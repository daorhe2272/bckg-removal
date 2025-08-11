import azure.functions as func
import logging
import json
import base64
import io
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import onnxruntime as ort

try:
    from azure.storage.blob import BlobServiceClient
    from azure.identity import DefaultAzureCredential
except Exception:  # pragma: no cover - optional in local
    BlobServiceClient = None  # type: ignore
    DefaultAzureCredential = None  # type: ignore


MODEL_FILES: Dict[str, Tuple[str, int]] = {
    # model_type -> (blob filename, preferred input size)
    "u2net": ("u2net.onnx", 320),
    "isnet": ("isnet-general-use.onnx", 1024),
    "isnet-general-use": ("isnet-general-use.onnx", 1024),
}


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def load_model_to_cache(model_filename: str) -> str:
    """Ensure model exists locally and return local path.

    Download from Azure Blob `models/<ENV_PREFIX>/` if needed, otherwise use local fallback
    at `functions/shared/models/` when present.
    """
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, model_filename)
    if os.path.exists(local_path):
        return local_path

    # Try local fallback first (developer may place files here)
    fallback_path = os.path.join(os.getcwd(), "shared", "models", model_filename)
    if os.path.exists(fallback_path):
        try:
            with open(fallback_path, "rb") as src, open(local_path, "wb") as dst:
                dst.write(src.read())
            return local_path
        except Exception:
            pass

    # Attempt to download from Azure Blob
    container_name = get_env("MODEL_CONTAINER_NAME", "models")
    path_prefix = get_env("MODEL_PATH_PREFIX", "production")  # e.g., production
    storage_account = get_env("AZURE_STORAGE_ACCOUNT_NAME")
    connection_string = get_env("AZURE_STORAGE_CONNECTION_STRING")

    blob_path = f"{path_prefix}/{model_filename}"
    try:
        if connection_string:
            bsc = BlobServiceClient.from_connection_string(connection_string)
        else:
            if not storage_account:
                raise RuntimeError("AZURE_STORAGE_ACCOUNT_NAME not set and no connection string available")
            if DefaultAzureCredential is None:
                raise RuntimeError("azure-identity is not available for DefaultAzureCredential")
            credential = DefaultAzureCredential()
            account_url = f"https://{storage_account}.blob.core.windows.net"
            bsc = BlobServiceClient(account_url=account_url, credential=credential)

        blob_client = bsc.get_blob_client(container=container_name, blob=blob_path)
        with open(local_path, "wb") as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        return local_path
    except Exception as ex:
        raise RuntimeError(f"Failed to obtain model '{model_filename}' from storage: {ex}")


_sessions: Dict[str, ort.InferenceSession] = {}


def get_session_for_model(model_filename: str) -> ort.InferenceSession:
    if model_filename in _sessions:
        return _sessions[model_filename]
    local_path = load_model_to_cache(model_filename)
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, int(get_env("ORT_INTRA_OP_THREADS", "1")))
    so.inter_op_num_threads = max(1, int(get_env("ORT_INTER_OP_THREADS", "1")))
    sess = ort.InferenceSession(local_path, providers=["CPUExecutionProvider"], sess_options=so)
    _sessions[model_filename] = sess
    return sess


def decode_image_to_pil(image_data_b64: str) -> Image.Image:
    # Supports both raw base64 and data URLs
    if image_data_b64.startswith("data:"):
        header, b64 = image_data_b64.split(",", 1)
    else:
        b64 = image_data_b64
    image_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def prepare_input_tensor(img: Image.Image, size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    orig_w, orig_h = img.size
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0  # HWC in [0,1]
    # CHW
    chw = np.transpose(arr, (2, 0, 1))
    chw = np.expand_dims(chw, 0)  # NCHW
    return chw, (orig_w, orig_h)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def run_inference(sess: ort.InferenceSession, input_tensor: np.ndarray) -> np.ndarray:
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: input_tensor})
    pred = outputs[0]
    pred = np.squeeze(pred)
    # Some models output logits; apply sigmoid; if already [0,1], this is largely idempotent
    pred = sigmoid(pred).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)
    return pred


def matte_to_rgba(original: Image.Image, matte: np.ndarray) -> bytes:
    matte_img = Image.fromarray((matte * 255).astype(np.uint8), mode="L")
    matte_resized = matte_img.resize(original.size, Image.BILINEAR)
    rgba = original.convert("RGBA")
    rgba.putalpha(matte_resized)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Background removal function triggered.")

    try:
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(json.dumps({"error": "Invalid JSON body"}), status_code=400, mimetype="application/json")

        image_data = (req_body or {}).get("image_data")
        model_type = (req_body or {}).get("model_type", "u2net").lower()

        if not image_data:
            return func.HttpResponse(json.dumps({"error": "No image data provided"}), status_code=400, mimetype="application/json")

        if model_type not in MODEL_FILES:
            return func.HttpResponse(
                json.dumps({"error": f"Unsupported model_type. Supported: {list(MODEL_FILES.keys())}"}),
                status_code=400,
                mimetype="application/json",
            )

        max_mb = int(get_env("PROCESSING_TIMEOUT_SECONDS", "30"))  # keep for run options if needed
        # Optional size check (raw base64 length approximation)
        max_image_mb = int(get_env("MAX_IMAGE_SIZE_MB", "10"))
        approx_bytes = int(len(image_data) * 0.75)
        if approx_bytes > max_image_mb * 1024 * 1024:
            return func.HttpResponse(
                json.dumps({"error": f"Image too large. Max {max_image_mb}MB"}),
                status_code=413,
                mimetype="application/json",
            )

        pil_img = decode_image_to_pil(image_data)

        model_filename, size = MODEL_FILES[model_type]
        session = get_session_for_model(model_filename)
        input_tensor, _ = prepare_input_tensor(pil_img, size)
        matte = run_inference(session, input_tensor)
        png_bytes = matte_to_rgba(pil_img, matte)
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("utf-8")

        return func.HttpResponse(
            json.dumps({
                "success": True,
                "result_image": data_url,
                "model_used": model_type,
            }),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.exception("Error processing image")
        return func.HttpResponse(
            json.dumps({"error": f"Processing failed: {str(e)}"}),
            status_code=500,
            mimetype="application/json",
        )