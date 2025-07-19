from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import httpx
import base64
import time
from typing import Optional

from app.core.config import settings

router = APIRouter()

@router.post("/image")
async def process_image(
    image: UploadFile = File(...),
    model: str = Form("u2net")
):
    """
    Process an image to remove background.
    
    Args:
        image: Image file to process
        model: Model type (u2net, isnet, u2netp)
    
    Returns:
        JSON response with processed image data
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image."
            )
        
        # Check file size
        content = await image.read()
        if len(content) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.MAX_IMAGE_SIZE_MB}MB"
            )
        
        # Validate model type
        valid_models = ['u2net', 'isnet', 'u2netp']
        if model not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Valid options: {valid_models}"
            )
        
        # Extract image metadata
        metadata = {
            "filename": image.filename,
            "content_type": image.content_type,
            "file_size_bytes": len(content),
            "file_size_kb": round(len(content) / 1024, 2)
        }
        
        # Encode image for Functions call
        image_base64 = base64.b64encode(content).decode()
        
        # Call Azure Functions
        async with httpx.AsyncClient(timeout=settings.PROCESSING_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{settings.FUNCTIONS_BASE_URL}/api/background_removal",
                json={
                    "image_data": image_base64,
                    "model_type": model
                }
            )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=503,
                detail="Image processing service unavailable"
            )
        
        result = response.json()
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "success": True,
            "processing_time": round(processing_time, 2),
            "image_metadata": metadata,
            "result_image": result.get("result_image"),
            "model_used": model
        })
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        )

@router.get("/status")
async def get_processing_status():
    """Get current processing service status."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{settings.FUNCTIONS_BASE_URL}/api/health"
            )
        
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "functions_available": response.status_code == 200,
            "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "functions_available": False,
            "error": str(e)
        } 