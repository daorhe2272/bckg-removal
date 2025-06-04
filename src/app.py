import os
import io
import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort

from PIL import Image

st.set_page_config(
    page_title="Herramienta de Eliminación de Fondo",
    page_icon="🖼️",
    layout="wide"
)

MODEL_PATH = "models/production/u2net.onnx"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modelo U2-Net no encontrado en {MODEL_PATH}")
        st.info("Asegúrate de que el archivo del modelo existe en el directorio models/production/")
        return None
    
    try:
        session = ort.InferenceSession(MODEL_PATH)
        return session
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert('RGB')
    img = img.resize((320, 320))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_mask(mask: np.ndarray, original_size: tuple) -> np.ndarray:
    mask = mask.squeeze()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, original_size)
    return mask

def apply_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    img_array = np.array(image.convert('RGBA'))
    mask_normalized = mask.astype(np.float32) / 255.0
    img_array[:, :, 3] = (mask_normalized * 255).astype(np.uint8)
    return Image.fromarray(img_array, 'RGBA')

def process_image(image: Image.Image, session):
    try:
        original_size = image.size
        input_array = preprocess_image(image)
        
        inputs = {session.get_inputs()[0].name: input_array}
        outputs = session.run(None, inputs)
        mask = outputs[0]
        
        processed_mask = postprocess_mask(mask, original_size)
        result_image = apply_mask_to_image(image, processed_mask)
        
        return result_image
    except Exception as e:
        st.error(f"Error al procesar la imagen: {str(e)}")
        return None

def image_to_bytes(image: Image.Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

st.title("🖼️ Herramienta de Eliminación de Fondo")
st.markdown("Sube una imagen para eliminar su fondo automáticamente usando IA")

session = load_model()

if session is None:
    st.stop()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📤 Subir Imagen")
    
    uploaded_file = st.file_uploader(
        "Arrastra y suelta una imagen aquí o haz clic para buscar",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Formatos soportados: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        st.subheader("Imagen Original")
        original_image = Image.open(uploaded_file)
        st.image(original_image, use_column_width=True)
        
        file_details = {
            "Nombre de archivo": uploaded_file.name,
            "Tamaño de archivo": f"{uploaded_file.size / 1024:.1f} KB",
            "Tamaño de imagen": f"{original_image.size[0]} × {original_image.size[1]} px"
        }
        
        st.json(file_details)

with col2:
    st.header("✨ Imagen Procesada")
    
    if uploaded_file is not None:
        if st.button("🚀 Eliminar Fondo", type="primary", use_container_width=True):
            with st.spinner("Procesando imagen con IA..."):
                original_image = Image.open(uploaded_file)
                processed_image = process_image(original_image, session)
                
                if processed_image:
                    st.session_state.processed_image = processed_image
                    st.session_state.original_filename = uploaded_file.name
        
        if "processed_image" in st.session_state:
            st.image(st.session_state.processed_image, use_column_width=True)
            
            filename_base = st.session_state.original_filename.rsplit('.', 1)[0]
            download_filename = f"{filename_base}_no_background.png"
            
            processed_image_bytes = image_to_bytes(st.session_state.processed_image)
            
            st.download_button(
                label="📥 Descargar Imagen Procesada",
                data=processed_image_bytes,
                file_name=download_filename,
                mime="image/png",
                type="secondary",
                use_container_width=True
            )
            
            st.success("✅ Fondo eliminado exitosamente!")
    else:
        st.info("👆 Sube una imagen a la izquierda para empezar")

st.markdown("---")
st.markdown(
    "**Cómo funciona:** Esta herramienta utiliza una red neuronal U2-Net para detectar y eliminar automáticamente los fondos de las imágenes. "
    "La imagen procesada tendrá un fondo transparente que podrás usar en tus proyectos."
)

with st.expander("ℹ️ Sobre la Tecnología"):
    st.markdown("""
    - **Modelo U2-Net**: Una arquitectura de aprendizaje profundo diseñada específicamente para la detección de objetos prominentes
    - **ONNX Runtime**: Motor de inferencia optimizado para el despliegue de modelos de IA multiplataforma
    - **Procesamiento de Imágenes**: OpenCV y PIL para la manipulación eficiente de imágenes
    - **Streamlit**: Framework web moderno para crear aplicaciones de datos interactivas
    """) 