import os
import io
import cv2
import glob
import json
import time
from datetime import datetime
import numpy as np
import streamlit as st
import onnxruntime as ort

from PIL import Image
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

load_dotenv()

st.set_page_config(
    page_title="Herramienta de Eliminación de Fondo",
    page_icon="🖼️",
    layout="wide"
)

def get_latest_model_path():
    """
    Busca el archivo .onnx más reciente en el directorio local de producción.
    
    Retorna:
        str: Ruta del modelo más reciente, o None si no se encuentra ninguno
    """
    production_dir = "models/production/"
    onnx_files = glob.glob(os.path.join(production_dir, "*.onnx"))
    
    if not onnx_files:
        return None
    
    latest_file = max(onnx_files, key=os.path.getmtime)
    return latest_file

MODEL_PATH = get_latest_model_path()

def get_azure_config():
    """
    Obtiene la configuración de Azure desde las variables de entorno.
    
    Retorna:
        dict: Diccionario con las credenciales y configuración de Azure
    """
    config = {
        'storage_account_name': os.getenv('AZURE_STORAGE_ACCOUNT_NAME'),
        'client_id': os.getenv('AZURE_CLIENT_ID'),
        'client_secret': os.getenv('AZURE_CLIENT_SECRET'),
        'tenant_id': os.getenv('AZURE_TENANT_ID'),
        'container_name': 'models'
    }
    return config

def log_prediction_to_blob(image_metadata, processing_time, success, error_message=None):
    """
    Registra las predicciones en Azure Blob Storage para monitoreo y análisis.
    
    Args:
        image_metadata (dict): Metadatos de la imagen procesada
        processing_time (float): Tiempo de procesamiento en segundos
        success (bool): Si el procesamiento fue exitoso
        error_message (str, optional): Mensaje de error si el procesamiento falló
    """
    try:
        # Determinar el entorno y archivo de log correspondiente
        environment = os.getenv('ENVIRONMENT', 'development').lower()
        
        if environment in ['development', 'dev', 'test']:
            log_file = 'logs/dev_predictions.log'
        else:
            log_file = 'logs/prod_predictions.log'
        
        config = get_azure_config()
        
        # Verificar que todas las credenciales estén disponibles
        required_credentials = [
            config['storage_account_name'],
            config['client_id'],
            config['client_secret'],
            config['tenant_id']
        ]
        
        if not all(required_credentials):
            st.warning("⚠️ No se pueden registrar las predicciones: Credenciales de Azure incompletas para el almacenamiento de logs.")
            return
        
        # Conectar a Azure Blob Storage usando DefaultAzureCredential
        account_url = f"https://{config['storage_account_name']}.blob.core.windows.net"
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        
        if not image_metadata or not isinstance(image_metadata, dict):
            st.warning("⚠️ Metadatos de imagen no disponibles para logging")
            return
            
        # Crear entrada de log con todos los detalles
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'entorno': environment,
            'nombre_imagen': image_metadata.get('name', 'desconocido'),
            'tamaño_imagen_kb': image_metadata.get('size_kb', 0),
            'ancho_imagen_px': image_metadata.get('width_px', 0),
            'alto_imagen_px': image_metadata.get('height_px', 0),
            'formato_imagen': image_metadata.get('format', 'desconocido'),
            'modo_imagen': image_metadata.get('mode', 'desconocido'),
            'tiempo_procesamiento_segundos': round(processing_time, 3),
            'éxito': success,
            'mensaje_error': error_message
        }
        
        try:
            # Intentar leer el contenido existente del archivo de log
            blob_client = blob_service_client.get_blob_client(container='models', blob=log_file)
            existing_content = blob_client.download_blob().readall().decode('utf-8')
        except Exception:
            # Si el archivo no existe, empezar con contenido vacío
            existing_content = ""
        
        # Agregar nueva entrada al contenido existente
        new_content = existing_content + json.dumps(log_entry) + '\n'
        
        # Subir el contenido actualizado
        blob_client.upload_blob(new_content, overwrite=True)
        
    except Exception as e:
        st.error(f"❌ Error al registrar la predicción en Azure Blob Storage: {str(e)}")
        with st.expander("🔍 Ver detalles del error de logging"):
            st.code(f"Error: {str(e)}\nMetadatos de la imagen: {json.dumps(image_metadata, indent=2)}")

def download_model_from_azure(status_placeholder=None):
    """
    Descarga el modelo más reciente desde Azure Blob Storage.
    
    Args:
        status_placeholder: Placeholder de Streamlit para mostrar el estado de la descarga
        
    Retorna:
        bool: True si la descarga fue exitosa, False en caso contrario
    """
    if status_placeholder is None:
        status_container = st.container()
        status_placeholder = status_container.empty()
    
    try:
        config = get_azure_config()
        
        # Verificar que todas las credenciales necesarias estén disponibles
        required_credentials = [
            config['storage_account_name'],
            config['client_id'],
            config['client_secret'],
            config['tenant_id']
        ]
        
        if not all(required_credentials):
            with status_placeholder.container():
                st.warning("⚠️ Credenciales de Azure incompletas.")
                st.info("Asegúrate de configurar todas las variables de entorno del Service Principal.")
            return False
        
        with status_placeholder.container():
            st.info(f"🔗 Conectando a Azure Storage: {config['storage_account_name']}")
        
        # Establecer conexión con Azure Blob Storage
        account_url = f"https://{config['storage_account_name']}.blob.core.windows.net"
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        
        container_client = blob_service_client.get_container_client(config['container_name'])
        
        with status_placeholder.container():
            st.info(f"📁 Verificando acceso al contenedor: {config['container_name']}")
        
        try:
            # Verificar que el contenedor sea accesible
            container_properties = container_client.get_container_properties()
            with status_placeholder.container():
                st.success(f"✅ Acceso al contenedor confirmado")
        except Exception as container_error:
            with status_placeholder.container():
                st.error(f"❌ Error al acceder al contenedor '{config['container_name']}': {str(container_error)}")
            return False
        
        with status_placeholder.container():
            st.info("🔍 Buscando archivos en Azure Blob Storage...")
            
        with st.spinner("Procesando archivos..."):
            # Obtener lista de todos los blobs en el contenedor
            all_blobs = list(container_client.list_blobs())
            
            onnx_blobs = []
            production_blobs = []
            
            # Buscar específicamente archivos .onnx en la carpeta production/
            for blob in container_client.list_blobs(name_starts_with="production/"):
                production_blobs.append(blob.name)
                if blob.name.endswith('.onnx'):
                    onnx_blobs.append(blob)
            
            with status_placeholder.container():
                if onnx_blobs:
                    st.info(f"📁 Encontrado {len(onnx_blobs)} modelo(s) en production/")
                    latest_name = max(onnx_blobs, key=lambda x: x.last_modified).name
                    st.text(f"  ✓ Más reciente: {latest_name}")
                else:
                    st.info(f"📊 Buscando en {len(all_blobs)} archivos del contenedor...")
            
            if not onnx_blobs:
                with status_placeholder.container():
                    st.error("❌ No se encontraron archivos .onnx en la ruta production/")
                    st.info("💡 Verifica que los archivos .onnx estén subidos en la ruta correcta")
                return False
            
            # Seleccionar el modelo más reciente basado en la fecha de modificación
            latest_blob = max(onnx_blobs, key=lambda x: x.last_modified)
            blob_name = latest_blob.name
            
            with status_placeholder.container():
                st.info(f"📥 Descargando: {os.path.basename(blob_name)}")
        
        # Crear directorio local si no existe
        os.makedirs("models/production", exist_ok=True)
        download_path = os.path.join("models/production", os.path.basename(blob_name))
        
        with st.spinner("📥 Descargando modelo..."):
            blob_client = blob_service_client.get_blob_client(container=config['container_name'], blob=blob_name)
            
            # Descargar y guardar el archivo
            with open(download_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            with status_placeholder.container():
                st.success(f"✅ Modelo descargado: {os.path.basename(blob_name)}")
            return True
            
    except Exception as e:
        with status_placeholder.container():
            st.error(f"❌ Error al descargar el modelo: {str(e)}")
            with st.expander("🔍 Ver detalles del error"):
                import traceback
                st.code(traceback.format_exc())
        return False

@st.cache_resource
def load_model():
    """
    Carga el modelo ONNX para inferencia. Usa caché de Streamlit para optimizar rendimiento.
    Si no encuentra un modelo local, intenta descargarlo desde Azure.
    
    Retorna:
        onnxruntime.InferenceSession: Sesión de inferencia del modelo, o None si falla
    """
    status_container = st.container()
    status_placeholder = status_container.empty()
    
    model_path = MODEL_PATH
    
    # Si no hay modelo local, intentar descarga desde Azure
    if model_path is None or not os.path.exists(model_path):
        with status_placeholder.container():
            st.info("🔍 Modelo no encontrado localmente. Intentando descargar desde Azure...")
        
        if not download_model_from_azure(status_placeholder):
            with status_placeholder.container():
                st.error("❌ No se pudo obtener el modelo desde Azure Blob Storage")
                st.markdown("""
                **📋 Verifica la configuración:**
                - Las variables de entorno de Azure estén configuradas correctamente
                - El Service Principal tenga permisos de lectura en el Blob Storage  
                - El modelo existe en la ruta especificada en Azure
                """)
            return None
        
        # Actualizar la ruta del modelo después de la descarga
        model_path = get_latest_model_path()
        
        if model_path is None:
            with status_placeholder.container():
                st.error("❌ No se encontró el modelo después de la descarga")
            return None
    else:
        with status_placeholder.container():
            st.info(f"🔍 Cargando modelo local: {os.path.basename(model_path)}")
    
    try:
        # Cargar el modelo usando ONNX Runtime
        session = ort.InferenceSession(model_path)
        with status_placeholder.container():
            st.success("🎯 Modelo cargado exitosamente!")
        return session
    except Exception as e:
        with status_placeholder.container():
            st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocesa la imagen para el modelo U2-Net.
    
    Args:
        image (PIL.Image): Imagen de entrada
        
    Retorna:
        np.ndarray: Array de imagen preprocesada para el modelo
    """
    # Convertir a RGB y redimensionar a 320x320 (tamaño esperado por U2-Net)
    img = image.convert('RGB')
    img = img.resize((320, 320))
    
    # Normalizar valores de píxel a rango [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Cambiar orden de dimensiones de HWC a CHW (canales primero)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Agregar dimensión de batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_mask(mask: np.ndarray, original_size: tuple) -> np.ndarray:
    """
    Postprocesa la máscara de salida del modelo.
    
    Args:
        mask (np.ndarray): Máscara de salida del modelo
        original_size (tuple): Tamaño original de la imagen (ancho, alto)
        
    Retorna:
        np.ndarray: Máscara procesada redimensionada al tamaño original
    """
    # Remover dimensiones extra y normalizar a rango [0, 255]
    mask = mask.squeeze()
    mask = (mask * 255).astype(np.uint8)
    
    # Redimensionar a tamaño original
    mask = cv2.resize(mask, original_size)
    return mask

def apply_mask_to_image(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Aplica la máscara a la imagen para crear fondo transparente.
    
    Args:
        image (PIL.Image): Imagen original
        mask (np.ndarray): Máscara de segmentación
        
    Retorna:
        PIL.Image: Imagen con fondo transparente
    """
    # Convertir imagen a RGBA para soporte de transparencia
    img_array = np.array(image.convert('RGBA'))
    
    # Normalizar máscara y aplicar como canal alfa
    mask_normalized = mask.astype(np.float32) / 255.0
    img_array[:, :, 3] = (mask_normalized * 255).astype(np.uint8)
    return Image.fromarray(img_array, 'RGBA')

def process_image(image: Image.Image, session, image_metadata=None):
    """
    Procesa una imagen completa: preprocesamiento, inferencia y postprocesamiento.
    
    Args:
        image (PIL.Image): Imagen de entrada
        session: Sesión de inferencia ONNX
        image_metadata (dict, optional): Metadatos para logging
        
    Retorna:
        PIL.Image: Imagen procesada con fondo removido, o None si hay error
    """
    start_time = time.time()
    
    try:
        # Guardar tamaño original para redimensionar la máscara final
        original_size = image.size
        
        # Preprocesar imagen para el modelo
        input_array = preprocess_image(image)
        
        # Ejecutar inferencia del modelo
        inputs = {session.get_inputs()[0].name: input_array}
        outputs = session.run(None, inputs)
        mask = outputs[0]
        
        # Postprocesar máscara y aplicar a imagen original
        processed_mask = postprocess_mask(mask, original_size)
        result_image = apply_mask_to_image(image, processed_mask)
        
        processing_time = time.time() - start_time
        
        # Registrar predicción exitosa en logs
        if image_metadata:
            log_prediction_to_blob(image_metadata, processing_time, success=True)
        
        return result_image
    except Exception as e:
        processing_time = time.time() - start_time
        error_message = str(e)
        
        # Registrar predicción fallida en logs
        if image_metadata:
            log_prediction_to_blob(image_metadata, processing_time, success=False, error_message=error_message)
        
        st.error(f"Error al procesar la imagen: {error_message}")
        return None

def image_to_bytes(image: Image.Image) -> bytes:
    """
    Convierte una imagen PIL a bytes para descarga.
    
    Args:
        image (PIL.Image): Imagen a convertir
        
    Retorna:
        bytes: Imagen en formato bytes PNG
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

st.title("🖼️ Herramienta de Eliminación de Fondo")
st.markdown("Sube una imagen para eliminar su fondo automáticamente usando IA")

with st.sidebar:
    st.header("⚙️ Configuración")
    config = get_azure_config()
    
    # Verificar si Azure está completamente configurado
    azure_configured = all([
        config['storage_account_name'],
        config['client_id'],
        config['client_secret'], 
        config['tenant_id']
    ])
    
    if azure_configured:
        st.success("✅ Azure Storage configurado")
        with st.expander("Ver detalles de configuración"):
            st.code(f"""
Storage Account: {config['storage_account_name']}
Container: {config['container_name']}
Model Search Path: production/*.onnx (latest)
Client ID: {config['client_id'][:6]}***
Tenant ID: {config['tenant_id'][:6]}***
            """)
    else:
        st.warning("⚠️ Azure Storage no configurado completamente")
        st.info("Variables de entorno requeridas para Service Principal:")
        st.code("""
AZURE_STORAGE_ACCOUNT_NAME
AZURE_CLIENT_ID
AZURE_CLIENT_SECRET  
AZURE_TENANT_ID
        """)

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
        
        st.subheader("📋 Detalles de la Imagen")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.metric(
                label="📄 Archivo",
                value=uploaded_file.name
            )
        
        with detail_col2:
            st.metric(
                label="💾 Tamaño",
                value=f"{uploaded_file.size / 1024:.1f} KB"
            )
        
        with detail_col3:
            st.metric(
                label="📐 Dimensiones",
                value=f"{original_image.size[0]} × {original_image.size[1]} px"
            )

with col2:
    st.header("✨ Imagen Procesada")
    
    if uploaded_file is not None:
        image_metadata = {
            'name': uploaded_file.name,
            'size_kb': round(uploaded_file.size / 1024, 2),
            'width_px': original_image.size[0],
            'height_px': original_image.size[1],
            'format': original_image.format or 'unknown',
            'mode': original_image.mode
        }
        
        if st.button("🚀 Eliminar Fondo", type="primary", use_container_width=True):
            with st.spinner("Procesando imagen con IA..."):
                processed_image = process_image(original_image, session, image_metadata)
                
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