import os
import io
import cv2
import glob
import json
import time
import numpy as np
import streamlit as st
import onnxruntime as ort

from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

load_dotenv()

def initialize_models_at_startup():
    """
    Inicializa todos los modelos disponibles al arranque del contenedor, antes de que se muestre la UI.
    Esto asegura que todos los modelos estén listos desde el primer uso.
    """
    print(f"[STARTUP] Verificando disponibilidad de modelos...")
    
    model_paths = get_all_model_paths()
    if model_paths:
        print(f"[STARTUP] ✅ {len(model_paths)} modelo(s) local(es) encontrado(s):")
        for path in model_paths:
            print(f"[STARTUP]   - {os.path.basename(path)}")
        return True
    
    print(f"[STARTUP] 📥 Modelos no encontrados localmente, descargando desde Azure...")
    
    # Intentar descarga sin UI de Streamlit
    try:
        config = get_azure_config()
        
        # Verificar credenciales
        required_credentials = [
            config['storage_account_name'],
            config['client_id'],
            config['client_secret'],
            config['tenant_id']
        ]
        
        if not all(required_credentials):
            print(f"[STARTUP] ⚠️ Credenciales de Azure incompletas, modelos se descargarán en primer uso")
            return False
        
        # Conectar a Azure Blob Storage
        account_url = f"https://{config['storage_account_name']}.blob.core.windows.net"
        credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
        container_client = blob_service_client.get_container_client(config['container_name'])
        
        # Buscar modelos .onnx en production/
        onnx_blobs = []
        for blob in container_client.list_blobs(name_starts_with="production/"):
            if blob.name.endswith('.onnx'):
                onnx_blobs.append(blob)
        
        if not onnx_blobs:
            print(f"[STARTUP] ❌ No se encontraron modelos en Azure, se intentará descargar en primer uso")
            return False
        
        # Descargar todos los modelos disponibles
        print(f"[STARTUP] 📥 Descargando {len(onnx_blobs)} modelo(s)...")
        
        # Crear directorio y descargar
        os.makedirs("models/production", exist_ok=True)
        
        for blob in onnx_blobs:
            blob_name = blob.name
            download_path = os.path.join("models/production", os.path.basename(blob_name))
            
            print(f"[STARTUP] 📥 Descargando: {os.path.basename(blob_name)}")
            
            blob_client = blob_service_client.get_blob_client(container=config['container_name'], blob=blob_name)
            
            with open(download_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        
        print(f"[STARTUP] ✅ {len(onnx_blobs)} modelo(s) descargado(s) exitosamente")
        return True
        
    except Exception as e:
        print(f"[STARTUP] ❌ Error al descargar modelos: {str(e)}")
        print(f"[STARTUP] ⚠️ Los modelos se descargarán en el primer uso")
        return False

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

def get_all_model_paths():
    """
    Busca todos los archivos .onnx disponibles en el directorio local de producción.
    
    Retorna:
        list: Lista de rutas de todos los modelos disponibles, ordenados por fecha de modificación (más reciente primero)
    """
    production_dir = "models/production/"
    onnx_files = glob.glob(os.path.join(production_dir, "*.onnx"))
    
    if not onnx_files:
        return []
    
    # Ordenar por fecha de modificación (más reciente primero)
    onnx_files.sort(key=os.path.getmtime, reverse=True)
    return onnx_files

def get_model_info(model_path):
    """
    Obtiene información detallada de un modelo específico.
    
    Args:
        model_path (str): Ruta del modelo
        
    Retorna:
        dict: Diccionario con información del modelo
    """
    if not os.path.exists(model_path):
        return None
    
    stats = os.stat(model_path)
    return {
        'name': os.path.basename(model_path),
        'path': model_path,
        'size_mb': round(stats.st_size / (1024 * 1024), 2),
        'modified_time': datetime.fromtimestamp(stats.st_mtime),
        'display_name': os.path.basename(model_path).replace('.onnx', '').replace('_', ' ').title()
    }

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

def download_all_models_from_azure(status_placeholder=None):
    """
    Descarga todos los modelos disponibles desde Azure Blob Storage.
    
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
                    for blob in onnx_blobs:
                        st.text(f"  ✓ {blob.name}")
                else:
                    st.info(f"📊 Buscando en {len(all_blobs)} archivos del contenedor...")
            
            if not onnx_blobs:
                with status_placeholder.container():
                    st.error("❌ No se encontraron archivos .onnx en la ruta production/")
                    st.info("💡 Verifica que los archivos .onnx estén subidos en la ruta correcta")
                return False
        
        # Crear directorio local si no existe
        os.makedirs("models/production", exist_ok=True)
        
        # Descargar todos los modelos
        downloaded_count = 0
        for blob in onnx_blobs:
            blob_name = blob.name
            download_path = os.path.join("models/production", os.path.basename(blob_name))
            
            with status_placeholder.container():
                st.info(f"📥 Descargando: {os.path.basename(blob_name)}")
            
            with st.spinner(f"📥 Descargando {os.path.basename(blob_name)}..."):
                blob_client = blob_service_client.get_blob_client(container=config['container_name'], blob=blob_name)
                
                # Descargar y guardar el archivo
                with open(download_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                
                downloaded_count += 1
        
        with status_placeholder.container():
            st.success(f"✅ {downloaded_count} modelo(s) descargado(s) exitosamente")
        return True
            
    except Exception as e:
        with status_placeholder.container():
            st.error(f"❌ Error al descargar los modelos: {str(e)}")
            with st.expander("🔍 Ver detalles del error"):
                import traceback
                st.code(traceback.format_exc())
        return False

@st.cache_resource
def load_all_models():
    """
    Carga todos los modelos ONNX disponibles para inferencia. Usa caché de Streamlit para optimizar rendimiento.
    Los modelos deberían estar disponibles localmente gracias a la inicialización al arranque.
    
    Retorna:
        dict: Diccionario con sesiones de inferencia de todos los modelos disponibles, o dict vacío si falla
    """
    model_paths = get_all_model_paths()
    
    # Si hay modelos disponibles localmente
    if model_paths:
        sessions = {}
        loaded_models = []
        failed_models = []
        
        for model_path in model_paths:
            try:
                session = ort.InferenceSession(model_path)
                model_name = os.path.basename(model_path)
                sessions[model_name] = session
                loaded_models.append(model_name)
            except Exception as e:
                failed_models.append(f"{os.path.basename(model_path)}: {str(e)}")
        
        if loaded_models:
            st.success(f"🎯 {len(loaded_models)} modelo(s) cargado(s) exitosamente!")
            if failed_models:
                st.warning(f"⚠️ {len(failed_models)} modelo(s) fallaron al cargar")
                with st.expander("Ver errores de carga"):
                    for error in failed_models:
                        st.code(error)
            return sessions
        else:
            st.error("❌ No se pudo cargar ningún modelo local")
            if failed_models:
                with st.expander("Ver errores de carga"):
                    for error in failed_models:
                        st.code(error)
    
    # Respaldo: si no hay modelos locales, intentar descarga con UI
    st.info("🔍 Modelos no encontrados localmente. Intentando descargar desde Azure...")
    
    if not download_all_models_from_azure():
        st.error("❌ No se pudo obtener los modelos desde Azure Blob Storage")
        st.markdown("""
        **📋 Verifica la configuración:**
        - Las variables de entorno de Azure estén configuradas correctamente
        - El Service Principal tenga permisos de lectura en el Blob Storage  
        - Los modelos existen en la ruta especificada en Azure
        """)
        return {}
    
    # Intentar cargar después de la descarga
    model_paths = get_all_model_paths()
    if model_paths:
        sessions = {}
        loaded_models = []
        failed_models = []
        
        for model_path in model_paths:
            try:
                session = ort.InferenceSession(model_path)
                model_name = os.path.basename(model_path)
                sessions[model_name] = session
                loaded_models.append(model_name)
            except Exception as e:
                failed_models.append(f"{os.path.basename(model_path)}: {str(e)}")
        
        if loaded_models:
            st.success(f"🎯 {len(loaded_models)} modelo(s) cargado(s) después de la descarga!")
            if failed_models:
                st.warning(f"⚠️ {len(failed_models)} modelo(s) fallaron al cargar")
            return sessions
        else:
            st.error("❌ No se pudo cargar ningún modelo después de la descarga")
            return {}
    
    st.error("❌ No se encontraron modelos después de la descarga")
    return {}

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

def optimize_image_size(image: Image.Image, max_dimension: int = 2000) -> tuple[Image.Image, bool]:
    """
    Optimiza el tamaño de la imagen para mejorar el rendimiento del procesamiento.
    Si la imagen es mayor a max_dimension píxeles en cualquier lado, la redimensiona
    manteniendo la proporción.
    
    Args:
        image (PIL.Image): Imagen original
        max_dimension (int): Dimensión máxima permitida (alto o ancho)
        
    Retorna:
        tuple: (imagen_optimizada, fue_redimensionada)
    """
    width, height = image.size
    
    # Si la imagen es pequeña, no hacer nada
    if width <= max_dimension and height <= max_dimension:
        return image, False
    
    # Calcular nuevo tamaño manteniendo proporción
    if width > height:
        # La imagen es más ancha que alta
        new_width = max_dimension
        new_height = int((height * max_dimension) / width)
    else:
        # La imagen es más alta que ancha
        new_height = max_dimension
        new_width = int((width * max_dimension) / height)
    
    # Redimensionar la imagen con alta calidad
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image, True

def process_image(image: Image.Image, session, image_metadata=None):
    """
    Procesa una imagen completa: optimización de tamaño, preprocesamiento, inferencia y postprocesamiento.
    
    Args:
        image (PIL.Image): Imagen de entrada
        session: Sesión de inferencia ONNX
        image_metadata (dict, optional): Metadatos para logging
        
    Retorna:
        PIL.Image: Imagen procesada con fondo removido, o None si hay error
    """
    start_time = time.time()
    
    try:
        # Optimizar tamaño de imagen para mejor rendimiento
        optimized_image, was_resized = optimize_image_size(image, max_dimension=2000)
        
        if was_resized:
            original_size = image.size
            optimized_size = optimized_image.size
            print(f"[PROCESSING] Imagen redimensionada: {original_size} → {optimized_size} para mejor rendimiento")
        
        # Usar la imagen optimizada para el procesamiento
        processing_size = optimized_image.size
        
        # Preprocesar imagen para el modelo
        input_array = preprocess_image(optimized_image)
        
        # Ejecutar inferencia del modelo
        inputs = {session.get_inputs()[0].name: input_array}
        outputs = session.run(None, inputs)
        mask = outputs[0]
        
        # Postprocesar máscara y aplicar a la imagen optimizada
        processed_mask = postprocess_mask(mask, processing_size)
        result_image = apply_mask_to_image(optimized_image, processed_mask)
        
        processing_time = time.time() - start_time
        
        # Registrar predicción exitosa en logs
        if image_metadata:
            # Actualizar metadatos con información de optimización
            if was_resized:
                image_metadata['optimized'] = True
                image_metadata['original_width_px'] = image.size[0]
                image_metadata['original_height_px'] = image.size[1]
                image_metadata['processed_width_px'] = processing_size[0]
                image_metadata['processed_height_px'] = processing_size[1]
            else:
                image_metadata['optimized'] = False
            
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
st.markdown("Sube una imagen para eliminar su fondo automáticamente usando IA. Selecciona entre múltiples modelos disponibles para obtener los mejores resultados.")

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
Model Search Path: production/*.onnx (todos)
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

initialize_models_at_startup()

session = load_all_models()

if not session:
    st.stop()

with st.sidebar:
    st.markdown("---")
    st.subheader("🤖 Selección de Modelo")
    
    model_paths = get_all_model_paths()
    if model_paths and session:
        # Crear opciones para el selector de modelos
        model_options = {}
        for model_path in model_paths:
            model_info = get_model_info(model_path)
            if model_info and model_info['name'] in session:
                model_options[model_info['display_name']] = model_info['name']
        
        if len(model_options) > 1:
            # Si hay múltiples modelos, mostrar selector
            selected_model_display = st.selectbox(
                "Elige el modelo a usar:",
                options=list(model_options.keys()),
                index=0,
                help="Selecciona el modelo que mejor se adapte a tus necesidades. Puedes cambiar entre modelos sin reiniciar la aplicación."
            )
            selected_model_name = model_options[selected_model_display]
            st.session_state.selected_model = selected_model_name
            
            # Mostrar información del modelo seleccionado
            selected_model_path = None
            for path in model_paths:
                if os.path.basename(path) == selected_model_name:
                    selected_model_path = path
                    break
            
            if selected_model_path:
                model_info = get_model_info(selected_model_path)
                if model_info:
                    st.info(f"📁 **{model_info['display_name']}**")
                    st.caption(f"💾 Tamaño: {model_info['size_mb']} MB")
                    st.caption(f"📅 Modificado: {model_info['modified_time'].strftime('%Y-%m-%d %H:%M')}")
        else:
            # Si solo hay un modelo, mostrarlo automáticamente
            single_model_name = list(model_options.values())[0]
            st.session_state.selected_model = single_model_name
            st.info(f"📁 **Modelo único disponible**")
            
            single_model_path = None
            for path in model_paths:
                if os.path.basename(path) == single_model_name:
                    single_model_path = path
                    break
            
            if single_model_path:
                model_info = get_model_info(single_model_path)
                if model_info:
                    st.caption(f"� Tamaño: {model_info['size_mb']} MB")
                    st.caption(f"📅 Modificado: {model_info['modified_time'].strftime('%Y-%m-%d %H:%M')}")
        
        # Mostrar resumen de todos los modelos disponibles
        with st.expander("Ver todos los modelos"):
            st.subheader("📊 Estado de los Modelos")
            for model_path in model_paths:
                model_info = get_model_info(model_path)
                if model_info:
                    status = "✅ Cargado" if model_info['name'] in session else "❌ Error"
                    st.text(f"{status} {model_info['display_name']} ({model_info['size_mb']} MB)")
    else:
        st.warning("⚠️ Modelos no disponibles")
        if 'selected_model' in st.session_state:
            del st.session_state.selected_model
    
    st.markdown("---")
    st.subheader("⚡ Optimización")
    st.caption("📐 Imágenes > 2000px se redimensionan automáticamente")
    st.caption("🚀 Mejora significativa en velocidad de procesamiento")
    st.caption("📱 Mantiene calidad y proporciones originales")

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
            # Verificar que hay un modelo seleccionado
            if 'selected_model' not in st.session_state:
                st.error("❌ No hay ningún modelo seleccionado. Verifica que los modelos estén cargados correctamente.")
                st.stop()
            
            selected_model_name = st.session_state.selected_model
            if selected_model_name not in session:
                st.error(f"❌ El modelo seleccionado '{selected_model_name}' no está disponible.")
                st.stop()
            
            selected_session = session[selected_model_name]
            
            # Verificar si la imagen será optimizada
            _, will_be_resized = optimize_image_size(original_image, max_dimension=2000)
            
            if will_be_resized:
                st.info("⚡ Imagen grande detectada. Se optimizará automáticamente para mejor rendimiento.")
            
            # Mostrar información del modelo que se está usando
            st.info(f"🤖 Procesando con modelo: **{selected_model_name.replace('.onnx', '').replace('_', ' ').title()}**")
            
            with st.spinner("Procesando imagen con IA..."):
                # Actualizar metadatos con información del modelo
                image_metadata['model_used'] = selected_model_name
                
                processed_image = process_image(original_image, selected_session, image_metadata)
                
                if processed_image:
                    st.session_state.processed_image = processed_image
                    st.session_state.original_filename = uploaded_file.name
                    st.session_state.was_optimized = will_be_resized
                    st.session_state.model_used = selected_model_name
        
        if "processed_image" in st.session_state:
            st.image(st.session_state.processed_image, use_column_width=True)
            
            # Mostrar información del modelo usado
            if st.session_state.get("model_used"):
                model_display_name = st.session_state.model_used.replace('.onnx', '').replace('_', ' ').title()
                st.success(f"🤖 Procesado con: **{model_display_name}**")
            
            # Mostrar información sobre la optimización si aplica
            if st.session_state.get("was_optimized", False):
                original_dims = f"{original_image.size[0]} × {original_image.size[1]}"
                processed_dims = f"{st.session_state.processed_image.size[0]} × {st.session_state.processed_image.size[1]}"
                st.info(f"⚡ Imagen optimizada: {original_dims} → {processed_dims} px para mejor rendimiento")
            
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
    "**Cómo funciona:** Esta herramienta utiliza redes neuronales U2-Net para detectar y eliminar automáticamente los fondos de las imágenes. "
    "Puedes elegir entre múltiples modelos disponibles, cada uno optimizado para diferentes tipos de imágenes. "
    "Las imágenes grandes (>2000px) se optimizan automáticamente para mayor velocidad manteniendo la calidad. "
    "La imagen procesada tendrá un fondo transparente que podrás usar en tus proyectos."
)

with st.expander("ℹ️ Sobre la Tecnología"):
    st.markdown("""
    - **Modelo U2-Net**: Una arquitectura de aprendizaje profundo diseñada específicamente para la detección de objetos prominentes
    - **ONNX Runtime**: Motor de inferencia optimizado para el despliegue de modelos de IA multiplataforma
    - **Procesamiento de Imágenes**: OpenCV y PIL para la manipulación eficiente de imágenes
    - **Streamlit**: Framework web moderno para crear aplicaciones de datos interactivas
    """) 