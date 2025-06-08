# Herramienta de Eliminación de Fondo

Una aplicación completa de Streamlit para la eliminación de fondo en imágenes asistida por IA utilizando el modelo U2-Net.

## 🚀 Características

- **Solución Todo en Uno**: Pipeline completo de procesamiento de imágenes en una sola aplicación.
- **Interfaz de Arrastrar y Soltar**: Carga de imágenes sencilla con soporte para múltiples formatos.
- **Procesamiento en Tiempo Real**: Inferencia directa de IA sin llamadas a API externas.
- **Opción de Descarga**: Descarga de imágenes procesadas con fondos transparentes.
- **Logging Completo**: Registro automático de todas las predicciones con metadatos detallados en Azure Blob Storage.
- **Suite de Pruebas**: Cobertura completa de pruebas para funcionalidad, rendimiento y robustez.
- **Lista para la Nube**: Optimizada para el despliegue en servicios gratuitos en la nube.

## 🏗️ Arquitectura

Esta es una única aplicación Streamlit que incluye:
- Interfaz de carga y previsualización de imágenes.
- Inferencia del modelo ONNX U2-Net.
- Preprocesamiento y postprocesamiento de imágenes.
- Funcionalidad de descarga.

## 🛠️ Ejecución Local

1. **Descargar el código fuente**:
```bash
git clone https://github.com/daorhe2272/bckg-removal.git
cd bckg-removal
```

2. **Ejecutar con Docker Compose**:
```bash
docker-compose up -d
```

La aplicación estará disponible en `http://localhost:8501`

## 🐳 Comandos Útiles

```bash
# Iniciar la aplicación
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener la aplicación
docker-compose down

# Reconstruir si hay cambios
docker-compose up -d --build
```

## 🧪 Ejecutar Pruebas

Ejecuta las pruebas usando Docker Compose:

```bash
# Ejecutar todas las pruebas
docker-compose exec bg-remover-app python run_tests.py

# Ejecutar pruebas con cobertura
docker-compose exec bg-remover-app python run_tests.py --coverage

# Ejecutar pruebas específicas con pytest
docker-compose exec bg-remover-app python -m pytest tests/ -v

# Ejecutar pruebas de logging específicamente
docker-compose exec bg-remover-app python -m pytest tests/test_app.py::TestLoggingFunctionality -v
```

## ☁️ Opciones de Despliegue en la Nube

### Streamlit Cloud (Recomendado)
1. Sube tu código a GitHub.
2. Conecta tu repositorio a [Streamlit Cloud](https://streamlit.io/cloud).
3. ¡Despliega con un clic!

### Railway
1. Instala Railway CLI: `npm install -g @railway/cli`
2. Inicia sesión: `railway login`
3. Despliega: `railway up`

### Heroku
1. Crea `Procfile`:
```
web: streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0
```
2. Despliega usando Heroku CLI o la integración de GitHub.

### Render
1. Conecta tu repositorio de GitHub.
2. Configura el comando de construcción: `pip install -r requirements.txt`
3. Configura el comando de inicio: `streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0`

## 📋 Requisitos

- Python 3.9+
- Archivo del modelo U2-Net en `../models/production/u2net.onnx`
- Todas las dependencias listadas en `requirements.txt`

## 🎯 Uso

1. **Subir**: Arrastra y suelta una imagen o haz clic para buscar.
2. **Procesar**: Haz clic en "Eliminar Fondo" para procesar la imagen.
3. **Descargar**: Usa el botón de descarga para guardar el resultado.

## 🖼️ Formatos Soportados

- PNG, JPG, JPEG, BMP, TIFF

## 📁 Estructura del Proyecto

```
├── src/
│   ├── app.py              # Aplicación principal de Streamlit
│   └── .streamlit/
│       └── config.toml     # Configuración de Streamlit
├── tests/                  # Suite de pruebas completa
│   ├── test_app.py         # Pruebas de la aplicación principal
│   ├── test_model.py       # Pruebas del modelo ONNX
│   └── test_streamlit_components.py  # Pruebas de componentes UI
├── requirements.txt        # Dependencias de Python
├── Dockerfile             # Configuración del contenedor
├── docker-compose.yml     # Orquestación con Docker Compose
├── run_tests.py           # Script ejecutor de pruebas
├── pytest.ini            # Configuración de pytest
└── README.md              # Este archivo
``` 