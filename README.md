# Herramienta de Eliminación de Fondo

Una aplicación completa de Streamlit para la eliminación de fondo en imágenes asistida por IA utilizando el modelo U2-Net.

## 🚀 Características

- **Solución Todo en Uno**: Pipeline completo de procesamiento de imágenes en una sola aplicación.
- **Interfaz de Arrastrar y Soltar**: Carga de imágenes sencilla con soporte para múltiples formatos.
- **Procesamiento en Tiempo Real**: Inferencia directa de IA sin llamadas a API externas.
- **Opción de Descarga**: Descarga de imágenes procesadas con fondos transparentes.
- **Lista para la Nube**: Optimizada para el despliegue en servicios gratuitos en la nube.

## 🏗️ Arquitectura

Esta es una única aplicación Streamlit que incluye:
- Interfaz de carga y previsualización de imágenes.
- Inferencia del modelo ONNX U2-Net.
- Preprocesamiento y postprocesamiento de imágenes.
- Funcionalidad de descarga.

## 🛠️ Configuración Local

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

2. Ejecuta la aplicación:
```bash
streamlit run app.py
```

La aplicación estará disponible en `http://localhost:8501`

## 🐳 Despliegue con Docker

Construye y ejecuta con Docker:

```bash
# Construir la imagen
docker build -t background-removal-app .

# Ejecutar el contenedor
docker run -p 8501:8501 background-removal-app
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
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```
2. Despliega usando Heroku CLI o la integración de GitHub.

### Render
1. Conecta tu repositorio de GitHub.
2. Configura el comando de construcción: `pip install -r requirements.txt`
3. Configura el comando de inicio: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

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
src/
├── app.py              # Aplicación principal de Streamlit
├── requirements.txt    # Dependencias de Python
├── Dockerfile          # Configuración del contenedor
├── .streamlit/
│   └── config.toml     # Configuración de Streamlit
└── README.md           # Este archivo
``` 