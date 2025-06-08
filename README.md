# Herramienta de Eliminación de Fondo

Una aplicación completa de Streamlit para la eliminación de fondo en imágenes asistida por IA utilizando el modelo U2-Net.

## **¡Prueba la App en Producción!**

### 🌐 **[CLICK AQUÍ PARA ACCEDER A LA APLICACIÓN](https://background-removal-app-febrhrcza2hrbbfj.eastus-01.azurewebsites.net)**

**La aplicación está desplegada y funcionando en Azure App Service con un pipeline completo de MLOps**

## Características

- **Solución Todo en Uno**: Pipeline completo de procesamiento de imágenes en una sola aplicación.
- **Interfaz de Arrastrar y Soltar**: Carga de imágenes sencilla con soporte para múltiples formatos.
- **Procesamiento en Tiempo Real**: Inferencia directa de IA sin llamadas a API externas.
- **Opción de Descarga**: Descarga de imágenes procesadas con fondos transparentes.
- **Logging Completo**: Registro automático de todas las predicciones con metadatos detallados en Azure Blob Storage.
- **Suite de Pruebas**: Cobertura completa de pruebas para funcionalidad, rendimiento y robustez.
- **Lista para la Nube**: Optimizada para el despliegue en servicios gratuitos en la nube.

## Arquitectura

Esta es una única aplicación Streamlit que incluye:
- Interfaz de carga y previsualización de imágenes.
- Inferencia del modelo ONNX U2-Net.
- Preprocesamiento y postprocesamiento de imágenes.
- Funcionalidad de descarga.

## Ejecución Local

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

## Comandos Útiles

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

## Ejecutar Pruebas

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

## Despliegue en la Nube

### **Producción Activa**
- **URL**: https://background-removal-app-febrhrcza2hrbbfj.eastus-01.azurewebsites.net
- **Plataforma**: Azure App Service
- **Contenedor**: GitHub Container Registry (ghcr.io)
- **CI/CD**: GitHub Actions

### **Pipeline MLOps Automatizado**
1. **Push a `prod`** → Ejecuta pruebas automáticamente
2. **Pruebas exitosas** → Construye imagen Docker
3. **Imagen construida** → Despliega a Azure App Service
4. **Aplicación actualizada** → Disponible en producción

### **Monitoreo**
- **Logs en tiempo real**: Azure App Service logs
- **Predicciones registradas**: Azure Blob Storage
- **Modelos**: Descargados automáticamente desde Azure Blob Storage

## Uso

1. **Subir**: Arrastra y suelta una imagen o haz clic para buscar.
2. **Procesar**: Haz clic en "Eliminar Fondo" para procesar la imagen.
3. **Descargar**: Usa el botón de descarga para guardar el resultado.

## Formatos Soportados

- PNG, JPG, JPEG, BMP, TIFF

## **Pipeline CI/CD Detallado**

### **Descripción General**
El proyecto utiliza **GitHub Actions** para implementar un pipeline completo de MLOps que automatiza el testing, construcción y despliegue de la aplicación.

### **Gestión de Ramas**
- **`dev`**: Desarrollo y testing
- **`prod`**: Producción (deployment automático)

### **Flujo de Trabajo (.github/workflows/test-and-deploy.yml)**

#### **Triggers**
```yaml
on:
  push:
    branches: [ prod, dev ]  # Se ejecuta en push a dev o prod
  pull_request:
    branches: [ prod ]       # Se ejecuta en PRs hacia prod
  workflow_dispatch:         # Ejecución manual
```

#### **Job 1: Test**
- **Runner**: Ubuntu Latest
- **Propósito**: Validar código y funcionalidad
- **Pasos**:
  1. Checkout del código
  2. Setup Python 3.9
  3. Cache de dependencias pip
  4. Instalación de dependencias
  5. Ejecución de suite de pruebas completa
- **Herramientas**: pytest, pytest-cov, pytest-mock

#### **Job 2: Build and Push**
- **Dependencia**: Job Test exitoso
- **Condición**: Solo en push a rama `prod`
- **Propósito**: Construir y publicar imagen Docker
- **Pasos**:
  1. Checkout del código
  2. Setup Docker Buildx
  3. Login a GitHub Container Registry
  4. Extracción de metadatos (tags, labels)
  5. Build y push de imagen Docker
- **Registro**: `ghcr.io/daorhe2272/bckg-removal`
- **Tags**: `latest`, `prod-{sha}`, `prod`

#### **Job 3: Deploy**
- **Dependencia**: Job Build and Push exitoso
- **Condición**: Solo en push a rama `prod`
- **Propósito**: Desplegar contenedor a Azure App Service
- **Pasos**:
  1. Login a Azure usando Service Principal
  2. Deploy a Azure App Service
- **Método**: `azure/webapps-deploy@v2`

### **Secretos de GitHub Requeridos**
```
AZURE_CLIENT_ID          # Service Principal ID
AZURE_CLIENT_SECRET      # Service Principal Secret  
AZURE_TENANT_ID          # Azure Tenant ID
AZURE_SUBSCRIPTION_ID    # Azure Subscription ID
```

### **Tecnologías del Pipeline**
- **CI/CD**: GitHub Actions
- **Testing**: pytest + coverage
- **Containerización**: Docker + Multi-stage builds
- **Registry**: GitHub Container Registry (ghcr.io)
- **Cloud Platform**: Azure App Service
- **Autenticación**: Azure Service Principal
- **Storage**: Azure Blob Storage

### **Gestión de Datos y Modelos**

#### **Persistencia en Azure Blob Storage**
Todos los datos de la aplicación se almacenan en **Azure Blob Storage** en contenedores externos, garantizando:
- ✅ **Persistencia**: Los datos sobreviven a reinicios de contenedores
- ✅ **Escalabilidad**: Almacenamiento ilimitado y distribuido
- ✅ **Accesibilidad**: Acceso desde múltiples entornos (local, dev, prod)

#### **📁 Estructura de Contenedores**
```
📦 Azure Blob Storage
├── 📁 models/
│   ├── 📁 production/          # Modelos en producción
│   ├── 📁 development/         # Modelos para testing local
│   ├── 📁 candidates/          # Modelos candidatos para promoción
│   ├── 📁 deprecated/          # Modelos anteriores archivados
│   └── 📁 rejected/            # Modelos que no pasaron validación
├── 📁 logs/
│   ├── 📄 dev_predictions.log  # Logs de predicciones en desarrollo
│   └── 📄 prod_predictions.log # Logs de predicciones en producción
└── 📁 test-data/
    ├── 📁 images/              # Imágenes de prueba
    └── 📁 masks/               # Anotaciones y ground truth
```

#### **Flujo de Promoción de Modelos**
```
1. Analistas de Datos
   ├── Ejecutan aplicación localmente
   ├── Usan modelos del directorio `development/`
   ├── Realizan experimentos y validaciones
   └── Promueven modelo a `candidates/`

2. Validación de Candidatos
   ├── Pruebas con datos reales
   ├── Evaluación de métricas
   ├── Tests de performance
   └── Decisión: `production/` o `rejected/`

3. Modelo en Producción
   ├── App Service descarga modelo más reciente de `production/`
   ├── Modelo anterior se mueve a `deprecated/`
   └── Nueva versión disponible automáticamente
```

#### **Roles y Responsabilidades**
- **Analistas de Datos**:
  - Desarrollo y testing de modelos localmente
  - Promoción de modelos candidatos
  - Validación con datos reales
- **Sistema Automatizado**:
  - Descarga del modelo más reciente en producción
  - Logging automático de predicciones
  - Gestión de ciclo de vida de modelos

#### **Configuración Automática**
- **Selección de Modelo**: La aplicación busca automáticamente el archivo `.onnx` más reciente en `models/production/`
- **Reinicio Inteligente**: Cada reinicio del contenedor actualiza al modelo más actual
- **Logging Diferenciado**: Registros separados según entorno (`ENVIRONMENT` variable)

### **Métricas y Calidad**
- **Cobertura de tests**: >85%
- **Tests automatizados**: Unidad, integración, performance
- **Validación de modelos**: Responsiveness y métricas
- **Health checks**: Endpoint de salud en contenedor

## 📁 Estructura del Proyecto

```
📦 Proyecto
├── 📁 src/
│   ├── 📄 app.py               # Aplicación principal de Streamlit
│   └── 📁 .streamlit/
│       └── 📄 config.toml      # Configuración de Streamlit
├── 📁 tests/                   # Suite de pruebas completa
│   ├── 📄 test_app.py          # Pruebas de la aplicación principal
│   ├── 📄 test_model.py        # Pruebas del modelo ONNX
│   └── 📄 test_streamlit_components.py  # Pruebas de componentes UI
├── 📄 requirements.txt         # Dependencias de Python
├── 📄 Dockerfile               # Configuración del contenedor
├── 📄 docker-compose.yml       # Orquestación con Docker Compose
├── 📄 run_tests.py             # Script ejecutor de pruebas
├── 📄 pytest.ini               # Configuración de pytest
└── 📄 README.md                # Este archivo
``` 