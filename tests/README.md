# Documentación de Pruebas

Este directorio contiene todas las pruebas unitarias e de integración para la aplicación de eliminación de fondos en imágenes.

## 📁 Estructura de Pruebas

```
📁 tests/
├── 📄 __init__.py                     # Inicialización del paquete
├── 📄 test_app.py                     # Pruebas de funcionalidad central
├── 📄 test_model.py                   # Pruebas de integración del modelo
├── 📄 test_streamlit_components.py    # Pruebas de componentes de UI
├── 📄 test_utils.py                   # Utilidades y ayudantes de prueba
└── 📄 README.md                       # Esta documentación
```

## Categorías de Pruebas

### 1. Pruebas Unitarias (`test_app.py`)
Prueba funciones individuales de forma aislada:
- **TestImagePreprocessing**: Pruebas para `preprocess_image()` (conversión RGB, redimensionado, normalización)
- **TestMaskPostprocessing**: Pruebas para `postprocess_mask()` (forma, tamaño, tipo de datos)
- **TestMaskApplication**: Pruebas para `apply_mask_to_image()` (transparencia, máscaras parciales)
- **TestImageConversion**: Pruebas para `image_to_bytes()` (formatos, recuperación)
- **TestModelLoading**: Pruebas para `load_model()` con mocks (éxito, errores, descarga Azure)
- **TestImageProcessing**: Pruebas de integración para `process_image()` (pipeline completo)
- **TestIntegration**: Pruebas de consistencia del pipeline completo
- **TestErrorHandling**: Pruebas para entradas inválidas y manejo de errores
- **TestPerformance**: Benchmarks de rendimiento para imágenes grandes
- **TestAzureIntegration**: Pruebas de integración con Azure Blob Storage
- **TestLoggingFunctionality**: Pruebas del sistema de logging de predicciones

### 2. Pruebas de Integración del Modelo (`test_model.py`)
Pruebas que requieren el modelo ONNX real:
- **TestModelResponsiveness**: Verifica carga, formas de entrada/salida, tiempo de inferencia
- **TestModelMetricStability**: Calcula IoU contra ground truth sintético, consistencia entre ejecuciones
- **TestModelRobustness**: Casos extremos (imágenes negro/blanco, tamaños pequeños, rectangulares, escala de grises)
- **TestModelPerformance**: Procesamiento por lotes, estabilidad de memoria

### 3. Pruebas de Componentes de Streamlit (`test_streamlit_components.py`)
Pruebas de funcionalidad de UI y específica de Streamlit:
- **TestStreamlitComponents**: Pruebas de componentes básicos y mensajes de error
- **TestImageUploadProcessing**: Manejo de archivos subidos, extracción de metadatos
- **TestDownloadFunctionality**: Generación de nombres de archivo de descarga
- **TestSessionStateHandling**: Persistencia y recuperación del estado de sesión
- **TestUIComponents**: Configuración de layouts, columnas, file uploader
- **TestErrorHandlingUI**: Visualización de errores, spinners de procesamiento
- **TestButtonInteractions**: Configuración y comportamiento de botones
- **TestInfoMessages**: Mensajes de éxito, información y expandibles

### 4. Utilidades de Prueba (`test_utils.py`)
Clases y funciones ayudantes para pruebas:
- **TestImageGenerator**: Crea imágenes sintéticas (colores sólidos, gradientes, tableros, círculos)
- **TestDataHelpers**: Utilidades de archivos temporales, conversión bytes/imagen
- **TestAssertions**: Funciones de validación para imágenes, arrays y máscaras
- **MockModelHelpers**: Creadores de sesiones ONNX simuladas (aleatorias y determinísticas)
- **PerformanceTestHelpers**: Medición de tiempo de ejecución y uso de memoria

## Ejecución de Pruebas

### Pruebas con Docker

La aplicación proporciona pruebas basadas en Docker para entornos de prueba consistentes y aislados utilizando un enfoque sencillo de Dockerfile multi-etapa.

#### Prerrequisitos para Pruebas con Docker
- Docker instalado
- Ejecutar desde el directorio raíz del proyecto

#### Pruebas Rápidas con Docker

**Construir imagen de prueba:**
```bash
docker build --target test -t bg-removal:test .
```

**Ejecutar todas las pruebas con cobertura:**
```bash
# Windows PowerShell
docker run --rm -v "${PWD}/htmlcov:/app/htmlcov" bg-removal:test python run_tests.py --coverage

# Linux/Mac/Git Bash  
docker run --rm -v "$(pwd)/htmlcov:/app/htmlcov" bg-removal:test python run_tests.py --coverage
```

**Ejecutar pruebas rápidas (excluyendo pruebas del modelo):**
```bash
docker run --rm bg-removal:test python run_tests.py --fast
```

**Ejecutar categorías de prueba específicas:**
```bash
docker run --rm bg-removal:test python run_tests.py --unit
docker run --rm bg-removal:test python run_tests.py --model
docker run --rm bg-removal:test python run_tests.py --streamlit
```

#### Pruebas con Docker Compose

**Ejecutar todas las pruebas:**
```bash
docker-compose exec bg-remover-app python run_tests.py
```

**Ejecutar con cobertura:**
```bash
docker-compose exec bg-remover-app python run_tests.py --coverage
```

**Ejecutar pruebas específicas:**
```bash
docker-compose exec bg-remover-app python -m pytest tests/test_app.py -v
```

#### Flujo de Trabajo de Pruebas con Docker

1. **Construir una vez**: `docker build --target test -t bg-removal:test .`
2. **Ejecutar según sea necesario**: Usar diferentes `--flags` con `run_tests.py`
3. **Informes de cobertura**: Montar volumen `htmlcov/` cuando se usa `--coverage`

## Marcadores de Prueba

Las pruebas están organizadas usando marcadores de pytest:

- `@pytest.mark.unit`: Pruebas unitarias para funciones individuales
- `@pytest.mark.integration`: Pruebas de integración para flujos de trabajo completos
- `@pytest.mark.model`: Pruebas que requieren el archivo del modelo real
- `@pytest.mark.performance`: Pruebas de rendimiento y tiempo
- `@pytest.mark.slow`: Pruebas que tardan más en ejecutarse

**Ejecutar pruebas por marcador:**
```bash
pytest tests/ -m "unit" -v
pytest tests/ -m "not model" -v  # Excluir pruebas del modelo
```

## Configuración de Pruebas

La configuración se maneja a través de `pytest.ini`:
- Patrones de descubrimiento de pruebas
- Configuración de cobertura
- Filtros de advertencias
- Formato de salida

## Benchmarks de Rendimiento

Las pruebas de rendimiento verifican:
- **Tiempo de preprocesamiento**: < 1 segundo para imágenes grandes
- **Tiempo de inferencia**: < 10 segundos por imagen
- **Estabilidad de memoria**: No hay fugas de memoria durante el procesamiento por lotes

## Pruebas de Errores

Las pruebas exhaustivas de manejo de errores cubren:
- **Entradas inválidas**: Valores nulos, tipos incorrectos
- **Operaciones de archivo**: Archivos faltantes, errores de permiso
- **Errores del modelo**: Fallos de carga, errores de inferencia
- **Desajustes de tamaño**: Dimensiones incompatibles de imagen/máscara

## Objetivos de Cobertura

Métricas de cobertura objetivo:
- **General**: > 85%
- **Funciones principales**: 100%
- **Manejo de errores**: > 95%
- **Componentes de UI**: > 80%

## Integración con CI/CD

Las pruebas están completamente integradas en el pipeline de GitHub Actions:

### Ejecución Automática
- **Trigger**: Push a ramas `dev` y `prod`
- **Pull Requests**: Hacia rama `prod`
- **Manual**: `workflow_dispatch`

### Proceso en GitHub Actions
1. **Setup del entorno** (Python 3.9, cache de dependencias)
2. **Instalación** de dependencias y herramientas de testing
3. **Ejecución** de suite completa de pruebas
4. **Generación** de reportes de cobertura
5. **Bloqueo** del deployment si las pruebas fallan

### Requisitos para Deployment
- **Todas las pruebas** deben pasar
- **Cobertura mínima** del 85%
- **Sin errores críticos** en el pipeline

## Solución de Problemas

### Comandos de Depuración

**Depuración local:**
```bash
pytest tests/test_app.py::TestImageProcessing::test_specific_function -v -s
```

**Depuración con Docker:**
```bash
# Shell interactivo en el contenedor de prueba
docker run --rm -it bg-removal:test bash

# Verificar contenido del contenedor
docker run --rm bg-removal:test ls -la /app/tests/

# Ejecutar prueba específica en el contenedor
docker run --rm bg-removal:test pytest tests/test_app.py -v
```
