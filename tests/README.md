# Documentación de Pruebas

Este directorio contiene todas las pruebas unitarias e de integración para la aplicación de eliminación de fondos en imágenes.

## Estructura de Pruebas

```
tests/
├── __init__.py                     # Inicialización del paquete
├── test_app.py                     # Pruebas de funcionalidad central
├── test_model.py                   # Pruebas de integración del modelo
├── test_streamlit_components.py    # Pruebas de componentes de UI
├── test_utils.py                   # Utilidades y ayudantes de prueba
└── README.md                       # Esta documentación
```

## Categorías de Pruebas

### 1. Pruebas Unitarias (`test_app.py`)
Prueba funciones individuales de forma aislada:
- **Preprocesamiento de Imágenes**: Pruebas para la función `preprocess_image()`
- **Postprocesamiento de Máscaras**: Pruebas para la función `postprocess_mask()`
- **Aplicación de Máscaras**: Pruebas para la función `apply_mask_to_image()`
- **Conversión de Imágenes**: Pruebas para la función `image_to_bytes()`
- **Carga del Modelo**: Pruebas para la función `load_model()` con mocks
- **Procesamiento de Imágenes**: Pruebas de integración para la función `process_image()`
- **Manejo de Errores**: Pruebas para diversas condiciones de error
- **Rendimiento**: Benchmarks básicos de rendimiento

### 2. Pruebas de Integración del Modelo (`test_model.py`)
Pruebas que requieren el modelo ONNX real:
- **Capacidad de Respuesta del Modelo**: Verifica que el modelo carga y produce salida
- **Estabilidad de Métricas**: Calcula IoU contra ground truth sintético
- **Robustez del Modelo**: Prueba casos extremos (imágenes en blanco/negro, diferentes tamaños)
- **Rendimiento**: Mide el tiempo de inferencia y el uso de memoria

### 3. Pruebas de Componentes de Streamlit (`test_streamlit_components.py`)
Pruebas de funcionalidad de UI y específica de Streamlit:
- **Configuración de Componentes**: Pruebas de configuración de widgets de Streamlit
- **Procesamiento de Carga de Archivos**: Pruebas de manejo de carga de imágenes
- **Gestión del Estado de Sesión**: Pruebas de persistencia de estado
- **Interacciones de UI**: Pruebas de funcionalidad de botones y descarga
- **Visualización de Errores**: Pruebas de presentación de mensajes de error

### 4. Utilidades de Prueba (`test_utils.py`)
Clases y funciones ayudantes para pruebas:
- **TestImageGenerator**: Crea imágenes de prueba sintéticas
- **TestDataHelpers**: Utilidades de manipulación de archivos y datos
- **TestAssertions**: Funciones de aserción personalizadas
- **MockModelHelpers**: Creadores de sesiones ONNX simuladas
- **PerformanceTestHelpers**: Utilidades de medición de rendimiento

## Ejecución de Pruebas

### Pruebas con Docker

La aplicación proporciona pruebas basadas en Docker para entornos de prueba consistentes y aislados utilizando un enfoque sencillo de Dockerfile multi-etapa.

#### Prerrequisitos para Pruebas con Docker
- Docker instalado
- Ejecutar desde el directorio `src/´

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
- **General**: > 90%
- **Funciones principales**: 100%
- **Manejo de errores**: > 95%
- **Componentes de UI**: > 80%

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
