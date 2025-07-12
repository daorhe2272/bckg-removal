import os
import io
import tempfile
import numpy as np

from PIL import Image
from typing import Tuple, List

class TestImageGenerator:
    
    @staticmethod
    def create_solid_color_image(size: Tuple[int, int], color: Tuple[int, int, int]) -> Image.Image:
        return Image.new('RGB', size, color=color)
    
    @staticmethod
    def create_gradient_image(size: Tuple[int, int]) -> Image.Image:
        width, height = size
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                image_array[y, x] = [
                    int(255 * x / width),
                    int(255 * y / height),
                    128
                ]
        
        return Image.fromarray(image_array)
    
    @staticmethod
    def create_checkerboard_image(size: Tuple[int, int], square_size: int = 20) -> Image.Image:
        width, height = size
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                if (x // square_size + y // square_size) % 2 == 0:
                    image_array[y, x] = [255, 255, 255]
                else:
                    image_array[y, x] = [0, 0, 0]
        
        return Image.fromarray(image_array)
    
    @staticmethod
    def create_circle_image(size: Tuple[int, int], center: Tuple[int, int] = None, 
                           radius: int = None) -> Tuple[Image.Image, np.ndarray]:
        width, height = size
        if center is None:
            center = (width // 2, height // 2)
        if radius is None:
            radius = min(width, height) // 4
        
        image_array = np.zeros((height, width, 3), dtype=np.uint8)
        mask_array = np.zeros((height, width), dtype=np.float32)
        
        center_x, center_y = center
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= radius:
                    image_array[y, x] = [255, 255, 255]
                    mask_array[y, x] = 1.0
                else:
                    image_array[y, x] = [128, 128, 128]
                    mask_array[y, x] = 0.0
        
        return Image.fromarray(image_array), mask_array

class TestDataHelpers:
    
    @staticmethod
    def create_temp_image_file(image: Image.Image, suffix: str = '.png') -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            image.save(tmp_file.name)
            return tmp_file.name
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
        byte_arr = io.BytesIO()
        image.save(byte_arr, format=format)
        return byte_arr.getvalue()
    
    @staticmethod
    def bytes_to_image(data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data))

class TestAssertions:
    
    @staticmethod
    def assert_image_properties(image: Image.Image, expected_size: Tuple[int, int] = None,
                               expected_mode: str = None):
        assert isinstance(image, Image.Image), "Objeto PIL Image esperado"
        
        if expected_size:
            assert image.size == expected_size, f"Tamaño esperado {expected_size}, obtenido {image.size}"
        
        if expected_mode:
            assert image.mode == expected_mode, f"Modo esperado {expected_mode}, obtenido {image.mode}"
    
    @staticmethod
    def assert_array_properties(array: np.ndarray, expected_shape: Tuple = None,
                               expected_dtype: np.dtype = None, 
                               expected_range: Tuple[float, float] = None):
        assert isinstance(array, np.ndarray), "Array de numpy esperado"
        
        if expected_shape:
            assert array.shape == expected_shape, f"Forma esperada {expected_shape}, obtenida {array.shape}"
        
        if expected_dtype:
            assert array.dtype == expected_dtype, f"Tipo de dato esperado {expected_dtype}, obtenido {array.dtype}"
        
        if expected_range:
            min_val, max_val = expected_range
            assert min_val <= array.min() <= array.max() <= max_val, \
                f"Valores del array fuera del rango esperado [{min_val}, {max_val}]"
    
    @staticmethod
    def assert_mask_validity(mask: np.ndarray):
        assert len(mask.shape) == 2, "La máscara debe ser 2D"
        assert mask.dtype in [np.uint8, np.float32], "La máscara debe ser uint8 o float32"
        
        if mask.dtype == np.uint8:
            assert 0 <= mask.min() <= mask.max() <= 255, "Los valores de la máscara uint8 deben estar en [0, 255]"
        else:
            assert 0.0 <= mask.min() <= mask.max() <= 1.0, "Los valores de la máscara float32 deben estar en [0.0, 1.0]"

class MockModelHelpers:
    
    @staticmethod
    def create_mock_onnx_session():
        from unittest.mock import Mock
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_input.shape = [1, 3, 320, 320]
        mock_session.get_inputs.return_value = [mock_input]
        
        def mock_run(output_names, input_dict):
            fake_output = np.random.rand(1, 1, 320, 320).astype(np.float32)
            return [fake_output]
        
        mock_session.run = mock_run
        return mock_session
    
    @staticmethod
    def create_deterministic_mock_session(output_value: float = 0.5):
        from unittest.mock import Mock
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_input.shape = [1, 3, 320, 320]
        mock_session.get_inputs.return_value = [mock_input]
        
        def mock_run(output_names, input_dict):
            fake_output = np.full((1, 1, 320, 320), output_value, dtype=np.float32)
            return [fake_output]
        
        mock_session.run = mock_run
        return mock_session

class PerformanceTestHelpers:
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> Tuple[float, any]:
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result
    
    @staticmethod
    def measure_memory_usage():
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None 

class StreamlitMockHelper:
    """Helper class to create consistent streamlit mocks across test files"""
    
    @staticmethod
    def create_streamlit_mocks():
        """Create standardized streamlit mocks for testing"""
        from unittest.mock import Mock
        import sys
        
        # Create context manager mock classes
        class MockStreamlitContainer:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def container(self):
                return MockStreamlitContainer()
            def empty(self):
                return MockStreamlitContainer()
            def __call__(self):
                return MockStreamlitContainer()
        
        class MockStreamlitSpinner:
            def __init__(self, text=""):
                self.text = text
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def __call__(self, text=""):
                return MockStreamlitSpinner(text)
        
        class MockStreamlitExpander:
            def __init__(self, label=""):
                self.label = label
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def __call__(self, label=""):
                return MockStreamlitExpander(label)
        
        class MockCacheResource:
            def __call__(self, func=None, **kwargs):
                if func is None:
                    return lambda f: f
                else:
                    return func
            def clear(self):
                pass
        
        # Create main mock object
        mock_st = Mock()
        
        # Setup context manager mocks
        mock_st.container = MockStreamlitContainer()
        mock_st.spinner = MockStreamlitSpinner()
        mock_st.expander = MockStreamlitExpander()
        mock_st.cache_resource = MockCacheResource()
        
        # Setup basic streamlit methods
        for method in [
            'error', 'info', 'success', 'warning', 'markdown', 'text', 'title', 
            'header', 'subheader', 'button', 'download_button', 'file_uploader', 
            'selectbox', 'image', 'stop', 'set_page_config', 'balloons', 'rerun'
        ]:
            setattr(mock_st, method, Mock())
        
        # Setup complex methods
        mock_st.sidebar = Mock()
        mock_st.columns = Mock(side_effect=lambda x: [Mock() for _ in range(x)])
        mock_st.session_state = {}
        
        return mock_st
    
    @staticmethod
    def setup_all_mocks():
        """Setup all required mocks for testing environment"""
        import sys
        from unittest.mock import Mock
        
        # Mock problematic imports
        mock_modules = {
            'cv2': Mock(),
            'onnxruntime': Mock(), 
            'azure.storage.blob': Mock(),
            'azure.identity': Mock(),
            'dotenv': Mock()
        }
        
        for module_name, mock_module in mock_modules.items():
            sys.modules[module_name] = mock_module
        
        # Setup streamlit mock
        mock_st = StreamlitMockHelper.create_streamlit_mocks()
        sys.modules['streamlit'] = mock_st
        
        return mock_st 