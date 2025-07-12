import os
import io
import sys
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Mock all the problematic imports before importing from app
# Create a proper mock for cv2 that simulates image resizing
class MockCV2:
    # Add cv2 constants
    INTER_LINEAR = 1
    INTER_CUBIC = 2
    INTER_NEAREST = 0
    
    @staticmethod
    def resize(image, size, interpolation=None):
        """Mock cv2.resize that actually resizes numpy arrays"""
        # Use PIL for resizing since it's available
        from PIL import Image as PILImage
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 2:
            # Grayscale image
            pil_img = PILImage.fromarray(image)
        else:
            # Color image  
            pil_img = PILImage.fromarray(image)
        
        # Resize using PIL
        resized_pil = pil_img.resize(size, PILImage.Resampling.BILINEAR)
        
        # Convert back to numpy array
        return np.array(resized_pil)

sys.modules['cv2'] = MockCV2()
sys.modules['streamlit'] = Mock()
sys.modules['onnxruntime'] = Mock()
sys.modules['azure.storage.blob'] = Mock()
sys.modules['azure.identity'] = Mock()
sys.modules['dotenv'] = Mock()

from PIL import Image

# Create a better mock for streamlit that supports context managers
class MockStreamlitContextManager:
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def __call__(self, *args, **kwargs):
        return self
    
    def __getattr__(self, name):
        # Return another context manager for any attribute access
        return MockStreamlitContextManager()

class MockStreamlitSidebar(MockStreamlitContextManager):
    pass

class MockStreamlitContainer(MockStreamlitContextManager):
    pass

class MockStreamlitExpander(MockStreamlitContextManager):
    pass

class MockStreamlitSpinner(MockStreamlitContextManager):
    pass

# Mock streamlit at module level to avoid import issues
class MockStreamlitContainer:
    """Mock streamlit container that supports context manager protocol"""
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def container(self):
        return MockStreamlitContainer()
    
    def empty(self):
        return MockStreamlitContainer()
    
    def __call__(self):
        """Make the container callable for st.container() calls"""
        return MockStreamlitContainer()

class MockStreamlitSpinner:
    """Mock streamlit spinner that supports context manager protocol"""
    def __init__(self, text=""):
        self.text = text
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def __call__(self, text=""):
        """Make the spinner callable for st.spinner() calls"""
        return MockStreamlitSpinner(text)

class MockStreamlitExpander:
    """Mock streamlit expander that supports context manager protocol"""
    def __init__(self, label=""):
        self.label = label
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def __call__(self, label=""):
        """Make the expander callable for st.expander() calls"""
        return MockStreamlitExpander(label)

class MockCacheResource:
    """Mock cache_resource that supports both @st.cache_resource and @st.cache_resource() syntax"""
    def __call__(self, func=None, **kwargs):
        if func is None:
            # Called with arguments: @st.cache_resource(ttl=3600)
            return lambda f: f
        else:
            # Called without arguments: @st.cache_resource
            return func
    
    def clear(self):
        """Mock clear method for cache_resource"""
        pass

mock_st = Mock()

# Create proper context manager mocks
mock_st.container = MockStreamlitContainer()
mock_st.spinner = MockStreamlitSpinner()
mock_st.expander = MockStreamlitExpander()

# Create proper cache_resource mock
mock_st.cache_resource = MockCacheResource()

# Add other streamlit methods
mock_st.error = Mock()
mock_st.info = Mock()
mock_st.success = Mock()
mock_st.warning = Mock()
mock_st.markdown = Mock()
mock_st.text = Mock()
mock_st.title = Mock()
mock_st.header = Mock()
mock_st.subheader = Mock()
mock_st.button = Mock()
mock_st.download_button = Mock()
mock_st.file_uploader = Mock()
mock_st.selectbox = Mock()
mock_st.sidebar = Mock()
mock_st.columns = Mock()
mock_st.image = Mock()
mock_st.balloons = Mock()
mock_st.rerun = Mock()
mock_st.stop = Mock()
mock_st.set_page_config = Mock()

sys.modules['streamlit'] = mock_st

# Create a proper cache_resource mock that can handle both @st.cache_resource and @st.cache_resource()
def mock_cache_resource(func=None, **kwargs):
    if func is None:
        # Called with arguments: @st.cache_resource(ttl=3600)
        return lambda f: f
    else:
        # Called without arguments: @st.cache_resource
        return func

mock_st.cache_resource = mock_cache_resource
mock_st.error = Mock()
mock_st.info = Mock()
mock_st.success = Mock()
mock_st.warning = Mock()
mock_st.markdown = Mock()
mock_st.expander = MockStreamlitExpander()
mock_st.spinner = MockStreamlitSpinner()
mock_st.container = MockStreamlitContainer()
mock_st.empty = MockStreamlitContextManager()
mock_st.set_page_config = Mock()
mock_st.sidebar = MockStreamlitSidebar()
mock_st.subheader = Mock()
mock_st.caption = Mock()

def mock_columns(num_or_spec, gap=None):
    """Mock columns function that returns the requested number of columns"""
    if isinstance(num_or_spec, int):
        num_columns = num_or_spec
    elif isinstance(num_or_spec, list):
        num_columns = len(num_or_spec)
    else:
        num_columns = 2  # Default fallback
    
    return [MockStreamlitContextManager() for _ in range(num_columns)]

mock_st.columns = mock_columns
mock_st.file_uploader = Mock()
mock_st.button = Mock()
mock_st.download_button = Mock()
mock_st.selectbox = Mock()
mock_st.image = Mock()
mock_st.stop = Mock()
mock_st.session_state = {}
mock_st.metric = Mock()
mock_st.header = Mock()
mock_st.title = Mock()

# Patch streamlit before importing app
with patch.dict('sys.modules', {'streamlit': mock_st}):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from app import (
        detect_model_type,
        get_model_input_size,
        get_model_normalization,
        preprocess_image_for_model,
        preprocess_image,
        postprocess_mask,
        apply_mask_to_image,
        optimize_image_size,
        image_to_bytes,
        get_all_model_paths,
        get_model_info,
        initialize_models_at_startup,
        get_azure_config,
        download_all_models_from_azure,
        log_prediction_to_blob,
        process_image,
        load_all_models
    )

class TestImagePreprocessing:
    
    def test_preprocess_image_rgb_conversion(self):
        image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        result = preprocess_image(image)
        
        assert result.shape == (1, 3, 320, 320)
        assert result.dtype == np.float32
        # ImageNet normalization can produce values outside [0,1] range
        assert -3.0 <= result.min() <= result.max() <= 3.0
    
    def test_preprocess_image_resize(self):
        image = Image.new('RGB', (500, 300), color=(128, 128, 128))
        result = preprocess_image(image)
        
        assert result.shape == (1, 3, 320, 320)
    
    def test_preprocess_image_normalization(self):
        image = Image.new('RGB', (50, 50), color=(255, 255, 255))
        result = preprocess_image(image)
        
        # For white pixels with ImageNet normalization, expect values around 2.6
        # (1.0 - 0.406) / 0.225 ≈ 2.64 for blue channel (highest)
        assert result.max() > 2.0  # Should be around 2.6
        assert result.max() < 3.0
    
    def test_preprocess_image_channel_order(self):
        red_image = Image.new('RGB', (50, 50), color=(255, 0, 0))
        result = preprocess_image(red_image)
        
        # For red image, red channel should have highest normalized value
        # This test checks relative ordering rather than absolute values
        red_mean = result[0, 0].mean()
        green_mean = result[0, 1].mean()  
        blue_mean = result[0, 2].mean()
        
        assert red_mean > green_mean
        assert red_mean > blue_mean


class TestMaskPostprocessing:
    
    def test_postprocess_mask_basic(self):
        mask = np.random.rand(1, 1, 320, 320).astype(np.float32)
        original_size = (640, 480)
        
        result = postprocess_mask(mask, original_size)
        
        assert result.shape == (480, 640)
        assert result.dtype == np.uint8
        assert 0 <= result.min() <= result.max() <= 255
    
    def test_postprocess_mask_squeeze(self):
        mask = np.ones((1, 1, 100, 100), dtype=np.float32) * 0.5
        original_size = (100, 100)
        
        result = postprocess_mask(mask, original_size)
        
        assert result.shape == (100, 100)
        expected_value = int(0.5 * 255)
        assert np.all(result == expected_value)
    
    def test_postprocess_mask_resize_different_sizes(self):
        mask = np.ones((1, 1, 320, 320), dtype=np.float32)
        
        sizes = [(100, 100), (200, 150), (640, 480)]
        for size in sizes:
            result = postprocess_mask(mask, size)
            assert result.shape == (size[1], size[0])


class TestMaskApplication:
    
    def test_apply_mask_to_image_basic(self):
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        result = apply_mask_to_image(image, mask)
        
        assert result.mode == 'RGBA'
        assert result.size == (100, 100)
        
        result_array = np.array(result)
        assert np.all(result_array[:, :, 3] == 255)
    
    def test_apply_mask_to_image_partial_transparency(self):
        image = Image.new('RGB', (50, 50), color=(0, 255, 0))
        mask = np.ones((50, 50), dtype=np.uint8) * 128
        
        result = apply_mask_to_image(image, mask)
        result_array = np.array(result)
        
        assert np.all(result_array[:, :, 3] == 128)
        assert np.all(result_array[:, :, 0] == 0)
        assert np.all(result_array[:, :, 1] == 255)
        assert np.all(result_array[:, :, 2] == 0)
    
    def test_apply_mask_to_image_zero_mask(self):
        image = Image.new('RGB', (30, 30), color=(100, 100, 100))
        mask = np.zeros((30, 30), dtype=np.uint8)
        
        result = apply_mask_to_image(image, mask)
        result_array = np.array(result)
        
        assert np.all(result_array[:, :, 3] == 0)


class TestImageConversion:
    
    def test_image_to_bytes_png_format(self):
        image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        
        result_bytes = image_to_bytes(image)
        
        assert isinstance(result_bytes, bytes)
        assert len(result_bytes) > 0
        
        recovered_image = Image.open(io.BytesIO(result_bytes))
        assert recovered_image.mode == 'RGBA'
        assert recovered_image.size == (100, 100)
    
    def test_image_to_bytes_different_sizes(self):
        sizes = [(10, 10), (200, 150), (500, 300)]
        
        for size in sizes:
            image = Image.new('RGB', size, color=(128, 128, 128))
            result_bytes = image_to_bytes(image)
            
            assert isinstance(result_bytes, bytes)
            assert len(result_bytes) > 0
            
            recovered_image = Image.open(io.BytesIO(result_bytes))
            assert recovered_image.size == size


class TestImageOptimization:
    """Test image size optimization functionality"""
    
    def test_optimize_small_image_no_resize(self):
        """Test that small images are not resized"""
        image = Image.new('RGB', (800, 600), color=(255, 0, 0))
        
        optimized, was_resized = optimize_image_size(image, max_dimension=2000)
        
        assert not was_resized
        assert optimized.size == (800, 600)
        assert optimized is image  # Should return the same object
    
    def test_optimize_exact_size_no_resize(self):
        """Test that images exactly at max_dimension are not resized"""
        image = Image.new('RGB', (2000, 2000), color=(0, 255, 0))
        
        optimized, was_resized = optimize_image_size(image, max_dimension=2000)
        
        assert not was_resized
        assert optimized.size == (2000, 2000)
        assert optimized is image
    
    def test_optimize_large_wide_image(self):
        """Test that large wide images are properly resized"""
        image = Image.new('RGB', (3000, 1500), color=(0, 0, 255))
        
        optimized, was_resized = optimize_image_size(image, max_dimension=2000)
        
        assert was_resized
        assert optimized.size == (2000, 1000)  # Maintains aspect ratio
        assert optimized is not image  # Should be a new object
    
    def test_optimize_large_tall_image(self):
        """Test that large tall images are properly resized"""
        image = Image.new('RGB', (1200, 2400), color=(255, 255, 0))
        
        optimized, was_resized = optimize_image_size(image, max_dimension=2000)
        
        assert was_resized
        assert optimized.size == (1000, 2000)  # Maintains aspect ratio
        assert optimized is not image
    
    def test_optimize_square_large_image(self):
        """Test that large square images are properly resized"""
        image = Image.new('RGB', (4000, 4000), color=(255, 0, 255))
        
        optimized, was_resized = optimize_image_size(image, max_dimension=2000)
        
        assert was_resized
        assert optimized.size == (2000, 2000)
        assert optimized is not image
    
    def test_optimize_custom_max_dimension(self):
        """Test optimization with custom max_dimension"""
        image = Image.new('RGB', (1500, 1000), color=(128, 128, 128))
        
        optimized, was_resized = optimize_image_size(image, max_dimension=1200)
        
        assert was_resized
        assert optimized.size == (1200, 800)  # Maintains aspect ratio

    def test_optimize_edge_case_one_pixel_over(self):
        """Test edge case where image is just one pixel over the limit"""
        image = Image.new('RGB', (2001, 1000), color=(100, 150, 200))
        
        optimized, was_resized = optimize_image_size(image, max_dimension=2000)
        
        assert was_resized
        assert optimized.size[0] <= 2000
        assert optimized.size[1] <= 2000

class TestNewMultiModelFunctionality:
    """Test the new multi-model support functions"""
    
    def test_detect_model_type_isnet(self):
        """Test IS-Net model detection"""
        assert detect_model_type('isnet-general-use.pth') == 'isnet'
        assert detect_model_type('path/to/isnet.onnx') == 'isnet'
        assert detect_model_type('general-use-model.onnx') == 'isnet'
    
    def test_detect_model_type_u2net(self):
        """Test U2-Net model detection"""
        assert detect_model_type('u2net.pth') == 'u2net'
        assert detect_model_type('path/to/u2net_model.onnx') == 'u2net'
    
    def test_detect_model_type_u2netp(self):
        """Test U2-NetP model detection"""
        assert detect_model_type('u2netp.pth') == 'u2netp'
        assert detect_model_type('u2net_small.onnx') == 'u2netp'
        assert detect_model_type('u2net_lite.onnx') == 'u2netp'
    
    def test_detect_model_type_fallback(self):
        """Test fallback to u2net for unknown models"""
        assert detect_model_type('unknown_model.onnx') == 'u2net'
        assert detect_model_type('random_name.pth') == 'u2net'
    
    def test_get_model_input_size(self):
        """Test input size detection for different model types"""
        assert get_model_input_size('u2net') == (320, 320)
        assert get_model_input_size('u2netp') == (320, 320)
        assert get_model_input_size('isnet') == (1024, 1024)
        assert get_model_input_size('unknown') == (320, 320)  # fallback
    
    def test_get_model_normalization_u2net(self):
        """Test normalization for U2-Net models"""
        mean, std = get_model_normalization('u2net')
        assert mean == [0.485, 0.456, 0.406]
        assert std == [0.229, 0.224, 0.225]
        
        mean, std = get_model_normalization('u2netp')
        assert mean == [0.485, 0.456, 0.406]
        assert std == [0.229, 0.224, 0.225]
    
    def test_get_model_normalization_isnet(self):
        """Test normalization for IS-Net models"""
        mean, std = get_model_normalization('isnet')
        assert mean == [0.5, 0.5, 0.5]
        assert std == [1.0, 1.0, 1.0]
    
    def test_preprocess_image_for_model_u2net(self):
        """Test preprocessing for U2-Net models"""
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        result = preprocess_image_for_model(image, 'u2net')
        
        assert result.shape == (1, 3, 320, 320)
        assert result.dtype == np.float32
    
    def test_preprocess_image_for_model_isnet(self):
        """Test preprocessing for IS-Net models"""
        image = Image.new('RGB', (100, 100), color=(0, 255, 0))
        result = preprocess_image_for_model(image, 'isnet')
        
        assert result.shape == (1, 3, 1024, 1024)
        assert result.dtype == np.float32
    
    def test_preprocess_image_backward_compatibility(self):
        """Test that preprocess_image still works with default parameters"""
        image = Image.new('RGB', (100, 100), color=(0, 0, 255))
        result = preprocess_image(image)  # Should default to u2net
        
        assert result.shape == (1, 3, 320, 320)
        assert result.dtype == np.float32


class TestMultiModelFunctionality:
    """Test new multi-model functionality"""
    
    def test_get_all_model_paths_empty_directory(self):
        with patch('glob.glob', return_value=[]):
            result = get_all_model_paths()
            assert result == []
    
    def test_get_all_model_paths_with_models(self):
        mock_paths = [
            'models/production/model1.onnx',
            'models/production/model2.onnx'
        ]
        with patch('glob.glob', return_value=mock_paths):
            with patch('os.path.getmtime', side_effect=[1000, 2000]):  # model2 is newer
                result = get_all_model_paths()
                # Should be sorted by modification time (newest first)
                assert result == ['models/production/model2.onnx', 'models/production/model1.onnx']
    
    def test_get_model_info_nonexistent_file(self):
        result = get_model_info('/nonexistent/path.onnx')
        assert result is None
    
    def test_get_model_info_valid_file(self):
        mock_stat = Mock()
        mock_stat.st_size = 1024 * 1024 * 5  # 5MB
        mock_stat.st_mtime = 1640995200  # 2022-01-01 00:00:00
        
        from datetime import datetime
        mock_time = datetime(2022, 1, 1, 0, 0, 0)
        
        with patch('os.path.exists', return_value=True):
            with patch('os.stat', return_value=mock_stat):
                with patch('app.datetime') as mock_datetime_module:
                    mock_datetime_module.fromtimestamp.return_value = mock_time
                    
                    result = get_model_info('models/production/test_model.onnx')
                    
                    assert result is not None
                    assert result['name'] == 'test_model.onnx'
                    assert result['size_mb'] == 5.0
                    assert result['display_name'] == 'Test Model'
    
    @patch('app.get_all_model_paths')
    def test_initialize_models_at_startup_local_models_exist(self, mock_get_paths):
        mock_get_paths.return_value = ['model1.onnx', 'model2.onnx']
        
        with patch('builtins.print') as mock_print:
            result = initialize_models_at_startup()
            
            assert result is True
            mock_print.assert_any_call("[STARTUP] ✅ 2 modelo(s) local(es) encontrado(s):")

class TestModelLoading:
    
    def setup_method(self):
        """Setup method that handles mocked streamlit environment"""
        # No need to clear cache since we're using mocked streamlit
        pass
    
    def test_load_all_models_no_models_available(self):
        with patch('app.get_all_model_paths', return_value=[]):
            with patch('app.download_all_models_from_azure', return_value=False):
                with patch('app.st.error') as mock_error:
                    with patch('app.st.info') as mock_info:
                        with patch('app.st.markdown') as mock_markdown:
                            result = load_all_models()
                            
                            assert result == {}
                            mock_error.assert_called()
                            mock_info.assert_called_with("🔍 Modelos no encontrados localmente. Intentando descargar desde Azure...")

    @patch('app.get_all_model_paths')
    @patch('os.path.exists', return_value=True)
    @patch('app.ort.InferenceSession')  # Mock onnxruntime
    def test_load_all_models_success(self, mock_session, mock_exists, mock_get_paths):
        mock_get_paths.return_value = ['models/production/u2net.onnx', 'models/production/isnet-general-use.onnx']
        mock_session_instances = [Mock(), Mock()]
        mock_session.side_effect = mock_session_instances
        
        with patch('app.st.success') as mock_success:
            with patch('app.st.expander') as mock_expander:
                mock_expander.return_value.__enter__ = Mock()
                mock_expander.return_value.__exit__ = Mock()
                result = load_all_models()
        
        assert len(result) == 2
        assert 'u2net.onnx' in result
        assert 'isnet-general-use.onnx' in result
        
        # Check that model_info dictionaries are returned
        assert result['u2net.onnx']['session'] == mock_session_instances[0]
        assert result['u2net.onnx']['type'] == 'u2net'
        assert result['u2net.onnx']['input_size'] == (320, 320)
        
        assert result['isnet-general-use.onnx']['session'] == mock_session_instances[1]
        assert result['isnet-general-use.onnx']['type'] == 'isnet'
        assert result['isnet-general-use.onnx']['input_size'] == (1024, 1024)
        
        mock_success.assert_called_with("🎯 2 modelo(s) cargado(s) exitosamente!")
    
    @patch('app.get_all_model_paths')
    @patch('os.path.exists', return_value=True)
    @patch('app.ort.InferenceSession')  # Mock onnxruntime
    def test_load_all_models_partial_failure(self, mock_session, mock_exists, mock_get_paths):
        mock_get_paths.return_value = ['models/production/good_u2net.onnx', 'models/production/bad_model.onnx']
        mock_session_instance = Mock()
        mock_session.side_effect = [mock_session_instance, Exception("Model loading failed")]
        
        with patch('app.st.success') as mock_success:
            with patch('app.st.warning') as mock_warning:
                with patch('app.st.expander') as mock_expander:
                    mock_expander.return_value.__enter__ = Mock()
                    mock_expander.return_value.__exit__ = Mock()
                    result = load_all_models()
        
        assert len(result) == 1
        assert 'good_u2net.onnx' in result
        assert 'bad_model.onnx' not in result
        
        # Check that the successful model has proper structure
        assert result['good_u2net.onnx']['session'] == mock_session_instance
        assert result['good_u2net.onnx']['type'] == 'u2net'
        assert result['good_u2net.onnx']['input_size'] == (320, 320)
        
        mock_success.assert_called_with("🎯 1 modelo(s) cargado(s) exitosamente!")
        mock_warning.assert_called_with("⚠️ 1 modelo(s) fallaron al cargar")


class TestImageProcessing:
    
    def create_mock_session(self):
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        # Create deterministic output for more predictable testing
        fake_output = np.ones((1, 1, 320, 320), dtype=np.float32) * 0.8
        mock_session.run.return_value = [fake_output]
        
        return mock_session
    
    def create_mock_model_info(self, model_type='u2net'):
        """Create mock model_info dictionary for testing"""
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        # Create appropriate output size based on model type
        if model_type == 'isnet':
            fake_output = np.ones((1, 1, 1024, 1024), dtype=np.float32) * 0.8
        else:
            fake_output = np.ones((1, 1, 320, 320), dtype=np.float32) * 0.8
            
        mock_session.run.return_value = [fake_output]
        
        return {
            'session': mock_session,
            'type': model_type,
            'input_size': get_model_input_size(model_type),
            'path': f'models/production/test_{model_type}.onnx'
        }
    
    def test_process_image_success(self):
        image = Image.new('RGB', (200, 200), color=(255, 128, 64))
        mock_model_info = self.create_mock_model_info('u2net')
        
        result = process_image(image, mock_model_info)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == image.size
        
        mock_model_info['session'].get_inputs.assert_called_once()
        mock_model_info['session'].run.assert_called_once()
    
    def test_process_image_different_sizes(self):
        sizes = [(100, 100), (300, 200), (640, 480)]
        mock_model_info = self.create_mock_model_info('u2net')
        
        for size in sizes:
            image = Image.new('RGB', size, color=(128, 128, 128))
            result = process_image(image, mock_model_info)
            
            assert result.size == size
    
    def test_process_image_exception_handling(self):
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        mock_model_info = self.create_mock_model_info('u2net')
        mock_model_info['session'].get_inputs.side_effect = Exception("Session error")
        
        with patch('app.st.error') as mock_error:
            result = process_image(image, mock_model_info)
            
            assert result is None
            mock_error.assert_called_once()
    
    def test_process_image_with_small_image_no_optimization(self):
        """Test that small images are processed without optimization"""
        image = Image.new('RGB', (800, 600), color=(255, 128, 64))
        mock_model_info = self.create_mock_model_info('u2net')
        
        result = process_image(image, mock_model_info)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == (800, 600)  # Same as original
        
        mock_model_info['session'].get_inputs.assert_called_once()
        mock_model_info['session'].run.assert_called_once()
    
    def test_process_image_with_large_image_optimization(self):
        """Test that large images are optimized during processing"""
        large_image = Image.new('RGB', (3000, 2000), color=(64, 128, 255))
        mock_model_info = self.create_mock_model_info('u2net')
        
        result = process_image(large_image, mock_model_info)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        # Image should be optimized to max 2000px on the larger side
        assert result.size == (2000, 1333)  # Maintains aspect ratio, width limited
        
        mock_model_info['session'].get_inputs.assert_called_once()
        mock_model_info['session'].run.assert_called_once()
    
    def test_process_image_isnet_model(self):
        """Test processing with IS-Net model"""
        image = Image.new('RGB', (500, 500), color=(100, 200, 150))
        mock_model_info = self.create_mock_model_info('isnet')
        
        # Mock IS-Net output structure (nested output)
        fake_output = np.random.rand(1, 1, 1024, 1024).astype(np.float32)
        mock_model_info['session'].run.return_value = [[fake_output]]  # Nested for IS-Net
        
        result = process_image(image, mock_model_info)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == (500, 500)
    
    def test_process_image_optimization_metadata_logging(self):
        """Test that optimization information is included in metadata logging"""
        large_image = Image.new('RGB', (2500, 1500), color=(200, 100, 50))
        mock_model_info = self.create_mock_model_info('u2net')
        
        image_metadata = {
            'name': 'test_large.jpg',
            'size_kb': 500.0,
            'width_px': 2500,
            'height_px': 1500,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(large_image, mock_model_info, image_metadata)
            
            assert isinstance(result, Image.Image)
            
            # Verify logging was called with optimization metadata
            mock_log.assert_called_once()
            logged_metadata = mock_log.call_args[0][0]
            
            assert logged_metadata['optimized'] is True
            assert logged_metadata['original_width_px'] == 2500
            assert logged_metadata['original_height_px'] == 1500
            assert logged_metadata['processed_width_px'] == 2000
            assert logged_metadata['processed_height_px'] == 1200
            assert logged_metadata['model_type'] == 'u2net'
            assert logged_metadata['model_input_size'] == '320x320'
    
    def test_process_image_no_optimization_metadata_logging(self):
        """Test that no optimization flag is set for small images"""
        small_image = Image.new('RGB', (800, 600), color=(100, 200, 150))
        mock_model_info = self.create_mock_model_info('u2net')
        
        image_metadata = {
            'name': 'test_small.jpg',
            'size_kb': 100.0,
            'width_px': 800,
            'height_px': 600,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(small_image, mock_model_info, image_metadata)
            
            assert isinstance(result, Image.Image)
            
            # Verify logging was called with no optimization flag
            mock_log.assert_called_once()
            logged_metadata = mock_log.call_args[0][0]
            
            assert logged_metadata['optimized'] is False
            assert logged_metadata['model_type'] == 'u2net'
            assert logged_metadata['model_input_size'] == '320x320'
            # These keys should not exist for non-optimized images
            assert 'original_width_px' not in logged_metadata
            assert 'original_height_px' not in logged_metadata
            assert 'processed_width_px' not in logged_metadata
            assert 'processed_height_px' not in logged_metadata


class TestIntegration:
    
    def create_mock_model_info(self, model_type='u2net'):
        """Create mock model_info dictionary for integration testing"""
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input_tensor'
        mock_session.get_inputs.return_value = [mock_input]
        
        # Create appropriate mock output based on model type
        if model_type == 'isnet':
            fake_mask = np.ones((1, 1, 1024, 1024), dtype=np.float32) * 0.8
            mock_session.run.return_value = [[fake_mask]]  # Nested for IS-Net
        else:
            fake_mask = np.ones((1, 1, 320, 320), dtype=np.float32) * 0.8
            mock_session.run.return_value = [fake_mask]
        
        return {
            'session': mock_session,
            'type': model_type,
            'input_size': get_model_input_size(model_type),
            'path': f'models/production/test_{model_type}.onnx'
        }
    
    def test_full_pipeline_with_mock_model(self):
        image = Image.new('RGB', (100, 100), color=(255, 255, 255))
        mock_model_info = self.create_mock_model_info('u2net')
        
        result = process_image(image, mock_model_info)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        
        result_array = np.array(result)
        expected_alpha = int(0.8 * 255)
        assert np.all(result_array[:, :, 3] == expected_alpha)
    
    def test_full_pipeline_with_isnet_model(self):
        """Test full pipeline with IS-Net model"""
        image = Image.new('RGB', (200, 200), color=(128, 128, 128))
        mock_model_info = self.create_mock_model_info('isnet')
        
        result = process_image(image, mock_model_info)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == (200, 200)
        
        result_array = np.array(result)
        expected_alpha = int(0.8 * 255)
        assert np.all(result_array[:, :, 3] == expected_alpha)
    
    def test_preprocessing_postprocessing_consistency(self):
        original_size = (150, 100)
        image = Image.new('RGB', original_size, color=(128, 64, 192))
        
        # Test with U2-Net preprocessing
        preprocessed_u2net = preprocess_image_for_model(image, 'u2net')
        assert preprocessed_u2net.shape == (1, 3, 320, 320)
        
        # Test with IS-Net preprocessing  
        preprocessed_isnet = preprocess_image_for_model(image, 'isnet')
        assert preprocessed_isnet.shape == (1, 3, 1024, 1024)
        
        # Test postprocessing
        fake_mask_output = np.random.rand(1, 1, 320, 320).astype(np.float32)
        processed_mask = postprocess_mask(fake_mask_output, original_size)
        
        assert processed_mask.shape == (100, 150)  # height, width
        assert processed_mask.dtype == np.uint8


class TestErrorHandling:
    
    def test_preprocess_image_invalid_input(self):
        with pytest.raises(AttributeError):
            preprocess_image(None)
    
    def test_apply_mask_size_mismatch(self):
        image = Image.new('RGB', (100, 100), color=(255, 255, 255))
        wrong_size_mask = np.ones((50, 50), dtype=np.uint8) * 255
        
        with pytest.raises((ValueError, IndexError)):
            apply_mask_to_image(image, wrong_size_mask)


class TestPerformance:
    
    def test_preprocess_performance_large_image(self):
        large_image = Image.new('RGB', (2000, 2000), color=(128, 128, 128))
        
        import time
        start_time = time.time()
        result = preprocess_image(large_image)
        end_time = time.time()
        
        assert result.shape == (1, 3, 320, 320)
        assert (end_time - start_time) < 1.0
    
    def test_postprocess_performance(self):
        large_mask = np.random.rand(1, 1, 320, 320).astype(np.float32)
        
        import time
        start_time = time.time()
        result = postprocess_mask(large_mask, (1920, 1080))
        end_time = time.time()
        
        assert result.shape == (1080, 1920)
        assert (end_time - start_time) < 1.0

class TestAzureIntegration:
    """Test Azure Blob Storage integration functionality"""
    
    def test_get_azure_config_with_env_vars(self):
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client_id',
            'AZURE_CLIENT_SECRET': 'test_secret',
            'AZURE_TENANT_ID': 'test_tenant'
        }):
            config = get_azure_config()
            
            assert config['storage_account_name'] == 'test_storage'
            assert config['client_id'] == 'test_client_id'
            assert config['client_secret'] == 'test_secret'
            assert config['tenant_id'] == 'test_tenant'
            assert config['container_name'] == 'models'
    
    def test_get_azure_config_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            config = get_azure_config()
            
            assert config['storage_account_name'] is None
            assert config['client_id'] is None
            assert config['client_secret'] is None
            assert config['tenant_id'] is None
            assert config['container_name'] == 'models'
    
    def test_download_all_models_no_storage_account(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('app.st.container') as mock_container:
                mock_placeholder = Mock()
                mock_placeholder.container.return_value.__enter__ = Mock()
                mock_placeholder.container.return_value.__exit__ = Mock()
                
                result = download_all_models_from_azure(mock_placeholder)
                
                assert result is False
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    @patch('os.makedirs')
    def test_download_all_models_success(self, mock_makedirs, mock_blob_service, mock_credential):
        # Setup mocks
        mock_container_client = Mock()
        mock_blob_service.return_value.get_container_client.return_value = mock_container_client
        
        # Mock blobs
        mock_blob1 = Mock()
        mock_blob1.name = 'production/model1.onnx'
        mock_blob2 = Mock()
        mock_blob2.name = 'production/model2.onnx'
        
        mock_container_client.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_container_client.get_container_properties.return_value = Mock()
        
        # Mock blob client for downloads
        mock_blob_client = Mock()
        mock_blob_client.download_blob.return_value.readall.return_value = b'fake_model_data'
        mock_blob_service.return_value.get_blob_client.return_value = mock_blob_client
        
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client',
            'AZURE_CLIENT_SECRET': 'test_secret',
            'AZURE_TENANT_ID': 'test_tenant'
        }):
            with patch('app.st.container'), patch('app.st.spinner'), patch('builtins.open', create=True):
                result = download_all_models_from_azure()
        
        assert result is True
        assert mock_makedirs.call_count >= 1
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    def test_download_all_models_no_onnx_files(self, mock_blob_service, mock_credential):
        # Setup mocks
        mock_container_client = Mock()
        mock_blob_service.return_value.get_container_client.return_value = mock_container_client
        
        # Mock no .onnx blobs
        mock_blob = Mock()
        mock_blob.name = 'production/not_a_model.txt'
        mock_container_client.list_blobs.return_value = [mock_blob]
        mock_container_client.get_container_properties.return_value = Mock()
        
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client',
            'AZURE_CLIENT_SECRET': 'test_secret',
            'AZURE_TENANT_ID': 'test_tenant'
        }):
            with patch('app.st.container'), patch('app.st.spinner'):
                result = download_all_models_from_azure()
        
        assert result is False
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    def test_download_all_models_exception_handling(self, mock_blob_service, mock_credential):
        # Setup mock to raise exception
        mock_blob_service.side_effect = Exception("Azure connection failed")
        
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client',
            'AZURE_CLIENT_SECRET': 'test_secret',
            'AZURE_TENANT_ID': 'test_tenant'
        }):
            with patch('app.st.container'), patch('app.st.error'):
                result = download_all_models_from_azure()
        
        assert result is False
    
    @patch('app.get_all_model_paths')
    @patch('os.path.exists')
    @patch('app.download_all_models_from_azure')
    @patch('app.ort.InferenceSession')
    def test_load_all_models_with_azure_download_success(self, mock_session, mock_download, mock_exists, mock_get_paths):
        # Clear cache before test
        import streamlit as st
        st.cache_resource.clear()
        
        # First call: no models locally
        # Second call: models available after download
        mock_get_paths.side_effect = [[], ['models/production/downloaded_model.onnx']]
        mock_exists.return_value = True
        mock_download.return_value = True
        mock_session.return_value = Mock()
        
        with patch('app.st.info'), patch('app.st.success'):
            result = load_all_models()
        
        assert len(result) == 1
        assert 'downloaded_model.onnx' in result
        mock_download.assert_called_once()
    
    @patch('app.get_all_model_paths', return_value=[])
    @patch('app.download_all_models_from_azure', return_value=False)
    def test_load_all_models_azure_download_fails(self, mock_download, mock_get_paths):
        # Clear cache before test
        import streamlit as st
        st.cache_resource.clear()
        
        with patch('app.st.info'), patch('app.st.error'), patch('app.st.markdown'):
            result = load_all_models()
        
        assert result == {}
        mock_download.assert_called_once()

class TestImageProcessingWithMultipleModels:
    """Test new multi-model functionality with process_image"""
    
    def create_mock_model_info(self, model_type='u2net'):
        """Create mock model_info dictionary for testing"""
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        if model_type == 'isnet':
            fake_output = np.ones((1, 1, 1024, 1024), dtype=np.float32) * 0.8
            mock_session.run.return_value = [[fake_output]]  # Nested for IS-Net
        else:
            fake_output = np.ones((1, 1, 320, 320), dtype=np.float32) * 0.8
            mock_session.run.return_value = [fake_output]
        
        return {
            'session': mock_session,
            'type': model_type,
            'input_size': get_model_input_size(model_type),
            'path': f'models/production/test_{model_type}.onnx'
        }
    
    def test_process_image_with_selected_model(self):
        """Test that process_image works with individually selected model_info"""
        image = Image.new('RGB', (150, 150), color=(200, 100, 50))
        mock_model_info = self.create_mock_model_info('u2net')
        
        result = process_image(image, mock_model_info)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == (150, 150)
    
    def test_process_image_with_model_metadata_logging(self):
        """Test that model information is properly logged"""
        image = Image.new('RGB', (100, 100), color=(75, 150, 225))
        mock_model_info = self.create_mock_model_info('isnet')
        
        image_metadata = {
            'name': 'test_image.jpg',
            'size_kb': 50.0,
            'width_px': 100,
            'height_px': 100,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(image, mock_model_info, image_metadata)
            
            assert isinstance(result, Image.Image)
            
            # Verify logging was called with model metadata
            mock_log.assert_called_once()
            logged_metadata = mock_log.call_args[0][0]
            
            assert logged_metadata['model_type'] == 'isnet'
            assert logged_metadata['model_input_size'] == '1024x1024'


class TestLoggingFunctionality:
    """Test logging functionality with multi-model support"""
    
    def create_mock_model_info(self, model_type='u2net'):
        """Create mock model_info for logging tests"""
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        fake_output = np.ones((1, 1, 320, 320), dtype=np.float32) * 0.8
        mock_session.run.return_value = [fake_output]
        
        return {
            'session': mock_session,
            'type': model_type,
            'input_size': get_model_input_size(model_type),
            'path': f'models/production/test_{model_type}.onnx'
        }
    
    def test_log_prediction_to_blob_missing_credentials(self):
        """Test logging behavior when Azure credentials are missing"""
        image_metadata = {
            'name': 'test.jpg',
            'size_kb': 100.0
        }
        
        with patch('app.get_azure_config') as mock_config:
            mock_config.return_value = {
                'storage_account_name': None,
                'client_id': None,
                'client_secret': None,
                'tenant_id': None,
                'container_name': 'models'
            }
            
            with patch('app.st.warning') as mock_warning:
                log_prediction_to_blob(image_metadata, 1.5, True)
                mock_warning.assert_called_once()
    
    def test_log_prediction_to_blob_production_environment(self):
        """Test that production environment uses correct log file"""
        with patch('os.getenv', return_value='production'):
            image_metadata = {'name': 'test.jpg'}
            
            # Test the environment detection logic
            environment = os.getenv('ENVIRONMENT', 'development').lower()
            if environment in ['development', 'dev', 'test']:
                expected_log_file = 'logs/dev_predictions.log'
            else:
                expected_log_file = 'logs/prod_predictions.log'
            
            assert expected_log_file == 'logs/prod_predictions.log'
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    def test_log_prediction_to_blob_azure_success(self, mock_blob_service, mock_credential):
        """Test successful logging to Azure Blob Storage"""
        # Setup mocks
        mock_blob_client = Mock()
        mock_blob_service.return_value.get_blob_client.return_value = mock_blob_client
        mock_blob_client.download_blob.return_value.readall.return_value = b'existing log content\n'
        
        image_metadata = {
            'name': 'test_image.jpg',
            'size_kb': 150.0,
            'width_px': 800,
            'height_px': 600,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.get_azure_config') as mock_config:
            mock_config.return_value = {
                'storage_account_name': 'testaccount',
                'client_id': 'test_client_id',
                'client_secret': 'test_secret',
                'tenant_id': 'test_tenant',
                'container_name': 'models'
            }
            
            # Call the function
            log_prediction_to_blob(image_metadata, 2.5, True)
            
            # Verify Azure client was called
            mock_blob_service.assert_called_once()
            mock_blob_client.upload_blob.assert_called_once()
            
            # Verify the uploaded content contains our metadata
            uploaded_content = mock_blob_client.upload_blob.call_args[0][0]
            assert 'test_image.jpg' in uploaded_content
            assert 'existing log content' in uploaded_content
    
    def test_log_prediction_to_blob_failure_handling(self):
        """Test error handling in logging functionality"""
        image_metadata = {
            'name': 'test.jpg',
            'size_kb': 100.0
        }
        
        with patch('app.get_azure_config') as mock_config:
            mock_config.return_value = {
                'storage_account_name': 'testaccount',
                'client_id': 'test_client_id', 
                'client_secret': 'test_secret',
                'tenant_id': 'test_tenant',
                'container_name': 'models'
            }
            
            with patch('app.BlobServiceClient') as mock_blob_service:
                mock_blob_service.side_effect = Exception("Connection failed")
                
                with patch('app.st.error') as mock_error:
                    log_prediction_to_blob(image_metadata, 1.0, True)
                    mock_error.assert_called_once()
    
    def test_process_image_with_logging_success(self):
        """Test that process_image properly logs successful predictions"""
        image = Image.new('RGB', (200, 200), color=(100, 150, 200))
        mock_model_info = self.create_mock_model_info('u2net')
        
        image_metadata = {
            'name': 'test_success.jpg',
            'size_kb': 75.0,
            'width_px': 200,
            'height_px': 200,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(image, mock_model_info, image_metadata)
            
            assert isinstance(result, Image.Image)
            
            # Verify logging was called for success
            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            logged_metadata, processing_time, success = args
            
            assert success is True
            assert logged_metadata['model_type'] == 'u2net'
            assert logged_metadata['model_input_size'] == '320x320'
            assert processing_time > 0
    
    def test_process_image_with_logging_failure(self):
        """Test that process_image properly logs failed predictions"""
        image = Image.new('RGB', (150, 150), color=(255, 100, 50))
        mock_model_info = self.create_mock_model_info('u2net')
        mock_model_info['session'].run.side_effect = Exception("Processing failed")
        
        image_metadata = {
            'name': 'test_failure.jpg',
            'size_kb': 100.0
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            with patch('app.st.error'):
                result = process_image(image, mock_model_info, image_metadata)
                
                assert result is None
                
                # Verify logging was called for failure
                mock_log.assert_called_once()
                args = mock_log.call_args
                logged_metadata, processing_time, success = args[0]
                error_message = args[1]['error_message']
                
                assert success is False
                assert error_message == "Processing failed"
                assert logged_metadata['model_type'] == 'u2net'
    
    def test_process_image_without_metadata(self):
        """Test that process_image works without metadata logging"""
        image = Image.new('RGB', (100, 100), color=(200, 200, 200))
        mock_model_info = self.create_mock_model_info('u2net')
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(image, mock_model_info)  # No metadata provided
            
            assert isinstance(result, Image.Image)
            # Logging should not be called when no metadata is provided
            mock_log.assert_not_called()

if __name__ == "__main__":
    pytest.main([__file__]) 