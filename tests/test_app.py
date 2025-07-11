import os
import io
import sys
import pytest
import numpy as np

from PIL import Image
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from app import (
    load_all_models,
    get_all_model_paths,
    get_model_info,
    initialize_models_at_startup,
    preprocess_image,
    postprocess_mask,
    apply_mask_to_image,
    process_image,
    image_to_bytes,
    get_azure_config,
    download_all_models_from_azure,
    log_prediction_to_blob,
    optimize_image_size
)

class TestImagePreprocessing:
    
    def test_preprocess_image_rgb_conversion(self):
        image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        result = preprocess_image(image)
        
        assert result.shape == (1, 3, 320, 320)
        assert result.dtype == np.float32
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_preprocess_image_resize(self):
        image = Image.new('RGB', (500, 300), color=(128, 128, 128))
        result = preprocess_image(image)
        
        assert result.shape == (1, 3, 320, 320)
    
    def test_preprocess_image_normalization(self):
        image = Image.new('RGB', (50, 50), color=(255, 255, 255))
        result = preprocess_image(image)
        
        expected_value = 255.0 / 255.0
        np.testing.assert_almost_equal(result.max(), expected_value, decimal=5)
    
    def test_preprocess_image_channel_order(self):
        red_image = Image.new('RGB', (50, 50), color=(255, 0, 0))
        result = preprocess_image(red_image)
        
        assert result[0, 0].mean() > result[0, 1].mean()
        assert result[0, 0].mean() > result[0, 2].mean()


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
        """Clear Streamlit cache before each test to avoid cached model sessions"""
        import streamlit as st
        st.cache_resource.clear()
    
    def test_load_all_models_no_models_available(self):
        with patch('app.get_all_model_paths', return_value=[]):
            with patch('app.download_all_models_from_azure', return_value=False):
                with patch('streamlit.error') as mock_error:
                    with patch('streamlit.info') as mock_info:
                        with patch('streamlit.markdown') as mock_markdown:
                            result = load_all_models()
                            
                            assert result == {}
                            mock_error.assert_called()
                            mock_info.assert_called_with("🔍 Modelos no encontrados localmente. Intentando descargar desde Azure...")

    @patch('app.get_all_model_paths')
    @patch('os.path.exists', return_value=True)
    @patch('onnxruntime.InferenceSession')
    def test_load_all_models_success(self, mock_session, mock_exists, mock_get_paths):
        mock_get_paths.return_value = ['models/production/model1.onnx', 'models/production/model2.onnx']
        mock_session_instances = [Mock(), Mock()]
        mock_session.side_effect = mock_session_instances
        
        with patch('streamlit.success') as mock_success:
            result = load_all_models()
        
        assert len(result) == 2
        assert 'model1.onnx' in result
        assert 'model2.onnx' in result
        assert result['model1.onnx'] == mock_session_instances[0]
        assert result['model2.onnx'] == mock_session_instances[1]
        mock_success.assert_called_with("🎯 2 modelo(s) cargado(s) exitosamente!")
    
    @patch('app.get_all_model_paths')
    @patch('os.path.exists', return_value=True)
    @patch('onnxruntime.InferenceSession')
    def test_load_all_models_partial_failure(self, mock_session, mock_exists, mock_get_paths):
        mock_get_paths.return_value = ['models/production/good_model.onnx', 'models/production/bad_model.onnx']
        mock_session_instance = Mock()
        mock_session.side_effect = [mock_session_instance, Exception("Model loading failed")]
        
        with patch('streamlit.success') as mock_success:
            with patch('streamlit.warning') as mock_warning:
                result = load_all_models()
        
        assert len(result) == 1
        assert 'good_model.onnx' in result
        assert 'bad_model.onnx' not in result
        mock_success.assert_called_with("🎯 1 modelo(s) cargado(s) exitosamente!")
        mock_warning.assert_called_with("⚠️ 1 modelo(s) fallaron al cargar")


class TestImageProcessing:
    
    def create_mock_session(self):
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        fake_output = np.random.rand(1, 1, 320, 320).astype(np.float32)
        mock_session.run.return_value = [fake_output]
        
        return mock_session
    
    def test_process_image_success(self):
        image = Image.new('RGB', (200, 200), color=(255, 128, 64))
        mock_session = self.create_mock_session()
        
        result = process_image(image, mock_session)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == image.size
        
        mock_session.get_inputs.assert_called_once()
        mock_session.run.assert_called_once()
    
    def test_process_image_different_sizes(self):
        sizes = [(100, 100), (300, 200), (640, 480)]
        mock_session = self.create_mock_session()
        
        for size in sizes:
            image = Image.new('RGB', size, color=(128, 128, 128))
            result = process_image(image, mock_session)
            
            assert result.size == size
    
    def test_process_image_exception_handling(self):
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        mock_session = Mock()
        mock_session.get_inputs.side_effect = Exception("Session error")
        
        with patch('streamlit.error') as mock_error:
            result = process_image(image, mock_session)
            
            assert result is None
            mock_error.assert_called_once()
    
    def test_process_image_with_small_image_no_optimization(self):
        """Test that small images are processed without optimization"""
        image = Image.new('RGB', (800, 600), color=(255, 128, 64))
        mock_session = self.create_mock_session()
        
        result = process_image(image, mock_session)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == (800, 600)  # Same as original
        
        mock_session.get_inputs.assert_called_once()
        mock_session.run.assert_called_once()
    
    def test_process_image_with_large_image_optimization(self):
        """Test that large images are optimized during processing"""
        large_image = Image.new('RGB', (3000, 2000), color=(64, 128, 255))
        mock_session = self.create_mock_session()
        
        result = process_image(large_image, mock_session)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        # Image should be optimized to max 2000px on the larger side
        assert result.size == (2000, 1333)  # Maintains aspect ratio, width limited
        
        mock_session.get_inputs.assert_called_once()
        mock_session.run.assert_called_once()
    
    def test_process_image_optimization_metadata_logging(self):
        """Test that optimization information is included in metadata logging"""
        large_image = Image.new('RGB', (2500, 1500), color=(200, 100, 50))
        mock_session = self.create_mock_session()
        
        image_metadata = {
            'name': 'test_large.jpg',
            'size_kb': 500.0,
            'width_px': 2500,
            'height_px': 1500,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(large_image, mock_session, image_metadata)
            
            assert isinstance(result, Image.Image)
            
            # Verify logging was called with optimization metadata
            mock_log.assert_called_once()
            logged_metadata = mock_log.call_args[0][0]
            
            assert logged_metadata['optimized'] is True
            assert logged_metadata['original_width_px'] == 2500
            assert logged_metadata['original_height_px'] == 1500
            assert logged_metadata['processed_width_px'] == 2000
            assert logged_metadata['processed_height_px'] == 1200
    
    def test_process_image_no_optimization_metadata_logging(self):
        """Test that no optimization flag is set for small images"""
        small_image = Image.new('RGB', (800, 600), color=(100, 200, 150))
        mock_session = self.create_mock_session()
        
        image_metadata = {
            'name': 'test_small.jpg',
            'size_kb': 100.0,
            'width_px': 800,
            'height_px': 600,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(small_image, mock_session, image_metadata)
            
            assert isinstance(result, Image.Image)
            
            # Verify logging was called with no optimization flag
            mock_log.assert_called_once()
            logged_metadata = mock_log.call_args[0][0]
            
            assert logged_metadata['optimized'] is False
            # These keys should not exist for non-optimized images
            assert 'original_width_px' not in logged_metadata
            assert 'original_height_px' not in logged_metadata
            assert 'processed_width_px' not in logged_metadata
            assert 'processed_height_px' not in logged_metadata


class TestIntegration:
    
    def test_full_pipeline_with_mock_model(self):
        image = Image.new('RGB', (100, 100), color=(255, 255, 255))
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input_tensor'
        mock_session.get_inputs.return_value = [mock_input]
        
        fake_mask = np.ones((1, 1, 320, 320), dtype=np.float32) * 0.8
        mock_session.run.return_value = [fake_mask]
        
        result = process_image(image, mock_session)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        
        result_array = np.array(result)
        expected_alpha = int(0.8 * 255)
        assert np.all(result_array[:, :, 3] == expected_alpha)
    
    def test_preprocessing_postprocessing_consistency(self):
        original_size = (150, 100)
        image = Image.new('RGB', original_size, color=(128, 64, 192))
        
        preprocessed = preprocess_image(image)
        assert preprocessed.shape == (1, 3, 320, 320)
        
        fake_mask_output = np.random.rand(1, 1, 320, 320).astype(np.float32)
        processed_mask = postprocess_mask(fake_mask_output, original_size)
        
        assert processed_mask.shape == (original_size[1], original_size[0])
        
        final_image = apply_mask_to_image(image, processed_mask)
        assert final_image.size == original_size
        assert final_image.mode == 'RGBA'


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
            with patch('streamlit.container') as mock_container:
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
            with patch('streamlit.container'), patch('streamlit.spinner'), patch('builtins.open', create=True):
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
            with patch('streamlit.container'), patch('streamlit.spinner'):
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
            with patch('streamlit.container'), patch('streamlit.error'):
                result = download_all_models_from_azure()
        
        assert result is False
    
    @patch('app.get_all_model_paths')
    @patch('os.path.exists')
    @patch('app.download_all_models_from_azure')
    @patch('onnxruntime.InferenceSession')
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
        
        with patch('streamlit.info'), patch('streamlit.success'):
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
        
        with patch('streamlit.info'), patch('streamlit.error'), patch('streamlit.markdown'):
            result = load_all_models()
        
        assert result == {}
        mock_download.assert_called_once()

class TestImageProcessingWithMultipleModels:
    """Test image processing with model selection"""
    
    def create_mock_session(self):
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        fake_output = np.random.rand(1, 1, 320, 320).astype(np.float32)
        mock_session.run.return_value = [fake_output]
        
        return mock_session
    
    def test_process_image_with_selected_model(self):
        """Test that process_image works with individually selected sessions"""
        image = Image.new('RGB', (200, 200), color=(255, 128, 64))
        mock_session = self.create_mock_session()
        
        result = process_image(image, mock_session)
        
        assert isinstance(result, Image.Image)
        assert result.mode == 'RGBA'
        assert result.size == image.size
        
        mock_session.get_inputs.assert_called_once()
        mock_session.run.assert_called_once()
    
    def test_process_image_with_model_metadata_logging(self):
        """Test that model information is included in metadata logging"""
        image = Image.new('RGB', (200, 200), color=(255, 128, 64))
        mock_session = self.create_mock_session()
        
        image_metadata = {
            'name': 'test_image.jpg',
            'size_kb': 50.0,
            'width_px': 200,
            'height_px': 200,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(image, mock_session, image_metadata)
            
            assert isinstance(result, Image.Image)
            mock_log.assert_called_once()
            # Model info should not be in this function anymore - it's added at UI level


class TestLoggingFunctionality:
    
    def test_log_prediction_to_blob_missing_credentials(self):
        image_metadata = {
            'name': 'test_image.jpg',
            'size_kb': 150.5,
            'width_px': 800,
            'height_px': 600,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch.dict(os.environ, {}, clear=True):
            with patch('streamlit.warning') as mock_warning:
                log_prediction_to_blob(image_metadata, 2.5, True)
                
                mock_warning.assert_called_once_with("⚠️ No se pueden registrar las predicciones: Credenciales de Azure incompletas para el almacenamiento de logs.")
    
    def test_log_prediction_to_blob_production_environment(self):
        """Test that production environment uses correct log file"""
        image_metadata = {'name': 'test.jpg'}
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}, clear=True):
            with patch('streamlit.warning') as mock_warning:
                log_prediction_to_blob(image_metadata, 1.0, True)
                
                mock_warning.assert_called_once_with("⚠️ No se pueden registrar las predicciones: Credenciales de Azure incompletas para el almacenamiento de logs.")
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    def test_log_prediction_to_blob_azure_success(self, mock_blob_service, mock_credential):
        image_metadata = {
            'name': 'test_image.png',
            'size_kb': 250.0,
            'width_px': 1024,
            'height_px': 768,
            'format': 'PNG',
            'mode': 'RGBA'
        }
        
        # Mock Azure Blob Service
        mock_blob_client = Mock()
        mock_blob_client.download_blob.return_value.readall.return_value = b'existing log content\n'
        
        mock_blob_service_instance = Mock()
        mock_blob_service_instance.get_blob_client.return_value = mock_blob_client
        mock_blob_service.return_value = mock_blob_service_instance
        
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client',
            'AZURE_CLIENT_SECRET': 'test_secret',
            'AZURE_TENANT_ID': 'test_tenant',
            'ENVIRONMENT': 'development'
        }):
            log_prediction_to_blob(image_metadata, 1.5, True)
            
            # Verify blob client was called with correct parameters
            mock_blob_service_instance.get_blob_client.assert_called_with(
                container='models', 
                blob='logs/dev_predictions.log'
            )
            mock_blob_client.download_blob.assert_called_once()
            mock_blob_client.upload_blob.assert_called_once()
            
            # Check that upload was called with the right content structure
            upload_call_args = mock_blob_client.upload_blob.call_args[0][0]
            assert 'existing log content' in upload_call_args
            assert 'test_image.png' in upload_call_args
            assert '"tiempo_procesamiento_segundos": 1.5' in upload_call_args
            assert ('"éxito": true' in upload_call_args or '"\\u00e9xito": true' in upload_call_args)
    
    def test_log_prediction_to_blob_failure_handling(self):
        image_metadata = {'name': 'test.jpg'}
        
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client',
            'AZURE_CLIENT_SECRET': 'test_secret',
            'AZURE_TENANT_ID': 'test_tenant'
        }):
            with patch('app.BlobServiceClient', side_effect=Exception("Azure error")):
                with patch('streamlit.error') as mock_error:
                    with patch('streamlit.expander') as mock_expander:
                        mock_expander.return_value.__enter__ = Mock()
                        mock_expander.return_value.__exit__ = Mock()
                        
                        try:
                            log_prediction_to_blob(image_metadata, 1.0, False, "Test error")
                            # Should not raise an exception
                        except Exception:
                            pytest.fail("log_prediction_to_blob should not raise exceptions")
                        
                        mock_error.assert_called_once()
    
    def test_process_image_with_logging_success(self):
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        fake_output = np.random.rand(1, 1, 320, 320).astype(np.float32)
        mock_session.run.return_value = [fake_output]
        
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        image_metadata = {
            'name': 'test.jpg',
            'size_kb': 10.5,
            'width_px': 100,
            'height_px': 100,
            'format': 'JPEG',
            'mode': 'RGB'
        }
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(image, mock_session, image_metadata)
            
            assert result is not None
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == image_metadata
            assert isinstance(call_args[0][1], float)
            assert call_args[1]['success'] is True
    
    def test_process_image_with_logging_failure(self):
        mock_session = Mock()
        mock_session.get_inputs.side_effect = Exception("Model error")
        
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        image_metadata = {'name': 'test.jpg'}
        
        with patch('app.log_prediction_to_blob') as mock_log:
            with patch('streamlit.error'):
                result = process_image(image, mock_session, image_metadata)
                
                assert result is None
                mock_log.assert_called_once()
                call_args = mock_log.call_args
                assert call_args[1]['success'] is False
                assert call_args[1]['error_message'] == "Model error"
    
    def test_process_image_without_metadata(self):
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_session.get_inputs.return_value = [mock_input]
        
        fake_output = np.random.rand(1, 1, 320, 320).astype(np.float32)
        mock_session.run.return_value = [fake_output]
        
        image = Image.new('RGB', (100, 100), color=(255, 0, 0))
        
        with patch('app.log_prediction_to_blob') as mock_log:
            result = process_image(image, mock_session)  # No metadata provided
            
            assert result is not None
            mock_log.assert_not_called()  # Should not log when no metadata provided

if __name__ == "__main__":
    pytest.main([__file__]) 