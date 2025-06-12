import os
import io
import sys
import pytest
import numpy as np

from PIL import Image
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from app import (
    load_model,
    preprocess_image,
    postprocess_mask,
    apply_mask_to_image,
    process_image,
    image_to_bytes,
    get_azure_config,
    download_model_from_azure,
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

class TestModelLoading:
    
    def test_load_model_file_not_exists(self):
        with patch('os.path.exists', return_value=False):
            with patch('app.download_model_from_azure', return_value=False):
                with patch('streamlit.error') as mock_error:
                    with patch('streamlit.info') as mock_info:
                        with patch('streamlit.markdown') as mock_markdown:
                            result = load_model()
                            
                            assert result is None
                            mock_error.assert_called_once()
                            mock_info.assert_called_once_with("🔍 Modelo no encontrado localmente. Intentando descargar desde Azure...")

    @patch('app.MODEL_PATH', 'models/production/u2net.onnx')
    @patch('os.path.exists', return_value=True)
    @patch('onnxruntime.InferenceSession')
    def test_load_model_success(self, mock_session, mock_exists):
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        with patch('streamlit.info'), patch('streamlit.success'):
            result = load_model()
        
        assert result == mock_session_instance
        mock_session.assert_called_once_with("models/production/u2net.onnx")
    
    @patch('app.MODEL_PATH', 'models/production/u2net.onnx')
    @patch('os.path.exists', return_value=True)
    @patch('onnxruntime.InferenceSession')
    def test_load_model_exception(self, mock_session, mock_exists):
        mock_session.side_effect = Exception("Model loading failed")
        
        with patch('streamlit.info'), patch('streamlit.error') as mock_error:
            result = load_model()
            
            assert result is None
            mock_error.assert_called_once()


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
    
    def test_download_model_no_storage_account(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('streamlit.warning') as mock_warning:
                result = download_model_from_azure()
                
                assert result is False
                mock_warning.assert_called_with("⚠️ Credenciales de Azure incompletas.")
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    @patch('os.makedirs')
    def test_download_model_success(self, mock_makedirs, mock_blob_service, mock_credential):
        # Setup mocks
        mock_blob = Mock()
        mock_blob.name = 'production/u2net.onnx'
        mock_blob.last_modified = '2023-01-01'
        
        mock_container_client = Mock()
        mock_container_client.get_container_properties.return_value = True
        mock_container_client.list_blobs.return_value = [mock_blob]
        
        mock_blob_client = Mock()
        mock_blob_client.download_blob.return_value.readall.return_value = b'fake model data'
        
        mock_blob_service_instance = Mock()
        mock_blob_service_instance.get_container_client.return_value = mock_container_client
        mock_blob_service_instance.get_blob_client.return_value = mock_blob_client
        mock_blob_service.return_value = mock_blob_service_instance
        
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client_id',
            'AZURE_CLIENT_SECRET': 'test_secret', 
            'AZURE_TENANT_ID': 'test_tenant'
        }):
            with patch('streamlit.spinner'), patch('streamlit.success') as mock_success:
                with patch('builtins.open', create=True) as mock_open:
                    mock_file = Mock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    result = download_model_from_azure()
                    
                    assert result is True
                    mock_file.write.assert_called_with(b'fake model data')
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    def test_download_model_blob_not_exists(self, mock_blob_service, mock_credential):
        # Setup mocks
        mock_container_client = Mock()
        mock_container_client.get_container_properties.return_value = True
        mock_container_client.list_blobs.return_value = []  # No blobs found
        
        mock_blob_service_instance = Mock()
        mock_blob_service_instance.get_container_client.return_value = mock_container_client
        mock_blob_service.return_value = mock_blob_service_instance
        
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client_id',
            'AZURE_CLIENT_SECRET': 'test_secret',
            'AZURE_TENANT_ID': 'test_tenant'
        }):
            with patch('streamlit.spinner'), patch('streamlit.error') as mock_error:
                with patch('os.makedirs'):
                    result = download_model_from_azure()
                    
                    assert result is False
                    mock_error.assert_called_with("❌ No se encontraron archivos .onnx en la ruta production/")
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    def test_download_model_exception_handling(self, mock_blob_service, mock_credential):
        # Setup mock to raise exception
        mock_blob_service.side_effect = Exception("Azure connection failed")
        
        with patch.dict(os.environ, {
            'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage',
            'AZURE_CLIENT_ID': 'test_client_id',
            'AZURE_CLIENT_SECRET': 'test_secret',
            'AZURE_TENANT_ID': 'test_tenant'
        }):
            with patch('streamlit.error') as mock_error:
                result = download_model_from_azure()
                
                assert result is False
                mock_error.assert_called_with("❌ Error al descargar el modelo: Azure connection failed")
    
    @patch('app.MODEL_PATH', None)
    @patch('app.get_latest_model_path')
    @patch('os.path.exists')
    @patch('app.download_model_from_azure')
    @patch('onnxruntime.InferenceSession')
    def test_load_model_with_azure_download_success(self, mock_session, mock_download, mock_exists, mock_get_latest):
        # First call returns False (no local model), subsequent calls return True (after download)
        mock_exists.side_effect = [False, True]
        mock_download.return_value = True
        mock_get_latest.return_value = 'models/production/u2net.onnx'
        
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        with patch('streamlit.info') as mock_info:
            with patch('streamlit.success') as mock_success:
                result = load_model()
                
                assert result == mock_session_instance
                mock_info.assert_called_with("🔍 Modelo no encontrado localmente. Intentando descargar desde Azure...")
                mock_download.assert_called_once()
                mock_success.assert_called_with("🎯 Modelo cargado exitosamente!")
    
    @patch('os.path.exists', return_value=False)
    @patch('app.download_model_from_azure', return_value=False)
    def test_load_model_azure_download_fails(self, mock_download, mock_exists):
        with patch('streamlit.info') as mock_info:
            with patch('streamlit.error') as mock_error:
                with patch('streamlit.markdown') as mock_markdown:
                    result = load_model()
                    
                    assert result is None
                    mock_info.assert_called_once_with("🔍 Modelo no encontrado localmente. Intentando descargar desde Azure...")
                    mock_download.assert_called_once()
                    mock_error.assert_called_with("❌ No se pudo obtener el modelo desde Azure Blob Storage")


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