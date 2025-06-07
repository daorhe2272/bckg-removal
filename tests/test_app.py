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
    download_model_from_azure
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
    
    @patch('os.path.exists', return_value=True)
    @patch('onnxruntime.InferenceSession')
    def test_load_model_success(self, mock_session, mock_exists):
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        
        result = load_model()
        
        assert result == mock_session_instance
        mock_session.assert_called_once_with("models/production/u2net.onnx")
    
    @patch('os.path.exists', return_value=True)
    @patch('onnxruntime.InferenceSession')
    def test_load_model_exception(self, mock_session, mock_exists):
        mock_session.side_effect = Exception("Model loading failed")
        
        with patch('streamlit.error') as mock_error:
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
            'MODEL_BLOB_PATH': 'models/test/model.onnx'
        }):
            config = get_azure_config()
            
            assert config['storage_account_name'] == 'test_storage'
            assert config['model_blob_path'] == 'models/test/model.onnx'
            assert config['container_name'] == 'models'
    
    def test_get_azure_config_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            config = get_azure_config()
            
            assert config['storage_account_name'] is None
            assert config['model_blob_path'] == 'models/production/u2net.onnx'
            assert config['container_name'] == 'models'
    
    def test_download_model_no_storage_account(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('streamlit.warning') as mock_warning:
                result = download_model_from_azure()
                
                assert result is False
                mock_warning.assert_called_with("⚠️ Azure Storage no configurado.")
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    @patch('os.makedirs')
    def test_download_model_success(self, mock_makedirs, mock_blob_service, mock_credential):
        # Setup mocks
        mock_blob_client = Mock()
        mock_blob_client.exists.return_value = True
        mock_blob_client.download_blob.return_value.readall.return_value = b'fake model data'
        
        mock_blob_service_instance = Mock()
        mock_blob_service_instance.get_blob_client.return_value = mock_blob_client
        mock_blob_service.return_value = mock_blob_service_instance
        
        with patch.dict(os.environ, {'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage'}):
            with patch('streamlit.spinner'), patch('streamlit.success') as mock_success:
                with patch('builtins.open', create=True) as mock_open:
                    mock_file = Mock()
                    mock_open.return_value.__enter__.return_value = mock_file
                    
                    result = download_model_from_azure()
                    
                    assert result is True
                    mock_success.assert_called_with("✅ Modelo descargado exitosamente desde Azure!")
                    mock_file.write.assert_called_with(b'fake model data')
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    def test_download_model_blob_not_exists(self, mock_blob_service, mock_credential):
        # Setup mocks
        mock_blob_client = Mock()
        mock_blob_client.exists.return_value = False
        
        mock_blob_service_instance = Mock()
        mock_blob_service_instance.get_blob_client.return_value = mock_blob_client
        mock_blob_service.return_value = mock_blob_service_instance
        
        with patch.dict(os.environ, {'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage'}):
            with patch('streamlit.spinner'), patch('streamlit.error') as mock_error:
                with patch('os.makedirs'):
                    result = download_model_from_azure()
                    
                    assert result is False
                    mock_error.assert_called_with("❌ Modelo no encontrado en Azure: models/production/u2net.onnx")
    
    @patch('app.DefaultAzureCredential')
    @patch('app.BlobServiceClient')
    def test_download_model_exception_handling(self, mock_blob_service, mock_credential):
        # Setup mock to raise exception
        mock_blob_service.side_effect = Exception("Azure connection failed")
        
        with patch.dict(os.environ, {'AZURE_STORAGE_ACCOUNT_NAME': 'test_storage'}):
            with patch('streamlit.error') as mock_error:
                with patch('streamlit.info') as mock_info:
                    result = download_model_from_azure()
                    
                    assert result is False
                    mock_error.assert_called_with("❌ Error al descargar el modelo desde Azure: Azure connection failed")
                    mock_info.assert_called_with("💡 Verifica la configuración de Azure y los permisos de acceso.")
    
    @patch('os.path.exists')
    @patch('app.download_model_from_azure')
    @patch('onnxruntime.InferenceSession')
    def test_load_model_with_azure_download_success(self, mock_session, mock_download, mock_exists):
        # First call returns False (no local model), subsequent calls return True (after download)
        mock_exists.side_effect = [False, True]
        mock_download.return_value = True
        
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


if __name__ == "__main__":
    pytest.main([__file__]) 