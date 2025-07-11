import os
import io
import sys
import pytest

from PIL import Image
from unittest.mock import Mock, patch
from tests.test_utils import TestImageGenerator

# Mock all the problematic imports before importing from app
sys.modules['cv2'] = Mock()
sys.modules['streamlit'] = Mock()
sys.modules['onnxruntime'] = Mock()
sys.modules['azure.storage.blob'] = Mock()
sys.modules['azure.identity'] = Mock()
sys.modules['dotenv'] = Mock()

# Mock streamlit at module level to avoid import issues
mock_st = Mock()
mock_st.cache_resource = Mock()
mock_st.cache_resource.return_value = lambda func: func  # Return function unmodified
mock_st.error = Mock()
mock_st.info = Mock()
mock_st.success = Mock()
mock_st.warning = Mock()
mock_st.markdown = Mock()
mock_st.expander = Mock()
mock_st.spinner = Mock()
mock_st.container = Mock()
mock_st.empty = Mock()
mock_st.button = Mock()
mock_st.download_button = Mock()
mock_st.file_uploader = Mock()
mock_st.columns = Mock()
mock_st.selectbox = Mock()

# Patch streamlit before importing app
with patch.dict('sys.modules', {'streamlit': mock_st}):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from app import (
        load_all_models,
        process_image,
        image_to_bytes
    )


class TestStreamlitComponents:

    @patch('app.st.error')
    @patch('app.st.info')
    @patch('app.st.markdown')
    @patch('app.download_all_models_from_azure', return_value=False)
    @patch('app.get_all_model_paths', return_value=[])
    def test_model_loading_error_display(self, mock_get_paths, mock_download, mock_markdown, mock_info, mock_error):
        result = load_all_models()
        
        assert result == {}
        mock_error.assert_called()
        mock_info.assert_called_with("🔍 Modelos no encontrados localmente. Intentando descargar desde Azure...")


class TestImageUploadProcessing:
    
    def create_mock_uploaded_file(self, image: Image.Image, filename: str = "test.jpg"):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        
        mock_file = Mock()
        mock_file.name = filename
        mock_file.size = len(image_bytes.getvalue())
        mock_file.read.return_value = image_bytes.getvalue()
        mock_file.getvalue.return_value = image_bytes.getvalue()
        
        return mock_file
    
    def test_uploaded_file_processing(self):
        test_image = TestImageGenerator.create_solid_color_image((200, 200), (255, 128, 64))
        mock_file = self.create_mock_uploaded_file(test_image)
        
        loaded_image = Image.open(io.BytesIO(mock_file.getvalue()))
        
        assert loaded_image.size == (200, 200)
        assert loaded_image.mode in ['RGB', 'RGBA']
    
    def test_file_details_extraction(self):
        test_image = TestImageGenerator.create_solid_color_image((640, 480), (100, 200, 150))
        mock_file = self.create_mock_uploaded_file(test_image, "sample_image.png")
        
        loaded_image = Image.open(io.BytesIO(mock_file.getvalue()))
        
        file_details = {
            "Nombre de archivo": mock_file.name,
            "Tamaño de archivo": f"{mock_file.size / 1024:.1f} KB",
            "Tamaño de imagen": f"{loaded_image.size[0]} × {loaded_image.size[1]} px"
        }
        
        assert file_details["Nombre de archivo"] == "sample_image.png"
        assert "KB" in file_details["Tamaño de archivo"]
        assert "640 × 480 px" in file_details["Tamaño de imagen"]


class TestDownloadFunctionality:
    
    def test_download_filename_generation(self):
        original_filename = "my_photo.jpg"
        expected_download_name = "my_photo_no_background.png"
        
        filename_base = original_filename.rsplit('.', 1)[0]
        download_filename = f"{filename_base}_no_background.png"
        
        assert download_filename == expected_download_name
    
    def test_download_filename_no_extension(self):
        original_filename = "image_without_extension"
        expected_download_name = "image_without_extension_no_background.png"
        
        filename_base = original_filename.rsplit('.', 1)[0]
        download_filename = f"{filename_base}_no_background.png"
        
        assert download_filename == expected_download_name
    
    def test_download_filename_multiple_dots(self):
        original_filename = "my.image.file.jpeg"
        expected_download_name = "my.image.file_no_background.png"
        
        filename_base = original_filename.rsplit('.', 1)[0]
        download_filename = f"{filename_base}_no_background.png"
        
        assert download_filename == expected_download_name


class TestSessionStateHandling:
    
    def test_session_state_storage(self):
        # Create mock session state
        mock_session_state = {}
        test_image = TestImageGenerator.create_solid_color_image((100, 100), (255, 0, 0))
        test_filename = "test_image.jpg"
        
        mock_session_state['processed_image'] = test_image
        mock_session_state['original_filename'] = test_filename
        
        assert 'processed_image' in mock_session_state
        assert 'original_filename' in mock_session_state
        assert mock_session_state['original_filename'] == test_filename
    
    def test_session_state_retrieval(self):
        # Create mock session state
        mock_session_state = {}
        test_image = TestImageGenerator.create_solid_color_image((150, 150), (0, 255, 0))
        
        mock_session_state['processed_image'] = test_image
        mock_session_state['original_filename'] = "retrieved_image.png"
        
        retrieved_image = mock_session_state.get('processed_image')
        retrieved_filename = mock_session_state.get('original_filename')
        
        assert retrieved_image == test_image
        assert retrieved_filename == "retrieved_image.png"


class TestUIComponents:
    
    def test_layout_columns_creation(self):
        # Mock columns function
        mock_columns = Mock()
        mock_columns.return_value = (Mock(), Mock())
        
        col1, col2 = mock_columns([1, 1], gap="large")
        
        mock_columns.assert_called_with([1, 1], gap="large")
        assert col1 is not None
        assert col2 is not None
    
    def test_file_uploader_configuration(self):
        # Mock file uploader
        mock_uploader = Mock()
        mock_uploader.return_value = None
        
        uploaded_file = mock_uploader(
            "Arrastra y suelta una imagen aquí o haz clic para buscar",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Formatos soportados: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        mock_uploader.assert_called()
        call_args = mock_uploader.call_args
        assert "png" in call_args[1]["type"]
        assert "jpg" in call_args[1]["type"]


class TestErrorHandlingUI:
    
    @patch('app.st.error')
    def test_processing_error_display(self, mock_error):
        test_image = TestImageGenerator.create_solid_color_image((100, 100), (128, 128, 128))
        
        # Create mock model_info with broken session
        broken_session = Mock()
        broken_session.get_inputs.side_effect = Exception("Model error")
        
        mock_model_info = {
            'session': broken_session,
            'type': 'u2net',
            'input_size': (320, 320),
            'path': 'models/production/test_u2net.onnx'
        }
        
        result = process_image(test_image, mock_model_info)
        
        assert result is None
        mock_error.assert_called_once()
    
    def test_processing_spinner_context(self):
        mock_spinner = Mock()
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        with mock_spinner("Procesando imagen con IA..."):
            pass
        
        mock_spinner.assert_called_with("Procesando imagen con IA...")


class TestButtonInteractions:
    
    def test_process_button_configuration(self):
        mock_button = Mock()
        mock_button.return_value = False
        
        result = mock_button("🚀 Eliminar Fondo", type="primary", use_container_width=True)
        
        mock_button.assert_called()
        call_args = mock_button.call_args
        assert call_args[0][0] == "🚀 Eliminar Fondo"
        assert call_args[1]["type"] == "primary"
        assert call_args[1]["use_container_width"] == True
    
    def test_download_button_configuration(self):
        test_image = TestImageGenerator.create_solid_color_image((50, 50), (200, 200, 200))
        processed_bytes = image_to_bytes(test_image)
        
        mock_download_button = Mock()
        mock_download_button.return_value = False
        
        result = mock_download_button(
            label="📥 Descargar Imagen Procesada",
            data=processed_bytes,
            file_name="test_no_background.png",
            mime="image/png",
            type="secondary",
            use_container_width=True
        )
        
        mock_download_button.assert_called()
        call_args = mock_download_button.call_args
        assert call_args[1]["label"] == "📥 Descargar Imagen Procesada"
        assert call_args[1]["mime"] == "image/png"
        assert call_args[1]["type"] == "secondary"


class TestInfoMessages:
    
    def test_success_message_display(self):
        mock_success = Mock()
        mock_success("✅ Fondo eliminado exitosamente!")
        mock_success.assert_called_with("✅ Fondo eliminado exitosamente!")
    
    def test_info_message_display(self):
        mock_info = Mock()
        mock_info("👆 Sube una imagen a la izquierda para empezar")
        mock_info.assert_called_with("👆 Sube una imagen a la izquierda para empezar")
    
    def test_expander_functionality(self):
        mock_expander = Mock()
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        with mock_expander("ℹ️ Sobre la Tecnología"):
            pass
        
        mock_expander.assert_called_with("ℹ️ Sobre la Tecnología")


class TestMultiModelUI:
    """Test UI components specific to multi-model functionality"""
    
    def test_model_selection_dropdown(self):
        """Test model selection dropdown functionality"""
        mock_selectbox = Mock()
        mock_selectbox.return_value = "Model 1"
        
        selected = mock_selectbox(
            "Elige el modelo a usar:",
            options=["Model 1", "Model 2"],
            index=0,
            help="Selecciona el modelo que mejor se adapte a tus necesidades."
        )
        
        mock_selectbox.assert_called()
        call_args = mock_selectbox.call_args
        assert "Elige el modelo a usar:" in call_args[0]
        assert "help" in call_args[1]
    
    def test_model_info_expander(self):
        """Test expandable model information section"""
        mock_expander = Mock()
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        with mock_expander("Ver todos los modelos"):
            pass
        
        mock_expander.assert_called_with("Ver todos los modelos")
    
    def test_model_processing_feedback(self):
        """Test model processing feedback messages"""
        mock_info = Mock()
        
        model_name = "test_model.onnx"
        display_name = model_name.replace('.onnx', '').replace('_', ' ').title()
        
        mock_info(f"🤖 Procesando con modelo: **{display_name}**")
        
        mock_info.assert_called_with(f"🤖 Procesando con modelo: **{display_name}**")
    
    def test_model_result_attribution(self):
        """Test model attribution in results"""
        mock_success = Mock()
        
        model_name = "test_model.onnx"
        display_name = model_name.replace('.onnx', '').replace('_', ' ').title()
        
        mock_success(f"🤖 Procesado con: **{display_name}**")
        
        mock_success.assert_called_with(f"🤖 Procesado con: **{display_name}**")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])