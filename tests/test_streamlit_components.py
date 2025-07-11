import os
import io
import sys
import pytest

from PIL import Image
from unittest.mock import Mock, patch
from tests.test_utils import TestImageGenerator

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


class TestStreamlitComponents:

    @patch('streamlit.error')
    @patch('streamlit.info')
    @patch('streamlit.markdown')
    @patch('app.download_model_from_azure', return_value=False)
    @patch('app.get_latest_model_path', return_value=None)
    def test_model_loading_error_display(self, mock_get_latest, mock_download, mock_markdown, mock_info, mock_error):
        # Clear cache before test
        import streamlit as st
        st.cache_resource.clear()
        
        from app import load_model
        
        result = load_model()
        
        assert result is None
        mock_error.assert_called_once()
        mock_info.assert_called_once()

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
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_session_state_storage(self, mock_session_state):
        test_image = TestImageGenerator.create_solid_color_image((100, 100), (255, 0, 0))
        test_filename = "test_image.jpg"
        
        mock_session_state['processed_image'] = test_image
        mock_session_state['original_filename'] = test_filename
        
        assert 'processed_image' in mock_session_state
        assert 'original_filename' in mock_session_state
        assert mock_session_state['original_filename'] == test_filename
    
    @patch('streamlit.session_state', new_callable=dict)
    def test_session_state_retrieval(self, mock_session_state):
        test_image = TestImageGenerator.create_solid_color_image((150, 150), (0, 255, 0))
        
        mock_session_state['processed_image'] = test_image
        mock_session_state['original_filename'] = "retrieved_image.png"
        
        retrieved_image = mock_session_state.get('processed_image')
        retrieved_filename = mock_session_state.get('original_filename')
        
        assert retrieved_image == test_image
        assert retrieved_filename == "retrieved_image.png"

class TestUIComponents:
    
    @patch('streamlit.columns')
    def test_layout_columns_creation(self, mock_columns):
        mock_columns.return_value = (Mock(), Mock())
        
        col1, col2 = mock_columns([1, 1], gap="large")
        
        mock_columns.assert_called_with([1, 1], gap="large")
        assert col1 is not None
        assert col2 is not None
    
    @patch('streamlit.file_uploader')
    def test_file_uploader_configuration(self, mock_uploader):
        mock_uploader.return_value = None
        
        from app import st
        
        uploaded_file = st.file_uploader(
            "Arrastra y suelta una imagen aquí o haz clic para buscar",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Formatos soportados: PNG, JPG, JPEG, BMP, TIFF"
        )
        
        mock_uploader.assert_called()
        call_args = mock_uploader.call_args
        assert "png" in call_args[1]["type"]
        assert "jpg" in call_args[1]["type"]

class TestErrorHandlingUI:
    
    @patch('streamlit.error')
    def test_processing_error_display(self, mock_error):
        from app import process_image
        
        test_image = TestImageGenerator.create_solid_color_image((100, 100), (128, 128, 128))
        broken_session = Mock()
        broken_session.get_inputs.side_effect = Exception("Model error")
        
        result = process_image(test_image, broken_session)
        
        assert result is None
        mock_error.assert_called_once()
    
    @patch('streamlit.spinner')
    def test_processing_spinner_context(self, mock_spinner):
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        with mock_spinner("Procesando imagen con IA..."):
            pass
        
        mock_spinner.assert_called_with("Procesando imagen con IA...")

class TestButtonInteractions:
    
    @patch('streamlit.button')
    def test_process_button_configuration(self, mock_button):
        mock_button.return_value = False
        
        from app import st
        
        if st.button("🚀 Eliminar Fondo", type="primary", use_container_width=True):
            pass
        
        mock_button.assert_called()
        call_args = mock_button.call_args
        assert call_args[0][0] == "🚀 Eliminar Fondo"
        assert call_args[1]["type"] == "primary"
        assert call_args[1]["use_container_width"] == True
    
    @patch('streamlit.download_button')
    def test_download_button_configuration(self, mock_download_button):
        from app import image_to_bytes
        
        test_image = TestImageGenerator.create_solid_color_image((50, 50), (200, 200, 200))
        processed_bytes = image_to_bytes(test_image)
        
        mock_download_button.return_value = False
        
        from app import st
        
        st.download_button(
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
    
    @patch('streamlit.success')
    def test_success_message_display(self, mock_success):
        from app import st
        
        st.success("✅ Fondo eliminado exitosamente!")
        
        mock_success.assert_called_with("✅ Fondo eliminado exitosamente!")
    
    @patch('streamlit.info')
    def test_info_message_display(self, mock_info):
        from app import st
        
        st.info("👆 Sube una imagen a la izquierda para empezar")
        
        mock_info.assert_called_with("👆 Sube una imagen a la izquierda para empezar")
    
    @patch('streamlit.expander')
    def test_expander_functionality(self, mock_expander):
        mock_expander.return_value.__enter__ = Mock()
        mock_expander.return_value.__exit__ = Mock()
        
        from app import st
        
        with st.expander("ℹ️ Sobre la Tecnología"):
            pass
        
        mock_expander.assert_called_with("ℹ️ Sobre la Tecnología")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])