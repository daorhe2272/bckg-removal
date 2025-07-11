import os
import sys
import pytest
import numpy as np

from PIL import Image
from typing import Tuple
from app import load_all_models, preprocess_image, process_image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


class TestModelResponsiveness:
    
    @pytest.fixture
    def model_sessions(self):
        """Load all available models and return the dictionary"""
        sessions = load_all_models()
        if not sessions:
            pytest.skip("No hay modelos disponibles para pruebas")
        return sessions
    
    @pytest.fixture
    def single_model_session(self, model_sessions):
        """Get a single model session for tests that need just one model"""
        # Use the first available model
        model_name = list(model_sessions.keys())[0]
        return model_sessions[model_name]
    
    @pytest.fixture
    def test_image(self):
        return Image.new('RGB', (256, 256), color=(128, 128, 128))
    
    def test_models_load_successfully(self, model_sessions):
        """Test that at least one model loads successfully"""
        assert len(model_sessions) > 0
        for model_name, session in model_sessions.items():
            assert session is not None
            assert hasattr(session, 'get_inputs')
            assert hasattr(session, 'run')
    
    def test_all_models_have_correct_input_shape(self, model_sessions):
        """Test that all loaded models have the expected input shape"""
        for model_name, session in model_sessions.items():
            inputs = session.get_inputs()
            assert len(inputs) == 1, f"Model {model_name} should have exactly one input"
            
            input_shape = inputs[0].shape
            assert len(input_shape) == 4, f"Model {model_name} input should be 4D"
            assert input_shape[1] == 3, f"Model {model_name} should expect 3 channels"
            assert input_shape[2] == 320, f"Model {model_name} should expect height 320"
            assert input_shape[3] == 320, f"Model {model_name} should expect width 320"
    
    def test_single_model_output_shape(self, single_model_session, test_image):
        """Test output shape using a single model"""
        preprocessed = preprocess_image(test_image)
        inputs = {single_model_session.get_inputs()[0].name: preprocessed}
        
        outputs = single_model_session.run(None, inputs)
        
        assert len(outputs) >= 1, "El modelo debe devolver al menos un parámetro de salida"
        main_output = outputs[0]
        assert main_output.shape == (1, 1, 320, 320)
        assert main_output.dtype == np.float32
    
    def test_single_model_output_range(self, single_model_session, test_image):
        """Test output value range using a single model"""
        preprocessed = preprocess_image(test_image)
        inputs = {single_model_session.get_inputs()[0].name: preprocessed}
        
        outputs = single_model_session.run(None, inputs)
        output = outputs[0]
        
        assert 0 <= output.min() <= output.max() <= 1
    
    def test_single_model_inference_time(self, single_model_session, test_image):
        """Test inference time using a single model"""
        import time
        
        preprocessed = preprocess_image(test_image)
        inputs = {single_model_session.get_inputs()[0].name: preprocessed}
        
        start_time = time.time()
        outputs = single_model_session.run(None, inputs)
        end_time = time.time()
        
        inference_time = end_time - start_time
        assert inference_time < 10.0


class TestModelMetricStability:
    
    @pytest.fixture
    def model_sessions(self):
        """Load all available models"""
        sessions = load_all_models()
        if not sessions:
            pytest.skip("No hay modelos disponibles para pruebas")
        return sessions
    
    @pytest.fixture
    def single_model_session(self, model_sessions):
        """Get a single model session for tests that need just one model"""
        model_name = list(model_sessions.keys())[0]
        return model_sessions[model_name]
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        gt_binary = (gt_mask > 0.5).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def create_synthetic_test_case(self) -> Tuple[Image.Image, np.ndarray]:
        image = Image.new('RGB', (320, 320), color=(128, 128, 128))
        
        image_array = np.array(image)
        center_x, center_y = 160, 160
        radius = 80
        
        y, x = np.ogrid[:320, :320]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        image_array[mask_circle] = [255, 255, 255]
        image_array[~mask_circle] = [0, 0, 0]
        
        synthetic_image = Image.fromarray(image_array)
        ground_truth = mask_circle.astype(np.float32)
        
        return synthetic_image, ground_truth
    
    def test_synthetic_high_contrast_iou(self, single_model_session):
        """Test IoU using a single model on synthetic data"""
        test_image, gt_mask = self.create_synthetic_test_case()
        
        preprocessed = preprocess_image(test_image)
        inputs = {single_model_session.get_inputs()[0].name: preprocessed}
        outputs = single_model_session.run(None, inputs)
        
        pred_mask = outputs[0].squeeze()
        
        iou = self.calculate_iou(pred_mask, gt_mask)
        
        assert iou > 0.3, f"IoU {iou} está por debajo del umbral mínimo de 0.3"
    
    def test_model_consistency_multiple_runs(self, single_model_session):
        """Test consistency across multiple runs with the same model"""
        test_image = Image.new('RGB', (256, 256), color=(200, 150, 100))
        
        results = []
        for _ in range(3):
            preprocessed = preprocess_image(test_image)
            inputs = {single_model_session.get_inputs()[0].name: preprocessed}
            outputs = single_model_session.run(None, inputs)
            results.append(outputs[0])
        
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0], results[i], decimal=6,
                err_msg="Las salidas del modelo no son consistentes entre ejecuciones"
            )
    
    def test_different_image_sizes_consistency(self, single_model_session):
        """Test consistency across different image sizes with a single model"""
        sizes = [(100, 100), (256, 256), (512, 384)]
        ious = []
        
        for size in sizes:
            test_image = Image.new('RGB', size, color=(255, 255, 255))
            
            result = process_image(test_image, single_model_session)
            if result:
                result_array = np.array(result)
                alpha_channel = result_array[:, :, 3] / 255.0
                
                ideal_mask = np.ones(alpha_channel.shape)
                iou = self.calculate_iou(alpha_channel, ideal_mask)
                ious.append(iou)
        
        if len(ious) > 1:
            iou_std = np.std(ious)
            assert iou_std < 0.3, f"La varianza del IoU {iou_std} es demasiado alta entre diferentes tamaños de imagen"

    def test_all_models_consistency_on_same_image(self, model_sessions):
        """Test that all models produce reasonable results on the same image"""
        if len(model_sessions) < 2:
            pytest.skip("Se necesitan al menos 2 modelos para esta prueba")
        
        test_image = Image.new('RGB', (256, 256), color=(200, 150, 100))
        results = {}
        
        for model_name, session in model_sessions.items():
            result = process_image(test_image, session)
            if result:
                results[model_name] = result
        
        # All models should produce valid results
        assert len(results) == len(model_sessions), "Todos los modelos deberían producir resultados válidos"
        
        # All results should have the same dimensions
        sizes = [result.size for result in results.values()]
        assert all(size == sizes[0] for size in sizes), "Todos los resultados deberían tener las mismas dimensiones"


class TestModelRobustness:
    
    @pytest.fixture
    def model_sessions(self):
        """Load all available models"""
        sessions = load_all_models()
        if not sessions:
            pytest.skip("No hay modelos disponibles para pruebas")
        return sessions
    
    @pytest.fixture
    def single_model_session(self, model_sessions):
        """Get a single model session for edge case tests"""
        model_name = list(model_sessions.keys())[0]
        return model_sessions[model_name]
    
    def test_edge_case_black_image(self, single_model_session):
        black_image = Image.new('RGB', (200, 200), color=(0, 0, 0))
        
        result = process_image(black_image, single_model_session)
        
        assert result is not None
        assert result.mode == 'RGBA'
        assert result.size == (200, 200)
    
    def test_edge_case_white_image(self, single_model_session):
        white_image = Image.new('RGB', (200, 200), color=(255, 255, 255))
        
        result = process_image(white_image, single_model_session)
        
        assert result is not None
        assert result.mode == 'RGBA'
        assert result.size == (200, 200)
    
    def test_edge_case_small_image(self, single_model_session):
        small_image = Image.new('RGB', (50, 50), color=(128, 128, 128))
        
        result = process_image(small_image, single_model_session)
        
        assert result is not None
        assert result.size == (50, 50)
    
    def test_edge_case_rectangular_image(self, single_model_session):
        rect_image = Image.new('RGB', (400, 200), color=(100, 150, 200))
        
        result = process_image(rect_image, single_model_session)
        
        assert result is not None
        assert result.size == (400, 200)
    
    def test_grayscale_converted_to_rgb(self, single_model_session):
        gray_image = Image.new('L', (150, 150), color=128)
        
        result = process_image(gray_image, single_model_session)
        
        assert result is not None
        assert result.mode == 'RGBA'

    def test_all_models_handle_edge_cases(self, model_sessions):
        """Test that all models can handle basic edge cases"""
        edge_case_images = [
            Image.new('RGB', (100, 100), color=(0, 0, 0)),     # Black
            Image.new('RGB', (100, 100), color=(255, 255, 255)), # White
            Image.new('L', (100, 100), color=128),              # Grayscale
        ]
        
        for model_name, session in model_sessions.items():
            for i, test_image in enumerate(edge_case_images):
                result = process_image(test_image, session)
                assert result is not None, f"Model {model_name} failed on edge case {i}"
                assert result.mode == 'RGBA'


class TestModelPerformance:
    
    @pytest.fixture
    def model_sessions(self):
        """Load all available models"""
        sessions = load_all_models()
        if not sessions:
            pytest.skip("No hay modelos disponibles para pruebas")
        return sessions
    
    @pytest.fixture
    def single_model_session(self, model_sessions):
        """Get a single model session for performance tests"""
        model_name = list(model_sessions.keys())[0]
        return model_sessions[model_name]
    
    def test_batch_processing_performance(self, single_model_session):
        """Test performance with batch processing using a single model"""
        import time
        
        images = [Image.new('RGB', (200, 200), color=(i*50, i*30, i*20)) for i in range(5)]
        
        start_time = time.time()
        results = []
        for image in images:
            result = process_image(image, single_model_session)
            results.append(result)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(images)
        
        assert all(r is not None for r in results)
        assert avg_time_per_image < 5.0
    
    def test_memory_usage_stability(self, single_model_session):
        """Test memory stability with repeated processing"""
        import gc
        
        for i in range(10):
            test_image = Image.new('RGB', (300, 300), color=(i*25, i*25, i*25))
            result = process_image(test_image, single_model_session)
            assert result is not None
            
            del result
            if i % 3 == 0:
                gc.collect()

    def test_comparative_performance_across_models(self, model_sessions):
        """Compare performance across all available models"""
        if len(model_sessions) < 2:
            pytest.skip("Se necesitan al menos 2 modelos para comparar rendimiento")
        
        import time
        test_image = Image.new('RGB', (300, 300), color=(128, 128, 128))
        performance_results = {}
        
        for model_name, session in model_sessions.items():
            start_time = time.time()
            result = process_image(test_image, session)
            end_time = time.time()
            
            processing_time = end_time - start_time
            performance_results[model_name] = {
                'time': processing_time,
                'success': result is not None
            }
        
        # All models should complete successfully
        for model_name, perf in performance_results.items():
            assert perf['success'], f"Model {model_name} failed to process image"
            assert perf['time'] < 10.0, f"Model {model_name} took too long: {perf['time']}s"
        
        # Print performance comparison for debugging
        print("\nModel Performance Comparison:")
        for model_name, perf in performance_results.items():
            print(f"  {model_name}: {perf['time']:.3f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 