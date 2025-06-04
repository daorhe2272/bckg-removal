import os
import sys
import pytest
import numpy as np

from PIL import Image
from typing import Tuple
from app import load_model, preprocess_image, process_image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


class TestModelResponsiveness:
    
    @pytest.fixture
    def model_session(self):
        session = load_model()
        if session is None:
            pytest.skip("Modelo no disponible para pruebas")
        return session
    
    @pytest.fixture
    def test_image(self):
        return Image.new('RGB', (256, 256), color=(128, 128, 128))
    
    def test_model_loads_successfully(self, model_session):
        assert model_session is not None
        assert hasattr(model_session, 'get_inputs')
        assert hasattr(model_session, 'run')
    
    def test_model_input_shape(self, model_session):
        inputs = model_session.get_inputs()
        assert len(inputs) == 1
        
        input_shape = inputs[0].shape
        assert len(input_shape) == 4
        assert input_shape[1] == 3
        assert input_shape[2] == 320
        assert input_shape[3] == 320
    
    def test_model_output_shape(self, model_session, test_image):
        preprocessed = preprocess_image(test_image)
        inputs = {model_session.get_inputs()[0].name: preprocessed}
        
        outputs = model_session.run(None, inputs)
        
        assert len(outputs) >= 1, "El modelo debe devolver al menos un parámetro de salida"
        main_output = outputs[0]
        assert main_output.shape == (1, 1, 320, 320)
        assert main_output.dtype == np.float32
    
    def test_model_output_range(self, model_session, test_image):
        preprocessed = preprocess_image(test_image)
        inputs = {model_session.get_inputs()[0].name: preprocessed}
        
        outputs = model_session.run(None, inputs)
        output = outputs[0]
        
        assert 0 <= output.min() <= output.max() <= 1
    
    def test_model_inference_time(self, model_session, test_image):
        import time
        
        preprocessed = preprocess_image(test_image)
        inputs = {model_session.get_inputs()[0].name: preprocessed}
        
        start_time = time.time()
        outputs = model_session.run(None, inputs)
        end_time = time.time()
        
        inference_time = end_time - start_time
        assert inference_time < 10.0


class TestModelMetricStability:
    
    @pytest.fixture
    def model_session(self):
        session = load_model()
        if session is None:
            pytest.skip("Modelo no disponible para pruebas")
        return session
    
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
    
    def test_synthetic_high_contrast_iou(self, model_session):
        test_image, gt_mask = self.create_synthetic_test_case()
        
        preprocessed = preprocess_image(test_image)
        inputs = {model_session.get_inputs()[0].name: preprocessed}
        outputs = model_session.run(None, inputs)
        
        pred_mask = outputs[0].squeeze()
        
        iou = self.calculate_iou(pred_mask, gt_mask)
        
        assert iou > 0.3, f"IoU {iou} está por debajo del umbral mínimo de 0.3"
    
    def test_model_consistency_multiple_runs(self, model_session):
        test_image = Image.new('RGB', (256, 256), color=(200, 150, 100))
        
        results = []
        for _ in range(3):
            preprocessed = preprocess_image(test_image)
            inputs = {model_session.get_inputs()[0].name: preprocessed}
            outputs = model_session.run(None, inputs)
            results.append(outputs[0])
        
        for i in range(1, len(results)):
            np.testing.assert_array_almost_equal(
                results[0], results[i], decimal=6,
                err_msg="Las salidas del modelo no son consistentes entre ejecuciones"
            )
    
    def test_different_image_sizes_consistency(self, model_session):
        sizes = [(100, 100), (256, 256), (512, 384)]
        ious = []
        
        for size in sizes:
            test_image = Image.new('RGB', size, color=(255, 255, 255))
            
            result = process_image(test_image, model_session)
            if result:
                result_array = np.array(result)
                alpha_channel = result_array[:, :, 3] / 255.0
                
                ideal_mask = np.ones(alpha_channel.shape)
                iou = self.calculate_iou(alpha_channel, ideal_mask)
                ious.append(iou)
        
        if len(ious) > 1:
            iou_std = np.std(ious)
            assert iou_std < 0.3, f"La varianza del IoU {iou_std} es demasiado alta entre diferentes tamaños de imagen"


class TestModelRobustness:
    
    @pytest.fixture
    def model_session(self):
        session = load_model()
        if session is None:
            pytest.skip("Modelo no disponible para pruebas")
        return session
    
    def test_edge_case_black_image(self, model_session):
        black_image = Image.new('RGB', (200, 200), color=(0, 0, 0))
        
        result = process_image(black_image, model_session)
        
        assert result is not None
        assert result.mode == 'RGBA'
        assert result.size == (200, 200)
    
    def test_edge_case_white_image(self, model_session):
        white_image = Image.new('RGB', (200, 200), color=(255, 255, 255))
        
        result = process_image(white_image, model_session)
        
        assert result is not None
        assert result.mode == 'RGBA'
        assert result.size == (200, 200)
    
    def test_edge_case_small_image(self, model_session):
        small_image = Image.new('RGB', (50, 50), color=(128, 128, 128))
        
        result = process_image(small_image, model_session)
        
        assert result is not None
        assert result.size == (50, 50)
    
    def test_edge_case_rectangular_image(self, model_session):
        rect_image = Image.new('RGB', (400, 200), color=(100, 150, 200))
        
        result = process_image(rect_image, model_session)
        
        assert result is not None
        assert result.size == (400, 200)
    
    def test_grayscale_converted_to_rgb(self, model_session):
        gray_image = Image.new('L', (150, 150), color=128)
        
        result = process_image(gray_image, model_session)
        
        assert result is not None
        assert result.mode == 'RGBA'


class TestModelPerformance:
    
    @pytest.fixture
    def model_session(self):
        session = load_model()
        if session is None:
            pytest.skip("Modelo no disponible para pruebas")
        return session
    
    def test_batch_processing_performance(self, model_session):
        import time
        
        images = [Image.new('RGB', (200, 200), color=(i*50, i*30, i*20)) for i in range(5)]
        
        start_time = time.time()
        results = []
        for image in images:
            result = process_image(image, model_session)
            results.append(result)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_image = total_time / len(images)
        
        assert all(r is not None for r in results)
        assert avg_time_per_image < 5.0
    
    def test_memory_usage_stability(self, model_session):
        import gc
        
        for i in range(10):
            test_image = Image.new('RGB', (300, 300), color=(i*25, i*25, i*25))
            result = process_image(test_image, model_session)
            assert result is not None
            
            del result
            if i % 3 == 0:
                gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 