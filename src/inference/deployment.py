"""
Deployment and Production Inference Module
==========================================

Optimized inference pipeline for production deployment with model serving,
batch processing, and performance monitoring.
"""

import torch
import torch.nn as nn
import torch.jit
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import logging
from pathlib import Path
import json
from dataclasses import dataclass
import concurrent.futures
from threading import Lock
import warnings
warnings.filterwarnings('ignore')


@dataclass
class InferenceResult:
    """Structure for inference results."""
    predictions: np.ndarray
    probabilities: np.ndarray
    inference_time: float
    preprocessing_time: float
    postprocessing_time: float
    model_version: str
    timestamp: float


class ModelOptimizer:
    """
    Model optimization for production deployment.
    """
    
    def __init__(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)):
        """
        Initialize model optimizer.
        
        Args:
            model: PyTorch model to optimize
            input_shape: Expected input shape
        """
        self.model = model
        self.input_shape = input_shape
        self.optimized_models = {}
    
    def quantize_model(self, 
                      quantization_type: str = 'dynamic',
                      save_path: Optional[str] = None) -> nn.Module:
        """
        Quantize model for reduced memory and faster inference.
        
        Args:
            quantization_type: Type of quantization ('dynamic', 'static')
            save_path: Path to save quantized model
        
        Returns:
            Quantized model
        """
        self.model.eval()
        
        if quantization_type == 'dynamic':
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
        else:
            # Static quantization (requires calibration data)
            # This is a simplified implementation
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            
            # Here you would run calibration data through the model
            # For demonstration, we'll skip this step
            
            quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        if save_path:
            torch.save(quantized_model.state_dict(), save_path)
            print(f"Quantized model saved to {save_path}")
        
        self.optimized_models['quantized'] = quantized_model
        return quantized_model
    
    def convert_to_torchscript(self, save_path: Optional[str] = None) -> torch.jit.ScriptModule:
        """
        Convert model to TorchScript for deployment.
        
        Args:
            save_path: Path to save TorchScript model
        
        Returns:
            TorchScript model
        """
        self.model.eval()
        
        # Create example input
        example_input = torch.randn(*self.input_shape)
        
        # Trace the model
        try:
            traced_model = torch.jit.trace(self.model, example_input)
        except Exception as e:
            print(f"Tracing failed: {e}, trying scripting...")
            traced_model = torch.jit.script(self.model)
        
        # Optimize for inference
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        if save_path:
            traced_model.save(save_path)
            print(f"TorchScript model saved to {save_path}")
        
        self.optimized_models['torchscript'] = traced_model
        return traced_model
    
    def convert_to_onnx(self, 
                       save_path: str,
                       opset_version: int = 11,
                       dynamic_axes: Optional[Dict] = None) -> str:
        """
        Convert model to ONNX format.
        
        Args:
            save_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes for variable input sizes
        
        Returns:
            Path to saved ONNX model
        """
        self.model.eval()
        
        # Create example input
        example_input = torch.randn(*self.input_shape)
        
        # Default dynamic axes
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            example_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
        print(f"ONNX model saved to {save_path}")
        return save_path
    
    def benchmark_models(self, 
                        num_iterations: int = 100,
                        warmup_iterations: int = 10) -> Dict[str, Dict]:
        """
        Benchmark different model formats.
        
        Args:
            num_iterations: Number of benchmark iterations
            warmup_iterations: Number of warmup iterations
        
        Returns:
            Benchmark results
        """
        results = {}
        example_input = torch.randn(*self.input_shape)
        
        # Benchmark original model
        results['original'] = self._benchmark_single_model(
            self.model, example_input, num_iterations, warmup_iterations
        )
        
        # Benchmark optimized models
        for name, model in self.optimized_models.items():
            if name == 'torchscript':
                results[name] = self._benchmark_single_model(
                    model, example_input, num_iterations, warmup_iterations
                )
        
        return results
    
    def _benchmark_single_model(self, 
                               model: nn.Module,
                               input_tensor: torch.Tensor,
                               num_iterations: int,
                               warmup_iterations: int) -> Dict[str, float]:
        """Benchmark a single model."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(input_tensor)
        
        # Actual benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'fps': fps,
            'total_time': end_time - start_time
        }


class ProductionInference:
    """
    Production-ready inference pipeline with optimization and monitoring.
    """
    
    def __init__(self, 
                 model_path: str,
                 class_names: List[str],
                 device: str = 'cpu',
                 model_format: str = 'pytorch',
                 batch_size: int = 1):
        """
        Initialize production inference pipeline.
        
        Args:
            model_path: Path to model file
            class_names: List of class names
            device: Device for inference ('cpu', 'cuda')
            model_format: Model format ('pytorch', 'torchscript', 'onnx')
            batch_size: Batch size for inference
        """
        self.model_path = model_path
        self.class_names = class_names
        self.device = device
        self.model_format = model_format
        self.batch_size = batch_size
        
        # Performance monitoring
        self.inference_stats = {
            'total_inferences': 0,
            'total_time': 0,
            'avg_inference_time': 0,
            'peak_memory': 0
        }
        
        # Thread safety
        self.lock = Lock()
        
        # Load model
        self.model = self._load_model()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        print(f"Production inference initialized with {model_format} model on {device}")
    
    def _load_model(self) -> Union[nn.Module, torch.jit.ScriptModule, ort.InferenceSession]:
        """Load model based on format."""
        if self.model_format == 'pytorch':
            model = torch.load(self.model_path, map_location=self.device)
            model.eval()
            return model
        
        elif self.model_format == 'torchscript':
            model = torch.jit.load(self.model_path, map_location=self.device)
            model.eval()
            return model
        
        elif self.model_format == 'onnx':
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and ort.get_device() == 'GPU':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            session = ort.InferenceSession(self.model_path, providers=providers)
            return session
        
        else:
            raise ValueError(f"Unsupported model format: {self.model_format}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for production monitoring."""
        logger = logging.getLogger('skin_disease_inference')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler('inference.log')
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def preprocess_image(self, image: np.ndarray, target_size: int = 224) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image as numpy array (H, W, C)
            target_size: Target image size
        
        Returns:
            Preprocessed image tensor
        """
        start_time = time.time()
        
        # Resize image
        image = cv2.resize(image, (target_size, target_size))
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        
        if self.device == 'cuda' and torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        preprocessing_time = time.time() - start_time
        return image_tensor, preprocessing_time
    
    def postprocess_predictions(self, 
                              outputs: Union[torch.Tensor, np.ndarray],
                              return_top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            return_top_k: Number of top predictions to return
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        start_time = time.time()
        
        # Convert to numpy if needed
        if torch.is_tensor(outputs):
            outputs = outputs.cpu().numpy()
        
        # Apply softmax
        probabilities = self._softmax(outputs)
        
        # Get predictions
        predictions = np.argmax(probabilities, axis=1)
        
        postprocessing_time = time.time() - start_time
        return predictions, probabilities, postprocessing_time
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def infer_single(self, image: np.ndarray) -> InferenceResult:
        """
        Perform inference on a single image.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            InferenceResult object
        """
        total_start_time = time.time()
        
        # Preprocess
        image_tensor, preprocess_time = self.preprocess_image(image)
        
        # Inference
        inference_start_time = time.time()
        
        if self.model_format == 'onnx':
            # ONNX inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: image_tensor.numpy()})
            outputs = outputs[0]
        else:
            # PyTorch inference
            with torch.no_grad():
                outputs = self.model(image_tensor)
                if torch.is_tensor(outputs):
                    outputs = outputs.cpu().numpy()
        
        inference_time = time.time() - inference_start_time
        
        # Postprocess
        predictions, probabilities, postprocess_time = self.postprocess_predictions(outputs)
        
        total_time = time.time() - total_start_time
        
        # Update statistics
        with self.lock:
            self.inference_stats['total_inferences'] += 1
            self.inference_stats['total_time'] += total_time
            self.inference_stats['avg_inference_time'] = (
                self.inference_stats['total_time'] / self.inference_stats['total_inferences']
            )
        
        # Log inference
        self.logger.info(f"Inference completed in {total_time:.3f}s - "
                        f"Predicted: {self.class_names[predictions[0]]} "
                        f"(confidence: {probabilities[0][predictions[0]]:.3f})")
        
        return InferenceResult(
            predictions=predictions,
            probabilities=probabilities,
            inference_time=inference_time,
            preprocessing_time=preprocess_time,
            postprocessing_time=postprocess_time,
            model_version="1.0",
            timestamp=time.time()
        )
    
    def infer_batch(self, images: List[np.ndarray]) -> List[InferenceResult]:
        """
        Perform batch inference on multiple images.
        
        Args:
            images: List of input images
        
        Returns:
            List of InferenceResult objects
        """
        # Process images in batches
        results = []
        
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            # Preprocess batch
            batch_tensors = []
            preprocess_times = []
            
            for image in batch_images:
                tensor, preprocess_time = self.preprocess_image(image)
                batch_tensors.append(tensor)
                preprocess_times.append(preprocess_time)
            
            # Combine into batch
            batch_tensor = torch.cat(batch_tensors, dim=0)
            
            # Batch inference
            inference_start_time = time.time()
            
            if self.model_format == 'onnx':
                input_name = self.model.get_inputs()[0].name
                outputs = self.model.run(None, {input_name: batch_tensor.numpy()})
                outputs = outputs[0]
            else:
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    if torch.is_tensor(outputs):
                        outputs = outputs.cpu().numpy()
            
            inference_time = time.time() - inference_start_time
            
            # Postprocess batch
            predictions, probabilities, postprocess_time = self.postprocess_predictions(outputs)
            
            # Create individual results
            for j in range(len(batch_images)):
                result = InferenceResult(
                    predictions=np.array([predictions[j]]),
                    probabilities=np.array([probabilities[j]]),
                    inference_time=inference_time / len(batch_images),
                    preprocessing_time=preprocess_times[j],
                    postprocessing_time=postprocess_time / len(batch_images),
                    model_version="1.0",
                    timestamp=time.time()
                )
                results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            stats = self.inference_stats.copy()
        
        # Add system info
        stats['device'] = self.device
        stats['model_format'] = self.model_format
        stats['batch_size'] = self.batch_size
        
        # Calculate throughput
        if stats['avg_inference_time'] > 0:
            stats['throughput_fps'] = 1.0 / stats['avg_inference_time']
        else:
            stats['throughput_fps'] = 0
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self.lock:
            self.inference_stats = {
                'total_inferences': 0,
                'total_time': 0,
                'avg_inference_time': 0,
                'peak_memory': 0
            }


class ModelServer:
    """
    Simple model server for REST API deployment.
    """
    
    def __init__(self, inference_pipeline: ProductionInference):
        """
        Initialize model server.
        
        Args:
            inference_pipeline: Production inference pipeline
        """
        self.inference_pipeline = inference_pipeline
        self.server_stats = {
            'requests_served': 0,
            'start_time': time.time(),
            'errors': 0
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'uptime': time.time() - self.server_stats['start_time'],
            'requests_served': self.server_stats['requests_served'],
            'errors': self.server_stats['errors'],
            'model_stats': self.inference_pipeline.get_performance_stats()
        }
    
    def predict(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Prediction endpoint.
        
        Args:
            image_data: Input image data
        
        Returns:
            Prediction response
        """
        try:
            result = self.inference_pipeline.infer_single(image_data)
            
            self.server_stats['requests_served'] += 1
            
            return {
                'success': True,
                'prediction': {
                    'class': self.inference_pipeline.class_names[result.predictions[0]],
                    'confidence': float(result.probabilities[0][result.predictions[0]]),
                    'all_probabilities': {
                        name: float(prob) for name, prob in 
                        zip(self.inference_pipeline.class_names, result.probabilities[0])
                    }
                },
                'timing': {
                    'total_time': result.inference_time + result.preprocessing_time + result.postprocessing_time,
                    'inference_time': result.inference_time,
                    'preprocessing_time': result.preprocessing_time,
                    'postprocessing_time': result.postprocessing_time
                },
                'timestamp': result.timestamp
            }
        
        except Exception as e:
            self.server_stats['errors'] += 1
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }


# Export classes
__all__ = [
    'ModelOptimizer',
    'ProductionInference', 
    'ModelServer',
    'InferenceResult'
]


if __name__ == "__main__":
    # Test deployment components
    print("Testing Deployment Components...")
    
    # Create a simple model for testing
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self, num_classes=8):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Linear(32, num_classes)
        
        def forward(self, x):
            features = self.features(x)
            return self.classifier(features.view(features.size(0), -1))
    
    # Test model optimization
    model = SimpleModel()
    optimizer = ModelOptimizer(model)
    
    # Test quantization
    quantized_model = optimizer.quantize_model()
    print("Model quantization completed")
    
    # Test TorchScript conversion
    torchscript_model = optimizer.convert_to_torchscript()
    print("TorchScript conversion completed")
    
    # Test benchmarking
    benchmark_results = optimizer.benchmark_models(num_iterations=10)
    print(f"Benchmark results: {benchmark_results}")
    
    print("Deployment testing completed successfully!")