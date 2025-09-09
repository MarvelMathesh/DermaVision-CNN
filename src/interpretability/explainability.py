"""
Interpretability and Explainability Module
==========================================

Implements state-of-the-art explainable AI techniques for skin disease detection,
including Grad-CAM, SHAP, LIME, and custom clinical feature visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime import lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available. Install with: pip install lime")

try:
    from captum.attr import IntegratedGradients, GradientShap, Occlusion, NoiseTunnel
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Captum not available. Install with: pip install captum")


class GradCAMAnalyzer:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for CNN interpretability.
    """
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Initialize Grad-CAM analyzer.
        
        Args:
            model: The trained model
            target_layer: Name of the target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for gradient capture."""
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        # Find target layer
        if self.target_layer:
            for name, module in self.model.named_modules():
                if name == self.target_layer:
                    self.hooks.append(module.register_forward_hook(forward_hook))
                    self.hooks.append(module.register_backward_hook(backward_hook))
                    break
        else:
            # Use the last convolutional layer by default
            conv_layers = [module for module in self.model.modules() 
                          if isinstance(module, nn.Conv2d)]
            if conv_layers:
                target_module = conv_layers[-1]
                self.hooks.append(target_module.register_forward_hook(forward_hook))
                self.hooks.append(target_module.register_backward_hook(backward_hook))
    
    def generate_cam(self, 
                     input_tensor: torch.Tensor,
                     target_class: Optional[int] = None,
                     normalize: bool = True) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)
            normalize: Whether to normalize the heatmap
        
        Returns:
            Grad-CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward(retain_graph=True)
        
        # Generate CAM
        if self.gradients is not None and self.activations is not None:
            # Pool gradients over spatial dimensions
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            
            # Weight activations by gradients
            for i in range(self.activations.size(1)):
                self.activations[:, i, :, :] *= pooled_gradients[i]
            
            # Create heatmap
            heatmap = torch.mean(self.activations, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            
            # Convert to numpy
            heatmap = heatmap.detach().cpu().numpy()
            
            if normalize:
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            return heatmap
        
        return None
    
    def visualize_cam(self,
                      input_image: np.ndarray,
                      heatmap: np.ndarray,
                      alpha: float = 0.4) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on input image.
        
        Args:
            input_image: Original image as numpy array (H, W, C)
            heatmap: Grad-CAM heatmap
            alpha: Blending factor
        
        Returns:
            Visualization as numpy array
        """
        # Resize heatmap to match input image
        heatmap_resized = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        if input_image.max() <= 1.0:
            input_image = (input_image * 255).astype(np.uint8)
        
        blended = cv2.addWeighted(input_image, 1-alpha, heatmap_colored, alpha, 0)
        
        return blended
    
    def cleanup(self):
        """Remove hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analyzer for model interpretability.
    """
    
    def __init__(self, model: nn.Module, background_data: torch.Tensor):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: The trained model
            background_data: Background dataset for SHAP baseline
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not available. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        
        # Create SHAP explainer
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def explain_prediction(self, 
                          input_tensor: torch.Tensor,
                          max_evals: int = 100) -> np.ndarray:
        """
        Generate SHAP values for input prediction.
        
        Args:
            input_tensor: Input image tensor
            max_evals: Maximum number of evaluations
        
        Returns:
            SHAP values as numpy array
        """
        with torch.no_grad():
            shap_values = self.explainer.shap_values(input_tensor, max_evals=max_evals)
        
        return shap_values
    
    def visualize_shap(self, 
                       input_image: np.ndarray,
                       shap_values: np.ndarray,
                       class_names: List[str]) -> plt.Figure:
        """
        Create SHAP visualization.
        
        Args:
            input_image: Original image
            shap_values: SHAP values for each class
            class_names: List of class names
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, len(class_names)//2 + 1, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, class_name in enumerate(class_names):
            if i < len(axes):
                shap.image_plot(
                    shap_values[i:i+1], 
                    input_image[np.newaxis], 
                    show=False
                )
                axes[i].set_title(f'{class_name}')
        
        # Hide unused subplots
        for i in range(len(class_names), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


class LIMEAnalyzer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) analyzer.
    """
    
    def __init__(self, model: nn.Module, class_names: List[str]):
        """
        Initialize LIME analyzer.
        
        Args:
            model: The trained model
            class_names: List of class names
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not available. Install with: pip install lime")
        
        self.model = model
        self.class_names = class_names
        self.explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(self, images: np.ndarray) -> np.ndarray:
        """
        Prediction function for LIME.
        
        Args:
            images: Batch of images as numpy array
        
        Returns:
            Prediction probabilities
        """
        self.model.eval()
        
        # Convert to tensor
        if images.ndim == 3:
            images = images[np.newaxis]
        
        # Normalize and convert to tensor
        images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2)
        images_tensor = (images_tensor / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
        
        with torch.no_grad():
            outputs = self.model(images_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def explain_prediction(self,
                          image: np.ndarray,
                          top_labels: int = 5,
                          num_samples: int = 1000) -> object:
        """
        Generate LIME explanation for image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            top_labels: Number of top labels to explain
            num_samples: Number of samples for explanation
        
        Returns:
            LIME explanation object
        """
        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples
        )
        
        return explanation
    
    def visualize_explanation(self, 
                            explanation: object,
                            label: int,
                            positive_only: bool = True) -> np.ndarray:
        """
        Visualize LIME explanation.
        
        Args:
            explanation: LIME explanation object
            label: Target label to visualize
            positive_only: Show only positive evidence
        
        Returns:
            Explanation image as numpy array
        """
        temp, mask = explanation.get_image_and_mask(
            label, 
            positive_only=positive_only,
            num_features=5,
            hide_rest=False
        )
        
        return temp


class ClinicalFeatureAnalyzer:
    """
    Analyzer for extracting and visualizing clinical features (ABCD rule).
    """
    
    def __init__(self):
        """Initialize clinical feature analyzer."""
        self.features = {
            'asymmetry': None,
            'border': None,
            'color': None,
            'diameter': None
        }
    
    def extract_asymmetry_score(self, image: np.ndarray, mask: np.ndarray) -> float:
        """
        Calculate asymmetry score of lesion.
        
        Args:
            image: Input image
            mask: Lesion segmentation mask
        
        Returns:
            Asymmetry score (0-1, higher = more asymmetric)
        """
        # Find lesion center
        moments = cv2.moments(mask)
        if moments['m00'] == 0:
            return 0.0
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        
        # Calculate asymmetry by comparing halves
        h, w = mask.shape
        
        # Vertical asymmetry
        left_half = mask[:, :w//2]
        right_half = cv2.flip(mask[:, w//2:], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        
        vertical_diff = np.sum(np.abs(
            left_half[:, :min_width] - right_half[:, :min_width]
        ))
        
        # Horizontal asymmetry
        top_half = mask[:h//2, :]
        bottom_half = cv2.flip(mask[h//2:, :], 0)
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        
        horizontal_diff = np.sum(np.abs(
            top_half[:min_height, :] - bottom_half[:min_height, :]
        ))
        
        # Normalize by lesion area
        lesion_area = np.sum(mask > 0)
        asymmetry_score = (vertical_diff + horizontal_diff) / (2 * lesion_area + 1e-8)
        
        return min(asymmetry_score, 1.0)
    
    def extract_border_irregularity(self, mask: np.ndarray) -> float:
        """
        Calculate border irregularity score.
        
        Args:
            mask: Lesion segmentation mask
        
        Returns:
            Border irregularity score (0-1, higher = more irregular)
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Calculate perimeter and area
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if area == 0:
            return 0.0
        
        # Circularity measure (1 = perfect circle, lower = more irregular)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
        
        # Convert to irregularity score
        irregularity = 1.0 - circularity
        
        return min(max(irregularity, 0.0), 1.0)
    
    def extract_color_variation(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """
        Extract color variation features.
        
        Args:
            image: Input image (RGB)
            mask: Lesion segmentation mask
        
        Returns:
            Dictionary of color features
        """
        # Extract lesion pixels
        lesion_pixels = image[mask > 0]
        
        if len(lesion_pixels) == 0:
            return {'color_std': 0.0, 'color_range': 0.0, 'dominant_colors': 0}
        
        # Color standard deviation
        color_std = np.mean(np.std(lesion_pixels, axis=0))
        
        # Color range
        color_range = np.mean(np.max(lesion_pixels, axis=0) - np.min(lesion_pixels, axis=0))
        
        # Number of dominant colors (simplified)
        from sklearn.cluster import KMeans
        
        try:
            # Reduce to main colors
            kmeans = KMeans(n_clusters=min(5, len(lesion_pixels)), random_state=42)
            kmeans.fit(lesion_pixels)
            dominant_colors = len(np.unique(kmeans.labels_))
        except:
            dominant_colors = 1
        
        return {
            'color_std': float(color_std / 255.0),
            'color_range': float(color_range / 255.0),
            'dominant_colors': dominant_colors
        }
    
    def estimate_diameter(self, mask: np.ndarray, pixels_per_mm: float = 10.0) -> float:
        """
        Estimate lesion diameter.
        
        Args:
            mask: Lesion segmentation mask
            pixels_per_mm: Pixels per millimeter calibration
        
        Returns:
            Estimated diameter in mm
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Use largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Find minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Convert to mm
        diameter_mm = (2 * radius) / pixels_per_mm
        
        return diameter_mm
    
    def analyze_abcd_features(self, 
                             image: np.ndarray,
                             mask: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive ABCD feature analysis.
        
        Args:
            image: Input image (RGB)
            mask: Lesion segmentation mask
        
        Returns:
            Dictionary of ABCD features
        """
        features = {}
        
        # A - Asymmetry
        features['asymmetry'] = self.extract_asymmetry_score(image, mask)
        
        # B - Border irregularity
        features['border_irregularity'] = self.extract_border_irregularity(mask)
        
        # C - Color variation
        color_features = self.extract_color_variation(image, mask)
        features.update(color_features)
        
        # D - Diameter
        features['diameter_mm'] = self.estimate_diameter(mask)
        
        # Calculate ABCD score (simplified)
        abcd_score = (
            features['asymmetry'] * 1.3 +
            features['border_irregularity'] * 0.1 +
            features['color_std'] * 0.5 +
            min(features['diameter_mm'] / 6.0, 1.0) * 0.5
        )
        
        features['abcd_score'] = min(abcd_score, 4.75)  # Max ABCD score
        
        return features


class InterpretabilityReport:
    """
    Comprehensive interpretability report generator.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 class_names: List[str],
                 device: torch.device):
        """
        Initialize report generator.
        
        Args:
            model: Trained model
            class_names: List of class names
            device: Computing device
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        
        # Initialize analyzers
        self.gradcam = GradCAMAnalyzer(model)
        self.lime = LIMEAnalyzer(model, class_names) if LIME_AVAILABLE else None
        self.clinical = ClinicalFeatureAnalyzer()
    
    def generate_comprehensive_report(self,
                                    image: np.ndarray,
                                    prediction: torch.Tensor,
                                    confidence: float,
                                    metadata: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive interpretability report.
        
        Args:
            image: Input image (RGB)
            prediction: Model prediction
            confidence: Prediction confidence
            metadata: Optional metadata
        
        Returns:
            Comprehensive report dictionary
        """
        report = {
            'prediction': {
                'class': self.class_names[prediction.item()],
                'confidence': confidence,
                'class_probabilities': {}
            },
            'visual_explanations': {},
            'clinical_features': {},
            'metadata': metadata or {}
        }
        
        # Convert image to tensor
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor / 255.0).to(self.device)
        
        # Grad-CAM analysis
        try:
            heatmap = self.gradcam.generate_cam(image_tensor, prediction.item())
            if heatmap is not None:
                visualization = self.gradcam.visualize_cam(image, heatmap)
                report['visual_explanations']['gradcam'] = {
                    'heatmap': heatmap,
                    'visualization': visualization
                }
        except Exception as e:
            print(f"Grad-CAM analysis failed: {e}")
        
        # LIME analysis (if available)
        if self.lime:
            try:
                lime_explanation = self.lime.explain_prediction(image)
                lime_vis = self.lime.visualize_explanation(lime_explanation, prediction.item())
                report['visual_explanations']['lime'] = {
                    'explanation': lime_explanation,
                    'visualization': lime_vis
                }
            except Exception as e:
                print(f"LIME analysis failed: {e}")
        
        # Clinical feature analysis (requires segmentation mask)
        # For demo purposes, create a simple mock mask
        mock_mask = self._create_mock_lesion_mask(image)
        try:
            clinical_features = self.clinical.analyze_abcd_features(image, mock_mask)
            report['clinical_features'] = clinical_features
        except Exception as e:
            print(f"Clinical feature analysis failed: {e}")
        
        return report
    
    def _create_mock_lesion_mask(self, image: np.ndarray) -> np.ndarray:
        """Create a mock lesion mask for demonstration."""
        # This is a simplified mock - in practice, you'd use a segmentation model
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple thresholding to find dark regions
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert to get lesion regions
        mask = 255 - mask
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def save_report(self, report: Dict, output_path: str):
        """Save interpretability report."""
        import json
        import pickle
        
        # Save JSON-serializable parts
        json_report = {
            'prediction': report['prediction'],
            'clinical_features': report['clinical_features'],
            'metadata': report['metadata']
        }
        
        with open(f"{output_path}_report.json", 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Save full report with visualizations
        with open(f"{output_path}_full_report.pkl", 'wb') as f:
            pickle.dump(report, f)
        
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    # Test interpretability components
    print("Testing Interpretability Components...")
    
    # Create mock model and data
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )
            self.classifier = nn.Linear(64, 8)
        
        def forward(self, x):
            features = self.features(x)
            return self.classifier(features.view(features.size(0), -1))
    
    # Test setup
    model = MockModel()
    class_names = ['melanoma', 'nevus', 'basal_cell_carcinoma', 'actinic_keratosis',
                   'benign_keratosis', 'dermatofibroma', 'vascular_lesion', 'squamous_cell_carcinoma']
    
    # Mock image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Test Grad-CAM
    gradcam = GradCAMAnalyzer(model)
    with torch.no_grad():
        prediction = model(image_tensor)
        predicted_class = prediction.argmax().item()
    
    heatmap = gradcam.generate_cam(image_tensor, predicted_class)
    if heatmap is not None:
        print(f"Grad-CAM heatmap shape: {heatmap.shape}")
    
    # Test Clinical Features
    clinical_analyzer = ClinicalFeatureAnalyzer()
    mock_mask = np.random.randint(0, 2, (224, 224)) * 255
    features = clinical_analyzer.analyze_abcd_features(image, mock_mask)
    print(f"Clinical features: {features}")
    
    # Cleanup
    gradcam.cleanup()
    
    print("Interpretability testing completed successfully!")