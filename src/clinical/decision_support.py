"""
Clinical Decision Support System
===============================

Implements clinical integration features for real-world deployment
of skin disease detection models.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClinicalPrediction:
    """Structure for clinical prediction results."""
    
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    risk_level: str
    recommended_action: str
    differential_diagnosis: List[Tuple[str, float]]
    clinical_features: Dict[str, float]
    uncertainty: Optional[float] = None
    explanation: Optional[str] = None
    timestamp: Optional[datetime] = None


class RiskStratificationSystem:
    """
    Risk stratification system for clinical decision support.
    """
    
    def __init__(self, 
                 class_names: List[str],
                 malignant_classes: List[str] = None,
                 risk_thresholds: Dict[str, float] = None):
        """
        Initialize risk stratification system.
        
        Args:
            class_names: List of all class names
            malignant_classes: List of malignant class names
            risk_thresholds: Thresholds for risk stratification
        """
        self.class_names = class_names
        self.malignant_classes = malignant_classes or ['melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma']
        
        # Default risk thresholds
        self.risk_thresholds = risk_thresholds or {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.7,
            'critical': 0.9
        }
        
        # Clinical guidelines
        self.clinical_guidelines = self._load_clinical_guidelines()
    
    def _load_clinical_guidelines(self) -> Dict:
        """Load clinical guidelines for different conditions."""
        return {
            'melanoma': {
                'description': 'Malignant melanoma - requires immediate specialist referral',
                'urgency': 'urgent',
                'follow_up': '2-week pathway',
                'additional_tests': ['dermoscopy', 'biopsy'],
                'patient_advice': 'Monitor for changes in size, shape, or color'
            },
            'basal_cell_carcinoma': {
                'description': 'Basal cell carcinoma - most common skin cancer',
                'urgency': 'routine',
                'follow_up': '4-6 weeks',
                'additional_tests': ['dermoscopy', 'possible biopsy'],
                'patient_advice': 'Protect from sun exposure'
            },
            'squamous_cell_carcinoma': {
                'description': 'Squamous cell carcinoma - requires specialist evaluation',
                'urgency': 'urgent',
                'follow_up': '2-week pathway',
                'additional_tests': ['dermoscopy', 'biopsy'],
                'patient_advice': 'Avoid sun exposure and monitor for growth'
            },
            'nevus': {
                'description': 'Benign nevus - routine monitoring recommended',
                'urgency': 'routine',
                'follow_up': 'annual check',
                'additional_tests': ['photography for monitoring'],
                'patient_advice': 'Self-examination monthly'
            },
            'actinic_keratosis': {
                'description': 'Actinic keratosis - precancerous lesion',
                'urgency': 'routine',
                'follow_up': '6-12 weeks',
                'additional_tests': ['dermoscopy'],
                'patient_advice': 'Sun protection essential'
            },
            'benign_keratosis': {
                'description': 'Benign seborrheic keratosis - no treatment required',
                'urgency': 'routine',
                'follow_up': 'as needed',
                'additional_tests': [],
                'patient_advice': 'Monitor for changes'
            },
            'dermatofibroma': {
                'description': 'Dermatofibroma - benign fibrous lesion',
                'urgency': 'routine',
                'follow_up': 'as needed',
                'additional_tests': [],
                'patient_advice': 'No specific monitoring required'
            },
            'vascular_lesion': {
                'description': 'Vascular lesion - typically benign',
                'urgency': 'routine',
                'follow_up': 'as needed',
                'additional_tests': ['ultrasound if large'],
                'patient_advice': 'Monitor for size changes'
            }
        }
    
    def stratify_risk(self, 
                     probabilities: np.ndarray,
                     uncertainty: Optional[float] = None,
                     clinical_features: Optional[Dict] = None) -> Tuple[str, str]:
        """
        Stratify risk level and recommend action.
        
        Args:
            probabilities: Class probabilities
            uncertainty: Prediction uncertainty
            clinical_features: Clinical feature scores
        
        Returns:
            Tuple of (risk_level, recommended_action)
        """
        # Calculate malignant probability
        malignant_prob = 0.0
        for i, class_name in enumerate(self.class_names):
            if class_name in self.malignant_classes and i < len(probabilities):
                malignant_prob += probabilities[i]
        
        # Adjust for uncertainty
        if uncertainty is not None and uncertainty > 0.5:
            # High uncertainty increases perceived risk
            malignant_prob = min(1.0, malignant_prob * (1 + uncertainty))
        
        # Adjust for clinical features (ABCD score)
        if clinical_features and 'abcd_score' in clinical_features:
            abcd_score = clinical_features['abcd_score']
            if abcd_score > 4.0:  # High ABCD score
                malignant_prob = min(1.0, malignant_prob * 1.2)
        
        # Determine risk level
        if malignant_prob >= self.risk_thresholds['critical']:
            risk_level = 'critical'
            action = 'immediate_referral'
        elif malignant_prob >= self.risk_thresholds['high']:
            risk_level = 'high'
            action = 'urgent_referral'
        elif malignant_prob >= self.risk_thresholds['medium']:
            risk_level = 'medium'
            action = 'routine_referral'
        elif malignant_prob >= self.risk_thresholds['low']:
            risk_level = 'low'
            action = 'monitoring'
        else:
            risk_level = 'very_low'
            action = 'reassurance'
        
        return risk_level, action
    
    def generate_recommendations(self, 
                               predicted_class: str,
                               risk_level: str,
                               action: str) -> Dict[str, str]:
        """
        Generate clinical recommendations.
        
        Args:
            predicted_class: Predicted disease class
            risk_level: Risk stratification level
            action: Recommended action
        
        Returns:
            Dictionary of recommendations
        """
        guidelines = self.clinical_guidelines.get(predicted_class, {})
        
        recommendations = {
            'primary_action': action,
            'urgency': guidelines.get('urgency', 'routine'),
            'follow_up': guidelines.get('follow_up', 'as needed'),
            'additional_tests': ', '.join(guidelines.get('additional_tests', [])),
            'patient_advice': guidelines.get('patient_advice', 'Follow up as directed'),
            'risk_level': risk_level
        }
        
        # Add risk-specific modifications
        if risk_level in ['critical', 'high']:
            recommendations['primary_action'] = 'urgent_specialist_referral'
            recommendations['follow_up'] = '2-week pathway'
        elif risk_level == 'medium':
            recommendations['follow_up'] = '4-6 weeks'
        
        return recommendations


class ClinicalDecisionSupport:
    """
    Main clinical decision support system.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 class_names: List[str],
                 device: torch.device = None):
        """
        Initialize clinical decision support system.
        
        Args:
            model: Trained model
            class_names: List of class names
            device: Computing device
        """
        self.model = model
        self.class_names = class_names
        self.device = device or torch.device('cpu')
        
        # Initialize components
        self.risk_system = RiskStratificationSystem(class_names)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, 
               image: torch.Tensor,
               clinical_features: Optional[Dict] = None,
               return_explanations: bool = True) -> ClinicalPrediction:
        """
        Generate clinical prediction with decision support.
        
        Args:
            image: Input image tensor
            clinical_features: Optional clinical features
            return_explanations: Whether to include explanations
        
        Returns:
            ClinicalPrediction object
        """
        with torch.no_grad():
            # Get model prediction
            if hasattr(self.model, 'forward'):
                outputs = self.model(image.to(self.device))
            else:
                # Handle uncertainty quantification models
                outputs, uncertainty = self.model(image.to(self.device), return_uncertainty=True)
                uncertainty = uncertainty.mean().item() if torch.is_tensor(uncertainty) else uncertainty
            
            # Convert to probabilities
            probabilities = F.softmax(outputs, dim=1)
            prob_values = probabilities.cpu().numpy().flatten()
            
            # Get predicted class
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = self.class_names[predicted_idx]
            confidence = prob_values[predicted_idx]
            
            # Create class probability dictionary
            class_probs = {name: float(prob) for name, prob in 
                          zip(self.class_names, prob_values)}
            
            # Risk stratification
            risk_level, action = self.risk_system.stratify_risk(
                prob_values, 
                uncertainty if 'uncertainty' in locals() else None,
                clinical_features
            )
            
            # Generate recommendations
            recommendations = self.risk_system.generate_recommendations(
                predicted_class, risk_level, action
            )
            
            # Differential diagnosis (top 3 predictions)
            sorted_indices = np.argsort(prob_values)[::-1]
            differential = [(self.class_names[idx], float(prob_values[idx])) 
                           for idx in sorted_indices[:3]]
            
            # Create clinical prediction
            prediction = ClinicalPrediction(
                predicted_class=predicted_class,
                confidence=confidence,
                class_probabilities=class_probs,
                risk_level=risk_level,
                recommended_action=recommendations['primary_action'],
                differential_diagnosis=differential,
                clinical_features=clinical_features or {},
                uncertainty=uncertainty if 'uncertainty' in locals() else None,
                explanation=self._generate_explanation(predicted_class, confidence, risk_level) if return_explanations else None,
                timestamp=datetime.now()
            )
        
        return prediction
    
    def _generate_explanation(self, 
                            predicted_class: str,
                            confidence: float,
                            risk_level: str) -> str:
        """Generate human-readable explanation."""
        explanation = f"The AI system predicts this lesion is most likely a {predicted_class} "
        explanation += f"with {confidence:.1%} confidence. "
        
        if risk_level in ['critical', 'high']:
            explanation += "This prediction suggests a potentially malignant lesion requiring urgent evaluation. "
        elif risk_level == 'medium':
            explanation += "This prediction suggests the lesion should be evaluated by a specialist. "
        else:
            explanation += "This prediction suggests a likely benign lesion with routine follow-up. "
        
        explanation += "Please note that AI predictions should always be interpreted alongside clinical judgment."
        
        return explanation
    
    def batch_predict(self, 
                     images: torch.Tensor,
                     clinical_features_list: Optional[List[Dict]] = None) -> List[ClinicalPrediction]:
        """
        Process multiple images in batch.
        
        Args:
            images: Batch of image tensors
            clinical_features_list: List of clinical features for each image
        
        Returns:
            List of ClinicalPrediction objects
        """
        predictions = []
        
        for i in range(images.shape[0]):
            image = images[i:i+1]  # Keep batch dimension
            clinical_features = clinical_features_list[i] if clinical_features_list else None
            
            prediction = self.predict(image, clinical_features)
            predictions.append(prediction)
        
        return predictions
    
    def export_prediction_report(self, 
                               prediction: ClinicalPrediction,
                               patient_info: Optional[Dict] = None,
                               include_image: bool = False) -> Dict:
        """
        Export prediction as structured clinical report.
        
        Args:
            prediction: Clinical prediction
            patient_info: Optional patient information
            include_image: Whether to include image data
        
        Returns:
            Structured clinical report
        """
        report = {
            'report_id': f"AI_DERM_{prediction.timestamp.strftime('%Y%m%d_%H%M%S')}",
            'timestamp': prediction.timestamp.isoformat(),
            'patient_info': patient_info or {},
            'ai_analysis': {
                'primary_diagnosis': prediction.predicted_class,
                'confidence': round(prediction.confidence, 3),
                'risk_level': prediction.risk_level,
                'differential_diagnosis': prediction.differential_diagnosis,
                'class_probabilities': {k: round(v, 3) for k, v in prediction.class_probabilities.items()}
            },
            'clinical_recommendations': {
                'recommended_action': prediction.recommended_action,
                'urgency': prediction.risk_level,
                'explanation': prediction.explanation
            },
            'clinical_features': prediction.clinical_features,
            'uncertainty': round(prediction.uncertainty, 3) if prediction.uncertainty else None,
            'disclaimer': "This AI analysis is intended to support clinical decision-making and should not replace professional medical judgment."
        }
        
        return report


class QualityAssurance:
    """
    Quality assurance system for clinical deployment.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize QA system.
        
        Args:
            confidence_threshold: Minimum confidence for reliable predictions
        """
        self.confidence_threshold = confidence_threshold
        self.quality_metrics = []
    
    def assess_prediction_quality(self, prediction: ClinicalPrediction) -> Dict[str, bool]:
        """
        Assess the quality of a prediction for clinical use.
        
        Args:
            prediction: Clinical prediction to assess
        
        Returns:
            Dictionary of quality indicators
        """
        quality_checks = {
            'sufficient_confidence': prediction.confidence >= self.confidence_threshold,
            'low_uncertainty': prediction.uncertainty < 0.5 if prediction.uncertainty else True,
            'clear_differential': len(prediction.differential_diagnosis) >= 2,
            'consistent_risk': self._check_risk_consistency(prediction),
            'valid_clinical_features': bool(prediction.clinical_features)
        }
        
        quality_checks['overall_quality'] = all([
            quality_checks['sufficient_confidence'],
            quality_checks['low_uncertainty'],
            quality_checks['consistent_risk']
        ])
        
        return quality_checks
    
    def _check_risk_consistency(self, prediction: ClinicalPrediction) -> bool:
        """Check if risk level is consistent with prediction."""
        malignant_classes = ['melanoma', 'basal_cell_carcinoma', 'squamous_cell_carcinoma']
        
        if prediction.predicted_class in malignant_classes:
            return prediction.risk_level in ['medium', 'high', 'critical']
        else:
            return prediction.risk_level in ['very_low', 'low', 'medium']
    
    def log_prediction(self, 
                      prediction: ClinicalPrediction,
                      quality_assessment: Dict[str, bool]):
        """Log prediction for quality monitoring."""
        log_entry = {
            'timestamp': prediction.timestamp,
            'predicted_class': prediction.predicted_class,
            'confidence': prediction.confidence,
            'risk_level': prediction.risk_level,
            'quality_score': sum(quality_assessment.values()) / len(quality_assessment),
            'quality_details': quality_assessment
        }
        
        self.quality_metrics.append(log_entry)
    
    def generate_quality_report(self) -> Dict:
        """Generate quality assurance report."""
        if not self.quality_metrics:
            return {'message': 'No predictions logged'}
        
        df = pd.DataFrame(self.quality_metrics)
        
        report = {
            'total_predictions': len(df),
            'average_confidence': df['confidence'].mean(),
            'average_quality_score': df['quality_score'].mean(),
            'predictions_by_class': df['predicted_class'].value_counts().to_dict(),
            'predictions_by_risk': df['risk_level'].value_counts().to_dict(),
            'quality_trends': {
                'high_quality_predictions': (df['quality_score'] >= 0.8).sum(),
                'medium_quality_predictions': ((df['quality_score'] >= 0.6) & (df['quality_score'] < 0.8)).sum(),
                'low_quality_predictions': (df['quality_score'] < 0.6).sum()
            }
        }
        
        return report


# Export classes
__all__ = [
    'ClinicalPrediction',
    'RiskStratificationSystem',
    'ClinicalDecisionSupport',
    'QualityAssurance'
]


if __name__ == "__main__":
    # Test clinical decision support system
    print("Testing Clinical Decision Support System...")
    
    # Mock model for testing
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.classifier = nn.Linear(100, num_classes)
        
        def forward(self, x):
            # Flatten input for testing
            x = x.view(x.size(0), -1)
            if x.size(1) != 100:
                x = torch.randn(x.size(0), 100)
            return self.classifier(x)
    
    # Test setup
    class_names = ['melanoma', 'nevus', 'basal_cell_carcinoma', 'actinic_keratosis',
                   'benign_keratosis', 'dermatofibroma', 'vascular_lesion', 'squamous_cell_carcinoma']
    
    model = MockModel(len(class_names))
    
    # Initialize clinical decision support
    cds = ClinicalDecisionSupport(model, class_names)
    
    # Test prediction
    dummy_image = torch.randn(1, 3, 224, 224)
    clinical_features = {'abcd_score': 3.2, 'diameter_mm': 5.1}
    
    prediction = cds.predict(dummy_image, clinical_features)
    
    print(f"Predicted class: {prediction.predicted_class}")
    print(f"Confidence: {prediction.confidence:.3f}")
    print(f"Risk level: {prediction.risk_level}")
    print(f"Recommended action: {prediction.recommended_action}")
    
    # Test quality assurance
    qa = QualityAssurance()
    quality_assessment = qa.assess_prediction_quality(prediction)
    print(f"Quality assessment: {quality_assessment}")
    
    # Test report export
    report = cds.export_prediction_report(prediction)
    print(f"Report generated with ID: {report['report_id']}")
    
    print("Clinical decision support testing completed successfully!")