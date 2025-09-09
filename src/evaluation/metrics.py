"""
Comprehensive Evaluation and Metrics Module
===========================================

Implements advanced evaluation metrics, bias assessment, and clinical validation
for skin disease detection models.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve,
    classification_report, cohen_kappa_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class MetricsCalculator:
    """
    Comprehensive metrics calculator for medical image classification.
    """
    
    def __init__(self, 
                 num_classes: int,
                 class_names: Optional[List[str]] = None,
                 average: str = 'weighted'):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: List of class names
            average: Averaging strategy ('micro', 'macro', 'weighted')
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]
        self.average = average
    
    def calculate_metrics(self, 
                         predictions: Union[np.ndarray, List],
                         targets: Union[np.ndarray, List],
                         probabilities: Optional[np.ndarray] = None,
                         prefix: str = '') -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            predictions: Predicted class labels
            targets: True class labels
            probabilities: Prediction probabilities (for AUC calculation)
            prefix: Prefix for metric names
        
        Returns:
            Dictionary of calculated metrics
        """
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        if len(predictions) == 0 or len(targets) == 0:
            return {}
        
        metrics = {}
        prefix = f"{prefix}/" if prefix and not prefix.endswith('/') else prefix
        
        # Basic accuracy metrics
        metrics[f'{prefix}accuracy'] = accuracy_score(targets, predictions)
        metrics[f'{prefix}balanced_accuracy'] = balanced_accuracy_score(targets, predictions)
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=self.average, zero_division=0
        )
        
        metrics[f'{prefix}precision_{self.average}'] = precision
        metrics[f'{prefix}recall_{self.average}'] = recall
        metrics[f'{prefix}f1_{self.average}'] = f1
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'{prefix}precision_{class_name}'] = precision_per_class[i]
                metrics[f'{prefix}recall_{class_name}'] = recall_per_class[i]
                metrics[f'{prefix}f1_{class_name}'] = f1_per_class[i]
        
        # Cohen's Kappa
        metrics[f'{prefix}cohen_kappa'] = cohen_kappa_score(targets, predictions)
        
        # Sensitivity and Specificity for binary problems or multiclass
        if self.num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
            metrics[f'{prefix}sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'{prefix}specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'{prefix}ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            metrics[f'{prefix}npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        # AUC metrics (if probabilities provided)
        if probabilities is not None:
            try:
                if self.num_classes == 2:
                    # Binary classification
                    metrics[f'{prefix}auc_roc'] = roc_auc_score(targets, probabilities[:, 1])
                else:
                    # Multiclass classification
                    targets_binarized = label_binarize(targets, classes=range(self.num_classes))
                    
                    # Macro-average AUC
                    auc_macro = roc_auc_score(targets_binarized, probabilities, 
                                            average='macro', multi_class='ovr')
                    metrics[f'{prefix}auc_roc_macro'] = auc_macro
                    
                    # Weighted-average AUC
                    auc_weighted = roc_auc_score(targets_binarized, probabilities,
                                               average='weighted', multi_class='ovr')
                    metrics[f'{prefix}auc_roc_weighted'] = auc_weighted
                    
                    # Per-class AUC
                    for i, class_name in enumerate(self.class_names):
                        if i < probabilities.shape[1]:
                            class_auc = roc_auc_score(targets_binarized[:, i], probabilities[:, i])
                            metrics[f'{prefix}auc_roc_{class_name}'] = class_auc
            
            except Exception as e:
                print(f"Warning: Could not calculate AUC metrics: {e}")
        
        return metrics
    
    def calculate_confusion_matrix(self, 
                                 predictions: np.ndarray,
                                 targets: np.ndarray,
                                 normalize: Optional[str] = None) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            normalize: Normalization method ('true', 'pred', 'all', None)
        
        Returns:
            Confusion matrix
        """
        return confusion_matrix(targets, predictions, normalize=normalize)
    
    def plot_confusion_matrix(self,
                            predictions: np.ndarray,
                            targets: np.ndarray,
                            normalize: Optional[str] = None,
                            title: str = 'Confusion Matrix',
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot confusion matrix heatmap.
        
        Args:
            predictions: Predicted labels
            targets: True labels
            normalize: Normalization method
            title: Plot title
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        cm = self.calculate_confusion_matrix(predictions, targets, normalize)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cmap='Blues', ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self,
                       targets: np.ndarray,
                       probabilities: np.ndarray,
                       title: str = 'ROC Curves',
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot ROC curves for multiclass classification.
        
        Args:
            targets: True labels
            probabilities: Prediction probabilities
            title: Plot title
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Binarize targets
        targets_binarized = label_binarize(targets, classes=range(self.num_classes))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            if i < probabilities.shape[1]:
                fpr, tpr, _ = roc_curve(targets_binarized[:, i], probabilities[:, i])
                auc = roc_auc_score(targets_binarized[:, i], probabilities[:, i])
                
                ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class BiasMetrics:
    """
    Bias and fairness metrics calculator for demographic groups.
    """
    
    def __init__(self):
        """Initialize bias metrics calculator."""
        pass
    
    def calculate_bias_metrics(self,
                             predictions: np.ndarray,
                             targets: np.ndarray,
                             metadata: List[Dict],
                             sensitive_attributes: List[str] = None,
                             prefix: str = '') -> Dict[str, float]:
        """
        Calculate bias metrics across demographic groups.
        
        Args:
            predictions: Model predictions
            targets: True labels
            metadata: List of metadata dictionaries
            sensitive_attributes: List of sensitive attribute names
            prefix: Prefix for metric names
        
        Returns:
            Dictionary of bias metrics
        """
        if not metadata or len(metadata) == 0:
            return {}
        
        sensitive_attributes = sensitive_attributes or ['skin_tone', 'gender', 'age_group']
        bias_metrics = {}
        prefix = f"{prefix}/" if prefix and not prefix.endswith('/') else prefix
        
        # Convert metadata to DataFrame for easier processing
        metadata_df = pd.DataFrame(metadata)
        
        for attr in sensitive_attributes:
            if attr not in metadata_df.columns:
                continue
            
            # Group-wise accuracy
            group_accuracies = self._calculate_group_accuracies(
                predictions, targets, metadata_df[attr]
            )
            
            if len(group_accuracies) > 1:
                # Statistical parity difference
                spd = self._statistical_parity_difference(
                    predictions, metadata_df[attr]
                )
                bias_metrics[f'{prefix}spd_{attr}'] = spd
                
                # Equalized odds difference
                eod = self._equalized_odds_difference(
                    predictions, targets, metadata_df[attr]
                )
                bias_metrics[f'{prefix}eod_{attr}'] = eod
                
                # Demographic parity difference
                dpd = max(group_accuracies.values()) - min(group_accuracies.values())
                bias_metrics[f'{prefix}dpd_{attr}'] = dpd
                
                # Individual group accuracies
                for group, accuracy in group_accuracies.items():
                    bias_metrics[f'{prefix}accuracy_{attr}_{group}'] = accuracy
        
        return bias_metrics
    
    def _calculate_group_accuracies(self,
                                  predictions: np.ndarray,
                                  targets: np.ndarray,
                                  groups: pd.Series) -> Dict[str, float]:
        """Calculate accuracy for each demographic group."""
        group_accuracies = {}
        
        for group in groups.unique():
            group_mask = groups == group
            if group_mask.sum() > 0:
                group_preds = predictions[group_mask]
                group_targets = targets[group_mask]
                accuracy = accuracy_score(group_targets, group_preds)
                group_accuracies[str(group)] = accuracy
        
        return group_accuracies
    
    def _statistical_parity_difference(self,
                                     predictions: np.ndarray,
                                     groups: pd.Series) -> float:
        """Calculate statistical parity difference."""
        positive_rates = {}
        
        for group in groups.unique():
            group_mask = groups == group
            if group_mask.sum() > 0:
                group_preds = predictions[group_mask]
                # Assuming positive class is anything > 0 (malignant)
                positive_rate = (group_preds > 0).mean()
                positive_rates[group] = positive_rate
        
        if len(positive_rates) < 2:
            return 0.0
        
        return max(positive_rates.values()) - min(positive_rates.values())
    
    def _equalized_odds_difference(self,
                                 predictions: np.ndarray,
                                 targets: np.ndarray,
                                 groups: pd.Series) -> float:
        """Calculate equalized odds difference."""
        tpr_rates = {}
        
        for group in groups.unique():
            group_mask = groups == group
            if group_mask.sum() > 0:
                group_preds = predictions[group_mask]
                group_targets = targets[group_mask]
                
                # True positive rate for positive class
                positive_mask = group_targets > 0
                if positive_mask.sum() > 0:
                    tpr = (group_preds[positive_mask] > 0).mean()
                    tpr_rates[group] = tpr
        
        if len(tpr_rates) < 2:
            return 0.0
        
        return max(tpr_rates.values()) - min(tpr_rates.values())


class ClinicalEvaluator:
    """
    Clinical evaluation metrics specific to dermatological diagnosis.
    """
    
    def __init__(self, malignant_classes: List[int] = None):
        """
        Initialize clinical evaluator.
        
        Args:
            malignant_classes: List of class indices considered malignant
        """
        self.malignant_classes = malignant_classes or [0]  # Assuming melanoma is class 0
    
    def calculate_clinical_metrics(self,
                                 predictions: np.ndarray,
                                 targets: np.ndarray,
                                 probabilities: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate clinical evaluation metrics.
        
        Args:
            predictions: Model predictions
            targets: True labels
            probabilities: Prediction probabilities
        
        Returns:
            Dictionary of clinical metrics
        """
        metrics = {}
        
        # Convert to binary: malignant vs benign
        pred_malignant = np.isin(predictions, self.malignant_classes)
        true_malignant = np.isin(targets, self.malignant_classes)
        
        # Clinical confusion matrix
        tn = np.sum((~pred_malignant) & (~true_malignant))
        fp = np.sum(pred_malignant & (~true_malignant))
        fn = np.sum((~pred_malignant) & true_malignant)
        tp = np.sum(pred_malignant & true_malignant)
        
        # Clinical metrics
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for malignant
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for benign
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision for malignant
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Precision for benign
        
        # Clinical accuracy
        metrics['clinical_accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Number needed to screen (NNS) - simplified
        if metrics['sensitivity'] > 0:
            metrics['nns'] = 1 / metrics['sensitivity']
        else:
            metrics['nns'] = float('inf')
        
        # Diagnostic odds ratio
        if fp > 0 and fn > 0:
            metrics['diagnostic_odds_ratio'] = (tp * tn) / (fp * fn)
        else:
            metrics['diagnostic_odds_ratio'] = float('inf')
        
        # F1-score for malignant class
        if metrics['sensitivity'] > 0 and metrics['ppv'] > 0:
            metrics['f1_malignant'] = 2 * (metrics['sensitivity'] * metrics['ppv']) / (metrics['sensitivity'] + metrics['ppv'])
        else:
            metrics['f1_malignant'] = 0
        
        return metrics
    
    def calculate_risk_stratification_metrics(self,
                                            probabilities: np.ndarray,
                                            targets: np.ndarray,
                                            risk_thresholds: List[float] = None) -> Dict[str, Dict]:
        """
        Calculate metrics for different risk stratification thresholds.
        
        Args:
            probabilities: Prediction probabilities for malignant class
            targets: True labels
            risk_thresholds: List of probability thresholds for risk stratification
        
        Returns:
            Dictionary of metrics for each threshold
        """
        risk_thresholds = risk_thresholds or [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Get malignant class probabilities
        if len(probabilities.shape) > 1:
            malignant_probs = probabilities[:, self.malignant_classes[0]]
        else:
            malignant_probs = probabilities
        
        true_malignant = np.isin(targets, self.malignant_classes)
        
        threshold_metrics = {}
        
        for threshold in risk_thresholds:
            pred_malignant = malignant_probs >= threshold
            
            # Calculate metrics at this threshold
            tn = np.sum((~pred_malignant) & (~true_malignant))
            fp = np.sum(pred_malignant & (~true_malignant))
            fn = np.sum((~pred_malignant) & true_malignant)
            tp = np.sum(pred_malignant & true_malignant)
            
            threshold_metrics[f'threshold_{threshold}'] = {
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                'referral_rate': pred_malignant.mean()
            }
        
        return threshold_metrics


class ModelComparison:
    """
    Compare multiple models and generate comprehensive evaluation reports.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize model comparison.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def compare_models(self,
                      model_results: Dict[str, Dict],
                      test_data: Dict) -> Dict:
        """
        Compare multiple models across various metrics.
        
        Args:
            model_results: Dictionary of model results
            test_data: Test dataset information
        
        Returns:
            Comprehensive comparison report
        """
        comparison = {
            'model_summary': {},
            'metric_comparison': {},
            'statistical_tests': {},
            'recommendations': {}
        }
        
        # Extract metrics for all models
        all_metrics = {}
        for model_name, results in model_results.items():
            all_metrics[model_name] = results.get('metrics', {})
        
        # Compare key metrics
        key_metrics = ['accuracy', 'balanced_accuracy', 'f1_weighted', 'auc_roc_weighted']
        
        for metric in key_metrics:
            metric_values = {}
            for model_name, metrics in all_metrics.items():
                if metric in metrics:
                    metric_values[model_name] = metrics[metric]
            
            if metric_values:
                comparison['metric_comparison'][metric] = {
                    'values': metric_values,
                    'best_model': max(metric_values.keys(), key=lambda k: metric_values[k]),
                    'worst_model': min(metric_values.keys(), key=lambda k: metric_values[k]),
                    'range': max(metric_values.values()) - min(metric_values.values())
                }
        
        # Model summary
        for model_name, results in model_results.items():
            comparison['model_summary'][model_name] = {
                'accuracy': results.get('metrics', {}).get('accuracy', 0),
                'parameter_count': results.get('parameter_count', 0),
                'inference_time': results.get('inference_time', 0),
                'model_size': results.get('model_size', 0)
            }
        
        return comparison


# Evaluation module initialization
__all__ = [
    'MetricsCalculator',
    'BiasMetrics', 
    'ClinicalEvaluator',
    'ModelComparison'
]


if __name__ == "__main__":
    # Test evaluation components
    print("Testing Evaluation Components...")
    
    # Mock data
    num_samples, num_classes = 100, 8
    predictions = np.random.randint(0, num_classes, num_samples)
    targets = np.random.randint(0, num_classes, num_samples)
    probabilities = np.random.rand(num_samples, num_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    
    # Test MetricsCalculator
    class_names = ['melanoma', 'nevus', 'basal_cell_carcinoma', 'actinic_keratosis',
                   'benign_keratosis', 'dermatofibroma', 'vascular_lesion', 'squamous_cell_carcinoma']
    
    metrics_calc = MetricsCalculator(num_classes, class_names)
    metrics = metrics_calc.calculate_metrics(predictions, targets, probabilities)
    
    print(f"Calculated {len(metrics)} metrics")
    print(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
    
    # Test BiasMetrics
    metadata = [{'skin_tone': np.random.choice(['light', 'medium', 'dark']),
                 'gender': np.random.choice(['male', 'female'])} 
                for _ in range(num_samples)]
    
    bias_metrics = BiasMetrics()
    bias_results = bias_metrics.calculate_bias_metrics(predictions, targets, metadata)
    
    print(f"Calculated {len(bias_results)} bias metrics")
    
    # Test ClinicalEvaluator
    clinical_eval = ClinicalEvaluator(malignant_classes=[0])  # melanoma
    clinical_metrics = clinical_eval.calculate_clinical_metrics(predictions, targets, probabilities)
    
    print(f"Clinical sensitivity: {clinical_metrics.get('sensitivity', 0):.4f}")
    print(f"Clinical specificity: {clinical_metrics.get('specificity', 0):.4f}")
    
    print("Evaluation testing completed successfully!")