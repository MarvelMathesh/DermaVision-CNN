"""
Advanced Loss Functions for Medical Image Classification
=======================================================

Implements specialized loss functions for skin disease detection including
Focal Loss, Label Smoothing, Knowledge Distillation, and Fairness-aware losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical image classification.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection. ICCV, 2017.
    """
    
    def __init__(self, 
                 alpha: Optional[Union[float, torch.Tensor]] = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: None)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
            ignore_index: Index to ignore in loss calculation
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(1) * alpha
        elif isinstance(alpha, list):
            self.alpha = torch.FloatTensor(alpha)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions of shape (N, C) where N is batch size, C is number of classes
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Computed focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            logpt = -ce_loss
            focal_loss = alpha_t * (1 - pt) ** self.gamma * logpt
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for improving model calibration and reducing overconfidence.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            num_classes: Number of classes
            smoothing: Smoothing parameter (default: 0.1)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            inputs: Predictions of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Computed label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -smooth_targets * log_probs
        return loss.sum(dim=1).mean()


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for transferring knowledge from teacher to student model.
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        """
        Initialize Distillation Loss.
        
        Args:
            temperature: Temperature for softmax (default: 4.0)
            alpha: Weight for distillation loss vs ground truth loss (default: 0.7)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_outputs: torch.Tensor, 
                targets: torch.Tensor,
                teacher_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_outputs: Student model predictions
            targets: Ground truth labels
            teacher_outputs: Teacher model predictions
        
        Returns:
            Combined distillation and ground truth loss
        """
        # Distillation loss
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_outputs / self.temperature, dim=1)
        
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Ground truth loss
        ground_truth_loss = self.ce_loss(student_outputs, targets)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * ground_truth_loss
        
        return total_loss
    
    def forward_with_teacher(self, student_outputs: torch.Tensor,
                           targets: torch.Tensor,
                           teacher_outputs: torch.Tensor) -> torch.Tensor:
        """Alias for forward method for compatibility."""
        return self.forward(student_outputs, targets, teacher_outputs)


class FairnessLoss(nn.Module):
    """
    Fairness-aware loss function to mitigate demographic bias in predictions.
    """
    
    def __init__(self, fairness_type: str = 'demographic_parity', lambda_fair: float = 0.1):
        """
        Initialize Fairness Loss.
        
        Args:
            fairness_type: Type of fairness constraint ('demographic_parity', 'equalized_odds')
            lambda_fair: Weight for fairness penalty (default: 0.1)
        """
        super().__init__()
        self.fairness_type = fairness_type
        self.lambda_fair = lambda_fair
    
    def forward(self, outputs: torch.Tensor, 
                targets: torch.Tensor,
                metadata: Dict) -> torch.Tensor:
        """
        Compute fairness penalty.
        
        Args:
            outputs: Model predictions
            targets: Ground truth labels
            metadata: Demographic metadata (skin_tone, gender, age, etc.)
        
        Returns:
            Fairness penalty term
        """
        if not metadata or 'skin_tone' not in metadata:
            return torch.tensor(0.0, device=outputs.device, requires_grad=True)
        
        skin_tones = metadata['skin_tone']
        predictions = F.softmax(outputs, dim=1)
        
        fairness_penalty = 0.0
        
        if self.fairness_type == 'demographic_parity':
            # Ensure equal positive prediction rates across groups
            fairness_penalty = self._demographic_parity_penalty(predictions, skin_tones)
        elif self.fairness_type == 'equalized_odds':
            # Ensure equal TPR and FPR across groups
            fairness_penalty = self._equalized_odds_penalty(predictions, targets, skin_tones)
        
        return self.lambda_fair * fairness_penalty
    
    def _demographic_parity_penalty(self, predictions: torch.Tensor, 
                                  skin_tones: List[str]) -> torch.Tensor:
        """Compute demographic parity penalty."""
        unique_tones = list(set(skin_tones))
        if len(unique_tones) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Calculate positive prediction rates for each group
        group_rates = {}
        for tone in unique_tones:
            tone_mask = torch.tensor([tone == st for st in skin_tones], 
                                   device=predictions.device)
            if tone_mask.sum() > 0:
                group_predictions = predictions[tone_mask]
                # Assuming class 0 is negative, others are positive (malignant)
                positive_rate = (group_predictions[:, 1:].sum(dim=1) > 0.5).float().mean()
                group_rates[tone] = positive_rate
        
        # Calculate penalty as variance of group rates
        if len(group_rates) >= 2:
            rates = torch.stack(list(group_rates.values()))
            penalty = torch.var(rates)
            return penalty
        
        return torch.tensor(0.0, device=predictions.device)
    
    def _equalized_odds_penalty(self, predictions: torch.Tensor,
                              targets: torch.Tensor,
                              skin_tones: List[str]) -> torch.Tensor:
        """Compute equalized odds penalty."""
        unique_tones = list(set(skin_tones))
        if len(unique_tones) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        penalty = 0.0
        pred_labels = torch.argmax(predictions, dim=1)
        
        # Calculate TPR and FPR for each group
        group_tpr = {}
        group_fpr = {}
        
        for tone in unique_tones:
            tone_mask = torch.tensor([tone == st for st in skin_tones], 
                                   device=predictions.device)
            if tone_mask.sum() > 0:
                group_preds = pred_labels[tone_mask]
                group_targets = targets[tone_mask]
                
                # TPR: True Positive Rate
                if (group_targets > 0).sum() > 0:
                    tp = ((group_preds > 0) & (group_targets > 0)).sum().float()
                    fn = ((group_preds == 0) & (group_targets > 0)).sum().float()
                    tpr = tp / (tp + fn + 1e-8)
                    group_tpr[tone] = tpr
                
                # FPR: False Positive Rate
                if (group_targets == 0).sum() > 0:
                    fp = ((group_preds > 0) & (group_targets == 0)).sum().float()
                    tn = ((group_preds == 0) & (group_targets == 0)).sum().float()
                    fpr = fp / (fp + tn + 1e-8)
                    group_fpr[tone] = fpr
        
        # Calculate penalty as variance of TPR and FPR across groups
        if len(group_tpr) >= 2:
            tpr_penalty = torch.var(torch.stack(list(group_tpr.values())))
            penalty += tpr_penalty
        
        if len(group_fpr) >= 2:
            fpr_penalty = torch.var(torch.stack(list(group_fpr.values())))
            penalty += fpr_penalty
        
        return penalty


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation-like problems or when dealing with very imbalanced classes.
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Predictions of shape (N, C)
            targets: Ground truth labels of shape (N,)
        
        Returns:
            Computed Dice loss
        """
        # Convert to one-hot encoding
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        
        # Apply softmax to inputs
        inputs_soft = F.softmax(inputs, dim=1)
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for i in range(num_classes):
            pred_i = inputs_soft[:, i]
            target_i = targets_one_hot[:, i]
            
            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Return 1 - mean Dice score
        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


class CombinedLoss(nn.Module):
    """
    Combined loss function that can mix multiple loss types with different weights.
    """
    
    def __init__(self, 
                 loss_types: List[str],
                 loss_weights: List[float],
                 num_classes: int,
                 **kwargs):
        """
        Initialize Combined Loss.
        
        Args:
            loss_types: List of loss function names
            loss_weights: Weights for each loss function
            num_classes: Number of classes
            **kwargs: Additional parameters for individual losses
        """
        super().__init__()
        self.loss_types = loss_types
        self.loss_weights = loss_weights
        self.losses = nn.ModuleDict()
        
        for loss_type in loss_types:
            if loss_type == 'focal':
                self.losses[loss_type] = FocalLoss(**kwargs.get('focal_params', {}))
            elif loss_type == 'label_smoothing':
                self.losses[loss_type] = LabelSmoothingLoss(
                    num_classes, **kwargs.get('smoothing_params', {})
                )
            elif loss_type == 'dice':
                self.losses[loss_type] = DiceLoss(**kwargs.get('dice_params', {}))
            elif loss_type == 'ce':
                self.losses[loss_type] = nn.CrossEntropyLoss(**kwargs.get('ce_params', {}))
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            inputs: Model predictions
            targets: Ground truth labels
        
        Returns:
            Weighted combination of losses
        """
        total_loss = 0.0
        
        for loss_type, weight in zip(self.loss_types, self.loss_weights):
            loss_value = self.losses[loss_type](inputs, targets)
            total_loss += weight * loss_value
        
        return total_loss


class UncertaintyLoss(nn.Module):
    """
    Loss function that incorporates prediction uncertainty for improved calibration.
    """
    
    def __init__(self, base_loss: nn.Module, uncertainty_weight: float = 0.1):
        """
        Initialize Uncertainty Loss.
        
        Args:
            base_loss: Base loss function (e.g., CrossEntropyLoss)
            uncertainty_weight: Weight for uncertainty penalty
        """
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_weight = uncertainty_weight
    
    def forward(self, 
                inputs: torch.Tensor, 
                targets: torch.Tensor,
                uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.
        
        Args:
            inputs: Model predictions
            targets: Ground truth labels
            uncertainty: Prediction uncertainty values
        
        Returns:
            Combined loss with uncertainty penalty
        """
        # Base classification loss
        base_loss_value = self.base_loss(inputs, targets)
        
        # Uncertainty penalty - encourage low uncertainty for correct predictions
        pred_labels = torch.argmax(inputs, dim=1)
        correct_mask = (pred_labels == targets).float()
        
        # Penalize high uncertainty for correct predictions
        uncertainty_penalty = (uncertainty * correct_mask).mean()
        
        # Encourage high uncertainty for incorrect predictions
        incorrect_mask = (pred_labels != targets).float()
        uncertainty_reward = -(uncertainty * incorrect_mask).mean()
        
        total_uncertainty_term = uncertainty_penalty + uncertainty_reward
        
        return base_loss_value + self.uncertainty_weight * total_uncertainty_term


# Factory function for creating loss functions
def create_loss_function(loss_config: Dict, num_classes: int) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.
    
    Args:
        loss_config: Configuration dictionary for loss function
        num_classes: Number of classes
    
    Returns:
        Initialized loss function
    """
    loss_type = loss_config.get('type', 'cross_entropy')
    
    if loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('alpha'),
            gamma=loss_config.get('gamma', 2.0),
            reduction=loss_config.get('reduction', 'mean')
        )
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(
            num_classes=num_classes,
            smoothing=loss_config.get('smoothing', 0.1)
        )
    elif loss_type == 'distillation':
        return DistillationLoss(
            temperature=loss_config.get('temperature', 4.0),
            alpha=loss_config.get('alpha', 0.7)
        )
    elif loss_type == 'fairness':
        return FairnessLoss(
            fairness_type=loss_config.get('fairness_type', 'demographic_parity'),
            lambda_fair=loss_config.get('lambda_fair', 0.1)
        )
    elif loss_type == 'dice':
        return DiceLoss(smooth=loss_config.get('smooth', 1.0))
    elif loss_type == 'combined':
        return CombinedLoss(
            loss_types=loss_config['loss_types'],
            loss_weights=loss_config['loss_weights'],
            num_classes=num_classes,
            **loss_config.get('params', {})
        )
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(
            weight=loss_config.get('class_weights'),
            reduction=loss_config.get('reduction', 'mean')
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    print("Testing Advanced Loss Functions...")
    
    # Mock data
    batch_size, num_classes = 8, 8
    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    teacher_outputs = torch.randn(batch_size, num_classes)
    uncertainty = torch.rand(batch_size)
    
    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    focal_result = focal_loss(inputs, targets)
    print(f"Focal Loss: {focal_result.item():.4f}")
    
    # Test Label Smoothing
    label_smoothing = LabelSmoothingLoss(num_classes, smoothing=0.1)
    smoothing_result = label_smoothing(inputs, targets)
    print(f"Label Smoothing Loss: {smoothing_result.item():.4f}")
    
    # Test Distillation Loss
    distillation_loss = DistillationLoss(temperature=4.0, alpha=0.7)
    distill_result = distillation_loss(inputs, targets, teacher_outputs)
    print(f"Distillation Loss: {distill_result.item():.4f}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dice_result = dice_loss(inputs, targets)
    print(f"Dice Loss: {dice_result.item():.4f}")
    
    # Test Combined Loss
    combined_loss = CombinedLoss(
        loss_types=['focal', 'label_smoothing'],
        loss_weights=[0.7, 0.3],
        num_classes=num_classes,
        focal_params={'gamma': 2.0},
        smoothing_params={'smoothing': 0.1}
    )
    combined_result = combined_loss(inputs, targets)
    print(f"Combined Loss: {combined_result.item():.4f}")
    
    print("Loss function testing completed successfully!")