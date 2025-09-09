"""Training module for skin disease detection."""

from .trainer import AdvancedTrainer
from .losses import (
    FocalLoss,
    LabelSmoothingLoss,
    DistillationLoss,
    FairnessLoss,
    create_loss_function
)

__all__ = [
    'AdvancedTrainer',
    'FocalLoss',
    'LabelSmoothingLoss',
    'DistillationLoss', 
    'FairnessLoss',
    'create_loss_function'
]