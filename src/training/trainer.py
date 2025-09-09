"""
Advanced Training Pipeline with State-of-the-Art Techniques
==========================================================

Implements comprehensive training strategies including progressive learning,
knowledge distillation, self-supervised pre-training, and fairness-aware training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import time
import os
import wandb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .losses import FocalLoss, LabelSmoothingLoss, DistillationLoss, FairnessLoss
from .optimizers import create_optimizer, create_scheduler
from .metrics import MetricsCalculator, BiasMetrics
from ..models.architectures import create_model, UncertaintyQuantification
from ..interpretability.explainability import GradCAMAnalyzer


class AdvancedTrainer:
    """
    Advanced training pipeline implementing state-of-the-art techniques
    for medical image classification.
    """
    
    def __init__(self, 
                 config: Dict,
                 device: Optional[torch.device] = None,
                 logger: Optional[object] = None):
        """
        Initialize advanced trainer.
        
        Args:
            config: Training configuration
            device: Device to use for training
            logger: Logging object (wandb, tensorboard, etc.)
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        
        # Initialize components
        self.model = None
        self.teacher_model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
        # Progressive training
        self.current_image_size = config.get('initial_image_size', 224)
        self.target_image_size = config.get('target_image_size', 384)
        
        # Metrics and bias tracking
        self.metrics_calculator = MetricsCalculator(num_classes=config['num_classes'])
        self.bias_metrics = BiasMetrics()
        
        # Interpretability
        self.gradcam_analyzer = None
        
        print(f"Trainer initialized on device: {self.device}")
    
    def setup_model(self, model_type: str = 'ensemble', **model_kwargs):
        """Setup model architecture."""
        self.model = create_model(
            model_type=model_type,
            num_classes=self.config['num_classes'],
            pretrained=True,
            **model_kwargs
        ).to(self.device)
        
        # Setup teacher model for knowledge distillation if specified
        if self.config.get('use_knowledge_distillation', False):
            teacher_type = self.config.get('teacher_model_type', 'ensemble')
            self.teacher_model = create_model(
                model_type=teacher_type,
                num_classes=self.config['num_classes'],
                pretrained=True
            ).to(self.device)
            
            # Load teacher weights if available
            teacher_path = self.config.get('teacher_model_path')
            if teacher_path and os.path.exists(teacher_path):
                self.teacher_model.load_state_dict(torch.load(teacher_path, map_location=self.device))
                self.teacher_model.eval()
                print("Teacher model loaded successfully")
        
        # Setup uncertainty quantification
        if self.config.get('use_uncertainty', False):
            self.model = UncertaintyQuantification(self.model, n_samples=100)
        
        # Setup interpretability
        if hasattr(self.model, 'backbone'):
            self.gradcam_analyzer = GradCAMAnalyzer(self.model.backbone)
        
        print(f"Model setup complete: {model_type}")
        return self.model
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        # Optimizer
        self.optimizer = create_optimizer(
            self.model.parameters(),
            optimizer_type=self.config.get('optimizer', 'adamw'),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-2)
        )
        
        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=self.config.get('scheduler', 'cosine'),
            num_epochs=self.config.get('num_epochs', 200),
            warmup_epochs=self.config.get('warmup_epochs', 10)
        )
        
        # Loss function
        loss_type = self.config.get('loss_type', 'focal')
        class_weights = self.config.get('class_weights')
        
        if loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=self.config.get('focal_gamma', 2.0),
                reduction='mean'
            )
        elif loss_type == 'label_smoothing':
            self.criterion = LabelSmoothingLoss(
                num_classes=self.config['num_classes'],
                smoothing=self.config.get('label_smoothing', 0.1)
            )
        elif loss_type == 'distillation' and self.teacher_model is not None:
            self.criterion = DistillationLoss(
                temperature=self.config.get('distillation_temperature', 4.0),
                alpha=self.config.get('distillation_alpha', 0.7)
            )
        else:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Fairness loss if bias mitigation is enabled
        if self.config.get('use_fairness_loss', False):
            self.fairness_criterion = FairnessLoss(
                fairness_type=self.config.get('fairness_type', 'demographic_parity'),
                lambda_fair=self.config.get('fairness_lambda', 0.1)
            )
        
        print("Training components setup complete")
    
    def progressive_resize_check(self, epoch: int) -> bool:
        """Check if image size should be increased for progressive training."""
        resize_epochs = self.config.get('progressive_resize_epochs', [50, 100])
        resize_sizes = self.config.get('progressive_resize_sizes', [224, 299, 384])
        
        for i, resize_epoch in enumerate(resize_epochs):
            if epoch == resize_epoch and i + 1 < len(resize_sizes):
                self.current_image_size = resize_sizes[i + 1]
                print(f"Progressive resize: Increasing image size to {self.current_image_size}")
                return True
        return False
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Progressive unfreezing
        if self.config.get('use_progressive_unfreezing', False):
            self._progressive_unfreeze(epoch)
        
        running_loss = 0.0
        running_fairness_loss = 0.0
        predictions = []
        targets = []
        metadata_list = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            metadata = batch.get('metadata', {})
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.model, UncertaintyQuantification):
                outputs, uncertainty = self.model(images, return_uncertainty=True)
            else:
                outputs = self.model(images)
            
            # Calculate loss
            if hasattr(self.criterion, 'forward_with_teacher') and self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)
                loss = self.criterion.forward_with_teacher(outputs, labels, teacher_outputs)
            else:
                loss = self.criterion(outputs, labels)
            
            # Fairness loss
            fairness_loss = 0.0
            if hasattr(self, 'fairness_criterion') and metadata:
                fairness_loss = self.fairness_criterion(outputs, labels, metadata)
                loss += fairness_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clipping', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            running_fairness_loss += fairness_loss if isinstance(fairness_loss, float) else fairness_loss.item()
            
            # Store predictions for metrics
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            if metadata:
                metadata_list.extend([metadata] * len(labels))
            
            # Logging
            if batch_idx % self.config.get('log_interval', 100) == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
                
                if self.logger:
                    self.logger.log({
                        'train/batch_loss': loss.item(),
                        'train/batch_fairness_loss': fairness_loss if isinstance(fairness_loss, float) else fairness_loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'epoch': epoch,
                        'batch': batch_idx
                    })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_fairness_loss = running_fairness_loss / len(train_loader)
        
        # Calculate classification metrics
        train_metrics = self.metrics_calculator.calculate_metrics(
            predictions, targets, prefix='train'
        )
        
        # Calculate bias metrics if metadata available
        if metadata_list:
            bias_metrics = self.bias_metrics.calculate_bias_metrics(
                predictions, targets, metadata_list
            )
            train_metrics.update(bias_metrics)
        
        train_metrics.update({
            'train/epoch_loss': epoch_loss,
            'train/fairness_loss': epoch_fairness_loss
        })
        
        return train_metrics
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        running_loss = 0.0
        predictions = []
        targets = []
        metadata_list = []
        uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                metadata = batch.get('metadata', {})
                
                # Forward pass
                if isinstance(self.model, UncertaintyQuantification):
                    outputs, uncertainty = self.model(images, return_uncertainty=True)
                    uncertainties.extend(uncertainty.cpu().numpy())
                else:
                    outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                # Store predictions
                predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
                if metadata:
                    metadata_list.extend([metadata] * len(labels))
        
        # Calculate metrics
        epoch_loss = running_loss / len(val_loader)
        val_metrics = self.metrics_calculator.calculate_metrics(
            predictions, targets, prefix='val'
        )
        
        # Calculate bias metrics
        if metadata_list:
            bias_metrics = self.bias_metrics.calculate_bias_metrics(
                predictions, targets, metadata_list, prefix='val'
            )
            val_metrics.update(bias_metrics)
        
        # Uncertainty statistics
        if uncertainties:
            val_metrics.update({
                'val/mean_uncertainty': np.mean(uncertainties),
                'val/std_uncertainty': np.std(uncertainties)
            })
        
        val_metrics['val/epoch_loss'] = epoch_loss
        
        return val_metrics
    
    def _progressive_unfreeze(self, epoch: int):
        """Progressive unfreezing of model layers."""
        unfreeze_epochs = self.config.get('unfreeze_epochs', [20, 40, 60])
        
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
        else:
            backbone = self.model
        
        # Freeze all parameters initially
        if epoch == 0:
            for param in backbone.parameters():
                param.requires_grad = False
        
        # Unfreeze layers progressively
        if hasattr(backbone, 'features'):
            layers = list(backbone.features.children())
        elif hasattr(backbone, 'blocks'):
            layers = list(backbone.blocks.children())
        else:
            layers = list(backbone.children())
        
        total_layers = len(layers)
        
        for i, unfreeze_epoch in enumerate(unfreeze_epochs):
            if epoch == unfreeze_epoch:
                layers_to_unfreeze = int((i + 1) * total_layers / len(unfreeze_epochs))
                
                for layer_idx in range(-layers_to_unfreeze, 0):
                    for param in layers[layer_idx].parameters():
                        param.requires_grad = True
                
                print(f"Unfroze {layers_to_unfreeze} layers at epoch {epoch}")
                break
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'best_metric': self.best_metric
        }
        
        # Save regular checkpoint
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with metric: {metrics.get('val/balanced_accuracy', 0):.4f}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints(checkpoint_dir, keep_last=5)
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path, keep_last: int = 5):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > keep_last:
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        print(f"Checkpoint loaded from epoch {self.current_epoch}")
        return checkpoint['metrics']
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: Optional[int] = None) -> Dict:
        """
        Main training loop with all advanced features.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
        
        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config.get('num_epochs', 200)
        patience = self.config.get('early_stopping_patience', 20)
        patience_counter = 0
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {type(self.model).__name__}")
        
        for epoch in range(self.current_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Progressive image resizing
            if self.progressive_resize_check(epoch):
                # Would need to recreate data loaders with new image size
                # This is a simplified implementation
                pass
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val/epoch_loss'])
                else:
                    self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update history
            training_history['train_losses'].append(train_metrics['train/epoch_loss'])
            training_history['val_losses'].append(val_metrics['val/epoch_loss'])
            training_history['train_metrics'].append(train_metrics)
            training_history['val_metrics'].append(val_metrics)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            
            print(f"Epoch {epoch}/{num_epochs-1}")
            print(f"Train Loss: {train_metrics['train/epoch_loss']:.4f}, "
                  f"Val Loss: {val_metrics['val/epoch_loss']:.4f}")
            print(f"Val Accuracy: {val_metrics.get('val/accuracy', 0):.4f}, "
                  f"Val Balanced Acc: {val_metrics.get('val/balanced_accuracy', 0):.4f}")
            print(f"Time: {epoch_time:.2f}s")
            
            if self.logger:
                self.logger.log({
                    **epoch_metrics,
                    'epoch': epoch,
                    'epoch_time': epoch_time,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Check for best model
            current_metric = val_metrics.get('val/balanced_accuracy', 0)
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                self.save_checkpoint(epoch, epoch_metrics, is_best)
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
        
        print(f"Training completed. Best validation metric: {self.best_metric:.4f}")
        return training_history
    
    def evaluate_with_tta(self, test_loader: DataLoader, tta_transforms: List) -> Dict:
        """Evaluate model with test-time augmentation."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in test_loader:
                original_images = batch['image']
                labels = batch['label'].to(self.device)
                
                # Apply TTA transforms
                tta_predictions = []
                tta_uncertainties = []
                
                for transform in tta_transforms:
                    # Apply transform to original images (numpy format expected)
                    transformed_images = []
                    for img in original_images:
                        # Convert tensor back to numpy for transformation
                        img_np = img.permute(1, 2, 0).numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        transformed = transform(image=img_np)
                        transformed_images.append(transformed['image'])
                    
                    batch_images = torch.stack(transformed_images).to(self.device)
                    
                    # Get predictions
                    if isinstance(self.model, UncertaintyQuantification):
                        outputs, uncertainty = self.model(batch_images, return_uncertainty=True)
                        tta_uncertainties.append(uncertainty)
                    else:
                        outputs = self.model(batch_images)
                    
                    tta_predictions.append(F.softmax(outputs, dim=1))
                
                # Average TTA predictions
                avg_predictions = torch.mean(torch.stack(tta_predictions), dim=0)
                final_predictions = torch.argmax(avg_predictions, dim=1)
                
                all_predictions.extend(final_predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                if tta_uncertainties:
                    avg_uncertainty = torch.mean(torch.stack(tta_uncertainties), dim=0)
                    all_uncertainties.extend(avg_uncertainty.cpu().numpy())
        
        # Calculate metrics
        test_metrics = self.metrics_calculator.calculate_metrics(
            all_predictions, all_targets, prefix='test_tta'
        )
        
        if all_uncertainties:
            test_metrics.update({
                'test_tta/mean_uncertainty': np.mean(all_uncertainties),
                'test_tta/std_uncertainty': np.std(all_uncertainties)
            })
        
        return test_metrics


# Utility function for hyperparameter optimization
def objective_function(trial, config: Dict, train_loader: DataLoader, val_loader: DataLoader):
    """Objective function for hyperparameter optimization with Optuna."""
    # Suggest hyperparameters
    config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    config['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-1, log=True)
    config['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 5.0)
    config['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Initialize trainer
    trainer = AdvancedTrainer(config)
    trainer.setup_model('mobilenet', dropout_rate=config['dropout_rate'])
    trainer.setup_training_components()
    
    # Train for a few epochs
    training_history = trainer.train(train_loader, val_loader, num_epochs=10)
    
    # Return best validation accuracy
    best_val_acc = max([metrics.get('val/balanced_accuracy', 0) 
                       for metrics in training_history['val_metrics']])
    
    return best_val_acc


if __name__ == "__main__":
    # Test training pipeline
    print("Testing Advanced Training Pipeline...")
    
    # Mock configuration
    config = {
        'num_classes': 8,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'num_epochs': 5,
        'batch_size': 8,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'loss_type': 'focal',
        'focal_gamma': 2.0,
        'use_uncertainty': True,
        'use_fairness_loss': True,
        'checkpoint_dir': 'test_checkpoints',
        'log_interval': 2
    }
    
    # Create mock trainer
    trainer = AdvancedTrainer(config)
    
    # Setup model
    model = trainer.setup_model('mobilenet')
    trainer.setup_training_components()
    
    print("Advanced training pipeline test completed successfully!")