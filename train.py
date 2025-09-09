"""
Main Training Script for Skin Disease Detection
===============================================

Orchestrates the complete training pipeline with all advanced features.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import wandb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import DataManager, create_mock_dataset_metadata
from src.models.architectures import create_model
from src.training.trainer import AdvancedTrainer
from src.training.losses import create_loss_function
from src.evaluation.metrics import MetricsCalculator
from src.interpretability.explainability import InterpretabilityReport


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: Dict):
    """Create necessary directories."""
    directories = [
        config.get('checkpoint_dir', 'checkpoints'),
        config.get('log_dir', 'logs'),
        config.get('results_dir', 'results'),
        'data/processed'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def initialize_wandb(config: Dict):
    """Initialize Weights & Biases logging."""
    if config.get('use_wandb', False):
        wandb.init(
            project=config.get('project_name', 'skin-disease-detection'),
            name=config.get('experiment_name', 'experiment'),
            config=config,
            tags=config.get('tags', [])
        )
        return wandb
    return None


def prepare_datasets(config: Dict) -> DataManager:
    """Prepare datasets and data loaders."""
    print("Preparing datasets...")
    
    # For demonstration, create mock datasets
    # In practice, replace with real dataset paths
    if config.get('use_mock_data', True):
        print("Creating mock dataset for demonstration...")
        
        # Create mock metadata files
        train_df = create_mock_dataset_metadata(5000, config['num_classes'])
        val_df = create_mock_dataset_metadata(1000, config['num_classes'])
        test_df = create_mock_dataset_metadata(1000, config['num_classes'])
        
        # Save mock metadata
        os.makedirs('data/processed', exist_ok=True)
        train_df.to_csv('data/processed/train_metadata.csv', index=False)
        val_df.to_csv('data/processed/val_metadata.csv', index=False)
        test_df.to_csv('data/processed/test_metadata.csv', index=False)
        
        # Update config with mock paths
        config['datasets'] = {
            'isic': {
                'metadata_path': 'data/processed/train_metadata.csv',
                'image_dir': 'data/processed/images'  # Would contain actual images
            }
        }
    
    # Initialize data manager
    data_manager = DataManager(
        config, 
        augmentation_severity=config.get('augmentation_severity', 'medium')
    )
    
    # Load datasets
    datasets = data_manager.load_datasets()
    
    # Create data loaders
    dataloaders = data_manager.create_dataloaders(
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
        use_weighted_sampling=config.get('use_weighted_sampling', True)
    )
    
    print(f"Datasets prepared: {list(datasets.keys())}")
    
    return data_manager


def setup_model_and_training(config: Dict, device: torch.device) -> AdvancedTrainer:
    """Setup model and training components."""
    print("Setting up model and training components...")
    
    # Initialize trainer
    trainer = AdvancedTrainer(config, device)
    
    # Setup model
    model = trainer.setup_model(
        model_type=config.get('model_type', 'ensemble'),
        dropout_rate=config.get('dropout_rate', 0.3)
    )
    
    # Setup training components
    trainer.setup_training_components()
    
    print(f"Model: {config.get('model_type', 'ensemble')}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return trainer


def train_model(trainer: AdvancedTrainer, 
               data_manager: DataManager,
               config: Dict,
               logger=None) -> Dict:
    """Execute training pipeline."""
    print("Starting training...")
    
    # Get data loaders
    train_loader = None
    val_loader = None
    
    # Find training and validation loaders
    for name, loader in data_manager.dataloaders.items():
        if 'train' in name:
            train_loader = loader
        elif 'val' in name:
            val_loader = loader
    
    if train_loader is None or val_loader is None:
        raise ValueError("Training or validation loader not found")
    
    # Train model
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.get('num_epochs', 200)
    )
    
    return training_history


def evaluate_model(trainer: AdvancedTrainer,
                  data_manager: DataManager,
                  config: Dict) -> Dict:
    """Evaluate trained model."""
    print("Evaluating model...")
    
    # Load best model
    best_model_path = Path(config.get('checkpoint_dir', 'checkpoints')) / 'best_model.pth'
    if best_model_path.exists():
        trainer.load_checkpoint(str(best_model_path))
        print("Loaded best model for evaluation")
    
    # Get test loader
    test_loader = None
    for name, loader in data_manager.dataloaders.items():
        if 'test' in name:
            test_loader = loader
            break
    
    if test_loader is None:
        print("No test loader found, using validation loader")
        for name, loader in data_manager.dataloaders.items():
            if 'val' in name:
                test_loader = loader
                break
    
    # Standard evaluation
    test_metrics = trainer.validate_epoch(test_loader, epoch=0)
    
    # Test-time augmentation evaluation
    if config.get('use_tta', True):
        tta_transforms = data_manager.augmentation.get_tta_transforms()
        tta_metrics = trainer.evaluate_with_tta(test_loader, tta_transforms)
        test_metrics.update(tta_metrics)
    
    return test_metrics


def generate_interpretability_report(trainer: AdvancedTrainer,
                                   data_manager: DataManager,
                                   config: Dict):
    """Generate interpretability reports for sample predictions."""
    print("Generating interpretability reports...")
    
    # Get class names
    class_names = [
        'melanoma', 'nevus', 'basal_cell_carcinoma', 'actinic_keratosis',
        'benign_keratosis', 'dermatofibroma', 'vascular_lesion', 'squamous_cell_carcinoma'
    ][:config['num_classes']]
    
    # Initialize report generator
    report_generator = InterpretabilityReport(
        model=trainer.model,
        class_names=class_names,
        device=trainer.device
    )
    
    # Get a few test samples
    test_loader = None
    for name, loader in data_manager.dataloaders.items():
        if 'test' in name or 'val' in name:
            test_loader = loader
            break
    
    if test_loader is None:
        print("No test data available for interpretability analysis")
        return
    
    # Generate reports for first few samples
    trainer.model.eval()
    sample_count = 0
    max_samples = config.get('interpretability_samples', 5)
    
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= max_samples:
                break
            
            images = batch['image'].to(trainer.device)
            labels = batch['label'].to(trainer.device)
            
            # Get predictions
            outputs = trainer.model(images)
            predictions = torch.argmax(outputs, dim=1)
            confidences = torch.softmax(outputs, dim=1)
            
            # Process each image in batch
            for i in range(min(images.size(0), max_samples - sample_count)):
                # Convert image back to numpy
                image_np = images[i].cpu().permute(1, 2, 0).numpy()
                image_np = ((image_np + 1) * 127.5).astype(np.uint8)  # Denormalize
                
                # Generate report
                report = report_generator.generate_comprehensive_report(
                    image=image_np,
                    prediction=predictions[i],
                    confidence=confidences[i, predictions[i]].item(),
                    metadata={'true_label': class_names[labels[i].item()]}
                )
                
                # Save report
                output_path = f"results/interpretability_sample_{sample_count}"
                report_generator.save_report(report, output_path)
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
    
    print(f"Generated {sample_count} interpretability reports")


def save_final_results(training_history: Dict,
                      test_metrics: Dict,
                      config: Dict):
    """Save final training results and metrics."""
    print("Saving final results...")
    
    results = {
        'config': config,
        'final_metrics': test_metrics,
        'training_history': {
            'train_losses': training_history['train_losses'],
            'val_losses': training_history['val_losses']
        },
        'best_metrics': {
            'best_val_accuracy': max([m.get('val/accuracy', 0) for m in training_history['val_metrics']]),
            'best_val_balanced_accuracy': max([m.get('val/balanced_accuracy', 0) for m in training_history['val_metrics']]),
            'best_val_f1': max([m.get('val/f1_weighted', 0) for m in training_history['val_metrics']])
        }
    }
    
    # Save results
    results_path = Path(config.get('results_dir', 'results'))
    results_path.mkdir(exist_ok=True)
    
    import json
    with open(results_path / 'final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save training history as CSV
    train_metrics_df = pd.DataFrame(training_history['train_metrics'])
    val_metrics_df = pd.DataFrame(training_history['val_metrics'])
    
    train_metrics_df.to_csv(results_path / 'train_metrics.csv', index=False)
    val_metrics_df.to_csv(results_path / 'val_metrics.csv', index=False)
    
    print(f"Results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Best Validation Accuracy: {results['best_metrics']['best_val_accuracy']:.4f}")
    print(f"Best Validation Balanced Accuracy: {results['best_metrics']['best_val_balanced_accuracy']:.4f}")
    print(f"Best Validation F1-Score: {results['best_metrics']['best_val_f1']:.4f}")
    
    if 'test/accuracy' in test_metrics:
        print(f"Final Test Accuracy: {test_metrics['test/accuracy']:.4f}")
    if 'test_tta/accuracy' in test_metrics:
        print(f"Test Accuracy (TTA): {test_metrics['test_tta/accuracy']:.4f}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Skin Disease Detection Training')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                       help='Only evaluate model, do not train')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Using default configuration.")
        config = {
            'num_classes': 8,
            'model_type': 'ensemble',
            'batch_size': 16,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'use_mock_data': True,
            'use_wandb': False,
            'augmentation_severity': 'medium',
            'use_tta': True,
            'interpretability_samples': 3
        }
    
    print("Starting Skin Disease Detection Training Pipeline")
    print(f"Configuration: {args.config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_directories(config)
    logger = initialize_wandb(config)
    
    try:
        # Prepare data
        data_manager = prepare_datasets(config)
        
        # Setup model and trainer
        trainer = setup_model_and_training(config, device)
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
            print(f"Resumed from checkpoint: {args.resume}")
        
        if not args.eval_only:
            # Train model
            training_history = trainer.train(
                train_loader=data_manager.dataloaders[list(data_manager.dataloaders.keys())[0]],
                val_loader=data_manager.dataloaders[list(data_manager.dataloaders.keys())[1]],
                num_epochs=config.get('num_epochs', 100)
            )
        else:
            training_history = {'train_losses': [], 'val_losses': [], 'train_metrics': [], 'val_metrics': []}
        
        # Evaluate model
        test_metrics = evaluate_model(trainer, data_manager, config)
        
        # Generate interpretability reports
        if config.get('generate_interpretability_reports', True):
            generate_interpretability_report(trainer, data_manager, config)
        
        # Save results
        save_final_results(training_history, test_metrics, config)
        
        print("\nTraining pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if logger:
            logger.finish()


if __name__ == "__main__":
    main()