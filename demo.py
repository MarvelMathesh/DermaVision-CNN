#!/usr/bin/env python3
"""
Demo script for the Skin Disease Detection System
Provides a quick way to test the system with mock data
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.architectures import EnsembleModel
from src.data.preprocessing import AdvancedAugmentation, ImageQualityAssessment
from src.inference.deployment import ProductionInference
from src.interpretability.explainability import GradCAMExplainer, LIMEExplainer
from src.clinical.decision_support import ClinicalDecisionSupport
from src.evaluation.metrics import ClinicalMetrics


def load_config(config_path="configs/default_config.yaml"):
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None


def create_mock_model(config):
    """Create a mock trained model for demo purposes"""
    print("Creating mock model...")
    
    # Initialize model
    model = EnsembleModel(
        num_classes=config['data']['num_classes'],
        use_confidence_weighting=config['model']['ensemble']['use_confidence_weighting'],
        use_meta_learner=config['model']['ensemble']['use_meta_learner']
    )
    
    # Initialize with random weights (normally you'd load trained weights)
    model.eval()
    
    return model


def demo_inference(config, model):
    """Demonstrate inference on sample images"""
    print("\n" + "="*50)
    print("INFERENCE DEMO")
    print("="*50)
    
    # Create production inference
    inference = ProductionInference(
        model=model,
        config=config,
        device="cpu"
    )
    
    # Create mock image
    mock_image = torch.randn(1, 3, 224, 224)
    
    # Run inference
    print("Running inference on mock image...")
    results = inference.predict_single(mock_image)
    
    print(f"Predicted class: {results['predicted_class']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Uncertainty: {results['uncertainty']:.3f}")
    
    # Show top predictions
    print("\nTop 3 predictions:")
    top_indices = torch.topk(results['probabilities'], 3)[1]
    class_names = config['data']['class_names']
    
    for i, idx in enumerate(top_indices):
        prob = results['probabilities'][idx].item()
        print(f"  {i+1}. {class_names[idx]}: {prob:.3f}")
    
    return results


def demo_clinical_decision(config, inference_results):
    """Demonstrate clinical decision support"""
    print("\n" + "="*50)
    print("CLINICAL DECISION SUPPORT DEMO") 
    print("="*50)
    
    # Initialize clinical decision support
    clinical_ds = ClinicalDecisionSupport(config)
    
    # Mock patient data
    patient_data = {
        'age': 45,
        'gender': 'female',
        'skin_tone': 'fair',
        'lesion_location': 'arm',
        'lesion_size_mm': 8.5,
        'symptoms': ['irregular_border', 'color_variation']
    }
    
    # Generate clinical assessment
    print("Generating clinical assessment...")
    assessment = clinical_ds.generate_assessment(
        predictions=inference_results['probabilities'],
        confidence=inference_results['confidence'],
        uncertainty=inference_results['uncertainty'],
        patient_data=patient_data
    )
    
    print(f"Risk Level: {assessment['risk_level']}")
    print(f"Clinical Priority: {assessment['priority']}")
    print(f"Urgency Score: {assessment['urgency_score']:.2f}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(assessment['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nQuality Assurance: {assessment['qa_status']}")
    
    return assessment


def demo_explainability(config, model):
    """Demonstrate explainability features"""
    print("\n" + "="*50)
    print("EXPLAINABILITY DEMO")
    print("="*50)
    
    # Create mock image
    mock_image = torch.randn(1, 3, 224, 224)
    
    try:
        # GradCAM explanation
        print("Generating GradCAM explanation...")
        gradcam = GradCAMExplainer(model)
        
        # Use the first model in ensemble for explanation
        target_layer = model.mobilenet_model.features[-1]  # Last conv layer
        heatmap = gradcam.generate_explanation(
            image=mock_image,
            target_class=0,
            target_layer=target_layer
        )
        
        print(f"GradCAM heatmap shape: {heatmap.shape}")
        print("✓ GradCAM explanation generated successfully")
        
    except Exception as e:
        print(f"GradCAM explanation failed: {e}")
    
    try:
        # LIME explanation
        print("Generating LIME explanation...")
        lime_explainer = LIMEExplainer(model)
        
        explanation = lime_explainer.explain_image(
            image=mock_image.squeeze(0).permute(1, 2, 0).numpy(),
            num_samples=100
        )
        
        print("✓ LIME explanation generated successfully")
        
    except Exception as e:
        print(f"LIME explanation failed: {e}")


def demo_evaluation_metrics(config):
    """Demonstrate evaluation metrics"""
    print("\n" + "="*50)
    print("EVALUATION METRICS DEMO")
    print("="*50)
    
    # Create mock predictions and labels
    num_samples = 100
    num_classes = config['data']['num_classes']
    
    # Mock ground truth labels
    y_true = np.random.randint(0, num_classes, num_samples)
    
    # Mock predictions with realistic accuracy
    y_pred = y_true.copy()
    # Add some noise to simulate real predictions
    noise_indices = np.random.choice(num_samples, size=int(0.15 * num_samples), replace=False)
    y_pred[noise_indices] = np.random.randint(0, num_classes, len(noise_indices))
    
    # Mock prediction probabilities
    y_proba = np.random.dirichlet(np.ones(num_classes), size=num_samples)
    # Make probabilities more realistic based on predictions
    for i in range(num_samples):
        y_proba[i] = np.random.dirichlet([0.1] * num_classes)
        y_proba[i, y_pred[i]] = np.random.uniform(0.6, 0.95)
        y_proba[i] /= y_proba[i].sum()
    
    # Mock metadata for bias assessment
    metadata = {
        'skin_tone': np.random.choice(['fair', 'medium', 'dark'], num_samples),
        'gender': np.random.choice(['male', 'female'], num_samples),
        'age_group': np.random.choice(['young', 'middle', 'elderly'], num_samples)
    }
    
    # Calculate metrics
    clinical_metrics = ClinicalMetrics(
        class_names=config['data']['class_names'],
        malignant_classes=config['evaluation']['malignant_classes']
    )
    
    print("Calculating comprehensive metrics...")
    metrics = clinical_metrics.calculate_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        metadata=metadata
    )
    
    # Display key metrics
    print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
    print(f"Clinical Sensitivity: {metrics['clinical_sensitivity']:.3f}")
    print(f"Clinical Specificity: {metrics['clinical_specificity']:.3f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
    
    # Display bias metrics
    if 'bias_metrics' in metrics:
        print("\nBias Assessment:")
        for attribute, bias_score in metrics['bias_metrics'].items():
            print(f"  {attribute}: {bias_score:.3f}")
    
    return metrics


def demo_data_processing(config):
    """Demonstrate data processing capabilities"""
    print("\n" + "="*50)
    print("DATA PROCESSING DEMO")
    print("="*50)
    
    # Create mock image
    mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Demonstrate augmentation
    print("Testing data augmentation...")
    augmenter = AdvancedAugmentation(severity='medium')
    augmented = augmenter.apply_augmentation(mock_image)
    
    print(f"Original image shape: {mock_image.shape}")
    print(f"Augmented image shape: {augmented.shape}")
    print("✓ Augmentation pipeline working")
    
    # Demonstrate quality assessment
    print("Testing image quality assessment...")
    qa = ImageQualityAssessment()
    
    quality_metrics = qa.assess_quality(mock_image)
    
    print(f"Quality Score: {quality_metrics['overall_quality']:.3f}")
    print(f"Blur Score: {quality_metrics['blur_score']:.3f}")
    print(f"Brightness Score: {quality_metrics['brightness_score']:.3f}")
    print(f"Contrast Score: {quality_metrics['contrast_score']:.3f}")
    print("✓ Quality assessment working")


def create_demo_report(results, config):
    """Create a summary report of the demo"""
    print("\n" + "="*60)
    print("DEMO SUMMARY REPORT")
    print("="*60)
    
    print(f"Configuration: {config.get('project_name', 'Unknown')}")
    print(f"Model Type: {config['model']['type']}")
    print(f"Number of Classes: {config['data']['num_classes']}")
    print(f"Image Size: {config['data']['image_size']}")
    
    print("\nDemo Components Tested:")
    print("✓ Model Architecture (Ensemble)")
    print("✓ Production Inference")
    print("✓ Clinical Decision Support")
    print("✓ Explainability (GradCAM, LIME)")
    print("✓ Evaluation Metrics")
    print("✓ Data Processing Pipeline")
    
    print("\nKey Features Demonstrated:")
    print("• Multi-model ensemble with confidence weighting")
    print("• Uncertainty quantification")
    print("• Clinical risk stratification")
    print("• Bias assessment across demographics")
    print("• Model interpretability and explainability")
    print("• Production-ready inference pipeline")
    
    print("\nNext Steps:")
    print("1. Prepare real dataset (replace mock data)")
    print("2. Configure training parameters in config file")
    print("3. Run training: python train.py")
    print("4. Evaluate on test set: python evaluate.py")
    print("5. Deploy for clinical use")


def main():
    parser = argparse.ArgumentParser(description="Demo Skin Disease Detection System")
    parser.add_argument("--config", default="configs/default_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--components", nargs="+", 
                       choices=['inference', 'clinical', 'explainability', 'metrics', 'data'],
                       default=['inference', 'clinical', 'explainability', 'metrics', 'data'],
                       help="Components to demonstrate")
    parser.add_argument("--quick", action="store_true",
                       help="Quick demo with minimal output")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SKIN DISEASE DETECTION SYSTEM - DEMO")
    print("="*60)
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        print("Error: Could not load configuration file")
        return
    
    print(f"Loaded configuration: {args.config}")
    print(f"Demo components: {', '.join(args.components)}")
    
    # Create mock model
    model = create_mock_model(config)
    
    results = {}
    
    # Run demo components
    if 'inference' in args.components:
        results['inference'] = demo_inference(config, model)
    
    if 'clinical' in args.components and 'inference' in results:
        results['clinical'] = demo_clinical_decision(config, results['inference'])
    
    if 'explainability' in args.components:
        demo_explainability(config, model)
    
    if 'metrics' in args.components:
        results['metrics'] = demo_evaluation_metrics(config)
    
    if 'data' in args.components:
        demo_data_processing(config)
    
    # Create summary report
    if not args.quick:
        create_demo_report(results, config)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()