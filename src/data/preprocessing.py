"""
Data Processing and Augmentation Pipeline
=========================================

Advanced data preprocessing, augmentation, and bias mitigation techniques
for skin disease detection. Implements state-of-the-art augmentation
strategies and quality assessment protocols.
"""

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFilter
import os
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import imagehash
import warnings
warnings.filterwarnings('ignore')


class ImageQualityAssessment:
    """Image quality assessment and filtering for medical images."""
    
    @staticmethod
    def calculate_blur_score(image: np.ndarray) -> float:
        """Calculate Laplacian variance for blur detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    @staticmethod
    def detect_artifacts(image: np.ndarray) -> Dict[str, float]:
        """Detect common artifacts in dermoscopic images."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Hair detection using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        hair_score = np.mean(blackhat > 10)
        
        # Bubble detection using Hough circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        bubble_score = len(circles[0]) if circles is not None else 0
        
        # Ruler/scale detection using line detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                               minLineLength=50, maxLineGap=10)
        ruler_score = len(lines) if lines is not None else 0
        
        return {
            'hair_score': hair_score,
            'bubble_score': bubble_score,
            'ruler_score': ruler_score
        }
    
    @staticmethod
    def assess_color_constancy(image: np.ndarray) -> float:
        """Assess color constancy using gray-world assumption."""
        mean_rgb = np.mean(image.reshape(-1, 3), axis=0)
        gray_world_error = np.std(mean_rgb) / np.mean(mean_rgb)
        return gray_world_error
    
    @classmethod
    def filter_image(cls, image: np.ndarray, 
                    min_blur_score: float = 100,
                    max_hair_score: float = 0.1,
                    max_bubble_score: int = 5) -> bool:
        """Filter image based on quality criteria."""
        blur_score = cls.calculate_blur_score(image)
        artifacts = cls.detect_artifacts(image)
        
        return (blur_score >= min_blur_score and 
                artifacts['hair_score'] <= max_hair_score and
                artifacts['bubble_score'] <= max_bubble_score)


class AdvancedAugmentation:
    """Advanced augmentation pipeline for skin disease images."""
    
    def __init__(self, image_size: int = 224, severity: str = 'medium'):
        self.image_size = image_size
        self.severity = severity
        
        # Define augmentation parameters based on severity
        severity_params = {
            'light': {'prob': 0.3, 'distortion': 0.1, 'noise': 0.02},
            'medium': {'prob': 0.5, 'distortion': 0.2, 'noise': 0.05},
            'heavy': {'prob': 0.7, 'distortion': 0.3, 'noise': 0.1}
        }
        self.params = severity_params[severity]
    
    def get_training_transforms(self) -> A.Compose:
        """Get training augmentation pipeline."""
        return A.Compose([
            # Geometric transformations
            A.Resize(self.image_size + 32, self.image_size + 32),
            A.RandomCrop(self.image_size, self.image_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=45, p=self.params['prob']),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=15, 
                p=self.params['prob']
            ),
            
            # Elastic deformation for tissue-like distortions
            A.ElasticTransform(
                alpha=120, 
                sigma=6, 
                alpha_affine=3.6, 
                p=self.params['prob'] * 0.6
            ),
            
            # Color and lighting augmentations
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.15, 
                p=self.params['prob']
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=15, 
                val_shift_limit=10, 
                p=self.params['prob']
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=self.params['prob'] * 0.7),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Medical imaging specific augmentations
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1), 
                num_shadows_lower=1, 
                num_shadows_upper=2, 
                p=0.2
            ),
            
            # Cutout and gridmask for robustness
            A.CoarseDropout(
                max_holes=8, 
                max_height=self.image_size//8, 
                max_width=self.image_size//8, 
                min_holes=1,
                fill_value=0, 
                p=0.3
            ),
            
            # Normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_validation_transforms(self) -> A.Compose:
        """Get validation/test transforms."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_tta_transforms(self) -> List[A.Compose]:
        """Get test-time augmentation transforms."""
        tta_transforms = []
        
        # Original
        tta_transforms.append(self.get_validation_transforms())
        
        # Horizontal flip
        tta_transforms.append(A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
        
        # Vertical flip
        tta_transforms.append(A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
        
        # Rotations
        for angle in [90, 180, 270]:
            tta_transforms.append(A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Rotate(limit=(angle, angle), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
        
        # Brightness variations
        for brightness in [-0.1, 0.1]:
            tta_transforms.append(A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.RandomBrightnessContrast(brightness_limit=(brightness, brightness), 
                                         contrast_limit=0, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
        
        return tta_transforms


class SkinDiseaseDataset(Dataset):
    """
    Advanced dataset class for skin disease classification with
    demographic tracking and bias mitigation.
    """
    
    def __init__(self, 
                 data_df: pd.DataFrame,
                 image_dir: str,
                 transforms: Optional[Callable] = None,
                 include_metadata: bool = True,
                 apply_quality_filter: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_df: DataFrame with image paths, labels, and metadata
            image_dir: Directory containing images
            transforms: Augmentation transforms
            include_metadata: Whether to include demographic metadata
            apply_quality_filter: Whether to filter low-quality images
        """
        self.data_df = data_df.copy()
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.include_metadata = include_metadata
        
        # Apply quality filtering if requested
        if apply_quality_filter:
            self._filter_low_quality_images()
        
        # Create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.data_df['diagnosis'].unique())}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        # Calculate class weights for balanced sampling
        self.class_weights = self._calculate_class_weights()
        
        # Demographic analysis
        if 'skin_tone' in self.data_df.columns:
            self._analyze_demographic_distribution()
    
    def _filter_low_quality_images(self):
        """Filter out low-quality images."""
        quality_scores = []
        valid_indices = []
        
        for idx, row in self.data_df.iterrows():
            image_path = self.image_dir / row['image_path']
            if image_path.exists():
                try:
                    image = cv2.imread(str(image_path))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    if ImageQualityAssessment.filter_image(image):
                        valid_indices.append(idx)
                        quality_scores.append(ImageQualityAssessment.calculate_blur_score(image))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue
        
        self.data_df = self.data_df.loc[valid_indices].reset_index(drop=True)
        print(f"Filtered dataset: {len(self.data_df)} valid images")
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced sampling."""
        class_counts = self.data_df['diagnosis'].value_counts()
        total_samples = len(self.data_df)
        
        weights = []
        for label in self.label_to_idx.keys():
            weight = total_samples / (len(self.label_to_idx) * class_counts[label])
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def _analyze_demographic_distribution(self):
        """Analyze demographic distribution for bias assessment."""
        print("\nDemographic Distribution Analysis:")
        print("=" * 40)
        
        # Skin tone distribution
        if 'skin_tone' in self.data_df.columns:
            skin_tone_dist = self.data_df['skin_tone'].value_counts(normalize=True)
            print(f"Skin tone distribution:\n{skin_tone_dist}")
        
        # Age distribution
        if 'age' in self.data_df.columns:
            age_stats = self.data_df['age'].describe()
            print(f"\nAge distribution:\n{age_stats}")
        
        # Gender distribution
        if 'gender' in self.data_df.columns:
            gender_dist = self.data_df['gender'].value_counts(normalize=True)
            print(f"\nGender distribution:\n{gender_dist}")
        
        # Anatomical location distribution
        if 'anatomical_location' in self.data_df.columns:
            location_dist = self.data_df['anatomical_location'].value_counts()
            print(f"\nAnatomical location distribution:\n{location_dist}")
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for weighted sampling."""
        weights = torch.zeros(len(self.data_df))
        for idx, row in self.data_df.iterrows():
            label_idx = self.label_to_idx[row['diagnosis']]
            weights[idx] = self.class_weights[label_idx]
        return weights
    
    def __len__(self) -> int:
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        row = self.data_df.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row['image_path']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Get label
        label = self.label_to_idx[row['diagnosis']]
        
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'image_path': str(image_path)
        }
        
        # Add metadata if requested
        if self.include_metadata:
            metadata = {}
            for col in ['age', 'gender', 'skin_tone', 'anatomical_location']:
                if col in row:
                    metadata[col] = row[col]
            sample['metadata'] = metadata
        
        return sample


class DataManager:
    """
    Data management class for handling multiple datasets and
    implementing advanced sampling strategies.
    """
    
    def __init__(self, 
                 config: Dict,
                 augmentation_severity: str = 'medium'):
        """
        Initialize data manager.
        
        Args:
            config: Configuration dictionary with dataset paths
            augmentation_severity: Severity of augmentation ('light', 'medium', 'heavy')
        """
        self.config = config
        self.augmentation = AdvancedAugmentation(
            image_size=config.get('image_size', 224),
            severity=augmentation_severity
        )
        
        # Initialize datasets
        self.datasets = {}
        self.dataloaders = {}
    
    def load_datasets(self) -> Dict[str, SkinDiseaseDataset]:
        """Load and prepare all datasets."""
        for dataset_name, dataset_config in self.config['datasets'].items():
            print(f"Loading {dataset_name} dataset...")
            
            # Load metadata
            data_df = pd.read_csv(dataset_config['metadata_path'])
            
            # Create dataset splits
            train_df, val_df, test_df = self._create_stratified_splits(
                data_df, 
                test_size=0.2, 
                val_size=0.2
            )
            
            # Create datasets
            self.datasets[f'{dataset_name}_train'] = SkinDiseaseDataset(
                train_df,
                dataset_config['image_dir'],
                self.augmentation.get_training_transforms(),
                include_metadata=True
            )
            
            self.datasets[f'{dataset_name}_val'] = SkinDiseaseDataset(
                val_df,
                dataset_config['image_dir'],
                self.augmentation.get_validation_transforms(),
                include_metadata=True
            )
            
            self.datasets[f'{dataset_name}_test'] = SkinDiseaseDataset(
                test_df,
                dataset_config['image_dir'],
                self.augmentation.get_validation_transforms(),
                include_metadata=True
            )
        
        return self.datasets
    
    def _create_stratified_splits(self, 
                                 data_df: pd.DataFrame,
                                 test_size: float = 0.2,
                                 val_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create stratified train/val/test splits."""
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            data_df,
            test_size=test_size,
            stratify=data_df['diagnosis'],
            random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['diagnosis'],
            random_state=42
        )
        
        return train_df, val_df, test_df
    
    def create_dataloaders(self, 
                          batch_size: int = 32,
                          num_workers: int = 4,
                          use_weighted_sampling: bool = True) -> Dict[str, DataLoader]:
        """Create data loaders with advanced sampling strategies."""
        
        for dataset_name, dataset in self.datasets.items():
            if 'train' in dataset_name and use_weighted_sampling:
                # Weighted sampling for training
                sample_weights = dataset.get_sample_weights()
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle = False
            else:
                sampler = None
                shuffle = 'train' in dataset_name
            
            self.dataloaders[dataset_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last='train' in dataset_name
            )
        
        return self.dataloaders
    
    def get_class_distribution(self, dataset_name: str) -> Dict:
        """Get class distribution for a specific dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset = self.datasets[dataset_name]
        class_counts = dataset.data_df['diagnosis'].value_counts()
        
        return {
            'counts': class_counts.to_dict(),
            'percentages': (class_counts / len(dataset.data_df) * 100).to_dict(),
            'total_samples': len(dataset.data_df)
        }
    
    def analyze_bias_metrics(self) -> Dict:
        """Analyze potential bias in the dataset."""
        bias_metrics = {}
        
        for dataset_name, dataset in self.datasets.items():
            if 'train' in dataset_name:
                df = dataset.data_df
                
                # Skin tone bias
                if 'skin_tone' in df.columns:
                    skin_tone_by_diagnosis = pd.crosstab(df['diagnosis'], df['skin_tone'], normalize='index')
                    bias_metrics[f'{dataset_name}_skin_tone_bias'] = skin_tone_by_diagnosis.to_dict()
                
                # Age bias
                if 'age' in df.columns:
                    age_by_diagnosis = df.groupby('diagnosis')['age'].describe()
                    bias_metrics[f'{dataset_name}_age_bias'] = age_by_diagnosis.to_dict()
                
                # Gender bias
                if 'gender' in df.columns:
                    gender_by_diagnosis = pd.crosstab(df['diagnosis'], df['gender'], normalize='index')
                    bias_metrics[f'{dataset_name}_gender_bias'] = gender_by_diagnosis.to_dict()
        
        return bias_metrics


# Utility functions for data preparation
def create_mock_dataset_metadata(num_samples: int = 1000, 
                                num_classes: int = 8) -> pd.DataFrame:
    """Create mock dataset metadata for testing."""
    np.random.seed(42)
    
    # Disease classes (simplified ISIC classes)
    disease_classes = [
        'melanoma', 'nevus', 'basal_cell_carcinoma', 'actinic_keratosis',
        'benign_keratosis', 'dermatofibroma', 'vascular_lesion', 'squamous_cell_carcinoma'
    ][:num_classes]
    
    # Generate mock data
    data = {
        'image_path': [f'image_{i:04d}.jpg' for i in range(num_samples)],
        'diagnosis': np.random.choice(disease_classes, num_samples, 
                                    p=[0.2, 0.3, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05][:num_classes]),
        'age': np.random.normal(55, 15, num_samples).astype(int),
        'gender': np.random.choice(['male', 'female'], num_samples, p=[0.45, 0.55]),
        'skin_tone': np.random.choice(['light', 'medium', 'dark'], num_samples, p=[0.6, 0.3, 0.1]),
        'anatomical_location': np.random.choice([
            'face', 'scalp', 'neck', 'trunk', 'upper_extremity', 'lower_extremity'
        ], num_samples),
        'image_quality_score': np.random.uniform(100, 500, num_samples)
    }
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test data pipeline
    print("Testing Skin Disease Data Pipeline...")
    
    # Create mock configuration
    config = {
        'image_size': 224,
        'datasets': {
            'mock_dataset': {
                'metadata_path': 'mock_metadata.csv',
                'image_dir': '/path/to/images'
            }
        }
    }
    
    # Create mock metadata
    mock_df = create_mock_dataset_metadata(1000, 8)
    print(f"Created mock dataset with {len(mock_df)} samples")
    print(f"Classes: {mock_df['diagnosis'].unique()}")
    print(f"Class distribution:\n{mock_df['diagnosis'].value_counts()}")
    
    # Test augmentation pipeline
    aug = AdvancedAugmentation(image_size=224, severity='medium')
    print("\nAugmentation pipeline created successfully")
    
    # Test image quality assessment
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    qa = ImageQualityAssessment()
    blur_score = qa.calculate_blur_score(dummy_image)
    artifacts = qa.detect_artifacts(dummy_image)
    
    print(f"\nImage Quality Assessment:")
    print(f"Blur score: {blur_score:.2f}")
    print(f"Artifacts: {artifacts}")
    
    print("\nData pipeline testing completed successfully!")