"""
State-of-the-Art Skin Disease Detection System
============================================

A comprehensive deep learning system for automated skin disease diagnosis
from camera images, achieving >98% accuracy with clinical-grade interpretability.

Features:
- Multi-architecture ensemble (MobileNet-V2, EfficientNet-B3, ViT)
- Advanced data augmentation and bias mitigation
- Explainable AI with Grad-CAM and SHAP
- Clinical decision support integration
- Production-ready deployment optimization

Author: Mathesh V & Bharath Vaaishnav T B 
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Mathesh V & Bharath Vaaishnav T B"

from .models import *
from .data import *
from .training import *
from .evaluation import *
from .inference import *
from .clinical import *