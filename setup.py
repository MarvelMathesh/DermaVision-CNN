#!/usr/bin/env python3
"""
Setup script for the Skin Disease Detection System
Installs dependencies and prepares the environment
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, check=True):
    """Run a command and optionally check for errors"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result


def install_dependencies():
    """Install required Python packages"""
    print("Installing Python dependencies...")
    
    # Core ML dependencies
    packages = [
        "torch>=1.12.0",
        "torchvision>=0.13.0", 
        "timm>=0.6.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "scipy>=1.8.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.9.0",
        "opencv-python>=4.6.0",
        "albumentations>=1.2.0",
        "Pillow>=9.0.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "tensorboard>=2.9.0",
        "wandb>=0.12.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
        "lime>=0.2.0",
        "shap>=0.41.0",
        "fastapi>=0.78.0",
        "uvicorn>=0.18.0",
        "requests>=2.28.0",
        "aiofiles>=0.8.0",
        "python-multipart>=0.0.5",
        "optuna>=3.0.0",
        "gradio>=3.1.0",
        "streamlit>=1.11.0"
    ]
    
    for package in packages:
        run_command(f"pip install {package}")


def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "data/mock",
        "checkpoints",
        "results/models",
        "results/evaluations", 
        "results/explanations",
        "logs/tensorboard",
        "models/optimized",
        "cache",
        "notebooks",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def download_mock_data():
    """Download or create mock data for testing"""
    print("Setting up mock data...")
    
    mock_data_script = """
import numpy as np
import pandas as pd
from PIL import Image
import os
from pathlib import Path

# Create mock metadata
np.random.seed(42)
n_samples = 1000

class_names = [
    'melanoma', 'nevus', 'basal_cell_carcinoma', 'actinic_keratosis',
    'benign_keratosis', 'dermatofibroma', 'vascular_lesion', 'squamous_cell_carcinoma'
]

demographics = ['fair', 'medium', 'dark']
genders = ['male', 'female', 'other']
age_groups = ['young', 'middle', 'elderly']

# Generate mock metadata
metadata = {
    'image_id': [f'img_{i:06d}' for i in range(n_samples)],
    'diagnosis': np.random.choice(class_names, n_samples),
    'skin_tone': np.random.choice(demographics, n_samples),
    'gender': np.random.choice(genders, n_samples),
    'age_group': np.random.choice(age_groups, n_samples),
    'confidence': np.random.uniform(0.7, 1.0, n_samples),
    'dataset': ['mock'] * n_samples
}

df = pd.DataFrame(metadata)

# Create mock images directory
mock_dir = Path('data/mock')
mock_dir.mkdir(parents=True, exist_ok=True)

# Save metadata
df.to_csv(mock_dir / 'metadata.csv', index=False)

# Create mock images (small colored squares)
print("Creating mock images...")
for i, row in df.iterrows():
    # Create a colored image based on diagnosis
    color_map = {
        'melanoma': (139, 69, 19),      # Brown
        'nevus': (205, 133, 63),        # Peru
        'basal_cell_carcinoma': (255, 182, 193),  # Light pink
        'actinic_keratosis': (255, 160, 122),     # Light salmon
        'benign_keratosis': (255, 218, 185),      # Peach puff
        'dermatofibroma': (160, 82, 45),          # Saddle brown
        'vascular_lesion': (255, 99, 71),         # Tomato
        'squamous_cell_carcinoma': (255, 20, 147) # Deep pink
    }
    
    color = color_map.get(row['diagnosis'], (128, 128, 128))
    
    # Add some noise to make it more realistic
    img = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
    img[:, :, 0] = np.clip(img[:, :, 0] + color[0], 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] + color[1], 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] + color[2], 0, 255)
    
    # Add some circular patterns to simulate lesions
    center = (112, 112)
    radius = np.random.randint(20, 60)
    y, x = np.ogrid[:224, :224]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img[mask] = [min(255, c + 30) for c in color]
    
    # Save image
    image = Image.fromarray(img)
    image.save(mock_dir / f'{row["image_id"]}.jpg')
    
    if (i + 1) % 100 == 0:
        print(f"Created {i + 1}/{len(df)} mock images")

print(f"Mock data created successfully!")
print(f"Images: {len(df)} in {mock_dir}")
print(f"Metadata: {mock_dir}/metadata.csv")
"""
    
    # Write and execute mock data script
    with open("create_mock_data.py", "w") as f:
        f.write(mock_data_script)
    
    run_command("python create_mock_data.py")
    os.remove("create_mock_data.py")


def setup_git_hooks():
    """Setup git hooks for code quality"""
    print("Setting up git hooks...")
    
    # Check if git is initialized
    if not Path(".git").exists():
        run_command("git init")
        print("Initialized git repository")
    
    # Create pre-commit hook
    pre_commit_hook = """#!/bin/sh
# Pre-commit hook for code quality

echo "Running pre-commit checks..."

# Check Python syntax
python -m py_compile src/**/*.py
if [ $? -ne 0 ]; then
    echo "Python syntax errors found. Commit aborted."
    exit 1
fi

echo "Pre-commit checks passed!"
"""
    
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        with open(hooks_dir / "pre-commit", "w") as f:
            f.write(pre_commit_hook)
        os.chmod(hooks_dir / "pre-commit", 0o755)
        print("Git pre-commit hook installed")


def create_environment_file():
    """Create environment file with dependencies"""
    print("Creating environment.yml file...")
    
    env_content = """name: skin-disease-detection
channels:
  - pytorch
  - conda-forge
  - defaults

dependencies:
  - python=3.9
  - pytorch>=1.12.0
  - torchvision>=0.13.0
  - cudatoolkit=11.6
  - pip
  - pip:
    - timm>=0.6.0
    - transformers>=4.20.0
    - albumentations>=1.2.0
    - opencv-python>=4.6.0
    - scikit-learn>=1.1.0
    - pandas>=1.4.0
    - matplotlib>=3.5.0
    - seaborn>=0.11.0
    - plotly>=5.9.0
    - tensorboard>=2.9.0
    - wandb>=0.12.0
    - onnx>=1.12.0
    - onnxruntime>=1.12.0
    - lime>=0.2.0
    - shap>=0.41.0
    - fastapi>=0.78.0
    - uvicorn>=0.18.0
    - gradio>=3.1.0
    - streamlit>=1.11.0
    - optuna>=3.0.0
    - pyyaml>=6.0
    - tqdm>=4.64.0
    - requests>=2.28.0
"""
    
    with open("environment.yml", "w") as f:
        f.write(env_content)


def verify_installation():
    """Verify that all components are installed correctly"""
    print("\nVerifying installation...")
    
    # Test imports
    test_imports = [
        "torch",
        "torchvision", 
        "timm",
        "transformers",
        "albumentations",
        "cv2",
        "sklearn",
        "pandas",
        "numpy",
        "matplotlib",
        "tensorboard",
        "onnx",
        "lime",
        "shap"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies manually")
        return False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU training only)")
    except:
        print("⚠ Could not check CUDA availability")
    
    print("\n✓ Installation verification complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Setup Skin Disease Detection System")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--skip-mock-data", action="store_true",
                       help="Skip mock data creation")
    parser.add_argument("--minimal", action="store_true",
                       help="Minimal setup (directories only)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Skin Disease Detection System Setup")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    if not args.minimal:
        # Install dependencies
        if not args.skip_deps:
            install_dependencies()
        
        # Create mock data
        if not args.skip_mock_data:
            download_mock_data()
        
        # Setup development tools
        setup_git_hooks()
        create_environment_file()
        
        # Verify installation
        if not args.skip_deps:
            verify_installation()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review and modify configs/default_config.yaml")
    print("2. Run: python train.py --config configs/default_config.yaml")
    print("3. For quick demo: python demo.py")
    print("4. For evaluation: python evaluate.py")
    print("\nFor help: python train.py --help")
    print("Documentation: README.md")


if __name__ == "__main__":
    main()