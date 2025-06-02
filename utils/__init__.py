# utils/__init__.py
"""
Utility modules for the Diabetes Prediction Flask App
"""

from .model_utils import ModelPredictor

__all__ = ['ModelPredictor']

# models/__init__.py
"""
Machine Learning Models Directory
"""

# This directory contains:
# - best_random_forest_model.pkl
# - best_logistic_model_with_smote.pkl
# - standard_scaler.pkl
# - logisticRegressionstandard_scaler.pkl
# - label_encoders.pkl
# - model_info.json

__version__ = "1.0.0"

# setup.py - Setup script for the application
#!/usr/bin/env python3
"""
Setup script for Diabetes Prediction Flask Application
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'models',
        'static/css',
        'static/js',
        'templates',
        'utils',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files"""
    init_files = ['utils/__init__.py', 'models/__init__.py']
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')
            print(f"✅ Created: {init_file}")

def check_model_files():
    """Check if model files are present"""
    model_files = [
        'models/best_random_forest_model.pkl',
        'models/best_logistic_model_with_smote.pkl',
        'models/standard_scaler.pkl',
        'models/logisticRegressionstandard_scaler.pkl'
    ]
    
    missing_files = []
    for model_file in model_files:
        if not os.path.exists(model_file):
            missing_files.append(model_file)
    
    if missing_files:
        print("⚠️  Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📋 Please copy your trained model files to the models/ directory")
    else:
        print("✅ All model files found!")

def create_model_info():
    """Create model info JSON file"""
    model_info = {
        "models": {
            "random_forest": {
                "name": "Random Forest Classifier",
                "file": "best_random_forest_model.pkl",
                "scaler": "standard_scaler.pkl",
                "description": "Ensemble method using multiple decision trees"
            },
            "logistic_regression": {
                "name": "Logistic Regression with SMOTE",
                "file": "best_logistic_model_with_smote.pkl",
                "scaler": "logisticRegressionstandard_scaler.pkl",
                "description": "Linear classifier with balanced sampling"
            }
        },
        "features": [
            "AGE", "Gender", "Urea", "HbA1c", "Chol", "BMI", "VLDL"
        ],
        "classes": ["N", "P", "Y"],
        "version": "1.0.0"
    }
    
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    print("✅ Created model_info.json")

def install_dependencies():
    """Install Python dependencies"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please run manually:")
        print("   pip install -r requirements.txt")

def create_env_file():
    """Create .env file for environment variables"""
    env_content = """# Flask Environment Configuration
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=your-secret-key-here

# Model Configuration
MODEL_DIR=models
"""
    
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")

def main():
    """Main setup function"""
    print("🚀 Setting up Diabetes Prediction Flask Application")
    print("=" * 50)
    
    create_directory_structure()
    create_init_files()
    create_model_info()
    create_env_file()
    check_model_files()
    
    print("\n📦 Installing dependencies...")
    install_dependencies()
    
    print("\n🎉 Setup complete!")
    print("\n📋 Next steps:")
    print("1. Copy your model files to the models/ directory")
    print("2. Run the application: python app.py")
    print("3. Open http://localhost:5000 in your browser")
    
    print("\n📁 Project structure created:")
    print("diabetes_prediction_app/")
    print("├── app.py")
    print("├── config.py") 
    print("├── requirements.txt")
    print("├── utils/")
    print("│   ├── __init__.py")
    print("│   └── model_utils.py")
    print("├── models/")
    print("│   ├── __init__.py")
    print("│   └── model_info.json")
    print("├── static/")
    print("│   ├── css/style.css")
    print("│   └── js/main.js")
    print("└── templates/")
    print("    ├── base.html")
    print("    ├── index.html")
    print("    └── results.html")

if __name__ == "__main__":
    main()