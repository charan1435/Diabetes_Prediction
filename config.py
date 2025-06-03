import os
from datetime import timedelta

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'diabetes-prediction-secret-key-2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Model settings
    MODEL_DIR = 'models'
    MAX_PREDICTION_BATCH_SIZE = 1000
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Session settings
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Feature validation ranges (updated with all features)
    FEATURE_RANGES = {
        'age': (0, 120),
        'urea': (0, 200),
        'cr': (0, 10),        # Creatinine
        'hba1c': (0, 20),
        'hdl': (0, 150),      # HDL Cholesterol
        'ldl': (0, 300),      # LDL Cholesterol
        'chol': (0, 500),     # Total Cholesterol
        'tg': (0, 1000),      # Triglycerides
        'bmi': (10, 60),
        'vldl': (0, 100)
    }
    
    # Model information
    MODEL_METADATA = {
        'random_forest': {
            'name': 'Random Forest Classifier',
            'description': 'Ensemble method using multiple decision trees',
            'best_for': 'Complex pattern recognition and feature interactions'
        },
        'logistic_regression': {
            'name': 'Logistic Regression with SMOTE',
            'description': 'Linear classifier with balanced sampling',
            'best_for': 'Interpretable predictions and handling imbalanced data'
        }
    }
    
    # Class information
    CLASS_INFO = {
        'N': {
            'name': 'Normal',
            'description': 'No diabetes detected',
            'color': 'success',
            'recommendation': 'Maintain healthy lifestyle'
        },
        'P': {
            'name': 'Pre-diabetes',
            'description': 'At risk for diabetes',
            'color': 'warning',
            'recommendation': 'Consider lifestyle changes and regular monitoring'
        },
        'Y': {
            'name': 'Diabetes',
            'description': 'Diabetes detected',
            'color': 'danger',
            'recommendation': 'Consult healthcare provider for proper treatment'
        }
    }
    
    # Feature information for documentation
    FEATURE_INFO = {
        'age': {
            'name': 'Age',
            'unit': 'years',
            'normal_range': '0-120',
            'description': 'Patient age in years'
        },
        'gender': {
            'name': 'Gender',
            'unit': '',
            'normal_range': 'Male/Female',
            'description': 'Patient gender'
        },
        'urea': {
            'name': 'Urea',
            'unit': 'mg/dL',
            'normal_range': '7-45',
            'description': 'Blood urea nitrogen level'
        },
        'cr': {
            'name': 'Creatinine',
            'unit': 'mg/dL',
            'normal_range': '0.6-1.2',
            'description': 'Serum creatinine level'
        },
        'hba1c': {
            'name': 'HbA1c',
            'unit': '%',
            'normal_range': '<5.7',
            'description': 'Glycated hemoglobin'
        },
        'hdl': {
            'name': 'HDL Cholesterol',
            'unit': 'mg/dL',
            'normal_range': 'M≥40, F≥50',
            'description': 'High-density lipoprotein cholesterol'
        },
        'ldl': {
            'name': 'LDL Cholesterol',
            'unit': 'mg/dL',
            'normal_range': '<100',
            'description': 'Low-density lipoprotein cholesterol'
        },
        'chol': {
            'name': 'Total Cholesterol',
            'unit': 'mg/dL',
            'normal_range': '<200',
            'description': 'Total cholesterol level'
        },
        'tg': {
            'name': 'Triglycerides',
            'unit': 'mg/dL',
            'normal_range': '<150',
            'description': 'Triglycerides level'
        },
        'bmi': {
            'name': 'BMI',
            'unit': 'kg/m²',
            'normal_range': '18.5-24.9',
            'description': 'Body Mass Index'
        },
        'vldl': {
            'name': 'VLDL',
            'unit': 'mg/dL',
            'normal_range': '<30',
            'description': 'Very low-density lipoprotein cholesterol'
        }
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}