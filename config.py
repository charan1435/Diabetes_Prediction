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
    
    # Feature validation ranges
    FEATURE_RANGES = {
        'age': (0, 120),
        'urea': (0, 200),
        'hba1c': (0, 20),
        'chol': (0, 500),
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