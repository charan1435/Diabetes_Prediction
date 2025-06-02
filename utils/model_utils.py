import pickle
import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime

class ModelPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = ['AGE', 'Gender', 'Urea', 'HbA1c', 'Chol', 'BMI', 'VLDL']
        self.class_names = ['N', 'P', 'Y']  # Normal, Pre-diabetes, Diabetes (adjust based on your data)
        
        self._load_models()
        self._load_scalers()
        self._load_encoders()
    
    def _load_models(self):
        """Load trained models"""
        model_paths = {
            'random_forest': 'models/best_random_forest_model.pkl',
            'logistic_regression': 'models/best_logistic_model_with_smote.pkl'
        }
        
        for model_name, path in model_paths.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    print(f"✅ Loaded {model_name} model")
                else:
                    print(f"⚠️ Model file not found: {path}")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
    
    def _load_scalers(self):
        """Load feature scalers"""
        scaler_paths = {
            'random_forest': 'models/standard_scaler.pkl',
            'logistic_regression': 'models/logisticRegressionstandard_scaler.pkl'
        }
        
        for model_name, path in scaler_paths.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
                    print(f"✅ Loaded {model_name} scaler")
                else:
                    print(f"⚠️ Scaler file not found: {path}")
            except Exception as e:
                print(f"❌ Error loading {model_name} scaler: {e}")
    
    def _load_encoders(self):
        """Load or create label encoders"""
        encoder_path = 'models/label_encoders.pkl'
        
        try:
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.encoders = pickle.load(f)
                print("✅ Loaded label encoders")
            else:
                # Create default encoders if not found
                self._create_default_encoders()
                print("✅ Created default label encoders")
        except Exception as e:
            print(f"❌ Error loading encoders: {e}")
            self._create_default_encoders()
    
    def _create_default_encoders(self):
        """Create default label encoders"""
        # Gender encoder
        gender_encoder = LabelEncoder()
        gender_encoder.classes_ = np.array(['Female', 'Male'])
        
        # Class encoder
        class_encoder = LabelEncoder()
        class_encoder.classes_ = np.array(self.class_names)
        
        self.encoders = {
            'gender': gender_encoder,
            'class': class_encoder
        }
        
        # Save encoders
        os.makedirs('models', exist_ok=True)
        with open('models/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
    
    def _preprocess_input(self, data, model_name):
        """Preprocess input data for prediction"""
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Map input keys to expected feature names
        feature_mapping = {
            'age': 'AGE',
            'gender': 'Gender',
            'urea': 'Urea',
            'hba1c': 'HbA1c',
            'chol': 'Chol',
            'bmi': 'BMI',
            'vldl': 'VLDL'
        }
        
        # Rename columns
        df = df.rename(columns=feature_mapping)
        
        # Encode gender
        if 'Gender' in df.columns and 'gender' in self.encoders:
            try:
                df['Gender'] = self.encoders['gender'].transform(df['Gender'])
            except ValueError:
                # Handle unknown gender values
                df['Gender'] = 0  # Default to first class
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Scale features
        if model_name in self.scalers:
            df_scaled = self.scalers[model_name].transform(df)
            return df_scaled
        else:
            return df.values
    
    def predict(self, data):
        """Make predictions using all available models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Preprocess data
                X = self._preprocess_input(data, model_name)
                
                # Make prediction
                prediction = model.predict(X)[0]
                
                # Get prediction probabilities if available
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    probabilities = {
                        self.class_names[i]: float(proba[i]) 
                        for i in range(len(proba))
                    }
                
                # Decode prediction
                if 'class' in self.encoders:
                    try:
                        predicted_class = self.encoders['class'].inverse_transform([prediction])[0]
                    except:
                        predicted_class = self.class_names[prediction] if prediction < len(self.class_names) else 'Unknown'
                else:
                    predicted_class = self.class_names[prediction] if prediction < len(self.class_names) else 'Unknown'
                
                predictions[model_name] = {
                    'prediction': predicted_class,
                    'prediction_code': int(prediction),
                    'probabilities': probabilities,
                    'confidence': float(max(probabilities.values())) if probabilities else None
                }
                
            except Exception as e:
                print(f"❌ Error predicting with {model_name}: {e}")
                predictions[model_name] = {
                    'prediction': 'Error',
                    'error': str(e)
                }
        
        return predictions
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'models_loaded': list(self.models.keys()),
            'scalers_loaded': list(self.scalers.keys()),
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'timestamp': datetime.now().isoformat()
        }
        
        for model_name, model in self.models.items():
            info[f'{model_name}_type'] = type(model).__name__
            if hasattr(model, 'n_estimators'):
                info[f'{model_name}_n_estimators'] = model.n_estimators
            if hasattr(model, 'C'):
                info[f'{model_name}_C'] = model.C
        
        return info
    
    def is_ready(self):
        """Check if predictor is ready to make predictions"""
        return len(self.models) > 0 and len(self.scalers) > 0