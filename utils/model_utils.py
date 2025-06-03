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
        # Use the detected feature order from your logs
        self.feature_names = ['AGE', 'Gender', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'BMI', 'VLDL']
        self.class_names = ['N', 'P', 'Y']  # Normal, Pre-diabetes, Diabetes
        
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
        print(f"🔍 Processing input for {model_name}")
        print(f"🔍 Expected features: {self.feature_names}")
        print(f"🔍 Input data keys: {list(data.keys())}")
        
        # Map input data to the correct order and feature names
        feature_mapping = {
            'age': 'AGE',
            'gender': 'Gender',
            'urea': 'Urea',
            'cr': 'Cr',
            'hba1c': 'HbA1c',
            'hdl': 'HDL',
            'ldl': 'LDL',
            'chol': 'Chol',
            'tg': 'TG',
            'bmi': 'BMI',
            'vldl': 'VLDL'
        }
        
        print(f"🔍 Feature mapping: {feature_mapping}")
        
        # Create a list in the exact order expected
        feature_values = []
        
        for expected_feature in self.feature_names:
            # Find the corresponding input key
            input_key = None
            for key, mapped_feature in feature_mapping.items():
                if mapped_feature == expected_feature:
                    input_key = key
                    break
            
            if input_key and input_key in data:
                value = data[input_key]
                
                # Handle gender encoding
                if expected_feature == 'Gender' and isinstance(value, str):
                    if 'gender' in self.encoders:
                        try:
                            value = self.encoders['gender'].transform([value])[0]
                            print(f"✅ Encoded gender: {data[input_key]} -> {value}")
                        except ValueError as e:
                            print(f"⚠️ Gender encoding error: {e}")
                            value = 0  # Default to first class
                    else:
                        # Manual encoding if no encoder available
                        value = 1 if value.lower() == 'male' else 0
                        print(f"✅ Manual gender encoding: {data[input_key]} -> {value}")
                
                feature_values.append(float(value))
            else:
                raise ValueError(f"Missing required feature: {expected_feature} (input key: {input_key})")
        
        # Convert to numpy array (this bypasses any feature name validation)
        X = np.array([feature_values])
        
        print(f"✅ Created feature array shape: {X.shape}")
        print(f"✅ Feature values: {X[0]}")
        
        # Scale features - using numpy array to avoid feature name issues
        if model_name in self.scalers:
            X_scaled = self.scalers[model_name].transform(X)
            print(f"✅ Applied scaling for {model_name}")
            return X_scaled
        else:
            print(f"⚠️ No scaler found for {model_name}")
            return X
    
    def predict(self, data):
        """Make predictions using all available models"""
        predictions = {}
        
        print(f"🚀 Starting prediction with data: {data}")
        
        for model_name, model in self.models.items():
            try:
                print(f"\n🔄 Processing {model_name}...")
                
                # Preprocess data
                X = self._preprocess_input(data, model_name)
                
                # Make prediction
                prediction = model.predict(X)[0]
                print(f"✅ {model_name} raw prediction: {prediction}")
                
                # Get prediction probabilities if available
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    probabilities = {
                        self.class_names[i]: float(proba[i]) 
                        for i in range(len(proba))
                    }
                    print(f"✅ {model_name} probabilities: {probabilities}")
                
                # Decode prediction
                if 'class' in self.encoders:
                    try:
                        predicted_class = self.encoders['class'].inverse_transform([prediction])[0]
                    except:
                        predicted_class = self.class_names[prediction] if prediction < len(self.class_names) else 'Unknown'
                else:
                    predicted_class = self.class_names[prediction] if prediction < len(self.class_names) else 'Unknown'
                
                print(f"✅ {model_name} final prediction: {predicted_class}")
                
                predictions[model_name] = {
                    'prediction': predicted_class,
                    'prediction_code': int(prediction),
                    'probabilities': probabilities,
                    'confidence': float(max(probabilities.values())) if probabilities else None
                }
                
            except Exception as e:
                print(f"❌ Error predicting with {model_name}: {e}")
                import traceback
                traceback.print_exc()
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
        return len(self.models) > 0 and len(self.scalers) > 0 and self.feature_names is not None