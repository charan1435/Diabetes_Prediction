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
        # These will be auto-detected from the actual models
        self.feature_names = None
        self.class_names = ['N', 'P', 'Y']  # Normal, Pre-diabetes, Diabetes
        
        self._load_models()
        self._load_scalers()
        self._detect_features()
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
    
    def _detect_features(self):
        """Auto-detect feature names from the trained models"""
        # Try to get feature names from the first available model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'feature_names_in_'):
                    # sklearn >= 1.0 stores feature names
                    self.feature_names = list(model.feature_names_in_)
                    print(f"✅ Detected feature names from {model_name}: {self.feature_names}")
                    break
                elif hasattr(model, 'n_features_'):
                    # Older sklearn versions, try to infer from scaler
                    if model_name in self.scalers:
                        scaler = self.scalers[model_name]
                        if hasattr(scaler, 'feature_names_in_'):
                            self.feature_names = list(scaler.feature_names_in_)
                            print(f"✅ Detected feature names from {model_name} scaler: {self.feature_names}")
                            break
                        else:
                            n_features = model.n_features_
                            print(f"⚠️ Model expects {n_features} features but no names found")
            except Exception as e:
                print(f"⚠️ Could not detect features from {model_name}: {e}")
        
        # If we couldn't detect features, use the common diabetes dataset features
        if self.feature_names is None:
            # Try the most common feature order from diabetes datasets
            possible_features = [
                ['AGE', 'Gender', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'BMI', 'VLDL'],
                ['AGE', 'Gender', 'Urea', 'Cr', 'HbA1c', 'HDL', 'LDL', 'Chol', 'TG', 'BMI', 'VLDL'],
                ['Age', 'Gender', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'BMI', 'VLDL'],
                ['Age', 'Gender', 'Urea', 'Cr', 'HbA1c', 'HDL', 'LDL', 'Chol', 'TG', 'BMI', 'VLDL']
            ]
            
            # Try each possible feature combination
            for features in possible_features:
                try:
                    # Test with dummy data
                    test_data = np.zeros((1, len(features)))
                    model = list(self.models.values())[0]
                    model.predict(test_data)
                    self.feature_names = features
                    print(f"✅ Successfully detected feature order: {self.feature_names}")
                    break
                except:
                    continue
            
            if self.feature_names is None:
                # Default fallback
                self.feature_names = ['AGE', 'Gender', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'BMI', 'VLDL']
                print(f"⚠️ Using default feature names: {self.feature_names}")
    
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
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Map input keys to expected feature names (case-insensitive)
        feature_mapping = {}
        for input_key in data.keys():
            for expected_feature in self.feature_names:
                if input_key.lower() == expected_feature.lower():
                    feature_mapping[input_key] = expected_feature
                    break
                elif input_key.lower() == 'age' and expected_feature.upper() == 'AGE':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'gender' and expected_feature.upper() == 'GENDER':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'cr' and expected_feature.upper() == 'CR':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'hba1c' and expected_feature.upper() == 'HBA1C':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'hdl' and expected_feature.upper() == 'HDL':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'ldl' and expected_feature.upper() == 'LDL':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'chol' and expected_feature.upper() == 'CHOL':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'tg' and expected_feature.upper() == 'TG':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'bmi' and expected_feature.upper() == 'BMI':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'vldl' and expected_feature.upper() == 'VLDL':
                    feature_mapping[input_key] = expected_feature
                elif input_key.lower() == 'urea' and expected_feature.upper() == 'UREA':
                    feature_mapping[input_key] = expected_feature
        
        print(f"🔍 Feature mapping: {feature_mapping}")
        
        # Rename columns
        df = df.rename(columns=feature_mapping)
        
        # Encode gender
        gender_col = None
        for col in df.columns:
            if col.upper() in ['GENDER', 'GENDER']:
                gender_col = col
                break
        
        if gender_col and 'gender' in self.encoders:
            try:
                df[gender_col] = self.encoders['gender'].transform(df[gender_col])
                print(f"✅ Encoded gender column: {gender_col}")
            except ValueError as e:
                print(f"⚠️ Gender encoding error: {e}")
                df[gender_col] = 0  # Default to first class
        
        # Ensure we have all required features
        missing_features = []
        for feature in self.feature_names:
            if feature not in df.columns:
                missing_features.append(feature)
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        print(f"✅ Final DataFrame shape: {df.shape}")
        print(f"✅ Final DataFrame columns: {list(df.columns)}")
        
        # Scale features
        if model_name in self.scalers:
            df_scaled = self.scalers[model_name].transform(df)
            print(f"✅ Applied scaling for {model_name}")
            return df_scaled
        else:
            print(f"⚠️ No scaler found for {model_name}")
            return df.values
    
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