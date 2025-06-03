#!/usr/bin/env python3
"""
Debug script to analyze your trained models and determine the exact feature requirements
"""

import pickle
import pandas as pd
import numpy as np
import os

def analyze_model(model_path, model_name):
    """Analyze a model file to understand its feature requirements"""
    print(f"\n{'='*50}")
    print(f"Analyzing {model_name}")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Model type: {type(model).__name__}")
        
        # Check for feature names
        if hasattr(model, 'feature_names_in_'):
            print(f"🔍 Feature names (sklearn >= 1.0): {list(model.feature_names_in_)}")
            return list(model.feature_names_in_)
        elif hasattr(model, 'n_features_'):
            print(f"🔍 Number of features: {model.n_features_}")
        elif hasattr(model, 'n_features_in_'):
            print(f"🔍 Number of features: {model.n_features_in_}")
        
        # Try to get more info
        if hasattr(model, 'get_params'):
            params = model.get_params()
            print(f"📋 Model parameters: {list(params.keys())[:5]}...")  # Show first 5 keys
        
        return None
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def analyze_scaler(scaler_path, scaler_name):
    """Analyze a scaler file to understand its feature requirements"""
    print(f"\n{'-'*30}")
    print(f"Analyzing {scaler_name} Scaler")
    print(f"{'-'*30}")
    
    if not os.path.exists(scaler_path):
        print(f"❌ Scaler file not found: {scaler_path}")
        return None
    
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"✅ Scaler loaded successfully")
        print(f"📊 Scaler type: {type(scaler).__name__}")
        
        # Check for feature names
        if hasattr(scaler, 'feature_names_in_'):
            print(f"🔍 Feature names: {list(scaler.feature_names_in_)}")
            return list(scaler.feature_names_in_)
        elif hasattr(scaler, 'n_features_in_'):
            print(f"🔍 Number of features: {scaler.n_features_in_}")
        
        # Check scaling parameters
        if hasattr(scaler, 'mean_'):
            print(f"📈 Feature means shape: {scaler.mean_.shape}")
        if hasattr(scaler, 'scale_'):
            print(f"📊 Feature scales shape: {scaler.scale_.shape}")
        
        return None
        
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return None

def test_feature_combinations(model_path, possible_features_list):
    """Test different feature combinations to find the correct one"""
    print(f"\n{'='*50}")
    print("Testing Feature Combinations")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        for i, features in enumerate(possible_features_list):
            print(f"\n🧪 Testing combination {i+1}: {features}")
            try:
                # Create dummy data
                dummy_data = np.random.rand(1, len(features))
                
                # Try prediction
                prediction = model.predict(dummy_data)
                print(f"✅ SUCCESS! This feature combination works")
                print(f"🎯 Correct features: {features}")
                return features
                
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                continue
        
        print(f"\n⚠️ None of the tested combinations worked")
        return None
        
    except Exception as e:
        print(f"❌ Error testing combinations: {e}")
        return None

def main():
    """Main analysis function"""
    print("🔍 DIABETES MODEL ANALYSIS")
    print("🔍" + "="*48)
    
    # Define paths
    model_paths = {
        'Random Forest': 'models/best_random_forest_model.pkl',
        'Logistic Regression': 'models/best_logistic_model_with_smote.pkl'
    }
    
    scaler_paths = {
        'Random Forest': 'models/standard_scaler.pkl',
        'Logistic Regression': 'models/logisticRegressionstandard_scaler.pkl'
    }
    
    # Analyze models
    detected_features = []
    for name, path in model_paths.items():
        features = analyze_model(path, name)
        if features:
            detected_features.append(features)
    
    # Analyze scalers
    for name, path in scaler_paths.items():
        features = analyze_scaler(path, name)
        if features:
            detected_features.append(features)
    
    # If we found features, we're done
    if detected_features:
        print(f"\n🎉 FOUND FEATURES!")
        for i, features in enumerate(detected_features):
            print(f"Feature set {i+1}: {features}")
        return
    
    # If no features detected, test common combinations
    print(f"\n🔬 No feature names detected. Testing common combinations...")
    
    possible_features = [
        # Different possible orders and naming conventions
        ['AGE', 'Gender', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'BMI', 'VLDL'],
        ['AGE', 'Gender', 'Urea', 'Cr', 'HbA1c', 'HDL', 'LDL', 'Chol', 'TG', 'BMI', 'VLDL'],
        ['Age', 'Gender', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'BMI', 'VLDL'],
        ['Age', 'Gender', 'Urea', 'Cr', 'HbA1c', 'HDL', 'LDL', 'Chol', 'TG', 'BMI', 'VLDL'],
        # Without some features
        ['AGE', 'Gender', 'Urea', 'HbA1c', 'Chol', 'BMI', 'VLDL'],
        ['Age', 'Gender', 'Urea', 'HbA1c', 'Chol', 'BMI', 'VLDL'],
        # Different naming
        ['AGE', 'Gender', 'Urea', 'Creatinine', 'HbA1c', 'Cholesterol', 'Triglycerides', 'HDL', 'LDL', 'BMI', 'VLDL'],
    ]
    
    # Test with first model
    first_model_path = list(model_paths.values())[0]
    correct_features = test_feature_combinations(first_model_path, possible_features)
    
    if correct_features:
        print(f"\n🎉 FOUND WORKING FEATURE COMBINATION!")
        print(f"✅ Use these features: {correct_features}")
        
        # Save the result
        result = {
            'feature_names': correct_features,
            'feature_count': len(correct_features)
        }
        
        with open('detected_features.json', 'w') as f:
            import json
            json.dump(result, f, indent=2)
        
        print(f"💾 Saved results to detected_features.json")
    else:
        print(f"\n❌ Could not determine correct features automatically")
        print(f"🔧 Manual investigation required")

if __name__ == "__main__":
    main()