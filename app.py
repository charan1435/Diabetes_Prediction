from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
import json
from utils.model_utils import ModelPredictor
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize model predictor
try:
    predictor = ModelPredictor()
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    predictor = None

@app.route('/')
def index():
    """Home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and make predictions"""
    if not predictor:
        flash('Models not loaded. Please check model files.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get form data
        form_data = {
            'age': float(request.form.get('age')),
            'gender': request.form.get('gender'),
            'urea': float(request.form.get('urea')),
            'hba1c': float(request.form.get('hba1c')),
            'chol': float(request.form.get('chol')),
            'bmi': float(request.form.get('bmi')),
            'vldl': float(request.form.get('vldl'))
        }
        
        # Validate data
        if any(v is None for v in form_data.values()):
            flash('Please fill in all fields', 'error')
            return redirect(url_for('index'))
        
        # Make predictions
        predictions = predictor.predict(form_data)
        
        return render_template('results.html', 
                             form_data=form_data, 
                             predictions=predictions)
    
    except ValueError as e:
        flash(f'Invalid input: {str(e)}', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Prediction error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)"""
    if not predictor:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['age', 'gender', 'urea', 'hba1c', 'chol', 'bmi', 'vldl']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'error': f'Missing fields: {missing_fields}'}), 400
        
        # Make predictions
        predictions = predictor.predict(data)
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'input_data': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/info')
def model_info():
    """Get information about loaded models"""
    if not predictor:
        return jsonify({'error': 'Models not loaded'}), 500
    
    return jsonify(predictor.get_model_info())

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'models_loaded': predictor is not None
    }
    return jsonify(status)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)