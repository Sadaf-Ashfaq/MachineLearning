import pickle
import pandas as pd
import numpy as np
import os

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models', 'trained_models')

# ============================================================
# LIVER DISEASE FUNCTIONS
# ============================================================

def load_liver_model():
    model_path = os.path.join(MODEL_DIR, 'liver_disease_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    encoder_path = os.path.join(MODEL_DIR, 'gender_encoder.pkl')
    features_path = os.path.join(MODEL_DIR, 'feature_columns.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        gender_encoder = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, scaler, gender_encoder, feature_columns

def predict_liver_disease(data):
    """
    data = {
        'age': 45,
        'gender': 'Male',
        'total_bilirubin': 1.2,
        'direct_bilirubin': 0.3,
        'alkaline_phosphotase': 200,
        'alamine_aminotransferase': 35,
        'aspartate_aminotransferase': 42,
        'total_proteins': 6.5,
        'albumin': 3.2,
        'ag_ratio': 0.9
    }
    """
    try:
        # Load model and preprocessors
        model, scaler, gender_encoder, feature_columns = load_liver_model()
        
        # Encode gender
        gender_encoded = gender_encoder.transform([data['gender']])[0]
        
        # Create feature array in correct order
        features = [
            data['age'],
            data['total_bilirubin'],
            data['direct_bilirubin'],
            data['alkaline_phosphotase'],
            data['alamine_aminotransferase'],
            data['aspartate_aminotransferase'],
            data['total_proteins'],
            data['albumin'],
            data['ag_ratio'],
            gender_encoded
        ]
        
        # Convert to DataFrame with correct column names
        feature_df = pd.DataFrame([features], columns=feature_columns)
        
        # Scale features
        features_scaled = scaler.transform(feature_df)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get risk probability (probability of disease)
        risk_probability = probability[1] * 100
        
        result = {
            'prediction': 'Disease Detected' if prediction == 1 else 'Healthy',
            'risk_percentage': round(risk_probability, 2),
            'status': 'danger' if prediction == 1 else 'success'
        }
        
        return result
    
    except Exception as e:
        return {'error': str(e)}


# ============================================================
# LIFESTYLE SCORE FUNCTIONS
# ============================================================

def load_lifestyle_model():
    model_path = os.path.join(MODEL_DIR, 'lifestyle_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'lifestyle_scaler.pkl')
    features_path = os.path.join(MODEL_DIR, 'lifestyle_features.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_columns = pickle.load(f)
    
    return model, scaler, feature_columns

def predict_lifestyle_score(data):
    """
    data = {
        'physical_activity': 45.5,
        'nutrition_score': 7.2,
        'stress_level': 4.5,
        'mindfulness': 15.3,
        'sleep_hours': 7.5,
        'hydration': 2.8,
        'bmi': 23.5,
        'alcohol': 3.2,
        'smoking': 0.0
    }
    """
    try:
        # Load model and preprocessors
        model, scaler, feature_columns = load_lifestyle_model()
        
        # Create feature array in correct order
        features = [
            data['physical_activity'],
            data['nutrition_score'],
            data['stress_level'],
            data['mindfulness'],
            data['sleep_hours'],
            data['hydration'],
            data['bmi'],
            data['alcohol'],
            data['smoking']
        ]
        
        # Convert to DataFrame with correct column names
        feature_df = pd.DataFrame([features], columns=feature_columns)
        
        # Scale features
        features_scaled = scaler.transform(feature_df)
        
        # Predict
        score = model.predict(features_scaled)[0]
        
        # Clip score to 0-100 range (just in case)
        score = max(0, min(100, score))
        
        # Determine health status based on score
        if score >= 75:
            status = 'Good'
            status_color = 'success'
            message = 'Excellent lifestyle! Keep it up!'
        elif score >= 50:
            status = 'Average'
            status_color = 'warning'
            message = 'Good progress! Some improvements needed.'
        else:
            status = 'Poor'
            status_color = 'danger'
            message = 'Needs significant improvement. Consult a health professional.'
        
        result = {
            'score': round(score, 2),
            'status': status,
            'status_color': status_color,
            'message': message
        }
        
        return result
    
    except Exception as e:
        return {'error': str(e)}