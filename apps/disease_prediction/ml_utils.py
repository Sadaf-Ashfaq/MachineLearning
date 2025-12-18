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
    


# ============================================================
# DIABETES PREDICTION FUNCTIONS
# ============================================================

def load_diabetes_model():
    model_path = os.path.join(MODEL_DIR, 'diabetes_prediction_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'diabetes_scaler.pkl')
    features_path = os.path.join(MODEL_DIR, 'diabetes_feature_names.pkl')
    metadata_path = os.path.join(MODEL_DIR, 'diabetes_metadata.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, feature_names, metadata

def predict_diabetes(data):
    """
    data = {
        'gender': 'Female',
        'age': 80.0,
        'hypertension': 0,
        'heart_disease': 1,
        'smoking_history': 'never',
        'bmi': 25.19,
        'HbA1c_level': 6.6,
        'blood_glucose_level': 140
    }
    """
    try:
        # Load model and preprocessors
        model, scaler, feature_names, metadata = load_diabetes_model()
        
        # Manual encoding for gender (0 = Female, 1 = Male)
        gender_map = {'Female': 0, 'Male': 1}
        gender_encoded = gender_map.get(data['gender'], 0)
        
        # Manual encoding for smoking_history
        smoking_map = {
            'never': 0,
            'No Info': 1,
            'current': 2,
            'former': 3,
            'ever': 4,
            'not current': 5
        }
        smoking_encoded = smoking_map.get(data['smoking_history'], 0)
        
        # Create feature array in correct order with encoded values
        features = [
            gender_encoded,  # Encoded gender
            float(data['age']),
            int(data['hypertension']),
            int(data['heart_disease']),
            smoking_encoded,  # Encoded smoking_history
            float(data['bmi']),
            float(data['HbA1c_level']),
            int(data['blood_glucose_level'])
        ]
        
        # Convert to DataFrame with correct column names
        feature_df = pd.DataFrame([features], columns=feature_names)
        
        # Check if scaling is needed (according to metadata)
        if metadata.get('uses_scaling', False):
            features_processed = scaler.transform(feature_df)
        else:
            features_processed = feature_df
        
        # Predict
        prediction = model.predict(features_processed)[0]
        probability = model.predict_proba(features_processed)[0]
        
        # Get risk probability (probability of diabetes)
        risk_probability = probability[1] * 100
        
        result = {
            'prediction': 'Diabetes Detected' if prediction == 1 else 'No Diabetes',
            'risk_percentage': round(risk_probability, 2),
            'status': 'danger' if prediction == 1 else 'success'
        }
        
        return result
    
    except Exception as e:
        return {'error': str(e)}
    

# ============================================================
# STROKE PREDICTION FUNCTIONS
# ============================================================

def load_stroke_model():
    model_path = os.path.join(MODEL_DIR, 'stroke_prediction_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'stroke_scaler.pkl')
    metadata_path = os.path.join(MODEL_DIR, 'stroke_model_metadata.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, metadata

def predict_stroke(data):
    """
    data = {
        'gender': 'Male',
        'age': 67.0,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 'formerly smoked'
    }
    """
    try:
        # Load model and preprocessors
        model, scaler, metadata = load_stroke_model()
        
        # Get the exact feature order from scaler
        if hasattr(scaler, 'feature_names_in_'):
            feature_order = scaler.feature_names_in_.tolist()
        else:
            # Fallback: use first 16 unique features from metadata
            feature_names = metadata.get('feature_names', [])
            seen = set()
            feature_order = []
            for name in feature_names:
                if name not in seen:
                    seen.add(name)
                    feature_order.append(name)
                if len(feature_order) == 16:  # We need exactly 16 features
                    break
        
        # Create one-hot encoded features dictionary
        features_dict = {
            'age': float(data['age']),
            'hypertension': int(data['hypertension']),
            'heart_disease': int(data['heart_disease']),
            'avg_glucose_level': float(data['avg_glucose_level']),
            'bmi': float(data['bmi']),
            
            # Gender encoding
            'gender_Male': 1 if data['gender'] == 'Male' else 0,
            'gender_Other': 1 if data['gender'] == 'Other' else 0,
            
            # Ever Married encoding
            'ever_married_Yes': 1 if data['ever_married'] == 'Yes' else 0,
            
            # Work Type encoding
            'work_type_Never_worked': 1 if data['work_type'] == 'Never_worked' else 0,
            'work_type_Private': 1 if data['work_type'] == 'Private' else 0,
            'work_type_Self-employed': 1 if data['work_type'] == 'Self-employed' else 0,
            'work_type_children': 1 if data['work_type'] == 'children' else 0,
            
            # Residence Type encoding
            'Residence_type_Urban': 1 if data['Residence_type'] == 'Urban' else 0,
            
            # Smoking Status encoding
            'smoking_status_formerly smoked': 1 if data['smoking_status'] == 'formerly smoked' else 0,
            'smoking_status_never smoked': 1 if data['smoking_status'] == 'never smoked' else 0,
            'smoking_status_smokes': 1 if data['smoking_status'] == 'smokes' else 0,
        }
        
        # Create list in the exact order the scaler expects
        feature_values = [features_dict[col] for col in feature_order]
        
        # Convert to DataFrame with exact column order
        feature_df = pd.DataFrame([feature_values], columns=feature_order)
        
        # Scale features
        features_scaled = scaler.transform(feature_df)
        
        # Get probability
        probability = model.predict_proba(features_scaled)[0]
        
        # Use best threshold from metadata
        best_threshold = metadata.get('best_threshold', 0.5)
        
        # Make prediction using custom threshold
        prediction = 1 if probability[1] >= best_threshold else 0
        
        # Get risk probability
        risk_probability = probability[1] * 100
        
        result = {
            'prediction': 'High Stroke Risk' if prediction == 1 else 'Low Stroke Risk',
            'risk_percentage': round(risk_probability, 2),
            'status': 'danger' if prediction == 1 else 'success'
        }
        
        return result
    
    except Exception as e:
        return {'error': str(e)}