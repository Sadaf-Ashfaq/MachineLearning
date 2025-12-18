import pickle
import pandas as pd
import numpy as np
import os

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models', 'trained_models')

# Load model and preprocessors
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
    

    #heart disease prediction function
import pickle
import numpy as np
import os

# =========================
# BASE DIRECTORY
# =========================
# ml_utils.py jis folder me hai uska path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# trained models ka folder
MODEL_DIR = os.path.join(BASE_DIR, 'ml_models', 'trained_models')


def predict_heart_disease(input_data: dict):
    """
    input_data example:
    {
        'Age': 45,
        'Sex': 'M',
        'ChestPainType': 'ATA',
        'RestingBP': 130,
        'Cholesterol': 230,
        'FastingBS': 0,
        'RestingECG': 'Normal',
        'MaxHR': 150,
        'ExerciseAngina': 'N',
        'Oldpeak': 1.2,
        'ST_Slope': 'Up'
    }
    """

    try:
        # =========================
        # LOAD MODEL & PREPROCESSORS
        # =========================
        model = pickle.load(
            open(os.path.join(MODEL_DIR, 'heart_disease_model.pkl'), 'rb')
        )

        scaler = pickle.load(
            open(os.path.join(MODEL_DIR, 'heart_scaler.pkl'), 'rb')
        )

        encoders = pickle.load(
            open(os.path.join(MODEL_DIR, 'heart_encoders.pkl'), 'rb')
        )

        feature_columns = pickle.load(
            open(os.path.join(MODEL_DIR, 'heart_feature_columns.pkl'), 'rb')
        )

        # =========================
        # ENCODE CATEGORICAL VALUES
        # =========================
        processed_data = {}

        for col in feature_columns:
            value = input_data[col]

            # agar column categorical hai to encoder use karo
            if col in encoders:
                processed_data[col] = encoders[col].transform([value])[0]
            else:
                processed_data[col] = value

        # =========================
        # CREATE INPUT ARRAY
        # =========================
        X = np.array(
            [processed_data[col] for col in feature_columns]
        ).reshape(1, -1)

        # scale input
        X_scaled = scaler.transform(X)

        # =========================
        # PREDICTION
        # =========================
        prediction = model.predict(X_scaled)[0]

        # disease class (1) ki probability
        risk_probability = model.predict_proba(X_scaled)[0][1] * 100

        # =========================
        # FINAL RESULT (UI FRIENDLY)
        # =========================
        return {
            "prediction": "High Risk" if prediction == 1 else "Low Risk",
            "risk_percentage": round(risk_probability, 2),
            "status": "danger" if prediction == 1 else "success"
        }

    except Exception as e:
        return {
            "error": str(e)
        }

