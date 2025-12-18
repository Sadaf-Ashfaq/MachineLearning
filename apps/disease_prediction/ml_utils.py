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
    try:
        model, scaler, gender_encoder, feature_columns = load_liver_model()
        gender_encoded = gender_encoder.transform([data['gender']])[0]

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

        df = pd.DataFrame([features], columns=feature_columns)
        scaled = scaler.transform(df)

        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1] * 100

        return {
            'prediction': 'Disease Detected' if pred == 1 else 'Healthy',
            'risk_percentage': round(prob, 2),
            'status': 'danger' if pred == 1 else 'success'
        }

    except Exception as e:
        return {'error': str(e)}


# ============================================================
# LIFESTYLE SCORE FUNCTIONS
# ============================================================

def load_lifestyle_model():
    with open(os.path.join(MODEL_DIR, 'lifestyle_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'lifestyle_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'lifestyle_features.pkl'), 'rb') as f:
        features = pickle.load(f)

    return model, scaler, features


def predict_lifestyle_score(data):
    try:
        model, scaler, features = load_lifestyle_model()

        df = pd.DataFrame([[ 
            data['physical_activity'], data['nutrition_score'], data['stress_level'],
            data['mindfulness'], data['sleep_hours'], data['hydration'],
            data['bmi'], data['alcohol'], data['smoking']
        ]], columns=features)

        scaled = scaler.transform(df)
        score = max(0, min(100, model.predict(scaled)[0]))

        if score >= 75:
            return {'score': score, 'status': 'Good', 'status_color': 'success'}
        elif score >= 50:
            return {'score': score, 'status': 'Average', 'status_color': 'warning'}
        else:
            return {'score': score, 'status': 'Poor', 'status_color': 'danger'}

    except Exception as e:
        return {'error': str(e)}


# ============================================================
# DIABETES PREDICTION FUNCTIONS
# ============================================================

def load_diabetes_model():
    with open(os.path.join(MODEL_DIR, 'diabetes_prediction_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'diabetes_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'diabetes_feature_names.pkl'), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'diabetes_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    return model, scaler, features, metadata


def predict_diabetes(data):
    try:
        model, scaler, features, metadata = load_diabetes_model()

        gender = 1 if data['gender'] == 'Male' else 0
        smoking_map = {'never':0,'No Info':1,'current':2,'former':3,'ever':4,'not current':5}
        smoking = smoking_map.get(data['smoking_history'], 0)

        df = pd.DataFrame([[ 
            gender, data['age'], data['hypertension'], data['heart_disease'],
            smoking, data['bmi'], data['HbA1c_level'], data['blood_glucose_level']
        ]], columns=features)

        processed = scaler.transform(df) if metadata.get('uses_scaling') else df
        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0][1] * 100

        return {
            'prediction': 'Diabetes Detected' if pred else 'No Diabetes',
            'risk_percentage': round(prob, 2),
            'status': 'danger' if pred else 'success'
        }

    except Exception as e:
        return {'error': str(e)}


# ============================================================
# STROKE PREDICTION FUNCTIONS
# ============================================================

def load_stroke_model():
    with open(os.path.join(MODEL_DIR, 'stroke_prediction_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'stroke_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'stroke_model_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    return model, scaler, metadata


def predict_stroke(data):
    try:
        model, scaler, metadata = load_stroke_model()
        cols = scaler.feature_names_in_

        features = {c:0 for c in cols}
        features.update({
            'age': data['age'],
            'hypertension': data['hypertension'],
            'heart_disease': data['heart_disease'],
            'avg_glucose_level': data['avg_glucose_level'],
            'bmi': data['bmi'],
            f"gender_{data['gender']}": 1,
            f"ever_married_{data['ever_married']}": 1,
            f"Residence_type_{data['Residence_type']}": 1,
            f"work_type_{data['work_type']}": 1,
            f"smoking_status_{data['smoking_status']}": 1
        })

        df = pd.DataFrame([[features[c] for c in cols]], columns=cols)
        scaled = scaler.transform(df)

        prob = model.predict_proba(scaled)[0][1]
        pred = prob >= metadata.get('best_threshold', 0.5)

        return {
            'prediction': 'High Stroke Risk' if pred else 'Low Stroke Risk',
            'risk_percentage': round(prob*100, 2),
            'status': 'danger' if pred else 'success'
        }

    except Exception as e:
        return {'error': str(e)}


# ============================================================
# OBESITY PREDICTION FUNCTIONS
# ============================================================

def load_obesity_model():
    with open(os.path.join(MODEL_DIR, 'obesity_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'obesity_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'obesity_target_encoder.pkl'), 'rb') as f:
        target_encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'obesity_label_encoders.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'obesity_features.pkl'), 'rb') as f:
        features = pickle.load(f)

    return model, scaler, target_encoder, label_encoders, features


def predict_obesity(data):
    try:
        model, scaler, target_encoder, encoders, features = load_obesity_model()
        bmi = data['weight'] / (data['height'] ** 2)

        df = pd.DataFrame([[ 
            encoders['Gender'].transform([data['gender']])[0],
            data['age'], data['height'], data['weight'],
            encoders['family_history'].transform([data['family_history']])[0],
            encoders['FAVC'].transform([data['favc']])[0],
            data['fcvc'], data['ncp'],
            encoders['CAEC'].transform([data['caec']])[0],
            encoders['SMOKE'].transform([data['smoke']])[0],
            data['ch2o'],
            encoders['SCC'].transform([data['scc']])[0],
            data['faf'], data['tue'],
            encoders['CALC'].transform([data['calc']])[0],
            encoders['MTRANS'].transform([data['mtrans']])[0],
            bmi
        ]], columns=features)

        scaled = scaler.transform(df)
        pred = model.predict(scaled)[0]
        label = target_encoder.inverse_transform([pred])[0]

        return {
            'prediction': label.replace('_',' ').title(),
            'bmi': round(bmi, 2),
            'confidence': round(max(model.predict_proba(scaled)[0]) * 100, 2)
        }

    except Exception as e:
        return {'error': str(e)}