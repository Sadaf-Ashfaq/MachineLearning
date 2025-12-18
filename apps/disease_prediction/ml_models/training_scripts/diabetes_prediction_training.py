import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / 'datasets' / 'diabetes_prediction_dataset.csv'

df = pd.read_csv(DATASET_PATH)
target_col = 'diabetes' if 'diabetes' in df.columns else df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

if y.dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
print(f"Class distribution in training: {np.bincount(y_train)}")

best_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced', 
    max_depth=None,  
    min_samples_split=10,
    min_samples_leaf=5  
)

print(f"\n{'='*50}")
print(f"Training Random Forest (Best Model)...")

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)

print(f"Random Forest Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Cross-Validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\n{'='*50}")
print(f"Best Model: Random Forest with accuracy: {accuracy:.4f}")

with open('../trained_models/diabetes_prediction_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('../trained_models/diabetes_scaler.pkl', 'wb') as f:
    pickle.dump(None, f)  

with open('../trained_models/diabetes_feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

metadata = {
    'model_name': 'Random Forest',
    'accuracy': accuracy,
    'uses_scaling': False
}
with open('../trained_models/diabetes_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\nModel saved to: trained_models/diabetes_prediction_model.pkl")
print("Scaler saved to: trained_models/diabetes_scaler.pkl")
print("Feature names saved to: trained_models/diabetes_feature_names.pkl")
print("Metadata saved to: trained_models/diabetes_metadata.pkl")

print(f"\nFinal Test Accuracy: {accuracy:.4f}")



import pickle

# Load and check metadata
with open('../trained_models/diabetes_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("Metadata contents:")
print(metadata)
print("\nMetadata type:", type(metadata))

# Load and check feature names
with open('../trained_models/diabetes_feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("\nFeature names:")
print(feature_names)