import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings("ignore")

print("Starting Heart Disease Model Training...")

# ======================================================
# 1. LOAD DATA
# ======================================================
print("\n1. Loading data...")
DATA_PATH = "../datasets/heart_disease_train.csv"
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip()

print(f"Dataset shape: {df.shape}")
print(df.head())

# ======================================================
# 2. DATA CLEANING
# ======================================================
print("\n2. Cleaning data...")

# Numerical columns
num_cols = [
    "Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"
]

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns
cat_cols = [
    "Sex", "ChestPainType", "RestingECG",
    "ExerciseAngina", "ST_Slope"
]

for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled!")

# ======================================================
# 3. ENCODING CATEGORICAL FEATURES
# ======================================================
print("\n3. Encoding categorical features...")

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
with open("../trained_models/heart_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Categorical features encoded and encoders saved!")

# ======================================================
# 4. PREPARE FEATURES & TARGET
# ======================================================
print("\n4. Preparing features and target...")

target_col = "HeartDisease"

feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols]
y = df[target_col]

print(f"Features shape: {X.shape}")
print("Target distribution:")
print(y.value_counts())

# ======================================================
# 5. TRAIN-TEST SPLIT
# ======================================================
print("\n5. Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")

# ======================================================
# 6. FEATURE SCALING
# ======================================================
print("\n6. Scaling features...")
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler
with open("../trained_models/heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Scaler saved!")

# ======================================================
# 7. TRAIN MODEL
# ======================================================
print("\n7. Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("Model trained successfully!")

# ======================================================
# 8. EVALUATION
# ======================================================
print("\n8. Evaluating model...")

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=["No Disease", "Disease"]))

print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))

# ======================================================
# 9. FEATURE IMPORTANCE
# ======================================================
print("\n9. Feature Importance:")

feature_importance = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print(feature_importance)

# ======================================================
# 10. SAVE MODEL & METADATA
# ======================================================
print("\n10. Saving model and files...")

with open("../trained_models/heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("../trained_models/heart_feature_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("\nMODEL TRAINING COMPLETE!")

print("\nSaved files:")
print("  - heart_disease_model.pkl")
print("  - heart_scaler.pkl")
print("  - heart_encoders.pkl")
print("  - heart_feature_columns.pkl")
