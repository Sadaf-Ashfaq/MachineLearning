import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Starting Liver Disease Model Training...")


# 1. LOAD DATA
print("\n1. Loading data...")
train_df = pd.read_csv('../datasets/liver_disease_train.csv', encoding='latin-1')
test_df = pd.read_excel('../datasets/liver_disease_test.xlsx')

# Clean column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# 2. DATA CLEANING
print("\n2. Cleaning data...")

# Handle missing values - Numerical (median)
numerical_cols = ['Age of the patient', 'Total Bilirubin', 'Direct Bilirubin', 
                  'Alkphos Alkaline Phosphotase', 'Sgpt Alamine Aminotransferase',
                  'Sgot Aspartate Aminotransferase', 'Total Protiens', 'ALB Albumin',
                  'A/G Ratio Albumin and Globulin Ratio']

for col in numerical_cols:
    if col in train_df.columns:
        median_val = train_df[col].median()
        train_df[col].fillna(median_val, inplace=True)
        if col in test_df.columns:
            test_df[col].fillna(median_val, inplace=True)

# Handle missing values - Gender (mode)
if train_df['Gender of the patient'].isnull().sum() > 0:
    mode_val = train_df['Gender of the patient'].mode()[0]
    train_df['Gender of the patient'].fillna(mode_val, inplace=True)
    if 'Gender of the patient' in test_df.columns:
        test_df['Gender of the patient'].fillna(mode_val, inplace=True)

print("Missing values handled!")

# 3. ENCODE GENDER
print("\n3. Encoding categorical features...")
le = LabelEncoder()
train_df['Gender_Encoded'] = le.fit_transform(train_df['Gender of the patient'])
if 'Gender of the patient' in test_df.columns:
    test_df['Gender_Encoded'] = le.transform(test_df['Gender of the patient'])

# Save label encoder
with open('../trained_models/gender_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Gender encoded and encoder saved!")

# 4. PREPARE FEATURES AND TARGET
print("\n4. Preparing features and target...")
feature_cols = numerical_cols + ['Gender_Encoded']
X = train_df[feature_cols]
y = train_df['Result']

# Convert Result: 1=Disease, 2=Healthy -> 1=Disease, 0=Healthy
y = (y == 1).astype(int)

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{pd.Series(y).value_counts()}")

# 5. TRAIN-TEST SPLIT
print("\n5. Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# 6. FEATURE SCALING
print("\n6. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save scaler
with open('../trained_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved!")

# 7. TRAIN MODEL
print("\n7. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight='balanced',  # Handle imbalanced data
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("Model trained!")

# 8. EVALUATE
print("\n8. Evaluating model...")
y_pred_train = model.predict(X_train_scaled)
y_pred_val = model.predict(X_val_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
val_acc = accuracy_score(y_val, y_pred_val)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

print("\nValidation Classification Report:")
print(classification_report(y_val, y_pred_val, target_names=['Healthy', 'Disease']))

print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_val))

# 9. FEATURE IMPORTANCE
print("\n9. Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# 10. SAVE MODEL
print("\n10. Saving model...")
with open('../trained_models/liver_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save feature columns
with open('../trained_models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)


print("MODEL TRAINING COMPLETE!")

print("\nSaved files:")
print("  - liver_disease_model.pkl")
print("  - scaler.pkl")
print("  - gender_encoder.pkl")
print("  - feature_columns.pkl")