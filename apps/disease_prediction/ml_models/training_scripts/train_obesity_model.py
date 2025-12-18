import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Starting Obesity Prediction Model Training...")

# 1. LOAD DATA
print("\n1. Loading data...")
df = pd.read_csv('../datasets/obesity.csv')
print(f"Original dataset shape: {df.shape}")

# Remove duplicates
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape}")

# 2. CREATE BMI FEATURE (Very Important!)
print("\n2. Creating BMI feature...")
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
print("BMI feature created!")
print(f"BMI range: {df['BMI'].min():.2f} - {df['BMI'].max():.2f}")

# 3. SEPARATE FEATURES AND TARGET
print("\n3. Preparing features and target...")
target_col = 'Obesity'
exclude_cols = [target_col]

X = df.drop(columns=exclude_cols).copy()
y = df[target_col].copy()

print(f"Features: {X.shape[1]}")
print(f"Target classes: {y.nunique()}")
print(f"\nClass distribution:")
print(y.value_counts().sort_index())

# 4. ENCODE CATEGORICAL FEATURES
print("\n4. Encoding categorical features...")

categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 
                        'SCC', 'CALC', 'MTRANS']
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 5. ENCODE TARGET VARIABLE
print("\n5. Encoding target variable...")
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

print("Target encoding:")
for i, label in enumerate(target_encoder.classes_):
    print(f"  {i}: {label}")

# 6. TRAIN-TEST SPLIT
print("\n6. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 7. FEATURE SCALING
print("\n7. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('../trained_models/obesity_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved!")

# 8. TRAIN LIGHTGBM MODEL (Optimized for high accuracy)
print("\n8. Training LightGBM Classifier...")

model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    num_leaves=50,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=20,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1,
    verbose=-1
)

model.fit(X_train_scaled, y_train)
print("Model trained!")

# 9. EVALUATE MODEL
print("\n9. Evaluating model...")

y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("MODEL PERFORMANCE")

print(f"\nTRAINING ACCURACY: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"TESTING ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")

if test_acc >= 0.9:
    print("\n🎉 EXCELLENT! Accuracy > 90%")
elif test_acc >= 0.8:
    print("\n✓ VERY GOOD! Accuracy > 80%")
elif test_acc >= 0.7:
    print("\n✓ GOOD! Accuracy > 70%")
else:
    print("\n⚠ Needs improvement")

print("DETAILED CLASSIFICATION REPORT")
print(classification_report(y_test, y_pred_test, 
                          target_names=target_encoder.classes_,
                          zero_division=0))

# 10. CONFUSION MATRIX
print("CONFUSION MATRIX")
cm = confusion_matrix(y_test, y_pred_test)
print(cm)

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, class_name in enumerate(target_encoder.classes_):
    class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  {class_name}: {class_acc*100:.2f}%")

# 11. FEATURE IMPORTANCE
print("FEATURE IMPORTANCE (Top 10)")
feature_names = X.columns.tolist()
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10))

# 12. SAMPLE PREDICTIONS
print("SAMPLE PREDICTIONS (First 10 Test Cases)")

# Fixed: y_test is already a numpy array, no need for .iloc or .values
predictions_df = pd.DataFrame({
    'Actual': target_encoder.inverse_transform(y_test[:10]),
    'Predicted': target_encoder.inverse_transform(y_pred_test[:10]),
    'Match': ['✓' if y_test[i] == y_pred_test[i] else '✗' for i in range(10)]
})

print(predictions_df)

# Fixed: y_test is already numpy array
correct_predictions = (y_test[:10] == y_pred_test[:10]).sum()
print(f"\nCorrect predictions: {correct_predictions}/10")

# 13. SAVE MODEL AND PREPROCESSORS

print("SAVING MODEL AND PREPROCESSORS")


# Save model
with open('../trained_models/obesity_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model saved: obesity_model.pkl")

# Save target encoder
with open('../trained_models/obesity_target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
print("✓ Target encoder saved: obesity_target_encoder.pkl")

# Save label encoders
with open('../trained_models/obesity_label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ Label encoders saved: obesity_label_encoders.pkl")

# Save feature columns
with open('../trained_models/obesity_features.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✓ Feature columns saved: obesity_features.pkl")

# 14. FINAL SUMMARY

print("TRAINING COMPLETE!")

print(f"\n✓ Final Test Accuracy: {test_acc*100:.2f}%")
print(f"✓ Model Type: LightGBM Classifier")
print(f"✓ Number of Classes: {len(target_encoder.classes_)}")
print(f"✓ Number of Features: {len(feature_names)}")
print(f"✓ Training Samples: {len(X_train)}")
print(f"✓ Test Samples: {len(X_test)}")

print("\nSaved files:")
print("  - obesity_model.pkl")
print("  - obesity_scaler.pkl")
print("  - obesity_target_encoder.pkl")
print("  - obesity_label_encoders.pkl")
print("  - obesity_features.pkl")

