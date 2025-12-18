import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Starting Lifestyle Score Model Training...")
print("="*60)

# 1. LOAD DATA
print("\n1. Loading data...")
df = pd.read_csv('../datasets/lifestyle_health.csv')
print(f"Dataset shape: {df.shape}")

# 2. PREPARE FEATURES AND TARGET
print("\n2. Preparing features and target...")
feature_cols = ['Physical_Activity', 'Nutrition_Score', 'Stress_Level', 
                'Mindfulness', 'Sleep_Hours', 'Hydration', 'BMI', 
                'Alcohol', 'Smoking']

X = df[feature_cols]
y = df['Overall_Health_Score']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target range: {y.min():.2f} - {y.max():.2f}")

# 3. TRAIN-TEST SPLIT
print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 4. FEATURE SCALING
print("\n4. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
with open('../trained_models/lifestyle_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved!")

# 5. TRAIN MODEL (XGBoost)
print("\n5. Training XGBoost Regressor...")
model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("Model trained!")

# 6. EVALUATE
print("\n6. Evaluating model...")
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Training metrics
train_r2 = r2_score(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)

# Testing metrics
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\nTRAINING METRICS:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  MAE: {train_mae:.4f}")

print("\nTESTING METRICS:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE: {test_mae:.4f}")

print("\nModel Performance:")
if test_r2 > 0.9:
    print("  ✓ Excellent! (R² > 0.9)")
elif test_r2 > 0.8:
    print("  ✓ Very Good! (R² > 0.8)")
elif test_r2 > 0.7:
    print("  ✓ Good! (R² > 0.7)")
else:
    print("  ⚠ Needs improvement (R² < 0.7)")

# 7. FEATURE IMPORTANCE
print("\n7. Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# 8. SAMPLE PREDICTIONS
print("\n8. Sample Predictions (First 5 test cases):")
comparison = pd.DataFrame({
    'Actual': y_test.values[:5],
    'Predicted': y_pred_test[:5],
    'Difference': y_test.values[:5] - y_pred_test[:5]
})
print(comparison)

# 9. SAVE MODEL
print("\n9. Saving model...")
with open('../trained_models/lifestyle_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save feature columns
with open('../trained_models/lifestyle_features.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)


print("MODEL TRAINING COMPLETE!")

print("\nSaved files:")
print("  - lifestyle_model.pkl")
print("  - lifestyle_scaler.pkl")
print("  - lifestyle_features.pkl")
