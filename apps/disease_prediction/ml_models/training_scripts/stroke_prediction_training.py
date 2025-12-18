# import pandas as pd
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
# from pathlib import Path

# DATASET_PATH = '../datasets/stroke_prediction.csv'
# MODEL_DIR = '../trained_models'

# Path(MODEL_DIR).mkdir(exist_ok=True)

# df = pd.read_csv(DATASET_PATH)

# if 'id' in df.columns:
#     df = df.drop('id', axis=1)

# categorical_cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
# df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# X = df_encoded.drop('stroke', axis=1)
# y = df_encoded['stroke']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# with open(f'{MODEL_DIR}/stroke_scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# print("Original dataset shape:", y_train.value_counts())
# print("Using class_weight='balanced' to handle imbalance")

# model = LogisticRegression(class_weight='balanced', max_iter=500, random_state=42)
# model.fit(X_train_scaled, y_train)

# y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
# best_threshold = 0.5
# best_f1 = 0

# for threshold in thresholds:
#     y_pred_adj = (y_pred_proba >= threshold).astype(int)
    
#     acc = accuracy_score(y_test, y_pred_adj)
#     prec = precision_score(y_test, y_pred_adj, zero_division=0)
#     rec = recall_score(y_test, y_pred_adj)
#     f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = threshold
    
#     print(f"\nThreshold: {threshold}")
#     print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
#     print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_adj)}")

# print(f"BEST THRESHOLD: {best_threshold} (F1-Score: {best_f1:.4f})")

# final_y_pred = (y_pred_proba >= best_threshold).astype(int)

# print("\nFINAL MODEL PERFORMANCE WITH THRESHOLD", best_threshold)
# print("="*60)
# print("Accuracy:", round(accuracy_score(y_test, final_y_pred), 4))
# print("\nClassification Report:\n", classification_report(y_test, final_y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, final_y_pred))

# metadata = {
#     'model_type': 'LogisticRegression',
#     'accuracy': float(accuracy_score(y_test, final_y_pred)),
#     'best_threshold': best_threshold,
#     'feature_names': X.columns.tolist()
# }

# with open(f'{MODEL_DIR}/stroke_model_metadata.pkl', 'wb') as f:
#     pickle.dump(metadata, f)

# with open(f'cd{MODEL_DIR}/stroke_prediction_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print(f"\nModel saved to {MODEL_DIR}/stroke_prediction_model.pkl")
# print(f"Metadata saved to {MODEL_DIR}/stroke_model_metadata.pkl")
# print(f"Note: For predictions, use threshold = {best_threshold}")

import pickle

with open('../trained_models/stroke_model_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("Metadata:")
print(metadata)