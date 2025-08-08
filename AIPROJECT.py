import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import requests

# STEP 1: Load and preprocess data
print("Loading and preprocessing data...")

# Load your dataset
df = pd.read_csv("student-data.csv")

# Save original columns to preserve before encoding (for example email, name)
original_cols = {}
for col in ['email', 'name']:
    if col in df.columns:
        original_cols[col] = df[col].copy()

# Encode all categorical columns
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Features and target
X = df.drop("passed", axis=1)
y = df["passed"]  # Already label-encoded

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 2: Train the model
print("Training Decision Tree model...")
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# STEP 3: Evaluate the model
print("Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Prediction summary
total = len(y_pred)
passed = np.sum(y_pred == 1)
failed = np.sum(y_pred == 0)

pass_percent = (passed / total) * 100
fail_percent = (failed / total) * 100

print(f"\nPrediction Summary:")
print(f"Total Students Predicted: {total}")
print(f"Passed: {passed} students ({pass_percent:.2f}%)")
print(f"Failed: {failed} students ({fail_percent:.2f}%)")

# STEP 4: Send failing students to n8n
import json

failed_indices = np.where(y_pred == 0)[0]

# Extract failing students' data from X_test
failing_students = X_test.iloc[failed_indices].copy()

# If you want to add back original email/name, map from original_cols if available
for col in ['email', 'name']:
    if col in original_cols:
        # Get the indexes of test data relative to original df
        # Because train_test_split shuffles indexes, get original indexes by matching
        # Here we do a crude approach by index alignment of X_test to original df rows
        # (This assumes the index wasn’t reset)
        failing_students[col] = original_cols[col].iloc[failing_students.index]

# Convert to dictionary list for JSON sending
payload = failing_students.reset_index(drop=True).to_dict(orient="records")

n8n_webhook_url = "https://mmss-123.app.n8n.cloud/webhook-test/reminder"

print(f"Sending {len(payload)} failing students to n8n...")

try:
    response = requests.post(n8n_webhook_url, json=payload)
    response.raise_for_status()
    print("Successfully sent data to n8n!")
except requests.exceptions.RequestException as e:
    print("Failed to send data to n8n:", e)

# STEP 5: Plot Feature Importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
