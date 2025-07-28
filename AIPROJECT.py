import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


# STEP 1: Load and preprocess data
print("Loading and preprocessing data...")

# Load your dataset
df = pd.read_csv("student-data.csv")

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


# STEP 4: Plot Feature Importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()


# STEP 5: Predict new students (automation task)
def predict_new_students(filename):
    print(f"\nPredicting new students from '{filename}'...")

    new_df = pd.read_csv(filename)

    # Encode using the same encoders
    for col in new_df.columns:
        if col in label_encoders:
            le = label_encoders[col]
            new_df[col] = new_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    predictions = model.predict(new_df)
    result_df = new_df.copy()
    result_df["Predicted_Passed"] = predictions
    result_df["Predicted_Passed"] = result_df["Predicted_Passed"].apply(lambda x: "yes" if x == 1 else "no")

    result_df.to_csv("prediction_output.csv", index=False)
    print("Predictions saved to 'prediction_output.csv'.")

