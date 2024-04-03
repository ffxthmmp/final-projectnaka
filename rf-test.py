import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.metrics import f1_score

# Load the data
df = pd.read_excel("CESresampling.xlsx")

# Separate features and target
X = df[['S_RANK', 'P_ID', 'R_ID', 'FEE_NAME', 'S_PARENT', 'GPA', 'GPA_MATCH', 'GPA_SCI']]
y = df['BRANCH']


# Splitting the transformed data for modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Load the saved RandomForest model from the file
model_path = "CES_RANDOM_FOREST_MODEL.joblib"
loaded_rf = load(model_path)

# Now, you can use the loaded model to make predictions
y_pred = loaded_rf.predict(X_test)
y_pred_train = loaded_rf.predict(X_train)

# Optionally, if you have the true labels for your test data, you can evaluate the model
accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, y_pred_train)
precision = precision_score(y_test, y_pred, average='weighted')  # Adjust for multi-class
recall = recall_score(y_test, y_pred, average='weighted')  # Adjust for multi-class
f1 = f1_score(y_test, y_pred, average='weighted')  # Adjust for multi-class

print(f"Accuracy of the loaded model: {accuracy}")
print(f"Train Accuracy of the loaded model: {train_accuracy}")
print(f"Precision of the loaded model: {precision}")
print(f"Recall of the loaded model: {recall}")
print(f"F1 Score of the loaded model: {f1}")