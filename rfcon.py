import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel("resampled_dataset.xlsx")

# Separate features and target
X = df[['S_RANK', 'P_ID', 'R_ID', 'FEE_NAME', 'S_PARENT', 'GPA', 'GPA_MATCH', 'GPA_SCI']]
y = df['BRANCH']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(bootstrap=False, n_estimators=300, max_depth=30, min_samples_leaf=1, min_samples_split=2, random_state=17)
rf.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = rf.predict(X_test)

# Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Adjust for multi-class
recall = recall_score(y_test, y_pred, average='weighted')  # Adjust for multi-class
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
