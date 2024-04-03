import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import joblib

# Load the dataset
data_path = 'CESresampling.xlsx'  # Adjust path as needed
df = pd.read_excel(data_path)

# Load the model
model_path = r'C:\project\code\RandomOverSampling\CES_RANDOM_FOREST_MODEL.pkl'
rf_model = joblib.load(model_path)

# Prepare features for prediction
features = ['S_RANK', 'P_ID', 'R_ID', 'FEE_NAME', 'S_PARENT', 'GPA', 'GPA_MATCH', 'GPA_SCI']
X = df[features]

# Make predictions
predictions = rf_model.predict(X)

# Define the class names as strings
class_names = ['DCA', 'DCM', 'MTA', 'ITD', 'IMI']

# Map numeric predictions to branch names using the provided mapping
mapping = {0: 'DCA', 1: 'DCM', 2: 'MTA', 3: 'ITD', 4: 'IMI'}
predicted_branches_str = [mapping[pred] for pred in predictions]

# Assuming the actual labels are in a column named 'BRANCH'
# Convert actual labels to strings based on your mapping
# Ensure the 'BRANCH' column contains numeric labels corresponding to the mapping
actual_labels_str = [mapping[label] for label in df['BRANCH']]

# Calculate confusion matrix with string labels
conf_matrix_str = confusion_matrix(actual_labels_str, predicted_branches_str)

# Convert the confusion matrix to a DataFrame for a nicer display
conf_matrix_df = pd.DataFrame(conf_matrix_str, index=class_names, columns=class_names)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix_df)

# Plot confusion matrix using Seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Calculate metrics without specifying labels to include all present in the actual and predicted labels
accuracy = accuracy_score(actual_labels_str, predicted_branches_str)
recall = recall_score(actual_labels_str, predicted_branches_str, average='weighted')
precision = precision_score(actual_labels_str, predicted_branches_str, average='weighted')
f1 = f1_score(actual_labels_str, predicted_branches_str, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
