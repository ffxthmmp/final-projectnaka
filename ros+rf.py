import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


# Correct the file path if necessary and ensure it points to the right file format
df = pd.read_excel("resampled_dataset.xlsx")  # Corrected function for reading an Excel file

# Separate features and target
X = df.drop('BRANCH', axis=1)
y = df['BRANCH']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=17)

# Enhanced Hyperparameter Grid
param_grid = {
    'n_estimators': [10,20, 30,40, 50, 100, 150, 200, 300],
    'max_depth': [10, 20, 30,40, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-2, verbose=2, scoring='accuracy')

# Training
grid_search.fit(X_train, y_train)

# Best Parameters and Accuracy
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)

# Prediction and Classification Report
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))