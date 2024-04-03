
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Load the dataset
df = pd.read_excel("resampled_dataset.xlsx")

# Separate features and target
X = df[['S_RANK', 'P_ID', 'R_ID',  'FEE_NAME', 'S_PARENT', 'GPA', 'GPA_MATCH', 'GPA_SCI']]
y = df['BRANCH']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20, 30, 100, 150],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_leaf_nodes': [None, 3, 5, 10, 20, 30]
}

# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=17)

# Setup the GridSearchCV object
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, n_jobs=-1)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Get the best accuracy score
print("Best accuracy achieved: {:.2f}%".format(grid_search.best_score_ * 100))

# Optional: Use the best estimator for further predictions
# best_clf = grid_search.best_estimator_
# predictions = best_clf.predict(X_test)
