# Import libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'decision_tree_model.joblib')  # consistent name

# Load the model
loaded_model = joblib.load('decision_tree_model.joblib')

# Test prediction
print("Joblib prediction:", loaded_model.predict([[5.1, 3.5, 1.4, 0.2]]))
