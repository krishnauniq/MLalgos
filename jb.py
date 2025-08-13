# Import libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import jb

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save the model
jb.dump(model, 'decision_tree_model.joblib')  # consistent name

# Load the model
loaded_model = jb.load('decision_tree_model.joblib')

# Test prediction
print("Joblib prediction:", loaded_model.predict([[5.1, 3.5, 1.4, 0.2]]))
