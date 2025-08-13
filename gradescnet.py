from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3️⃣ Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(
    n_estimators=50,      # number of boosting stages
    learning_rate=0.1,    # step size shrinkage
    max_depth=3,          # depth of each tree
    random_state=42
)

# 4️⃣ Train
gb_model.fit(X_train, y_train)

# 5️⃣ Predict
y_pred = gb_model.predict(X_test)

# 6️⃣ Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
