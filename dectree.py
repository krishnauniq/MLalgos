from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = [[25],[45],[65],[85],[105]]
y = [0,0,0,1,1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("will ??",model.predict([[155]]))

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
tree.plot_tree(model, feature_names=["Age"], class_names=["No", "Yes"], filled=True)
plt.show()
