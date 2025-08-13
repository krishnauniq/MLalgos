from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X,y = iris.data, iris.target

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

bagm =RandomForestClassifier(
    n_estimators = 10,
    max_features='sqrt',
    random_state = 42
)

bagm.fit(X_train,y_train)

y_pred = bagm.predict(X_test)

print("Accuracy score :", accuracy_score(y_test,y_pred))