from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Example dataset
X = [[5], [10], [15], [20]]  # Features
y = ['A', 'A', 'B', 'B']     # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
prediction = model.predict([[12]])
print(prediction)  # Output: ['B']
