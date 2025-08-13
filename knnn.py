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

# 📊 Example (Classification):

# Suppose you want to classify whether a fruit is an "Apple" or "Orange" based on features like weight and color:
#    •	If K = 3, and among the 3 nearest fruits to your test point, 
#       2 are apples and 1 is orange → classify it as Apple.

# ✅ Advantages:
# Simple and intuitive
# No training time (lazy learner)
# Works well with small datasets

# ❌ Disadvantages:
# Slow at prediction for large datasets
# Sensitive to irrelevant features or noise
# Needs proper scaling (since it's distance-based)