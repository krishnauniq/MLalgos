# # Logistic Regression Example

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt

# # Step 1: Create the dataset
# data = {
#     'Age': [22, 25, 47, 52, 46, 56, 23, 24, 28, 33, 45, 60],
#     'Purchased': [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
# }
# df = pd.DataFrame(data)

# # Step 2: Prepare features and labels
# X = df[['Age']]  # 2D DataFrame
# y = df['Purchased']  # 1D Series

# # Step 3: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Step 4: Create and train the model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Step 5: Make predictions
# y_pred = model.predict(X_test)

# # Step 6: Evaluate the model
# print("Test Set Predictions:", y_pred)
# print("Actual Values:", y_test.values)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# # Step 7: Predict for a new user
# new_age = 30
# result = model.predict([[new_age]])[0]
# print(f"Will a {new_age}-year-old purchase? {'Yes' if result == 1 else 'No'}")

# # Step 8: Plotting the sigmoid curve (optional visualization)
# import numpy as np
# ages = np.linspace(20, 65, 100).reshape(-1, 1)
# probabilities = model.predict_proba(ages)[:, 1]

# plt.plot(ages, probabilities, color='red', label='Sigmoid Curve')
# plt.scatter(X, y, color='blue', label='Data Points')
# plt.xlabel("Age")
# plt.ylabel("Probability of Purchase")
# plt.title("Logistic Regression - Age vs Purchase")
# plt.legend()
# plt.grid(True)
# plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Example dataset
import pandas as pd

data = {'Age': [22, 25, 47, 52, 46, 56],
        'Purchased': [0, 0, 1, 1, 1, 1]}
df = pd.DataFrame(data)

X = df[['Age']]             # Features
y = df['Purchased']         # Target (0 or 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Will a 30-year-old purchase?", model.predict([[30]]))

