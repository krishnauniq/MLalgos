import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt 

# Example dataset
data = {'Experience': [1, 2, 3, 4, 5, 6], 'Salary': [30000, 35000, 40000, 45000, 50000, 55000]}
df = pd.DataFrame(data)

X = df[['Experience']]   # Features
y = df['Salary']         # Target


Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=0)

# Create a linear regression model
model = LinearRegression()
model.fit(Xtrain,ytrain)

y_pred = model.predict(Xtest)

# Plotting
# plt.scatter(X, y, color='blue', label='Actual')
# plt.plot(X, model.predict(X), color='red', label='Regression Line')
# plt.xlabel('Experience')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()

print("Predicted salary for 8 years experience: ₹", model.predict([[8]])[0])
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# 6. Print evaluation results
print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")