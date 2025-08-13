import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

ypred = model.predict(Xtest)

# Plotting
# plt.scatter(X, y, color='blue', label='Actual')
# plt.plot(X, model.predict(X), color='red', label='Regression Line')
# plt.xlabel('Experience')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()

print("Predicted salary for 8 years experience: ₹", model.predict([[8]])[0])
