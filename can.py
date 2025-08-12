import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df= pd.read_csv('canada_per_capita_income.csv')
print(df.head())

plt.xlabel('year')
plt.ylabel('per capita income (US$)')
plt.title('Canada Per Capita Income Over Years')
plt.scatter(df['year'], df['per capita income (US$)'], color='blue', marker='o')
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['year']], df['per capita income (US$)'])

predicted_income = reg.predict([[2020]])
print(f"Predicted per capita income for 2020: ${predicted_income[0]:,.2f}")

z=pd.get_dummies('areas.csv')
print(z.head())