import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')
print(df.head())

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(df['area'], df['price'], color='red', marker='+')
plt.show() 

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])

prediction = reg.predict([[3300]])
print(f"Predicted price for 3300 sq.ft area: {prediction[0]}")
