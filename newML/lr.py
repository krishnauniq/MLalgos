# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # tell sklearn to output Pandas DataFrames
# from sklearn import set_config
# set_config(transform_output="pandas")

# # Generate a synthetic dataset
# np.random.seed(42)
# num_samples = 100
# X = 2 * np.random.rand(num_samples, 1)  # A single feature
# y = 4 + 3 * X + np.random.randn(num_samples, 1)  # Linear relationship with some noise

# # Create a DataFrame
# df = pd.DataFrame({'feature': X.flatten(), 'target': y.flatten()})

# print("First 5 rows of the synthetic dataset:")
# print(df.head())

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(figsize=(8,6))
# df.plot(kind='scatter',x='feature',y='target',ax=ax,color='blue',alpha=0.5)
# plt.title('Synthetic Dataset : Feature vs Target')
# plt.show()

# X=df[['feature']]
# y=df['target']
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
# print(f'train x dataset :{len(X_train)} samples')
# print(f'train y dataset :{len(y_test)} samples')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import set_config

# Optional: make sklearn outputs prettier
set_config(transform_output="pandas")

# 1. Generate synthetic dataset
np.random.seed(42)
num_samples = 100
X = 2 * np.random.rand(num_samples, 1)
y = 4 + 3 * X + np.random.randn(num_samples, 1)

# 2. Create DataFrame
df = pd.DataFrame({'feature': X.flatten(), 'target': y.flatten()})
print("First 5 rows of the synthetic dataset:")
print(df.head())

# 3. Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# 4. Predict
y_pred = model.predict(X)

# 5. Evaluate the model
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
