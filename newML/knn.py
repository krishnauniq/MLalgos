import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Create a simple dataset
data_knn = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'feature2': [1, 2, 1, 2, 3, 3, 4, 4, 5, 5],
    'target': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
})

model = KNeighborsClassifier(n_neighbors=3)
# Define features (X) and target (y)
X_knn = data_knn[['feature1', 'feature2']]
y_knn = data_knn['target']
model.fit(X_knn, y_knn)

# plot points and decision boundary
fig, ax = plt.subplots(figsize=(8, 4))

data_knn.plot(kind='scatter', x='feature1', y='feature2', c='target', 
              colormap='viridis', ax=ax, s=100, alpha=0.7)

# Create a grid to plot the decision boundary
x_min, x_max = data_knn['feature1'].min() - 1, data_knn['feature1'].max() + 1
y_min, y_max = data_knn['feature2'].min() - 1, data_knn['feature2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Predict on the grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
# Reshape the predictions back to the grid shape
Z = Z.reshape(xx.shape)
# Plot the decision boundary
ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
ax.set_title('K-Nearest Neighbors Decision Boundary')
plt.show()