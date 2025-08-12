from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data (n_samples x n_features)
# Why standardize? 
# PCA is sensitive to the scale of data. Features with bigger ranges will dominate.
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2, 1.6],
              [1, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# Standardize data

# StandardScaler subtracts the mean and divides by standard deviation for each feature.
# After scaling, each feature will have mean ≈ 0 and variance = 1.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA

# fit_transform:
# fit calculates principal components from the scaled data.
# transform projects the data onto the principal component(s), producing the new reduced dataset.


pca = PCA(n_components=1)  # reduce to 1D
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Principal components:\n", pca.components_)
print("Transformed data:\n", X_pca)
