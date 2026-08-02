import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load dataset
data = load_digits()
X = data.data
y = data.target

# Standardize data
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
std[std == 0] = 1

X_standard = (X - mean) / std

# Covariance matrix
cov_matrix = np.cov(X_standard.T)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues
sorted_index = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Select first two components
principal_components = sorted_eigenvectors[:, :2]

# Transform data
X_pca = np.dot(X_standard, principal_components)

# Plot
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA from Scratch")
plt.show()