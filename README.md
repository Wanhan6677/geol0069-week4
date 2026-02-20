# geol0069-week4
Unsupervised learning helps us find patterns in data without using predefined labels. In this notebook, we focus on a practical workflow for classifying sea ice and leads in Earth observation data.

The notebook includes two main tasks:
1. Discrimination of Sea ice and lead based on image classification based on Sentinel-2 optical data. 
2. Discrimination of Sea ice and lead based on altimetry data classification based on Sentinel-3 altimetry data.

from google.colab import drive  # Mount your google drive to use Google Colab and interact directly
drive.mount('/content/drive')

Basic Code Implementation
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('K-means')
plt.show()
