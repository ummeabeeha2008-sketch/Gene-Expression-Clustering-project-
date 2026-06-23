"""
Gene Expression Clustering Project
------------------------------------
Goal: Use unsupervised machine learning to group samples based on their
measurements (acting as a stand-in for gene expression values), WITHOUT
telling the algorithm the correct answer. Then check if the groups it
finds on its own match reality.

Dataset: Breast Cancer Wisconsin Dataset (same as before)
This time we IGNORE the malignant/benign labels during clustering, and
only use them afterward to check our results.
"""

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 2: Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)  # the "expression-like" measurements
y_true = data.target  # real labels (we'll only use this to CHECK our clustering later)

print("Dataset shape:", X.shape)

# Step 3: Scale the data
# Clustering is very sensitive to scale, so this step matters a lot here
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Reduce to 2 dimensions using PCA (for visualization only)
# We have 30 features - PCA compresses them into 2 "summary" dimensions
# so we can actually plot and see the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nVariance explained by 2 PCA components:", pca.explained_variance_ratio_)

# Step 5: K-Means Clustering
# Tell it to find 2 groups (we're guessing 2 since we expect malignant/benign-like split)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Step 6: Compare K-Means clusters to the real labels
comparison = pd.DataFrame({"KMeans Cluster": kmeans_labels, "Actual Label": y_true})
print("\n--- K-Means vs Actual Labels ---")
print(pd.crosstab(comparison["KMeans Cluster"], comparison["Actual Label"]))

# Step 7: Visualize K-Means clusters using PCA-reduced data
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", alpha=0.6)
plt.title("K-Means Clustering (PCA-reduced view)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.savefig("kmeans_clusters.png")
plt.close()
print("\nSaved K-Means visualization as kmeans_clusters.png")

# Step 8: Hierarchical Clustering (a different clustering approach)
hierarchical = AgglomerativeClustering(n_clusters=2)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

comparison_h = pd.DataFrame({"Hierarchical Cluster": hierarchical_labels, "Actual Label": y_true})
print("\n--- Hierarchical Clustering vs Actual Labels ---")
print(pd.crosstab(comparison_h["Hierarchical Cluster"], comparison_h["Actual Label"]))

# Step 9: Plot a Dendrogram (visual "tree" showing how samples group together)
# Using only the first 30 samples so the diagram stays readable
linked = linkage(X_scaled[:30], method="ward")
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram (first 30 samples)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.savefig("dendrogram.png")
plt.close()
print("Saved dendrogram as dendrogram.png")

print("\nDone! Check the folder for kmeans_clusters.png and dendrogram.png")