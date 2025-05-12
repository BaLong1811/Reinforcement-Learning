import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


from sklearn.metrics import silhouette_score
import numpy as np

# Load the data
train_data = pd.read_csv('dungeon_sensorstats_partC.csv')

# Select features
features = train_data[['height', 'strength']]

# # Convert 'alignment' from string to numeric values
# alignment_mapping = {'good': 1, 'neutral': 0, 'evil': -1}
# features['alignment'] = features['alignment'].map(alignment_mapping)

# Check for NaN values
print(features.isnull().sum())  # This will show you how many NaNs are in each column

# Option 1: Drop rows with NaN values
features_cleaned = features.dropna()

# # Option 2: Fill NaN values with a default value (e.g., mean of the column)
# features_cleaned = features.fillna(features.mean())

# KMeans Clustering
kmeans = KMeans(n_clusters=3)
kmeans_clusters = kmeans.fit_predict(features_cleaned)

# Gaussian Clustering (Gaussian Mixture Models)
gmm = GaussianMixture(n_components=5)
gmm_clusters = gmm.fit_predict(features_cleaned)

# Plotting the results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# KMeans plot
plt.rcParams.update({'font.size': 18})
axes[0].scatter(features_cleaned['height'], features_cleaned['strength'], c=kmeans_clusters, cmap='viridis')
axes[0].set_title('KMeans Clustering')
axes[0].set_xlabel('Height', fontsize=18)
axes[0].set_ylabel('Strength', fontsize=18)

# Gaussian Clustering plot
axes[1].scatter(features_cleaned['height'], features_cleaned['strength'], c=gmm_clusters, cmap='viridis')
axes[1].set_title('Gaussian Clustering')
axes[1].set_xlabel('Height', fontsize=18)
axes[1].set_ylabel('Strength', fontsize=18)

plt.tight_layout()
plt.show()

# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(features_cleaned)

# # Plot for K-Means
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=kmeans_clusters, cmap='viridis', label='K-Means')
# plt.title('K-Means Clusters')
# plt.legend()
# plt.show()

# # Plot for GMM
# plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=gmm_clusters, cmap='viridis', label='GMM')
# plt.title('GMM Clusters')
# plt.legend()
# plt.show()


#### Justification for the number of clusters ####

# Determine the optimal number of clusters for KMeans
kmeans_inertia = []
silhouette_scores = []

cluster_range = range(2, 10)  # Test cluster numbers from 2 to 9
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features_cleaned)
    kmeans_inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features_cleaned, clusters))

# Plot Elbow Method and Silhouette Score for KMeans
plt.rcParams.update({'font.size': 18})
fig, ax1 = plt.subplots(figsize=(8, 6))
ax2 = ax1.twinx()
ax1.plot(cluster_range, kmeans_inertia, 'g-^', label='Inertia (Elbow Method)')
ax2.plot(cluster_range, silhouette_scores, 'b-o', label='Silhouette Score')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia', color='g')
ax2.set_ylabel('Silhouette Score', color='b')
ax1.set_title('Optimal Number of Clusters for KMeans')
plt.legend(loc='best')
plt.show()

# Determine the optimal number of clusters for GMM
bic_scores = []
aic_scores = []

for k in cluster_range:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(features_cleaned)
    bic_scores.append(gmm.bic(features_cleaned))
    aic_scores.append(gmm.aic(features_cleaned))

# Plot BIC and AIC for GMM
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, bic_scores, 'r-s', label='BIC')
plt.plot(cluster_range, aic_scores, 'm-d', label='AIC')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.title('Optimal Number of Clusters for GMM')
plt.legend(loc='best')
plt.show()


