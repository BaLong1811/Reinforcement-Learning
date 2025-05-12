import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load the data
train_data = pd.read_csv('dungeon_sensorstats.csv')

# Select features
features = train_data[['height', 'weight']]

# KMeans Clustering
kmeans = KMeans(n_clusters=4)
kmeans_clusters = kmeans.fit_predict(features)

# Gaussian Clustering (Gaussian Mixture Models)
gmm = GaussianMixture(n_components=4)
gmm_clusters = gmm.fit_predict(features)

# Plotting the results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# KMeans plot

plt.rcParams.update({'font.size': 18})
axes[0].scatter(features['height'], features['weight'], c=kmeans_clusters, cmap='viridis')
axes[0].set_title('KMeans Clustering')
axes[0].set_xlabel('Height')
axes[0].set_ylabel('Weight')

# Gaussian Clustering plot
axes[1].scatter(features['height'], features['weight'], c=gmm_clusters, cmap='viridis')
axes[1].set_title('Gaussian Clustering')
axes[1].set_xlabel('Height')
axes[1].set_ylabel('Weight')

plt.tight_layout()
plt.show()

#### Justification for the number of clusters ####

# Determine the optimal number of clusters for KMeans
kmeans_inertia = []
silhouette_scores = []

cluster_range = range(2, 10)  # Test cluster numbers from 2 to 9
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(features)
    kmeans_inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(features, clusters))

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
    gmm.fit(features)
    bic_scores.append(gmm.bic(features))
    aic_scores.append(gmm.aic(features))

# Plot BIC and AIC for GMM
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, bic_scores, 'r-s', label='BIC')
plt.plot(cluster_range, aic_scores, 'm-d', label='AIC')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.title('Optimal Number of Clusters for GMM')
plt.legend(loc='best')
plt.show()
