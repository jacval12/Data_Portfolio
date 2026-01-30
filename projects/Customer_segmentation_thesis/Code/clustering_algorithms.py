# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:59:21 2025

@author: ASUS
"""
 
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import skfuzzy as fuzz


   
def kmeans_analysis_and_plot(features, dataset_label="RFM", show_plots=True):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    ks = range(2, 15)
    wcss = []
    silhouette_scores = []
    calinski_scores = []
    dbi_scores = []
    bss_values = []

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)

        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, labels))
        calinski_scores.append(calinski_harabasz_score(scaled_features, labels))
        dbi_scores.append(davies_bouldin_score(scaled_features, labels))

        total_ss = np.sum((scaled_features - np.mean(scaled_features, axis=0)) ** 2)
        bss = total_ss - kmeans.inertia_
        bss_values.append(bss)

    if show_plots:
        plt.figure(figsize=(16, 10))
        plt.suptitle(f"K-Means algorithm ({dataset_label})")

        plt.subplot(2, 3, 1)
        plt.plot(ks, wcss, marker='o')
        plt.title(f'Elbow Method (WCSS) for {dataset_label}')
        plt.xlabel('k')
        plt.ylabel('WCSS')

        plt.subplot(2, 3, 2)
        plt.plot(ks, silhouette_scores, marker='o', color='green')
        plt.title(f'Silhouette Score for {dataset_label}')
        plt.xlabel('k')
        plt.ylabel('Score')
 
        plt.subplot(2, 3, 3)
        plt.plot(ks, calinski_scores, marker='o', color='orange')
        plt.title(f'Calinski-Harabasz Index for {dataset_label}')
        plt.xlabel('k')
        plt.ylabel('Score')

        plt.subplot(2, 3, 4)
        plt.plot(ks, dbi_scores, marker='o', color='red')
        plt.title(f'Davies-Bouldin Index for {dataset_label}')
        plt.xlabel('k')
        plt.ylabel('Score (lower is better)')

        plt.subplot(2, 3, 5)
        plt.plot(ks, bss_values, marker='o', color='purple')
        plt.title(f'BSS for {dataset_label}')
        plt.xlabel('k')
        plt.ylabel('BSS')

        plt.tight_layout()
        plt.show()

    silhouette_array = np.array(silhouette_scores)
    optimal_k = ks[np.argmax(silhouette_array)]

    for i in range(1, len(silhouette_array) - 1):
        if silhouette_array[i] > silhouette_array[i - 1] and silhouette_array[i] > silhouette_array[i + 1]:
            optimal_k = ks[i]
            break
 
    print(f"Selected optimal k: {optimal_k}")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(scaled_features)

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)
    explained_variance = pca.explained_variance_ratio_.sum()
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)

    if show_plots:
        plt.figure(figsize=(8, 6))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=final_labels, cmap='viridis', s=50)
        plt.colorbar(label='Cluster Label')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        cmap = cm.get_cmap('viridis', optimal_k)
        handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(optimal_k)]
        plt.legend(handles=handles, title='Clusters')
        plt.title(f'2D Clusters (k={optimal_k}) - KMeans on {dataset_label} with PCA')
        plt.show()

    metrics_df = pd.DataFrame({
        "k": list(ks),
        "WCSS": wcss,
        "Silhouette_score": silhouette_scores,
        "Calinski_harabasz_score": calinski_scores,
        "Davies_bouldin_score(DBI)": dbi_scores,
        "BSS": bss_values
    })

    pca_info = {
        "explained_variance_ratio_sum": explained_variance,
        "loadings": loadings
    }

    return final_labels, optimal_k, metrics_df, pca_info


 

def hierarchical_analysis_and_plot(features, dataset_label="RFMD", show_plots=True):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    ks = range(2, 15)
    wcss_list = []
    silhouette_list = []
    calinski_list = []
    dbi_list = []
    bss_list = []

    total_ss = np.sum((scaled_features - np.mean(scaled_features, axis=0)) ** 2)

    Z = linkage(scaled_features, method='ward')

    for k in ks:
        labels = fcluster(Z, k, criterion='maxclust')
        centroids = np.array([scaled_features[labels == i].mean(axis=0) for i in np.unique(labels)])
        wcss = 0
        for i in np.unique(labels):
            cluster_points = scaled_features[labels == i]
            wcss += np.sum((cluster_points - centroids[i - 1]) ** 2)
        bss = total_ss - wcss

        wcss_list.append(wcss)
        silhouette_list.append(silhouette_score(scaled_features, labels))
        calinski_list.append(calinski_harabasz_score(scaled_features, labels))
        dbi_list.append(davies_bouldin_score(scaled_features, labels))
        bss_list.append(bss)

    silhouette_array = np.array(silhouette_list)
    optimal_k = ks[np.argmax(silhouette_array)]

    for i in range(1, len(silhouette_array) - 1):
        if silhouette_array[i] > silhouette_array[i - 1] and silhouette_array[i] > silhouette_array[i + 1]:
            optimal_k = ks[i]
            break

    print(f"Selected optimal k: {optimal_k}")

    final_labels = fcluster(Z, optimal_k, criterion='maxclust')

    # PCA for plotting and explained variance
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)
    explained_variance = pca.explained_variance_ratio_.sum()
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)

    if show_plots:
        plt.figure(figsize=(16, 10))
        plt.suptitle(f"Hierarchical Clustering Evaluation Metrics on {dataset_label}", fontsize=16)

        plt.subplot(2, 3, 1)
        plt.plot(ks, wcss_list, marker='o')
        plt.title('Elbow Method (WCSS)')
        plt.xlabel('k')
        plt.ylabel('WCSS')

        plt.subplot(2, 3, 2)
        plt.plot(ks, silhouette_list, marker='o', color='green')
        plt.title('Silhouette Score')
        plt.xlabel('k')
        plt.ylabel('Score')

        plt.subplot(2, 3, 3)
        plt.plot(ks, calinski_list, marker='o', color='orange')
        plt.title('Calinski-Harabasz Index')
        plt.xlabel('k')
        plt.ylabel('Score')

        plt.subplot(2, 3, 4)
        plt.plot(ks, dbi_list, marker='o', color='red')
        plt.title('Davies-Bouldin Index')
        plt.xlabel('k')
        plt.ylabel('Score (lower is better)')

        plt.subplot(2, 3, 5)
        plt.plot(ks, bss_list, marker='o', color='purple')
        plt.title('Between-Group Sum of Squares (BSS)')
        plt.xlabel('k')
        plt.ylabel('BSS')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        plt.figure(figsize=(12, 6))
        dendrogram(Z, truncate_mode='level', p=5)
        plt.axhline(y=Z[-(optimal_k - 1), 2], color='r', linestyle='--', label=f'k = {optimal_k}')
        plt.title(f'Dendrogram for {dataset_label} Data')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=final_labels, cmap='viridis', s=50)
        plt.colorbar(label='Cluster Label')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        cmap = cm.get_cmap('viridis', optimal_k)
        handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(optimal_k)]
        plt.legend(handles=handles, title='Clusters')
        plt.title(f'2D Clusters (k={optimal_k}) - Hierarchical on {dataset_label} (Ward + PCA)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    metrics_df = pd.DataFrame({
        "k": list(ks),
        "WCSS": wcss_list,
        "Silhouette_score": silhouette_list,
        "Calinski_harabasz_score": calinski_list,
        "Davies_bouldin_score(DBI)": dbi_list,
        "BSS": bss_list
    })

    pca_info = {
        "explained_variance_ratio_sum": explained_variance,
        "loadings": loadings
    }

    return final_labels, optimal_k, metrics_df, pca_info





"""
Fuzzy C-Means clustering code inspired by:

@misc{geeksforgeeks_fuzzyclustering,
  author       = {GeeksforGeeks},
  title        = {ML | Fuzzy clustering},
  year         = {2021},
  url          = {https://www.geeksforgeeks.org/ml-fuzzy-clustering/},
  note         = {Accessed: 2025-05-16}
}

MADE BY CHATGPT
"""



def fuzzy_cmeans_analysis_and_plot(features, dataset_label="RFM", show_plots=True):
    from itertools import product
    from sklearn.decomposition import PCA

    def fuzzy_entropy(u):
        u_safe = np.clip(u, 1e-10, 1)
        entropy = -np.sum(u_safe * np.log(u_safe)) / u.shape[1]
        return entropy

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features_t = scaled_features.T

    ks = range(2, 15)
    m_values = [1.5, 2.0, 2.5]
    error_values = [0.001, 0.005]
    maxiter_values = [500, 1000]

    final_labels = None
    optimal_k = None
    best_params = None
    best_membership = None
    metrics_list = []

    candidate_models = []

    for m, error, maxiter in product(m_values, error_values, maxiter_values):
        for k in ks:
            cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
                scaled_features_t, c=k, m=m, error=error, maxiter=maxiter, init=None
            )
            cluster_labels = np.argmax(u, axis=0)
            silhouette = silhouette_score(scaled_features, cluster_labels)
            calinski = calinski_harabasz_score(scaled_features, cluster_labels)
            dbi = davies_bouldin_score(scaled_features, cluster_labels)
            entropy = fuzzy_entropy(u)

            metrics = {
                "k": k,
                "m": m,
                "error": error,
                "maxiter": maxiter,
                "FPC": fpc,
                "Silhouette_score": silhouette,
                "Calinski_harabasz_score": calinski,
                "Davies_bouldin_score(DBI)": dbi,
                "Entropy": entropy
            }
            metrics_list.append(metrics)

            candidate_models.append(
                (fpc, -entropy, silhouette, calinski, -dbi, k, m, error, maxiter, cluster_labels, u)
            )

    candidate_models.sort(reverse=True)

    for fpc, neg_entropy, silhouette, calinski, neg_dbi, k, m, error, maxiter, cluster_labels, u in candidate_models:
        if k != 2:
            optimal_k = k
            best_params = (m, error, maxiter)
            final_labels = cluster_labels
            best_membership = u
            break

    if optimal_k is None:
        fpc, neg_entropy, silhouette, calinski, neg_dbi, k, m, error, maxiter, cluster_labels, u = candidate_models[0]
        optimal_k = k
        best_params = (m, error, maxiter)
        final_labels = cluster_labels
        best_membership = u

    print(f"Best parameters found: m={best_params[0]}, error={best_params[1]}, maxiter={best_params[2]}")
    print(f"Best k (excluding k=2 if possible): {optimal_k}")

    metrics_df = pd.DataFrame(metrics_list)

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)
    explained_variance = pca.explained_variance_ratio_.sum()
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)

    if show_plots:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.cm as cm

        plt.figure(figsize=(12, 5))

        best_m, best_error, best_maxiter = best_params
        filtered = metrics_df[
            (metrics_df["m"] == best_m) &
            (metrics_df["error"] == best_error) &
            (metrics_df["maxiter"] == best_maxiter)
        ].sort_values(by="k")

        plt.subplot(1, 2, 1)
        plt.plot(filtered["k"], filtered["FPC"], marker='o', label='FPC')
        plt.axvline(optimal_k, color='red', linestyle='--', label=f'Chosen k = {optimal_k}')
        plt.title("FPC vs Number of clusters k")
        plt.xlabel("Number of clusters k")
        plt.ylabel("FPC (higher better)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(filtered["k"], filtered["Entropy"], marker='o', color='purple', label='Entropy')
        plt.axvline(optimal_k, color='red', linestyle='--', label=f'Chosen k = {optimal_k}')
        plt.title("Entropy vs Number of clusters k")
        plt.xlabel("Number of clusters k")
        plt.ylabel("Entropy (lower better)")
        plt.legend()
        plt.suptitle(f"FPC and Entropy vs k (best params: m={best_m}, error={best_error}, maxiter={best_maxiter}, dataset: {dataset_label})")
        plt.tight_layout()
        plt.show()

        n_clusters = optimal_k
        cmap = cm.get_cmap('viridis', n_clusters)

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        axs[0].scatter(features_2d[:, 0], features_2d[:, 1],
                       c=final_labels, cmap='viridis', s=50)
        handles = [mpatches.Patch(color=cmap(i), label=f'Cluster {i}') for i in range(n_clusters)]
        axs[0].legend(handles=handles, title='Clusters')
        axs[0].set_title(f'Fuzzy C-Means (k={n_clusters}) - Hard Clustering (PCA 2D)')
        axs[0].set_xlabel('PCA Component 1')
        axs[0].set_ylabel('PCA Component 2')
        axs[0].grid(True)

        scatter = axs[1].scatter(features_2d[:, 0], features_2d[:, 1],
                                 c=best_membership[0], cmap='viridis', s=50)
        fig.colorbar(scatter, ax=axs[1], label='Membership to Cluster 0')
        axs[1].set_title('Fuzzy C-Means - Soft Membership to Cluster 0')
        axs[1].set_xlabel('PCA Component 1')
        axs[1].set_ylabel('PCA Component 2')
        axs[1].grid(True)

        plt.suptitle(f"Best Fuzzy C-Means Clustering on {dataset_label} Data (m={best_m}, error={best_error}, maxiter={best_maxiter})")
        plt.tight_layout()
        plt.show()

    pca_info = {
        "explained_variance_ratio_sum": explained_variance,
        "loadings": loadings
    }

    return final_labels, optimal_k, metrics_df, best_params, pca_info





"""
CHATGP: DBSCAN 
"""

 
def dbscan_analysis_and_plot(features, dataset_label="RFM", eps_range=(0.1, 2.0), eps_step=0.1, min_samples_range=(2, 10), show_plots=True):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    eps_values = np.arange(eps_range[0], eps_range[1] + eps_step, eps_step)
    min_samples_values = range(min_samples_range[0], min_samples_range[1])

    results = []

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(scaled_features)

            unique_labels = set(labels)
            if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels and list(labels).count(-1) == len(labels)):
                continue

            score = silhouette_score(scaled_features, labels)
            results.append((eps, min_samples, score))

    if not results:
        raise ValueError("No valid clustering found for given DBSCAN parameters.")

    results.sort(key=lambda x: x[2], reverse=True)
    best_eps, best_min_samples, best_score = results[0]

    print(f"Best DBSCAN params for {dataset_label}: eps={best_eps}, min_samples={best_min_samples}, silhouette={best_score:.4f}")

    best_params = {"eps": best_eps, "min_samples": best_min_samples, "silhouette_score": best_score}

    # PCA and fit DBSCAN with best params
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)
    explained_variance = pca.explained_variance_ratio_.sum()
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)

    dbscan_best = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    final_labels = dbscan_best.fit_predict(scaled_features)

    if show_plots:
        from sklearn.neighbors import NearestNeighbors

        k_for_plot = best_min_samples - 1
        neighbors = NearestNeighbors(n_neighbors=k_for_plot)
        neighbors_fit = neighbors.fit(scaled_features)
        distances, indices = neighbors_fit.kneighbors(scaled_features)

        k_distances = np.sort(distances[:, k_for_plot - 1])
        plt.figure(figsize=(8, 4))
        plt.plot(k_distances)
        plt.title(f'k-Distance Plot ({dataset_label})\n(k = {k_for_plot})')
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'Distance to {k_for_plot}-th Nearest Neighbor')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        unique_labels = np.unique(final_labels)
        has_noise = -1 in unique_labels
        valid_labels = [label for label in unique_labels if label != -1]
        n_clusters = len(valid_labels)
        label_to_index = {label: idx for idx, label in enumerate(valid_labels)}
        color_indices = [label_to_index[label] if label != -1 else n_clusters for label in final_labels]
        colors = cm.get_cmap('viridis', n_clusters + (1 if has_noise else 0))

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=color_indices, cmap=colors, s=50)
        plt.colorbar(scatter, label='Cluster Label')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        handles = []
        for label in unique_labels:
            if label == -1:
                handles.append(mpatches.Patch(color=colors(n_clusters), label='Noise'))
            else:
                idx = label_to_index[label]
                handles.append(mpatches.Patch(color=colors(idx), label=f'Cluster {label}'))

        plt.legend(handles=handles, title='Clusters')
        plt.title(f'2D Clusters (DBSCAN on {dataset_label}) with PCA\n'
                  f'eps={round(best_eps, 2)}, min_samples={best_min_samples}')
        plt.tight_layout()
        plt.show()

    metrics_df = pd.DataFrame(results, columns=["eps", "min_samples", "Silhouette_score"])

    pca_info = {
        "explained_variance_ratio_sum": explained_variance,
        "loadings": loadings
    }

    return final_labels, best_params, metrics_df, pca_info


 
 

