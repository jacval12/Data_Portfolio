# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 19:18:37 2025

@author: JACOBO VALDERRAMA ROVIRA

"""

"""
CONTEXT: Customer segmentation is a very important topic for all kinds of 
business, the fact that considering customers as different from one another 
can help target them better and as a consequence have better KPIs like ROI.


"""
  
# main.py
#Libraries used for handling data
import pandas as pd
import numpy as np
from datetime import datetime
import polars as pl
import fastexcel
from time import perf_counter
#Libraries for Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

import seaborn as sns
#from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from datetime import datetime as dt
from datetime import timedelta


#Libraries for clustering and machine learning
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.express as px
import skfuzzy as fuzz
from scipy.signal import argrelextrema

from sklearn.decomposition import PCA
#  Hierarchical Clustering tools
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
 
#This part is to call the functions from other files
from preprocessing import load_and_preprocess_data
from analysis import analyze_and_visualize_data
from rfmd_clv_analysis import compute_rfmd_and_clv
from clustering_algorithms import dbscan_analysis_and_plot
from clustering_algorithms import fuzzy_cmeans_analysis_and_plot
from clustering_algorithms import hierarchical_analysis_and_plot
from clustering_algorithms import kmeans_analysis_and_plot

#In this part of the code we import the processed dataframes 
retail_orders, retail_orders_known, retail_orders_unknown = load_and_preprocess_data("Online_Retail.xlsx")
#In the next part of the code we will check the dataset of known customers
results = analyze_and_visualize_data(retail_orders_known, show_plots=True)
#In this next part of the code we will we call the function that creates RFMD table including CLV
RFMD_DF = compute_rfmd_and_clv(retail_orders_known, show_plots=True) # THIS FUNCTION returns a tuple so we need the first element to acces the RFMD_TABLE

RFMD_TABLE = RFMD_DF[0] #We create the RFMD_TABLE 

#Get the feautres for each RFM and RFMD
RFM_features = RFMD_TABLE[["Recency","Frequency","Monetary"]]

RFMD_features = RFMD_TABLE[["Recency","Frequency","Monetary","Diversity"]]

#the next we will import the results of K-Means for both RFM and RFMD
final_labels_Kmeans_rfm, optimal_k_kmeans_rfm, metrics_df_Kmeans_rfm,kmeans_pca_info_rfm=kmeans_analysis_and_plot(RFM_features, dataset_label="RFM", show_plots=True)

final_labels_Kmeans_rfmd, optimal_k_kmeans_rfmd, metrics_df_Kmeans_rfmd,kmeans_pca_info_rfmd = kmeans_analysis_and_plot(RFMD_features, dataset_label="RFMD", show_plots=True)
 
#the next we will import the results of Hierarchical Clustering(Ward) for both RFM and RFMD
final_labels_hierarchical_rfm, optimal_k_hierarchical_rfm, metrics_df_hierarchical_rfm,hierar_pca_info_rfm=hierarchical_analysis_and_plot(RFM_features, dataset_label="RFM", show_plots=True)

final_labels_hierarchical_rfmd, optimal_k_hierarchical_rfmd, metrics_df_hierarchical_rfmd,hierar_pca_info_rfmd=hierarchical_analysis_and_plot(RFMD_features, dataset_label="RFMD", show_plots=True)

#the next we will import the results of Fuzzy C-Means for both RFM and RFMD
final_labels_fuzzy_rfm, optimal_k_fuzzy_rfm, metrics_df_fuzzy_rfm,best_params_fuzzy_rfm,fuzzy_pca_info_rfm =fuzzy_cmeans_analysis_and_plot(RFM_features, dataset_label="RFM", show_plots=True)

final_labels_fuzzy_rfmd, optimal_k_fuzzy_rfmd, metrics_df_fuzzy_rfmd, best_params_fuzzy_rfmd,fuzzy_pca_info_rfmd =fuzzy_cmeans_analysis_and_plot(RFMD_features, dataset_label="RFMD", show_plots=True)

#the next we will import the results of DBSCAN for both RFM and RFMD

final_labels_DBSCAN_rfm, best_params_DBSCAN_rfm, metrics_df_DBSCAN_rfm, dbscan_pca_info_rfm = dbscan_analysis_and_plot(RFM_features, dataset_label="RFM", eps_range=(0.1, 2.0), eps_step=0.1, min_samples_range=(2, 10), show_plots=True)

final_labels_DBSCAN_rfmd, best_params_DBSCAN_rfmd, metrics_df_DBSCAN_rfmd,dbscan_pca_info_rfmd = dbscan_analysis_and_plot(RFMD_features, dataset_label="RFMD", eps_range=(0.1, 2.0), eps_step=0.1, min_samples_range=(2, 10), show_plots=True)

 

"""
In the next part we will show important results
"""
#Results section


#RFMD ANALYSIS
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Boxplots of RFMD Features', fontsize=16)

sns.boxplot(y='Recency', data=RFMD_TABLE, ax=axes[0, 0])
axes[0, 0].set_title('Recency')

sns.boxplot(y='Frequency', data=RFMD_TABLE, ax=axes[0, 1])
axes[0, 1].set_title('Frequency')

sns.boxplot(y='Monetary', data=RFMD_TABLE, ax=axes[1, 0])
axes[1, 0].set_title('Monetary')

sns.boxplot(y='Diversity', data=RFMD_TABLE, ax=axes[1, 1])
axes[1, 1].set_title('Diversity')
plt.show()




#Summary of best metrics for each method


#Summary of best metrics for K-Means
optimal_metrics_kmeans_rfm = metrics_df_Kmeans_rfm[metrics_df_Kmeans_rfm['k'] == optimal_k_kmeans_rfm]
optimal_metrics_kmeans_rfmd = metrics_df_Kmeans_rfmd[metrics_df_Kmeans_rfmd['k'] == optimal_k_kmeans_rfmd]

#Summary of best metrics for Hierarchical Clustering
optimal_metrics_hierarchical_rfm = metrics_df_hierarchical_rfm[metrics_df_hierarchical_rfm['k'] == optimal_k_kmeans_rfm]
optimal_metrics_hierarchical_rfmd = metrics_df_hierarchical_rfmd[metrics_df_hierarchical_rfmd['k'] == optimal_k_kmeans_rfmd]

#Summary of best metrics for Fuzzy C-Means
optimal_metrics_df_fuzzy_rfm = metrics_df_fuzzy_rfm[metrics_df_fuzzy_rfm['k']==optimal_k_fuzzy_rfm]
optimal_metrics_df_fuzzy_rfmd = metrics_df_fuzzy_rfmd[metrics_df_fuzzy_rfmd['k']==optimal_k_fuzzy_rfmd]

#Summary of best metrics for DBSCAN
optimal_metrics_df_DBSCAN_rfm = metrics_df_DBSCAN_rfm[
    (metrics_df_DBSCAN_rfm['eps'] == best_params_DBSCAN_rfm['eps']) &
    (metrics_df_DBSCAN_rfm['min_samples'] == best_params_DBSCAN_rfm['min_samples'])
]

optimal_metrics_df_DBSCAN_rfmd = metrics_df_DBSCAN_rfmd[
    (metrics_df_DBSCAN_rfmd['eps'] == best_params_DBSCAN_rfmd['eps']) &
    (metrics_df_DBSCAN_rfmd['min_samples'] == best_params_DBSCAN_rfmd['min_samples'])
]

#Best metrics RFM


best_metrics_rfm = pd.DataFrame({
    "Method": ["K-Means", "Hierarchical Clustering (Ward)", "Fuzzy C-Means"],
    "Silhouette": [
        optimal_metrics_kmeans_rfm['Silhouette_score'].values[0],
        optimal_metrics_hierarchical_rfm['Silhouette_score'].values[0],
        optimal_metrics_df_fuzzy_rfm['Silhouette_score'].values[0],
    ],
    "Calinski-Harabasz": [
        optimal_metrics_kmeans_rfm['Calinski_harabasz_score'].values[0],
        optimal_metrics_hierarchical_rfm['Calinski_harabasz_score'].values[0],
        optimal_metrics_df_fuzzy_rfm['Calinski_harabasz_score'].values[0],
    ],
    "Davies-Bouldin": [
        optimal_metrics_kmeans_rfm['Davies_bouldin_score(DBI)'].values[0],
        optimal_metrics_hierarchical_rfm['Davies_bouldin_score(DBI)'].values[0],
        optimal_metrics_df_fuzzy_rfm['Davies_bouldin_score(DBI)'].values[0],
    ],
    "Optimal k": [
        optimal_metrics_kmeans_rfm['k'].values[0],
        optimal_metrics_hierarchical_rfm['k'].values[0],
        optimal_metrics_df_fuzzy_rfm['k'].values[0],
        
    ]
})
  
print(best_metrics_rfm)

dbscan_metrics_rfm = pd.DataFrame({
    "Method": ["DBSCAN"],
    "eps": [optimal_metrics_df_DBSCAN_rfm['eps'].values[0]],
    "min_samples": [optimal_metrics_df_DBSCAN_rfm['min_samples'].values[0]],
    "Silhouette_score": [optimal_metrics_df_DBSCAN_rfm['Silhouette_score'].values[0]]
})

print("Best metrics for DBSCAN (RFM):")
print(dbscan_metrics_rfm)


print(best_metrics_rfm.to_dict())
print(dbscan_metrics_rfm.to_dict())

#Best metrics RFMD

best_metrics_rfmd = pd.DataFrame({
    "Method": ["K-Means", "Hierarchical Clustering (Ward)", "Fuzzy C-Means"],
    "Silhouette": [
        optimal_metrics_kmeans_rfmd['Silhouette_score'].values[0],
        optimal_metrics_hierarchical_rfmd['Silhouette_score'].values[0],
        optimal_metrics_df_fuzzy_rfmd['Silhouette_score'].values[0],
    ],
    "Calinski-Harabasz": [
        optimal_metrics_kmeans_rfmd['Calinski_harabasz_score'].values[0],
        optimal_metrics_hierarchical_rfmd['Calinski_harabasz_score'].values[0],
        optimal_metrics_df_fuzzy_rfmd['Calinski_harabasz_score'].values[0],
    ],
    "Davies-Bouldin": [
        optimal_metrics_kmeans_rfmd['Davies_bouldin_score(DBI)'].values[0],
        optimal_metrics_hierarchical_rfmd['Davies_bouldin_score(DBI)'].values[0],
        optimal_metrics_df_fuzzy_rfmd['Davies_bouldin_score(DBI)'].values[0],
    ],
    "Optimal k": [
        optimal_metrics_kmeans_rfmd['k'].values[0],
        optimal_metrics_hierarchical_rfmd['k'].values[0],
        optimal_metrics_df_fuzzy_rfmd['k'].values[0],
        ]
})

print(best_metrics_rfmd)


dbscan_metrics_rfmd = pd.DataFrame({
    "Method": ["DBSCAN"],
    "eps": [optimal_metrics_df_DBSCAN_rfmd['eps'].values[0]],
    "min_samples": [optimal_metrics_df_DBSCAN_rfmd['min_samples'].values[0]],
    "Silhouette_score": [optimal_metrics_df_DBSCAN_rfmd['Silhouette_score'].values[0]]
})

print("Best metrics for DBSCAN (RFMD):")
print(dbscan_metrics_rfmd)

print(best_metrics_rfmd.to_dict())
print(dbscan_metrics_rfmd.to_dict())

#results RFM elbow vs silhouette in each method

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# K-Means Elbow Plot (RFM)
axs[0, 0].plot(metrics_df_Kmeans_rfm['k'], metrics_df_Kmeans_rfm['WCSS'], marker='o', color='tab:blue')
axs[0, 0].axvline(optimal_metrics_kmeans_rfm['k'].values[0], color='gray', linestyle='--', label=f'Optimal k = {optimal_metrics_kmeans_rfm['k'].values[0]}')
axs[0, 0].set_title('K-Means Elbow Plot (RFM)')
axs[0, 0].set_xlabel('Number of Clusters (k)')
axs[0, 0].set_ylabel('Inertia')
axs[0, 0].legend()
axs[0, 0].grid(True)

# K-Means Silhouette Plot (RFM)
axs[0, 1].plot(metrics_df_Kmeans_rfm['k'], metrics_df_Kmeans_rfm['Silhouette_score'], marker='o', color='tab:green')
axs[0, 1].axvline(optimal_metrics_kmeans_rfm['k'].values[0], color='gray', linestyle='--', label=f'Optimal k = {optimal_metrics_kmeans_rfm['k'].values[0]}')
axs[0, 1].set_title('K-Means Silhouette Score (RFM)')
axs[0, 1].set_xlabel('Number of Clusters (k)')
axs[0, 1].set_ylabel('Silhouette Score')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Hierarchical Elbow Plot (RFM)
axs[1, 0].plot(metrics_df_hierarchical_rfm['k'], metrics_df_hierarchical_rfm['WCSS'], marker='s', color='tab:orange')
axs[1, 0].axvline(optimal_metrics_hierarchical_rfm['k'].values[0], color='gray', linestyle='--', label=f'Optimal k = {optimal_metrics_hierarchical_rfm['k'].values[0]}')
axs[1, 0].set_title('Hierarchical Elbow Plot (RFM)')
axs[1, 0].set_xlabel('Number of Clusters (k)')
axs[1, 0].set_ylabel('Inertia')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Hierarchical Silhouette Plot (RFM)
axs[1, 1].plot(metrics_df_hierarchical_rfm['k'], metrics_df_hierarchical_rfm['Silhouette_score'], marker='s', color='tab:red')
axs[1, 1].axvline(optimal_metrics_hierarchical_rfm['k'].values[0], color='gray', linestyle='--', label=f'Optimal k = {optimal_metrics_hierarchical_rfm['k'].values[0]}')
axs[1, 1].set_title('Hierarchical Silhouette Score (RFM)')
axs[1, 1].set_xlabel('Number of Clusters (k)')
axs[1, 1].set_ylabel('Silhouette Score')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at the top for suptitle
plt.show()

#results RFMD elbow vs silhouette in each method

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# K-Means Elbow Plot (RFMD)
axs[0, 0].plot(metrics_df_Kmeans_rfmd['k'], metrics_df_Kmeans_rfmd['WCSS'], marker='o', color='tab:blue')
axs[0, 0].axvline(optimal_metrics_kmeans_rfmd['k'].values[0], color='gray', linestyle='--', label=f'Optimal k = {optimal_metrics_kmeans_rfmd['k'].values[0]}')
axs[0, 0].set_title('K-Means Elbow Plot (RFMD)')
axs[0, 0].set_xlabel('Number of Clusters (k)')
axs[0, 0].set_ylabel('Inertia')
axs[0, 0].legend()
axs[0, 0].grid(True)

# K-Means Silhouette Plot (RFMD)
axs[0, 1].plot(metrics_df_Kmeans_rfmd['k'], metrics_df_Kmeans_rfmd['Silhouette_score'], marker='o', color='tab:green')
axs[0, 1].axvline(optimal_metrics_kmeans_rfmd['k'].values[0], color='gray', linestyle='--', label=f'Optimal k = {optimal_metrics_kmeans_rfmd['k'].values[0]}')
axs[0, 1].set_title('K-Means Silhouette Score (RFMD)')
axs[0, 1].set_xlabel('Number of Clusters (k)')
axs[0, 1].set_ylabel('Silhouette Score')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Hierarchical Elbow Plot (RFMD)
axs[1, 0].plot(metrics_df_hierarchical_rfmd['k'], metrics_df_hierarchical_rfmd['WCSS'], marker='s', color='tab:orange')
axs[1, 0].axvline(optimal_metrics_hierarchical_rfmd['k'].values[0], color='gray', linestyle='--', label=f'Optimal k = {optimal_metrics_hierarchical_rfmd['k'].values[0]}')
axs[1, 0].set_title('Hierarchical Elbow Plot (RFMD)')
axs[1, 0].set_xlabel('Number of Clusters (k)')
axs[1, 0].set_ylabel('Inertia')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Hierarchical Silhouette Plot (RFMD)
axs[1, 1].plot(metrics_df_hierarchical_rfmd['k'], metrics_df_hierarchical_rfmd['Silhouette_score'], marker='s', color='tab:red')
axs[1, 1].axvline(optimal_metrics_hierarchical_rfmd['k'].values[0], color='gray', linestyle='--', label=f'Optimal k = {optimal_metrics_hierarchical_rfmd['k'].values[0]}')
axs[1, 1].set_title('Hierarchical Silhouette Score (RFMD)')
axs[1, 1].set_xlabel('Number of Clusters (k)')
axs[1, 1].set_ylabel('Silhouette Score')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space at the top for suptitle
plt.show()



#\subsubsection{Metrics Used to Compare Clustering Models}

#RFM
#kmeans
calinski_score_kmeans_rfm = optimal_metrics_kmeans_rfm['Calinski_harabasz_score'].values[0]
dbi_score_kmeans_rfm = optimal_metrics_kmeans_rfm['Davies_bouldin_score(DBI)'].values[0]
#hierarchical
calinski_score_hierarchical_rfm = optimal_metrics_hierarchical_rfm['Calinski_harabasz_score'].values[0]
dbi_score_hierarchical_rfm = optimal_metrics_hierarchical_rfm['Davies_bouldin_score(DBI)'].values[0]
#fuzzy
calinski_score_fuzzy_rfm = optimal_metrics_df_fuzzy_rfm['Calinski_harabasz_score'].values[0]
dbi_score_fuzzy_rfm = optimal_metrics_df_fuzzy_rfm['Davies_bouldin_score(DBI)'].values[0]

#Calinski rfm
fig, ax = plt.subplots(figsize=(12, 6))  # wider figure for spacing
methods = ['K-Means', 'Hierarchical Clustering(Ward)', 'Fuzzy C-Means']
scores = [calinski_score_kmeans_rfm, calinski_score_hierarchical_rfm, calinski_score_fuzzy_rfm]
colors = ['tab:blue', 'tab:orange', 'tab:green']
bars = ax.bar(methods, scores, color=colors, edgecolor='black')
ax.set_ylabel('Calinski-Harabasz Score')
ax.set_title('Calinski-Harabasz Score by Clustering Method (RFM)')
ax.grid(axis='y', linestyle='--', alpha=0.7)  # horizontal grid lines for easier reading
# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',  # format to 1 decimal place
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # offset label slightly above bar
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=10)
plt.tight_layout()  # better padding around plot
plt.show()

#DBI rfm
fig, ax = plt.subplots(figsize=(12, 6))  # wider figure for spacing
methods = ['K-Means', 'Hierarchical Clustering(Ward)', 'Fuzzy C-Means']
scores = [dbi_score_kmeans_rfm, dbi_score_hierarchical_rfm, dbi_score_fuzzy_rfm]
colors = ['tab:blue', 'tab:orange', 'tab:green']
bars = ax.bar(methods, scores, color=colors, edgecolor='black')
ax.set_ylabel('Davies_bouldin_score(DBI)')
ax.set_title('Davies_bouldin_score(DBI) by Clustering Method (RFM)')
ax.grid(axis='y', linestyle='--', alpha=0.7)  # horizontal grid lines for easier reading
# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',  # format to 1 decimal place
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # offset label slightly above bar
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=10)
plt.tight_layout()  # better padding around plot
plt.show()


#RFMD

#kmeans
calinski_score_kmeans_rfmd = optimal_metrics_kmeans_rfmd['Calinski_harabasz_score'].values[0]
dbi_score_kmeans_rfmd = optimal_metrics_kmeans_rfmd['Davies_bouldin_score(DBI)'].values[0]
#hierarchical
calinski_score_hierarchical_rfmd = optimal_metrics_hierarchical_rfmd['Calinski_harabasz_score'].values[0]
dbi_score_hierarchical_rfmd = optimal_metrics_hierarchical_rfmd['Davies_bouldin_score(DBI)'].values[0]
#fuzzy
calinski_score_fuzzy_rfmd = optimal_metrics_df_fuzzy_rfmd['Calinski_harabasz_score'].values[0]
dbi_score_fuzzy_rfmd = optimal_metrics_df_fuzzy_rfmd['Davies_bouldin_score(DBI)'].values[0]

#Calinski RFMD
fig, ax = plt.subplots(figsize=(12, 6))  # wider figure for spacing
methods = ['K-Means', 'Hierarchical Clustering(Ward)', 'Fuzzy C-Means']
scores = [calinski_score_kmeans_rfmd, calinski_score_hierarchical_rfmd, calinski_score_hierarchical_rfmd]
colors = ['tab:blue', 'tab:orange', 'tab:green']
bars = ax.bar(methods, scores, color=colors, edgecolor='black')
ax.set_ylabel('Calinski-Harabasz Score')
ax.set_title('Calinski-Harabasz Score by Clustering Method (RFMD)')
ax.grid(axis='y', linestyle='--', alpha=0.7)  # horizontal grid lines for easier reading
# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',  # format to 1 decimal place
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # offset label slightly above bar
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=10)
plt.tight_layout()  # better padding around plot
plt.show()
 
#DBI RFMD
fig, ax = plt.subplots(figsize=(12, 6))  # wider figure for spacing
methods = ['K-Means', 'Hierarchical Clustering(Ward)', 'Fuzzy C-Means']
scores = [dbi_score_kmeans_rfmd, dbi_score_hierarchical_rfmd, dbi_score_fuzzy_rfmd]
colors = ['tab:blue', 'tab:orange', 'tab:green']
bars = ax.bar(methods, scores, color=colors, edgecolor='black')
ax.set_ylabel('Davies_bouldin_score(DBI)')
ax.set_title('Davies_bouldin_score(DBI) by Clustering Method (RFMD)')
ax.grid(axis='y', linestyle='--', alpha=0.7)  # horizontal grid lines for easier reading
# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',  # format to 1 decimal place
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # offset label slightly above bar
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=10)
plt.tight_layout()  # better padding around plot
plt.show()


#RESULTS CLV
#In the next part we assign the labels to the RFMD TABLE DATAFRAME
RFMD_TABLE["Kmeans_RFM_label"] = final_labels_Kmeans_rfm
RFMD_TABLE["Kmeans_RFMD_label"] = final_labels_Kmeans_rfmd
RFMD_TABLE["Hierarchical_RFM_label"] = final_labels_hierarchical_rfm
RFMD_TABLE["Hierarchical_RFMD_label"] = final_labels_hierarchical_rfmd
RFMD_TABLE["Fuzzy_RFM_label"] = final_labels_fuzzy_rfm
RFMD_TABLE["Fuzzy_RFMD_label"] = final_labels_fuzzy_rfmd
RFMD_TABLE["DBSCAN_RFM_label"] = final_labels_DBSCAN_rfm
RFMD_TABLE["DBSCAN_RFMD_label"] = final_labels_DBSCAN_rfmd


 
datasets = {
    "RFMD": ["Kmeans_RFMD_label", "Hierarchical_RFMD_label", "Fuzzy_RFMD_label", "DBSCAN_RFMD_label"],
    "RFM": ["Kmeans_RFM_label", "Hierarchical_RFM_label", "Fuzzy_RFM_label", "DBSCAN_RFM_label"]
}
method_titles = ["K-Means", "Hierarchical", "Fuzzy C-Means", "DBSCAN"]

for dataset_name, label_cols in datasets.items():
    # CLV boxplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (col, title) in enumerate(zip(label_cols, method_titles)):
        sns.boxplot(x=RFMD_TABLE[col], y=RFMD_TABLE["CLV"], palette="pastel", ax=axes[i])
        axes[i].set_title(f"CLV per Cluster - {title}")
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("CLV")
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)
    plt.suptitle(f"CLV Distribution per Cluster ({dataset_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Barplots: number of customers
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, (col, title) in enumerate(zip(label_cols, method_titles)):
        cluster_counts = RFMD_TABLE[col].value_counts().sort_index()
        axes[i].bar(cluster_counts.index, cluster_counts.values, color='skyblue')
        axes[i].set_title(f"Cluster Sizes - {title}")
        axes[i].set_xlabel("Cluster")
        axes[i].set_ylabel("Number of Customers")
        axes[i].grid(axis='y', linestyle='--', alpha=0.6)
    plt.suptitle(f"Cluster Sizes per Method ({dataset_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    # --- Print Average CLV Table per method including customer count ---
    print(f"\n=== Average CLV and Customer Count per Cluster ({dataset_name}) ===")
    for col, title in zip(label_cols, method_titles):
        grouped = RFMD_TABLE.groupby(col).agg(
            Average_CLV=('CLV', 'mean'),
            Customer_Count=('CLV', 'count')
        ).sort_index().reset_index()

        grouped.columns = ["Cluster", "Average CLV", "Number of Customers"]

        print(f"\n--- {title} ---")
        print(grouped.round(2).to_string(index=False))


#In this part we will complement the part of results of the cluster pca diagrams


#kmeans variance
explained_variance_ratio_rfm_kmeans_pca = kmeans_pca_info_rfm['explained_variance_ratio_sum']
explained_variance_ratio_rfmd_kmeans_pca = kmeans_pca_info_rfmd['explained_variance_ratio_sum']

#kmeans loadings
loadings_rfm_kmeans_pca = kmeans_pca_info_rfm['loadings']
loadings_rfmd_kmeans_pca = kmeans_pca_info_rfmd['loadings']

#hierachical variance
explained_variance_ratio_rfm_hierarchical_pca = hierar_pca_info_rfm['explained_variance_ratio_sum']
explained_variance_ratio_rfmd_hierarchical_pca = hierar_pca_info_rfmd['explained_variance_ratio_sum']

#hierachical loadings
loadings_rfm_hierarchical_pca = hierar_pca_info_rfm['loadings']
loadings_rfmd_hierarchical_pca = hierar_pca_info_rfmd['loadings']

#fuzzy variance
explained_variance_ratio_rfm_fuzzy_pca = fuzzy_pca_info_rfm['explained_variance_ratio_sum']
explained_variance_ratio_rfmd_fuzzy_pca = fuzzy_pca_info_rfmd['explained_variance_ratio_sum']

#fuzzy loadings
loadings_rfm_fuzzy_pca = fuzzy_pca_info_rfm['loadings']
loadings_rfmd_fuzzy_pca = fuzzy_pca_info_rfmd['loadings']

#DBSCAN variance
explained_variance_ratio_rfm_dbscan_pca = dbscan_pca_info_rfm['explained_variance_ratio_sum']
explained_variance_ratio_rfmd_dbscan_pca = dbscan_pca_info_rfmd['explained_variance_ratio_sum']

#DBSCAN loadings
loadings_rfm_dbscan_pca = dbscan_pca_info_rfm['loadings']
loadings_rfmd_dbscan_pca = dbscan_pca_info_rfmd['loadings']


loadings = [loadings_rfm_kmeans_pca,loadings_rfmd_kmeans_pca,loadings_rfm_hierarchical_pca,loadings_rfmd_hierarchical_pca, loadings_rfm_fuzzy_pca, loadings_rfmd_fuzzy_pca,loadings_rfm_dbscan_pca,loadings_rfmd_dbscan_pca]

titles = [
    "RFM - KMeans PCA Loadings",
    "RFMD - KMeans PCA Loadings",
    "RFM - Hierarchical PCA Loadings",
    "RFMD - Hierarchical PCA Loadings",
    "RFM - Fuzzy PCA Loadings",
    "RFMD - Fuzzy PCA Loadings",
    "RFM - DBSCAN PCA Loadings",
    "RFMD - DBSCAN PCA Loadings"
]

for loading, title in zip(loadings, titles):
    features = loading.index.tolist()
    pc1 = loading['PC1'].abs()
    pc2 = loading['PC2'].abs()
    x = np.arange(len(features))
    width = 0.4

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # PC1 plot
    axs[0].bar(x, pc1, width, color='skyblue')
    axs[0].set_title('PC1 Loadings')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(features, rotation=45, ha='right')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid(True, axis='y')

    ymax_pc1 = max(pc1) * 1.15  # Add 15% headroom
    axs[0].set_ylim(0, ymax_pc1)
    for i, val in enumerate(pc1):
        axs[0].text(i, val + ymax_pc1*0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # PC2 plot
    axs[1].bar(x, pc2, width, color='lightgreen')
    axs[1].set_title('PC2 Loadings')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(features, rotation=45, ha='right')
    axs[1].grid(True, axis='y')

    ymax_pc2 = max(pc2) * 1.15
    axs[1].set_ylim(0, ymax_pc2)
    for i, val in enumerate(pc2):
        axs[1].text(i, val + ymax_pc2*0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()





print("The last of the code")





"""
In this section we will analyze everything but for uknown customers 


retail_orders_unknown["InvoiceNo"] = retail_orders_unknown["InvoiceNo"].astype(str)


InvoiceNo_unique = retail_orders_unknown['InvoiceNo'].drop_duplicates() # Create a dataset with only unique values for InvoiceNo

index = 1
for value in InvoiceNo_unique:
   new_id = f"UNK{str(index).zfill(2)}"  # Create ID like UNK01, UNK02
   retail_orders_unknown.loc[retail_orders_unknown['InvoiceNo'] == value, 'CustomerID'] = new_id
   index += 1  
   #chatGPT was used here for this

"""





