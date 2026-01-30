# -*- coding: utf-8 -*-
"""
Created on Thu May 22 15:02:22 2025
@author: ASUS
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def compute_rfmd_and_clv(retail_orders_known, show_plots=True):
    # Recency calculation
    latest_general_date = retail_orders_known["InvoiceDate"].max()
    latest_date_by_customer = retail_orders_known.groupby("CustomerID")["InvoiceDate"].max().reset_index()
    latest_date_by_customer["Recency"] = (latest_general_date - latest_date_by_customer["InvoiceDate"]).dt.days

    # Frequency and Monetary calculation
    FM = retail_orders_known.groupby("CustomerID").agg(
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Total_Price", "sum")
    ).reset_index()

    RFM_TABLE = pd.merge(FM, latest_date_by_customer[["CustomerID", "Recency"]], on="CustomerID")

    # RFM scores
    RFM_TABLE["R_Score"] = pd.qcut(RFM_TABLE['Recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    RFM_TABLE["F_Score"] = pd.qcut(RFM_TABLE['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    RFM_TABLE["M_Score"] = pd.qcut(RFM_TABLE['Monetary'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    RFM_TABLE["RFM_Score"] = RFM_TABLE["R_Score"] + RFM_TABLE["F_Score"] + RFM_TABLE["M_Score"]

    # Diversity (D)
    Products_diversity = retail_orders_known.groupby("CustomerID")["StockCode"].nunique().reset_index()
    Products_diversity.rename(columns={"StockCode": "Diversity"}, inplace=True)

    RFMD_TABLE = pd.merge(RFM_TABLE, Products_diversity, on="CustomerID", how="left")
    RFMD_TABLE["D_Score"] = pd.qcut(RFMD_TABLE['Diversity'], q=5, labels=False, duplicates='drop') + 1
    RFMD_TABLE["RFMD_Score"] = RFMD_TABLE["R_Score"] + RFMD_TABLE["F_Score"] + RFMD_TABLE["M_Score"] + RFMD_TABLE["D_Score"]

    # Reorder columns
    RFMD_TABLE = RFMD_TABLE[["CustomerID", "Recency", "Frequency", "Monetary", "Diversity",
                             "R_Score", "F_Score", "M_Score", "D_Score", "RFM_Score", "RFMD_Score"]]

    # Churn determination
    RFMD_TABLE["Churned"] = (RFMD_TABLE["Recency"] > 90).astype(int)

    # Summary statistics
    small_RFMD = RFMD_TABLE[["Recency", "Frequency", "Monetary", "Diversity"]]
    rfmd_stats = pd.DataFrame({
        'Mean': small_RFMD.mean(),
        'Median': small_RFMD.median(),
        'Std': small_RFMD.std()
    }).round(0)
    print("RFMD Summary Stats:\n", rfmd_stats)

    # CLV Calculation
    T = 3  # estimated average customer lifespan in years
    discount_rate = 0.10  # annual discount rate
    CAC = 100  # assumed customer acquisition cost
    cost_ratio = 0.2  # assumed operating cost ratio

    clv_values = []
    for _, row in RFMD_TABLE.iterrows():
        annual_revenue = row["Monetary"] / T
        annual_cost = annual_revenue * cost_ratio
        clv = 0
        for t in range(1, int(T) + 1):
            clv += (annual_revenue - annual_cost) / ((1 + discount_rate) ** t)
        clv -= CAC
        clv_values.append(clv)

    RFMD_TABLE["CLV"] = clv_values

    if show_plots:
        # RFM Heatmap
        pivot_mscore = RFMD_TABLE.pivot_table(index='R_Score', columns='F_Score', values='M_Score', aggfunc='mean', fill_value=0)
        pivot_count = RFMD_TABLE.pivot_table(index='R_Score', columns='F_Score', values='Monetary', aggfunc='count', fill_value=0)
        annot_labels = pivot_mscore.round(1).astype(str) + "\n(n=" + pivot_count.astype(int).astype(str) + ")"

        plt.figure(figsize=(10, 7))
        sns.heatmap(pivot_mscore, annot=annot_labels, fmt='', cmap='YlGnBu', cbar_kws={'label': 'Average Monetary Score'})
        plt.gca().invert_yaxis()
        plt.xlabel('Frequency Score')
        plt.ylabel('Recency Score')
        plt.title('RFM Segmentation Heatmap')
        plt.tight_layout()
        plt.show()

        # RFMD Histograms
        colors = {
            "Recency": "#1f77b4",
            "Frequency": "#2ca02c",
            "Monetary": "#ffd700",
            "Diversity": "#800080"
        }

        plt.figure(figsize=(15, 12))
        for idx, column in enumerate(["Recency", "Frequency", "Monetary", "Diversity"], 1):
            plt.subplot(2, 2, idx)
            sns.histplot(RFMD_TABLE[column], bins=50, kde=True, color=colors[column])
            plt.title(f'{column} Distribution')
            plt.xlabel(column)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

        # RFMD Subplot Heatmaps by D_Score
        d_scores = sorted(RFMD_TABLE["D_Score"].dropna().unique())
        n = len(d_scores)
        cols = 3
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)

        for i, d in enumerate(d_scores):
            row, col = divmod(i, cols)
            ax = axes[row, col]

            subset = RFMD_TABLE[RFMD_TABLE["D_Score"] == d]
            pivot_avg = subset.pivot_table(index="R_Score", columns="F_Score", values="M_Score", aggfunc="mean", fill_value=0)
            pivot_count = subset.pivot_table(index="R_Score", columns="F_Score", values="M_Score", aggfunc="count", fill_value=0)
            annot_labels = pivot_avg.round(1).astype(str) + "\n(n=" + pivot_count.astype(int).astype(str) + ")"

            sns.heatmap(pivot_avg, annot=annot_labels, fmt="", cmap="YlGnBu",
                        cbar=(i == 0), ax=ax, cbar_kws={"label": "Avg Monetary Score"})
            ax.set_title(f"D_Score = {d}")
            ax.set_xlabel("Frequency Score")
            ax.set_ylabel("Recency Score")
            ax.invert_yaxis()

        # Hide any unused subplots
        for j in range(n, rows * cols):
            fig.delaxes(axes.flat[j])

        plt.tight_layout()
        plt.show()

    return RFMD_TABLE, rfmd_stats
