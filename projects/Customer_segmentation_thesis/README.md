📊 Customer Segmentation using RFMD, CLV and Clustering Algorithms
Master Thesis – Customer Analytics & Machine Learning

This repository contains the full implementation of my master’s thesis focused on customer segmentation in retail using traditional marketing techniques combined with machine learning clustering algorithms.
The objective of the project is to identify meaningful customer segments and provide actionable marketing recommendations based on purchasing behavior.

🔍 Research Motivation

Customer segmentation is a key component of modern marketing strategies. Traditional approaches such as RFM (Recency, Frequency, Monetary) are widely used but often lack granularity.
This thesis extends RFM by incorporating Product Diversity (RFMD) and Customer Lifetime Value (CLV) to improve segmentation quality and business interpretability.

The study compares multiple clustering techniques to determine which method offers the best balance between performance, interpretability, and practicality.

❓ Research Questions

Which clustering algorithms provide the most meaningful customer segments based on RFM and RFMD features?

Does adding product diversity (D) improve customer segmentation results?

How do customer segments differ in terms of value (CLV), engagement, and purchasing behavior?

Which segmentation approach is most suitable for real-world marketing decision-making?

How can these segments be translated into targeted marketing strategies?

🗂 Dataset

Source: UCI Machine Learning Repository

Dataset: Online Retail Transactions (UK-based retailer)

Time period: December 2010 – December 2011

Size: ~541,000 transactions

⚙️ Methodology Overview

Data Preprocessing

Data cleaning and validation

Handling missing customer identifiers

Feature engineering

Feature Construction

RFM and RFMD metrics

Customer Lifetime Value (CLV) estimation

Clustering Algorithms

K-Means

Hierarchical Clustering

Fuzzy C-Means

DBSCAN

Evaluation Metrics

Silhouette Score

Calinski-Harabasz Index

Davies-Bouldin Index

Visualization

PCA used for 2D visualization only

🧠 Key Results & Findings

RFMD provided more interpretable and actionable segments than traditional RFM.

K-Means delivered the best overall balance between performance and interpretability.

High-value customers were characterized by higher product diversity and frequency.

CLV analysis enabled clear differentiation between customer segments.

🎯 Marketing Insights

The identified clusters support targeted marketing strategies such as:

Re-engagement campaigns for disengaged customers

Retention strategies for high-value customers

Onboarding actions for new customers

🧱 Project Structure
Customer Segmentation Thesis/
├── main.py
├── preprocessing.py
├── analysis.py
├── rfmd_clv_analysis.py
├── clustering_algorithms.py
├── README.md
├── requirements.txt
└── thesis.pdf

▶️ How to Run the Project
pip install -r requirements.txt
python main.py

🛠 Technologies Used

Python

Pandas / Polars

NumPy

Scikit-learn

SciPy

Scikit-fuzzy

Matplotlib / Seaborn / Plotly

📄 Thesis Document

The complete thesis document is included in this repository as a PDF file.
