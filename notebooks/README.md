# Jupyter Notebooks

The notebooks document the experimental workflow for the
E-Commerce Personalization project.

## Notebook 1 — Data Cleaning

`01_data_cleaning.ipynb`

Covers:

- Loading the Online Retail dataset
- Inspecting dataset structure
- Missing-value analysis
- Duplicate removal
- Removal of invalid transactions
- Customer-level feature construction

The resulting customer-level features are:

- `NumPurchases`
- `TotalQuantity`
- `TotalSpending`
- `AvgUnitPrice`

## Notebook 2 — Exploratory Analysis

`02_exploratory_analysis.ipynb`

Explores customer purchasing behavior and transaction-level patterns
through statistical analysis and visualizations.

## Notebook 3 — Customer Segmentation

`03_customer_segmentation.ipynb`

Covers:

- Loading customer-level features
- Selecting the four behavioral features
- Standardization using `StandardScaler`
- Preparation of the feature matrix for clustering

## Notebook 4 — Expectation-Maximization Clustering

`04_expectation_maximization_clustering.ipynb`

Covers:

- Gaussian Mixture Model (GMM) clustering
- Evaluation of candidate cluster counts
- AIC and BIC model selection
- Final cluster assignment
- Cluster-size analysis
- Cluster profiling
- PCA-based visualization

The evaluated range was `k = 2` through `k = 10`, with the final
experiment using 10 components because this was the minimum AIC/BIC
value within the evaluated range.

## Current Repository Scope

The repository currently contains the implemented data preparation,
customer feature engineering, exploratory analysis, and EM/GMM
clustering workflow.

A deep-learning recommendation implementation and separate evaluation
notebook are not currently included in the repository.
