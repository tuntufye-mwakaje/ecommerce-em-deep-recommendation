# E-Commerce Personalization using Expectation-Maximization Clustering and Deep Recommendation

## Overview

This project investigates a hybrid approach to e-commerce personalization
that combines customer behavioral segmentation using
**Expectation-Maximization (EM) / Gaussian Mixture Models (GMM)** with
a planned deep-learning recommendation stage.

The current repository implements and documents the customer analytics
and segmentation pipeline using the Online Retail dataset. The
repository is structured to support subsequent integration of a
deep-learning recommendation component.

## Problem Statement

Customers exhibit different purchasing behaviors, including differences
in purchase frequency, quantity, spending, and price sensitivity.

A personalization system can use these behavioral differences to
identify customer segments before applying recommendation techniques.

This project therefore investigates a workflow in which:

1. Customer purchasing behavior is transformed into customer-level
   behavioral features.
2. Customers are segmented using EM/GMM clustering.
3. The resulting segments can subsequently be incorporated into a
   recommendation model.

## Current Objectives

The implemented repository currently focuses on:

- Analyzing customer purchasing behavior
- Cleaning transaction-level data
- Constructing customer-level behavioral features
- Standardizing behavioral features for clustering
- Evaluating candidate GMM cluster counts
- Comparing AIC and BIC across candidate models
- Assigning customers to behavioral clusters
- Profiling the resulting customer segments
- Visualizing customer segments

The broader project objective is to integrate these customer segments
into a deep-learning recommendation system.

## Dataset

The project uses the **Online Retail** transactional dataset.

The dataset contains transaction records including:

- Invoice number
- Stock code
- Product description
- Quantity
- Invoice date
- Unit price
- Customer ID
- Country

The raw dataset is not included in this repository.

See `data/README.md` for dataset information and acquisition
instructions.

## Implemented Methodology

```text
Online Retail Transactions
            |
            v
      Data Cleaning
            |
            v
 Customer-Level Features
            |
            v
    Exploratory Analysis
            |
            v
     Feature Scaling
            |
            v
       EM / GMM
        Clustering
            |
            v
     Cluster Analysis
            |
            v
     Visualization

### 1. Data Cleaning

The transaction-level preparation includes:

- Removing records with missing `CustomerID`
- Removing duplicate records
- Removing transactions with `Quantity <= 0`
- Removing transactions with `UnitPrice <= 0`

The resulting customer-level dataset is constructed from valid
transactions.

### 2. Customer-Level Feature Engineering

Four behavioral features are constructed for each customer:

| Feature | Description |
|---|---|
| `NumPurchases` | Number of unique invoices |
| `TotalQuantity` | Total quantity purchased |
| `TotalSpending` | Total transaction spending |
| `AvgUnitPrice` | Mean unit price across transactions |

### 3. Feature Scaling

The four customer-level behavioral features are standardized using
`StandardScaler` before clustering.

### 4. EM / GMM Clustering

Customer segmentation is performed using
`sklearn.mixture.GaussianMixture`.

The clustering experiment evaluates candidate models with:

- `n_components = 2` through `10`
- `covariance_type = "full"`
- `random_state = 42`

AIC and BIC are recorded for each candidate cluster count.

Within the evaluated range of 2–10 components, the recorded minimum
AIC and BIC occur at 10 components. The final documented experiment
therefore uses 10 GMM components.

### 5. Cluster Analysis

The final experiment assigns each customer to one of the resulting
clusters and examines:

- Cluster sizes
- Cluster proportions
- Behavioral feature profiles
- PCA-based two-dimensional visualization

The resulting clusters include both large customer groups and several
very small groups. These outcomes are retained as part of the
experimental result rather than being manually removed or merged.

## Repository Structure

```text
ecommerce-em-deep-recommendation/
¦
+-- data/
¦   +-- README.md
¦
+-- models/
¦   +-- README.md
¦
+-- notebooks/
¦   +-- 01_data_cleaning.ipynb
¦   +-- 02_exploratory_analysis.ipynb
¦   +-- 03_customer_segmentation.ipynb
¦   +-- 04_expectation_maximization_clustering.ipynb
¦   +-- README.md
¦
+-- results/
¦   +-- Figures/
¦   +-- README.md
¦
+-- src/
¦   +-- __init__.py
¦   +-- data/
¦   ¦   +-- __init__.py
¦   ¦   +-- loader.py
¦   +-- preprocessing/
¦   ¦   +-- __init__.py
¦   ¦   +-- cleaning.py
¦   +-- features/
¦   ¦   +-- __init__.py
¦   ¦   +-- customer_features.py
¦   +-- clustering/
¦   ¦   +-- __init__.py
¦   ¦   +-- gmm.py
¦   +-- README.md
¦
+-- .gitignore
+-- LICENSE
+-- README.md

## Source Code

Reusable Python functionality is currently provided for:

- Dataset loading
- Transaction cleaning
- Customer-level feature engineering
- Gaussian Mixture Model evaluation
- Final GMM fitting
- Cluster assignment
- Customer cluster profiling

The EM/GMM experimentation remains documented in the Jupyter
notebooks, while reusable clustering functions are also available in
`src/clustering/gmm.py`.

## Results and Visualizations

The repository currently contains figures documenting:

- Customer spending distribution
- Customer purchase frequency
- Purchase frequency versus total spending
- Behavioral-feature correlations
- AIC/BIC model selection
- Cluster distribution
- Cluster profiles
- PCA visualization of customer clusters

These outputs are available under:

```text
results/Figures/
## Deep Recommendation Stage

The original project concept includes a deep-learning recommendation
stage in which customer behavioral segments can be incorporated into
personalized product recommendation.

A complete deep-recommendation implementation, trained model artifact,
and recommendation evaluation pipeline are **not currently included
in this repository**.

This stage is therefore treated as planned/future development rather
than presented as an implemented result.

## Reproducibility

The public repository does not contain the raw Online Retail dataset.

To reproduce the implemented analysis:

1. Obtain the Online Retail dataset from its original source.
2. Place the dataset in the local working environment.
3. Run the notebooks in sequence.
4. Follow the feature construction and scaling steps documented in the
   notebooks.
5. Run the EM/GMM model-selection and clustering notebook.

The raw dataset should not be committed to the public repository.

## License

See the `LICENSE` file for project licensing information.
