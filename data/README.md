# Dataset

This project uses the Online Retail dataset (transactions from a UK-based online retailer). The raw dataset is **not** included in this repository.

This document explains where to obtain the dataset, how to prepare the local input expected by the data-cleaning notebook, and how the cleaned customer-level data is produced for the downstream notebooks.

## Source

UCI Machine Learning Repository / Kaggle

Link: https://archive.ics.uci.edu/ml/datasets/Online+Retail

Please review the dataset page for license and usage restrictions before using the data. You must download the dataset yourself.

## Why the raw data is not included

The raw dataset is not included in this repository to avoid redistributing the source dataset and to keep the public repository lightweight.

## Local dataset preparation

The data-cleaning notebook currently expects a CSV file named:

`OnlineRetail.csv`

Place this file in the notebook's working directory before running `notebooks/01_data_cleaning.ipynb`.

The notebook reads the file using:

```python
pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")
```

The notebook then removes invalid transaction records, constructs customer-level behavioral features, and saves the resulting customer-level dataset for subsequent analysis.
