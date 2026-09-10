import pandas as pd


def clean_transactions(df):
    """
    Perform basic cleaning for transaction-level data.

    The function removes duplicate records and rows missing
    essential customer/product information.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw transaction data.

    Returns
    -------
    pandas.DataFrame
        Cleaned transaction data.
    """
    cleaned = df.copy()

    cleaned = cleaned.drop_duplicates()

    required_columns = ["CustomerID", "StockCode", "Quantity", "UnitPrice"]

    available_columns = [
        column for column in required_columns
        if column in cleaned.columns
    ]

    if available_columns:
        cleaned = cleaned.dropna(subset=available_columns)

    return cleaned
