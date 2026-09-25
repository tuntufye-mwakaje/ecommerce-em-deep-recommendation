import pandas as pd


def clean_transactions(df):
    """
    Clean Online Retail transaction-level data.

    Cleaning operations:
    1. Remove duplicate rows.
    2. Remove transactions with missing CustomerID.
    3. Remove transactions with Quantity <= 0.
    4. Remove transactions with UnitPrice <= 0.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw transaction-level data.

    Returns
    -------
    pandas.DataFrame
        Cleaned transaction-level data.
    """
    required_columns = [
        "CustomerID",
        "InvoiceNo",
        "Quantity",
        "UnitPrice",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    cleaned = df.copy()

    cleaned = cleaned.dropna(subset=["CustomerID"])

    cleaned = cleaned.drop_duplicates()

    cleaned = cleaned[
        (cleaned["Quantity"] > 0)
        & (cleaned["UnitPrice"] > 0)
    ].copy()

    return cleaned
