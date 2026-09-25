import pandas as pd


def build_customer_features(df):
    """
    Build customer-level behavioral features from transaction data.

    Expected transaction columns:
        CustomerID
        InvoiceNo
        Quantity
        UnitPrice

    Features produced:
        CustomerID
        NumPurchases
        TotalQuantity
        TotalSpending
        AvgUnitPrice

    NumPurchases is the number of unique invoices for each customer.
    TotalSpending is calculated from Quantity * UnitPrice.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned transaction-level data.

    Returns
    -------
    pandas.DataFrame
        Customer-level behavioral features.
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

    data = df.copy()

    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    customer_features = (
        data.groupby("CustomerID")
        .agg(
            NumPurchases=("InvoiceNo", "nunique"),
            TotalQuantity=("Quantity", "sum"),
            TotalSpending=("TotalPrice", "sum"),
            AvgUnitPrice=("UnitPrice", "mean"),
        )
        .reset_index()
    )

    return customer_features
