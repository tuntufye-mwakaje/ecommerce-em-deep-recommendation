import pandas as pd


def build_customer_features(df):
    """
    Build customer-level behavioral features.

    Expected transaction columns:
        CustomerID
        Quantity
        UnitPrice

    Returns
    -------
    pandas.DataFrame
        Customer-level behavioral features.
    """
    required_columns = ["CustomerID", "Quantity", "UnitPrice"]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    data = df.copy()

    data["Spending"] = data["Quantity"] * data["UnitPrice"]

    customer_features = (
        data.groupby("CustomerID")
        .agg(
            NumPurchases=("Quantity", "count"),
            TotalQuantity=("Quantity", "sum"),
            TotalSpending=("Spending", "sum"),
            AvgUnitPrice=("UnitPrice", "mean"),
        )
        .reset_index()
    )

    return customer_features
