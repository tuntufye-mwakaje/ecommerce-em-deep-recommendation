import pandas as pd
from sklearn.preprocessing import StandardScaler


def standardize_customer_features(customer_features, feature_columns):
    """
    Standardize selected customer features using StandardScaler.

    Parameters
    ----------
    customer_features : pandas.DataFrame
        Customer-level feature table.
    feature_columns : list
        Names of the numerical feature columns to standardize.

    Returns
    -------
    scaled_features : pandas.DataFrame
        Standardized features with the original index and column names.
    scaler : StandardScaler
        Fitted scaler instance.
    """
    missing = [
        column
        for column in feature_columns
        if column not in customer_features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required customer features: {', '.join(missing)}"
        )

    scaler = StandardScaler()
    X = customer_features[feature_columns]

    X_scaled = scaler.fit_transform(X)

    scaled_features = pd.DataFrame(
        X_scaled,
        columns=feature_columns,
        index=customer_features.index,
    )

    return scaled_features, scaler
