import pandas as pd
from sklearn.mixture import GaussianMixture


def evaluate_gmm_models(
    X,
    cluster_range=range(2, 11),
    covariance_type="full",
    random_state=42,
):
    """
    Fit GMM models across a range of cluster counts and calculate AIC/BIC.

    Parameters
    ----------
    X : pandas.DataFrame or array-like
        Feature matrix used for clustering.
    cluster_range : iterable of int
        Candidate numbers of GMM components.
    covariance_type : str
        Covariance structure passed to GaussianMixture.
    random_state : int
        Random seed used by GaussianMixture.

    Returns
    -------
    model_selection : pandas.DataFrame
        AIC and BIC values for each candidate cluster count.
    models : dict
        Fitted GMM models keyed by number of components.
    """
    models = {}
    results = []

    for k in cluster_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            random_state=random_state,
        )

        gmm.fit(X)

        models[k] = gmm
        results.append(
            {
                "Clusters": k,
                "AIC": gmm.aic(X),
                "BIC": gmm.bic(X),
            }
        )

    model_selection = pd.DataFrame(results)

    return model_selection, models


def fit_final_gmm(
    X,
    n_components=10,
    covariance_type="full",
    random_state=42,
):
    """
    Fit the final GMM model used for customer segmentation.

    Parameters
    ----------
    X : pandas.DataFrame or array-like
        Feature matrix used for clustering.
    n_components : int
        Number of GMM components.
    covariance_type : str
        Covariance structure passed to GaussianMixture.
    random_state : int
        Random seed used by GaussianMixture.

    Returns
    -------
    GaussianMixture
        Fitted GMM model.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )

    gmm.fit(X)

    return gmm


def assign_clusters(X, model):
    """
    Assign each observation to a GMM cluster.

    Parameters
    ----------
    X : pandas.DataFrame or array-like
        Feature matrix.
    model : GaussianMixture
        Fitted GMM model.

    Returns
    -------
    numpy.ndarray
        Predicted cluster labels.
    """
    return model.predict(X)


def build_cluster_profiles(customer_features, cluster_labels):
    """
    Build mean behavioral profiles for each customer cluster.

    Parameters
    ----------
    customer_features : pandas.DataFrame
        Unscaled customer-level behavioral features.
    cluster_labels : array-like
        Cluster assignment for each customer.

    Returns
    -------
    pandas.DataFrame
        Mean behavioral feature values by cluster.
    """
    required_features = [
        "NumPurchases",
        "TotalQuantity",
        "TotalSpending",
        "AvgUnitPrice",
    ]

    missing = [
        column
        for column in required_features
        if column not in customer_features.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required customer features: {', '.join(missing)}"
        )

    if len(customer_features) != len(cluster_labels):
        raise ValueError(
            "Number of customer records must match number of cluster labels."
        )

    data = customer_features[required_features].copy()
    data["Cluster"] = cluster_labels

    return (
        data.groupby("Cluster")[required_features]
        .mean()
    )
