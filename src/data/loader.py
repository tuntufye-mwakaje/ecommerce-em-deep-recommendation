from pathlib import Path
import pandas as pd


def load_csv(file_path):
    """
    Load a CSV dataset into a pandas DataFrame.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)
