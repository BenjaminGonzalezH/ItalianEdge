"""Utilities to read solution matrices from disk.

This module supports CSV, fixed-width text, and pickle files and returns:
1) A NumPy matrix with the loaded values.
2) The list of column names (genes symbos or ID).
"""

######### Libraries #########
import numpy as np                                          # Efficient Math Operations.
import pandas as pd                                         # Dataframe managment.
import csv                                                  # Read csv.
from pathlib import Path                                    # Confort about paths managments.
from typing import Tuple                                    # Improve functions specs.


######### Internal Functions #########

def _clean_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    """Validate and normalize a DataFrame before converting to NumPy."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Loaded object is not a pandas DataFrame.")

    # Remove index columns often introduced by CSV exports.
    df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False)]

    if df.empty:
        raise ValueError("DataFrame is empty.")

    return df.to_numpy(), list(df.columns)


def _read_csv(filepath: str) -> pd.DataFrame:
    """Read CSV while trying to auto-detect delimiter."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            sample = f.read(2048)
            delimiter = csv.Sniffer().sniff(sample).delimiter
    except Exception:
        # Fallback to comma if detection fails.
        delimiter = ","

    return pd.read_csv(filepath, sep=delimiter)


def _read_fwf(filepath: str) -> pd.DataFrame:
    """Read fixed-width formatted text."""
    return pd.read_fwf(filepath)


def _read_pkl(filepath: str) -> pd.DataFrame:
    """Read a pickled pandas DataFrame."""
    return pd.read_pickle(filepath)


######### Constanst #########
# Reader dispatch table by extension / declared format.
READERS = {
    "csv": _read_csv,
    "fwf": _read_fwf,
    "txt": _read_fwf,
    "pkl": _read_pkl,
}

######### Main Function #########
def read_solutions_file(filepath: str) -> Tuple[np.ndarray, list[str]]:
    """Read a solutions file using its filename extension."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower().replace(".", "")

    if suffix not in READERS:
        raise ValueError(f"Unsupported format: {suffix}")

    df = READERS[suffix](filepath)
    return _clean_dataframe(df)