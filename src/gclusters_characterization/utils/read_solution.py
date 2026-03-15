"""ReadSolution

This module provides utilities to load clustering solution matrices from disk.
It supports multiple file formats and converts them into a standardized
NumPy matrix along with the list of column names.

Functions
1. read_solutions_file – Load a solutions file and return the matrix and gene names.
2. _clean_dataframe – Validate and normalize a DataFrame before conversion.
3. _read_csv – Read CSV files with automatic delimiter detection.
4. _read_fwf – Read fixed-width formatted text files.
5. _read_pkl – Load a pickled pandas DataFrame.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from typing import Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Internal functions
# ──────────────────────────────────────────────────────────────────────────────

def _clean_dataframe(df: pd.DataFrame) -> Tuple[np.ndarray, list[str]]:
    """
    Validate and normalize a DataFrame before converting it to NumPy.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame loaded from an external file.

    Returns
    -------
    matrix : numpy.ndarray
        Numerical matrix extracted from the DataFrame.
    columns : list[str]
        List of column names after cleaning.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Loaded object is not a pandas DataFrame.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    df = df.loc[:, ~df.columns.str.contains("Unnamed", case=False)]

    if df.shape[1] == 0:
        raise ValueError("No usable columns after removing 'Unnamed' columns.")

    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().all().all():
        raise ValueError("All values became NaN after numeric conversion.")

    return df.to_numpy(), list(df.columns)


def _read_csv(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file while attempting to detect the delimiter.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the parsed CSV content.
    """

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            sample = f.read(2048)
            delimiter = csv.Sniffer().sniff(sample).delimiter
    except Exception:
        # Fallback to comma if detection fails.
        delimiter = ","

    return pd.read_csv(filepath, sep=delimiter)


def _read_fwf(filepath: str) -> pd.DataFrame:
    """
    Read a fixed-width formatted text file.

    Parameters
    ----------
    filepath : str
        Path to the text file.

    Returns
    -------
    pandas.DataFrame
        Parsed DataFrame.
    """
    return pd.read_fwf(filepath)


def _read_pkl(filepath: str) -> pd.DataFrame:
    """
    Load a pickled pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the pickle file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame.
    """
    return pd.read_pickle(filepath)


# ──────────────────────────────────────────────────────────────────────────────
# Reader registry
# ──────────────────────────────────────────────────────────────────────────────

READERS = {
    "csv": _read_csv,
    "fwf": _read_fwf,
    "txt": _read_fwf,
    "pkl": _read_pkl,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main API
# ──────────────────────────────────────────────────────────────────────────────

def read_solutions_file(filepath: str) -> Tuple[np.ndarray, list[str]]:
    """
    Load a solutions file and convert it to a matrix representation.

    The file format is inferred from its extension. Supported formats
    include CSV, fixed-width text, and pickled pandas DataFrames.

    Parameters
    ----------
    filepath : str
        Path to the solutions file.

    Returns
    -------
    matrix : numpy.ndarray
        Matrix containing the solution values.
    columns : list[str]
        List of column names representing genes or identifiers.
    """

    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = path.suffix.lower().replace(".", "")

    if suffix not in READERS:
        raise ValueError(f"Unsupported format: {suffix}")

    df = READERS[suffix](filepath)

    return _clean_dataframe(df)