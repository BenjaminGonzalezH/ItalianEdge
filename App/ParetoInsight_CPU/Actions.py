"""
Utilities for saving and loading matrices and DataFrames.

Purpose of this file:
- Provide helper functions to persist NumPy matrices to disk.
- Provide helper functions to load and optionally display stored matrices.
- Provide helper functions to export pandas DataFrames in common formats.
"""

######### Libraries #########
from dataclasses import dataclass                   # Decorator to automatically generate special methods (e.g., __init__).
from enum import Enum                               # Define enumerations for controlled string values.
from pathlib import Path                            # Object-oriented filesystem path handling.
from typing import Any, Optional, Union             # Improve type hints and function signatures.
import logging                                      # Advanced logging system for status and error messages.
import numpy as np                                  # Efficient numerical computations.
import pandas as pd                                 # DataFrame manipulation and data analysis.
import matplotlib                                   # Plot configuration (backend management).


######### Configurations #########
matplotlib.use("Agg")                               # Use non-GUI backend (thread-safe for headless environments).
logger = logging.getLogger(__name__)                # Initialize module-level logger.


######### Classes #########

class MatrixSaveMode(str, Enum):
    """Enumeration of supported matrix file formats."""
    BINARY_NPY = "npy"        # np.save -> .npy (fastest and native NumPy format)
    COMPRESSED_NPZ = "npz"    # np.savez_compressed -> .npz (compressed archive)
    TEXT_CSV = "csv"          # np.savetxt -> .csv (portable text format)


@dataclass(frozen=True)
class MatrixSaveOptions:
    """
    Configuration options for saving matrices.
    
    Attributes:
        mode: File format used to save the matrix.
        delimiter: Delimiter used when saving as text.
        fmt: Format string for text-based numeric output.
        allow_pickle: Applied during loading (not saving).
        verbose: If True, prints additional status messages.
    """
    mode: MatrixSaveMode = MatrixSaveMode.BINARY_NPY
    delimiter: str = ","
    fmt: str = "%.6f"
    allow_pickle: bool = False  # Only relevant for loading.
    verbose: bool = False


class DataFrameFormat(str, Enum):
    """Enumeration of supported DataFrame export formats."""
    CSV = "csv"
    EXCEL = "excel"
    PARQUET = "parquet"


######### Data Types #########
PathLike = Union[str, Path]     # Accept both string and Path objects.


######### Internal Functions #########

def _as_path(path: PathLike) -> Path:
    """Ensure the input is converted to a Path object."""
    return path if isinstance(path, Path) else Path(path)


def _log_or_print(msg: str, verbose: bool = False) -> None:
    """
    Library-friendly output handler:
    - Always logs the message using the module logger.
    - Optionally prints the message if verbose=True.
    """
    logger.info(msg)
    if verbose:
        print(msg)


######### Main Functions #########

def ensure_parent_dir(filepath: PathLike) -> Path:
    """
    Ensure that the parent directory of `filepath` exists.
    
    If it does not exist, it will be created automatically.
    
    Returns:
        Resolved Path object for convenience.
    """
    p = _as_path(filepath)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    return p

def save_matrix(
    matrix: np.ndarray,
    filepath: PathLike,
    options: MatrixSaveOptions = MatrixSaveOptions(),
) -> Path:
    """
    Save a NumPy array to disk in different formats.

    Supported modes:
        - "npy": Saves as .npy (fastest and native NumPy format).
        - "npz": Saves as compressed .npz (smaller file size).
        - "csv": Saves as delimited text (portable but slower).

    Args:
        matrix: NumPy array to save.
        filepath: Output file path.
        options: Configuration options for saving.

    Returns:
        Path to the created file.

    Raises:
        TypeError: If matrix is not a NumPy array.
        ValueError: If an unsupported save mode is provided.
        OSError: If a file system error occurs.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"`matrix` must be a numpy.ndarray, got: {type(matrix)}")

    p = ensure_parent_dir(filepath)

    # Normalize suffix based on mode if user omitted it
    desired_suffix = f".{options.mode.value}"
    if p.suffix.lower() != desired_suffix:
        p = p.with_suffix(desired_suffix)

    try:
        if options.mode == MatrixSaveMode.BINARY_NPY:
            np.save(p, matrix)
            # np.save keeps suffix; if p ends with .npy it won't double it
            _log_or_print(f"Matrix saved (NPY) at: {p}", options.verbose)

        elif options.mode == MatrixSaveMode.COMPRESSED_NPZ:
            # store under a stable key for easy load
            np.savez_compressed(p, matrix=matrix)
            _log_or_print(f"Matrix saved (NPZ compressed) at: {p}", options.verbose)

        elif options.mode == MatrixSaveMode.TEXT_CSV:
            np.savetxt(p, matrix, delimiter=options.delimiter, fmt=options.fmt)
            _log_or_print(f"Matrix saved (text) at: {p}", options.verbose)

        else:
            raise ValueError(f"Unsupported mode: {options.mode}")

        return p

    except OSError as e:
        logger.exception("I/O error while saving matrix.")
        raise
    except Exception:
        logger.exception("Unexpected error while saving matrix.")
        raise


def load_matrix(
    filepath: PathLike,
    allow_pickle: bool = False,
    key: str = "matrix",
    display: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Load a matrix saved as .npy or .npz.

    Args:
        filepath: path to the file (.npy or .npz). If suffix is missing,
                  it will try .npy then .npz.
        allow_pickle: forwarded to np.load (use False by default for safety).
        key: for .npz files, which key to read from (default "matrix").
        display: if True, prints the matrix (useful for debugging).
        verbose: if True, prints status messages.

    Returns:
        numpy.ndarray

    Raises:
        FileNotFoundError, ValueError, KeyError
    """
    p = _as_path(filepath)

    candidates = []
    if p.suffix:
        candidates = [p]
    else:
        candidates = [p.with_suffix(".npy"), p.with_suffix(".npz")]

    found: Optional[Path] = None
    for c in candidates:
        if c.exists():
            found = c
            break

    if found is None:
        raise FileNotFoundError(
            f"Matrix file not found. Tried: {', '.join(str(c) for c in candidates)}"
        )

    try:
        data = np.load(found, allow_pickle=allow_pickle)

        if isinstance(data, np.lib.npyio.NpzFile):
            if key not in data.files:
                raise KeyError(
                    f"Key '{key}' not found in NPZ file. Available: {data.files}"
                )
            matrix = data[key]
        else:
            matrix = data

        _log_or_print(f"Matrix loaded from: {found}", verbose)

        if display:
            print(matrix)

        return matrix

    except Exception:
        logger.exception("Error while loading matrix.")
        raise

def save_dataframe(
    dataframe: pd.DataFrame,
    filepath: PathLike,
    format: str = "csv",
    verbose: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Save a pandas DataFrame using CSV, Excel, or Parquet.

    Args:
        dataframe: DataFrame to save.
        filepath: output path.
        format: "csv" | "excel" | "parquet"
        verbose: if True, prints status.
        **kwargs: forwarded to the underlying pandas writer.

    Returns:
        Path to the created file.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"`dataframe` must be a pandas.DataFrame, got: {type(dataframe)}")

    fmt = format.lower().strip()
    try:
        df_format = DataFrameFormat(fmt)
    except ValueError:
        raise ValueError("Unsupported format. Use 'csv', 'excel', or 'parquet'.")

    p = ensure_parent_dir(filepath)

    # Normalize file extension if missing/wrong
    if df_format == DataFrameFormat.CSV:
        if p.suffix.lower() != ".csv":
            p = p.with_suffix(".csv")
        dataframe.to_csv(p, index=False, **kwargs)
        _log_or_print(f"Results saved as CSV at: {p}", verbose)

    elif df_format == DataFrameFormat.EXCEL:
        if p.suffix.lower() not in (".xlsx", ".xls"):
            p = p.with_suffix(".xlsx")
        # Provide a safe default engine if user didn't set it
        kwargs.setdefault("engine", "openpyxl")
        dataframe.to_excel(p, index=False, **kwargs)
        _log_or_print(f"Results saved as Excel at: {p}", verbose)

    elif df_format == DataFrameFormat.PARQUET:
        if p.suffix.lower() != ".parquet":
            p = p.with_suffix(".parquet")
        dataframe.to_parquet(p, index=False, **kwargs)
        _log_or_print(f"Results saved as Parquet at: {p}", verbose)

    return p
