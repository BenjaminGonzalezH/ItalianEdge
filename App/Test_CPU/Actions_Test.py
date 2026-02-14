"""
Unit tests for Actions.py

Purpose of this file:
- Validate matrix save/load behavior across supported formats.
- Validate DataFrame export formats.
- Verify expected errors are raised for invalid inputs.
- Ensure no residual files are left after execution.
"""

######### Libraries #########
import unittest                     # Test framework.
import tempfile                     # Use of temporal files.
import numpy as np                  # Efficient numerical computations.
import pandas as pd                 # DataFrame manipulation and data analysis.
from pathlib import Path            # Object-oriented filesystem path handling.

# Imports from the librarie to test.
from ParetoInsight_CPU.Actions import (
    save_matrix,
    load_matrix,
    save_dataframe,
    MatrixSaveOptions,
    MatrixSaveMode,
)


class TestMatrixIO(unittest.TestCase):
    """Test suite for matrix save and load functionality."""

    def setUp(self):
        """Create temporary directory and sample matrix."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.matrix = np.array([[1, 2], [3, 4]])

    def tearDown(self):
        """Cleanup temporary directory automatically."""
        self.temp_dir.cleanup()

    ##########################
    # Matrix Save Tests
    ##########################

    def test_save_and_load_npy(self):
        """Confirm .npy save and load works correctly."""
        filepath = self.base_path / "matrix"
        save_matrix(self.matrix, filepath)

        loaded = load_matrix(filepath)
        np.testing.assert_array_equal(self.matrix, loaded)

    def test_save_and_load_npz(self):
        """Confirm compressed .npz save and load works correctly."""
        filepath = self.base_path / "matrix"
        options = MatrixSaveOptions(mode=MatrixSaveMode.COMPRESSED_NPZ)

        save_matrix(self.matrix, filepath, options)
        loaded = load_matrix(filepath)

        np.testing.assert_array_equal(self.matrix, loaded)

    def test_save_and_load_csv(self):
        """Confirm text CSV save and load works correctly."""
        filepath = self.base_path / "matrix"
        options = MatrixSaveOptions(mode=MatrixSaveMode.TEXT_CSV)

        save_matrix(self.matrix, filepath, options)

        # CSV load requires suffix
        loaded = np.loadtxt(filepath.with_suffix(".csv"), delimiter=",")
        np.testing.assert_array_equal(self.matrix, loaded)

    def test_invalid_matrix_type(self):
        """Confirm non-numpy input raises TypeError."""
        with self.assertRaises(TypeError):
            save_matrix([[1, 2], [3, 4]], self.base_path / "bad")

    def test_load_nonexistent_file(self):
        """Confirm missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_matrix(self.base_path / "missing_file")


class TestDataFrameIO(unittest.TestCase):
    """Test suite for DataFrame save functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        self.df = pd.DataFrame({
            "A": [1, 2],
            "B": [3, 4]
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    ##########################
    # DataFrame Save Tests
    ##########################

    def test_save_csv(self):
        """Confirm DataFrame saves correctly as CSV."""
        filepath = self.base_path / "df"
        save_dataframe(self.df, filepath, format="csv")

        loaded = pd.read_csv(filepath.with_suffix(".csv"))
        pd.testing.assert_frame_equal(self.df, loaded)

    def test_save_excel(self):
        """Confirm DataFrame saves correctly as Excel."""
        filepath = self.base_path / "df"
        save_dataframe(self.df, filepath, format="excel")

        loaded = pd.read_excel(filepath.with_suffix(".xlsx"))
        pd.testing.assert_frame_equal(self.df, loaded)

    def test_save_parquet(self):
        """Confirm DataFrame saves correctly as Parquet."""
        filepath = self.base_path / "df"
        save_dataframe(self.df, filepath, format="parquet")

        loaded = pd.read_parquet(filepath.with_suffix(".parquet"))
        pd.testing.assert_frame_equal(self.df, loaded)

    def test_invalid_format(self):
        """Confirm unsupported format raises ValueError."""
        with self.assertRaises(ValueError):
            save_dataframe(self.df, self.base_path / "df", format="json")

    def test_invalid_dataframe_type(self):
        """Confirm non-DataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            save_dataframe([1, 2, 3], self.base_path / "df")


if __name__ == "__main__":
    unittest.main()
