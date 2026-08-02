"""Unit tests for ReadSolution module.

These tests verify correct file loading, error handling,
and normalization behavior for supported file formats.
"""

import unittest
import tempfile
import pandas as pd
import pickle

from biocluster.utils.read_solution import (
    read_solutions_file,
    _clean_dataframe,
)


class TestReadSolutionsFile(unittest.TestCase):

    def test_read_csv_basic(self):
        """Verify a small CSV file loads correctly."""

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("Gene1,Gene2\n1,2\n3,4\n")
            path = f.name

        matrix, genes = read_solutions_file(path)

        self.assertEqual(genes, ["Gene1", "Gene2"])
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(matrix[0, 0], 1)


    def test_semicolon_csv(self):
        """Verify delimiter auto-detection works."""

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("Gene1;Gene2\n1;2\n3;4\n")
            path = f.name

        matrix, genes = read_solutions_file(path)

        self.assertEqual(genes, ["Gene1", "Gene2"])
        self.assertEqual(matrix.shape, (2, 2))


    def test_pickle_dataframe(self):
        """Verify pickled DataFrames are loaded correctly."""

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df.to_pickle(path)

        matrix, genes = read_solutions_file(path)

        self.assertEqual(genes, ["A", "B"])
        self.assertEqual(matrix.shape, (2, 2))


    def test_pickle_not_dataframe(self):
        """Verify pickled objects that are not DataFrames raise TypeError."""

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        with open(path, "wb") as f:
            pickle.dump([1, 2, 3], f)

        with self.assertRaises(TypeError):
            read_solutions_file(path)


    def test_file_not_found(self):
        """Verify missing files raise FileNotFoundError."""

        with self.assertRaises(FileNotFoundError):
            read_solutions_file("missing_file.csv")


    def test_unsupported_format(self):
        """Verify unsupported extensions raise ValueError."""

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{}")
            path = f.name

        with self.assertRaises(ValueError):
            read_solutions_file(path)


    def test_clean_dataframe_removes_unnamed(self):
        """Verify automatic removal of 'Unnamed' columns."""

        df = pd.DataFrame({
            "Unnamed: 0": [0, 1],
            "Gene1": [1, 2],
            "Gene2": [3, 4],
        })

        matrix, genes = _clean_dataframe(df)

        self.assertEqual(genes, ["Gene1", "Gene2"])
        self.assertEqual(matrix.shape, (2, 2))


    def test_clean_dataframe_empty(self):
        """Verify empty DataFrames raise ValueError."""

        df = pd.DataFrame()

        with self.assertRaises(ValueError):
            _clean_dataframe(df)


    def test_clean_dataframe_invalid_type(self):
        """Verify invalid input types raise TypeError."""

        with self.assertRaises(TypeError):
            _clean_dataframe([1, 2, 3])

    def test_only_unnamed_columns(self):
        """Verify error when only 'Unnamed' columns exist."""

        df = pd.DataFrame({
            "Unnamed: 0": [1, 2],
            "Unnamed: 1": [3, 4]
        })

        with self.assertRaises(ValueError):
            _clean_dataframe(df)

    def test_all_values_become_nan(self):
        """Verify error when numeric conversion produces only NaN."""

        df = pd.DataFrame({
            "Gene1": ["A", "B"],
            "Gene2": ["C", "D"]
        })

        with self.assertRaises(ValueError):
            _clean_dataframe(df)

if __name__ == "__main__":
    unittest.main()