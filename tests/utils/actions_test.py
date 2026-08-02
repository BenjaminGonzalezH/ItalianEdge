"""Unit tests for Actions module.

These tests validate matrix persistence, loading behavior,
and DataFrame export functionality across supported formats.
"""

import unittest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from biocluster.utils.actions import (
    save_matrix,
    load_matrix,
    save_dataframe,
    ensure_parent_dir,
    MatrixSaveOptions,
    MatrixSaveMode,
)


class TestMatrixIO(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        self.matrix = np.array([[1, 2], [3, 4]])
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_npy(self):
        filepath = self.base / "matrix"

        save_matrix(self.matrix, filepath)

        loaded = load_matrix(filepath)

        np.testing.assert_array_equal(self.matrix, loaded)

    def test_save_and_load_npz(self):
        filepath = self.base / "matrix"
        opts = MatrixSaveOptions(mode=MatrixSaveMode.COMPRESSED_NPZ)

        save_matrix(self.matrix, filepath, opts)

        loaded = load_matrix(filepath)

        np.testing.assert_array_equal(self.matrix, loaded)

    def test_save_and_load_csv(self):
        filepath = self.base / "matrix"
        opts = MatrixSaveOptions(mode=MatrixSaveMode.TEXT_CSV)

        save_matrix(self.matrix, filepath, opts)

        loaded = np.loadtxt(filepath.with_suffix(".csv"), delimiter=",")

        np.testing.assert_array_equal(self.matrix, loaded)

    def test_invalid_matrix_type(self):
        with self.assertRaises(TypeError):
            save_matrix([[1, 2], [3, 4]], self.base / "bad")

    def test_invalid_save_mode(self):
        filepath = self.base / "matrix"

        class FakeMode:
            value = "bad"

        opts = MatrixSaveOptions(mode=FakeMode())

        with self.assertRaises(ValueError):
            save_matrix(self.matrix, filepath, opts)

    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_matrix(self.base / "missing")

    def test_npz_missing_key(self):
        filepath = self.base / "matrix.npz"

        np.savez_compressed(filepath, other=np.array([1, 2]))

        with self.assertRaises(KeyError):
            load_matrix(filepath)

    def test_display_flag(self):
        filepath = self.base / "matrix"

        save_matrix(self.matrix, filepath)

        loaded = load_matrix(filepath, display=True)

        np.testing.assert_array_equal(self.matrix, loaded)


class TestDataFrameIO(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name)
        logging.disable(logging.CRITICAL)

        self.df = pd.DataFrame({
            "A": [1, 2],
            "B": [3, 4]
        })

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_csv(self):
        path = self.base / "df"

        save_dataframe(self.df, path, format="csv")

        loaded = pd.read_csv(path.with_suffix(".csv"))

        pd.testing.assert_frame_equal(self.df, loaded)

    def test_save_excel(self):
        path = self.base / "df"

        save_dataframe(self.df, path, format="excel")

        loaded = pd.read_excel(path.with_suffix(".xlsx"))

        pd.testing.assert_frame_equal(self.df, loaded)

    def test_save_parquet(self):
        path = self.base / "df"

        save_dataframe(self.df, path, format="parquet")

        loaded = pd.read_parquet(path.with_suffix(".parquet"))

        pd.testing.assert_frame_equal(self.df, loaded)

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            save_dataframe(self.df, self.base / "df", format="json")

    def test_invalid_dataframe_type(self):
        with self.assertRaises(TypeError):
            save_dataframe([1, 2, 3], self.base / "df")

    def test_extension_correction(self):
        path = self.base / "df.txt"

        save_dataframe(self.df, path, format="csv")

        self.assertTrue((self.base / "df.csv").exists())


class TestUtilities(unittest.TestCase):

    def test_ensure_parent_dir(self):

        with tempfile.TemporaryDirectory() as tmp:

            path = Path(tmp) / "nested" / "file.npy"

            ensure_parent_dir(path)

            self.assertTrue((Path(tmp) / "nested").exists())


if __name__ == "__main__":
    unittest.main()