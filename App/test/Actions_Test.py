######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import pandas as pd                 # Dataframe managment.
import os                           # OS calls.
import io                           # Input-Output 
import sys                          # syscalls.
from CoMOcG.Actions import (
    save_matrix, 
    save_matrix_uncompresed, 
    load_and_display_matrix, 
    plot_html_heatmap, 
    save_dataframe, 
    Pairs_Ordered
)

class TestMatrixFunctions(unittest.TestCase):

    def setUp(self):
        # Create test data.
        self.matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.filepath_npy = "test_data/test_matrix"
        self.filepath_csv = "test_data/test_dataframe.csv"
        self.filepath_txt = "test_data/test_matrix.txt"
        self.filepath_html = "test_data/test_heatmap.html"
        self.df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4]
        })

        # Create test directory.
        os.makedirs("test_data", exist_ok=True)

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def test_save_and_load_matrix(self):
        save_matrix(self.matrix, self.filepath_npy)
        loaded = load_and_display_matrix(f"{self.filepath_npy}.npy")
        self.assertTrue(np.array_equal(self.matrix, loaded))

    def test_save_matrix_uncompressed(self):
        save_matrix_uncompresed(self.matrix, self.filepath_txt)
        loaded_txt = np.loadtxt(self.filepath_txt, delimiter=",")
        self.assertTrue(np.allclose(self.matrix, loaded_txt))

    def test_save_dataframe_csv(self):
        save_dataframe(self.df, self.filepath_csv, "csv")
        loaded_df = pd.read_csv(self.filepath_csv)
        pd.testing.assert_frame_equal(self.df, loaded_df)

    def test_plot_html_heatmap(self):
        plot_html_heatmap(self.matrix, self.filepath_html)
        self.assertTrue(os.path.exists(self.filepath_html))

    def test_pairs_ordered_default(self):
        mat = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        result = Pairs_Ordered(mat)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertListEqual(list(result.columns), ['solution_ID_1', 'solution_ID_2', 'value'])
        self.assertEqual(len(result), 6)

    def test_pairs_ordered_with_desc(self):
        mat = np.array([
            [0, 2],
            [2, 0]
        ])
        result = Pairs_Ordered(mat, desc=True)
        self.assertGreaterEqual(result.iloc[0]['value'], result.iloc[-1]['value'])

    def test_pairs_ordered_error_handling(self):
        result = Pairs_Ordered(np.array([1, 2, 3]))  # Not square
        self.assertTrue(result.empty)

    def tearDown(self):
        # Clean every file.
        for f in [
            f"{self.filepath_npy}.npy", 
            self.filepath_csv, 
            self.filepath_txt, 
            self.filepath_html
        ]:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists("test_data"):
            os.rmdir("test_data")

        # Activate prints.
        sys.stdout = self._original_stdout

# Execution.
if __name__ == "__main__":
    unittest.main()