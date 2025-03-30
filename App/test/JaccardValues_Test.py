######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import pandas as pd                 # Dataframe managment.
import sys                          # syscalls.
import io                           # Input-Output 
from CoMOcG.JaccardValues import (
    compute_jaccard,
    process_JaccardValues,
    Jaccar_similarityClusters,
    compute_rand_index,
    process_RandValues,
    compare_solution_pair,
    find_equivalent_clusters
)

class TestClusteringSimilarity(unittest.TestCase):

    def setUp(self):
        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def tearDown(self):
        # Activate prints.
        sys.stdout = self._original_stdout

    def test_compute_jaccard_basic(self):
        A = np.array([1, 2, 1])
        B = np.array([1, 2, 2])
        score = compute_jaccard(A, B)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_compute_jaccard_shape_mismatch(self):
        A = np.array([1, 2])
        B = np.array([1, 2, 3])
        result = compute_jaccard(A, B)
        self.assertIsNone(result)

    def test_process_JaccardValues(self):
        Matrix = [np.array([1, 2, 1]), np.array([1, 1, 2])]
        result = process_JaccardValues(Matrix, n_threads=2)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.allclose(result[0, 0], 1.0))

    def test_Jaccar_similarityClusters(self):
        S1 = [{0, 1}, {2}]
        S2 = [{1, 2}, {0}]
        result = Jaccar_similarityClusters(S1, S2)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_compute_rand_index(self):
        A = np.array([0, 1, 0])
        B = np.array([1, 1, 0])
        result = compute_rand_index(A, B, adjusted=False)
        self.assertTrue(0 <= result <= 1)

    def test_process_RandValues(self):
        Matrix = [np.array([0, 1, 0]), np.array([1, 1, 0])]
        result = process_RandValues(Matrix, n_threads=2, adjusted=False)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue((result >= 0).all() and (result <= 1).all())

    def test_compare_solution_pair(self):
        S1 = [[{0, 1}, {2}], [{1, 2}, {0}]]
        result = compare_solution_pair(0, 1, S1)
        self.assertTrue(isinstance(result, list))
        self.assertTrue(all(len(t) == 3 for t in result))

    def test_find_equivalent_clusters(self):
        S1 = [[{0, 1}, {2}], [{1, 2}, {0}], [{0, 2}, {1}]]
        result = find_equivalent_clusters(S1)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Solution Pair", result.columns)
        self.assertIn("Equivalent Clusters", result.columns)
        self.assertIn("Jaccard Similarities", result.columns)

if __name__ == '__main__':
    unittest.main()