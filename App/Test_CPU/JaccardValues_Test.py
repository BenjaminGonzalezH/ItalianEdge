######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import pandas as pd                 # Dataframes managment.
import sys                          # syscalls.
import io                           # Input-Output 
from ParetoInsight_CPU.JaccardValues import (
    JaccardIndexSolutions,
    JaccardIndexClusters,
    CompareSolutionsPair,
    FindEquivalentClusters
)

class TestClusteringJaccard(unittest.TestCase):
    
    ########################## Test's Initialization ##########################
    def setUp(self):
        self.matrix = np.array([
            [0, 0, 1, 1],  # Solution 1: [0,0], [1,1]
            [1, 1, 0, 0],  # Solution 2: [1,1], [0,0]
            [0, 1, 0, 1]   # Solution 3: [0,1], [0,1]
        ])
        # equivalent sets of the previous solutions.
        self.solutions_sets = [
            [set(['a', 'b']), set(['c', 'd'])],
            [set(['a', 'c']), set(['b', 'd'])]
        ]

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    ########################## Tests ##########################
    def test_jaccard_index_solutions(self):
        jaccard = JaccardIndexSolutions(self.matrix, n_threads=2)
        self.assertTrue(np.allclose(np.diag(jaccard), 1.0))
        self.assertTrue(np.allclose(jaccard, jaccard.T))

    def test_jaccard_index_clusters(self):
        result = JaccardIndexClusters(self.solutions_sets[0], self.solutions_sets[1])
        self.assertEqual(result.shape, (2, 2))

    def test_compare_solutions_pair(self):
        out = CompareSolutionsPair(0, 1, self.solutions_sets * 2)
        self.assertIsInstance(out, list)
        self.assertTrue(all(isinstance(x, tuple) and len(x) == 3 for x in out))

    def test_find_equivalent_clusters(self):
        df = FindEquivalentClusters(self.solutions_sets * 2)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(
            set(df.columns),
            {'Solution 1', 'Solution 2', 'Cluster 1', 'Cluster 2', 'Jaccard Similarity'}
        )

    def test_jaccard_index_solutions_empty(self):
        with self.assertRaises(RuntimeError):
            JaccardIndexSolutions(np.empty((0, 4)), n_threads=2)

    def test_jaccard_index_solutions_one_gene(self):
        with self.assertRaises(RuntimeError):
            JaccardIndexSolutions(np.array([[0], [1]]), n_threads=2)

    def test_jaccard_index_clusters_type_error(self):
        with self.assertRaises(RuntimeError):
            JaccardIndexClusters([1, 2], [set([1])])

    def test_jaccard_index_clusters_empty(self):
        with self.assertRaises(RuntimeError):
            JaccardIndexClusters([], [set([1])])

if __name__ == "__main__":
    unittest.main()