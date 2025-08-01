import unittest
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from ParetoInsight_CPU.JaccardValues import (
    JaccardIndexSolutions,
    JaccardIndexClusters,
    CompareSolutionsPair,
    FindEquivalentClusters
)

class TestClusteringJaccard(unittest.TestCase):
    def setUp(self):
        # Matriz de soluciones artificial, cada fila es una solución, cada columna un gen
        # Soluciones: Dos agrupaciones diferentes de 4 genes
        self.matrix = np.array([
            [0, 0, 1, 1],  # Solución 1: [0,0], [1,1]
            [1, 1, 0, 0],  # Solución 2: [1,1], [0,0]
            [0, 1, 0, 1]   # Solución 3: [0,1,0,1]
        ])
        # Representación directa de clústeres como listas de sets
        self.solutions_sets = [
            [set(['a', 'b']), set(['c', 'd'])],
            [set(['a', 'c']), set(['b', 'd'])]
        ]

    def test_jaccard_index_solutions(self):
        jaccard = JaccardIndexSolutions(self.matrix, n_threads=2)
        # Diagonal en 1 y simetría
        self.assertTrue(np.allclose(np.diag(jaccard), 1.0))
        self.assertTrue(np.allclose(jaccard, jaccard.T))

    def test_jaccard_index_clusters(self):
        result = JaccardIndexClusters(self.solutions_sets[0], self.solutions_sets[1])
        self.assertEqual(result.shape, (2, 2))
        # Por ejemplo, todos intersectan al menos en un elemento

    def test_compare_solutions_pair(self):
        out = CompareSolutionsPair(0, 1, self.solutions_sets * 2)
        self.assertIsInstance(out, list)
        # Cada elemento es tupla con dos índices y una similitud
        self.assertTrue(all(isinstance(x, tuple) and len(x) == 3 for x in out))

    def test_find_equivalent_clusters(self):
        # Compara ambas soluciones entre sí
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