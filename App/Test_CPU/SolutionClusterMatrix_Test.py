######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import sys                          # syscalls.
import io                           # Input-Output 
from ParetoInsight_CPU.SolutionClusterMatrix import (
    ProcessSolution,
    SolutionClusterMatrix
)

class TestSolutionClustering(unittest.TestCase):

    ########################## Test's Initialization ##########################
    def setUp(self):
        self.solution = np.array([0, 0, 1, 1, 0])
        self.genes_str = ["GeneA", "GeneB", "GeneC", "GeneD", "GeneE"]
        self.genes_int = [101, 102, 103, 104, 105]

        self.matrix = np.array([
            [0, 0, 1, 1, 0],
            [1, 1, 2, 2, 1],
        ])

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()        

    ########################## Tests ##########################
    def test_process_solution_str_genes(self):
        clusters = ProcessSolution(self.solution, self.genes_str)
        # Deben ser 2 clusters
        self.assertEqual(len(clusters), 2)
        # Uno de los clusters debe contener "GeneA", "GeneB", "GeneE"
        found = any(set(["GeneA", "GeneB", "GeneE"]) == group for group in clusters)
        self.assertTrue(found)

    def test_process_solution_int_genes(self):
        clusters = ProcessSolution(self.solution, self.genes_int)
        self.assertEqual(len(clusters), 2)
        found = any(set([101, 102, 105]) == group for group in clusters)
        self.assertTrue(found)

    def test_solution_cluster_matrix(self):
        result = SolutionClusterMatrix(self.matrix, self.genes_str)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(clusters, list) for clusters in result))
        self.assertTrue(all(isinstance(cl, set) for group in result for cl in group))

    def test_empty_solution(self):
        result = ProcessSolution(np.array([]), self.genes_str)
        self.assertListEqual(result,[])

    def test_mismatch_length(self):
        with self.assertRaises(RuntimeError):
            ProcessSolution(np.array([0, 1, 2]), ["A", "B"])

    def test_matrix_shape_error(self):
        with self.assertRaises(Exception):
            SolutionClusterMatrix(np.array([1, 0, 1]), self.genes_str)

if __name__ == "__main__":
    unittest.main()
