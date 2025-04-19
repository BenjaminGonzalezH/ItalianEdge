######### Libraries #########
import unittest                         # Test interface.
import sys                              # syscalls.
import numpy as np                      # Numbers ADT managment.
import io                               # Input-Output
from CoMOcG.SolutionClusterMatrix import (
    ProcessSolution,
    SolutionClusterMatrix
)

######### Testing #########

class TestSolutionClusterMatrix(unittest.TestCase):

    def setUp(self):
        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def tearDown(self):
        # Activate prints.
        sys.stdout = self._original_stdout

    def test_process_solution_ids_basic(self):
        solution = np.array([1, 2, 1, 2])
        genes = ['GeneA', 'GeneB', 'GeneC', 'GeneD']
        clusters = ProcessSolution(solution, genes)

        # Esperado: dos clústeres: [GeneA, GeneC] y [GeneB, GeneD]
        expected_clusters = [{'GeneA', 'GeneC'}, {'GeneB', 'GeneD'}]

        # Comparación ignorando orden
        self.assertEqual(len(clusters), 2)
        self.assertTrue(any(cluster == expected_clusters[0] for cluster in clusters))
        self.assertTrue(any(cluster == expected_clusters[1] for cluster in clusters))

    def test_process_solution_ids_invalid_input(self):
        solution = "not a numpy array"
        genes = ['Gene1', 'Gene2']
        result = ProcessSolution(solution, genes)
        self.assertIsNone(result)

    def test_solution_cluster_matrix_gene_id(self):
        Matrix = [
            np.array([1, 2, 1]),
            np.array([2, 2, 1])
        ]
        genes = ['GeneX', 'GeneY', 'GeneZ']
        result = SolutionClusterMatrix(Matrix, genes, max_workers=2)

        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(sol, list) for sol in result))
        self.assertTrue(all(isinstance(cluster, set) for sol in result for cluster in sol))

        all_genes = set().union(*[set.union(*sol) for sol in result])
        self.assertEqual(all_genes, set(genes))

    def test_solution_cluster_matrix_empty(self):
        result = SolutionClusterMatrix([], [], max_workers=2)
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()