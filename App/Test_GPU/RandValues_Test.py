######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import cupy as cp                   # CPU math structures.
import sys                          # syscalls.
import io                           # Input-Output.
from ParetoInsight_GPU.RandValues import (
    RandIndexSolutionsCupy
)

class TestRandIndexSolutionsCupy(unittest.TestCase):
    def setUp(self):
        # Tres soluciones para 4 genes
        self.cpu_matrix = np.array([
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1]
        ])
        self.gpu_matrix = cp.array(self.cpu_matrix)

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()           

    def test_shape_and_type(self):
        result = RandIndexSolutionsCupy(self.gpu_matrix)
        self.assertEqual(result.shape, (3,3))
        self.assertIsInstance(result, cp.ndarray)

    def test_diag_ones(self):
        result = RandIndexSolutionsCupy(self.gpu_matrix)
        diag = cp.asnumpy(cp.diag(result))
        self.assertTrue(np.allclose(diag, 1.0))

    def test_symmetry(self):
        result = RandIndexSolutionsCupy(self.gpu_matrix)
        mat = cp.asnumpy(result)
        self.assertTrue(np.allclose(mat, mat.T))

    def test_bounds(self):
        result = RandIndexSolutionsCupy(self.gpu_matrix)
        mat = cp.asnumpy(result)
        self.assertTrue(np.all((mat >= 0.0) & (mat <= 1.0)))

    def test_empty_matrix(self):
        with self.assertRaises(RuntimeError):
            RandIndexSolutionsCupy(cp.empty((0, 4)))

    def test_too_few_genes(self):
        with self.assertRaises(RuntimeError):
            RandIndexSolutionsCupy(cp.array([[1], [0]]))

    def test_identical_solutions(self):
        mat = cp.array([[1,1,1,1],[1,1,1,1]])
        res = RandIndexSolutionsCupy(mat)
        self.assertTrue(cp.allclose(res, cp.ones_like(res)))

    def test_completely_different_solutions(self):
        mat = cp.array([[0,1,2,3],[4,5,6,7]])
        res = RandIndexSolutionsCupy(mat)
        arr = cp.asnumpy(res)
        self.assertTrue(np.all(arr >= 0.0))
        self.assertTrue(np.all(arr <= 1.0))
        self.assertTrue(np.allclose(np.diag(arr), 1.0))

if __name__ == "__main__":
    unittest.main()
