######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import sys                          # syscalls.
import io                           # Input-Output 
from ParetoInsight_CPU.RandValues import (
    RandIndexSolutions
)

class TestRandIndexSolutions(unittest.TestCase):

    ########################## Test's Initialization ##########################
    def setUp(self):
        # Tres soluciones distintas de agrupamiento de 4 genes
        self.matrix = np.array([
            [0, 0, 1, 1],  # Solución 1: grupos [0,1], [2,3]
            [1, 1, 0, 0],  # Solución 2: grupos invertidos respecto a la 1
            [0, 1, 0, 1]   # Solución 3: grupos alternados
        ])

        self.diferents = np.array([
            [0, 1, 2, 3],
            [3, 2, 1, 0]
        ])

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()        

    ########################## Tests ##########################
    def test_regular_behavior(self):
        rand = RandIndexSolutions(self.matrix, n_threads=2)
        self.assertTrue(np.allclose(rand, rand.T))
        self.assertTrue(np.allclose(np.diag(rand), 1.0))
        self.assertEqual(rand.shape, (3, 3))
        self.assertTrue(np.all((rand >= 0) & (rand <= 1)))

    def test_empty_matrix(self):
        with self.assertRaises(RuntimeError):
            RandIndexSolutions(np.empty((0, 4)), n_threads=2)

    def test_insufficient_columns(self):
        with self.assertRaises(RuntimeError):
            RandIndexSolutions(np.zeros((2, 1)), n_threads=2)

    def test_identical_solutions(self):
        matrix = np.tile([2, 2, 1, 1], (2, 1))
        rand = RandIndexSolutions(matrix, n_threads=2)
        self.assertTrue(np.allclose(rand, 1.0))

    def test_completely_different_solutions(self):
        rand = RandIndexSolutions(self.diferents, n_threads=2)
        self.assertTrue(np.all(rand <= 1.0))
        self.assertTrue(np.all(rand >= 0.0))
        self.assertTrue(np.allclose(np.diag(rand), 1.0))

if __name__ == "__main__":
    unittest.main()
