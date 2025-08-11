######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import cupy as cp                   # CPU math structures.
import sys                          # syscalls.
import io                           # Input-Output.
from ParetoInsight_GPU.JaccardValues import (
    JaccardIndexSolutions
)

class TestJaccardIndexSolutions(unittest.TestCase):
    
    ########################## Test's Initialization ##########################
    def setUp(self):
        self.cpu_matrix = np.array([
            [0, 0, 1, 1],   # Solution 1: [0,0], [1,1]
            [1, 1, 0, 0],   # Solution 2: [1,1], [0,0]
            [0, 1, 0, 1]    # Solution 3: [0,1], [0,1]
        ])
        self.gpu_matrix = cp.array(self.cpu_matrix)
        
        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()        

    def test_shape_and_type(self):
        result = JaccardIndexSolutions(self.gpu_matrix)
        self.assertEqual(result.shape, (3,3))
        self.assertIsInstance(result, cp.ndarray)
    
    def test_diag_ones(self):
        result = JaccardIndexSolutions(self.gpu_matrix)
        diag = cp.asnumpy(cp.diag(result))
        self.assertTrue(np.allclose(diag, 1.0))
    
    def test_symmetry(self):
        result = JaccardIndexSolutions(self.gpu_matrix)
        mat = cp.asnumpy(result)
        self.assertTrue(np.allclose(mat, mat.T))
    
    def test_bounds(self):
        result = JaccardIndexSolutions(self.gpu_matrix)
        mat = cp.asnumpy(result)
        self.assertTrue(np.all((mat >= 0.0) & (mat <= 1.0)))
    
    def test_empty_matrix(self):
        with self.assertRaises(RuntimeError):
            JaccardIndexSolutions(cp.empty((0, 4)))

    def test_too_few_genes(self):
        with self.assertRaises(RuntimeError):
            JaccardIndexSolutions(cp.array([[1], [0]]))

if __name__ == "__main__":
    unittest.main()
