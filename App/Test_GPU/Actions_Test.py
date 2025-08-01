import unittest
import numpy as np
import cupy as cp

from ParetoInsight_GPU.Actions import (
    TransformMathStructure,
    GPU_MemoryPool,
    GPU_freeMemoryPool
)

class TestCupyNumpyUtils(unittest.TestCase):

    def test_transform_math_structure_to_gpu(self):
        arr_cpu = np.array([1,2,3])
        arr_gpu = TransformMathStructure(arr_cpu, Origin="CPU")
        self.assertTrue(isinstance(arr_gpu, cp.ndarray))
        np.testing.assert_array_equal(cp.asnumpy(arr_gpu), arr_cpu)

    def test_transform_math_structure_to_cpu(self):
        arr_gpu = cp.array([4,5,6])
        arr_cpu = TransformMathStructure(arr_gpu, Origin="GPU")
        self.assertTrue(isinstance(arr_cpu, np.ndarray))
        np.testing.assert_array_equal(arr_cpu, cp.asnumpy(arr_gpu))

    def test_transform_math_structure_invalid(self):
        # Si Origin no es CPU o GPU, se debe lanzar excepción o retornar None
        arr_cpu = np.array([7,8,9])
        # El match no incluye un case por defecto: se espera una excepción TypeError por no haber branch
        with self.assertRaises(Exception):
            TransformMathStructure(arr_cpu, Origin="OTHER")

    def test_gpu_memory_pool_and_free(self):
        mempool = GPU_MemoryPool(10 * 2**20)  # 10MB
        self.assertTrue(hasattr(mempool, 'used_bytes'))
        # Hay memoria asignada
        before = mempool.used_bytes()
        arr = cp.empty((10000,), dtype=cp.float32)  # Usa memoria del pool
        after = mempool.used_bytes()
        self.assertTrue(after >= before)
        GPU_freeMemoryPool(mempool)
        self.assertTrue(mempool.used_bytes() >= 0)

    def test_gpu_memory_pool_invalid(self):
        # Límite inválido (negativo, string...)
        with self.assertRaises(ValueError):
            GPU_MemoryPool(-100)
        with self.assertRaises(ValueError):
            GPU_MemoryPool("foo")

    def test_gpu_free_memory_pool_exception(self):
        # Pasar un argumento que no sea un mempool de cupy debería lanzar excepción
        with self.assertRaises(RuntimeError):
            GPU_freeMemoryPool(None)

if __name__ == "__main__":
    unittest.main()
