######### Libraries #########
import unittest
import sys
import os
import io
from contextlib import redirect_stdout
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix

######### Module Path #########
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ProportionMatrix import (
    compute_connectivity,
    connectivityMatrix,
    sum_connectivity_matrices,
    ProportionMatrix_Similarity,
    ProportionMatrix_Disimilarity
)

class TestComputeConnectivity(unittest.TestCase):

    def test_simple_case(self):
        # Test case.
        solution = [1, 2, 2, 1]
        expected_matrix = np.array([[0, 0, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, 0]])

        # Convert expected matrix into CSR.
        expected_csr = csr_matrix(expected_matrix)

        # RUN.
        result_csr = compute_connectivity(solution)

        # Compare solutions.
        self.assertTrue((result_csr != expected_csr).nnz == 0, 
                        "La matriz CSR generada no es igual a la esperada.")

    def test_not_a_list(self):
        # Test case with a string.
        solution = "fake_list"
        
        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            result_csr = compute_connectivity(solution)
        # Take message.
        mensaje_impreso = f.getvalue()
        
        # Check message and output.
        self.assertEqual(mensaje_impreso, 
                         "Error de tipo: La solución debe ser una lista o una secuencia.\n")
        self.assertEqual(result_csr,None)

    def test_empty_list(self):
        # Test case with empty list.
        solution = []
        
        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            result_csr = compute_connectivity(solution)
        # Take message.
        mensaje_impreso = f.getvalue()
        
        # Check message and output.
        self.assertEqual(mensaje_impreso, 
                         "Error de valor: La solución no puede estar vacía.\n")
        self.assertEqual(result_csr,None)

class TestConnectivityMatrix(unittest.TestCase):
    def test_connectivityMatrix(self):
        # Test case with multiple solutions.
        SolutionsMatrix = [
            [1, 1, 2, 2, 5, 5],
            [3, 3, 3, 4, 6, 6],
            [7, 7, 7, 7, 7, 7],
            [10, 10, 11, 12, 12, 12],
        ]

        expected_matrices = [
            csr_matrix(np.array([[0, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 1, 0]])),
    
            csr_matrix(np.array([[0, 1, 1, 0, 0, 0],
                                 [1, 0, 1, 0, 0, 0],
                                 [1, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 1, 0]])),
    
            csr_matrix(np.array([[0, 1, 1, 1, 1, 1],
                                 [1, 0, 1, 1, 1, 1],
                                 [1, 1, 0, 1, 1, 1],
                                 [1, 1, 1, 0, 1, 1],
                                 [1, 1, 1, 1, 0, 1],
                                 [1, 1, 1, 1, 1, 0]])),
    
            csr_matrix(np.array([[0, 1, 0, 0, 0, 0],
                                 [1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 1],
                                 [0, 0, 0, 1, 0, 1],
                                 [0, 0, 0, 1, 1, 0]]))
        ]
    
        # RUN.
        result_matrices = connectivityMatrix(SolutionsMatrix)

        # Compare all solutions.
        for result_csr, expected_csr in zip(result_matrices, expected_matrices):
            self.assertTrue((result_csr != expected_csr).nnz == 0, "La matriz CSR generada no es igual a la esperada.")

class TestSumConnectivityMatrix(unittest.TestCase):
    def test_sum_connectivity_matrices(self):
        # Test case with multiple solutions.
        SolutionsMatrix = [
            [1, 1, 2, 2, 5, 5],
            [3, 3, 3, 4, 6, 6],
            [7, 7, 7, 7, 7, 7],
            [10, 10, 11, 12, 12, 12],
        ]

        expected_matrix = csr_matrix(np.array([[0, 4, 2, 1, 1, 1],
                                               [4, 0, 2, 1, 1, 1],
                                               [2, 2, 0, 2, 1, 1],
                                               [1, 1, 2, 0, 2, 2],
                                               [1, 1, 1, 2, 0, 4],
                                               [1, 1, 1, 2, 4, 0]], dtype=float))

        # RUN.
        connectivity_matrices = connectivityMatrix(SolutionsMatrix)
        summed_matrix = sum_connectivity_matrices(connectivity_matrices)

        # Check.
        self.assertTrue(np.array_equal(summed_matrix.toarray(), expected_matrix.toarray()), 
                        "La matriz sumada no es igual a la esperada.")

    def test_not_a_list(self):
        # Test Case with empty list.
        connectivity_matrices = "UwU"

        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            summed_matrix = sum_connectivity_matrices(connectivity_matrices)
        # Take message.
        mensaje_impreso = f.getvalue()

        # Debería retornar None para entradas vacías
        self.assertEqual(mensaje_impreso, 
                         "Error de tipo: Todos los elementos deben ser matrices dispersas CSR.\n")
        self.assertEqual(summed_matrix,None)

    def test_empty_input(self):
        # Test Case with empty list.
        connectivity_matrices = []

        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            summed_matrix = sum_connectivity_matrices(connectivity_matrices)
        # Take message.
        mensaje_impreso = f.getvalue()

        # Debería retornar None para entradas vacías
        self.assertEqual(mensaje_impreso, 
                         "Error de valor: La lista de matrices de conectividad está vacía.\n")
        self.assertEqual(summed_matrix,None)

class TestProportionMatrix_Similarity(unittest.TestCase):
    def test_divide_by_total_solutions(self):
        # Example Matrix.
        summed_matrix = csr_matrix(np.array([[2, 4], [4, 6]]))
        total_solutions = 2

        # Expected result.
        expected_matrix = np.array([[1.0, 2.0], [2.0, 3.0]])

        # RUN.
        divided_matrix = ProportionMatrix_Similarity(summed_matrix, total_solutions)

        # Check.
        self.assertTrue(np.array_equal(divided_matrix.toarray(), expected_matrix), 
                        "La matriz dividida no es igual a la esperada.")
        
    def test_not_a_list(self):
        # Test case with a string.
        solution = "fake_list"
        
        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            result_csr = ProportionMatrix_Similarity(solution,2)
        # Take message.
        mensaje_impreso = f.getvalue()
        
        # Check message and output.
        self.assertEqual(mensaje_impreso, 
                         "Error de tipo: La matriz sumada debe ser del tipo csr_matrix.\n")
        self.assertEqual(result_csr,None)

    def test_empty_list(self):
        # Test case with empty list.
        solution = csr_matrix((0,0))
        
        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            result_csr = ProportionMatrix_Similarity(solution,0)
        # Take message.
        mensaje_impreso = f.getvalue()
        
        # Check message and output.
        self.assertEqual(mensaje_impreso, 
                         "Error de valor: El número total de soluciones no puede ser cero.\n")
        self.assertEqual(result_csr,None)

    def test_divide_by_zero(self):
        # Matriz de ejemplo
        summed_matrix = csr_matrix(np.array([[2, 4], [4, 6]]))

        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            result_csr = ProportionMatrix_Similarity(summed_matrix,0)
        # Take message.
        mensaje_impreso = f.getvalue()
        
        # Check message and output.
        self.assertEqual(mensaje_impreso, 
                         "Error de valor: El número total de soluciones no puede ser cero.\n")
        self.assertEqual(result_csr,None)

class TestProportionMatrix_Disimilarity(unittest.TestCase):
    def test_subtract_from_one(self):
        # Matriz de ejemplo después de la división
        divided_matrix = csr_matrix(np.array([[0.5, 0.25], [0.25, 0.75]]))

        # Matriz esperada después de restar cada valor a 1
        expected_matrix = np.array([[0.5, 0.75], [0.75, 0.25]])

        # Llamar a la función
        subtracted_matrix = ProportionMatrix_Disimilarity(divided_matrix)

        # Comparar la matriz resultante con la esperada
        self.assertTrue(np.array_equal(subtracted_matrix.toarray(), expected_matrix), 
                        "La matriz restada no es igual a la esperada.")

if __name__ == '__main__':
    unittest.main()