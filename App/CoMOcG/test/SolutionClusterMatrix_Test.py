######### Libraries #########
import unittest
import sys
import os
import io
from contextlib import redirect_stdout

######### Module Path #########
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from SolutionClusterMatrix import (
    ProcessSolution_IDs,
    ProcessSolution_noIDs,
    SolutionClusterMatrix_GeneID,
    SolutionClusterMatrix_NoGeneID
)

######### Testing #########

class TestProcessSolutionsID(unittest.TestCase):   
    def test_happy_road(self):
        # Test data.
        list = [1, 2, 3, 2, 1, 3, 4, 3]
        genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8"]
        
        # Expected result.
        Compare_with = [['GENE1', 'GENE5'], ['GENE2', 'GENE4'], ['GENE3', 'GENE6', 'GENE8'], ['GENE7']]

        # Run.
        result = ProcessSolution_IDs(list, genes)

        # Check.
        self.assertEqual(Compare_with,result)

    def test_non_list(self):
        # Test data.
        list = None
        genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8"]

        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            result = ProcessSolution_IDs(list, genes)
        # Take message.
        mensaje_impreso = f.getvalue()

        # Check.
        self.assertEqual(mensaje_impreso, 
                         "Error: La entrada debe ser una lista de valores numéricos.\n")
        self.assertEqual(result,None)

    def test_empty_list(self):
        # Test data.
        list = []
        genes = []
        
        #RUN.
        result = ProcessSolution_IDs(list, genes)
        
        # Check.
        self.assertEqual(result,[])

class TestProcessSolutionsnoID(unittest.TestCase): 
    def test_happy_road(self):
        # Test data.
        list = [1, 2, 3, 2, 1, 3, 4, 3]

        # Expected result.
        Compare_with = [[0, 4], [1, 3], [2, 5, 7], [6]]

        # Run.
        result = ProcessSolution_noIDs(list)

        # Check.
        self.assertEqual(Compare_with,result)

    def test_non_list(self):
        # Test data.
        list = None

        # Not using the print message.
        f = io.StringIO()
        with redirect_stdout(f):
            #RUN.
            result = ProcessSolution_noIDs(list)
        # Take message.
        mensaje_impreso = f.getvalue()

        # Check.
        self.assertEqual(mensaje_impreso, 
                         "Error: La entrada debe ser una lista de valores numéricos.\n")
        self.assertEqual(result,None)

    def test_empty_list(self):
        # Test data.
        list = []
        
        #RUN.
        result = ProcessSolution_noIDs(list)
        
        # Check.
        self.assertEqual(result,[]) 

class TestSolutionClusterMatrix_GeneID(unittest.TestCase):   
    def test_happy_road(self):
        # Test data.
        Matrix = [
            [1, 2, 2, 1],
            [1, 1, 2, 2],
            [2, 3, 3, 2]
        ]
        genes = ["GENE1", "GENE2", "GENE3", "GENE4"]

        # Expected result.
        compare_with = [[['GENE1', 'GENE4'], ['GENE2', 'GENE3']], 
                        [['GENE1', 'GENE2'], ['GENE3', 'GENE4']], 
                        [['GENE1', 'GENE4'], ['GENE2', 'GENE3']]]
        
        # Run.
        result = SolutionClusterMatrix_GeneID(Matrix, genes, 4)

        # Check.
        self.assertEqual(result, compare_with) 

class TestSolutionClusterMatrix_NoGeneID(unittest.TestCase): 
    def test_happy_road(self):
        # Test data.
        Matrix = [
            [1, 2, 2, 1],
            [1, 1, 2, 2],
            [2, 3, 3, 2]
        ]

        # Expected result.
        compare_with = [
            [[0, 3], [1, 2]], 
            [[0, 1], [2, 3]], 
            [[0, 3], [1, 2]]
        ]

        # Run.
        result = SolutionClusterMatrix_NoGeneID(Matrix, 4)

        # Check.
        self.assertEqual(result, compare_with)

if __name__ == '__main__':
    unittest.main()