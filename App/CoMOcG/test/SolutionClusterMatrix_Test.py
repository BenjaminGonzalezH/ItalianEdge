######### Libraries #########
import unittest
import sys
import os
import csv
import tempfile
import random
import string
import numpy as np
import numpy.testing as npt

######### Module Path #########
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from SolutionClusterMatrix import (
    ProcessSolution_IDs,
    ProcessSolution_noIDs,
    SolutionClusterMatrix_GeneID,
    SolutionClusterMatrix_NoGeneID
)

######### Aux functions #########

######### Testing #########

class TestProcessSolutionsID(unittest.TestCase):
    
    def test_happy_road(self):
        list = [1, 2, 3, 2, 1, 3, 4, 3]
        genes = ["GENE1", "GENE2", "GENE3", "GENE4", "GENE5", "GENE6", "GENE7", "GENE8"]
        result = ProcessSolution_IDs(list, genes)
        print(result)

class TestProcessSolutionsID(unittest.TestCase):
    
    def test_happy_road(self):
        list = [1, 2, 3, 2, 1, 3, 4, 3]
        result = ProcessSolution_noIDs(list)
        print(result)


class TestSolutionClusterMatrix_GeneID(unittest.TestCase):
    
    def test_happy_road(self):
        # Entrada básica de prueba
        Matrix = [
            [1, 2, 2, 1],
            [1, 1, 2, 2],
            [2, 3, 3, 2]
        ]
        genes = ["GENE1", "GENE2", "GENE3", "GENE4"]
        result = SolutionClusterMatrix_GeneID(Matrix, genes, 4)
        print(result)

class TestSolutionClusterMatrix_NoGeneID(unittest.TestCase):
    
    def test_happy_road(self):
        # Entrada básica de prueba
        Matrix = [
            [1, 2, 2, 1],
            [1, 1, 2, 2],
            [2, 3, 3, 2]
        ]
        result = SolutionClusterMatrix_NoGeneID(Matrix, 4)
        print(result)



if __name__ == '__main__':
    unittest.main()