######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import sys                          # syscalls.
import io                           # Input-Output.
from CoMOcG.SolutionComposition import AmountGenes_Equals

class TestSolutionCompositions(unittest.TestCase):

    def setUp(self):
        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def tearDown(self):
        # Activate prints.
        sys.stdout = self._original_stdout
    
    def test_basic_overlap(self):
        S1 = [{'A', 'B'}, {'C'}]
        S2 = [{'B'}, {'C', 'D'}]
        result = AmountGenes_Equals(S1, S2)

        expected = np.array([
            [1, 0],  # {'A', 'B'} ∩ {'B'} = 1, ∩ {'C', 'D'} = 0
            [0, 1]   # {'C'} ∩ {'B'} = 0, ∩ {'C', 'D'} = 1
        ])

        np.testing.assert_array_equal(result, expected)

    def test_empty_clusters(self):
        S1 = [{'A', 'B'}, set()]
        S2 = [{'B', 'C'}, {'D'}]
        result = AmountGenes_Equals(S1, S2)

        expected = np.array([
            [1, 0],
            [0, 0]
        ])

        np.testing.assert_array_equal(result, expected)

    def test_empty_inputs(self):
        result = AmountGenes_Equals([], [{'A'}])
        self.assertTrue(result.size == 0)  # en vez de assertRaises

    def test_invalid_inputs_type(self):
        result = AmountGenes_Equals("not_a_list", [{'A'}])
        self.assertTrue(result.size == 0)

    def test_shape_of_output(self):
        S1 = [{'G1'}, {'G2'}]
        S2 = [{'G1'}, {'G2'}, {'G3'}]
        result = AmountGenes_Equals(S1, S2)
        self.assertEqual(result.shape, (2, 3))

if __name__ == '__main__':
    unittest.main()