######### Libraries #########
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from ReadSolution import (
    ReadDataFrame
)

######### Testing #########
class TestReadSolution(unittest.TestCase):
    def test_read_valid_file(self):
        pass

if __name__ == '__main__':
    unittest.main()