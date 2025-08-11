######### Libraries #########
import unittest                     # Test interface.
import sys                          # syscalls.
import io                           # Input-Output 
import os                           # OS calls.
import csv                          # CSV files.
from ParetoInsight_CPU.ReadSolution import ReadSolutionsFile

class TestReadSolutionsFile(unittest.TestCase):

    ########################## Test's Initialization ##########################
    def setUp(self):
        self.file_with_ids = "test_with_ids.csv"
        with open(self.file_with_ids, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["SolID", "Gene1", "Gene2"])
            writer.writerow(["Sol1", 2, 3])
            writer.writerow(["Sol2", 2, 3])
            writer.writerow(["Sol3", 3, 3])

        self.file_without_ids = "test_without_ids.csv"
        with open(self.file_without_ids, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Gene1", "Gene2", "Gene3"])
            writer.writerow([1, 2, 3])
            writer.writerow([2, 2, 3])
            writer.writerow([3, 3, 3])

        # Archivo vacío
        self.empty_file = "empty.csv"
        with open(self.empty_file, "w") as f:
            pass

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()        

    ########################## Test's Deletion ##########################
    def tearDown(self):
        for filename in [self.file_with_ids, self.file_without_ids, self.empty_file]:
            if os.path.exists(filename):
                os.remove(filename)

    ########################## Tests ##########################
    def test_read_with_ids(self):
        matrix, genes = ReadSolutionsFile(self.file_with_ids, format="csv")
        self.assertEqual(genes, ["SolID", "Gene1", "Gene2"])
        self.assertEqual(matrix.shape, (3, 3))

    def test_read_without_ids(self):
        matrix, genes = ReadSolutionsFile(self.file_without_ids, format="csv")
        self.assertEqual(genes, ["Gene1", "Gene2", "Gene3"])
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(matrix[0, 0], 1)

    def test_empty_file(self):
        with self.assertRaises(Exception):
            ReadSolutionsFile(self.empty_file, format="csv")

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            ReadSolutionsFile("non_existent.csv", format="csv")

    def test_unsupported_format(self):
        with self.assertRaises(Exception):
            ReadSolutionsFile(self.file_with_ids, format="json")

if __name__ == '__main__':
    unittest.main()