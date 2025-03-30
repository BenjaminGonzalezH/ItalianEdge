######### Libraries #########
import unittest
import sys
import os
import csv

######### Module Path #########
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from CoMOcG.ReadSolution import (
    read_csv_part,
    ReadInputCSV_threads
)

class TestReadInputCSV(unittest.TestCase):

    def setUp(self):
        # Crea archivo CSV temporal con IDs
        self.file_with_ids = "test_with_ids.csv"
        with open(self.file_with_ids, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["SolID", "Gene1", "Gene2"])
            writer.writerow(["Sol1", 2, 3])
            writer.writerow(["Sol2", 2, 3])
            writer.writerow(["Sol3", 3, 3])

        # Crea archivo CSV temporal sin IDs
        self.file_without_ids = "test_without_ids.csv"
        with open(self.file_without_ids, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Gene1", "Gene2", "Gene3"])
            writer.writerow([1, 2, 3])
            writer.writerow([2, 2, 3])
            writer.writerow([3, 3, 3])

        # Crea archivo vacío
        self.empty_file = "empty.csv"
        with open(self.empty_file, "w") as f:
            pass

    def tearDown(self):
        for filename in [self.file_with_ids, self.file_without_ids, self.empty_file]:
            if os.path.exists(filename):
                os.remove(filename)

    def test_read_csv_part_with_id(self):
        rows = read_csv_part(self.file_with_ids, start_row=1, chunk_size=2, flag_solutions_id=0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ['2', '3'])

    def test_read_csv_part_without_id(self):
        rows = read_csv_part(self.file_without_ids, start_row=1, chunk_size=2, flag_solutions_id=1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1], ['2', '2', '3'])

    def test_read_input_csv_threads_with_id(self):
        genes, n, matrix = ReadInputCSV_threads(self.file_with_ids, n_workers=2, solutions_id_colum=0)
        self.assertEqual(genes, ['Gene1', 'Gene2'])
        self.assertEqual(n, 2)
        self.assertEqual(matrix.shape, (3, 2))

    def test_read_input_csv_threads_without_id(self):
        genes, n, matrix = ReadInputCSV_threads(self.file_without_ids, n_workers=2, solutions_id_colum=1)
        self.assertEqual(genes, ['Gene1', 'Gene2', 'Gene3'])
        self.assertEqual(n, 3)
        self.assertEqual(matrix.shape, (3, 3))

    def test_empty_file(self):
        with self.assertRaises(ValueError):
            ReadInputCSV_threads(self.empty_file, n_workers=2)

    def test_invalid_worker_number(self):
        with self.assertRaises(ValueError):
            ReadInputCSV_threads(self.file_with_ids, n_workers=0)

    def test_file_not_found(self):
        with self.assertRaises(ValueError):
            ReadInputCSV_threads("non_existent.csv", n_workers=2)

if __name__ == '__main__':
    unittest.main()