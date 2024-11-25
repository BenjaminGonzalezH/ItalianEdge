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
from ReadSolution import (
    ReadInputCSV_processes,
    ReadInputCSV_threads
)

######### Aux functions #########
def dumb_alphaNumeric():
    letras = ''.join(random.choices(string.ascii_uppercase, k=6)) 
    numeros = ''.join(random.choices(string.digits, k=4))
    return letras + numeros

def generate_solution(numero, tam):
    solucion = [f"Solution {numero}"]
    solucion.extend(random.randint(1, 4) for _ in range(tam))
    return solucion

def generate_solution_noID(tam):
    solucion = [random.randint(1, 4) for _ in range(tam)]
    return solucion

######### Testing #########

class TestReadSolution_Processes(unittest.TestCase):

    def test_read_valid_file(self):

        # Create temporal CSV.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            
            ####################### Create Dummy Variables.
            writer = csv.writer(temp_csv)
            compareMatrix = []

            # Write dumb genes identificators.
            dumb_genes = [dumb_alphaNumeric() for _ in range(1000)]
            writer.writerow(dumb_genes)

            # Write solutions.
            for i in range(500):
                row = generate_solution(i, 1000)
                writer.writerow(row)
                compareMatrix.append(row[1:])
            compareMatrix = np.array(compareMatrix, dtype=int)

            # Close file.
            temp_csv.close()
            ###############################################

            # RUN
            genes, num_genes, Matrix = ReadInputCSV_processes(temp_csv.name, n_jobs=5)
            os.remove(temp_csv.name)

            # Checks.
            self.assertEqual(genes, dumb_genes)
            self.assertEqual(num_genes, 1000)
            npt.assert_array_equal(Matrix, compareMatrix)

    def test_empty_file(self):
        # Create temporal CSV.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_csv:
            temp_csv.close()
            try:
                with self.assertRaises(ValueError) as context:
                    ReadInputCSV_processes(temp_csv.name, n_jobs=2)
                self.assertEqual(str(context.exception), "Input error: Empty file.")
            finally:
                os.remove(temp_csv.name)

    def test_no_solution_id(self):
         # Create temporal CSV without solutions id's.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            compareMatrix = []
            dumb_genes = [dumb_alphaNumeric() for _ in range(1000)]
            writer.writerow(dumb_genes)
            for _ in range(500):
                row = generate_solution_noID(1000)
                writer.writerow(row)
                compareMatrix.append(row)
            compareMatrix = np.array(compareMatrix, dtype=int)

            temp_csv.close()

            # RUN.
            genes, num_genes, Matrix = ReadInputCSV_processes(temp_csv.name, n_jobs=3, solutions_id_colum=1)
            os.remove(temp_csv.name)

            # Checks
            self.assertEqual(genes, dumb_genes)
            self.assertEqual(num_genes, 1000)
            npt.assert_array_equal(Matrix, compareMatrix)

    def test_invalid_n_jobs(self):
        # Create a valid file.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            writer.writerow([dumb_alphaNumeric() for _ in range(10)])
            for _ in range(5):
                writer.writerow(generate_solution_noID(10))
            temp_csv.close()

            try:
                with self.assertRaises(ValueError) as context:
                    ReadInputCSV_processes(temp_csv.name, n_jobs=-2)
                self.assertIn("n_jobs must be a positive integer or -1.", str(context.exception))
            finally:
                os.remove(temp_csv.name)

class TestReadSolution_Threads(unittest.TestCase):

    def test_read_valid_file(self):

        # Create temporal CSV.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            
            ####################### Create Dummy Variables.
            writer = csv.writer(temp_csv)
            compareMatrix = []

            # Write dumb genes identificators.
            dumb_genes = [dumb_alphaNumeric() for _ in range(1000)]
            writer.writerow(dumb_genes)

            # Write solutions.
            for i in range(500):
                row = generate_solution(i, 1000)
                writer.writerow(row)
                compareMatrix.append(row[1:])
            compareMatrix = np.array(compareMatrix, dtype=int)

            # Close file.
            temp_csv.close()
            ###############################################

            # RUN
            genes, num_genes, Matrix = ReadInputCSV_threads(temp_csv.name, n_workers=5)
            os.remove(temp_csv.name)

            # Checks.
            self.assertEqual(genes, dumb_genes)
            self.assertEqual(num_genes, 1000)
            npt.assert_array_equal(Matrix, compareMatrix)

    def test_empty_file(self):
        # Create temporal CSV.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_csv:
            temp_csv.close()
            try:
                with self.assertRaises(ValueError) as context:
                    ReadInputCSV_threads(temp_csv.name, n_workers=2)
                self.assertEqual(str(context.exception), "Input error: Empty file.")
            finally:
                os.remove(temp_csv.name)

    def test_no_solution_id(self):
         # Create temporal CSV without solutions id's.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            compareMatrix = []
            dumb_genes = [dumb_alphaNumeric() for _ in range(1000)]
            writer.writerow(dumb_genes)
            for _ in range(500):
                row = generate_solution_noID(1000)
                writer.writerow(row)
                compareMatrix.append(row)
            compareMatrix = np.array(compareMatrix, dtype=int)

            temp_csv.close()

            # RUN.
            genes, num_genes, Matrix = ReadInputCSV_threads(temp_csv.name, n_workers=3, solutions_id_colum=1)
            os.remove(temp_csv.name)

            # Checks
            self.assertEqual(genes, dumb_genes)
            self.assertEqual(num_genes, 1000)
            npt.assert_array_equal(Matrix, compareMatrix)

    def test_invalid_n_jobs(self):
        # Create a valid file.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            writer.writerow([dumb_alphaNumeric() for _ in range(10)])
            for _ in range(5):
                writer.writerow(generate_solution_noID(10))
            temp_csv.close()

            try:
                with self.assertRaises(ValueError) as context:
                    ReadInputCSV_threads(temp_csv.name, n_workers=-2)
                self.assertIn("n_jobs must be a positive integer or -1.", str(context.exception))
            finally:
                os.remove(temp_csv.name)

if __name__ == '__main__':
    unittest.main()