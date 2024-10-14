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
    ReadInputcsv,
    ReadInputcsv_noID
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

def generate_solution_noID(numero, tam):
    solucion = [random.randint(1, 4) for _ in range(tam)]
    return solucion

######### Testing #########
class TestReadSolution(unittest.TestCase):
    def test_read_valid_file(self):
        # Create a csv tempfile for this test with identificator temp_csv.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            compareMatrix = []
            # Write file.
            dumb_genes = [dumb_alphaNumeric() for _ in range(20)]
            writer.writerow(dumb_genes)
            for i in range(0,101):
                row = generate_solution(i,20)
                writer.writerow(row)
                compareMatrix.append(row[1:len(row)+1])
            compareMatrix = np.array(compareMatrix, dtype=int)

            # close tempfile to re-open it later.
            temp_csv.close()
            
            # function.
            Matrix, genes, num_genes = ReadInputcsv(temp_csv.name)

            # Delete the file. Also, NamedTemporaryFile do it, but
            # this happend when you close the file.
            os.remove(temp_csv.name)

            # AssertEquals
            self.assertEqual(genes, dumb_genes)
            self.assertEqual(num_genes, 20)
            npt.assert_array_equal(Matrix, compareMatrix)

    def test_read_filenotfound(self):       
        # Obtain ValueError from function.
        with self.assertRaises(ValueError) as context:
            ReadInputcsv("NoFile")
        self.assertEqual(str(context.exception), "File in NoFile does not exists.")

    def test_read_Emptyfile(self):
        # Create a csv tempfile for this test with identificator temp_csv.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            csv.writer(temp_csv)
            # close tempfile to re-open it later.
            temp_csv.close()

            # Obtain ValueError from function.
            with self.assertRaises(ValueError) as context:
                ReadInputcsv(temp_csv.name)
            self.assertEqual(str(context.exception), "Empty file.")

            os.remove(temp_csv.name)

    def test_read_Emptyrow(self):
        # Create a csv tempfile for this test with identificator temp_csv.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            
            # Write file.
            for _ in range(10):
                writer.writerow([])
            empty_matrix = np.empty((9, 0), dtype=np.int64)
            
            # close tempfile to re-open it later.
            temp_csv.close()

            # function.
            Matrix, genes, num_genes = ReadInputcsv(temp_csv.name)

            # Delete the file. Also, NamedTemporaryFile do it, but
            # this happend when you close the file.
            os.remove(temp_csv.name)

            # AssertEquals
            self.assertEqual(genes, [])
            self.assertEqual(num_genes, 0)
            self.assertTrue(np.array_equal(Matrix, empty_matrix))

class TestReadSolutionNoID(unittest.TestCase):
    def test_read_valid_file(self):
        # Create a csv tempfile for this test with identificator temp_csv.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            compareMatrix = []
            # Write file.
            dumb_genes = [dumb_alphaNumeric() for _ in range(20)]
            writer.writerow(dumb_genes)
            for i in range(0,101):
                row = generate_solution_noID(i,20)
                writer.writerow(row)
                compareMatrix.append(row[0:len(row)+1])
            compareMatrix = np.array(compareMatrix, dtype=int)

            # close tempfile to re-open it later.
            temp_csv.close()
            
            # function.
            Matrix, genes, num_genes = ReadInputcsv_noID(temp_csv.name)

            # Delete the file. Also, NamedTemporaryFile do it, but
            # this happend when you close the file.
            os.remove(temp_csv.name)

            # AssertEquals
            self.assertEqual(genes, dumb_genes)
            self.assertEqual(num_genes, 20)
            npt.assert_array_equal(Matrix, compareMatrix)

    def test_read_filenotfound(self):       
        # Obtain ValueError from function.
        with self.assertRaises(ValueError) as context:
            ReadInputcsv_noID("NoFile")
        self.assertEqual(str(context.exception), "File in NoFile does not exists.")

    def test_read_Emptyfile(self):
        # Create a csv tempfile for this test with identificator temp_csv.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            csv.writer(temp_csv)
            # close tempfile to re-open it later.
            temp_csv.close()

            # Obtain ValueError from function.
            with self.assertRaises(ValueError) as context:
                ReadInputcsv_noID(temp_csv.name)
            self.assertEqual(str(context.exception), "Empty file.")

            os.remove(temp_csv.name)

    def test_read_Emptyrow(self):
        # Create a csv tempfile for this test with identificator temp_csv.
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, newline='') as temp_csv:
            writer = csv.writer(temp_csv)
            
            # Write file.
            for _ in range(10):
                writer.writerow([])
            empty_matrix = np.empty((9, 0), dtype=np.int64)
            
            # close tempfile to re-open it later.
            temp_csv.close()

            # function.
            Matrix, genes, num_genes = ReadInputcsv_noID(temp_csv.name)

            # Delete the file. Also, NamedTemporaryFile do it, but
            # this happend when you close the file.
            os.remove(temp_csv.name)

            # AssertEquals
            self.assertEqual(genes, [])
            self.assertEqual(num_genes, 0)
            self.assertTrue(np.array_equal(Matrix, empty_matrix))


if __name__ == '__main__':
    unittest.main()