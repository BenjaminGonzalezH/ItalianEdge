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
    ReadInputcsv
)

######### Aux functions #########
def dumb_alphaNumeric():
    letras = ''.join(random.choices(string.ascii_uppercase, k=6))  # 6 letras aleatorias
    numeros = ''.join(random.choices(string.digits, k=4))  # 4 dígitos aleatorios
    return letras + numeros

def generate_solution(numero, tam):
    solucion = [f"Solution {numero}"]
    solucion.extend(random.randint(1, 4) for _ in range(tam))  # 3 números aleatorios entre 1 y 10
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
                compareMatrix.append([row[1:len(row)+1]])
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
        

if __name__ == '__main__':
    unittest.main()