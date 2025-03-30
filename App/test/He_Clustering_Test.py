######### Libraries #########
import unittest                     # Test interface.
import sys                          # syscalls.
import os                           # OS calls.
import numpy as np                  # Numbers ADT managment.
import io                           # Input-Output.
from CoMOcG.He_Clustering import (
    He_clustering
)

class TestHeClustering(unittest.TestCase):

    def setUp(self):
        # Datos simulados para testeo
        self.distance_matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.5, 0.0]
        ])
        self.genes = ["GeneA", "GeneB", "GeneC"]
        self.output_dir = "test_dendrogram"
        self.output_file = "test_output.html"

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def tearDown(self):
        # Eliminar archivos de prueba si existen
        html_path = os.path.join(self.output_dir, self.output_file)
        if os.path.exists(html_path):
            os.remove(html_path)
        if os.path.exists(self.output_dir):
            os.rmdir(self.output_dir)

        # Activate prints.
        sys.stdout = self._original_stdout

    def test_he_clustering_basic(self):
        result = He_clustering(
            distance_matrix=self.distance_matrix,
            genes=self.genes,
            num_groups=2,
            save_path=self.output_dir,
            dendrogram_file=self.output_file,
            method="single"
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(self.genes))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, self.output_file)))

    def test_invalid_distance_matrix_shape(self):
        # Matriz no cuadrada
        bad_matrix = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]])
        result = He_clustering(bad_matrix, self.genes)
        self.assertIsNone(result)

    def test_genes_and_matrix_size_mismatch(self):
        bad_genes = ["GeneA", "GeneB"]
        result = He_clustering(self.distance_matrix, bad_genes)
        self.assertIsNone(result)

    def test_zero_clusters(self):
        result = He_clustering(self.distance_matrix, self.genes, num_groups=0)
        self.assertIsNone(result)
        

if __name__ == "__main__":
    unittest.main()
