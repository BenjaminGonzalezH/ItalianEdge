######### Libraries #########
import unittest                     # Test interface.
import numpy as np                  # Numbers ADT managment.
import pandas as pd                 # Dataframe managment.
import sys                          # syscalls.
import io                           # Input-Output 
from ParetoInsight_CPU.WangIndex import (
    SimilarityIndexMatrix,
    Solution_Wang_index_similarity_Python,
    AnnotationFromEntrezIDs
)

class TestWangIndex(unittest.TestCase):

    def setUp(self):
        # Human genes.
        self.genes = ["TP53", "BRCA1", "EGFR", "MYC"]

        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def test_similarity_index_matrix(self):
        """Prueba que SimilarityIndexMatrix devuelva una matriz cuadrada simétrica."""
        matrix = SimilarityIndexMatrix(
            genes=self.genes,
            gaf_name="goa_human",
            ontology="BP",
            measure="wang",
            groupwise="bma",
            download_gaf=True,
            transform=False,
            load_go_terms=True
        )
        # Validaciones
        self.assertIsInstance(matrix, np.ndarray)
        self.assertEqual(matrix.shape, (len(self.genes), len(self.genes)))
        self.assertTrue(np.allclose(matrix, matrix.T))   # simétrica
        self.assertTrue(np.allclose(np.diag(matrix), 1)) # diagonal con 1

    def test_solution_wang_index_similarity(self):
        """Prueba de matriz de similitud entre soluciones usando datos ficticios."""
        ids = self.genes
        similarity_matrix = np.identity(len(ids))  # matriz identidad
        df = pd.DataFrame({
            "Solution 1": [0, 0],
            "Solution 2": [1, 1],
            "Cluster 1": [0, 1],
            "Cluster 2": [1, 0],
            "Jaccard Similarity": [0.5, 0.8]
        })
        groups_structure = [
            [set(["TP53", "BRCA1"])],
            [set(["EGFR", "MYC"])]
        ]

        result = Solution_Wang_index_similarity_Python(ids, similarity_matrix, df, groups_structure, num_threads=1)

        # Validaciones
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.allclose(np.diag(result), 1)) # diagonal con 1

    def test_annotation_from_entrez(self):
        """Prueba que se devuelvan anotaciones válidas desde gprofiler."""
        entrez_ids = ["7157", "672"]  # TP53 y BRCA1
        annotations = AnnotationFromEntrezIDs(entrez_ids, Ontology=["GO:BP"], organism="hsapiens")

        # Validaciones
        self.assertIsInstance(annotations, dict)
        self.assertTrue(all(isinstance(k, str) for k in annotations.keys()))
        self.assertTrue(all(isinstance(v, list) for v in annotations.values()))

if __name__ == "__main__":
    unittest.main()
