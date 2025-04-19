######### Libraries #########
import unittest
import os
import numpy as np
from tempfile import TemporaryDirectory

# Asumiendo que la función fue importada así:
from CoMOcG.Dual_Heatmap import plot_dual_heatmap_two_colors

class TestDualHeatmapPlot(unittest.TestCase):
    
    def setUp(self):
        # Crear dos matrices de prueba
        self.matrix_upper = np.array([
            [0.0, 0.3, 0.6],
            [np.nan, 0.0, 0.5],
            [np.nan, np.nan, 0.0]
        ])

        self.matrix_lower = np.array([
            [0.0, np.nan, np.nan],
            [0.3, 0.0, np.nan],
            [0.2, 0.4, 0.0]
        ])

        # Crear carpeta temporal para los archivos
        self.tempdir = TemporaryDirectory()
        self.output_file = os.path.join(self.tempdir.name, "test_dual_heatmap.html")
    
    def test_plot_dual_heatmap_creation(self):
        # Ejecutar la función
        plot_dual_heatmap_two_colors(
            matrix_upper=self.matrix_upper,
            matrix_lower=self.matrix_lower,
            save_filepath=self.output_file,
            title="Test Jaccard vs Wang"
        )
        
        # Verificar si el archivo fue creado
        self.assertTrue(os.path.exists(self.output_file), "El archivo HTML no fue creado correctamente.")
        
        # Verificar contenido mínimo
        with open(self.output_file, 'r') as f:
            content = f.read()
            self.assertIn("Plotly.newPlot", content)
            self.assertIn("Jaccard", content)
            self.assertIn("Wang", content)

    def test_plot_with_mismatched_shape_should_raise(self):
        with self.assertRaises(ValueError):
            plot_dual_heatmap_two_colors(
                matrix_upper=np.array([[1, 2], [3, 4]]),
                matrix_lower=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                save_filepath=self.output_file
            )

    def tearDown(self):
        # Eliminar carpeta temporal
        self.tempdir.cleanup()

# Ejecución directa
if __name__ == '__main__':
    unittest.main()