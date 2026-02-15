"""
Unit tests for He_Clustering module.

Purpose of this file:
- Validate correct hierarchical clustering behavior.
- Validate input validation logic.
- Ensure dendrogram export works without leaving residual files.
- Confirm optional return modes (fig/html).
"""

######### Libraries #########
import unittest
import tempfile
import os
import numpy as np
import logging

from ParetoInsight_CPU.He_Clustering import (
    he_clustering,
    compute_hierarchical_clustering,
    ClusteringOptions,
    DendrogramOptions,
    ExportOptions,
)


class TestHeClustering(unittest.TestCase):
    """Test suite for clustering logic, validation, and export behavior."""

    ########################## Test Initialization ##########################
    def setUp(self):
        """
        Build reusable small symmetric distance matrix.
        4 genes with two obvious clusters:
            Cluster A: 0-1
            Cluster B: 2-3
        """
        # Disable logging during tests
        logging.disable(logging.CRITICAL)

        self.genes = ["G1", "G2", "G3", "G4"]

        self.distance_matrix = np.array([
            [0.0, 0.1, 1.0, 1.0],
            [0.1, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.2],
            [1.0, 1.0, 0.2, 0.0],
        ])

    ########################## Core Clustering Tests ##########################

    def test_compute_hierarchical_clustering_basic(self):
        """Purpose: confirm clustering produces correct number of labels."""
        Z, labels = compute_hierarchical_clustering(
            self.distance_matrix,
            self.genes,
            ClusteringOptions(num_groups=2)
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_he_clustering_returns_labels_only(self):
        """Purpose: confirm default behavior returns only cluster labels."""
        labels = he_clustering(
            self.distance_matrix,
            self.genes,
            clustering=ClusteringOptions(num_groups=2)
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_num_groups_one(self):
        """Purpose: confirm num_groups=1 assigns all genes to same cluster."""
        labels = he_clustering(
            self.distance_matrix,
            self.genes,
            clustering=ClusteringOptions(num_groups=1)
        )

        self.assertEqual(len(np.unique(labels)), 1)

    ########################## Validation Tests ##########################

    def test_non_square_matrix(self):
        """Purpose: confirm non-square matrix raises RuntimeError."""
        bad_matrix = np.array([[0, 1, 2]])
        with self.assertRaises(RuntimeError):
            he_clustering(bad_matrix, self.genes)

    def test_non_symmetric_matrix(self):
        """Purpose: confirm non-symmetric matrix raises error."""
        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = 0.9  # break symmetry

        with self.assertRaises(RuntimeError):
            he_clustering(bad_matrix, self.genes)

    def test_negative_values(self):
        """Purpose: confirm negative distances are rejected."""
        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = -1.0
        bad_matrix[1, 0] = -1.0

        with self.assertRaises(RuntimeError):
            he_clustering(bad_matrix, self.genes)

    def test_nan_values(self):
        """Purpose: confirm NaN values are rejected."""
        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = np.nan
        bad_matrix[1, 0] = np.nan

        with self.assertRaises(RuntimeError):
            he_clustering(bad_matrix, self.genes)

    def test_gene_length_mismatch(self):
        """Purpose: confirm mismatch between genes and matrix size raises error."""
        with self.assertRaises(RuntimeError):
            he_clustering(self.distance_matrix, ["G1", "G2"])

    ########################## Dendrogram & Export Tests ##########################

    def test_return_figure(self):
        """Purpose: confirm return_fig=True returns a plotly Figure."""
        labels, fig = he_clustering(
            self.distance_matrix,
            self.genes,
            clustering=ClusteringOptions(num_groups=2),
            return_fig=True
        )

        self.assertEqual(len(labels), 4)
        self.assertTrue(hasattr(fig, "to_html"))

    def test_return_html(self):
        """Purpose: confirm return_html=True returns HTML string."""
        labels, html = he_clustering(
            self.distance_matrix,
            self.genes,
            clustering=ClusteringOptions(num_groups=2),
            return_html=True
        )

        self.assertIn("<div", html)
        self.assertEqual(len(labels), 4)

    def test_export_html_temp_directory(self):
        """
        Purpose:
        - Confirm HTML file is created.
        - Ensure no residual files remain outside TemporaryDirectory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "dendrogram.html")

            labels = he_clustering(
                self.distance_matrix,
                self.genes,
                clustering=ClusteringOptions(num_groups=2),
                save_html_to=filepath,
                export=ExportOptions(verbose=False)
            )

            self.assertTrue(os.path.exists(filepath))
            self.assertEqual(len(labels), 4)

        # After exiting block, directory is automatically removed.
        self.assertFalse(os.path.exists(tmpdir))

    ########################## Stability Tests ##########################

    def test_different_linkage_methods(self):
        """Purpose: confirm different linkage methods do not crash."""
        for method in ["single", "complete", "average"]:
            labels = he_clustering(
                self.distance_matrix,
                self.genes,
                clustering=ClusteringOptions(num_groups=2, method=method)
            )
            self.assertEqual(len(labels), 4)


if __name__ == "__main__":
    unittest.main()
