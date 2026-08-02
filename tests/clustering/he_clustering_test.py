"""
Unit tests for He_Clustering module.

Purpose of this file:
- Validate hierarchical clustering behavior.
- Validate input validation logic.
- Confirm dendrogram generation and export.
- Validate optional return modes.
"""

######### Libraries #########
import unittest
import tempfile
import os
import numpy as np
import logging

from biocluster.clustering.he_clustering import (
    he_clustering,
    compute_hierarchical_clustering,
    ClusteringOptions,
    DendrogramOptions,
    ExportOptions,
)


class TestHeClustering(unittest.TestCase):
    """Test suite validating clustering logic and export behavior."""

    ########################## Test Initialization ##########################

    def setUp(self):
        """
        Create reusable symmetric distance matrix.

        Structure:
        Cluster A -> genes 0 and 1
        Cluster B -> genes 2 and 3
        """

        # Disable logs and prints produced by verbose options
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
        """
        Confirm clustering produces valid outputs.

        Checks:
        - linkage matrix shape
        - label count
        - cophenetic coefficient type
        """

        Z, labels, cophenetic_corr = compute_hierarchical_clustering(
            self.distance_matrix,
            self.genes,
            ClusteringOptions(num_groups=2)
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertTrue(isinstance(cophenetic_corr, float))
        self.assertTrue(Z.shape[1] == 4)

    def test_he_clustering_returns_labels_only(self):
        """
        Confirm default behavior returns cluster labels.
        """

        labels = he_clustering(
            self.distance_matrix,
            self.genes,
            clustering=ClusteringOptions(num_groups=2)
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_num_groups_one(self):
        """
        Confirm num_groups=1 places all genes in one cluster.
        """

        labels = he_clustering(
            self.distance_matrix,
            self.genes,
            clustering=ClusteringOptions(num_groups=1)
        )

        self.assertEqual(len(np.unique(labels)), 1)

    ########################## Validation Tests ##########################

    def test_non_square_matrix(self):
        """Confirm non-square matrix triggers validation error."""

        bad_matrix = np.array([[0, 1, 2]])

        with self.assertRaises(RuntimeError):
            he_clustering(bad_matrix, self.genes)

    def test_non_symmetric_matrix(self):
        """Confirm non-symmetric matrix triggers validation error."""

        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = 0.9

        with self.assertRaises(RuntimeError):
            he_clustering(bad_matrix, self.genes)

    def test_negative_values(self):
        """Confirm negative distances are rejected."""

        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = -1.0
        bad_matrix[1, 0] = -1.0

        with self.assertRaises(RuntimeError):
            he_clustering(bad_matrix, self.genes)

    def test_nan_values(self):
        """Confirm NaN values are rejected."""

        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = np.nan
        bad_matrix[1, 0] = np.nan

        with self.assertRaises(RuntimeError):
            he_clustering(bad_matrix, self.genes)

    def test_gene_length_mismatch(self):
        """Confirm mismatch between gene labels and matrix size."""

        with self.assertRaises(RuntimeError):
            he_clustering(self.distance_matrix, ["G1", "G2"])

    ########################## Dendrogram & Export Tests ##########################

    def test_return_figure(self):
        """
        Confirm return_fig=True returns a Plotly Figure.
        """

        labels, fig = he_clustering(
            self.distance_matrix,
            self.genes,
            clustering=ClusteringOptions(num_groups=2),
            return_fig=True
        )

        self.assertEqual(len(labels), 4)
        self.assertTrue(hasattr(fig, "to_html"))

    def test_return_html(self):
        """
        Confirm return_html=True returns HTML output.
        """

        labels, html = he_clustering(
            self.distance_matrix,
            self.genes,
            clustering=ClusteringOptions(num_groups=2),
            return_html=True
        )

        self.assertEqual(len(labels), 4)
        self.assertTrue("<div" in html)

    def test_export_html_temp_directory(self):
        """
        Validate HTML export using temporary directory.
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

        self.assertFalse(os.path.exists(tmpdir))

    ########################## Stability Tests ##########################

    def test_different_linkage_methods(self):
        """
        Confirm multiple linkage methods run without failure.
        """

        for method in ["single", "complete", "average"]:

            labels = he_clustering(
                self.distance_matrix,
                self.genes,
                clustering=ClusteringOptions(num_groups=2, method=method)
            )

            self.assertEqual(len(labels), 4)

    ########################## Dataclass field tests ##########################

    def test_verbose_field_is_overridable(self):
        """
        ClusteringOptions.verbose must be a proper dataclass field so that
        callers can suppress output via ClusteringOptions(verbose=False).

        This would raise TypeError if verbose were a bare class variable
        instead of an annotated field.
        """
        opts_silent = ClusteringOptions(verbose=False)
        self.assertFalse(opts_silent.verbose)

        opts_default = ClusteringOptions()
        self.assertTrue(opts_default.verbose)


if __name__ == "__main__":
    unittest.main()