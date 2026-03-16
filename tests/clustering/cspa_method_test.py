"""
Unit tests for CSPA method utilities.

Purpose
-------
Validate functionality of the CSPA consensus clustering pipeline,
including spectral clustering, embedding visualization, and HTML export.

Coverage goals
--------------
• validate matrix validation logic
• validate spectral clustering output
• validate embedding generation
• validate Plotly figure generation
• validate HTML export
• validate end-to-end pipeline execution
"""

######### Libraries #########

import unittest
import tempfile
import numpy as np
from pathlib import Path

import plotly.graph_objects as go

from gclusters_characterization.clustering.cspa_method import (
    CSPAOptions,
    EmbedOptions,
    ExportHTML,
    _validate_square_numeric,
    _validate_genes,
    _validate_coincidence_matrix,
    cspa_spectral_from_coincidence,
    _factorize_labels,
    _spectral_embedding_2d,
    build_click_highlight_embedding_figure,
    figure_to_html_with_click_highlight,
    plot_embedding_click_highlight,
    cspa_method,
)


class TestCSPAMethod(unittest.TestCase):
    """
    Test suite for CSPA consensus clustering module.
    """

    ##############################
    # Test initialization
    ##############################

    def setUp(self):
        """
        Create small deterministic coincidence matrix.
        """

        self.coincidence = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ])

        self.genes = ["g1", "g2", "g3", "g4"]

        self.options = CSPAOptions(n_clusters=2, random_state=0)

        self.embed = EmbedOptions(n_components=2)

        self.export = ExportHTML(verbose=False)

    ##############################
    # Validation helpers
    ##############################

    def test_validate_square_numeric_valid(self):
        """Valid square numeric matrix should pass."""
        _validate_square_numeric(self.coincidence, "test")

    def test_validate_square_numeric_invalid_type(self):
        """Non-numpy input must raise TypeError."""
        with self.assertRaises(TypeError):
            _validate_square_numeric([[1, 2], [3, 4]], "test")

    def test_validate_square_numeric_not_square(self):
        """Non-square matrices must raise ValueError."""
        with self.assertRaises(ValueError):
            _validate_square_numeric(np.ones((2, 3)), "test")

    def test_validate_genes_length(self):
        """Gene list must match matrix size."""
        with self.assertRaises(ValueError):
            _validate_genes(["a", "b"], 4)

    def test_validate_coincidence_matrix(self):
        """Valid coincidence matrix should pass validation."""
        _validate_coincidence_matrix(self.coincidence)

    ##############################
    # Spectral clustering
    ##############################

    def test_cspa_spectral_output_shape(self):
        """Spectral clustering must return label vector."""
        labels = cspa_spectral_from_coincidence(
            self.coincidence,
            self.options,
        )

        self.assertEqual(labels.shape, (4,))
        self.assertTrue(len(np.unique(labels)) <= 2)

    def test_cspa_invalid_cluster_count(self):
        """Invalid cluster count must raise error."""
        with self.assertRaises(ValueError):
            bad = CSPAOptions(n_clusters=10)
            cspa_spectral_from_coincidence(self.coincidence, bad)

    ##############################
    # Factorization
    ##############################

    def test_factorize_labels(self):
        """Label factorization should produce compact codes."""
        labels = np.array(["A", "A", "B"])

        unique, codes = _factorize_labels(labels)

        self.assertEqual(len(unique), 2)
        self.assertEqual(len(codes), 3)

    ##############################
    # Spectral embedding
    ##############################

    def test_spectral_embedding_shape(self):
        """Embedding must return coordinates matrix."""
        coords = _spectral_embedding_2d(
            self.coincidence,
            self.embed,
        )

        self.assertEqual(coords.shape, (4, 2))

    ##############################
    # Plotly figure generation
    ##############################

    def test_build_embedding_figure(self):
        """Figure generation must produce Plotly figure."""
        labels = np.array([0, 0, 1, 1])

        fig = build_click_highlight_embedding_figure(
            self.coincidence,
            labels,
            self.genes,
        )

        self.assertTrue(isinstance(fig, go.Figure))
        self.assertTrue(len(fig.data) > 0)

    ##############################
    # HTML export
    ##############################

    def test_html_generation(self):
        """HTML export must produce valid HTML string."""
        labels = np.array([0, 0, 1, 1])

        fig = build_click_highlight_embedding_figure(
            self.coincidence,
            labels,
            self.genes,
        )

        html = figure_to_html_with_click_highlight(fig)

        self.assertTrue(isinstance(html, str))
        self.assertIn("<script>", html)

    ##############################
    # HTML file export
    ##############################

    def test_html_file_export(self):
        """Exporting HTML should write a file."""
        labels = np.array([0, 0, 1, 1])

        with tempfile.TemporaryDirectory() as tmp:

            path = Path(tmp) / "plot.html"

            plot_embedding_click_highlight(
                affinity_matrix=self.coincidence,
                labels=labels,
                genes=self.genes,
                save_html_to=path,
            )

            self.assertTrue(path.exists())

    ##############################
    # Pipeline execution
    ##############################

    def test_cspa_pipeline_labels(self):
        """Pipeline should return cluster labels."""
        labels, fig, html = cspa_method(
            coincidence_matrix=self.coincidence,
            genes=self.genes,
            cspa=self.options,
        )

        self.assertEqual(labels.shape, (4,))
        self.assertIsNone(fig)
        self.assertIsNone(html)

    def test_cspa_pipeline_return_fig(self):
        """Pipeline must return figure when requested."""
        labels, fig, html = cspa_method(
            coincidence_matrix=self.coincidence,
            genes=self.genes,
            cspa=self.options,
            return_fig=True,
        )

        self.assertTrue(isinstance(fig, go.Figure))

    def test_cspa_pipeline_return_html(self):
        """Pipeline must return HTML when requested."""
        labels, fig, html = cspa_method(
            coincidence_matrix=self.coincidence,
            genes=self.genes,
            cspa=self.options,
            return_html=True,
        )

        self.assertTrue(isinstance(html, str))


if __name__ == "__main__":
    unittest.main()