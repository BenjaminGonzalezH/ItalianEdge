"""
Unit tests for Heatmaps utilities.

Purpose:
- Validate normal and edge-case behavior of heatmap generation.
- Ensure validation logic works correctly.
- Verify HTML generation without leaving residual files.
- Confirm downsampling behavior.
"""

######### Libraries #########
import unittest
import os
import numpy as np
import tempfile

from Graphs.Heatmaps import (
    plot_html_heatmap,
    plot_dual_heatmap_two_colors,
    HeatmapScaleOptions,
    HeatmapExportOptions
)


class TestHeatmaps(unittest.TestCase):
    """Test suite for Heatmaps module."""

    ##########################
    # Setup / Teardown
    ##########################

    def setUp(self):
        """Create reusable small matrices for testing."""
        self.small_matrix = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)

        self.large_matrix = np.random.rand(2000, 2000)

        self.invalid_matrix = np.array([1, 2, 3])  # Not 2D

    ##########################
    # Basic functionality
    ##########################

    def test_plot_returns_figure(self):
        """Ensure plot_html_heatmap returns a Plotly Figure when requested."""
        fig = plot_html_heatmap(
            self.small_matrix,
            save_filepath=None,
            return_fig=True
        )
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)

    def test_plot_returns_html(self):
        """Ensure plot_html_heatmap returns HTML string when requested."""
        html = plot_html_heatmap(
            self.small_matrix,
            save_filepath=None,
            return_html=True
        )
        self.assertIsInstance(html, str)
        self.assertIn("<div", html)

    ##########################
    # Validation errors
    ##########################

    def test_invalid_matrix_type(self):
        """Non-numpy input must raise TypeError."""
        with self.assertRaises(TypeError):
            plot_html_heatmap([[1, 2], [3, 4]])

    def test_invalid_matrix_dimension(self):
        """Non-2D matrix must raise ValueError."""
        with self.assertRaises(ValueError):
            plot_html_heatmap(self.invalid_matrix)

    def test_empty_matrix(self):
        """Empty matrix must raise ValueError."""
        with self.assertRaises(ValueError):
            plot_html_heatmap(np.array([[]]))

    ##########################
    # Downsampling behavior
    ##########################

    def test_downsampling_triggered(self):
        """Large matrix should trigger downsampling."""
        scale_opts = HeatmapScaleOptions(max_dim=500)

        fig = plot_html_heatmap(
            self.large_matrix,
            save_filepath=None,
            scale=scale_opts,
            return_fig=True
        )

        z_shape = fig.data[0].z.shape
        self.assertLessEqual(z_shape[0], 500)
        self.assertLessEqual(z_shape[1], 500)

    ##########################
    # File writing
    ##########################

    def test_html_file_creation(self):
        """Ensure HTML file is written and removed properly."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            tmp_path = tmp.name

        try:
            plot_html_heatmap(
                self.small_matrix,
                save_filepath=tmp_path,
                export=HeatmapExportOptions(verbose=False)
            )
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    ##########################
    # Dual heatmap
    ##########################

    def test_dual_heatmap_basic(self):
        """Ensure dual heatmap returns figure correctly."""
        upper = self.small_matrix
        lower = self.small_matrix

        fig = plot_dual_heatmap_two_colors(
            upper,
            lower,
            save_filepath=None,
            return_fig=True
        )

        self.assertIsNotNone(fig)
        # 2 heatmaps + 2 highlight traces
        self.assertEqual(len(fig.data), 4)

    def test_dual_invalid_shape(self):
        """Different shapes must raise ValueError."""
        with self.assertRaises(ValueError):
            plot_dual_heatmap_two_colors(
                self.small_matrix,
                np.random.rand(4, 4),
                save_filepath=None
            )

    def test_dual_not_square(self):
        """Non-square matrices must raise ValueError."""
        nonsquare = np.random.rand(3, 4)
        with self.assertRaises(ValueError):
            plot_dual_heatmap_two_colors(
                nonsquare,
                nonsquare,
                save_filepath=None
            )


if __name__ == "__main__":
    unittest.main()
