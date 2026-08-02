"""
Unit tests for the heatmaps visualization module.

Purpose of this file:
- Validate the clustered heatmap figure/HTML generation (HoloViews + Bokeh).
- Validate row/column dendrogram toggling.
- Validate input validation logic.
- Validate downsampling for large matrices.
- Confirm HTML export writes to disk without leaving residual files.
"""

######### Libraries #########
import unittest
import tempfile
import os
import logging
import numpy as np

from biocluster.visualization.heatmaps import (
    plot_clustered_heatmap,
    HeatmapExportOptions,
    HeatmapScaleOptions,
    ClusteringOptions,
)


class TestPlotClusteredHeatmap(unittest.TestCase):
    """Test suite validating the clustered heatmap public API."""

    ########################## Test Initialization ##########################

    def setUp(self):
        """
        Build a small, reusable, symmetric similarity matrix (values in
        [0, 1], unit diagonal) plus matching labels.
        """

        # Disable logs and prints produced by verbose options
        logging.disable(logging.CRITICAL)

        rng = np.random.default_rng(0)
        n = 6
        sim = rng.random((n, n))
        sim = (sim + sim.T) / 2.0
        np.fill_diagonal(sim, 1.0)

        self.matrix = sim
        self.labels = [f"G{i}" for i in range(n)]
        self.export = HeatmapExportOptions(verbose=False)

    def tearDown(self):
        """Re-enable logging so it does not leak into other test modules."""
        logging.disable(logging.NOTSET)

    ########################## Core Rendering Tests ##########################

    def test_returns_layout_with_both_dendrograms_by_default(self):
        """
        Confirm the default configuration (both dendrograms enabled)
        returns a HoloViews Layout object.
        """

        layout = plot_clustered_heatmap(
            self.matrix, self.labels, save_filepath=None, return_fig=True, export=self.export,
        )

        self.assertTrue(hasattr(layout, "opts"))

    def test_returns_heatmap_element_without_dendrograms(self):
        """
        Confirm disabling both row/col clustering returns a bare HeatMap
        element (no dendrogram panels composed around it).
        """

        layout = plot_clustered_heatmap(
            self.matrix, self.labels, save_filepath=None, return_fig=True, export=self.export,
            clustering=ClusteringOptions(cluster_rows=False, cluster_cols=False),
        )

        self.assertEqual(type(layout).__name__, "HeatMap")

    def test_returns_layout_with_only_row_dendrogram(self):
        """Confirm enabling only row clustering still returns a composed Layout."""

        layout = plot_clustered_heatmap(
            self.matrix, self.labels, save_filepath=None, return_fig=True, export=self.export,
            clustering=ClusteringOptions(cluster_rows=True, cluster_cols=False),
        )

        self.assertTrue(hasattr(layout, "opts"))
        self.assertNotEqual(type(layout).__name__, "HeatMap")

    def test_return_html_is_standalone_bokeh_document(self):
        """Confirm return_html=True yields a standalone Bokeh HTML string."""

        html = plot_clustered_heatmap(
            self.matrix, self.labels, save_filepath=None, return_html=True, export=self.export,
        )

        self.assertIsInstance(html, str)
        self.assertIn("bokeh", html.lower())

    def test_return_both_fig_and_html(self):
        """Confirm return_fig=True and return_html=True yields a (layout, html) tuple."""

        layout, html = plot_clustered_heatmap(
            self.matrix, self.labels, save_filepath=None,
            return_fig=True, return_html=True, export=self.export,
        )

        self.assertTrue(hasattr(layout, "opts"))
        self.assertIsInstance(html, str)

    def test_default_return_is_none(self):
        """Confirm the default call (no return flags) returns None."""

        result = plot_clustered_heatmap(self.matrix, self.labels, save_filepath=None, export=self.export)

        self.assertIsNone(result)

    ########################## Validation Tests ##########################

    def test_non_ndarray_matrix_raises(self):
        """Confirm a non-ndarray matrix raises TypeError."""

        with self.assertRaises(TypeError):
            plot_clustered_heatmap([[1, 2], [3, 4]], save_filepath=None, export=self.export)

    def test_non_square_matrix_raises(self):
        """Confirm a non-square matrix raises ValueError."""

        with self.assertRaises(ValueError):
            plot_clustered_heatmap(np.zeros((2, 3)), save_filepath=None, export=self.export)

    def test_too_small_matrix_raises(self):
        """Confirm a 1x1 matrix raises ValueError (needs at least 2 rows/cols)."""

        with self.assertRaises(ValueError):
            plot_clustered_heatmap(np.array([[1.0]]), save_filepath=None, export=self.export)

    def test_labels_length_mismatch_raises(self):
        """Confirm a labels list not matching the matrix size raises ValueError."""

        with self.assertRaises(ValueError):
            plot_clustered_heatmap(self.matrix, self.labels[:2], save_filepath=None, export=self.export)

    def test_empty_matrix_raises(self):
        """Confirm an empty matrix raises ValueError."""

        with self.assertRaises(ValueError):
            plot_clustered_heatmap(np.empty((0, 0)), save_filepath=None, export=self.export)

    ########################## Downsampling Tests ##########################

    def test_large_matrix_is_downsampled(self):
        """
        Confirm a matrix larger than max_dim is pooled down and still
        produces a valid figure (labels are regenerated as plain indices).
        """

        rng = np.random.default_rng(1)
        big = rng.random((30, 30))
        big = (big + big.T) / 2.0
        np.fill_diagonal(big, 1.0)

        layout = plot_clustered_heatmap(
            big, save_filepath=None, return_fig=True, export=self.export,
            scale=HeatmapScaleOptions(max_dim=10, downsample_mode="pool_mean"),
        )

        self.assertTrue(hasattr(layout, "opts"))

    ########################## Export Tests ##########################

    def test_export_html_temp_directory(self):
        """
        Validate HTML export using a temporary directory that is fully
        cleaned up afterward (no residual files left on disk).
        """

        with tempfile.TemporaryDirectory() as tmpdir:

            filepath = os.path.join(tmpdir, "heatmap.html")

            result = plot_clustered_heatmap(
                self.matrix, self.labels, save_filepath=filepath, export=self.export,
            )

            self.assertIsNone(result)
            self.assertTrue(os.path.exists(filepath))

        self.assertFalse(os.path.exists(tmpdir))


if __name__ == "__main__":
    unittest.main()
