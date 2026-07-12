"""
Unit tests for He_Inconsistency_Clustering module.

Purpose of this file:
- Validate automatic (inconsistency-coefficient-based) clustering behavior.
- Validate input validation logic.
- Confirm two-panel figure generation and export.
- Validate optional return modes and candidate ranking.
"""

######### Libraries #########
import unittest
import tempfile
import os
import numpy as np
import logging

from gclusters_characterization.clustering.he_inconsistency_clustering import (
    he_inconsistency_clustering,
    compute_inconsistency_clustering,
    InconsistencyClusteringOptions,
    DendrogramOptions,
    ExportOptions,
)


class TestHeInconsistencyClustering(unittest.TestCase):
    """Test suite validating automatic clustering logic and export behavior."""

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

    def tearDown(self):
        """Re-enable logging so it does not leak into other test modules."""
        logging.disable(logging.NOTSET)

    ########################## Core Clustering Tests ##########################

    def test_compute_inconsistency_clustering_basic(self):
        """
        Confirm automatic clustering produces valid outputs.

        Checks:
        - linkage matrix shape
        - label count
        - cophenetic coefficient type
        - inconsistency report structure
        - inconsistency matrix shape
        """

        Z, labels, cophenetic_corr, report, R = compute_inconsistency_clustering(
            self.distance_matrix,
            self.genes,
            InconsistencyClusteringOptions(verbose=False),
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertTrue(isinstance(cophenetic_corr, float))
        self.assertEqual(Z.shape[1], 4)
        self.assertEqual(R.shape, Z.shape)
        self.assertEqual(len(report), 1)
        self.assertEqual(
            set(report[0].keys()),
            {"rank", "k", "coefficient", "cut_height", "merge_height", "labels", "selected"},
        )
        self.assertTrue(report[0]["selected"])

    def test_he_inconsistency_clustering_returns_labels_only(self):
        """
        Confirm default behavior returns cluster labels.
        """

        labels = he_inconsistency_clustering(
            self.distance_matrix,
            self.genes,
            clustering=InconsistencyClusteringOptions(verbose=False),
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_n_candidates_expands_report(self):
        """
        Confirm n_candidates controls how many ranked candidates are returned.
        """

        _, _, _, report, _ = compute_inconsistency_clustering(
            self.distance_matrix,
            self.genes,
            InconsistencyClusteringOptions(verbose=False, n_candidates=3),
        )

        self.assertEqual(len(report), 3)
        self.assertEqual([entry["rank"] for entry in report], [1, 2, 3])

    def test_min_max_clusters_constraints(self):
        """
        Confirm min_clusters/max_clusters bound the selected candidate.
        """

        _, _, _, report, _ = compute_inconsistency_clustering(
            self.distance_matrix,
            self.genes,
            InconsistencyClusteringOptions(verbose=False, min_clusters=2, max_clusters=2),
        )

        self.assertEqual(report[0]["k"], 2)

    ########################## Validation Tests ##########################

    def test_too_few_elements(self):
        """Confirm fewer than 3 elements raises a validation error."""

        small_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])

        with self.assertRaises(ValueError):
            compute_inconsistency_clustering(
                small_matrix,
                ["G1", "G2"],
                InconsistencyClusteringOptions(verbose=False),
            )

    def test_non_square_matrix(self):
        """Confirm non-square matrix triggers validation error."""

        bad_matrix = np.array([[0, 1, 2]])

        with self.assertRaises(RuntimeError):
            he_inconsistency_clustering(bad_matrix, self.genes)

    def test_non_symmetric_matrix(self):
        """Confirm non-symmetric matrix triggers validation error."""

        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = 0.9

        with self.assertRaises(RuntimeError):
            he_inconsistency_clustering(bad_matrix, self.genes)

    def test_negative_values(self):
        """Confirm negative distances are rejected."""

        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = -1.0
        bad_matrix[1, 0] = -1.0

        with self.assertRaises(RuntimeError):
            he_inconsistency_clustering(bad_matrix, self.genes)

    def test_nan_values(self):
        """Confirm NaN values are rejected."""

        bad_matrix = self.distance_matrix.copy()
        bad_matrix[0, 1] = np.nan
        bad_matrix[1, 0] = np.nan

        with self.assertRaises(RuntimeError):
            he_inconsistency_clustering(bad_matrix, self.genes)

    def test_gene_length_mismatch(self):
        """Confirm mismatch between gene labels and matrix size."""

        with self.assertRaises(RuntimeError):
            he_inconsistency_clustering(self.distance_matrix, ["G1", "G2"])

    def test_selected_rank_out_of_range(self):
        """Confirm an out-of-range selected_rank raises a clean error."""

        with self.assertRaises(RuntimeError):
            he_inconsistency_clustering(
                self.distance_matrix,
                self.genes,
                clustering=InconsistencyClusteringOptions(verbose=False),
                selected_rank=5,
            )

    ########################## Figure & Export Tests ##########################

    def test_return_figure(self):
        """
        Confirm return_fig=True returns a Plotly Figure.
        """

        labels, fig = he_inconsistency_clustering(
            self.distance_matrix,
            self.genes,
            clustering=InconsistencyClusteringOptions(verbose=False),
            return_fig=True,
        )

        self.assertEqual(len(labels), 4)
        self.assertTrue(hasattr(fig, "to_html"))

    def test_return_html(self):
        """
        Confirm return_html=True returns HTML output.
        """

        labels, html = he_inconsistency_clustering(
            self.distance_matrix,
            self.genes,
            clustering=InconsistencyClusteringOptions(verbose=False),
            return_html=True,
        )

        self.assertEqual(len(labels), 4)
        self.assertTrue("<div" in html)

    def test_return_report(self):
        """
        Confirm return_report=True returns the ranked inconsistency report.
        """

        labels, report = he_inconsistency_clustering(
            self.distance_matrix,
            self.genes,
            clustering=InconsistencyClusteringOptions(verbose=False, n_candidates=2),
            return_report=True,
        )

        self.assertEqual(len(labels), 4)
        self.assertEqual(len(report), 2)

    def test_export_html_temp_directory(self):
        """
        Validate HTML export using a temporary directory that is fully
        cleaned up afterward (no residual files left on disk).
        """

        with tempfile.TemporaryDirectory() as tmpdir:

            filepath = os.path.join(tmpdir, "inconsistency.html")

            labels = he_inconsistency_clustering(
                self.distance_matrix,
                self.genes,
                clustering=InconsistencyClusteringOptions(verbose=False),
                save_html_to=filepath,
                export=ExportOptions(verbose=False),
            )

            self.assertTrue(os.path.exists(filepath))
            self.assertEqual(len(labels), 4)

        self.assertFalse(os.path.exists(tmpdir))

    ########################## Dataclass field tests ##########################

    def test_verbose_field_is_overridable(self):
        """
        InconsistencyClusteringOptions.verbose must be a proper dataclass
        field so that callers can suppress output via
        InconsistencyClusteringOptions(verbose=False).

        This would raise TypeError if verbose were a bare class variable
        instead of an annotated field.
        """
        opts_silent = InconsistencyClusteringOptions(verbose=False)
        self.assertFalse(opts_silent.verbose)

        opts_default = InconsistencyClusteringOptions()
        self.assertTrue(opts_default.verbose)


if __name__ == "__main__":
    unittest.main()
