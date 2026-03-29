"""
Unit tests for similarity_threshold module.

Purpose
-------
Validate Gaussian Mixture Model (GMM)-based threshold estimation utilities.

Coverage goals
--------------
- Validate input checking
- Validate similarity metric combination
- Validate GMM threshold computation
- Validate Gaussian intersection extraction
- Validate interactive HTML plotting
- Validate single-column and combined-column APIs
- Validate file export behavior using temporary files
"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture

from gclusters_characterization.clustering.similarity_threshold import (
    GMMThresholdOptions,
    _validate_values,
    combine_similarity_metrics,
    _extract_gmm_parameters,
    _compute_gaussian_intersections,
    compute_gmm_threshold,
    _plot_gmm_html,
    estimate_similarity_threshold,
    estimate_similarity_threshold_combined,
)


class TestSimilarityThreshold(unittest.TestCase):
    """
    Test suite for GMM-based similarity threshold estimation.
    """

    def setUp(self):
        """
        Build deterministic bimodal similarity distributions.

        The two modes are intentionally separated so that
        Gaussian intersections are easy to detect.
        """

        rng = np.random.default_rng(0)

        cluster1 = rng.normal(0.20, 0.03, 200)
        cluster2 = rng.normal(0.75, 0.05, 200)

        self.values = np.concatenate([cluster1, cluster2])

        self.df = pd.DataFrame({
            "similarity_1": self.values,
            "similarity_2": self.values * 0.95 + 0.02,
        })

        self.options = GMMThresholdOptions(
            n_components=2,
            random_state=0,
            bins=30,
            verbose=False,
        )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def test_validate_values_valid(self):
        """Valid 1D finite arrays should pass validation."""
        _validate_values(self.values)

    def test_validate_values_invalid_dimension(self):
        """Non-1D arrays must raise ValueError."""
        bad = np.array([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            _validate_values(bad)

    def test_validate_values_too_few(self):
        """Very small arrays must raise ValueError."""
        bad = np.array([0.1, 0.2, 0.3])

        with self.assertRaises(ValueError):
            _validate_values(bad)

    def test_validate_values_nan(self):
        """Arrays containing NaN must raise ValueError."""
        bad = self.values.copy()
        bad[0] = np.nan

        with self.assertRaises(ValueError):
            _validate_values(bad)

    # --------------------------------------------------
    # Metric combination
    # --------------------------------------------------

    def test_combine_similarity_metrics_mean(self):
        """Mean combination must match arithmetic average."""
        a = np.array([0.2, 0.4, 0.6])
        b = np.array([0.4, 0.6, 0.8])

        opts = GMMThresholdOptions(combination_method="mean", verbose=False)

        combined = combine_similarity_metrics(a, b, opts)

        expected = np.array([0.3, 0.5, 0.7])
        np.testing.assert_allclose(combined, expected)

    def test_combine_similarity_metrics_weighted_mean(self):
        """Weighted mean must respect provided weights."""
        a = np.array([0.2, 0.4])
        b = np.array([0.6, 0.8])

        opts = GMMThresholdOptions(
            combination_method="weighted_mean",
            weight_1=1.0,
            weight_2=3.0,
            verbose=False,
        )

        combined = combine_similarity_metrics(a, b, opts)

        expected = (1.0 * a + 3.0 * b) / 4.0
        np.testing.assert_allclose(combined, expected)

    def test_combine_similarity_metrics_product(self):
        """Product combination must multiply values element-wise."""
        a = np.array([0.2, 0.5])
        b = np.array([0.5, 0.4])

        opts = GMMThresholdOptions(combination_method="product", verbose=False)

        combined = combine_similarity_metrics(a, b, opts)
        expected = np.array([0.1, 0.2])

        np.testing.assert_allclose(combined, expected)

    def test_combine_similarity_metrics_geometric(self):
        """Geometric combination must compute sqrt(a*b)."""
        a = np.array([0.25, 0.36])
        b = np.array([0.81, 0.25])

        opts = GMMThresholdOptions(combination_method="geometric", verbose=False)

        combined = combine_similarity_metrics(a, b, opts)
        expected = np.sqrt(a * b)

        np.testing.assert_allclose(combined, expected)

    def test_combine_similarity_metrics_harmonic(self):
        """Harmonic combination must return positive bounded values."""
        a = np.array([0.2, 0.4, 0.6])
        b = np.array([0.3, 0.5, 0.9])

        opts = GMMThresholdOptions(combination_method="harmonic", verbose=False)

        combined = combine_similarity_metrics(a, b, opts)

        self.assertEqual(combined.shape, a.shape)
        self.assertTrue(np.all(np.isfinite(combined)))

    def test_combine_similarity_metrics_shape_mismatch(self):
        """Mismatched input lengths must raise ValueError."""
        a = np.array([0.1, 0.2])
        b = np.array([0.1, 0.2, 0.3])

        with self.assertRaises(ValueError):
            combine_similarity_metrics(a, b, self.options)

    def test_combine_similarity_metrics_invalid_method(self):
        """Unknown combination methods must raise ValueError."""
        a = np.array([0.1, 0.2])
        b = np.array([0.1, 0.2])

        opts = GMMThresholdOptions(combination_method="invalid", verbose=False)

        with self.assertRaises(ValueError):
            combine_similarity_metrics(a, b, opts)

    # --------------------------------------------------
    # GMM parameter extraction and intersections
    # --------------------------------------------------

    def test_extract_gmm_parameters_sorted(self):
        """Extracted Gaussian parameters must be sorted by mean."""
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(self.values.reshape(-1, 1))

        means, stds, weights = _extract_gmm_parameters(gmm)

        self.assertEqual(len(means), 2)
        self.assertEqual(len(stds), 2)
        self.assertEqual(len(weights), 2)
        self.assertTrue(np.all(np.diff(means) >= 0))

    def test_compute_gaussian_intersections_basic(self):
        """Separated Gaussians should produce at least one intersection."""
        means = np.array([0.2, 0.8])
        stds = np.array([0.05, 0.05])
        weights = np.array([0.5, 0.5])

        intersections = _compute_gaussian_intersections(means, stds, weights)

        self.assertTrue(len(intersections) >= 1)
        self.assertTrue(0.2 < intersections[0] < 0.8)

    # --------------------------------------------------
    # Core GMM threshold computation
    # --------------------------------------------------

    def test_compute_gmm_threshold(self):
        """Threshold computation should return sorted thresholds and fitted model."""
        thresholds, gmm = compute_gmm_threshold(self.values, self.options)

        self.assertTrue(isinstance(thresholds, list))
        self.assertTrue(len(thresholds) >= 1)
        self.assertTrue(all(np.isfinite(thresholds)))
        self.assertTrue(isinstance(gmm, GaussianMixture))
        self.assertTrue(thresholds == sorted(thresholds))

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------

    def test_plot_gmm_html_returns_figure(self):
        """GMM plot helper should return a Plotly figure."""
        thresholds, gmm = compute_gmm_threshold(self.values, self.options)

        fig = _plot_gmm_html(
            self.values,
            thresholds,
            gmm,
            self.options,
            title="Test Plot",
        )

        self.assertTrue(isinstance(fig, go.Figure))
        self.assertTrue(len(fig.data) >= 2)

    # --------------------------------------------------
    # High-level API: single column
    # --------------------------------------------------

    def test_estimate_similarity_threshold_basic(self):
        """Single-column API should return a sorted list of thresholds."""
        thresholds = estimate_similarity_threshold(
            self.df,
            "similarity_1",
            options=self.options,
            plot=False,
        )

        self.assertTrue(isinstance(thresholds, list))
        self.assertTrue(len(thresholds) >= 1)
        self.assertTrue(thresholds == sorted(thresholds))

    def test_estimate_similarity_threshold_missing_column(self):
        """Missing columns must raise ValueError."""
        with self.assertRaises(ValueError):
            estimate_similarity_threshold(
                self.df,
                "missing_column",
                options=self.options,
            )

    def test_estimate_similarity_threshold_plot_and_save(self):
        """Single-column API should export HTML when requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "gmm_single.html"

            thresholds = estimate_similarity_threshold(
                self.df,
                "similarity_1",
                options=self.options,
                plot=True,
                save_html_to=filepath,
            )

            self.assertTrue(filepath.exists())
            self.assertTrue(len(thresholds) >= 1)

    # --------------------------------------------------
    # High-level API: combined columns
    # --------------------------------------------------

    def test_estimate_similarity_threshold_combined_basic(self):
        """Combined-column API must return thresholds; column added when opt-in."""
        df_copy = self.df.copy()

        thresholds = estimate_similarity_threshold_combined(
            df_copy,
            ("similarity_1", "similarity_2"),
            options=self.options,
            plot=False,
            add_combined_column=True,
        )

        self.assertTrue(isinstance(thresholds, list))
        self.assertTrue(len(thresholds) >= 1)
        self.assertIn("combined_similarity", df_copy.columns)

    def test_estimate_no_combined_column_by_default(self):
        """Combined-column API must NOT mutate the caller's DataFrame by default."""
        df_copy = self.df.copy()
        original_columns = set(df_copy.columns)

        estimate_similarity_threshold_combined(
            df_copy,
            ("similarity_1", "similarity_2"),
            options=self.options,
            plot=False,
        )

        self.assertEqual(set(df_copy.columns), original_columns,
                         "'combined_similarity' column was added without opt-in")

    def test_estimate_similarity_threshold_combined_missing_columns(self):
        """Missing combined columns must raise ValueError."""
        with self.assertRaises(ValueError):
            estimate_similarity_threshold_combined(
                self.df.copy(),
                ("similarity_1", "missing"),
                options=self.options,
            )

    def test_estimate_similarity_threshold_combined_plot_and_save(self):
        """Combined-column API should export HTML correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "gmm_combined.html"
            df_copy = self.df.copy()

            thresholds = estimate_similarity_threshold_combined(
                df_copy,
                ("similarity_1", "similarity_2"),
                options=self.options,
                plot=True,
                save_html_to=filepath,
            )

            self.assertTrue(filepath.exists())
            self.assertTrue(len(thresholds) >= 1)


if __name__ == "__main__":
    unittest.main()