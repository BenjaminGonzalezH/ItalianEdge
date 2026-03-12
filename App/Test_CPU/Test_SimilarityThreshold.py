"""
Unit tests for SimilarityThreshold module.

Purpose
-------
- Validate GMM threshold estimation.
- Validate input validation.
- Ensure plotting/export works correctly.
- Confirm optional return modes.
"""

######### Libraries #########

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import logging

from ParetoInsight_CPU.SimilarityThreshold import (
    compute_gmm_threshold,
    plot_gmm_threshold,
    estimate_similarity_threshold,
    GMMThresholdOptions,
)


class TestSimilarityThreshold(unittest.TestCase):

    ##########################
    # Test Initialization
    ##########################

    def setUp(self):
        """
        Generate deterministic bimodal distribution
        to simulate similarity values.
        """

        logging.disable(logging.CRITICAL)

        np.random.seed(0)

        cluster1 = np.random.normal(0.2, 0.03, 200)
        cluster2 = np.random.normal(0.75, 0.05, 200)

        self.values = np.concatenate([cluster1, cluster2])

        self.df = pd.DataFrame({
            "similarity": self.values
        })

    ##########################
    # Core Computation Tests
    ##########################

    def test_compute_gmm_threshold(self):
        """
        Confirm threshold estimation returns valid value.
        """

        threshold, model = compute_gmm_threshold(self.values)

        self.assertTrue(0.3 < threshold < 0.6)
        self.assertEqual(model.n_components, 2)

    ##########################
    # DataFrame Wrapper Tests
    ##########################

    def test_estimate_threshold_from_dataframe(self):

        threshold = estimate_similarity_threshold(
            self.df,
            column="similarity"
        )

        self.assertTrue(0.3 < threshold < 0.6)

    ##########################
    # Visualization Tests
    ##########################

    def test_plot_generation(self):

        threshold, model = compute_gmm_threshold(self.values)

        fig = plot_gmm_threshold(
            self.values,
            threshold,
            model,
            return_fig=True
        )

        self.assertTrue(hasattr(fig, "savefig"))

    ##########################
    # Export Tests
    ##########################

    def test_png_export(self):
        """
        Ensure PNG export works correctly.
        """

        threshold, model = compute_gmm_threshold(self.values)

        with tempfile.TemporaryDirectory() as tmpdir:

            filepath = os.path.join(tmpdir, "gmm_plot.png")

            plot_gmm_threshold(
                self.values,
                threshold,
                model,
                save_png_to=filepath
            )

            self.assertTrue(os.path.exists(filepath))

        self.assertFalse(os.path.exists(tmpdir))

    ##########################
    # Validation Tests
    ##########################

    def test_invalid_input_type(self):

        with self.assertRaises(TypeError):
            compute_gmm_threshold("not an array")

    def test_invalid_dimension(self):

        bad_values = np.array([[1, 2], [3, 4]])

        with self.assertRaises(ValueError):
            compute_gmm_threshold(bad_values)

    def test_nan_values(self):

        bad_values = self.values.copy()
        bad_values[0] = np.nan

        with self.assertRaises(ValueError):
            compute_gmm_threshold(bad_values)

    def test_missing_column(self):

        with self.assertRaises(ValueError):

            estimate_similarity_threshold(
                self.df,
                column="missing_column"
            )


if __name__ == "__main__":
    unittest.main()