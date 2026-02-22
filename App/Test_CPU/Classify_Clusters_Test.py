"""
Unit tests for Classify_Clusters module.

Objetivo:
- Validar combinación de scores.
- Validar detección de umbral GMM con evidencia estadística.
- Confirmar rechazo de GMM en distribución unimodal.
- Validar método histogram valley.
- Validar método percentile.
- Confirmar manejo de errores.
"""

import unittest
import numpy as np
import logging

from ParetoInsight_CPU.Classify_Clusters import (
    combined_score,
    CombinedScoreOptions,
    compute_threshold,
    GMMThresholdOptions,
    HistogramValleyOptions,
    PercentileOptions,
)


class TestClassifyClusters(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL)
        np.random.seed(42)

        # Distribución bimodal clara
        low = np.random.normal(0.25, 0.05, 300)
        high = np.random.normal(0.80, 0.05, 300)
        self.bimodal = np.clip(np.concatenate([low, high]), 0, 1)

        # Distribución unimodal
        self.unimodal = np.clip(np.random.normal(0.5, 0.05, 600), 0, 1)

    # ─────────────────────────────────────────────
    # Combined Score Tests
    # ─────────────────────────────────────────────

    def test_combined_geometric(self):
        J = np.array([0.5, 0.8])
        W = np.array([0.5, 0.2])

        result = combined_score(J, W)
        expected = np.sqrt(J * W)

        np.testing.assert_allclose(result, expected)

    def test_combined_linear(self):
        J = np.array([0.4])
        W = np.array([0.6])

        result = combined_score(
            J, W,
            options=CombinedScoreOptions(method="linear", alpha=0.7)
        )

        self.assertAlmostEqual(result[0], 0.7*0.4 + 0.3*0.6)

    def test_combined_invalid_range(self):
        J = np.array([1.2])
        W = np.array([0.5])
        with self.assertRaises(ValueError):
            combined_score(J, W)

    # ─────────────────────────────────────────────
    # GMM Threshold Tests
    # ─────────────────────────────────────────────

    def test_gmm_bimodal_detected(self):
        result = compute_threshold(
            self.bimodal,
            method="gmm",
            gmm=GMMThresholdOptions(
                min_separation_index=1.0,
                min_bic_improvement=5.0
            )
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.details["validation"].accepted)
        self.assertGreater(result.threshold, 0.3)
        self.assertLess(result.threshold, 0.75)

    def test_gmm_unimodal_rejected(self):
        result = compute_threshold(
            self.unimodal,
            method="gmm",
            gmm=GMMThresholdOptions(
                min_separation_index=1.5,
                min_bic_improvement=10.0
            )
        )

        self.assertIsNone(result)

    def test_gmm_small_sample(self):
        small = np.array([0.1, 0.2, 0.3])
        result = compute_threshold(
            small,
            method="gmm"
        )
        self.assertIsNone(result)

    # ─────────────────────────────────────────────
    # Histogram Valley Tests
    # ─────────────────────────────────────────────

    def test_histogram_valley_detected(self):
        result = compute_threshold(
            self.bimodal,
            method="histogram_valley",
            hist=HistogramValleyOptions(bins=50)
        )

        self.assertIsNotNone(result)
        self.assertGreater(result.threshold, 0.3)
        self.assertLess(result.threshold, 0.75)

    def test_histogram_valley_unimodal(self):
        result = compute_threshold(
            self.unimodal,
            method="histogram_valley"
        )
        # Puede devolver None o threshold inestable.
        # Validamos que si existe, no sea extremo.
        if result is not None:
            self.assertGreater(result.threshold, 0.1)
            self.assertLess(result.threshold, 0.9)

    # ─────────────────────────────────────────────
    # Percentile Tests
    # ─────────────────────────────────────────────

    def test_percentile_threshold(self):
        result = compute_threshold(
            self.bimodal,
            method="percentile",
            perc=PercentileOptions(percentile=10.0)
        )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.threshold, 0.0)
        self.assertLessEqual(result.threshold, 1.0)

    def test_percentile_invalid(self):
        with self.assertRaises(ValueError):
            compute_threshold(
                self.bimodal,
                method="percentile",
                perc=PercentileOptions(percentile=150.0)
            )

    # ─────────────────────────────────────────────
    # Error Handling
    # ─────────────────────────────────────────────

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            compute_threshold(self.bimodal, method="invalid_method")


if __name__ == "__main__":
    unittest.main()