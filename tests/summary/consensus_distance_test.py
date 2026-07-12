"""
Unit tests for the consensus_distance summary module.

Purpose of this file:
- Validate the per-solution distance-to-consensus scoring table.
- Validate outlier selection (z-score threshold and fixed top-K).
- Validate input validation logic.
- Confirm the two-panel summary figure is generated correctly.
"""

######### Libraries #########
import unittest
import numpy as np
import pandas as pd

from gclusters_characterization.summary.consensus_distance import (
    compute_consensus_distance_scores,
    identify_outlier_solutions_vs_consensus,
    get_outlier_solution_indices,
    plot_consensus_distance_summary,
    ConsensusDistanceOptions,
)


class TestConsensusDistance(unittest.TestCase):
    """Test suite validating consensus-distance scoring and outlier detection."""

    ########################## Test Initialization ##########################

    def setUp(self):
        """
        Create a reusable set of distances-to-consensus.

        Structure: five "typical" solutions clustered around 0.1-0.13,
        plus one clear outlier at 0.9.
        """

        self.distances = [0.10, 0.12, 0.11, 0.13, 0.90, 0.10]
        self.labels = ["s0", "s1", "s2", "s3", "s4", "s5"]
        self.solution_matrix = np.array([[1, 1, 2, 2]] * 6)

    ########################## Scoring Tests ##########################

    def test_compute_scores_basic(self):
        """
        Confirm the scores table has the expected shape, columns, and
        descending sort by distance.
        """

        df = compute_consensus_distance_scores(self.distances)

        self.assertEqual(len(df), 6)
        self.assertListEqual(list(df.columns), ["Solution", "distance", "z_score"])
        self.assertTrue(df["distance"].is_monotonic_decreasing)
        self.assertEqual(df.iloc[0]["Solution"], 4)

    def test_compute_scores_with_labels_and_solution_matrix(self):
        """
        Confirm optional "label" and "n_clusters" columns are added when
        the corresponding inputs are provided.
        """

        df = compute_consensus_distance_scores(
            self.distances, labels=self.labels, solution_matrix=self.solution_matrix
        )

        self.assertIn("label", df.columns)
        self.assertIn("n_clusters", df.columns)
        self.assertEqual(df.iloc[0]["label"], "s4")
        self.assertEqual(df.iloc[0]["n_clusters"], 2)

    def test_zero_variance_distances_yield_zero_z_scores(self):
        """
        Confirm identical distances (zero standard deviation) do not
        produce NaN/Inf z-scores.
        """

        df = compute_consensus_distance_scores([0.5, 0.5, 0.5])

        self.assertTrue((df["z_score"] == 0.0).all())

    ########################## Validation Tests ##########################

    def test_too_few_solutions_raises(self):
        """Confirm fewer than 3 distances raises a validation error."""

        with self.assertRaises(ValueError):
            compute_consensus_distance_scores([0.1, 0.2])

    def test_labels_length_mismatch_raises(self):
        """Confirm mismatched labels length raises a validation error."""

        with self.assertRaises(ValueError):
            compute_consensus_distance_scores(self.distances, labels=["only_one"])

    def test_solution_matrix_length_mismatch_raises(self):
        """Confirm mismatched solution_matrix length raises a validation error."""

        with self.assertRaises(ValueError):
            compute_consensus_distance_scores(
                self.distances, solution_matrix=self.solution_matrix[:2]
            )

    ########################## Outlier Selection Tests ##########################

    def test_identify_outliers_by_z_threshold(self):
        """
        Confirm the clear outlier (distance=0.90) is flagged under the
        default z-score threshold.
        """

        outliers = identify_outlier_solutions_vs_consensus(
            self.distances, self.labels, self.solution_matrix,
            options=ConsensusDistanceOptions(z_threshold=1.0),
        )

        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers.iloc[0]["Solution"], 4)

    def test_identify_outliers_by_top_k(self):
        """Confirm top_k overrides the z-score threshold and returns exactly k rows."""

        outliers = identify_outlier_solutions_vs_consensus(
            self.distances, options=ConsensusDistanceOptions(top_k=2)
        )

        self.assertEqual(len(outliers), 2)

    def test_identify_outliers_invalid_top_k_raises(self):
        """Confirm a non-positive top_k raises a validation error."""

        with self.assertRaises(ValueError):
            identify_outlier_solutions_vs_consensus(
                self.distances, options=ConsensusDistanceOptions(top_k=0)
            )

    def test_no_outliers_returns_empty_dataframe(self):
        """Confirm an unreachable z-threshold returns an empty DataFrame."""

        outliers = identify_outlier_solutions_vs_consensus(
            self.distances, options=ConsensusDistanceOptions(z_threshold=100.0)
        )

        self.assertTrue(outliers.empty)

    ########################## Index Extraction Tests ##########################

    def test_get_outlier_solution_indices(self):
        """Confirm indices are extracted, sorted, and de-duplicated."""

        outliers = identify_outlier_solutions_vs_consensus(
            self.distances, options=ConsensusDistanceOptions(top_k=3)
        )

        indices = get_outlier_solution_indices(outliers)

        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(indices), 3)

    def test_get_outlier_solution_indices_empty(self):
        """Confirm an empty outliers table yields an empty index list."""

        self.assertEqual(get_outlier_solution_indices(pd.DataFrame()), [])

    ########################## Figure Tests ##########################

    def test_plot_returns_figure_with_traces(self):
        """Confirm the summary figure is built with the expected panels."""

        scores_df = compute_consensus_distance_scores(self.distances)
        outliers_df = identify_outlier_solutions_vs_consensus(
            self.distances, options=ConsensusDistanceOptions(z_threshold=1.0)
        )

        fig = plot_consensus_distance_summary(scores_df, outliers_df)

        self.assertTrue(hasattr(fig, "to_html"))
        self.assertGreater(len(fig.data), 0)

    def test_plot_with_empty_scores_returns_empty_figure(self):
        """Confirm an empty scores_df short-circuits to an empty Figure."""

        fig = plot_consensus_distance_summary(pd.DataFrame(), pd.DataFrame())

        self.assertTrue(hasattr(fig, "to_html"))
        self.assertEqual(len(fig.data), 0)


if __name__ == "__main__":
    unittest.main()
