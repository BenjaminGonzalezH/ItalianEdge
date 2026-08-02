"""
Unit tests for the semantic_structural_discrepancy summary module.

Purpose of this file:
- Validate the element-wise Wang-vs-Jaccard discrepancy matrix.
- Validate the per-pair discrepancy profile and z-scoring.
- Validate outlier selection (z-score threshold, fixed top-K, both directions).
- Confirm the two-panel summary figure is generated correctly.
"""

######### Libraries #########
import unittest

import numpy as np

from biocluster.summary.semantic_structural_discrepancy import (
    DiscrepancyOptions,
    _hierarchical_order,
    compute_discrepancy_matrix,
    compute_pairwise_discrepancy_profile,
    identify_discrepant_solution_pairs,
    plot_discrepancy_summary,
)


class TestComputeDiscrepancyMatrix(unittest.TestCase):
    """Test suite validating the element-wise discrepancy matrix."""

    def setUp(self):
        """
        Solutions 0/1 are semantically similar but structurally different
        (Wang high, Jaccard low); solutions 2/3 are the opposite.
        """

        self.wang = np.array(
            [
                [1.0, 0.9, 0.2, 0.3],
                [0.9, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.8],
                [0.3, 0.2, 0.8, 1.0],
            ]
        )
        self.jaccard = np.array(
            [
                [1.0, 0.1, 0.2, 0.3],
                [0.1, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.1],
                [0.3, 0.2, 0.1, 1.0],
            ]
        )

    def test_discrepancy_values(self):
        """Confirm discrepancy = wang - jaccard element-wise, off-diagonal."""

        disc = compute_discrepancy_matrix(self.wang, self.jaccard)

        self.assertAlmostEqual(disc[0, 1], 0.8)
        self.assertAlmostEqual(disc[2, 3], 0.7)

    def test_diagonal_is_nan(self):
        """Confirm the diagonal (self-comparison) is masked as NaN."""

        disc = compute_discrepancy_matrix(self.wang, self.jaccard)

        self.assertTrue(np.isnan(np.diag(disc)).all())

    def test_shape_mismatch_raises(self):
        """Confirm mismatched matrix shapes raise ValueError."""

        with self.assertRaises(ValueError):
            compute_discrepancy_matrix(self.wang, self.jaccard[:3, :3])

    def test_non_square_raises(self):
        """Confirm a non-square input matrix raises ValueError."""

        with self.assertRaises(ValueError):
            compute_discrepancy_matrix(np.zeros((2, 3)), np.zeros((2, 3)))


class TestPairwiseDiscrepancyProfile(unittest.TestCase):
    """Test suite validating the per-pair discrepancy profile."""

    def setUp(self):
        self.wang = np.array(
            [
                [1.0, 0.9, 0.2, 0.3],
                [0.9, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.8],
                [0.3, 0.2, 0.8, 1.0],
            ]
        )
        self.jaccard = np.array(
            [
                [1.0, 0.1, 0.2, 0.3],
                [0.1, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.1],
                [0.3, 0.2, 0.1, 1.0],
            ]
        )
        self.labels = ["a", "b", "c", "d"]

    def test_profile_columns_and_row_count(self):
        """
        Confirm the profile has one row per unordered pair (n*(n-1)/2 = 6
        for n=4) and the expected columns.
        """

        profile = compute_pairwise_discrepancy_profile(self.wang, self.jaccard)

        self.assertEqual(len(profile), 6)
        self.assertListEqual(
            list(profile.columns),
            [
                "Solution 1",
                "Solution 2",
                "wang",
                "jaccard",
                "discrepancy",
                "score",
                "z_score",
            ],
        )
        self.assertTrue(profile["score"].is_monotonic_decreasing)

    def test_profile_values_match_source_matrices(self):
        """Confirm wang/jaccard/discrepancy columns match the source matrices for a known pair."""

        profile = compute_pairwise_discrepancy_profile(self.wang, self.jaccard)

        row = profile[(profile["Solution 1"] == 0) & (profile["Solution 2"] == 1)].iloc[
            0
        ]
        self.assertAlmostEqual(row["wang"], 0.9)
        self.assertAlmostEqual(row["jaccard"], 0.1)
        self.assertAlmostEqual(row["discrepancy"], 0.8)

    def test_profile_with_labels(self):
        """Confirm optional "label 1"/"label 2" columns are added and correctly aligned."""

        profile = compute_pairwise_discrepancy_profile(
            self.wang, self.jaccard, labels=self.labels
        )

        self.assertIn("label 1", profile.columns)
        self.assertIn("label 2", profile.columns)

        row = profile[(profile["Solution 1"] == 0) & (profile["Solution 2"] == 1)].iloc[
            0
        ]
        self.assertEqual(row["label 1"], "a")
        self.assertEqual(row["label 2"], "b")

    def test_wang_over_jaccard_direction_favors_high_wang_pairs(self):
        """
        Confirm the default direction (wang_over_jaccard) ranks the
        Wang-high/Jaccard-low pairs (0,1) and (2,3) at the top.
        """

        profile = compute_pairwise_discrepancy_profile(
            self.wang,
            self.jaccard,
            options=DiscrepancyOptions(direction="wang_over_jaccard"),
        )

        top_two = set(zip(profile.head(2)["Solution 1"], profile.head(2)["Solution 2"]))
        self.assertEqual(top_two, {(0, 1), (2, 3)})

    def test_jaccard_over_wang_direction_flips_ranking(self):
        """
        Confirm jaccard_over_wang inverts the sign, ranking the
        Wang-high/Jaccard-low pairs (0,1) and (2,3) at the BOTTOM.
        """

        profile = compute_pairwise_discrepancy_profile(
            self.wang,
            self.jaccard,
            options=DiscrepancyOptions(direction="jaccard_over_wang"),
        )

        bottom_two = set(
            zip(profile.tail(2)["Solution 1"], profile.tail(2)["Solution 2"])
        )
        self.assertEqual(bottom_two, {(0, 1), (2, 3)})

    def test_labels_length_mismatch_raises(self):
        """Confirm mismatched labels length raises ValueError."""

        with self.assertRaises(ValueError):
            compute_pairwise_discrepancy_profile(
                self.wang, self.jaccard, labels=["only_one"]
            )


class TestHierarchicalOrder(unittest.TestCase):
    """Test suite validating the hierarchical-clustering row/column order used for the heatmap."""

    def setUp(self):
        self.wang = np.array(
            [
                [1.0, 0.9, 0.2, 0.3],
                [0.9, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.8],
                [0.3, 0.2, 0.8, 1.0],
            ]
        )
        self.jaccard = np.array(
            [
                [1.0, 0.1, 0.2, 0.3],
                [0.1, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.1],
                [0.3, 0.2, 0.1, 1.0],
            ]
        )

    def test_order_is_a_valid_permutation(self):
        """Confirm the returned order visits every solution index exactly once."""

        disc = compute_discrepancy_matrix(self.wang, self.jaccard)

        order = _hierarchical_order(disc)

        self.assertListEqual(sorted(int(i) for i in order), [0, 1, 2, 3])

    def test_order_handles_nan_diagonal(self):
        """Confirm the NaN diagonal from compute_discrepancy_matrix doesn't break clustering."""

        disc = compute_discrepancy_matrix(self.wang, self.jaccard)
        self.assertTrue(np.isnan(np.diag(disc)).all())

        order = _hierarchical_order(disc)

        self.assertEqual(len(order), 4)


class TestIdentifyDiscrepantSolutionPairs(unittest.TestCase):
    """Test suite validating outlier-pair selection on top of the pairwise profile."""

    def setUp(self):
        self.wang = np.array(
            [
                [1.0, 0.9, 0.2, 0.3],
                [0.9, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.8],
                [0.3, 0.2, 0.8, 1.0],
            ]
        )
        self.jaccard = np.array(
            [
                [1.0, 0.1, 0.2, 0.3],
                [0.1, 1.0, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.1],
                [0.3, 0.2, 0.1, 1.0],
            ]
        )

    def test_z_threshold_selection(self):
        """Confirm z-score threshold flags exactly the expected outlier pairs."""

        pairs_df, outliers_df = identify_discrepant_solution_pairs(
            self.wang, self.jaccard, options=DiscrepancyOptions(z_threshold=0.5)
        )

        self.assertEqual(len(pairs_df), 6)
        flagged = set(zip(outliers_df["Solution 1"], outliers_df["Solution 2"]))
        self.assertEqual(flagged, {(0, 1), (2, 3)})

    def test_top_k_selection(self):
        """Confirm top_k overrides the z-score threshold."""

        _, outliers_df = identify_discrepant_solution_pairs(
            self.wang, self.jaccard, options=DiscrepancyOptions(top_k=2)
        )

        self.assertEqual(len(outliers_df), 2)

    def test_invalid_top_k_raises(self):
        """Confirm a non-positive top_k raises ValueError."""

        with self.assertRaises(ValueError):
            identify_discrepant_solution_pairs(
                self.wang, self.jaccard, options=DiscrepancyOptions(top_k=0)
            )

    ########################## Figure Tests ##########################

    def test_plot_returns_figure_with_traces(self):
        """Confirm the two-panel summary figure (scatter + heatmap) is generated with traces."""

        _, outliers_df = identify_discrepant_solution_pairs(
            self.wang, self.jaccard, options=DiscrepancyOptions(z_threshold=0.5)
        )

        fig = plot_discrepancy_summary(self.wang, self.jaccard, outliers_df)

        self.assertTrue(hasattr(fig, "to_html"))
        trace_types = [type(t).__name__ for t in fig.data]
        self.assertIn("Scatter", trace_types)
        self.assertIn("Heatmap", trace_types)
        self.assertNotIn("Bar", trace_types)

    def test_plot_with_no_outliers(self):
        """Confirm the figure still renders when no pairs are flagged."""

        _, outliers_df = identify_discrepant_solution_pairs(
            self.wang, self.jaccard, options=DiscrepancyOptions(z_threshold=100.0)
        )

        fig = plot_discrepancy_summary(self.wang, self.jaccard, outliers_df)

        self.assertTrue(outliers_df.empty)
        self.assertTrue(hasattr(fig, "to_html"))

    def test_plot_reorders_heatmap_by_default(self):
        """
        Confirm the heatmap panel (reorder=True, the default) reorders rows/
        columns via hierarchical clustering while keeping the original
        solution indices as tick labels (as a permutation of 0..n-1).
        """

        _, outliers_df = identify_discrepant_solution_pairs(
            self.wang, self.jaccard, options=DiscrepancyOptions(z_threshold=0.5)
        )

        fig = plot_discrepancy_summary(self.wang, self.jaccard, outliers_df)

        heatmap = next(t for t in fig.data if type(t).__name__ == "Heatmap")
        self.assertListEqual(sorted(int(v) for v in heatmap.x), [0, 1, 2, 3])
        self.assertListEqual(list(heatmap.x), list(heatmap.y))
        self.assertIn("(ordenado)", fig.layout.annotations[1].text)

    def test_plot_reorder_false_keeps_natural_order(self):
        """Confirm reorder=False keeps the heatmap in natural solution order."""

        _, outliers_df = identify_discrepant_solution_pairs(
            self.wang, self.jaccard, options=DiscrepancyOptions(z_threshold=0.5)
        )

        fig = plot_discrepancy_summary(
            self.wang, self.jaccard, outliers_df, reorder=False
        )

        heatmap = next(t for t in fig.data if type(t).__name__ == "Heatmap")
        self.assertListEqual(list(heatmap.x), ["0", "1", "2", "3"])
        self.assertNotIn("(ordenado)", fig.layout.annotations[1].text)

    def test_plot_reorder_skipped_below_three_solutions(self):
        """Confirm reordering is skipped (natural order used) when n < 3, even with reorder=True."""

        wang = np.array([[1.0, 0.5], [0.5, 1.0]])
        jaccard = np.array([[1.0, 0.2], [0.2, 1.0]])
        _, outliers_df = identify_discrepant_solution_pairs(
            wang, jaccard, options=DiscrepancyOptions(z_threshold=100.0)
        )

        fig = plot_discrepancy_summary(wang, jaccard, outliers_df, reorder=True)

        heatmap = next(t for t in fig.data if type(t).__name__ == "Heatmap")
        self.assertListEqual(list(heatmap.x), ["0", "1"])


if __name__ == "__main__":
    unittest.main()
