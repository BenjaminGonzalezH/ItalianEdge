"""
Unit tests for the gene_overlap summary module.

Purpose of this file:
- Validate shared/disjoint gene-set computation for cluster pairs.
- Validate gene frequency counting and the automatic knee/elbow cutoff.
- Validate the end-to-end gene summarization pipeline.
- Validate input validation logic and figure generation.
"""

######### Libraries #########
import unittest
import warnings
import numpy as np
import pandas as pd

from gclusters_characterization.summary.gene_overlap import (
    compute_gene_overlap_dataframe,
    compute_gene_frequencies,
    compute_frequency_cutoff,
    plot_frequency_cutoff,
    summarize_genes,
    GeneSummaryResult,
)


class TestComputeGeneOverlapDataframe(unittest.TestCase):
    """Test suite validating shared/disjoint gene-set extraction per cluster pair."""

    def setUp(self):
        """
        Two solutions with a partial overlap between their clusters:
        Solution 0, cluster 0 -> {G1, G2, G3}
        Solution 1, cluster 1 -> {G3, G4, G5}
        Shared: {G3}. Disjoint (symmetric difference): {G1, G2, G4, G5}.
        """

        self.solution_cluster_matrix = [
            [{"G1", "G2", "G3"}, {"G4", "G5"}],
            [{"G1", "G2"}, {"G3", "G4", "G5"}],
        ]
        self.pairs_df = pd.DataFrame({
            "Solution 1": [0],
            "Cluster 1": [0],
            "Solution 2": [1],
            "Cluster 2": [1],
        })

    def test_shared_mode(self):
        """Confirm shared mode returns the intersection of gene sets."""

        result = compute_gene_overlap_dataframe(self.pairs_df, self.solution_cluster_matrix, mode="shared")

        self.assertIn("Shared Genes", result.columns)
        self.assertEqual(result.loc[0, "Shared Genes"], {"G3"})

    def test_disjoint_mode(self):
        """Confirm disjoint mode returns the symmetric difference of gene sets."""

        result = compute_gene_overlap_dataframe(self.pairs_df, self.solution_cluster_matrix, mode="disjoint")

        self.assertIn("Disjoint Genes", result.columns)
        self.assertEqual(result.loc[0, "Disjoint Genes"], {"G1", "G2", "G4", "G5"})

    def test_missing_required_columns_raises(self):
        """Confirm a DataFrame missing required columns raises ValueError."""

        with self.assertRaises(ValueError):
            compute_gene_overlap_dataframe(pd.DataFrame({"a": [1]}), self.solution_cluster_matrix)

    def test_invalid_mode_raises(self):
        """Confirm an unsupported mode raises ValueError."""

        with self.assertRaises(ValueError):
            compute_gene_overlap_dataframe(self.pairs_df, self.solution_cluster_matrix, mode="bad")


class TestGeneFrequencies(unittest.TestCase):
    """Test suite validating gene frequency counting."""

    def test_compute_gene_frequencies_counts_and_sorts(self):
        """Confirm frequencies are counted correctly and sorted descending."""

        df = pd.DataFrame({
            "Shared Genes": [{"G1", "G2"}, {"G1"}, {"G1", "G3"}],
        })

        freq = compute_gene_frequencies(df, "Shared Genes")

        self.assertEqual(freq.loc[freq["Gene"] == "G1", "Frequency"].iloc[0], 3)
        self.assertTrue(freq["Frequency"].is_monotonic_decreasing)

    def test_compute_gene_frequencies_empty_input(self):
        """Confirm an empty/degenerate gene column returns an empty frequency table."""

        df = pd.DataFrame({"Shared Genes": [set(), set()]})

        freq = compute_gene_frequencies(df, "Shared Genes")

        self.assertTrue(freq.empty)
        self.assertListEqual(list(freq.columns), ["Gene", "Frequency"])


class TestFrequencyCutoff(unittest.TestCase):
    """Test suite validating the automatic (knee/elbow) frequency cutoff."""

    def test_cutoff_on_bimodal_distribution(self):
        """
        Confirm the knee/elbow method separates a clear low-frequency group
        from a clear high-frequency group.
        """

        freq_df = pd.DataFrame({
            "Gene": [f"low_{i}" for i in range(10)] + [f"high_{i}" for i in range(10)],
            "Frequency": [1] * 10 + [10] * 10,
        })

        cutoff = compute_frequency_cutoff(freq_df)

        self.assertGreater(cutoff, 1)
        self.assertLessEqual(cutoff, 10)

    def test_cutoff_empty_dataframe_returns_zero(self):
        """Confirm an empty frequency table returns a cutoff of 0."""

        self.assertEqual(compute_frequency_cutoff(pd.DataFrame(columns=["Gene", "Frequency"])), 0)

    def test_cutoff_constant_frequency_returns_that_value(self):
        """Confirm a frequency table with max frequency <= 1 short-circuits sensibly."""

        freq_df = pd.DataFrame({"Gene": ["G1", "G2"], "Frequency": [1, 1]})

        self.assertEqual(compute_frequency_cutoff(freq_df), 1)

    def test_unsupported_method_raises(self):
        """Confirm an unimplemented cutoff method raises ValueError."""

        freq_df = pd.DataFrame({"Gene": ["G1"], "Frequency": [1]})

        with self.assertRaises(ValueError):
            compute_frequency_cutoff(freq_df, method="otsu")

    ########################## Figure Tests ##########################

    def test_plot_frequency_cutoff_returns_figure(self):
        """Confirm the cutoff histogram is generated with at least one trace."""

        freq_df = pd.DataFrame({"Gene": ["G1", "G2", "G3"], "Frequency": [1, 2, 3]})

        fig = plot_frequency_cutoff(freq_df, cutoff=2)

        self.assertTrue(hasattr(fig, "to_html"))
        self.assertGreater(len(fig.data), 0)

    def test_plot_frequency_cutoff_empty_returns_empty_figure(self):
        """Confirm an empty frequency table short-circuits to an empty Figure."""

        fig = plot_frequency_cutoff(pd.DataFrame(columns=["Gene", "Frequency"]), cutoff=1)

        self.assertTrue(hasattr(fig, "to_html"))
        self.assertEqual(len(fig.data), 0)


class TestSummarizeGenes(unittest.TestCase):
    """Test suite validating the end-to-end gene summarization pipeline."""

    def setUp(self):
        """
        Reusable overlap table (Shared Genes) plus a matching gene-gene
        similarity matrix covering all genes referenced in the sets.
        """

        self.overlap_df = pd.DataFrame({
            "Shared Genes": [{"G1", "G2"}, {"G1"}, {"G1", "G3"}],
        })
        self.gene_ids = ["G1", "G2", "G3"]
        self.similarity_matrix = np.eye(3)

    def test_summarize_genes_basic(self):
        """
        Confirm the pipeline selects genes at/above the cutoff and returns
        a similarity submatrix restricted to those genes.
        """

        result = summarize_genes(
            self.overlap_df, "Shared Genes", self.gene_ids, self.similarity_matrix,
            min_gene_frequency=1,
        )

        self.assertIsInstance(result, GeneSummaryResult)
        self.assertIn("G1", result.selected_genes)
        self.assertEqual(result.cutoff_used, 1)
        self.assertFalse(result.similarity_submatrix.empty)
        self.assertFalse(result.cooccurrence_df.empty)

    def test_summarize_genes_auto_cutoff(self):
        """Confirm min_gene_frequency=None triggers the automatic Otsu cutoff."""

        result = summarize_genes(
            self.overlap_df, "Shared Genes", self.gene_ids, self.similarity_matrix,
        )

        self.assertIsInstance(result.cutoff_used, (int, np.integer))

    def test_summarize_genes_max_genes_cap(self):
        """Confirm max_genes caps the number of selected genes."""

        result = summarize_genes(
            self.overlap_df, "Shared Genes", self.gene_ids, self.similarity_matrix,
            min_gene_frequency=1, max_genes=1,
        )

        self.assertEqual(len(result.selected_genes), 1)

    def test_summarize_genes_missing_column_raises(self):
        """Confirm a missing gene_column raises ValueError."""

        with self.assertRaises(ValueError):
            summarize_genes(self.overlap_df, "Nonexistent Column", self.gene_ids, self.similarity_matrix)

    def test_summarize_genes_non_square_matrix_raises(self):
        """Confirm a non-square similarity matrix raises ValueError."""

        with self.assertRaises(ValueError):
            summarize_genes(self.overlap_df, "Shared Genes", self.gene_ids, np.zeros((2, 3)))

    def test_summarize_genes_matrix_size_mismatch_raises(self):
        """Confirm a similarity matrix size not matching gene_ids raises ValueError."""

        with self.assertRaises(ValueError):
            summarize_genes(self.overlap_df, "Shared Genes", ["G1"], self.similarity_matrix)

    def test_summarize_genes_empty_gene_sets_returns_empty_result(self):
        """Confirm an overlap table with only empty sets returns an empty result."""

        empty_df = pd.DataFrame({"Shared Genes": [set(), set()]})

        result = summarize_genes(empty_df, "Shared Genes", self.gene_ids, self.similarity_matrix)

        self.assertEqual(result.selected_genes, [])
        self.assertTrue(result.similarity_submatrix.empty)

    def test_summarize_genes_unreachable_cutoff_returns_partial_empty_result(self):
        """
        Confirm a cutoff higher than any observed frequency yields no
        selected genes, while still returning the full frequency table.
        """

        result = summarize_genes(
            self.overlap_df, "Shared Genes", self.gene_ids, self.similarity_matrix,
            min_gene_frequency=100,
        )

        self.assertEqual(result.selected_genes, [])
        self.assertFalse(result.frequency_df.empty)
        self.assertEqual(result.cutoff_used, 100)

    def test_summarize_genes_missing_gene_warns_without_stray_output(self):
        """
        Confirm that a selected gene absent from gene_ids/similarity matrix
        triggers exactly one captured UserWarning (not printed to stderr)
        and the submatrix is restricted to the genes actually found.
        """

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")

            result = summarize_genes(
                self.overlap_df, "Shared Genes", ["G1"], np.eye(1),
                min_gene_frequency=1,
            )

            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            self.assertEqual(len(user_warnings), 1)
            self.assertIn("not found in similarity matrix", str(user_warnings[0].message))

        self.assertListEqual(list(result.similarity_submatrix.index), ["G1"])


if __name__ == "__main__":
    unittest.main()
