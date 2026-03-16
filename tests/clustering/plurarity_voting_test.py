"""
Unit tests for plurality voting ensemble clustering.

Goals
-----
• Validate input validation
• Validate similarity computation
• Validate partition stability
• Validate label alignment
• Validate plurality voting consensus
• Validate Plotly stability visualization
• Validate end-to-end plurality voting algorithm
"""

######### Libraries #########

import unittest
import tempfile
import numpy as np
from pathlib import Path

import plotly.graph_objects as go

from gclusters_characterization.clustering.plurarity_voting import (
    PVOptions,
    _validate_solutions_matrix,
    _partition_similarity,
    compute_partition_stability,
    _align_partition,
    _plurality_vote,
    plot_partition_stability,
    plurality_voting,
)


class TestPluralityVoting(unittest.TestCase):

    ############################
    # Setup
    ############################

    def setUp(self):
        """
        Create small deterministic clustering solutions.
        """

        self.solutions = np.array([
            [1,1,2,2],
            [1,1,2,2],
            [2,2,1,1]
        ])

    ############################
    # Validation
    ############################

    def test_validate_matrix_valid(self):
        """Valid matrix should pass validation."""
        _validate_solutions_matrix(self.solutions)

    def test_validate_matrix_type(self):
        """Non-numpy input should raise TypeError."""
        with self.assertRaises(TypeError):
            _validate_solutions_matrix([[1,2],[3,4]])

    def test_validate_matrix_dimension(self):
        """1D matrix must raise ValueError."""
        with self.assertRaises(ValueError):
            _validate_solutions_matrix(np.array([1,2,3]))

    ############################
    # Similarity
    ############################

    def test_partition_similarity_ari(self):
        """ARI similarity must return valid value."""
        s = _partition_similarity(
            self.solutions[0],
            self.solutions[1],
            "ari"
        )

        self.assertTrue(-1 <= s <= 1)

    def test_partition_similarity_rand(self):
        """Rand similarity must return value between 0 and 1."""
        s = _partition_similarity(
            self.solutions[0],
            self.solutions[1],
            "rand"
        )

        self.assertTrue(0 <= s <= 1)

    def test_partition_similarity_invalid_metric(self):
        """Invalid metric should raise ValueError."""
        with self.assertRaises(ValueError):
            _partition_similarity(self.solutions[0], self.solutions[1], "bad")

    ############################
    # Stability
    ############################

    def test_compute_partition_stability_shape(self):
        """Stability vector should match number of solutions."""
        stability = compute_partition_stability(
            self.solutions,
            "ari"
        )

        self.assertEqual(stability.shape[0], 3)

    ############################
    # Alignment
    ############################

    def test_align_partition(self):
        """Aligned partition should preserve cluster structure."""
        reference = np.array([1,1,2,2])
        target = np.array([2,2,1,1])

        aligned = _align_partition(reference, target)

        self.assertTrue(np.array_equal(aligned, reference))

    ############################
    # Plurality vote
    ############################

    def test_plurality_vote(self):
        """Plurality vote should produce consensus labels."""
        matrix = np.array([
            [1,1,2,2],
            [1,1,2,2],
            [1,1,2,2]
        ])

        consensus = _plurality_vote(matrix)

        self.assertTrue(np.array_equal(consensus, [1,1,2,2]))

    ############################
    # Plot stability
    ############################

    def test_plot_partition_stability_return_fig(self):
        """Plot function should return Plotly figure."""
        stability = np.array([0.9,0.8,0.7])

        fig = plot_partition_stability(
            stability,
            reference_index=0,
            return_fig=True
        )

        self.assertTrue(isinstance(fig, go.Figure))

    def test_plot_partition_stability_html(self):
        """Plot function should return HTML."""
        stability = np.array([0.9,0.8,0.7])

        html = plot_partition_stability(
            stability,
            reference_index=1,
            return_html=True
        )

        self.assertTrue(isinstance(html, str))

    def test_plot_partition_stability_save(self):
        """Plot function should save HTML to file."""
        stability = np.array([0.9,0.8,0.7])

        with tempfile.TemporaryDirectory() as tmp:

            path = Path(tmp) / "stability.html"

            plot_partition_stability(
                stability,
                reference_index=0,
                save_html_to=str(path)
            )

            self.assertTrue(path.exists())

    ############################
    # Main algorithm
    ############################

    def test_plurality_voting_basic(self):
        """Plurality voting should return consensus and stability."""
        consensus, ref_idx, stability = plurality_voting(self.solutions)

        self.assertEqual(consensus.shape[0], 4)
        self.assertTrue(0 <= ref_idx < 3)
        self.assertEqual(stability.shape[0], 3)

    def test_plurality_voting_invalid_metric(self):
        """Invalid similarity metric should raise ValueError."""
        options = PVOptions(similarity_metric="bad")

        with self.assertRaises(ValueError):
            plurality_voting(
                self.solutions,
                options=options
            )

    def test_plurality_voting_with_plot(self):
        """Algorithm should run when stability plotting is enabled."""
        with tempfile.TemporaryDirectory() as tmp:

            path = Path(tmp) / "plot.html"

            consensus, ref_idx, stability = plurality_voting(
                self.solutions,
                plot_stability=True,
                save_plot_to=str(path)
            )

            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()