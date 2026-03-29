"""
Unit tests for plurality voting ensemble clustering.

Goals
-----
• Validate input validation
• Validate label alignment
• Validate plurality voting consensus and confidence
• Validate Plotly confidence visualization
• Validate end-to-end plurality voting algorithm
"""

######### Libraries #########

import unittest
import tempfile
import numpy as np
from pathlib import Path

import plotly.graph_objects as go

from gclusters_characterization.clustering.plurality_voting import (
    PVOptions,
    _validate_solutions_matrix,
    _align_partition,
    _plurality_vote,
    plot_consensus_confidence,
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

        self.genes = ["AASD", "ASDSD", "QWEX", "DFS"]

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

    def test_plurality_vote_labels(self):
        """Plurality vote should produce correct consensus labels."""
        matrix = np.array([
            [1,1,2,2],
            [1,1,2,2],
            [1,1,2,2]
        ])

        labels, confidence = _plurality_vote(matrix)

        self.assertTrue(np.array_equal(labels, [1,1,2,2]))

    def test_plurality_vote_confidence(self):
        """Confidence values must be between 0 and 1."""
        matrix = np.array([
            [1,1,2,2],
            [1,1,2,2],
            [2,2,1,1]
        ])

        _, confidence = _plurality_vote(matrix)

        self.assertTrue(np.all(confidence >= 0))
        self.assertTrue(np.all(confidence <= 1))

    ############################
    # Plot confidence
    ############################

    def test_plot_confidence_return_fig(self):
        """Plot function should return Plotly figure."""
        confidence = np.array([1.0, 0.8, 0.6])

        fig = plot_consensus_confidence(
            confidence,
            self.genes,
            return_fig=True
        )

        self.assertTrue(isinstance(fig, go.Figure))

    def test_plot_confidence_html(self):
        """Plot function should return HTML."""
        confidence = np.array([1.0, 0.8, 0.6])

        html = plot_consensus_confidence(
            confidence,
            self.genes,
            return_html=True
        )

        self.assertTrue(isinstance(html, str))

    def test_plot_confidence_save(self):
        """Plot function should save HTML to file."""
        confidence = np.array([1.0, 0.8, 0.6])

        with tempfile.TemporaryDirectory() as tmp:

            path = Path(tmp) / "confidence.html"

            plot_consensus_confidence(
                confidence,
                self.genes,
                save_html_to=str(path)
            )

            self.assertTrue(path.exists())

    ############################
    # Main algorithm
    ############################

    def test_plurality_voting_basic(self):
        """Plurality voting should return consensus and confidence."""
        consensus, confidence = plurality_voting(self.solutions, self.genes)

        self.assertEqual(consensus.shape[0], 4)
        self.assertEqual(confidence.shape[0], 4)

    def test_plurality_voting_confidence_range(self):
        """Confidence must always be between 0 and 1."""
        _, confidence = plurality_voting(self.solutions, self.genes)

        self.assertTrue(np.all(confidence >= 0))
        self.assertTrue(np.all(confidence <= 1))

    def test_plurality_voting_with_plot(self):
        """Algorithm should generate HTML when plotting is enabled."""
        with tempfile.TemporaryDirectory() as tmp:

            path = Path(tmp) / "confidence.html"

            plurality_voting(
                self.solutions,
                self.genes,
                plot_confidence=True,
                save_plot_to=str(path)
            )

            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()