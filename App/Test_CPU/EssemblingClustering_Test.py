# EssemblingClustering_Test.py

"""
Unit tests for EssemblingClustering module.

Validates:
- CSPA spectral clustering from coincidence matrix
- Global-stability plurality voting
- Interactive embedding plot
- HTML export behavior
- Input validation robustness
"""

import unittest
import tempfile
import os
import numpy as np
import logging

from ParetoInsight_CPU.EssemblingClustering import (
    cspa_spectral_from_coincidence,
    plurality_voting_stable,
    build_click_highlight_embedding_figure,
    plot_embedding_click_highlight,
    ensemble_cspa,
    ensemble_plurality_voting,
    CSPAOptions,
    PVOptions,
    EmbedOptions,
    ExportHTML,
)


class TestEssemblingClustering(unittest.TestCase):

    # ─────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────

    def setUp(self):
        logging.disable(logging.CRITICAL)

        self.genes = ["G1", "G2", "G3", "G4"]

        # Coincidence matrix for 2 clear clusters
        # Cluster A: 0-1
        # Cluster B: 2-3
        self.coincidence = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ])

        self.labels_matrix = np.array([
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])

    # ─────────────────────────────────────────────
    # CSPA Tests
    # ─────────────────────────────────────────────

    def test_cspa_basic(self):
        labels = cspa_spectral_from_coincidence(
            self.coincidence,
            CSPAOptions(n_clusters=2)
        )
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_cspa_invalid_matrix(self):
        bad = np.array([[1, 0, 1]])
        with self.assertRaises(ValueError):
            cspa_spectral_from_coincidence(
                bad,
                CSPAOptions(n_clusters=2)
            )

    def test_cspa_too_many_clusters(self):
        with self.assertRaises(ValueError):
            cspa_spectral_from_coincidence(
                self.coincidence,
                CSPAOptions(n_clusters=10)
            )

    # ─────────────────────────────────────────────
    # Plurality Voting Stable
    # ─────────────────────────────────────────────

    def test_plurality_voting_stable_basic(self):
        consensus = plurality_voting_stable(self.labels_matrix)
        self.assertEqual(len(consensus), 4)
        self.assertEqual(len(np.unique(consensus)), 2)

    def test_plurality_single_partition(self):
        single = np.array([[0, 1, 1, 0]])
        consensus = plurality_voting_stable(single)
        self.assertTrue(np.array_equal(consensus, np.array([0,1,1,0])))

    def test_plurality_invalid_input(self):
        with self.assertRaises(ValueError):
            plurality_voting_stable(np.array([]))

    # ─────────────────────────────────────────────
    # Embedding Plot
    # ─────────────────────────────────────────────

    def test_embedding_returns_figure(self):
        labels = np.array([0,0,1,1])
        fig = build_click_highlight_embedding_figure(
            self.coincidence,
            labels,
            self.genes
        )
        self.assertTrue(hasattr(fig, "to_html"))

    def test_embedding_html_contains_js(self):
        labels = np.array([0,0,1,1])
        html = plot_embedding_click_highlight(
            self.coincidence,
            labels,
            self.genes,
            return_html=True
        )
        self.assertIn("plotly_click", html)

    # ─────────────────────────────────────────────
    # Pipeline Tests
    # ─────────────────────────────────────────────

    def test_ensemble_cspa_pipeline(self):
        labels, fig, html = ensemble_cspa(
            self.coincidence,
            self.genes,
            cspa=CSPAOptions(n_clusters=2),
            return_fig=True,
            return_html=True
        )

        self.assertEqual(len(labels), 4)
        self.assertTrue(hasattr(fig, "to_html"))
        self.assertIn("<div", html)

    def test_ensemble_pv_pipeline(self):
        consensus, fig, html = ensemble_plurality_voting(
            self.labels_matrix,
            self.coincidence,
            self.genes,
            return_fig=True,
            return_html=True
        )

        self.assertEqual(len(consensus), 4)
        self.assertTrue(hasattr(fig, "to_html"))
        self.assertIn("<div", html)

    # ─────────────────────────────────────────────
    # HTML Export Safety
    # ─────────────────────────────────────────────

    def test_export_html_temp_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "ensemble.html")

            labels, _, _ = ensemble_cspa(
                self.coincidence,
                self.genes,
                cspa=CSPAOptions(n_clusters=2),
                save_html_to=filepath,
                export=ExportHTML(verbose=False)
            )

            self.assertTrue(os.path.exists(filepath))
            self.assertEqual(len(labels), 4)

        self.assertFalse(os.path.exists(tmpdir))


if __name__ == "__main__":
    unittest.main()