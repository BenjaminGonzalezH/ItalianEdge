"""
Unit tests for GoNetwork module (go3-based refactor).

Purpose:
- Validate correct graph construction.
- Validate filtering behavior.
- Validate HTML export without residual files.
- Ensure return modes (fig/html) work correctly.
- Avoid real GO downloads using mocking.
"""

import unittest
import tempfile
import os
import logging
from unittest.mock import patch

import numpy as np

from Graphs.GoNetwork import (
    plot_go_interaction_network_html,
    GoNetworkOptions,
)


class TestGoNetwork(unittest.TestCase):

    ############################
    # Setup
    ############################

    def setUp(self):
        # Disable logging noise
        logging.disable(logging.CRITICAL)

        self.gene2terms = {
            "G1": ["GO:0001", "GO:0002"],
            "G2": ["GO:0001"],
            "G3": ["GO:0002"],
        }

        self.term_pvalues = {
            "GO:0001": 0.01,
            "GO:0002": 0.05,
        }

        self.options = GoNetworkOptions(
            similarity_threshold=0.0,  # force edges
            min_genes_per_term=1,
            layout="spring",
            verbose=False,
        )

    ############################
    # Mock helpers
    ############################

    def _mock_go3(self):
        """
        Mock go3 behavior to avoid real OBO/GAF loading.
        """
        import Graphs.GoNetwork as GoNetwork
        return patch.multiple(
            GoNetwork.go3,
            load_go_terms=lambda x: None,
            load_gaf=lambda x: {},
            build_term_counter=lambda x: {},
            semantic_similarity=lambda a, b, m, c: 0.8 if a != b else 1.0
        )

    ############################
    # Core functionality tests
    ############################

    def test_returns_figure(self):
        with self._mock_go3():
            fig = plot_go_interaction_network_html(
                self.gene2terms,
                self.term_pvalues,
                gaf_path="dummy.gaf",
                obo_path="dummy.obo",
                options=self.options,
                return_fig=True
            )

        self.assertTrue(hasattr(fig, "to_html"))

    def test_returns_html(self):
        with self._mock_go3():
            html = plot_go_interaction_network_html(
                self.gene2terms,
                self.term_pvalues,
                gaf_path="dummy.gaf",
                obo_path="dummy.obo",
                options=self.options,
                return_html=True
            )

        self.assertIn("<html", html.lower())

    def test_filtering_min_genes(self):
        opts = GoNetworkOptions(
            similarity_threshold=0.0,
            min_genes_per_term=3,  # no term reaches 3 genes
            verbose=False
        )

        with self._mock_go3():
            with self.assertRaises(ValueError):
                plot_go_interaction_network_html(
                    self.gene2terms,
                    self.term_pvalues,
                    gaf_path="dummy.gaf",
                    obo_path="dummy.obo",
                    options=opts
                )

    ############################
    # Export tests
    ############################

    def test_html_export_temp_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "go_network.html")

            with self._mock_go3():
                plot_go_interaction_network_html(
                    self.gene2terms,
                    self.term_pvalues,
                    gaf_path="dummy.gaf",
                    obo_path="dummy.obo",
                    options=self.options,
                    save_html_to=filepath
                )

            self.assertTrue(os.path.exists(filepath))

        # Directory should be removed automatically
        self.assertFalse(os.path.exists(tmpdir))

    ############################
    # Validation tests
    ############################

    def test_invalid_gene2terms(self):
        with self.assertRaises(ValueError):
            plot_go_interaction_network_html(
                gene2terms={},
                term_pvalues=self.term_pvalues,
                gaf_path="dummy.gaf",
                obo_path="dummy.obo",
            )

    def test_invalid_term_pvalues(self):
        with self.assertRaises(ValueError):
            plot_go_interaction_network_html(
                gene2terms=self.gene2terms,
                term_pvalues=None,
                gaf_path="dummy.gaf",
                obo_path="dummy.obo",
            )


if __name__ == "__main__":
    unittest.main()