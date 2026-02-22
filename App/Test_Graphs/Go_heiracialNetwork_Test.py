"""
Unit tests for GoHierarchicalNetwork module.

Goals:
- Validate hierarchical DAG building logic.
- Validate ontology filtering.
- Validate return modes (fig/html).
- Ensure no residual files remain.
- Avoid real OBO download (mocked).
"""

import unittest
import tempfile
import os
import logging
from unittest.mock import patch, MagicMock

import networkx as nx

from Graphs.Go_heiracialNetwork import (
    plot_go_hierarchy_html,
    GoHierarchyOptions,
)

# ──────────────────────────────────────────────────────────────
# Fake GO DAG for testing (avoid real goatools + OBO file)
# ──────────────────────────────────────────────────────────────

class FakeGOTerm:
    def __init__(self, go_id, name, namespace, parents=None):
        self.id = go_id
        self.name = name
        self.namespace = namespace
        self.parents = parents or []


class FakeGODag(dict):
    def __init__(self):
        super().__init__()

        # Simple BP hierarchy:
        # GO:0001 (root)
        #   ├── GO:0002
        #   │     └── GO:0004
        #   └── GO:0003

        root = FakeGOTerm("GO:0001", "Root BP", "biological_process", [])
        t2 = FakeGOTerm("GO:0002", "Child A", "biological_process", [root])
        t3 = FakeGOTerm("GO:0003", "Child B", "biological_process", [root])
        t4 = FakeGOTerm("GO:0004", "Grandchild", "biological_process", [t2])

        self["GO:0001"] = root
        self["GO:0002"] = t2
        self["GO:0003"] = t3
        self["GO:0004"] = t4

    def __contains__(self, item):
        return dict.__contains__(self, item)


# ──────────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────────

class TestGoHierarchicalNetwork(unittest.TestCase):

    def setUp(self):
        logging.disable(logging.CRITICAL)

        # gene -> terms format
        self.gene2terms = {
            "GeneA": ["GO:0002", "GO:0004"],
            "GeneB": ["GO:0003"],
        }

        self.term_pvalues = {
            "GO:0002": 0.001,
            "GO:0003": 0.02,
            "GO:0004": 0.0001,
        }

        self.options = GoHierarchyOptions(
            ontology="BP",
            max_terms=10,
            include_ancestors=True,
            verbose=False,
            download_obo_if_missing=False,
        )

    # ──────────────────────────────────────────────────────────
    # Core functionality tests
    # ──────────────────────────────────────────────────────────

    @patch("Graphs.Go_heiracialNetwork._ensure_obo")
    @patch("Graphs.Go_heiracialNetwork.GODag")
    def test_basic_figure_generation(self, mock_godag, mock_ensure):
        mock_ensure.return_value = "fake.obo"
        mock_godag.return_value = FakeGODag()

        fig = plot_go_hierarchy_html(
            self.gene2terms,
            self.term_pvalues,
            options=self.options,
            save_html_to=None,
            return_fig=True,
        )

        self.assertTrue(hasattr(fig, "to_html"))

    @patch("Graphs.Go_heiracialNetwork._ensure_obo")
    @patch("Graphs.Go_heiracialNetwork.GODag")
    def test_return_html(self, mock_godag, mock_ensure):
        mock_ensure.return_value = "fake.obo"
        mock_godag.return_value = FakeGODag()

        html = plot_go_hierarchy_html(
            self.gene2terms,
            self.term_pvalues,
            options=self.options,
            save_html_to=None,
            return_html=True,
        )

        self.assertIn("<div", html)
        self.assertIn("GO Hierarchical DAG", html)

    @patch("Graphs.Go_heiracialNetwork._ensure_obo")
    @patch("Graphs.Go_heiracialNetwork.GODag")
    def test_return_fig_and_html(self, mock_godag, mock_ensure):
        mock_ensure.return_value = "fake.obo"
        mock_godag.return_value = FakeGODag()

        fig, html = plot_go_hierarchy_html(
            self.gene2terms,
            self.term_pvalues,
            options=self.options,
            save_html_to=None,
            return_fig=True,
            return_html=True,
        )

        self.assertTrue(hasattr(fig, "to_html"))
        self.assertIn("<div", html)

    # ──────────────────────────────────────────────────────────
    # File export test (no residuals)
    # ──────────────────────────────────────────────────────────

    @patch("Graphs.Go_heiracialNetwork._ensure_obo")
    @patch("Graphs.Go_heiracialNetwork.GODag")
    def test_html_file_export(self, mock_godag, mock_ensure):
        mock_ensure.return_value = "fake.obo"
        mock_godag.return_value = FakeGODag()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "go_test.html")

            plot_go_hierarchy_html(
                self.gene2terms,
                self.term_pvalues,
                options=self.options,
                save_html_to=filepath,
            )

            self.assertTrue(os.path.exists(filepath))

        # Ensure cleanup
        self.assertFalse(os.path.exists(tmpdir))

    # ──────────────────────────────────────────────────────────
    # Error handling
    # ──────────────────────────────────────────────────────────

    def test_invalid_input_dict(self):
        with self.assertRaises(ValueError):
            plot_go_hierarchy_html(
                {},
                self.term_pvalues,
                options=self.options,
                save_html_to=None,
            )

    def test_invalid_pvalues_type(self):
        with self.assertRaises(TypeError):
            plot_go_hierarchy_html(
                self.gene2terms,
                "invalid",
                options=self.options,
                save_html_to=None,
            )

    @patch("Graphs.Go_heiracialNetwork._ensure_obo")
    @patch("Graphs.Go_heiracialNetwork.GODag")
    def test_no_terms_after_filter(self, mock_godag, mock_ensure):
        mock_ensure.return_value = "fake.obo"
        mock_godag.return_value = FakeGODag()

        options = GoHierarchyOptions(
            ontology="MF",  # none match (fake DAG only BP)
            download_obo_if_missing=False,
            verbose=False,
        )

        with self.assertRaises(ValueError):
            plot_go_hierarchy_html(
                self.gene2terms,
                self.term_pvalues,
                options=options,
                save_html_to=None,
            )


if __name__ == "__main__":
    unittest.main()