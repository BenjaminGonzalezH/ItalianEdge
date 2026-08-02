"""
go_hierarchical_network_test.py

Unit tests for GO hierarchical DAG visualization module.
"""

import unittest
from unittest.mock import patch
from pathlib import Path

from biocluster.visualization.go_hierarchical_network import (
    GoHierarchyOptions,
    _select_target_terms,
    plot_go_hierarchy_html,
)


class FakeTerm:
    def __init__(self, name, namespace, parents=None):
        self.name = name
        self.namespace = namespace
        self.parents = parents or []


def _make_bp_dag():
    return {
        "GO:1": FakeTerm("term1", "biological_process"),
        "GO:2": FakeTerm("term2", "biological_process"),
        "GO:3": FakeTerm("term3", "biological_process"),
    }


class TestSelectTargetTerms(unittest.TestCase):

    def setUp(self):
        self.go_dag = _make_bp_dag()
        self.term2genes = {
            "GO:1": ["g1", "g2"],
            "GO:2": ["g3"],
            "GO:3": ["g4"],
        }
        self.pvalues = {"GO:1": 0.01, "GO:2": 0.05}  # GO:3 has no p-value

    def test_restrict_to_enriched_filters_non_enriched_terms(self):
        options = GoHierarchyOptions(restrict_to_enriched=True)
        result = _select_target_terms(self.go_dag, self.term2genes, self.pvalues, options)
        self.assertIn("GO:1", result)
        self.assertIn("GO:2", result)
        self.assertNotIn("GO:3", result)

    def test_restrict_to_enriched_disabled_keeps_all_terms(self):
        options = GoHierarchyOptions(restrict_to_enriched=False)
        result = _select_target_terms(self.go_dag, self.term2genes, self.pvalues, options)
        self.assertIn("GO:3", result)

    def test_significance_threshold_filters_non_significant_terms(self):
        options = GoHierarchyOptions(restrict_to_enriched=False, significance_threshold=0.01)
        result = _select_target_terms(self.go_dag, self.term2genes, self.pvalues, options)
        self.assertIn("GO:1", result)
        self.assertNotIn("GO:2", result)
        self.assertNotIn("GO:3", result)

    def test_no_candidates_raises_value_error(self):
        options = GoHierarchyOptions(restrict_to_enriched=True, significance_threshold=0.0001)
        with self.assertRaises(ValueError):
            _select_target_terms(self.go_dag, self.term2genes, self.pvalues, options)


class TestPlotGoHierarchyHtml(unittest.TestCase):

    def setUp(self):
        self.term2genes = {
            "GO:1": ["g1", "g2"],
            "GO:2": ["g3"],
            "GO:3": ["g4"],
        }
        self.pvalues = {"GO:1": 0.01, "GO:2": 0.05}  # GO:3 has no p-value

    @patch("biocluster.visualization.go_hierarchical_network._ensure_obo", return_value=Path("go.obo"))
    @patch("biocluster.visualization.go_hierarchical_network.GODag", return_value=_make_bp_dag())
    def test_restrict_to_enriched_excludes_non_enriched_term(self, *_):
        fig = plot_go_hierarchy_html(
            self.term2genes,
            self.pvalues,
            options=GoHierarchyOptions(
                include_ancestors=False,
                restrict_to_enriched=True,
                download_obo_if_missing=False,
            ),
            save_html_to=None,
            return_fig=True,
        )

        rect_names = {s["name"] for s in fig.layout.shapes}
        self.assertIn("rect_GO:1", rect_names)
        self.assertIn("rect_GO:2", rect_names)
        self.assertNotIn("rect_GO:3", rect_names)

    @patch("biocluster.visualization.go_hierarchical_network._ensure_obo", return_value=Path("go.obo"))
    @patch("biocluster.visualization.go_hierarchical_network.GODag", return_value=_make_bp_dag())
    def test_restrict_to_enriched_disabled_includes_all_terms(self, *_):
        fig = plot_go_hierarchy_html(
            self.term2genes,
            self.pvalues,
            options=GoHierarchyOptions(
                include_ancestors=False,
                restrict_to_enriched=False,
                download_obo_if_missing=False,
            ),
            save_html_to=None,
            return_fig=True,
        )

        rect_names = {s["name"] for s in fig.layout.shapes}
        self.assertIn("rect_GO:3", rect_names)

    @patch("biocluster.visualization.go_hierarchical_network._ensure_obo", return_value=Path("go.obo"))
    @patch("biocluster.visualization.go_hierarchical_network.GODag", return_value=_make_bp_dag())
    def test_all_terms_filtered_out_raises_value_error(self, *_):
        with self.assertRaises(ValueError):
            plot_go_hierarchy_html(
                self.term2genes,
                self.pvalues,
                options=GoHierarchyOptions(
                    include_ancestors=False,
                    restrict_to_enriched=True,
                    significance_threshold=0.0001,
                    download_obo_if_missing=False,
                ),
                save_html_to=None,
                return_fig=True,
            )


if __name__ == "__main__":
    unittest.main()
