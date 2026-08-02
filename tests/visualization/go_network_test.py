"""
go_network_test.py

Unit tests for GO interaction network module.
"""

import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

from biocluster.visualization.go_network import (
    plot_go_interaction_network_html,
    GoNetworkOptions,
)


class TestGoNetwork(unittest.TestCase):

    def setUp(self):
        self.data = {"gene1": ["GO:1", "GO:2"]}
        self.pvals = {"GO:1": 0.01, "GO:2": 0.05}

    @patch("biocluster.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("biocluster.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_gaf", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_go_terms")
    def test_basic_execution(self, *_):
        fig = plot_go_interaction_network_html(
            self.data,
            self.pvals,
            "fake.gaf",
            "fake.obo",
            return_fig=True
        )
        self.assertIsNotNone(fig)

    @patch("biocluster.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("biocluster.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_gaf", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_go_terms")
    def test_html_output(self, *_):
        html = plot_go_interaction_network_html(
            self.data,
            self.pvals,
            "fake.gaf",
            "fake.obo",
            return_html=True
        )
        self.assertTrue(isinstance(html, str))

    @patch("biocluster.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("biocluster.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_gaf", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_go_terms")
    def test_save_html(self, *_):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.html"

            plot_go_interaction_network_html(
                self.data,
                self.pvals,
                "fake.gaf",
                "fake.obo",
                save_html_to=path
            )

            self.assertTrue(path.exists())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            plot_go_interaction_network_html({}, {}, "", "")

    def test_invalid_layout(self):
        with patch("biocluster.visualization.go_network.go3.load_go_terms"), \
             patch("biocluster.visualization.go_network.go3.load_gaf", return_value={}), \
             patch("biocluster.visualization.go_network.go3.build_term_counter", return_value={}), \
             patch("biocluster.visualization.go_network.go3.semantic_similarity", return_value=0.9):

            with self.assertRaises(ValueError):
                plot_go_interaction_network_html(
                    self.data,
                    self.pvals,
                    "fake.gaf",
                    "fake.obo",
                    options=GoNetworkOptions(layout="invalid"),
                    return_fig=True
                )

    ########################## restrict_to_enriched / significance_threshold ##########################

    @patch("biocluster.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("biocluster.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_gaf", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_go_terms")
    def test_restrict_to_enriched_filters_non_enriched_terms(self, *_):
        data = {"gene1": ["GO:1", "GO:2", "GO:3"]}
        pvals = {"GO:1": 0.01, "GO:2": 0.05}  # GO:3 has no p-value, i.e. not enriched

        fig = plot_go_interaction_network_html(
            data,
            pvals,
            "fake.gaf",
            "fake.obo",
            options=GoNetworkOptions(restrict_to_enriched=True),
            return_fig=True,
        )

        node_names = set(fig.data[1].text)
        self.assertNotIn("GO:3", node_names)
        self.assertIn("GO:1", node_names)
        self.assertIn("GO:2", node_names)

    @patch("biocluster.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("biocluster.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_gaf", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_go_terms")
    def test_restrict_to_enriched_disabled_keeps_all_terms(self, *_):
        data = {"gene1": ["GO:1", "GO:2", "GO:3"]}
        pvals = {"GO:1": 0.01, "GO:2": 0.05}

        fig = plot_go_interaction_network_html(
            data,
            pvals,
            "fake.gaf",
            "fake.obo",
            options=GoNetworkOptions(restrict_to_enriched=False),
            return_fig=True,
        )

        self.assertIn("GO:3", set(fig.data[1].text))

    @patch("biocluster.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("biocluster.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_gaf", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_go_terms")
    def test_significance_threshold_filters_non_significant_terms(self, *_):
        fig = plot_go_interaction_network_html(
            self.data,
            self.pvals,
            "fake.gaf",
            "fake.obo",
            options=GoNetworkOptions(significance_threshold=0.01),
            return_fig=True,
        )

        node_names = set(fig.data[1].text)
        self.assertIn("GO:1", node_names)
        self.assertNotIn("GO:2", node_names)

    @patch("biocluster.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("biocluster.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_gaf", return_value={})
    @patch("biocluster.visualization.go_network.go3.load_go_terms")
    def test_all_terms_filtered_out_raises_value_error(self, *_):
        with self.assertRaises(ValueError):
            plot_go_interaction_network_html(
                self.data,
                self.pvals,
                "fake.gaf",
                "fake.obo",
                options=GoNetworkOptions(significance_threshold=0.0001),
                return_fig=True,
            )


if __name__ == "__main__":
    unittest.main()