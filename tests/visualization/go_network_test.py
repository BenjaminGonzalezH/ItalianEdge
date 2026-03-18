"""
go_network_test.py

Unit tests for GO interaction network module.
"""

import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path

from gclusters_characterization.visualization.go_network import plot_go_interaction_network_html


class TestGoNetwork(unittest.TestCase):

    def setUp(self):
        self.data = {"gene1": ["GO:1", "GO:2"]}
        self.pvals = {"GO:1": 0.01, "GO:2": 0.05}

    @patch("gclusters_characterization.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("gclusters_characterization.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("gclusters_characterization.visualization.go_network.go3.load_gaf", return_value={})
    @patch("gclusters_characterization.visualization.go_network.go3.load_go_terms")
    def test_basic_execution(self, *_):
        fig = plot_go_interaction_network_html(
            self.data,
            self.pvals,
            "fake.gaf",
            "fake.obo",
            return_fig=True
        )
        self.assertIsNotNone(fig)

    @patch("gclusters_characterization.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("gclusters_characterization.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("gclusters_characterization.visualization.go_network.go3.load_gaf", return_value={})
    @patch("gclusters_characterization.visualization.go_network.go3.load_go_terms")
    def test_html_output(self, *_):
        html = plot_go_interaction_network_html(
            self.data,
            self.pvals,
            "fake.gaf",
            "fake.obo",
            return_html=True
        )
        self.assertTrue(isinstance(html, str))

    @patch("gclusters_characterization.visualization.go_network.go3.semantic_similarity", return_value=0.9)
    @patch("gclusters_characterization.visualization.go_network.go3.build_term_counter", return_value={})
    @patch("gclusters_characterization.visualization.go_network.go3.load_gaf", return_value={})
    @patch("gclusters_characterization.visualization.go_network.go3.load_go_terms")
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
        with patch("gclusters_characterization.visualization.go_network.go3.load_go_terms"), \
             patch("gclusters_characterization.visualization.go_network.go3.load_gaf", return_value={}), \
             patch("gclusters_characterization.visualization.go_network.go3.build_term_counter", return_value={}), \
             patch("gclusters_characterization.visualization.go_network.go3.semantic_similarity", return_value=0.9):

            from gclusters_characterization.visualization.go_network import GoNetworkOptions

            with self.assertRaises(ValueError):
                plot_go_interaction_network_html(
                    self.data,
                    self.pvals,
                    "fake.gaf",
                    "fake.obo",
                    options=GoNetworkOptions(layout="invalid"),
                    return_fig=True
                )


if __name__ == "__main__":
    unittest.main()