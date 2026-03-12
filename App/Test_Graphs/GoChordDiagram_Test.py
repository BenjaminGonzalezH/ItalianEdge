"""
GoChordDiagram_Test.py

Unit tests for GoChordDiagram module.

Tests
-----
- gene2terms conversion
- chord diagram generation
- HTML export
- filtering behavior
- input validation
"""

######### Libraries #########

import unittest
import tempfile
import os
import logging

from Graphs.GoChordDiagram import (
    plot_go_chord_html,
    GoChordOptions,
    _gene2terms_to_dataframe
)


class TestGoChordDiagram(unittest.TestCase):

    ##################################
    # Setup
    ##################################

    def setUp(self):

        logging.disable(logging.CRITICAL)

        self.gene2terms = {
            "TP53": ["GO:0008285", "GO:0006355"],
            "BRCA1": ["GO:0006281"],
            "RAD51": ["GO:0006281"]
        }

    ##################################
    # Conversion test
    ##################################

    def test_gene2terms_conversion(self):

        options = GoChordOptions()

        df = _gene2terms_to_dataframe(self.gene2terms, options)

        self.assertEqual(len(df), 4)
        self.assertIn("gene", df.columns)
        self.assertIn("go_term", df.columns)

    ##################################
    # Chord generation
    ##################################

    def test_plot_generation(self):

        chord = plot_go_chord_html(
            self.gene2terms,
            return_plot=True
        )

        self.assertIsNotNone(chord)

    ##################################
    # HTML export
    ##################################

    def test_html_export(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            path = os.path.join(tmpdir, "chord.html")

            plot_go_chord_html(
                self.gene2terms,
                save_html_to=path
            )

            self.assertTrue(os.path.exists(path))

    ##################################
    # Filtering test
    ##################################

    def test_term_frequency_filter(self):

        options = GoChordOptions(
            min_gene_frequency=2
        )

        df = _gene2terms_to_dataframe(
            self.gene2terms,
            options
        )

        # GO:0006281 appears twice
        self.assertTrue("GO:0006281" in df["go_term"].values)

    ##################################
    # Max terms per gene
    ##################################

    def test_max_terms_per_gene(self):

        options = GoChordOptions(
            max_terms_per_gene=1
        )

        df = _gene2terms_to_dataframe(
            self.gene2terms,
            options
        )

        counts = df.groupby("gene").size()

        self.assertTrue(all(counts <= 1))

    ##################################
    # Empty input validation
    ##################################

    def test_empty_input(self):

        with self.assertRaises(ValueError):

            plot_go_chord_html(
                {},
                return_plot=True
            )

    ##################################
    # Invalid type
    ##################################

    def test_invalid_input_type(self):

        with self.assertRaises(TypeError):

            plot_go_chord_html(
                ["not", "a", "dict"],
                return_plot=True
            )


##################################
# Run tests
##################################

if __name__ == "__main__":

    unittest.main()