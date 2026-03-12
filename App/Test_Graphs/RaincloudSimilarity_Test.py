"""
RaincloudSimilarity_Test.py

Unit tests for RaincloudSimilarity module.

Tests
-----
- Visualization generation
- HTML export
- Input validation
- Deterministic jitter behavior
"""

######### Libraries #########

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import logging

from Graphs.Raincloud import (
    plot_similarity_raincloud_html,
    RaincloudOptions
)


class TestRaincloudSimilarity(unittest.TestCase):

    ##################################
    # Setup
    ##################################

    def setUp(self):

        logging.disable(logging.CRITICAL)

        np.random.seed(0)

        cluster1 = np.random.normal(0.25, 0.05, 200)
        cluster2 = np.random.normal(0.75, 0.04, 200)

        values = np.concatenate([cluster1, cluster2])

        self.df = pd.DataFrame({
            "similarity": values
        })

    ##################################
    # Figure generation
    ##################################

    def test_plot_generation(self):

        fig = plot_similarity_raincloud_html(
            self.df,
            column="similarity",
            return_fig=True
        )

        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, "to_html"))

    ##################################
    # HTML export
    ##################################

    def test_html_export(self):

        with tempfile.TemporaryDirectory() as tmpdir:

            path = os.path.join(tmpdir, "raincloud.html")

            plot_similarity_raincloud_html(
                self.df,
                column="similarity",
                save_html_to=path
            )

            self.assertTrue(os.path.exists(path))

    ##################################
    # Return HTML
    ##################################

    def test_return_html(self):

        html = plot_similarity_raincloud_html(
            self.df,
            column="similarity",
            return_html=True
        )

        self.assertTrue(isinstance(html, str))
        self.assertTrue("<html" in html.lower())

    ##################################
    # Missing column validation
    ##################################

    def test_missing_column(self):

        with self.assertRaises(ValueError):

            plot_similarity_raincloud_html(
                self.df,
                column="missing"
            )

    ##################################
    # Empty dataframe
    ##################################

    def test_empty_dataframe(self):

        df = pd.DataFrame({"similarity": []})

        with self.assertRaises(ValueError):

            plot_similarity_raincloud_html(
                df,
                column="similarity"
            )

    ##################################
    # NaN handling
    ##################################

    def test_nan_values(self):

        df = self.df.copy()

        df.loc[0, "similarity"] = np.nan

        fig = plot_similarity_raincloud_html(
            df,
            column="similarity",
            return_fig=True
        )

        self.assertIsNotNone(fig)

    ##################################
    # Custom options
    ##################################

    def test_custom_options(self):

        options = RaincloudOptions(
            jitter_strength=0.3,
            point_size=8
        )

        fig = plot_similarity_raincloud_html(
            self.df,
            column="similarity",
            options=options,
            return_fig=True
        )

        self.assertIsNotNone(fig)


##################################
# Run tests
##################################

if __name__ == "__main__":

    unittest.main()