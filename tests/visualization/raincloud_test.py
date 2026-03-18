"""
raincloud_test.py

Unit tests for RainCloud visualization module.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd

from gclusters_characterization.visualization.raincloud import (
    plot_similarity_raincloud_html,
    RaincloudOptions
)


class TestRaincloud(unittest.TestCase):

    def setUp(self):
        """Create simple numeric dataset."""
        self.df = pd.DataFrame({
            "sim": [0.1, 0.2, 0.3, 0.4]
        })

    def test_basic_plot(self):
        """Plot should return a figure."""
        fig = plot_similarity_raincloud_html(
            self.df,
            "sim",
            return_fig=True
        )
        self.assertIsNotNone(fig)

    def test_return_html(self):
        """HTML output should be returned."""
        html = plot_similarity_raincloud_html(
            self.df,
            "sim",
            return_html=True
        )
        self.assertTrue(isinstance(html, str))

    def test_save_html(self):
        """HTML file should be created."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rain.html"

            plot_similarity_raincloud_html(
                self.df,
                "sim",
                save_html_to=path
            )

            self.assertTrue(path.exists())

    def test_invalid_dataframe(self):
        """Non-dataframe input should raise."""
        with self.assertRaises(TypeError):
            plot_similarity_raincloud_html([], "sim")

    def test_missing_column(self):
        """Missing column should raise."""
        with self.assertRaises(ValueError):
            plot_similarity_raincloud_html(self.df, "bad")

    def test_empty_values_after_nan(self):
        """All NaN should raise."""
        df = pd.DataFrame({"sim": [None, None]})

        with self.assertRaises(ValueError):
            plot_similarity_raincloud_html(df, "sim")

    def test_return_both(self):
        """Return both fig and html."""
        fig, html = plot_similarity_raincloud_html(
            self.df,
            "sim",
            return_fig=True,
            return_html=True
        )

        self.assertIsNotNone(fig)
        self.assertTrue(isinstance(html, str))


if __name__ == "__main__":
    unittest.main()