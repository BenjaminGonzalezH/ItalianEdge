"""
test_go_utils.py

Unit tests for go_utils module.

Coverage goals:
- File download logic (mocked)
- GAF and gene_info handling
- Mapping utilities
- Error handling and edge cases
"""

import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import gzip
import shutil
from pathlib import Path

import pandas as pd

from gclusters_characterization.go.go_utils import (
    download_file,
    gunzip_file,
    ensure_gaf_file,
    ensure_gene_info_file,
    build_gaf_gene_mappings,
    map_genes_using_gaf,
    load_gene_info,
    entrez_to_symbol_ncbi,
    DownloadOptions,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def create_dummy_gz(content: bytes, path: str):
    with gzip.open(path, "wb") as f:
        f.write(content)


def create_dummy_gaf(path: str):
    content = """!gaf-version: 2.2
DB\tGENE1\tSYMBOL1
DB\tGENE2\tSYMBOL2
"""
    with open(path, "w") as f:
        f.write(content)


def create_dummy_gene_info(path: str):
    df = pd.DataFrame({
        "GeneID": ["1", "2"],
        "Symbol": ["A", "B"]
    })
    df.to_csv(path, sep="\t", index=False)


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

class TestFileOperations(unittest.TestCase):

    @patch("gclusters_characterization.go.go_utils.requests.get")
    def test_download_file(self, mock_get):
        """Download should write file correctly."""

        mock_response = MagicMock()
        mock_response.iter_content = lambda chunk_size: [b"data"]
        mock_response.raise_for_status = lambda: None

        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(tmp, "file.txt")

            path = download_file("http://fake-url", dest)

            self.assertTrue(os.path.exists(path))

            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"data")

    def test_gunzip_file(self):
        """Gunzip should extract file correctly."""

        with tempfile.TemporaryDirectory() as tmp:
            gz_path = os.path.join(tmp, "file.gz")
            out_path = os.path.join(tmp, "file.txt")

            create_dummy_gz(b"hello", gz_path)

            gunzip_file(gz_path, out_path)

            with open(out_path, "rb") as f:
                self.assertEqual(f.read(), b"hello")


class TestEnsureFiles(unittest.TestCase):

    @patch("gclusters_characterization.go.go_utils.download_file")
    @patch("gclusters_characterization.go.go_utils.gunzip_file")
    def test_ensure_gaf_download(self, mock_gunzip, mock_download):
        """Ensure GAF downloads when missing."""

        with tempfile.TemporaryDirectory() as tmp:

            tmp = Path(tmp)

            # Crear archivo gz real
            fake_gz = tmp / "file.gz"
            fake_gz.write_bytes(b"dummy")

            mock_download.return_value = fake_gz

            # Simular gunzip → crear archivo final .gaf
            def fake_gunzip(src, dest):
                Path(dest).write_text("dummy gaf")

            mock_gunzip.side_effect = fake_gunzip

            result = ensure_gaf_file(
                "goa_human",
                out_dir=tmp,
                download_if_missing=True
            )

            self.assertTrue(result.exists())

    def test_ensure_gaf_no_download(self):
        """Should fail if file missing and download disabled."""

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                ensure_gaf_file(
                    "goa_human",
                    out_dir=tmp,
                    download_if_missing=False
                )

    def test_invalid_species(self):
        """Invalid species should raise."""

        with self.assertRaises(KeyError):
            ensure_gaf_file("invalid_species")

    @patch("gclusters_characterization.go.go_utils.download_file")
    @patch("gclusters_characterization.go.go_utils.gunzip_file")
    def test_ensure_gene_info_download(self, mock_gunzip, mock_download):
        """Ensure gene_info downloads."""

        with tempfile.TemporaryDirectory() as tmp:

            tmp = Path(tmp)

            fake_gz = tmp / "file.gz"
            fake_gz.write_bytes(b"dummy")

            mock_download.return_value = fake_gz

            def fake_gunzip(src, dest):
                Path(dest).write_text("gene_info")

            mock_gunzip.side_effect = fake_gunzip

            result = ensure_gene_info_file(
                "goa_human",
                out_dir=tmp,
                download_if_missing=True
            )

            self.assertTrue(result.exists())


class TestGAFMapping(unittest.TestCase):

    def test_build_gaf_mappings(self):
        """Mappings should be created correctly."""

        with tempfile.TemporaryDirectory() as tmp:
            gaf = os.path.join(tmp, "test.gaf")
            create_dummy_gaf(gaf)

            id2sym, sym2id = build_gaf_gene_mappings(gaf)

            self.assertEqual(id2sym["GENE1"], "SYMBOL1")
            self.assertEqual(sym2id["SYMBOL2"], "GENE2")

    def test_map_genes_symbol(self):
        """Mapping to symbols should work."""

        with tempfile.TemporaryDirectory() as tmp:
            gaf = os.path.join(tmp, "test.gaf")
            create_dummy_gaf(gaf)

            result = map_genes_using_gaf(
                ["GENE1", "UNKNOWN"],
                gaf,
                to="symbol"
            )

            self.assertEqual(result, ["SYMBOL1", "UNKNOWN"])

    def test_map_genes_id(self):
        """Mapping to IDs should work."""

        with tempfile.TemporaryDirectory() as tmp:
            gaf = os.path.join(tmp, "test.gaf")
            create_dummy_gaf(gaf)

            result = map_genes_using_gaf(
                ["SYMBOL1"],
                gaf,
                to="id"
            )

            self.assertEqual(result, ["GENE1"])


class TestGeneInfo(unittest.TestCase):

    def test_load_gene_info(self):
        """Gene info should load as DataFrame."""

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gene_info.tsv")
            create_dummy_gene_info(path)

            df = load_gene_info(path)

            self.assertIn("GeneID", df.columns)
            self.assertEqual(len(df), 2)

    def test_entrez_to_symbol(self):
        """Mapping from Entrez IDs should work."""

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gene_info.tsv")
            create_dummy_gene_info(path)

            result = entrez_to_symbol_ncbi(
                ["1", "999"],
                path,
                na_value="NA"
            )

            self.assertEqual(result, ["A", "NA"])

    def test_invalid_gene_info_columns(self):
        """Missing columns should raise."""

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gene_info.tsv")

            pd.DataFrame({"X": [1]}).to_csv(path, sep="\t", index=False)

            with self.assertRaises(KeyError):
                entrez_to_symbol_ncbi(["1"], path)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()