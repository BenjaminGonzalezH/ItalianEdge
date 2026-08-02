"""
go_utils: Utility functions for Gene Ontology (GO) data handling. This module provides:
    - Download and extraction of GO resources (GAF, gene_info)
    - Gene identifier mapping (Entrez to Symbol)
    - Gene information loading (NCBI gene_info)

The module is intentionally stateless:
- No caching is performed
- All operations are explicit and reproducible

Functions
1. download_file           – Download a file from a URL, with retry and backoff.
2. gunzip_file             – Extract a .gz file to a destination path.
3. ensure_gaf_file         – Ensure a species GAF annotation file exists locally, downloading it if missing.
4. ensure_gene_info_file   – Ensure a species gene_info file exists locally, downloading it if missing.
5. load_gene_info          – Load an NCBI gene_info file into a DataFrame.
6. entrez_to_symbol_ncbi   – Convert Entrez IDs to gene symbols using a gene_info file.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import gzip
import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd
import requests

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Resource definitions
# ---------------------------------------------------------------------

# GAFs from the GO Consortium (current.geneontology.org/annotations/)
# gene_info per species from NCBI (ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/...)
#
# References:
# - Annotations directory: https://current.geneontology.org/annotations/ :contentReference[oaicite:3]{index=3}
# - General NCBI gene_info: https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz :contentReference[oaicite:4]{index=4}
# - Example species gene_info (H. sapiens): .../GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz :contentReference[oaicite:5]{index=5}

SPECIES_RESOURCES: dict[str, dict[str, str]] = {
    # Human: GOA + NCBI gene_info Mammalia
    "goa_human": {
        "gaf_gz": "https://current.geneontology.org/annotations/goa_human.gaf.gz",
        "gene_info_gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
    },
    # Mouse (MGI)
    "mgi": {
        "gaf_gz": "https://current.geneontology.org/annotations/mgi.gaf.gz",
        "gene_info_gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Mus_musculus.gene_info.gz",
    },
    # Fly (FB)
    "fb": {
        "gaf_gz": "https://current.geneontology.org/annotations/fb.gaf.gz",
        "gene_info_gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Invertebrates/Drosophila_melanogaster.gene_info.gz",
    },
    # Zebrafish (ZFIN)
    "zfin": {
        "gaf_gz": "https://current.geneontology.org/annotations/zfin.gaf.gz",
        "gene_info_gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Vertebrates_other/Danio_rerio.gene_info.gz",
    },
    # Yeast (SGD)
    "sgd": {
        "gaf_gz": "https://current.geneontology.org/annotations/sgd.gaf.gz",
        "gene_info_gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Fungi/Saccharomyces_cerevisiae.gene_info.gz",
    },
    # Arabidopsis (TAIR)
    "tair": {
        "gaf_gz": "https://current.geneontology.org/annotations/tair.gaf.gz",
        "gene_info_gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Plants/Arabidopsis_thaliana.gene_info.gz",
    },
    # WormBase
    "wb": {
        "gaf_gz": "https://current.geneontology.org/annotations/wb.gaf.gz",
        "gene_info_gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Invertebrates/Caenorhabditis_elegans.gene_info.gz",
    },
}


# ---------------------------------------------------------------------
# Configuration classes
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadOptions:
    """
    Configuration for HTTP downloads.

    Parameters
    ----------
    timeout_seconds : float
        Timeout for HTTP requests.

    retries : int
        Number of retry attempts on failure.

    backoff_base_seconds : float
        Base time for exponential backoff.

    chunk_size : int
        Chunk size for streaming downloads.
    """

    timeout_seconds: float = 60.0
    retries: int = 3
    backoff_base_seconds: float = 0.6
    chunk_size: int = 1024 * 1024


@dataclass(frozen=True)
class GeneInfoOptions:
    """
    Configuration for gene_info handling.

    Parameters
    ----------
    na_value : str
        Value used when mapping fails.

    download_if_missing : bool
        Whether gene_info should be downloaded if not found.
    """

    na_value: str = "NA"
    download_if_missing: bool = False


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------


def _as_path(p: PathLike) -> Path:
    """Convert input into Path object."""
    return p if isinstance(p, Path) else Path(p)


def _retry_get(url: str, opts: DownloadOptions) -> requests.Response:
    """
    Perform HTTP GET with retry logic.

    Raises
    ------
    RuntimeError if all retries fail.
    """
    last_exc: Exception | None = None

    for attempt in range(opts.retries):
        try:
            r = requests.get(url, stream=True, timeout=opts.timeout_seconds)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            sleep_s = opts.backoff_base_seconds * (2**attempt)

            logger.warning(
                "Retry %d/%d for %s after error: %s",
                attempt + 1,
                opts.retries,
                url,
                e,
            )

            time.sleep(sleep_s)

    raise RuntimeError(f"Failed download: {url}") from last_exc


# ---------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------


def download_file(
    url: str, dest: PathLike, opts: DownloadOptions = DownloadOptions()
) -> Path:
    """
    Download file from URL.

    Parameters
    ----------
    url : str
        Source URL.

    dest : PathLike
        Destination file path.

    Returns
    -------
    Path
        Downloaded file path.
    """
    dest_p = _as_path(dest)
    dest_p.parent.mkdir(parents=True, exist_ok=True)

    r = _retry_get(url, opts)

    with open(dest_p, "wb") as f:
        for chunk in r.iter_content(chunk_size=opts.chunk_size):
            if chunk:
                f.write(chunk)

    return dest_p


def gunzip_file(gz_path: PathLike, out_path: PathLike) -> Path:
    """
    Extract .gz file.

    Returns
    -------
    Path
        Extracted file path.
    """
    gz_p = _as_path(gz_path)
    out_p = _as_path(out_path)

    out_p.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(gz_p, "rb") as f_in, open(out_p, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return out_p


# ---------------------------------------------------------------------
# Resource management
# ---------------------------------------------------------------------


def ensure_gaf_file(
    species_key: str,
    *,
    out_dir: PathLike = ".",
    download_if_missing: bool = True,
    download_opts: DownloadOptions = DownloadOptions(),
) -> Path:
    """
    Ensure GAF file is available locally.

    Returns
    -------
    Path to .gaf file
    """
    if species_key not in SPECIES_RESOURCES:
        raise KeyError(f"Unknown species: {species_key}")

    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gaf_path = out_dir / f"{species_key}.gaf"

    if gaf_path.exists():
        return gaf_path

    if not download_if_missing:
        raise FileNotFoundError(gaf_path)

    gz = download_file(
        SPECIES_RESOURCES[species_key]["gaf_gz"],
        gaf_path.with_suffix(".gaf.gz"),
        download_opts,
    )
    gunzip_file(gz, gaf_path)

    gz.unlink(missing_ok=True)
    return gaf_path


def ensure_gene_info_file(
    species_key: str,
    *,
    out_dir: PathLike = ".",
    download_if_missing: bool = True,
    download_opts: DownloadOptions = DownloadOptions(),
) -> Path:
    """
    Ensure gene_info file exists locally.

    Returns
    -------
    Path
    """
    if species_key not in SPECIES_RESOURCES:
        raise KeyError(f"Unknown species: {species_key}")

    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{species_key}.gene_info"

    if out_path.exists():
        return out_path

    if not download_if_missing:
        raise FileNotFoundError(out_path)

    gz = download_file(
        SPECIES_RESOURCES[species_key]["gene_info_gz"],
        out_path.with_suffix(".gz"),
        download_opts,
    )
    gunzip_file(gz, out_path)

    gz.unlink(missing_ok=True)
    return out_path


# ---------------------------------------------------------------------
# Gene mapping
# ---------------------------------------------------------------------


def load_gene_info(gene_info_path: PathLike) -> pd.DataFrame:
    """
    Load gene_info file (NCBI).
    """
    return pd.read_csv(gene_info_path, sep="\t", dtype=str)


def entrez_to_symbol_ncbi(
    entrez_ids: Sequence[str | int],
    gene_info_path: PathLike,
    *,
    na_value: str = "NA",
) -> list[str]:
    """
    Convert Entrez IDs to gene symbols using gene_info.
    """
    df = load_gene_info(gene_info_path)

    mapping = df.set_index("GeneID")["Symbol"].to_dict()

    return [mapping.get(str(e), na_value) for e in entrez_ids]
