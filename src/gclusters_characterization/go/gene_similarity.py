"""
WangIndex:contentReference[oaicite:2]{index=2}o)

Objetivos:
- Descarga opcional de GAF y gene_info por especie (URLs centralizadas).
- Sin cache para gene_info (load_gene_info siempre lee desde disco).
- Cálculo de matriz de similitud entre genes con GO3 (Wang u otras).
- Cálculo de similitud entre SOLUCIONES usando SOLO información biológica:
  - Similaridad entre clusters via promedio de similitud génica (matriz GO3)
  - Emparejamiento de clusters (Hungarian) maximizando similaridad biológica
  - Similaridad solución-solución como promedio (o ponderado) de matches

Notas:
- GO3 ya maneja paralelismo interno (set_num_threads).
- Logging en vez de prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, Literal

import logging
import os
import gzip
import shutil
import time

import numpy as np
import pandas as pd
import requests

import go3
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

Ontology = Literal["BP", "MF", "CC"]
Groupwise = Literal["bma", "max"]
SimilarityMeasure = str  # go3 admite "wang", "lin", etc.
ClusterWeighting = Literal["uniform", "size"]


# -----------------------------------------------------------------------------
# 1) Recursos por especie (GAF + gene_info)
# -----------------------------------------------------------------------------
# GAFs del GO Consortium (current.geneontology.org/annotations/)
# gene_info por especie en NCBI (ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/...)
#
# Referencias:
# - Directorio de annotations: https://current.geneontology.org/annotations/ :contentReference[oaicite:3]{index=3}
# - gene_info general en NCBI: https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz :contentReference[oaicite:4]{index=4}
# - Ejemplo species gene_info (H. sapiens): .../GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz :contentReference[oaicite:5]{index=5}

SPECIES_RESOURCES: Dict[str, Dict[str, str]] = {
    # Humanos: GOA + NCBI gene_info Mammalia
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


# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DownloadOptions:
    timeout_seconds: float = 60.0
    retries: int = 3
    backoff_base_seconds: float = 0.6  # sleep = base * 2^attempt
    chunk_size: int = 1024 * 1024  # 1MB


@dataclass(frozen=True)
class GeneInfoOptions:
    na_value: str = "NA"
    download_if_missing: bool = False
    # No cache: siempre se lee del archivo al llamar load_gene_info


@dataclass(frozen=True)
class GeneSimilarityOptions:
    ontology: Ontology = "BP"
    measure: SimilarityMeasure = "wang"
    groupwise: Groupwise = "bma"
    load_go_terms: bool = True
    num_threads_go3: int = 0  # go3.set_num_threads(0) = auto/por defecto


@dataclass(frozen=True)
class SolutionSimilarityOptions:
    weighting: ClusterWeighting = "uniform"  # "uniform" o "size"
    na_value: str = "NA"
    # Si hay genes fuera de ids/sim_matrix, se ignoran.


# -----------------------------------------------------------------------------
# Utilidades I/O (descarga + gunzip)
# -----------------------------------------------------------------------------

def _as_path(p: PathLike) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _retry_get(url: str, opts: DownloadOptions) -> requests.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(opts.retries):
        try:
            r = requests.get(url, stream=True, timeout=opts.timeout_seconds)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            sleep_s = opts.backoff_base_seconds * (2 ** attempt)
            logger.warning("Download retry %d/%d for %s after error: %s (sleep %.2fs)",
                           attempt + 1, opts.retries, url, e, sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to download after {opts.retries} retries: {url}") from last_exc


def download_file(url: str, dest: PathLike, opts: DownloadOptions = DownloadOptions()) -> Path:
    """
    Descarga URL a archivo dest (sin descomprimir).
    """
    dest_p = _as_path(dest)
    if dest_p.parent and not dest_p.parent.exists():
        dest_p.parent.mkdir(parents=True, exist_ok=True)

    r = _retry_get(url, opts)
    with open(dest_p, "wb") as f:
        for chunk in r.iter_content(chunk_size=opts.chunk_size):
            if chunk:
                f.write(chunk)

    logger.info("Downloaded: %s -> %s", url, dest_p)
    return dest_p


def gunzip_file(gz_path: PathLike, out_path: PathLike) -> Path:
    """
    Descomprime .gz a out_path.
    """
    gz_p = _as_path(gz_path)
    out_p = _as_path(out_path)
    if out_p.parent and not out_p.parent.exists():
        out_p.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(gz_p, "rb") as f_in:
        with open(out_p, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    logger.info("Uncompressed: %s -> %s", gz_p, out_p)
    return out_p


def ensure_gaf_file(
    species_key: str,
    *,
    out_dir: PathLike = ".",
    download_if_missing: bool = True,
    download_opts: DownloadOptions = DownloadOptions(),
) -> Path:
    """
    Asegura un .gaf (NO gz) listo para ser leído.
    Retorna ruta al .gaf.
    """
    if species_key not in SPECIES_RESOURCES:
        raise KeyError(f"Unknown species_key: {species_key}. Available: {list(SPECIES_RESOURCES)}")

    out_dir_p = _as_path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    gaf_name = f"{species_key}.gaf"
    gaf_path = out_dir_p / gaf_name
    if gaf_path.exists():
        return gaf_path

    if not download_if_missing:
        raise FileNotFoundError(f"GAF not found and download disabled: {gaf_path}")

    url = SPECIES_RESOURCES[species_key]["gaf_gz"]
    gz_path = out_dir_p / f"{gaf_name}.gz"
    download_file(url, gz_path, opts=download_opts)
    gunzip_file(gz_path, gaf_path)

    # borrar gz para no dejar basura
    try:
        gz_path.unlink(missing_ok=True)
    except Exception:
        pass

    return gaf_path


def ensure_gene_info_file(
    species_key: str,
    *,
    out_dir: PathLike = ".",
    download_if_missing: bool = False,
    download_opts: DownloadOptions = DownloadOptions(),
) -> Path:
    """
    Asegura gene_info (NO gz) listo para ser leído.
    Retorna ruta al .gene_info (o .tsv equivalente).

    Importante:
    - Descarga es opcional y por defecto NO descarga.
    """
    if species_key not in SPECIES_RESOURCES:
        raise KeyError(f"Unknown species_key: {species_key}. Available: {list(SPECIES_RESOURCES)}")

    out_dir_p = _as_path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_name = f"{species_key}.gene_info"
    out_path = out_dir_p / out_name
    if out_path.exists():
        return out_path

    if not download_if_missing:
        raise FileNotFoundError(f"gene_info not found and download disabled: {out_path}")

    url = SPECIES_RESOURCES[species_key]["gene_info_gz"]
    gz_path = out_dir_p / f"{out_name}.gz"
    download_file(url, gz_path, opts=download_opts)
    gunzip_file(gz_path, out_path)

    # borrar gz para no dejar basura
    try:
        gz_path.unlink(missing_ok=True)
    except Exception:
        pass

    return out_path


# -----------------------------------------------------------------------------
# Mapeos (GAF + gene_info)
# -----------------------------------------------------------------------------

def build_gaf_gene_mappings(gaf_path: PathLike) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Construye dicts:
      - id_to_symbol
      - symbol_to_id

    Nota:
    En GAF, las columnas típicas son:
      DB (0), DB_Object_ID (1), DB_Object_Symbol (2), ...
    """
    p = _as_path(gaf_path)
    if not p.exists():
        raise FileNotFoundError(f"GAF file not found: {p}")

    id_to_symbol: Dict[str, str] = {}
    symbol_to_id: Dict[str, str] = {}

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gene_id = parts[1]
            gene_symbol = parts[2]
            if gene_id and gene_symbol:
                id_to_symbol[gene_id] = gene_symbol
                symbol_to_id[gene_symbol] = gene_id

    return id_to_symbol, symbol_to_id


def map_genes_using_gaf(
    genes: Sequence[str],
    gaf_path: PathLike,
    *,
    to: Literal["symbol", "id"] = "symbol",
) -> List[str]:
    """
    Transforma lista de genes usando SOLO el GAF (DB_Object_ID <-> DB_Object_Symbol).
    """
    if not isinstance(genes, (list, tuple)):
        raise TypeError("genes must be a list/tuple of strings.")
    id_to_symbol, symbol_to_id = build_gaf_gene_mappings(gaf_path)

    out: List[str] = []
    for g in genes:
        s = str(g)
        if to == "symbol":
            out.append(id_to_symbol.get(s, s))
        elif to == "id":
            out.append(symbol_to_id.get(s, s))
        else:
            raise ValueError("to must be 'symbol' or 'id'")
    return out


def load_gene_info(
    gene_info_path: PathLike,
) -> pd.DataFrame:
    """
    Carga gene_info desde disco (SIN CACHE por requerimiento).
    El archivo gene_info de NCBI es tab-delimited.

    Retorna DataFrame dtype=str.
    """
    p = _as_path(gene_info_path)
    if not p.exists():
        raise FileNotFoundError(f"gene_info file not found: {p}")

    df = pd.read_csv(p, sep="\t", dtype=str, low_memory=False)
    return df


def entrez_to_symbol_ncbi(
    entrez_ids: Sequence[Union[str, int]],
    gene_info_path: PathLike,
    *,
    na_value: str = "NA",
) -> List[str]:
    """
    Convierte Entrez GeneID -> Symbol usando gene_info de NCBI.
    SIN CACHE: siempre lee el archivo al llamar.
    """
    if not isinstance(entrez_ids, (list, tuple)):
        raise TypeError("entrez_ids must be list/tuple.")

    df = load_gene_info(gene_info_path)
    if "GeneID" not in df.columns or "Symbol" not in df.columns:
        raise ValueError("gene_info must contain 'GeneID' and 'Symbol' columns.")

    mapping = df.set_index("GeneID")["Symbol"].to_dict()
    return [mapping.get(str(e), na_value) for e in entrez_ids]


# -----------------------------------------------------------------------------
# Similitud gen-gen con GO3
# -----------------------------------------------------------------------------

def compute_gene_similarity_matrix_go3(
    genes: Sequence[str],
    species_key: str,
    *,
    go3_opts: GeneSimilarityOptions = GeneSimilarityOptions(),
    out_dir: PathLike = ".",
    download_gaf_if_missing: bool = True,
    transform_genes_with_gaf: bool = False,
    gene_info_path: Optional[PathLike] = None,
    entrez_to_symbol: bool = False,
    gene_info_opts: GeneInfoOptions = GeneInfoOptions(),
    download_opts: DownloadOptions = DownloadOptions(),
) -> Tuple[List[str], np.ndarray]:
    """
    Calcula matriz de similitud entre genes usando GO3.

    Flujo de mapeo (opcional):
    - Si entrez_to_symbol=True: requiere gene_info_path o download_if_missing (gene_info_opts)
      y convierte GeneID -> Symbol.
    - Si transform_genes_with_gaf=True: usa GAF para mapear ID<->Symbol (según contenido del GAF).

    Retorna:
      ordered_genes, similarity_matrix (NxN, float)
    """
    if not isinstance(genes, (list, tuple)) or len(genes) == 0:
        raise ValueError("genes must be a non-empty list/tuple of strings.")

    if go3_opts.ontology not in ("BP", "MF", "CC"):
        raise ValueError("ontology must be 'BP', 'MF' or 'CC'.")
    if go3_opts.groupwise not in ("bma", "max"):
        raise ValueError("groupwise must be 'bma' or 'max'.")

    go3.set_num_threads(int(go3_opts.num_threads_go3))

    # Cargar GO terms si se pide
    if go3_opts.load_go_terms:
        go3.load_go_terms()

    # Asegurar GAF
    gaf_path = ensure_gaf_file(
        species_key,
        out_dir=out_dir,
        download_if_missing=download_gaf_if_missing,
        download_opts=download_opts,
    )

    # Preparar lista de genes
    genes_list = [str(g) for g in genes]

    # Entrez -> Symbol usando gene_info
    if entrez_to_symbol:
        if gene_info_path is None:
            # intenta asegurar/descargar según gene_info_opts
            gene_info_path = ensure_gene_info_file(
                species_key,
                out_dir=out_dir,
                download_if_missing=bool(gene_info_opts.download_if_missing),
                download_opts=download_opts,
            )
        genes_list = entrez_to_symbol_ncbi(genes_list, gene_info_path, na_value=gene_info_opts.na_value)

    # Mapear con GAF (si se solicita)
    if transform_genes_with_gaf:
        genes_list = map_genes_using_gaf(genes_list, gaf_path, to="symbol")

    # Cargar anotaciones y computar matriz de distancias con GO3
    annotations = go3.load_gaf(str(gaf_path))  # go3 espera path/alias de su loader
    counter = go3.build_term_counter(annotations)

    ordered, dist = go3.gene_distance_matrix(
        genes_list,
        ontology=go3_opts.ontology,
        similarity=go3_opts.measure,
        groupwise=go3_opts.groupwise,
        counter=counter,
        distance_transform="one_minus",  # produce dist = 1 - sim
    )

    # dist es distancia: queremos similitud
    sim = 1.0 - np.array(dist, dtype=np.float64)
    return list(ordered), sim


# -----------------------------------------------------------------------------
# Similitud solución-solución usando SOLO info biológica
# -----------------------------------------------------------------------------


def solution_wang_similarity_from_dataframe(
    ids: Sequence[str],
    gene_similarity_matrix: np.ndarray,
    reference_df: pd.DataFrame,
    solutions: Sequence[Sequence[Set[str]]],
    *,
    na_value: str = "NA",
    normalize_matrix: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute Wang biological similarity using a reference DataFrame that already
    contains Jaccard cluster matches.

    Parameters
    ----------
    ids:
        Ordered gene identifiers corresponding to gene_similarity_matrix.
    gene_similarity_matrix:
        NxN GO3 similarity matrix (Wang or other).
    reference_df:
        DataFrame with columns:
            - Solution 1
            - Solution 2
            - Cluster 1
            - Cluster 2
            - Jaccard Similarity
    solutions:
        List of solutions, each solution is list[set[str]].
    na_value:
        Gene label to ignore.
    normalize_matrix:
        Whether to normalize final solution matrix to [0,1].

    Returns
    -------
    solution_similarity_matrix (Wang only)
    updated_dataframe (with new column "Wang Similarity")
    """

    required_cols = {
        "Solution 1",
        "Solution 2",
        "Cluster 1",
        "Cluster 2",
        "Jaccard Similarity",
    }
    if not required_cols.issubset(reference_df.columns):
        raise ValueError(f"reference_df must contain columns: {required_cols}")

    n = len(solutions)
    final_matrix = np.zeros((n, n), dtype=np.float64)

    ids_list = [str(x) for x in ids]
    id_to_idx = {g: i for i, g in enumerate(ids_list)}

    wang_values = []

    for idx, row in reference_df.iterrows():

        s1 = int(row["Solution 1"])
        s2 = int(row["Solution 2"])
        c1 = int(row["Cluster 1"])
        c2 = int(row["Cluster 2"])

        cluster_a = solutions[s1][c1]
        cluster_b = solutions[s2][c2]

        idx_a = [
            id_to_idx[g] for g in cluster_a
            if g != na_value and g in id_to_idx
        ]
        idx_b = [
            id_to_idx[g] for g in cluster_b
            if g != na_value and g in id_to_idx
        ]

        if not idx_a or not idx_b:
            wang = 0.0
        else:
            submatrix = gene_similarity_matrix[np.ix_(idx_a, idx_b)]
            wang = float(np.nanmean(submatrix))

        wang_values.append(wang)

        final_matrix[s1, s2] += wang
        final_matrix[s2, s1] += wang

    reference_df = reference_df.copy()
    reference_df["Wang Similarity"] = wang_values

    if normalize_matrix and np.max(final_matrix) > 0:
        final_matrix /= np.max(final_matrix)

    np.fill_diagonal(final_matrix, 1.0)

    return final_matrix, reference_df


# -----------------------------------------------------------------------------
# Convenience: construir soluciones (labels -> list[set])
# -----------------------------------------------------------------------------

def labels_to_clusters(labels: Sequence[Union[int, str]], genes: Sequence[str]) -> List[Set[str]]:
    """
    Convierte vector labels (n_genes) y genes (n_genes) a list[set] clusters.
    """
    if len(labels) != len(genes):
        raise ValueError("labels and genes must have same length.")
    clusters: Dict[str, Set[str]] = {}
    for g, lab in zip(genes, labels):
        key = str(lab)
        clusters.setdefault(key, set()).add(str(g))
    return list(clusters.values())