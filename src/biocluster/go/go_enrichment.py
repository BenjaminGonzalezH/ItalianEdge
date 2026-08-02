"""
go_enrichment: Gene Ontology (GO) enrichment and annotation utilities using g:Profiler.
This module provides tools to:
    • Perform enrichment analysis on a gene set.
    • Map genes to GO terms (annotation).
    • Handle large gene lists using chunking and parallelization.
    • Ensure robustness via retries and stable preprocessing.

Functions
1. go_enrichment              – Perform GO enrichment analysis on a gene set.
2. annotation_from_entrez_ids – Map genes to GO terms using chunked parallel queries.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Libraries
# ──────────────────────────────────────────────────────────────────────────────
from dataclasses        import dataclass
from typing             import Dict, Iterator, List, Sequence, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from gprofiler          import GProfiler
import logging
import time
import math
import numpy  as np
import pandas as pd


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class GoEnrichmentOptions:
    """
    Configuration for GO enrichment analysis.
    """
    organism: str            = "hsapiens"
    sources: Tuple[str, ...] = ("GO:BP",)
    user_threshold: float    = 0.05
    include_evidences: bool  = False
    request_retries: int        = 3
    backoff_base_seconds: float = 0.6
    compute_gene_ratio: bool = True
    compute_qscore: bool     = True
    sort_by: str             = "p_value"


@dataclass(frozen=True)
class AnnotationOptions:
    """
    Configuration for GO annotation queries.
    """
    organism: str               = "hsapiens"
    sources: Tuple[str, ...]    = ("GO:BP", "GO:CC", "GO:MF")
    chunk_size: int             = 500
    n_threads: int              = 4
    request_retries: int        = 3
    backoff_base_seconds: float = 0.6


# ─────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────

def _as_str_list(values: Union[Sequence, np.ndarray]) -> List[str]:
    """
    Normalize input values to a clean list of strings.

    Removes:
    • None values
    • empty strings
    • whitespace

    Parameters
    ----------
    values : sequence
        Input gene identifiers.

    Returns
    -------
    list[str]
        Clean list of string identifiers.
    """

    if values is None:
        raise ValueError("Input list must not be None.")

    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError(f"Expected list/tuple/ndarray/Series, got: {type(values)}")

    return [
        str(v).strip()
        for v in values
        if v is not None and str(v).strip() != ""
    ]


def _dedup_stable(values: List[str]) -> List[str]:
    """
    Remove duplicates while preserving original order.

    This ensures deterministic behavior.

    Returns
    -------
    list[str]
    """

    seen = set()
    out = []

    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)

    return out


def _iter_chunks(seq: Sequence[str], chunk_size: int) -> Iterator[List[str]]:
    """
    Split a sequence into fixed-size chunks.

    Used for annotation queries to avoid large requests.

    Parameters
    ----------
    seq : list[str]
    chunk_size : int

    Yields
    ------
    list[str]
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")

    for i in range(0, len(seq), chunk_size):
        yield list(seq[i:i + chunk_size])


def _retry_call(fn, *, retries: int, backoff_base: float, retry_exceptions: Tuple[type, ...]):
    """
    Execute a function with retry logic and exponential backoff.

    Parameters
    ----------
    fn : callable
    retries : int
    backoff_base : float
    retry_exceptions : tuple

    Returns
    -------
    Any
        Result of function execution.
    """

    last_exc = None

    for attempt in range(retries):

        try:
            return fn()

        except retry_exceptions as e:
            last_exc = e
            sleep_s = backoff_base * (2 ** attempt)

            logger.warning(
                "Retry %d/%d after error: %s (sleep %.2fs)",
                attempt + 1, retries, e, sleep_s
            )

            time.sleep(sleep_s)

    raise RuntimeError(f"Request failed after {retries} retries") from last_exc


def _safe_neglog10(p: Union[float, int]) -> float:
    """
    Compute -log10(p) safely.

    Handles:
    • p = 0
    • NaN / Inf

    Returns
    -------
    float
    """

    try:
        p = float(p)
    except Exception:
        return float("nan")

    if not math.isfinite(p):
        return float("nan")

    if p <= 0.0:
        p = 1e-300

    return -math.log10(p)


# ─────────────────────────────────────────────
# Enrichment
# ─────────────────────────────────────────────

def go_enrichment(entrez_ids, options=GoEnrichmentOptions()) -> pd.DataFrame:
    """
    Perform GO enrichment analysis.

    Parameters
    ----------
    entrez_ids : list[str or int]
        Input gene identifiers.

    options : GoEnrichmentOptions

    Returns
    -------
    pd.DataFrame
        Enrichment results sorted by significance.
    """
    if not (0.0 < float(options.user_threshold) <= 1.0):
        raise ValueError("options.user_threshold must be in (0, 1].")
    if len(options.sources) == 0:
        raise ValueError("options.sources must not be empty.")

    genes = _dedup_stable(_as_str_list(entrez_ids))

    if len(genes) == 0:
        raise ValueError("entrez_ids is empty after cleaning.")

    gp = GProfiler(return_dataframe=True)

    def _do():
        return gp.profile(
            organism=options.organism,
            query=genes,
            user_threshold=options.user_threshold,
            sources=list(options.sources),
            no_evidences=(not options.include_evidences),
        )

    results = _retry_call(
        _do,
        retries=options.request_retries,
        backoff_base=options.backoff_base_seconds,
        retry_exceptions=(Exception,),
    )

    if not isinstance(results, pd.DataFrame) or results.empty:
        return pd.DataFrame()

    df = results.copy()

    # Rename precision → gene_ratio
    if options.compute_gene_ratio and "precision" in df.columns:
        df = df.rename(columns={"precision": "gene_ratio"})

    # Compute qscore
    if options.compute_qscore and "p_value" in df.columns:

        pvals = df["p_value"].apply(_safe_neglog10)

        if "gene_ratio" in df.columns:
            df["qscore"] = pvals * df["gene_ratio"]
        else:
            df["qscore"] = pvals

    # Sort results
    if options.sort_by in df.columns:
        df = df.sort_values(options.sort_by).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────
# Annotation
# ─────────────────────────────────────────────

def annotation_from_entrez_ids(entrez_ids, options=AnnotationOptions()) -> Dict[str, List[str]]:
    """
    Map genes to GO terms using chunked parallel queries.

    Returns
    -------
    dict[str, list[str]]
        Gene → list of GO terms
    """

    genes = _dedup_stable(_as_str_list(entrez_ids))

    if len(genes) == 0:
        raise ValueError("entrez_ids is empty after cleaning.")

    if len(options.sources) == 0:
        raise ValueError("options.sources must not be empty.")

    gp = GProfiler(return_dataframe=True)

    def _query_block(block):

        def _do():
            return gp.profile(
                organism=options.organism,
                query=block,
                user_threshold=1.0,
                sources=list(options.sources),
                no_evidences=False,
            )

        return _retry_call(
            _do,
            retries=options.request_retries,
            backoff_base=options.backoff_base_seconds,
            retry_exceptions=(Exception,),
        )

    frames = []
    n_chunks = 0
    n_failed = 0

    with ThreadPoolExecutor(max_workers=options.n_threads) as ex:

        futures = {
            ex.submit(_query_block, b): b
            for b in _iter_chunks(genes, options.chunk_size)
        }
        n_chunks = len(futures)

        for f in as_completed(futures):
            try:
                df = f.result()
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                n_failed += 1
                logger.warning(
                    "Annotation chunk failed (%d/%d): %s",
                    n_failed, n_chunks, exc,
                )

    if n_failed:
        logger.warning(
            "%d/%d annotation chunks failed; gene mapping may be incomplete.",
            n_failed, n_chunks,
        )

    if not frames:
        return {}

    merged = pd.concat(frames)

    if "intersections" not in merged.columns or "native" not in merged.columns:
        logger.warning(
            "g:Profiler response missing required columns; returning empty mapping."
        )
        return {}

    exploded = merged[["intersections", "native"]].explode("intersections")

    mapping = (
        exploded.groupby("intersections")["native"]
        .agg(lambda s: sorted(set(map(str, s))))
        .to_dict()
    )

    return {str(k): v for k, v in mapping.items()}