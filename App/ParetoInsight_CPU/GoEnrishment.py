"""
GoEnrishment.py (refactor, large-scale ready)

Funciones:
- go_enrichment: Enrichment analysis (NO chunking del query; científicamente correcto).
- annotation_from_entrez_ids: Gene -> términos (SÍ chunking + paralelismo, seguro).

Notas científicas:
- g:Profiler (g:GOSt) usa test hipergeométrico + corrección por múltiples tests.
- Por defecto aplica g:SCS y reporta p-values ajustados en los resultados. :contentReference[oaicite:2]{index=2}

Este módulo reemplaza prints por logging y agrega:
- Validaciones fuertes
- Orden determinista
- Retries con backoff exponencial para fallos de red
- Opciones configurables (dataclasses)
- Wrappers con tus nombres originales para compatibilidad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import logging
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from gprofiler import GProfiler

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────────────────────

def _as_str_list(values: Union[Sequence, np.ndarray]) -> List[str]:
    if values is None:
        raise ValueError("Input list must not be None.")
    if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
        raise TypeError(f"Expected list/tuple/ndarray/Series, got: {type(values)}")
    out = [str(v).strip() for v in values if v is not None and str(v).strip() != ""]
    return out

def _dedup_stable(values: List[str]) -> List[str]:
    # preserva el primer orden de aparición (determinista)
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _iter_chunks(seq: Sequence[str], chunk_size: int) -> Iterator[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i:i + chunk_size])

def _retry_call(fn, *, retries: int, backoff_base: float, retry_exceptions: Tuple[type, ...]) -> any:
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except retry_exceptions as e:
            last_exc = e
            sleep_s = backoff_base * (2 ** attempt)
            logger.warning("Retry %d/%d after error: %s (sleep %.2fs)", attempt + 1, retries, e, sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"Request failed after {retries} retries") from last_exc

def _safe_neglog10(p: Union[float, int]) -> float:
    # evita -inf si p=0 o NaN; usa un piso conservador
    try:
        p = float(p)
    except Exception:
        return float("nan")
    if not math.isfinite(p):
        return float("nan")
    if p <= 0.0:
        p = 1e-300
    return -math.log10(p)

# ──────────────────────────────────────────────────────────────
# Opciones / Config
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoEnrichmentOptions:
    organism: str = "hsapiens"
    sources: Tuple[str, ...] = ("GO:BP",)
    user_threshold: float = 0.05
    include_evidences: bool = False  # si False -> no_evidences=True

    # Robustez red
    request_retries: int = 3
    backoff_base_seconds: float = 0.6

    # Output / scoring
    compute_gene_ratio: bool = True   # renombra precision -> gene_ratio si existe
    compute_qscore: bool = True       # score custom documentado
    sort_by: str = "p_value"          # 'p_value' suele ser ajustado por g:SCS en g:Profiler :contentReference[oaicite:3]{index=3}


@dataclass(frozen=True)
class AnnotationOptions:
    organism: str = "hsapiens"
    sources: Tuple[str, ...] = ("GO:BP", "GO:CC", "GO:MF")
    chunk_size: int = 500
    n_threads: int = 4

    # Robustez red
    request_retries: int = 3
    backoff_base_seconds: float = 0.6

# ──────────────────────────────────────────────────────────────
# API principal
# ──────────────────────────────────────────────────────────────

def go_enrichment(
    entrez_ids: Sequence[Union[str, int, np.str_]],
    options: GoEnrichmentOptions = GoEnrichmentOptions(),
) -> pd.DataFrame:
    """
    Enrichment (g:Profiler g:GOSt) para un *set* de genes (NO se chunkea).

    Args:
        entrez_ids: lista de genes (Entrez IDs como str/int). Se normaliza a str.
        options: configuración (organism, sources, threshold, retries, etc.)

    Returns:
        DataFrame ordenado por options.sort_by (por defecto p_value).
        Si no hay resultados, retorna DataFrame vacío.

    Correctitud:
        No se divide el query en chunks, porque cambia el test y la corrección múltiple. :contentReference[oaicite:4]{index=4}
    """
    genes = _as_str_list(entrez_ids)
    genes = _dedup_stable(genes)

    if len(genes) == 0:
        raise ValueError("entrez_ids is empty after cleaning.")
    if len(options.sources) == 0:
        raise ValueError("options.sources must not be empty.")
    if not (0.0 < float(options.user_threshold) <= 1.0):
        raise ValueError("options.user_threshold must be in (0, 1].")

    gp = GProfiler(return_dataframe=True)

    def _do():
        # Nota: gprofiler-official usa profile(**kwargs) con estos nombres en tu código actual.
        return gp.profile(
            organism=options.organism,
            query=genes,
            user_threshold=float(options.user_threshold),
            sources=list(options.sources),
            no_evidences=(not options.include_evidences),
        )

    try:
        results = _retry_call(
            _do,
            retries=int(options.request_retries),
            backoff_base=float(options.backoff_base_seconds),
            retry_exceptions=(Exception,),
        )
    except Exception as e:
        raise RuntimeError("Error during g:Profiler enrichment request") from e

    if not isinstance(results, pd.DataFrame) or results.empty:
        logger.info("No enrichment terms found for query (n=%d).", len(genes))
        return pd.DataFrame()

    df = results.copy()

    # gene_ratio: gprofiler suele traer 'precision' (ratio de query en término) en tu versión actual
    if options.compute_gene_ratio and "precision" in df.columns and "gene_ratio" not in df.columns:
        df = df.rename(columns={"precision": "gene_ratio"})

    # qscore custom: (-log10(p)) * gene_ratio
    if options.compute_qscore:
        if "p_value" in df.columns:
            pvals = df["p_value"].apply(_safe_neglog10)
            if "gene_ratio" in df.columns:
                df["qscore"] = pvals * pd.to_numeric(df["gene_ratio"], errors="coerce")
            else:
                df["qscore"] = pvals
        else:
            logger.warning("Column 'p_value' not found; qscore not computed.")

    # Orden
    sort_col = options.sort_by if options.sort_by in df.columns else "p_value"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=True, na_position="last").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def annotation_from_entrez_ids(
    entrez_ids: Sequence[Union[str, int, np.str_]],
    options: AnnotationOptions = AnnotationOptions(),
) -> Dict[str, List[str]]:
    """
    Gene -> lista de términos (native IDs de g:Profiler), usando chunking + paralelismo.

    Retorna:
        dict[gene_id_str] = [term1, term2, ...]
    """
    genes = _as_str_list(entrez_ids)
    genes = _dedup_stable(genes)
    if len(genes) == 0:
        raise ValueError("entrez_ids is empty after cleaning.")
    if len(options.sources) == 0:
        raise ValueError("options.sources must not be empty.")

    gp = GProfiler(return_dataframe=True)

    def _query_block(block: List[str]) -> pd.DataFrame:
        def _do():
            # user_threshold=1.0 -> sin filtro estadístico (solo queremos anotación / intersecciones)
            return gp.profile(
                organism=options.organism,
                query=block,
                no_evidences=False,
                user_threshold=1.0,
                sources=list(options.sources),
            )

        return _retry_call(
            _do,
            retries=int(options.request_retries),
            backoff_base=float(options.backoff_base_seconds),
            retry_exceptions=(Exception,),
        )

    blocks = list(_iter_chunks(genes, int(options.chunk_size)))

    frames: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, int(options.n_threads))) as ex:
        futs = [ex.submit(_query_block, b) for b in blocks]
        for fut in as_completed(futs):
            try:
                df = fut.result()
            except Exception as e:
                logger.warning("Annotation chunk failed: %s", e)
                continue
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)

    if not frames:
        return {}

    merged = pd.concat(frames, axis=0, ignore_index=True)

    # Esperado en gprofiler DF:
    # - 'intersections' contiene genes del query que intersectan el término
    # - 'native' es el ID del término (GO:..., REAC:..., etc.)
    if "intersections" not in merged.columns or "native" not in merged.columns:
        logger.warning("g:Profiler response missing 'intersections'/'native'; returning empty mapping.")
        return {}

    exploded = merged[["intersections", "native"]].explode("intersections")
    exploded = exploded.dropna(subset=["intersections", "native"])

    # Agrupar
    gene_to_terms = (
        exploded.groupby("intersections")["native"]
        .agg(lambda s: sorted(set(map(str, s))))  # determinista: set + sorted
        .to_dict()
    )

    # Normalizar keys a str y asegurar que estén en la lista original (limpio)
    out: Dict[str, List[str]] = {}
    gene_set = set(genes)
    for k, v in gene_to_terms.items():
        ks = str(k)
        if ks in gene_set:
            out[ks] = list(v)
    return out


# ──────────────────────────────────────────────────────────────
# Wrappers compat con tu API original
# ──────────────────────────────────────────────────────────────

def GoEnrichment(
    entrez_ids: List[str],
    organism: str = "hsapiens",
    Ontology: List[str] = ["GO:BP"],
    evidences: bool = False,
) -> pd.DataFrame:
    """
    Wrapper compatible con tu función original.
    """
    opts = GoEnrichmentOptions(
        organism=organism,
        sources=tuple(Ontology),
        user_threshold=0.05,
        include_evidences=bool(evidences),
    )
    return go_enrichment(entrez_ids, options=opts)


def AnnotationFromEntrezIDs(
    entrez_ids: List[np.str_],
    Ontology: List[str] = ["GO:BP", "GO:CC", "GO:MF"],
    organism: str = "hsapiens",
) -> Dict[str, List[str]]:
    """
    Wrapper compatible con tu función original.
    """
    opts = AnnotationOptions(
        organism=organism,
        sources=tuple(Ontology),
        chunk_size=500,
        n_threads=4,
    )
    return annotation_from_entrez_ids(entrez_ids, options=opts)