"""
MappingEntrez.py (refactor)

Objetivo:
- Convertir una lista de IDs/símbolos génicos a EntrezID usando:
  1) gProfiler (primero)
  2) MyGene.info (fallback para los no mapeados)

Regla determinista:
- Si un gen se mapea a múltiples EntrezIDs, se selecciona el EntrezID numéricamente más bajo.

Notas de diseño:
- Logging en vez de prints.
- Retries con backoff exponencial para robustez de red.
- Control configurable de chunking y paralelismo.
- Tipos homogéneos: EntrezIDs retornan siempre como str (o na_value).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from gprofiler import GProfiler
import mygene


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MappingOptions:
    organism_gp: str = "hsapiens"
    tax_id: int = 9606
    scopes_mg: Sequence[str] = ("symbol", "alias", "tair", "accession", "refseq")
    na_value: str = "NA"

    # Performance / estabilidad
    n_threads: int = 4
    chunk_size: int = 250  # recomendado 200–300 para estabilidad
    request_retries: int = 3
    backoff_base_seconds: float = 0.6  # sleep = base * (2**attempt)
    timeout_seconds: Optional[float] = None  # mygene usa requests internamente (no siempre configurable)


# ──────────────────────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────────────────────

def _iter_chunks(seq: Sequence[str], n: int) -> Iterator[List[str]]:
    if n <= 0:
        raise ValueError("chunk_size must be > 0")
    for i in range(0, len(seq), n):
        yield list(seq[i : i + n])


def _min_entrez_str(values: Union[str, int, float, List, tuple, None]) -> Optional[str]:
    """
    Normaliza el/los entrez candidates y retorna el menor como str.
    Retorna None si no hay candidato numérico.
    """
    if values is None:
        return None

    if isinstance(values, (list, tuple)):
        nums = []
        for v in values:
            if v is None:
                continue
            s = str(v).strip()
            if s.isnumeric():
                nums.append(int(s))
        return str(min(nums)) if nums else None

    s = str(values).strip()
    if s.isnumeric():
        return str(int(s))
    return None


def _retry_call(fn, *, retries: int, backoff_base: float, retry_exceptions: tuple) -> any:
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except retry_exceptions as e:
            last_exc = e
            sleep_s = backoff_base * (2 ** attempt)
            logger.warning("Retry %d/%d after error: %s (sleep %.2fs)", attempt + 1, retries, e, sleep_s)
            time.sleep(sleep_s)
    # agotar retries
    raise RuntimeError(f"Request failed after {retries} retries") from last_exc


# ──────────────────────────────────────────────────────────────
# gProfiler
# ──────────────────────────────────────────────────────────────

def _map_with_gprofiler(
    genes: Sequence[str],
    options: MappingOptions,
) -> Dict[str, str]:
    """
    Retorna mapping incoming_gene -> min_entrez_as_str (solo válidos).
    """
    if not genes:
        return {}

    def _do():
        gp = GProfiler(return_dataframe=True)
        df = gp.convert(
            organism=options.organism_gp,
            query=list(genes),
            target_namespace="ENTREZGENE_ACC",
        )
        return df

    try:
        conversion = _retry_call(
            _do,
            retries=options.request_retries,
            backoff_base=options.backoff_base_seconds,
            retry_exceptions=(requests.exceptions.RequestException, Exception),
        )
    except Exception as e:
        logger.warning("gProfiler failed: %s", e)
        return {}

    if not isinstance(conversion, pd.DataFrame) or conversion.empty:
        return {}

    # Filtrar y normalizar a numérico de forma vectorizada
    # Esperado: columnas "incoming" y "converted"
    if "incoming" not in conversion.columns or "converted" not in conversion.columns:
        logger.warning("gProfiler response missing expected columns.")
        return {}

    conv = conversion.loc[conversion["converted"].notna(), ["incoming", "converted"]].copy()
    conv["converted_num"] = pd.to_numeric(conv["converted"], errors="coerce")
    conv = conv.dropna(subset=["converted_num"])
    if conv.empty:
        return {}

    # Seleccionar el mínimo por incoming (determinista)
    grouped = conv.groupby("incoming", as_index=False)["converted_num"].min()
    mapping = {str(row["incoming"]): str(int(row["converted_num"])) for _, row in grouped.iterrows()}
    return mapping


# ──────────────────────────────────────────────────────────────
# MyGene.info
# ──────────────────────────────────────────────────────────────

def _query_mygene_chunk(
    mg: mygene.MyGeneInfo,
    chunk: Sequence[str],
    options: MappingOptions,
) -> pd.DataFrame:
    """
    Retorna DataFrame con índice = query string (row.name) y columna 'entrezgene'.
    """
    if not chunk:
        return pd.DataFrame()

    def _do():
        # as_dataframe=True devuelve un DF indexado por query (gene/id)
        return mg.querymany(
            list(chunk),
            scopes=list(options.scopes_mg),
            fields="entrezgene",
            species=options.tax_id,
            as_dataframe=True,
        )

    df = _retry_call(
        _do,
        retries=options.request_retries,
        backoff_base=options.backoff_base_seconds,
        retry_exceptions=(requests.exceptions.RequestException, Exception),
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    # normalizar notfound
    if "notfound" not in df.columns:
        df["notfound"] = False
    else:
        df["notfound"] = df["notfound"].fillna(False)

    if "entrezgene" not in df.columns:
        return pd.DataFrame()

    valid = df[(~df["notfound"]) & (df["entrezgene"].notnull())].copy()
    return valid


def _map_with_mygene(
    genes: Sequence[str],
    options: MappingOptions,
) -> Dict[str, str]:
    """
    Retorna mapping gene -> min_entrez_as_str (solo válidos).
    """
    if not genes:
        return {}

    mg = mygene.MyGeneInfo()
    mapping: Dict[str, str] = {}

    blocks = list(_iter_chunks(list(genes), options.chunk_size))

    # Paralelizar por chunk
    with ThreadPoolExecutor(max_workers=max(1, int(options.n_threads))) as ex:
        futures = [ex.submit(_query_mygene_chunk, mg, block, options) for block in blocks]

        frames: List[pd.DataFrame] = []
        for fut in as_completed(futures):
            try:
                df = fut.result()
            except Exception as e:
                logger.warning("MyGene chunk failed: %s", e)
                continue

            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)

    if not frames:
        return {}

    merged = pd.concat(frames, axis=0)

    # merged index = query string. entrezgene puede ser escalar o lista.
    for idx, row in merged.iterrows():
        ent = row.get("entrezgene", None)
        mn = _min_entrez_str(ent)
        if mn is not None:
            mapping[str(idx)] = mn

    return mapping


# ──────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────

def convert_to_entrez_id(
    symbol_list: Sequence[str],
    options: MappingOptions = MappingOptions(),
) -> List[str]:
    """
    Convertir lista de IDs/símbolos a EntrezID.

    Estrategia:
    1) gProfiler para todos
    2) MyGene.info para los no mapeados

    Retorna:
    - Lista (mismo orden que symbol_list) con EntrezID como str o options.na_value.
    """
    if not isinstance(symbol_list, (list, tuple)) or len(symbol_list) == 0:
        raise ValueError("symbol_list must be a non-empty list/tuple of strings.")

    genes = [str(g) for g in symbol_list]
    conversion_dict: Dict[str, str] = {}

    # 1) gProfiler
    gp_map = _map_with_gprofiler(genes, options)
    conversion_dict.update(gp_map)

    mapped_genes = set(gp_map.keys())
    unmapped = [g for g in genes if g not in mapped_genes]

    logger.info("gProfiler → %d transformed, %d no match.", len(mapped_genes), len(unmapped))

    # 2) MyGene.info (fallback)
    if unmapped:
        logger.info("MyGene.info query for %d genes...", len(unmapped))
        mg_map = _map_with_mygene(unmapped, options)
        conversion_dict.update(mg_map)
        logger.info("MyGene.info → %d genes converted.", len(mg_map))

        still_unmapped = set(unmapped) - set(mg_map.keys())
        if still_unmapped:
            logger.warning("Warning: %d genes still unmapped by MyGene.info.", len(still_unmapped))

    # reconstruir en el orden original
    return [conversion_dict.get(g, options.na_value) for g in genes]


# Backwards-compatible alias (tu nombre original)
def Convert_To_Entrez_ID(
    symbol_list: List[str],
    organism_gp: str = "hsapiens",
    taxID: int = 9606,
    scopes_mg: List[str] = ["symbol", "alias", "tair", "accession", "refseq"],
    na_value: str = "NA",
    n_threads: int = 4,
) -> List[str]:
    """
    Wrapper compatible con la firma original.
    """
    opts = MappingOptions(
        organism_gp=organism_gp,
        tax_id=taxID,
        scopes_mg=tuple(scopes_mg),
        na_value=na_value,
        n_threads=n_threads,
    )
    return convert_to_entrez_id(symbol_list, options=opts)