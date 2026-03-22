"""
MappingEntrez.py (refactor - standardized US English comments)

Purpose:
- Convert a list of gene identifiers/symbols to Entrez IDs using:
  1) gProfiler (primary)
  2) MyGene.info (fallback for unmapped genes)

Deterministic rule:
- If a gene maps to multiple Entrez IDs, the smallest numeric ID is selected.

Notes:
- Outputs are always strings (or na_value if missing).
- Network robustness handled via retry logic.
"""

######### Libraries #########
import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Union

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from gprofiler import GProfiler
import mygene

logger = logging.getLogger(__name__)


######### Configuration #########
@dataclass(frozen=True)
class MappingOptions:
    """
    MappingOptions(class): Configuration parameters for Entrez ID conversion.

    Parameters
    ----------
    organism_gp : str
        Organism identifier for gProfiler.
    tax_id : int
        Taxonomy ID for MyGene queries.
    scopes_mg : Sequence[str]
        Fields used by MyGene for matching.
    na_value : str
        Value used when mapping fails.
    n_threads : int
        Number of threads for parallel queries.
    chunk_size : int
        Size of chunks for batch processing.
    request_retries : int
        Number of retries for failed requests.
    backoff_base_seconds : float
        Base time for exponential backoff.
    timeout_seconds : Optional[float]
        Optional timeout for requests.
    """

    organism_gp: str = "hsapiens"
    tax_id: int = 9606
    scopes_mg: Sequence[str] = ("symbol", "alias", "tair", "accession", "refseq", "ensembl.gene")
    na_value: str = "NA"

    n_threads: int = 4
    chunk_size: int = 250
    request_retries: int = 3
    backoff_base_seconds: float = 0.6
    timeout_seconds: Optional[float] = None


######### Utilities #########
def _iter_chunks(seq: Sequence[str], n: int) -> Iterator[List[str]]:
    """
    IterChunks(function): Split a sequence into chunks of size n.

    Parameters
    ----------
    seq : Sequence[str]
        Input sequence.
    n : int
        Chunk size.

    Returns
    -------
    Iterator[List[str]]
        Iterator over chunks.
    """

    if n <= 0:
        raise ValueError("chunk_size must be > 0")

    for i in range(0, len(seq), n):
        yield list(seq[i: i + n])


def _min_entrez_str(values: Union[str, int, float, List, tuple, None]) -> Optional[str]:
    """
    MinEntrezStr(function): Return the smallest valid Entrez ID as string.

    Parameters
    ----------
    values : Union[str, int, float, List, tuple, None]
        Candidate Entrez IDs.

    Returns
    -------
    Optional[str]
        Minimum Entrez ID as string or None if invalid.
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


def _retry_call(fn, *, retries: int, backoff_base: float, retry_exceptions: tuple):
    """
    RetryCall(function): Execute a function with retry and exponential backoff.

    Parameters
    ----------
    fn : callable
        Function to execute.
    retries : int
        Number of retry attempts.
    backoff_base : float
        Base delay for exponential backoff.
    retry_exceptions : tuple
        Exceptions that trigger retry.

    Returns
    -------
    any
        Result of the function call.
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
                attempt + 1,
                retries,
                e,
                sleep_s
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"Request failed after {retries} retries") from last_exc


######### gProfiler Mapping #########
def _map_with_gprofiler(
    genes: Sequence[str],
    options: MappingOptions
) -> Dict[str, str]:
    """
    MapWithGProfiler(function): Map genes to Entrez IDs using gProfiler.

    Parameters
    ----------
    genes : Sequence[str]
        List of gene identifiers.
    options : MappingOptions
        Configuration object.

    Returns
    -------
    Dict[str, str]
        Mapping from gene to Entrez ID.
    """

    if not genes:
        return {}

    def _do():
        gp = GProfiler(return_dataframe=True)
        return gp.convert(
            organism=options.organism_gp,
            query=list(genes),
            target_namespace="ENTREZGENE_ACC"
        )

    try:
        conversion = _retry_call(
            _do,
            retries=options.request_retries,
            backoff_base=options.backoff_base_seconds,
            retry_exceptions=(requests.exceptions.RequestException, Exception)
        )
    except Exception as e:
        logger.warning("gProfiler failed: %s", e)
        return {}

    if not isinstance(conversion, pd.DataFrame) or conversion.empty:
        return {}

    if "incoming" not in conversion.columns or "converted" not in conversion.columns:
        logger.warning("gProfiler response missing expected columns.")
        return {}

    conv = conversion.loc[conversion["converted"].notna(), ["incoming", "converted"]].copy()
    conv["converted_num"] = pd.to_numeric(conv["converted"], errors="coerce")
    conv = conv.dropna(subset=["converted_num"])

    if conv.empty:
        return {}

    grouped = conv.groupby("incoming", as_index=False)["converted_num"].min()

    return {
        str(row["incoming"]): str(int(row["converted_num"]))
        for _, row in grouped.iterrows()
    }


######### MyGene Mapping #########
def _query_mygene_chunk(
    mg: mygene.MyGeneInfo,
    chunk: Sequence[str],
    options: MappingOptions
) -> pd.DataFrame:
    """
    QueryMyGeneChunk(function): Query MyGene for a chunk of genes.

    Parameters
    ----------
    mg : mygene.MyGeneInfo
        MyGene client instance.
    chunk : Sequence[str]
        Subset of genes.
    options : MappingOptions
        Configuration object.

    Returns
    -------
    pd.DataFrame
        DataFrame with Entrez mappings.
    """

    if not chunk:
        return pd.DataFrame()

    def _do():
        return mg.querymany(
            list(chunk),
            scopes=list(options.scopes_mg),
            fields="entrezgene",
            species=options.tax_id,
            as_dataframe=True
        )

    df = _retry_call(
        _do,
        retries=options.request_retries,
        backoff_base=options.backoff_base_seconds,
        retry_exceptions=(requests.exceptions.RequestException, Exception)
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    if "notfound" not in df.columns:
        df["notfound"] = False
    else:
        df["notfound"] = df["notfound"].fillna(False)

    if "entrezgene" not in df.columns:
        return pd.DataFrame()

    return df[(~df["notfound"]) & (df["entrezgene"].notnull())].copy()


def _map_with_mygene(
    genes: Sequence[str],
    options: MappingOptions
) -> Dict[str, str]:
    """
    MapWithMyGene(function): Map genes to Entrez IDs using MyGene.

    Parameters
    ----------
    genes : Sequence[str]
        List of gene identifiers.
    options : MappingOptions
        Configuration object.

    Returns
    -------
    Dict[str, str]
        Mapping from gene to Entrez ID.
    """

    if not genes:
        return {}

    mg = mygene.MyGeneInfo()
    mapping: Dict[str, str] = {}

    blocks = list(_iter_chunks(list(genes), options.chunk_size))

    with ThreadPoolExecutor(max_workers=max(1, int(options.n_threads))) as ex:
        futures = [ex.submit(_query_mygene_chunk, mg, block, options) for block in blocks]

        frames = []
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

    for idx, row in merged.iterrows():
        ent = row.get("entrezgene", None)
        mn = _min_entrez_str(ent)
        if mn is not None:
            mapping[str(idx)] = mn

    return mapping


######### Main Function #########
def convert_to_entrez_id(
    symbol_list: Sequence[str],
    options: MappingOptions = MappingOptions()
) -> List[str]:
    """
    ConvertToEntrezID(function): Convert gene symbols to Entrez IDs.

    Parameters
    ----------
    symbol_list : Sequence[str]
        Input gene symbols.
    options : MappingOptions
        Configuration object.

    Returns
    -------
    List[str]
        List of Entrez IDs aligned with input order.
    """

    if not isinstance(symbol_list, (list, tuple)) or len(symbol_list) == 0:
        raise ValueError("symbol_list must be a non-empty list/tuple of strings.")

    genes = [str(g) for g in symbol_list]
    conversion_dict: Dict[str, str] = {}

    gp_map = _map_with_gprofiler(genes, options)
    conversion_dict.update(gp_map)

    unmapped = [g for g in genes if g not in gp_map]

    if unmapped:
        mg_map = _map_with_mygene(unmapped, options)
        conversion_dict.update(mg_map)

    return [conversion_dict.get(g, options.na_value) for g in genes]
