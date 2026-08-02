"""
Refactored reproducible example pipeline for biocluster.

This script is intended as an executable demonstration of the main package
components, not as a unit test suite. It improves readability, portability,
error handling, output organization, and biological identifier consistency.

Main refactoring goals
----------------------
1. Map genes to Entrez IDs and symbols from the beginning of the pipeline.
2. Move representative/disjoint-group summary analysis into its own section.
3. Separate computation, visualization, and output organization.
4. Reduce or explain the Python 3.13 + joblib/loky threading shutdown warning.
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# STANDARD LIBRARY IMPORTS
# ---------------------------------------------------------------------

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional
import logging
import multiprocessing
import os
import platform
import warnings

# ---------------------------------------------------------------------
# THIRD-PARTY IMPORTS
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# RUNTIME / ENVIRONMENT CONFIGURATION
# ---------------------------------------------------------------------

# Helps joblib/loky on Windows avoid noisy CPU detection warnings.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "6")

# Python 3.13 + loky/joblib can emit a shutdown-time dummy-thread warning.
# Using "spawn" tends to be the safest cross-platform option in examples.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # Safe when the interpreter has already initialized the context.
    pass

# This does not solve the root cause in all environments, but reduces noise.
warnings.filterwarnings(
    "ignore",
    message=".*Could not find the number of physical cores.*",
)

# ---------------------------------------------------------------------
# PACKAGE IMPORTS
# ---------------------------------------------------------------------

import biocluster.utils.read_solution as RD
import biocluster.utils.actions as AC

import biocluster.clustering.consensus_matrix as CM
import biocluster.clustering.he_clustering as HC
import biocluster.clustering.jaccard_values as JV
import biocluster.clustering.rand_values as RV
import biocluster.clustering.solutioncluster_matrix as SCM

import biocluster.go.mapping_entrez as ME
import biocluster.go.go_enrichment as GOeP
import biocluster.go.go_utils as Gutils
import biocluster.go.gene_similarity as GS

import biocluster.visualization.heatmaps as Heat
import biocluster.visualization.go_plots as Gplot
import biocluster.visualization.go_network as Gnet
import biocluster.visualization.go_hierarchical_network as GHnet

import biocluster.summary.gene_overlap as GOL

LOGGER = logging.getLogger("biocluster_example")


# ---------------------------------------------------------------------
# CONFIGURATION CLASSES
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PipelinePaths:
    """Centralized filesystem layout used by the example pipeline."""

    base_dir: Path
    input_file: Path
    output_dir: Path
    resources_dir: Path
    file_tag: str = "File_3"

    @property
    def output_file_dir(self) -> Path:
        """Return the default subdirectory used for outputs."""
        return self.output_dir / self.file_tag

    @classmethod
    def build(
        cls,
        input_filename: str = "archivo_prueba_3_25_133.csv",
        file_tag: str = "File_3",
    ) -> "PipelinePaths":
        """Build default paths relative to the script location."""
        base_dir = Path(__file__).resolve().parent
        return cls(
            base_dir=base_dir,
            input_file=base_dir / "examples_tests_files" / input_filename,
            output_dir=base_dir / "Results",
            resources_dir=base_dir / "resources",
            file_tag=file_tag,
        )


@dataclass(frozen=True)
class PipelineConfig:
    """Execution options for the demo pipeline."""

    hierarchical_clusters: int = 4
    parallel_scm: bool = True
    max_workers: int = 6

    go_species_key: str = "tair"
    go_mapping_organism: str = "athaliana"
    go_tax_id: int = 3702
    go_hierarchy_ontology: str = "BP"
    min_genes_per_term: int = 20
    summary_min_genes_per_term: int = 5

    summary_combined_threshold: float = 0.17
    summary_min_gene_frequency: int = 50

    verbose: bool = True
    explain_threading_warning: bool = True


@dataclass
class PipelineState:
    """Mutable state container shared across pipeline stages."""

    # Raw inputs
    matrix: Optional[np.ndarray] = None
    genes_raw: Optional[list[str]] = None

    # Normalized identifiers used through the pipeline
    genes_for_mapping: Optional[list[str]] = None
    entrez_ids: Optional[list[Any]] = None
    gene_symbols: Optional[list[str]] = None

    # Consensus / clustering artifacts
    prop_matrix: Optional[np.ndarray] = None
    dist_matrix: Optional[np.ndarray] = None
    consensus_arrays: Optional[np.ndarray] = None

    # Solution-level metrics
    jaccard_matrix: Optional[np.ndarray] = None
    rand_matrix: Optional[np.ndarray] = None
    adjusted_rand_matrix: Optional[np.ndarray] = None

    # Cluster-level structures
    solution_cluster_matrix: Any = None
    jaccard_equivalents_df: Any = None
    rand_equivalents_df: Any = None
    adjusted_rand_equivalents_df: Any = None

    # GO / biological similarity
    wang_gene_matrix: Optional[np.ndarray] = None
    wang_solution_matrix: Optional[np.ndarray] = None
    wang_enriched_df: Any = None
    df_gene_distance: Any = None
    gaf_path: Optional[Path] = None
    gene_info_path: Optional[Path] = None
    obo_path: Optional[Path] = None

    # Summary / representative groups section
    disjoint_genes_df: Any = None
    summary_frequency_df: Any = None
    summary_bio_sub_df: Any = None
    summary_cooc_df: Any = None
    summary_cluster_bio: Optional[np.ndarray] = None
    summary_cluster_cooc: Optional[np.ndarray] = None

    # Auxiliary notes
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------
# ENVIRONMENT FUNCTIONS
# ---------------------------------------------------------------------

def configure_logging(verbose: bool = True) -> None:
    """Configure logging for the example run."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_directories(paths: PipelinePaths) -> None:
    """Create all directories used by the pipeline."""
    directories = [
        paths.output_dir,
        paths.output_file_dir,
        paths.resources_dir,
        paths.output_file_dir / "Solutions_metrics",
        paths.output_file_dir / "Visualizations",
        paths.output_file_dir / "Go_plots",
        paths.output_file_dir / "Go_numeric_results",
        paths.output_file_dir / "Go_Enrichment_Results",
        paths.output_file_dir / "summary",
        paths.output_file_dir / "summary" / "go_group_analysis",
        paths.output_file_dir / "dendograms",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def timed_step(name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute a callable, log elapsed time, and return its result."""
    LOGGER.info("Starting: %s", name)
    start = perf_counter()
    result = func(*args, **kwargs)
    elapsed = perf_counter() - start
    LOGGER.info("Completed: %s (%.4f s)", name, elapsed)
    return result


def safe_list_replace(values: list[str], old_value: str, new_value: str) -> None:
    """Replace a value in a list only when it exists."""
    try:
        values[values.index(old_value)] = new_value
    except ValueError:
        LOGGER.warning("Value %s not found; replacement skipped.", old_value)


def save_matrix(matrix: np.ndarray, filepath: Path) -> None:
    """Save a matrix using package-native save options."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    options = AC.MatrixSaveOptions(
        mode=AC.MatrixSaveMode.TEXT_CSV,
        verbose=True,
        fmt="%.6f",
        delimiter=",",
    )
    AC.save_matrix(matrix, str(filepath), options)


def log_threading_warning_explanation(cfg: PipelineConfig) -> None:
    """Explain the shutdown-time dummy-thread warning when requested."""
    if not cfg.explain_threading_warning:
        return

    if platform.python_version().startswith("3.13"):
        LOGGER.warning(
            "If you see '_DeleteDummyThreadOnDel' warnings at interpreter shutdown, "
            "they are usually related to Python 3.13 + joblib/loky thread cleanup. "
            "They are typically emitted after the pipeline has finished and usually "
            "do not invalidate computed results."
        )


# ---------------------------------------------------------------------
# IDENTIFIER NORMALIZATION
# ---------------------------------------------------------------------

def apply_initial_gene_fixes(raw_genes: list[str]) -> list[str]:
    """
    Apply known input replacements before Entrez mapping.

    This keeps symbol/identifier normalization consistent from the beginning
    of the pipeline rather than only inside GO sections.
    """
    genes = list(raw_genes)

    # Thaliana-oriented replacements used in the original example.
    safe_list_replace(genes, "AB000623", "AT2G06230")
    safe_list_replace(genes, "T04323", "PDF1.2")
    safe_list_replace(genes, "Y11607", "WRKY11")
    safe_list_replace(genes, "AB003040", "AT4G31750")
    safe_list_replace(genes, "Z56278", "LAZ5")
    safe_list_replace(genes, "AJ000470", "GPX2")
    safe_list_replace(genes, "L15389", "TBL15")
    safe_list_replace(genes, "AF062901", "MYB68")
    safe_list_replace(genes, "Z26426", "GSTF6")
    safe_list_replace(genes, "AF087932", "HPL1")

    return genes


def apply_symbol_fixes(symbols: list[str]) -> list[str]:
    """
    Apply known output symbol replacements after Entrez -> symbol conversion.
    """
    fixed = list(symbols)

    safe_list_replace(fixed, "AT2G06230", "TBL9")
    safe_list_replace(fixed, "AT3G20230", "uL18-L4")
    safe_list_replace(fixed, "RABE1b", "ATRABE1B")
    safe_list_replace(fixed, "HLECRK", "LecRK-V.5")
    safe_list_replace(fixed, "AT1G56190", "cPGK2")
    safe_list_replace(fixed, "AT5G60390", "EF1alpha")
    safe_list_replace(fixed, "AT1G67750", "PLL16")
    safe_list_replace(fixed, "AT2G45820", "Rem1.3")
    safe_list_replace(fixed, "AT2G14850", "ADA1A")
    safe_list_replace(fixed, "AT4G14980", "AO1")
    safe_list_replace(fixed, "AT2G05510", "OXR2")

    return fixed


def prepare_go_resources(paths: PipelinePaths, cfg: PipelineConfig) -> tuple[Path, Path, Path]:
    """Ensure GO auxiliary resources are locally available."""
    url_obo = "https://current.geneontology.org/ontology/go.obo"

    obo_file = timed_step(
        "Download GO OBO",
        Gutils.download_file,
        url_obo,
        dest=paths.resources_dir / "go.obo",
    )
    gaf_path = timed_step(
        "Ensure GAF file",
        Gutils.ensure_gaf_file,
        cfg.go_species_key,
        out_dir=paths.resources_dir,
    )
    gene_info_path = timed_step(
        "Ensure gene info file",
        Gutils.ensure_gene_info_file,
        cfg.go_species_key,
        out_dir=paths.resources_dir,
    )

    return Path(gaf_path), Path(gene_info_path), Path(obo_file)


def prepare_gene_identifiers(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """
    Normalize gene identifiers from the beginning of the pipeline.

    Order:
    1. Load raw genes
    2. Apply known identifier fixes
    3. Convert to Entrez IDs
    4. Convert Entrez IDs to symbols
    5. Apply symbol-level fixes
    """
    if state.genes_raw is None:
        raise RuntimeError("Genes must be loaded before identifier normalization.")

    state.genes_for_mapping = apply_initial_gene_fixes(state.genes_raw)

    mapping_options = ME.MappingOptions(
        organism_gp=cfg.go_mapping_organism,
        tax_id=cfg.go_tax_id,
    )

    state.entrez_ids = timed_step(
        "Convert genes to Entrez IDs",
        ME.convert_to_entrez_id,
        state.genes_for_mapping,
        options=mapping_options,
    )

    gaf_path, gene_info_path, obo_path = prepare_go_resources(paths, cfg)
    state.gaf_path = gaf_path
    state.gene_info_path = gene_info_path
    state.obo_path = obo_path

    state.gene_symbols = timed_step(
        "Convert Entrez IDs to gene symbols",
        Gutils.entrez_to_symbol_ncbi,
        entrez_ids=state.entrez_ids,
        gene_info_path=str(gene_info_path),
    )
    state.gene_symbols = apply_symbol_fixes(list(state.gene_symbols))

    LOGGER.info("Normalized %d genes for downstream analyses.", len(state.gene_symbols))


# ---------------------------------------------------------------------
# PIPELINE EXECUTIONS (INPUT)
# ---------------------------------------------------------------------

def load_input_solutions(paths: PipelinePaths, state: PipelineState) -> None:
    """Load the solution matrix and ordered gene list from the input file."""
    if not paths.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {paths.input_file}")

    matrix, genes = timed_step(
        "Read solutions file",
        RD.read_solutions_file,
        str(paths.input_file),
    )

    state.matrix = matrix
    state.genes_raw = list(genes)

    LOGGER.info("Loaded matrix shape: %s", getattr(matrix, "shape", None))
    LOGGER.info("Loaded %d raw genes.", len(state.genes_raw))


# ---------------------------------------------------------------------
# PIPELINE EXECUTIONS (CORE STEPS)
# ---------------------------------------------------------------------

def compute_consensus(paths: PipelinePaths, state: PipelineState) -> None:
    """Compute and save consensus and distance matrices."""
    if state.matrix is None:
        raise RuntimeError("Matrix must be loaded before computing consensus.")

    prop_matrix, dist_matrix = timed_step(
        "Consensus matrix",
        CM.consensus_matrix,
        state.matrix,
    )
    state.prop_matrix = prop_matrix
    state.dist_matrix = dist_matrix

    save_matrix(prop_matrix, paths.output_file_dir / "Prop_matrix.csv")
    save_matrix(dist_matrix, paths.output_file_dir / "Dist_matrix.csv")


def run_consensus_clustering(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Run clustering/ensembling methods and append consensus solutions."""
    if (
        state.dist_matrix is None
        or state.prop_matrix is None
        or state.matrix is None
        or state.gene_symbols is None
    ):
        raise RuntimeError("Consensus inputs are not ready.")

    genes = state.gene_symbols

    options_hierarchical = {
        "single": HC.ClusteringOptions(num_groups=cfg.hierarchical_clusters, method="single", sym_tol=1e-6),
        "complete": HC.ClusteringOptions(num_groups=cfg.hierarchical_clusters, method="complete", sym_tol=1e-6),
        "average": HC.ClusteringOptions(num_groups=cfg.hierarchical_clusters, method="average", sym_tol=1e-6),
        "weighted": HC.ClusteringOptions(num_groups=cfg.hierarchical_clusters, method="weighted", sym_tol=1e-6),
        "centroid": HC.ClusteringOptions(num_groups=cfg.hierarchical_clusters, method="centroid", sym_tol=1e-6),
        "median": HC.ClusteringOptions(num_groups=cfg.hierarchical_clusters, method="median", sym_tol=1e-6),
        "ward": HC.ClusteringOptions(num_groups=cfg.hierarchical_clusters, method="ward", sym_tol=1e-6),
    }

    consensus_arrays: list[np.ndarray] = []

    for key, option in options_hierarchical.items():
        hierarchical_solution = timed_step(
            f"Hierarchical clustering ({key})",
            HC.he_clustering,
            state.dist_matrix,
            genes,
            option,
            save_html_to=str(paths.output_file_dir / "dendograms" / f"Dendogram_{key}.html"),
        )
        consensus_arrays.append(hierarchical_solution)

    # Preserve prior behavior: append one consensus solution to the original matrix.
    state.matrix = np.vstack([state.matrix, consensus_arrays[0]])

    state.consensus_arrays = np.stack(consensus_arrays, axis=0)
    consensus_df = pd.DataFrame(data=state.consensus_arrays, columns=genes)
    AC.save_dataframe(
        consensus_df,
        filepath=paths.output_file_dir / "Dataframe_Consensus_Solutions.csv",
    )


def compute_solution_metrics(paths: PipelinePaths, state: PipelineState) -> None:
    """Compute solution-level similarity matrices and persist them."""
    if state.matrix is None:
        raise RuntimeError("Matrix must be available before computing metrics.")

    state.jaccard_matrix = timed_step(
        "Jaccard between solutions",
        JV.jaccard_index_solutions,
        state.matrix,
    )
    state.rand_matrix = timed_step(
        "Rand index between solutions",
        RV.rand_index_solutions,
        state.matrix,
    )
    state.adjusted_rand_matrix = timed_step(
        "Adjusted Rand index between solutions",
        RV.adjusted_rand_index_solutions,
        state.matrix,
    )

    save_matrix(state.jaccard_matrix, paths.output_file_dir / "Solutions_metrics" / "jaccard_matrix.csv")
    save_matrix(state.rand_matrix, paths.output_file_dir / "Solutions_metrics" / "rand_matrix.csv")
    save_matrix(state.adjusted_rand_matrix, paths.output_file_dir / "Solutions_metrics" / "adjusted_rand_matrix.csv")


def compute_cluster_equivalences(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Build solution-cluster representation and equivalent-cluster tables."""
    if state.matrix is None or state.gene_symbols is None:
        raise RuntimeError("Matrix and gene symbols are required to compute cluster equivalences.")

    state.solution_cluster_matrix = timed_step(
        "Solution-cluster matrix",
        SCM.solution_cluster_matrix,
        state.matrix,
        state.gene_symbols,
        parallel=cfg.parallel_scm,
        max_workers=cfg.max_workers,
    )

    state.jaccard_equivalents_df = timed_step(
        "Equivalent clusters by Jaccard",
        JV.find_equivalent_clusters_jaccard,
        state.solution_cluster_matrix,
    )
    AC.save_dataframe(
        state.jaccard_equivalents_df,
        str(paths.output_file_dir / "Solutions_metrics" / "Jaccard_Equivalentes.csv"),
    )

    state.rand_equivalents_df = timed_step(
        "Equivalent clusters by Rand",
        RV.find_equivalent_clusters_rand,
        state.solution_cluster_matrix,
    )
    AC.save_dataframe(
        state.rand_equivalents_df,
        str(paths.output_file_dir / "Solutions_metrics" / "Rand_Equivalentes.csv"),
    )

    state.adjusted_rand_equivalents_df = timed_step(
        "Equivalent clusters by Adjusted Rand",
        RV.find_equivalent_clusters_rand,
        state.solution_cluster_matrix,
        metric="adjusted_rand",
    )
    AC.save_dataframe(
        state.adjusted_rand_equivalents_df,
        str(paths.output_file_dir / "Solutions_metrics" / "Adjusted_Rand_Equivalentes.csv"),
    )


# ---------------------------------------------------------------------
# GO / BIOLOGICAL SIMILARITY SECTION
# ---------------------------------------------------------------------

def compute_global_go_enrichment(paths: PipelinePaths, state: PipelineState) -> None:
    """Run GO enrichment for the full normalized gene list."""
    if state.entrez_ids is None:
        raise RuntimeError("Entrez IDs are required for GO enrichment.")

    go_df = timed_step(
        "GO enrichment from full Entrez ID set",
        GOeP.go_enrichment,
        state.entrez_ids,
    )
    AC.save_dataframe(
        go_df,
        str(paths.output_file_dir / "Go_Enrichment_Results" / "Enrichment_Full_Gene_Set.csv"),
    )


def compute_go_similarity_matrices(paths: PipelinePaths, state: PipelineState) -> None:
    """Compute GO-based gene similarity matrices."""
    if state.gene_symbols is None or state.obo_path is None or state.gaf_path is None:
        raise RuntimeError("GO resources and gene symbols are required.")

    go_indexes = {
        "wang": GS.GeneSimilarityOptions(ontology="BP", measure="wang", groupwise="bma", distance_method="auto"),
        "resnik": GS.GeneSimilarityOptions(ontology="MF", measure="resnik", groupwise="hausdorff", distance_method="auto"),
        "lin": GS.GeneSimilarityOptions(ontology="CC", measure="lin", groupwise="max", distance_method="auto"),
        "simrel": GS.GeneSimilarityOptions(ontology="BP", measure="simrel", groupwise="bma", distance_method="auto"),
    }

    list_symbols: list[list[str]] = []
    list_matrices: list[np.ndarray] = []
    distances_dfs: list[pd.DataFrame] = []

    for index, option in go_indexes.items():
        ordered_symbols, gene_matrix = timed_step(
            f"Compute GO3 {index} gene similarity matrix",
            GS.compute_gene_similarity_matrix_by_batch,
            state.gene_symbols,
            obo_path=state.obo_path,
            gaf_path=state.gaf_path,
            go3_opts=option,
        )

        ordered_symbols = list(ordered_symbols)
        list_symbols.append(ordered_symbols)
        list_matrices.append(gene_matrix)

        df_matrix = pd.DataFrame(gene_matrix, index=ordered_symbols, columns=ordered_symbols)
        distances_dfs.append(df_matrix)
        AC.save_dataframe(
            df_matrix,
            str(paths.output_file_dir / "Go_numeric_results" / f"{index}_index.csv"),
        )

    # Preserve Wang as the default matrix for downstream biological summaries.
    state.df_gene_distance = distances_dfs[0]
    state.gene_symbols = list_symbols[0]
    state.wang_gene_matrix = list_matrices[0]


def compute_solution_go_similarity(paths: PipelinePaths, state: PipelineState) -> None:
    """Compute GO solution similarity from equivalent cluster matches."""
    if (
        state.gene_symbols is None
        or state.wang_gene_matrix is None
        or state.jaccard_equivalents_df is None
        or state.solution_cluster_matrix is None
    ):
        raise RuntimeError("Wang matrix, symbols, cluster equivalences, and SCM are required.")

    wang_solution_matrix, wang_enriched_df = timed_step(
        "Compute solution GO similarity from dataframe",
        GS.solution_go_similarity_from_dataframe,
        state.gene_symbols,
        state.wang_gene_matrix,
        "Wang",
        state.jaccard_equivalents_df,
        state.solution_cluster_matrix,
        normalize_matrix=True,
    )

    state.wang_solution_matrix = wang_solution_matrix
    state.wang_enriched_df = wang_enriched_df

    AC.save_dataframe(
        wang_enriched_df,
        str(paths.output_file_dir / "Equivalentes_con_wang.csv"),
    )


def compute_go_sections(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Run GO-related enrichment and Wang similarity sections."""
    if state.gene_symbols is None:
        raise RuntimeError("Gene identifiers must be prepared before GO sections.")

    compute_global_go_enrichment(paths, state)
    compute_go_similarity_matrices(paths, state)
    compute_solution_go_similarity(paths, state)
    run_go_visualizations(paths, cfg, state)


# ---------------------------------------------------------------------
# VISUALIZATION SECTION
# ---------------------------------------------------------------------

def run_go_visualizations(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Generate GO visual outputs for the first SCM cluster when available."""
    if state.solution_cluster_matrix is None:
        raise RuntimeError("SCM results are required for GO visualizations.")

    first_cluster_genes = list(state.solution_cluster_matrix[0][1])
    if not first_cluster_genes:
        LOGGER.warning("First cluster is empty; GO visualizations skipped.")
        return

    mapping_options = ME.MappingOptions(
        organism_gp=cfg.go_mapping_organism,
        tax_id=cfg.go_tax_id,
    )
    enrichment_options = GOeP.GoEnrichmentOptions(organism=cfg.go_mapping_organism)
    annotation_options = GOeP.AnnotationOptions(organism=cfg.go_mapping_organism)

    entrez_ids_cluster = timed_step(
        "Convert first cluster to Entrez IDs",
        ME.convert_to_entrez_id,
        first_cluster_genes,
        options=mapping_options,
    )

    go_df_cluster = timed_step(
        "GO enrichment for first cluster",
        GOeP.go_enrichment,
        entrez_ids_cluster,
        options=enrichment_options,
    )
    gene_to_terms = timed_step(
        "GO annotations from first cluster Entrez IDs",
        GOeP.annotation_from_entrez_ids,
        entrez_ids_cluster,
        options=annotation_options,
    )

    AC.save_dataframe(
        go_df_cluster,
        str(paths.output_file_dir / "Go_Enrichment_Results" / "Enrichment_First_Cluster.csv"),
    )

    term_pvalues = go_df_cluster.set_index("native")["p_value"].to_dict()

    Gplot.plot_gene_ratio(
        go_df_cluster,
        save_path=str(paths.output_file_dir / "Go_plots" / "GR.html"),
    )
    Gplot.plot_qscore(
        go_df_cluster,
        save_path=str(paths.output_file_dir / "Go_plots" / "QS.html"),
    )

    if state.gaf_path is None:
        raise FileNotFoundError("GAF path not available for GO visualization.")
    if state.obo_path is None:
        raise FileNotFoundError("GO OBO path not available for GO visualization.")

    network_options = Gnet.GoNetworkOptions(min_genes_per_term=cfg.min_genes_per_term)
    Gnet.plot_go_interaction_network_html(
        gene_to_terms,
        term_pvalues,
        gaf_path=str(state.gaf_path),
        obo_path=str(state.obo_path),
        options=network_options,
        save_html_to=str(paths.output_file_dir / "Go_plots" / "Net.html"),
    )

    hierarchy_options = GHnet.GoHierarchyOptions(
        ontology=cfg.go_hierarchy_ontology,
        min_genes_per_term=cfg.min_genes_per_term,
        obo_path=str(state.obo_path),
    )
    GHnet.plot_go_hierarchy_html(
        gene_to_terms,
        term_pvalues,
        options=hierarchy_options,
        save_html_to=str(paths.output_file_dir / "Go_plots" / "Tree.html"),
    )


def run_heatmaps(paths: PipelinePaths, state: PipelineState) -> None:
    """Generate general similarity heatmaps."""
    if state.prop_matrix is None or state.jaccard_matrix is None:
        raise RuntimeError("Consensus and Jaccard matrices are required for heatmaps.")

    Heat.plot_clustered_heatmap(
        state.prop_matrix,
        labels=state.gene_symbols,
        save_filepath=str(paths.output_file_dir / "Visualizations" / "Prop_matrix.html"),
        x_label="Gene",
        y_label="Gene",
        title="Gene similarity based on co-assignment proportion",
        z_label="Co-assignment proportion",
    )

    Heat.plot_clustered_heatmap(
        state.jaccard_matrix,
        save_filepath=str(paths.output_file_dir / "Visualizations" / "JaccardS.html"),
        x_label="Solution",
        y_label="Solution",
        title="Jaccard similarity between solutions",
        z_label="Jaccard",
    )

    if state.wang_gene_matrix is not None:
        Heat.plot_clustered_heatmap(
            state.wang_gene_matrix,
            labels=state.gene_symbols,
            save_filepath=str(paths.output_file_dir / "Visualizations" / "Wang.html"),
            x_label="Gene",
            y_label="Gene",
            title="Wang similarity between genes",
            z_label="Wang",
        )


# ---------------------------------------------------------------------
# SUMMARY / REPRESENTATIVE GROUPS SECTION
# ---------------------------------------------------------------------

def compute_summary_inputs(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """
    Compute disjoint/representative group summary inputs.

    This section is intentionally separated from the main pipeline body
    so that the last summary stage does not remain mixed with the core flow.
    """
    if state.wang_enriched_df is None:
        raise RuntimeError("Wang enriched dataframe is required for summary analysis.")
    if state.matrix is None or state.gene_symbols is None or state.wang_gene_matrix is None:
        raise RuntimeError("Matrix, symbols, and Wang gene matrix are required for summary analysis.")

    matrix_help = timed_step(
        "Auxiliary solution-cluster matrix for summary",
        SCM.solution_cluster_matrix,
        state.matrix,
        state.gene_symbols,
        parallel=cfg.parallel_scm,
        max_workers=cfg.max_workers,
    )

    summary_df = state.wang_enriched_df.copy()
    summary_df["combined"] = (
        summary_df["Jaccard Similarity"] + summary_df["Wang Similarity"]
    ) / 2.0
    state.wang_enriched_df = summary_df

    # gene_overlap's compute_gene_overlap_dataframe does not filter by a
    # similarity/threshold column itself (every row of the input is
    # processed), so the combined-score cutoff is applied here first.
    filtered_pairs_df = summary_df[
        summary_df["combined"] >= cfg.summary_combined_threshold
    ].reset_index(drop=True)

    state.disjoint_genes_df = timed_step(
        "Compute disjoint genes dataframe",
        GOL.compute_gene_overlap_dataframe,
        filtered_pairs_df,
        matrix_help,
        mode="disjoint",
    )

    AC.save_dataframe(
        state.disjoint_genes_df,
        filepath=paths.output_file_dir / "summary" / "Disjoint_genes.csv",
    )

    summary_result = timed_step(
        "Summarize disjoint genes",
        GOL.summarize_genes,
        state.disjoint_genes_df,
        gene_column="Disjoint Genes",
        gene_ids=state.gene_symbols,
        gene_similarity_matrix=state.wang_gene_matrix,
        min_gene_frequency=cfg.summary_min_gene_frequency,
        label="Disjoint Genes",
    )

    freq_df = summary_result.frequency_df
    bio_sub_df = summary_result.similarity_submatrix
    cooc_df = summary_result.cooccurrence_df

    state.summary_frequency_df = freq_df
    state.summary_bio_sub_df = bio_sub_df
    state.summary_cooc_df = cooc_df

    AC.save_dataframe(
        freq_df,
        filepath=paths.output_file_dir / "summary" / "Gene_frequency.csv",
    )
    AC.save_dataframe(
        bio_sub_df,
        filepath=paths.output_file_dir / "summary" / "Biological_submatrix.csv",
    )
    AC.save_dataframe(
        cooc_df,
        filepath=paths.output_file_dir / "summary" / "Cooccurrence_matrix.csv",
    )


def run_summary_visualizations(paths: PipelinePaths, state: PipelineState) -> None:
    """Generate visualizations for the representative/disjoint groups section."""
    if state.summary_bio_sub_df is None or state.summary_cooc_df is None:
        raise RuntimeError("Summary matrices must be computed before visualization.")

    Heat.plot_clustered_heatmap(
        state.summary_bio_sub_df.to_numpy(),
        labels=state.summary_bio_sub_df.columns.to_list(),
        save_filepath=str(paths.output_file_dir / "summary" / "Biological_matrix.html"),
    )
    Heat.plot_clustered_heatmap(
        state.summary_cooc_df.to_numpy(),
        labels=state.summary_cooc_df.columns.to_list(),
        save_filepath=str(paths.output_file_dir / "summary" / "Cooccurrence_matrix.html"),
    )


def compute_summary_clusters(paths: PipelinePaths, state: PipelineState) -> None:
    """Cluster the summary matrices to study representative group structure."""
    if state.summary_bio_sub_df is None or state.summary_cooc_df is None:
        raise RuntimeError("Summary matrices are required before summary clustering.")

    bio_matrix = state.summary_bio_sub_df.to_numpy()
    cooc_matrix = state.summary_cooc_df.to_numpy()

    distance_bio = 1 - bio_matrix
    np.fill_diagonal(distance_bio, 0)

    distance_cooc = 1 - cooc_matrix
    np.fill_diagonal(distance_cooc, 0)

    state.summary_cluster_bio = timed_step(
        "Hierarchical clustering for biological summary matrix",
        HC.he_clustering,
        distance_bio,
        genes=state.summary_bio_sub_df.columns.to_list(),
        save_html_to=paths.output_file_dir / "summary" / "Dendogram_bio.html",
    )

    state.summary_cluster_cooc = timed_step(
        "Hierarchical clustering for co-occurrence summary matrix",
        HC.he_clustering,
        distance_cooc,
        genes=state.summary_cooc_df.columns.to_list(),
        save_html_to=paths.output_file_dir / "summary" / "Dendogram_cooc.html",
    )


def run_summary_go_analysis(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Run GO analysis over the representative groups section."""
    if (
        state.summary_bio_sub_df is None
        or state.summary_cluster_bio is None
        or state.summary_cluster_cooc is None
    ):
        raise RuntimeError("Summary clusters and matrices are required for GO group analysis.")

    ids = state.summary_bio_sub_df.columns.to_list()

    mapping_options = ME.MappingOptions(
        organism_gp=cfg.go_mapping_organism,
        tax_id=cfg.go_tax_id,
    )
    enrichment_options = GOeP.GoEnrichmentOptions(organism=cfg.go_mapping_organism)
    annotation_options = GOeP.AnnotationOptions(organism=cfg.go_mapping_organism)

    entrez = timed_step(
        "Convert summary group IDs to Entrez",
        ME.convert_to_entrez_id,
        ids,
        options=mapping_options,
    )

    summary_matrix = np.stack((state.summary_cluster_bio, state.summary_cluster_cooc))
    solutions_disjointive = timed_step(
        "Build solution-cluster matrix for summary groups",
        SCM.solution_cluster_matrix,
        matrix=summary_matrix,
        genes=entrez,
    )

    LOGGER.info("Summary group SCM: %s", solutions_disjointive)

    target_group = list(solutions_disjointive[1][0])

    go_df_cluster = timed_step(
        "GO enrichment for summary target group",
        GOeP.go_enrichment,
        target_group,
        options=enrichment_options,
    )
    gene_to_terms = timed_step(
        "GO annotations for summary target group",
        GOeP.annotation_from_entrez_ids,
        target_group,
        options=annotation_options,
    )

    AC.save_dataframe(
        go_df_cluster,
        filepath=paths.output_file_dir / "summary" / "go_group_analysis" / "Enrichment_summary_group.csv",
    )

    term_pvalues = go_df_cluster.set_index("native")["p_value"].to_dict()

    Gplot.plot_gene_ratio(
        go_df_cluster,
        save_path=str(paths.output_file_dir / "summary" / "go_group_analysis" / "GR.html"),
    )
    Gplot.plot_qscore(
        go_df_cluster,
        save_path=str(paths.output_file_dir / "summary" / "go_group_analysis" / "QS.html"),
    )

    if state.gaf_path is None or state.obo_path is None:
        raise FileNotFoundError("GO resources are required for summary GO visualizations.")

    network_options = Gnet.GoNetworkOptions(min_genes_per_term=cfg.summary_min_genes_per_term)
    Gnet.plot_go_interaction_network_html(
        gene_to_terms,
        term_pvalues,
        gaf_path=str(state.gaf_path),
        obo_path=str(state.obo_path),
        options=network_options,
        save_html_to=str(paths.output_file_dir / "summary" / "go_group_analysis" / "Net.html"),
    )

    hierarchy_options = GHnet.GoHierarchyOptions(
        ontology=cfg.go_hierarchy_ontology,
        min_genes_per_term=cfg.summary_min_genes_per_term,
        obo_path=str(state.obo_path),
    )
    GHnet.plot_go_hierarchy_html(
        gene_to_terms,
        term_pvalues,
        options=hierarchy_options,
        save_html_to=str(paths.output_file_dir / "summary" / "go_group_analysis" / "Tree.html"),
    )


def run_summary_section(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Run the full representative/disjoint-groups summary section."""
    compute_summary_inputs(paths, cfg, state)
    run_summary_visualizations(paths, state)
    compute_summary_clusters(paths, state)
    run_summary_go_analysis(paths, cfg, state)


# ---------------------------------------------------------------------
# OUTPUT SUMMARY
# ---------------------------------------------------------------------

def summarize_outputs(paths: PipelinePaths) -> None:
    """Log the generated files for easier review."""
    if not paths.output_file_dir.exists():
        LOGGER.warning("Output directory does not exist: %s", paths.output_file_dir)
        return

    generated_files = sorted(
        p.relative_to(paths.output_file_dir)
        for p in paths.output_file_dir.rglob("*")
        if p.is_file()
    )
    LOGGER.info("Generated %d output files.", len(generated_files))
    for file_path in generated_files:
        LOGGER.info("Output: %s", file_path)


# ---------------------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------------------

def run_pipeline(input_filename: str = "archivo_prueba_3_25_133.csv") -> PipelineState:
    """
    Execute the end-to-end demonstration pipeline.

    High-level order
    ----------------
    1. Read raw solutions
    2. Normalize gene identifiers from the beginning
    3. Compute consensus and ensemble clustering
    4. Compute structural similarities
    5. Compute GO-based similarities
    6. Generate general visualizations
    7. Run separate representative-group summary section
    """
    cfg = PipelineConfig()
    paths = PipelinePaths.build(input_filename=input_filename, file_tag="File_3")

    configure_logging(cfg.verbose)
    ensure_directories(paths)
    log_threading_warning_explanation(cfg)

    state = PipelineState()

    load_input_solutions(paths, state)
    prepare_gene_identifiers(paths, cfg, state)

    compute_consensus(paths, state)
    run_consensus_clustering(paths, cfg, state)
    compute_solution_metrics(paths, state)
    compute_cluster_equivalences(paths, cfg, state)

    compute_go_sections(paths, cfg, state)
    run_heatmaps(paths, state)

    # Fully separated representative/disjoint-group section.
    run_summary_section(paths, cfg, state)

    summarize_outputs(paths)
    return state


if __name__ == "__main__":
    requested_input = os.environ.get("BIOCLUSTER_INPUT_FILE", "archivo_prueba_3_25_133.csv")
    run_pipeline(input_filename=requested_input)