"""
Refactored reproducible example pipeline for gclusters_characterization.

This script is intended as an executable demonstration of the main package
components, not as a unit test suite. It improves readability, portability,
error handling, and output organization compared to the original monolithic
script.
"""

# Package imports.
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Optional
import logging
import os
import numpy as np
import pandas as pd

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

# Package imports.
import gclusters_characterization.utils.read_solution as RD
import gclusters_characterization.utils.actions       as AC

import gclusters_characterization.clustering.consensus_matrix       as CM
import gclusters_characterization.clustering.he_clustering          as HC
import gclusters_characterization.clustering.jaccard_values         as JV
import gclusters_characterization.clustering.rand_values            as RV
import gclusters_characterization.clustering.cspa_method            as CSPA
import gclusters_characterization.clustering.plurarity_voting       as PV
import gclusters_characterization.clustering.solutioncluster_matrix as SCM
import gclusters_characterization.clustering.similarity_threshold   as ST

import gclusters_characterization.go.mapping_entrez  as ME
import gclusters_characterization.go.go_enrishment   as GOeP
import gclusters_characterization.go.go_utils        as Gutils
import gclusters_characterization.go.gene_similarity as GS

import gclusters_characterization.visualization.heatmaps            as Heat
import gclusters_characterization.visualization.go_plots            as Gplot
import gclusters_characterization.visualization.go_network          as Gnet
import gclusters_characterization.visualization.go_heiracialNetwork as GHnet
import gclusters_characterization.visualization.raincloud           as RC
import gclusters_characterization.visualization.cir_go              as GCD

LOGGER = logging.getLogger("gclusters_example")

# --------------------------------------------------------------
# CONFIGURATION CLASSES.
# --------------------------------------------------------------

@dataclass(frozen=True)
class PipelinePaths:
    """Centralized filesystem layout used by the example pipeline."""

    base_dir: Path
    input_file: Path
    output_dir: Path
    resources_dir: Path

    @property
    def output_file_dir(self) -> Path:
        """Return the default subdirectory used for outputs."""
        return self.output_dir / "File_3"

    @classmethod
    def build(cls, input_filename: str = "archivo_prueba_3_25_133.csv") -> "PipelinePaths":
        """Build default paths relative to the script location."""
        base_dir = Path(__file__).resolve().parent
        return cls(
            base_dir=base_dir,
            input_file=base_dir / "examples_tests_files" / input_filename,
            output_dir=base_dir / "Results",
            resources_dir=base_dir / "resources",
        )

@dataclass(frozen=True)
class PipelineConfig:
    """Execution options for the demo pipeline."""

    cspa_clusters: int = 4
    embedding_components: int = 4
    parallel_scm: bool = True
    max_workers: int = 6
    gmm_components: int = 4
    go_species_key: str = "tair"
    go_hierarchy_ontology: str = "BP"
    min_genes_per_term: int = 50
    verbose: bool = True

@dataclass
class PipelineState:
    """Mutable state container shared across pipeline stages."""

    matrix: Optional[np.ndarray]                = None
    genes: Optional[list[str]]                  = None
    prop_matrix: Optional[np.ndarray]           = None
    dist_matrix: Optional[np.ndarray]           = None
    consensus_arrays: Optional[np.ndarray]      = None
    jaccard_matrix: Optional[np.ndarray]        = None
    rand_matrix: Optional[np.ndarray]           = None
    adjusted_rand_matrix: Optional[np.ndarray]  = None
    solution_cluster_matrix: Any                = None
    jaccard_equivalents_df: Any                 = None
    rand_equivalents_df: Any                    = None
    adjusted_rand_equivalents_df: Any           = None
    entrez_ids: Optional[list[Any]]             = None
    gene_symbols: Optional[list[str]]           = None
    wang_gene_matrix: Optional[np.ndarray]      = None
    wang_solution_matrix: Optional[np.ndarray]  = None
    wang_enriched_df: Any                       = None


# --------------------------------------------------------------
# ENVIROMENT FUNCTIONS.
# --------------------------------------------------------------

def configure_logging(verbose: bool = True) -> None:
    """Configure logging for the example run."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def ensure_directories(paths: PipelinePaths) -> None:
    """Create output and resource directories if needed."""
    paths.output_file_dir.mkdir(parents=True, exist_ok=True)
    paths.resources_dir.mkdir(parents=True, exist_ok=True)

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
        LOGGER.warning("Symbol %s not found; replacement skipped.", old_value)


# --------------------------------------------------------------
# PIPELINE EXECUTIONS (UTILS).
# --------------------------------------------------------------

def save_matrix(matrix: np.ndarray, filepath: Path) -> None:
    """Save a matrix using package-native save options."""
    options = AC.MatrixSaveOptions(
        mode=AC.MatrixSaveMode.TEXT_CSV,
        verbose=True,
        fmt="%.2f",
        delimiter=","
    )
    AC.save_matrix(matrix, str(filepath), options)

def load_input_solutions(paths: PipelinePaths, state: PipelineState) -> None:
    """Load the solution matrix and ordered gene list from the input file."""
    if not paths.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {paths.input_file}")
    

    matrix, genes = timed_step("Read solutions file", RD.read_solutions_file, str(paths.input_file))
    state.matrix = matrix
    state.genes = list(genes)

    LOGGER.info("Loaded matrix shape: %s", getattr(matrix, "shape", None))
    LOGGER.info("Loaded %d genes.", len(state.genes))



# --------------------------------------------------------------
# PIPELINE EXECUTIONS (STEPS).
# --------------------------------------------------------------

def compute_consensus(paths: PipelinePaths, state: PipelineState) -> None:
    """Compute and save consensus and distance matrices."""
    if state.matrix is None:
        raise RuntimeError("Matrix must be loaded before computing consensus.")

    prop_matrix, dist_matrix = timed_step("Consensus matrix", CM.consensus_matrix, state.matrix)
    state.prop_matrix = prop_matrix
    state.dist_matrix = dist_matrix

    save_matrix(prop_matrix, paths.output_file_dir / "Prop_matrix.csv")
    save_matrix(dist_matrix, paths.output_file_dir / "Dist_matrix.csv")

def run_consensus_clustering(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Run clustering/ensembling methods and append consensus solutions."""
    if state.dist_matrix is None or state.prop_matrix is None or state.matrix is None or state.genes is None:
        raise RuntimeError("Consensus inputs are not ready.")

    # Options for heiracial clustering.
    options_heiracial = {
        "options_single":       HC.ClusteringOptions(num_groups=4, method="single",   sym_tol=1e-6),
        "options_complete":     HC.ClusteringOptions(num_groups=4, method="complete", sym_tol=1e-6),
        "options_average":      HC.ClusteringOptions(num_groups=4, method="average",  sym_tol=1e-6),
        "options_weighted":     HC.ClusteringOptions(num_groups=4, method="weighted", sym_tol=1e-6),
        "options_centroid":     HC.ClusteringOptions(num_groups=4, method="centroid", sym_tol=1e-6),
        "options_median":       HC.ClusteringOptions(num_groups=4, method="median",   sym_tol=1e-6),
        "options_ward":         HC.ClusteringOptions(num_groups=4, method="ward",     sym_tol=1e-6)
    }

    # Options for CSPA Method.
    options_cspa = {
        "options_kmeans":       CSPA.CSPAOptions(n_clusters=cfg.cspa_clusters, assign_labels="kmeans"),
        "options_discretize":   CSPA.CSPAOptions(n_clusters=cfg.cspa_clusters, assign_labels="discretize"),
        "options_cluster_qr":   CSPA.CSPAOptions(n_clusters=cfg.cspa_clusters, assign_labels="cluster_qr")
    }

    state.consensus_arrays = []
    for key, option in options_heiracial.items():
        hierarchical_solution = timed_step(
            "Hierarchical clustering",
            HC.he_clustering,
            state.dist_matrix,
            state.genes,
            option,
            save_html_to=str(paths.output_file_dir / "dendograms" / f"Dendogram_file_3_{key}.html")
        )
        state.consensus_arrays.append(hierarchical_solution)

    embed_options = CSPA.EmbedOptions(n_components=cfg.embedding_components)
    for key, option in options_cspa.items():
        cspa_solution, _, _ = timed_step(
            "CSPA consensus",
            CSPA.cspa_method,
            state.prop_matrix,
            state.genes,
            cspa=option,
            embed=embed_options,
            save_html_to=str(paths.output_file_dir / "enbedding" / f"Essem_CSPA_file_3_{key}.html"),
        )
        state.consensus_arrays.append(cspa_solution)

    plurality_solution, _ = timed_step(
        "Plurality voting consensus",
        PV.plurality_voting,
        state.matrix,
        state.genes,
        plot_confidence=True,
        save_plot_to=str(paths.output_file_dir / "Plurarity Voting" / "Essem_PV.html"),
    )
    state.consensus_arrays.append(plurality_solution)

    # Default method in previous work.
    state.matrix = np.vstack([state.matrix, state.consensus_arrays[0]])

    # Save dataframe with functions.
    state.consensus_arrays = np.stack(state.consensus_arrays, axis=0)
    consensus = pd.DataFrame(data=state.consensus_arrays, columns=state.genes)
    AC.save_dataframe(consensus, filepath=paths.output_file_dir / "Dataframe_Consensus_Solutions.csv")

def compute_solution_metrics(paths: PipelinePaths, state: PipelineState) -> None:
    """Compute solution-level similarity matrices and persist them."""
    if state.matrix is None:
        raise RuntimeError("Matrix must be available before computing metrics.")

    state.jaccard_matrix =       timed_step("Jaccard between solutions",             JV.jaccard_index_solutions,       state.matrix)
    state.rand_matrix =          timed_step("Rand index between solutions",          RV.rand_index_solutions,          state.matrix)
    state.adjusted_rand_matrix = timed_step("Adjusted Rand index between solutions", RV.adjusted_rand_index_solutions, state.matrix)

    save_matrix(state.jaccard_matrix,       paths.output_file_dir / "Solutions_metrics" / "jacca_matrix.csv")
    save_matrix(state.rand_matrix,          paths.output_file_dir / "Solutions_metrics" / "rand_matrix.csv")
    save_matrix(state.adjusted_rand_matrix, paths.output_file_dir / "Solutions_metrics" / "adj_rand_matrix.csv")

def compute_cluster_equivalences(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Build solution-cluster representation and equivalent-cluster tables."""
    if state.matrix is None or state.genes is None:
        raise RuntimeError("Matrix and genes are required to compute cluster equivalences.")

    state.solution_cluster_matrix = timed_step(
        "Solution-cluster matrix",
        SCM.solution_cluster_matrix,
        state.matrix,
        state.genes,
        parallel=cfg.parallel_scm,
        max_workers=cfg.max_workers,
    )

    state.jaccard_equivalents_df = timed_step(
        "Equivalent clusters by Jaccard",
        JV.find_equivalent_clusters_jaccard,
        state.solution_cluster_matrix,
    )
    AC.save_dataframe(state.jaccard_equivalents_df, str(paths.output_file_dir / "Solutions_metrics" / "jacc_Equivalentes.csv"))

    state.rand_equivalents_df = timed_step(
        "Equivalent clusters by Rand",
        RV.find_equivalent_clusters_rand,
        state.solution_cluster_matrix,
    )
    AC.save_dataframe(state.rand_equivalents_df, str(paths.output_file_dir / "Solutions_metrics" / "Rand_Equivalentes.csv"))

    state.adjusted_rand_equivalents_df = timed_step(
        "Equivalent clusters by Adjusted Rand",
        RV.find_equivalent_clusters_rand,
        state.solution_cluster_matrix,
        metric="adjusted_rand",
    )
    AC.save_dataframe(
        state.adjusted_rand_equivalents_df,
        str(paths.output_file_dir / "Solutions_metrics" / "A_Rand_Equivalentes.csv"),
    )

def prepare_go_resources(paths: PipelinePaths, cfg: PipelineConfig) -> tuple[Path, Path]:
    """Ensure GO auxiliary resources are locally available."""
    url_obo  = "https://current.geneontology.org/ontology/go.obo"
    obo_file =       timed_step("Download obo",          Gutils.download_file,         url_obo,            dest= paths.resources_dir / "go.obo")
    gaf_path =       timed_step("Ensure GAF file",       Gutils.ensure_gaf_file,       cfg.go_species_key, out_dir= paths.resources_dir)
    gene_info_path = timed_step("Ensure gene info file", Gutils.ensure_gene_info_file, cfg.go_species_key, out_dir= paths.resources_dir)

    return Path(gaf_path), Path(gene_info_path), Path(obo_file)

def compute_go_sections(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Run GO-related enrichment and Wang similarity sections when possible."""

    if state.genes is None or state.jaccard_equivalents_df is None or state.solution_cluster_matrix is None:
        raise RuntimeError("GO sections require genes, equivalent clusters, and SCM results.")

    # Thaliana genes replacements (133 genes).
    safe_list_replace(state.genes, "AB000623", "AT2G06230")          # Dummy change - AB000623 is from Nicotiana tabacum.
    safe_list_replace(state.genes, "T04323", "PDF1.2")
    safe_list_replace(state.genes, "Y11607", "WRKY11")               # Dummy change - Y11607 is from Medicago sativa.
    safe_list_replace(state.genes, "AB003040", "AT4G31750")
    safe_list_replace(state.genes, "Z56278", "LAZ5")                 # Dummy change - Y11607 is from Vicia faba var. minor
    safe_list_replace(state.genes, "AJ000470", "GPX2")
    safe_list_replace(state.genes, "L15389", "TBL15")                # Dummy change - Not founded in National Library Medicina.
    safe_list_replace(state.genes, "AF062901", "MYB68")
    safe_list_replace(state.genes, "Z26426", "GSTF6")
    safe_list_replace(state.genes, "AF087932", "HPL1")

    options = ME.MappingOptions(organism_gp='athaliana', tax_id=3702)
    state.entrez_ids = timed_step("Convert genes to Entrez ID", ME.convert_to_entrez_id, state.genes, options= options)
    go_df = timed_step("GO enrichment from Entrez IDs", GOeP.go_enrichment, state.entrez_ids)
    AC.save_dataframe(go_df, str(paths.output_file_dir / "Go_Enrichment_Results" /"Enrichment_Example_Python.csv"))


    gaf_path, gene_info_path, obo_path = prepare_go_resources(paths, cfg)
    state.gene_symbols = timed_step( "Convert Entrez IDs to gene symbols",
        Gutils.entrez_to_symbol_ncbi,
        entrez_ids=state.entrez_ids,
        gene_info_path=str(gene_info_path)
    )

    # Human symbols replacements (500 genes).
    #safe_list_replace(state.gene_symbols, "MALAT1", "URS000001C914_9606")
    #safe_list_replace(state.gene_symbols, "C1orf56", "MENT")

    # Human symbols replacements (500 genes).
    safe_list_replace(state.gene_symbols, "AT2G06230", "TBL9")
    safe_list_replace(state.gene_symbols, "AT3G20230", "uL18-L4")
    safe_list_replace(state.gene_symbols, "RABE1b", "ATRABE1B")
    safe_list_replace(state.gene_symbols, "HLECRK", "LecRK-V.5")
    safe_list_replace(state.gene_symbols, "AT1G56190", "cPGK2")
    safe_list_replace(state.gene_symbols, "AT5G60390", "EF1alpha")
    safe_list_replace(state.gene_symbols, "AT1G67750", "PLL16")
    safe_list_replace(state.gene_symbols, "AT2G45820", "Rem1.3")
    safe_list_replace(state.gene_symbols, "AT2G14850", "ADA1A")
    
    # Dummy changes due evidence.
    safe_list_replace(state.gene_symbols, "AT4G14980", "AO1")
    safe_list_replace(state.gene_symbols, "AT2G05510", "OXR2")

    # Gene similarity configurations.
    go_indexes = {
        "wang":     GS.GeneSimilarityOptions(ontology="BP", measure="wang", groupwise="bma", distance_method="auto"),
        "resnik":   GS.GeneSimilarityOptions(ontology="MF", measure="resnik" , groupwise="hausdorff", distance_method="auto"),
        "lin":      GS.GeneSimilarityOptions(ontology="CC", measure="lin" , groupwise="max", distance_method="auto"),
        "simrel": GS.GeneSimilarityOptions(ontology="BP", measure="simrel" , groupwise="bma", distance_method="auto")
    }

    list_simbols = []
    list_matrices = []
    for index, option in go_indexes.items():
        ordered_symbols, gene_matrix = timed_step(
            "Compute GO3 Wang gene similarity matrix",
            GS.compute_gene_similarity_matrix_go3,
            state.gene_symbols,
            obo_path = obo_path,
            gaf_path = gaf_path,
            go3_opts = option
        )
        list_simbols.append(list(ordered_symbols))
        list_matrices.append(gene_matrix)
        Dataframe_wang = pd.DataFrame(gene_matrix, columns=state.gene_symbols)
        AC.save_dataframe(Dataframe_wang, str(paths.output_file_dir / "Go_numeric_results" / f"{index}_index.csv"))

    state.gene_symbols = list(list_simbols[0])
    state.wang_gene_matrix = list_matrices[0]

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
    AC.save_dataframe(wang_enriched_df, str(paths.output_file_dir / "Equivalentes_con_wang.csv"))

    run_go_visualizations(paths, cfg, state)
    run_threshold_estimation(paths, cfg, state)

def run_go_visualizations(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Generate GO visual outputs for the first SCM cluster when available."""
    if state.solution_cluster_matrix is None:
        raise RuntimeError("SCM results are required for GO visualizations.")

    first_cluster_genes = list(state.solution_cluster_matrix[0][1])
    if not first_cluster_genes:
        LOGGER.warning("First cluster is empty; GO visualizations skipped.")
        return

    options = ME.MappingOptions(organism_gp='athaliana', tax_id=3702)
    options_e = GOeP.GoEnrichmentOptions(organism='athaliana')
    options_a = GOeP.AnnotationOptions(organism='athaliana')
    entrez_ids_cluster = timed_step("Convert first cluster to Entrez IDs", ME.convert_to_entrez_id, first_cluster_genes, options= options)
    go_df_cluster = timed_step("GO enrichment for first cluster", GOeP.go_enrichment, entrez_ids_cluster, options= options_e)
    gene_to_terms = timed_step("GO annotations from Entrez IDs", GOeP.annotation_from_entrez_ids, entrez_ids_cluster, options=options_a)

    term_pvalues = go_df_cluster.set_index("native")["p_value"].to_dict()

    Gplot.plot_gene_ratio(go_df_cluster, save_path=str(paths.output_file_dir / "Go_plots" / "GR.html"))
    Gplot.plot_qscore(go_df_cluster, save_path=str(paths.output_file_dir / "Go_plots" / "QS.html"))

    gaf_candidates = [
        paths.resources_dir / f"{cfg.go_species_key}.gaf",
        paths.base_dir / f"{cfg.go_species_key}.gaf",
    ]
    gaf_path = next((candidate for candidate in gaf_candidates if candidate.exists()), None)
    if gaf_path is None:
        raise FileNotFoundError(
            f"GAF file not found for visualization. Expected '{cfg.go_species_key}.gaf'."
        )

    obo_candidates = [
        paths.resources_dir / "go.obo",
        paths.base_dir / "go.obo",
    ]
    obo_path = next((candidate for candidate in obo_candidates if candidate.exists()), None)
    if obo_path is None:
        raise FileNotFoundError("GO OBO file not found for GO visualization steps.")

    network_options = Gnet.GoNetworkOptions(min_genes_per_term=cfg.min_genes_per_term)
    Gnet.plot_go_interaction_network_html(
        gene_to_terms,
        term_pvalues,
        gaf_path=str(gaf_path),
        obo_path=str(obo_path),
        options=network_options,
        save_html_to=str(paths.output_file_dir / "Go_plots" / "Net.html"),
    )

    hierarchy_options = GHnet.GoHierarchyOptions(
        ontology=cfg.go_hierarchy_ontology,
        min_genes_per_term=cfg.min_genes_per_term,
        obo_path=str(obo_path),
    )
    GHnet.plot_go_hierarchy_html(
        gene_to_terms,
        term_pvalues,
        options=hierarchy_options,
        save_html_to=str(paths.output_file_dir / "Go_plots" / "Tree.html"),
    )

    GCD.plot_cirgo(
        gene_to_terms,
        save_html_to=str(paths.output_file_dir / "Go_plots" / "go_circle.html"),
    )

def run_heatmaps(paths: PipelinePaths, state: PipelineState) -> None:
    """Generate general similarity heatmaps."""
    if state.prop_matrix is None or state.jaccard_matrix is None:
        raise RuntimeError("Consensus and Jaccard matrices are required for heatmaps.")

    Heat.plot_html_heatmap(
        state.prop_matrix,
        str(paths.output_file_dir / "Visualizations" / "Prop_matrix.html"),
        x_label="Gene",
        y_label="Gene",
        title="Gene similarity based on co-assignment proportion",
        z_label="Co-assignment proportion",
        tooltip_format="Gene_1: %{x}<br>Gene_2: %{y}<br>Proportion: %{z:.2f}",
    )

    Heat.plot_html_heatmap(
        state.jaccard_matrix,
        save_filepath=str(paths.output_file_dir / "Visualizations" / "JaccardS.html"),
        x_label="Solution",
        y_label="Solution",
        title="Jaccard similarity between solutions",
        z_label="Jaccard",
        tooltip_format="Solution_1: %{x}<br>Solution_2: %{y}<br>Jaccard: %{z:.2f}",
    )

    if state.wang_gene_matrix is not None:
        Heat.plot_html_heatmap(
            state.wang_gene_matrix,
            save_filepath=str(paths.output_file_dir / "Visualizations" / "Wang.html"),
            x_label="Gene",
            y_label="Gene",
            title="Wang similarity between genes",
            z_label="Wang",
            tooltip_format="Gene_1: %{x}<br>Gene_2: %{y}<br>Wang: %{z:.2f}",
        )

    if state.wang_solution_matrix is not None:
        Heat.plot_dual_heatmap_two_colors(
            state.jaccard_matrix,
            state.wang_solution_matrix,
            str(paths.output_file_dir / "Visualizations" / "Dual.html"),
        )

def run_threshold_estimation(paths: PipelinePaths, cfg: PipelineConfig, state: PipelineState) -> None:
    """Estimate single and combined thresholds from equivalence tables."""
    if state.jaccard_equivalents_df is None:
        raise RuntimeError("Jaccard equivalence dataframe is required for threshold estimation.")

    gmm_options = ST.GMMThresholdOptions(n_components=cfg.gmm_components)

    jaccard_threshold = ST.estimate_similarity_threshold(
        state.jaccard_equivalents_df,
        column="Jaccard Similarity",
        options=gmm_options,
        plot=True,
        save_html_to=str(paths.output_file_dir / "Visualizations" / "gmm_threshold.html"),
    )
    LOGGER.info("Estimated Jaccard threshold: %s", jaccard_threshold)

    if state.wang_enriched_df is not None:
        wang_threshold = ST.estimate_similarity_threshold(
            state.wang_enriched_df,
            column="Wang Similarity",
            options=gmm_options,
            plot=True,
            save_html_to=str(paths.output_file_dir / "Visualizations" / "gmm_threshold_wang_df.html"),
        )
        LOGGER.info("Estimated threshold over enriched dataframe: %s", wang_threshold)

        combined_thresholds = ST.estimate_similarity_threshold_combined(
            state.wang_enriched_df,
            columns=["Jaccard Similarity", "Wang Similarity"],
            options=gmm_options,
            plot=True,
            save_html_to=str(paths.output_file_dir / "Visualizations" / "gmm_threshold_combined.html"),
        )
        LOGGER.info("Estimated combined thresholds: %s", combined_thresholds)

        RC.plot_similarity_raincloud_html(
            state.wang_enriched_df,
            column="Jaccard Similarity",
            save_html_to=str(paths.output_file_dir / "similarity_raincloud.html"),
        )


# --------------------------------------------------------------
# MAIN INTERFACE FOR EXECUTION.
# --------------------------------------------------------------

def summarize_outputs(paths: PipelinePaths) -> None:
    """Log the generated files for easier review."""
    if not paths.output_file_dir.exists():
        LOGGER.warning("Output directory does not exist: %s", paths.output_file_dir)
        return

    generated_files = sorted(p.relative_to(paths.output_file_dir) for p in paths.output_file_dir.rglob("*") if p.is_file())
    LOGGER.info("Generated %d output files.", len(generated_files))
    for file_path in generated_files:
        LOGGER.info("Output: %s", file_path)


def run_pipeline(input_filename: str = "archivo_prueba_1_116_500.csv") -> PipelineState:
    """Execute the end-to-end demonstration pipeline."""
    cfg = PipelineConfig()
    paths = PipelinePaths.build(input_filename=input_filename)

    configure_logging(cfg.verbose)
    ensure_directories(paths)

    state = PipelineState()

    load_input_solutions(paths, state)
    compute_consensus(paths, state)
    run_consensus_clustering(paths, cfg, state)
    compute_solution_metrics(paths, state)
    compute_cluster_equivalences(paths, cfg, state)
    compute_go_sections(paths, cfg, state)
    run_heatmaps(paths, state)
    summarize_outputs(paths)

    return state


if __name__ == "__main__":
    requested_input = os.environ.get("GCLUSTERS_INPUT_FILE", "archivo_prueba_3_25_133.csv")
    run_pipeline(input_filename=requested_input)
