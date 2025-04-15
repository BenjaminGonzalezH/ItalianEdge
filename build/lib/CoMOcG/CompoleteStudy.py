######### Libraries #########
from concurrent.futures import ProcessPoolExecutor                  # Process Administration.
import numpy as np                                                  # Efficient Math Operations.
import os                                                           # OS callings.
import matplotlib                                                   # Graph construction.
import pandas as pd                                                 # Dataframe managment.
from ast import literal_eval                                        # Evaluate stings.
from concurrent.futures import as_completed                         # Process data.


######### Configurations #########
matplotlib.use('Agg')                               # Not GUI use, this is for conflicts between process.

######### Own Libraries #########
import CoMOcG.GoEnrischment as GOe                  # GO terms enrichment.
import CoMOcG.Go_Plots as plots                     # Gene Ontology graph functions.
import CoMOcG.Actions as Ac                         # Reiterative actions (heatmaps and saves).


######### Functions #########

def process_cluster_pair(
        row_dict: dict, 
        groups_structure: list[list[set]], 
        organism: str, 
        ont: str, 
        convert_ids: bool, 
        keytype: str, 
        directory: str):
    """
    process_cluster_pair (function): Performs GO overrepresentation analysis for a specific pair of solutions and equivalent clusters.

    This function takes a DataFrame row with pairs of solutions and their equivalent clusters, identifies genes in common
    (intersection) between the clusters, performs GO enrichment analysis on the shared genes, and saves the results
    as files (CSV, HTML, PNG, PDF) for easy viewing and analysis.

    Parameters:
    - row_dict (dict): dictionary with the keys 'Solution Pair' (tuple of solution indices) and 'Equivalent Clusters' (list of cluster pairs).
    - groups_structure (list[list[list[set]]): Structure with the clusters of each solution as gene sets.
    - organism (str): GO annotation database (default: “org.Hs.eg.db”).
    - ont (str): GO ontology to be used ('BP', 'CC' or 'MF').
    - convert_ids (bool): Indicates whether to convert identifiers (e.g. SYMBOL to ENTREZ).
    - keytype (str): Type of identifier that the genes have (for example, 'SYMBOL', 'ENTREZID').
    - directory (str): Path of the directory where the results and graphics will be saved.

    Returns:
    - str: Success, warning or error message describing the result of processing the solution pair.

    """
    try:
        # Prepare elements and check pair of solutions.
        group_i, group_j = row_dict['Solution Pair']
        equivalent_pairs = row_dict['Equivalent Clusters']
        if group_i >= len(groups_structure) or group_j >= len(groups_structure):
            return f"Skip: Invalid group indices {group_i}, {group_j}"

        # Obtain clusters of solution.
        sets_i = groups_structure[group_i]
        sets_j = groups_structure[group_j]
        for elem_i, elem_j in equivalent_pairs:
            if elem_i >= len(sets_i) or elem_j >= len(sets_j):
                continue

            # Intersecction of equivalent cluusters.
            set_i = sets_i[elem_i]
            set_j = sets_j[elem_j]
            intersection = set_i & set_j

            if not intersection:
                return f"No intersection: groups {group_i},{group_j} - clusters {elem_i},{elem_j}"

            # Go enrichment and check if dataframe is empty.
            GO_DF = GOe.perform_go_enrichment(
                list(intersection),
                organism=organism,
                ont=ont,
                convert_ids=convert_ids,
                keytype=keytype
            )
            if GO_DF is None or GO_DF.empty:
                return f"[WARNING] No valid go enrichment {group_i},{group_j} - clusters {elem_i},{elem_j}"

            # Create graphs.
            prefix = f"{directory}Solutions_{group_i},{group_j}_Clusters_{elem_i},{elem_j}"
            Ac.save_dataframe(GO_DF, f"{prefix}_Enrichment.csv")
            plots.plot_gene_ratio(GO_DF, f"{prefix}_GeneRatio.html")
            plots.plot_qscore(GO_DF, f"{prefix}_Qscore.html")
            plots.plot_go_interaction_network_rpy2(
                list(intersection),
                save_path=f"{prefix}_Network.png",
                organism=organism,
                aspect=ont,
                convert_ids=convert_ids,
                keytype=keytype
            )
            plots.create_go_tree_rpy2(
                GO_DF,
                save_path=f"{prefix}_Tree.pdf"
            )

        return f"✓ Processiong solutions {group_i},{group_j}"

    except Exception as e:
        return f"[ERROR] Fail in solutions duo {group_i},{group_j}: {e}"

def Complete_Study(
        df: pd.DataFrame,
        groups_structure: list[list[set]],
        organism: str = "org.Hs.eg.db",
        ont: str = "BP",
        convert_ids: bool = True,
        keytype: str = "SYMBOL",
        directory: str = "",
        max_process: int = 4
    ) -> None:
    """
    Complete_Study (function): Perform GO enrichment and generate plots from intersected gene sets.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with pairs of equivalent clusters.
    - groups_structure (list[list[set]]): Matrix from 'SolutionClusterMatrix'.
    - organism (str): Organism database (default: "org.Hs.eg.db").
    - ont (str): Ontology to study (BP, CC, MF).
    - convert_ids (bool): Whether to convert symbols to Entrez IDs.
    - keytype (str): Gene identifier type (e.g. SYMBOL).
    - directory (str): Directory to save generated files.
    """
    try:
        # Create directory.
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[✓] Directory created: {directory}")

        # Prepare dataframe.
        df = df.copy()
        df['Solution Pair'] = df['Solution Pair'].apply(lambda x: literal_eval(x.replace('np.float64', '')) if isinstance(x, str) else x)
        df['Equivalent Clusters'] = df['Equivalent Clusters'].apply(lambda x: literal_eval(x.replace('np.float64', '')) if isinstance(x, str) else x)

        # Arguments per row.
        rows = df.to_dict('records')
        args_list = [(row, groups_structure, organism, ont, convert_ids, keytype, directory) for row in rows]

        # Execution using designated cores.
        with ProcessPoolExecutor(max_workers=max_process) as executor:
            futures = [executor.submit(process_cluster_pair, *args) for args in args_list]

            # Converge results.
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        print(result)
                except Exception as e:
                    print(f"[ERROR] {e}")

    except Exception as e:
        print(f"[ERROR] General failure in function: {e}")
