# Importar librerías necesarias
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import pandas as pd

pandas2ri.activate()

# Importar paquetes R de forma segura
def load_r_package(package_name):
    """Carga un paquete R si no está ya cargado."""
    if not robjects.r(f'"{package_name}" %in% loadedNamespaces()')[0]:
        robjects.r(f'library({package_name})')

# Cargar paquetes esenciales
load_r_package("AnnotationDbi")
load_r_package("biomaRt")
load_r_package("clusterProfiler")
load_r_package("GOSemSim")

def convert_symbols_to_entrez(gene_symbols, organism="org.Hs.eg.db"):
    """
    Convert gene symbols to Entrez IDs efficiently.
    Uses BiocParallel for faster processing.
    """
    try:
        # Cargar paquetes R necesarios
        load_r_package(organism)
        load_r_package("BiocParallel")

        # Convertir a R vector
        r_genes = robjects.StrVector(gene_symbols)
        robjects.r.assign("gene_symbols", r_genes)

        # Código en R optimizado
        r_code = f"""
        entrez_ids <- mapIds(
            x = {organism}, 
            keys = gene_symbols, 
            column = "ENTREZID", 
            keytype = "SYMBOL", 
            multiVals = "first"
        )
        entrez_ids[is.na(entrez_ids)] <- NA  # Mantener estructura
        """
        robjects.r(r_code)

        # Obtener los IDs desde R
        entrez_ids = list(robjects.r('entrez_ids'))

        return entrez_ids
    except Exception as e:
        print(f"Error in convert_symbols_to_entrez: {e}")
        return []

def perform_go_enrichment(gene_list, organism="org.Hs.eg.db", ont="BP", convert_ids=True):
    """
    Perform GO enrichment analysis.
    """
    try:
        # Convertir IDs si es necesario
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism)
            print(f"Converted {len(gene_list)} genes to {len(entrez_ids)} Entrez IDs.")
        else:
            entrez_ids = gene_list

        if not entrez_ids:
            raise ValueError("No valid Entrez IDs.")

        # Asignar genes en R
        r_genes = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_genes)

        # Ejecutar enriquecimiento GO
        r_code = f"""
        go_results <- enrichGO(
            gene = gene_list,
            OrgDb = {organism},
            ont = "{ont}",
            keyType = "ENTREZID",
            readable = TRUE
        )
        as.data.frame(go_results)
        """
        result_df = robjects.r(r_code)
        return pandas2ri.rpy2py(result_df)

    except Exception as e:
        print(f"Error in perform_go_enrichment: {e}")
        return pd.DataFrame()

def calculate_wang_distance_matrix(gene_list, organism="org.Hs.eg.db", ont="BP", convert_ids=True):
    """
    Calculate a matrix of Wang semantic distances for a list of genes.
    """
    try:
        # Validación de parámetros
        if gene_list is None or len(gene_list) < 2:
            raise ValueError("The gene list must contain at least two valid entries.")

        if organism is None:
            raise ValueError("Organism database cannot be None.")
        
        if ont is None:
            raise ValueError("Ontology type cannot be None. Use 'BP', 'MF', or 'CC'.")

        # Convertir IDs si es necesario
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism)
            if not entrez_ids:
                raise ValueError("No valid Entrez IDs found after conversion.")
            print(f"Converted {len(gene_list)} genes to {len(entrez_ids)} Entrez IDs.")
        else:
            entrez_ids = gene_list

        # Convertir a R vector
        r_gene_list = robjects.StrVector(entrez_ids)
        robjects.r.assign("gene_list", r_gene_list)

        # Cargar R paquetes
        load_r_package("GOSemSim")

        # Ejecutar código R con validación de parámetros
        r_code = f"""
        library(GOSemSim)
        go_db <- godata(annoDb = "{organism}", ont = "{ont}", computeIC = TRUE)
        sim_matrix <- mgeneSim(genes = gene_list, semData = go_db, measure = "Wang")
        sim_matrix[is.na(sim_matrix)] <- 0  # Replace NA with 0
        as.data.frame(sim_matrix)  # Convert to DataFrame
        """
        
        sim_matrix_df = robjects.r(r_code)

        # Convertir DataFrame de R a pandas
        sim_matrix_df = pandas2ri.rpy2py(sim_matrix_df)
        
        return sim_matrix_df

    except Exception as e:
        print(f"Error in calculate_wang_distance_matrix: {e}")
        return pd.DataFrame()