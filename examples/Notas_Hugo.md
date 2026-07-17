# Notas para Hugo — cambios al pipeline

Hugo, dejo este documento como un conjunto de instrucciones para que veas las modificaciones hechas al pipeline, dado que muchos elementos se modificaron o directamente se eliminaron. Por lo tanto, el pipeline tendrá un cambio importante respecto a cómo estaba planteado en un inicio.

## 1. Archivos de ejemplo

Ignóralos: los dejaré como una interfaz de pruebas de rendimiento o como scripts de referencia para quienes se interesen en el proyecto más adelante. El archivo que te interesa a ti es **`Pipeline_documented.ipynb`**.

## 2. Módulos eliminados

Se eliminaron el módulo asociado a CSPA, el uso del `DualHeatmap` y el detector de umbral por GMM, por las siguientes razones:

- **CSPA**: no logré justificar con claridad la relación entre el agrupamiento espectral y este método, por lo que es un elemento que podría considerarse incorrecto dentro del análisis si lo dejaba sin esa fundamentación.
- **DualHeatmap**: lo eliminé por ahora porque, como visualización, resultaba demasiado pesado para la librería, y no encontré una alternativa más liviana que soportara hover/tooltips interactivos equivalentes a los de Plotly. De todas formas, el estudio de soluciones con alta discrepancia entre el índice de Wang y el de Jaccard se mantiene dentro del módulo `summary` (`semantic_structural_discrepancy.identify_discrepant_solution_pairs` + `plot_discrepancy_summary`).
- **Detector de umbral por GMM**: lo eliminé buscando un enfoque más simple de explicar, sin caer en un proceso tan complejo. Su reemplazo es el corte automático por método de Otsu (`gene_overlap.compute_frequency_cutoff`, con soporte para `kneed`), que ya está integrado en el módulo de `gene_overlap`.

## 3. Nueva función de similitud semántica

Creé la función `compute_gene_similarity_matrix_by_batch`, que ya no tiene los defectos de la original (`compute_gene_similarity_matrix_go3`): calcula el índice semántico directamente y sin error. Si no puede calcular el índice semántico de un gen respecto a los demás, ese valor queda en cero dentro de la matriz, en lugar de fallar.

## 4. Cómo armar un pipeline completo a partir de `Pipeline_documented.ipynb`

El notebook tiene mucha información explicativa y elementos asociados que no son estrictamente necesarios para un pipeline productivo. Si se quiere implementar uno completo, el orden sería el siguiente:

1. **Carga y consenso (se mantiene igual que antes)**: cargar los CSV (`read_solution.read_solutions_file`), hacer el análisis cuantitativo de las soluciones y generar la solución de consenso (`consensus_matrix.consensus_matrix` + `he_clustering.he_clustering` probando los distintos métodos de linkage, seleccionando el mejor según el coeficiente de correlación cofenética).
2. **Análisis biológico**: conversión de símbolos a Entrez ID (`mapping_entrez.convert_to_entrez_id`, `go_utils.entrez_to_symbol_ncbi`) y cálculo del índice semántico entre genes junto con su heatmap (`gene_similarity.compute_gene_similarity_matrix_by_batch` + `heatmaps.plot_clustered_heatmap`).
3. **Similitud entre soluciones**: agregar un apartado u opción para generar una matriz de similitud entre soluciones que combine la matriz de Jaccard con la de Wang (por ejemplo, con la función `average_matrices` ya definida en el notebook, probando sus distintos métodos de promedio: aritmético, ponderado, geométrico y armónico).
4. **Grupos de soluciones similares**: usar el fragmento de agrupamiento dinámico (`he_inconsistency_clustering.he_inconsistency_clustering`, vía `InconsistencyClusteringOptions`) para encontrar grupos de soluciones similares, y a esos grupos aplicarles el análisis de `gene_overlap` (`compute_gene_overlap_dataframe`, `compute_gene_frequencies`, `compute_frequency_cutoff`, `summarize_genes`, correspondiente al último apartado del notebook) junto con las visualizaciones de Gene Ontology (`go_plots`, `go_network`, `go_he_network`) sobre los genes resultantes.
5. **Identificación adicional de soluciones**: por último, los otros dos códigos de identificación de soluciones (del apartado "Solutions identification"): las soluciones estadísticamente distintas a la solución de consenso (`consensus_distance.compute_consensus_distance_scores` / `identify_outlier_solutions_vs_consensus`) y aquellas soluciones que presentan discrepancia entre el índice de Wang y el de Jaccard (`semantic_structural_discrepancy.identify_discrepant_solution_pairs`).
