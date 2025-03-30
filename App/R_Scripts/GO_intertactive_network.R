function(gene_list, save_path, similarity_threshold, organism, aspect, width, height, res) {
  # gene_list: List with Entrez IDs.
  # save_path: Path to store the plot.
  # similarity_threshold: Minimum similarity to create an edge.
  # organism: Name of the organism annotation package (e.g., "org.Hs.eg.db").
  # aspect: Ontology type (BP, MF, CC).
  # width, height, res: Image configuration.
  
  # Load necessary libraries.
  library(clusterProfiler)
  library(enrichplot)
  
  # Dynamically load the organism database
  tryCatch({
    OrgDb <- get(organism, envir = asNamespace(organism))
  }, error = function(e) {
    stop(paste("Error: Unable to load organism database:", organism, "\nMake sure the package is installed."))
  })
  
  tryCatch({
    # Perform GO enrichment analysis.
    ego <- enrichGO(gene = gene_list, ont = aspect, OrgDb = OrgDb)
    edox <- pairwise_termsim(ego)
    
    # Save the plot if save_path is provided.
    if (!is.null(save_path)) {
      png(save_path, width = width, height = height, res = res)
    }
    
    # Plot emapplot.
    print(emapplot(edox))
    
    if (!is.null(save_path)) {
      dev.off()
    }
  }, error = function(e) {
    stop("R encountered an error: ", e$message)
  })
}