function(go_ids, aspect, max_nodes, save_path, p_adjust_values, descriptions) {{
  library(GO.db)
  library(GOstats)
  library(graph)
  library(Rgraphviz)
  
  # Get terms and create graph
  go_terms <- unique(go_ids)
  gograph <- GOGraph(go_terms, {go_parents})
  
  # Filter nodes
  valid_nodes <- intersect(nodes(gograph), go_terms)
  gograph <- subGraph(valid_nodes, gograph)
  
  # Limit nodes if needed
  if (numNodes(gograph) > max_nodes) {{
    valid_nodes <- sample(valid_nodes, max_nodes)
    gograph <- subGraph(valid_nodes, gograph)
  }}
  
  # Function to get significance level (1-9)
  get_significance_level <- function(p_value) {{
    if (is.na(p_value)) return(0)
    if (p_value <= 5e-10) return(9)
    if (p_value <= 5e-9) return(8)
    if (p_value <= 5e-8) return(7)
    if (p_value <= 5e-7) return(6)
    if (p_value <= 5e-6) return(5)
    if (p_value <= 5e-5) return(4)
    if (p_value <= 5e-4) return(3)
    if (p_value <= 5e-3) return(2)
    if (p_value <= 0.05) return(1)
    return(0)
  }}
  
  # Function to get color based on significance level
  get_node_color <- function(p_value) {{
    level <- get_significance_level(p_value)
    colors <- c("#FFFFFF", "#FFF7EC", "#FEE8C8", "#FDD49E", 
                "#FDBB84", "#FC8D59", "#EF6548", "#D7301F",
                "#B30000", "#7F0000")
    return(colors[level + 1])
  }}
  
  # Create node labels with GO ID and term name
  node_labels <- sapply(nodes(gograph), function(x) {{
    node_labels <- sapply(nodes(gograph), function(x) {{
      p_val <- p_adjust_values[[x]]
      desc <- descriptions[[x]]
      sig_level <- get_significance_level(p_val)
      
      # Manejar NA en descripciones
      if (is.null(desc) || is.na(desc)) {{
        desc <- "Unknown Term"
      }}
      
      # Forzar salto de línea si el texto es demasiado largo
      desc_wrapped <- paste(strwrap(desc, width=30), collapse="\\n")
      
      sprintf("%s\\n(p=%0.2e)", desc_wrapped, p_val)
    }})
  }})
  
  # Set node attributes
  nAttrs <- list()
  nAttrs$label <- node_labels
  names(nAttrs$label) <- nodes(gograph)
  
  # Set colors based on significance
  node_colors <- sapply(nodes(gograph), 
                        function(x) get_node_color(p_adjust_values[[x]]))
  names(node_colors) <- nodes(gograph)
  nAttrs$fillcolor <- node_colors
  
  # Set other attributes
  nAttrs$shape <- rep("box", length(nodes(gograph)))
  names(nAttrs$shape) <- nodes(gograph)
  nAttrs$fontsize <- rep(20, length(nodes(gograph)))
  names(nAttrs$fontsize) <- nodes(gograph)
  
  # Edge attributes
  eAttrs <- list()
  eAttrs$arrowhead <- rep("vee", length(edges(gograph)))
  names(eAttrs$arrowhead) <- edges(gograph)
  
  # Generate plot
  if (!is.null(save_path)) {{
    pdf(save_path, width=15, height=12)
  }}
  
  # Create layout with top to bottom direction
  lay <- layoutGraph(gograph, layoutType="dot", 
                     attrs=list(graph=list(rankdir="TB")))
  
  # Plot graph
  plot(lay,
       main=paste("GO", aspect, "DAG - Heat Response Terms"),
       nodeAttrs=nAttrs,
       edgeAttrs=eAttrs,
       attrs=list(
         node=list(
           shape="box",
           style="filled",
           width=3.5,  # Increased width for better text display
           height=1.2
         ),
         edge=list(
           color="black",
           dir="forward"
         )
       ))
  
  # Add legend for significance levels
  legend("bottomright", 
         legend=paste("p≤", c("0.05", "5e-3", "5e-4", "5e-5",
                              "5e-6", "5e-7", "5e-8", "5e-9", "5e-10")),
         fill=c("#FFF7EC", "#FEE8C8", "#FDD49E", "#FDBB84",
                "#FC8D59", "#EF6548", "#D7301F", "#B30000",
                "#7F0000"),
         border="black",
         title="Significance Levels")
  
  if (!is.null(save_path)) {{
    dev.off()
  }}
  
  return(valid_nodes)
}}