######### Libraries #########
import numpy as np                                          # Efficient Math Operations.
import rpy2.robjects as robjects                            # Transport Python data to R enviroment.
from rpy2.robjects import pandas2ri                         # R dataframe into Pandas dataframe.
import rpy2.robjects.vectors as r_vectors                   # Transport Python data to R enviroment (vectors).
import plotly.express as px                                 # HTML interactive plots.
import pandas as pd                                         # Dataframes.

######### Own Libraries #########
from CoMOcG.GoEnrischment import convert_symbols_to_entrez

######### Functions #########

"""
This block contains all main functions.
"""

def plot_gene_ratio(
        df: pd.DataFrame, 
        save_path:str = 'geneRatioPlot.html'
        ) -> None:
    """
    plot_gene_ratio (function): Create a HTML that allocates the gene ratio plot.

    Parameters:
    - df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    - save_path (str): Path to save the interactive plot as an HTML file.
    """
    try:
        # Take dataframe necesary data.
        sorted_df = df.sort_values(['p.adjust', 'GeneRatio', 'Count'], ascending=[True, False, False])
        sorted_df['GeneRatio'] = sorted_df['GeneRatio'].apply(lambda x: round(eval(x), 2))
        
        # Draw plot.
        fig = px.scatter(
            sorted_df,
            x='GeneRatio',
            y='Description',
            size='Count',
            color='p.adjust',
            color_continuous_scale='viridis',
            title="Gene Ratio for GO Terms",
            labels={"GeneRatio": "Gene Ratio", "p.adjust": "Adjusted p-value"},
            hover_data={'Description': True, 'Count': True, 'p.adjust': True}
        )
        
        # Save to HTML.
        fig.write_html(save_path)
        print(f"Interactive Gene Ratio plot saved as: {save_path}")

    except Exception as e:
        print(f"Error creating interactive Gene Ratio plot: {str(e)}")

def plot_qscore(
        df: pd.DataFrame, 
        save_path:str = 'Qplot.html'
        ) -> None:
    """
    plot_qscore (function): Same as 'plot_qscore' function.
    Create a HTML that allocates the equivalent plot but interactive.

    Parameters:
    df (pd.DataFrame): DataFrame with enriched GO terms and associated data.
    save_path (str): Path to save the interactive plot as an HTML file.
    """
    try:
        df['qScore'] = -np.log10(df['p.adjust'])  # Calculate qScore.

        # Create an interactive bar chart.
        fig = px.bar(
            df.sort_values('qScore', ascending=True),
            y='Description',
            x='qScore',
            title="qScore for GO Terms",
            labels={"qScore": "Qscore"},
            color='qScore',
            color_continuous_scale='viridis'
        )
        
        # Save to HTML.
        fig.write_html(save_path)
        print(f"Interactive qScore plot saved as: {save_path}")

    except Exception as e:
        print(f"Error creating interactive qScore plot: {str(e)}")

def plot_go_interaction_network_rpy2(
        gene_list: list[str], 
        organism: str = "org.Hs.eg.db", 
        aspect: str = "BP", 
        similarity_threshold: float = 0.7, 
        p_value_cutoff: float = 0.05,
        save_path: str = None, 
        convert_ids: bool = True,
        keytype: str = "SYMBOL", 
        width: int = 10,   # pulgadas (usado en SVG)
        height: int = 8,   # pulgadas (usado en SVG)
        res: int = 150,    # solo usado en PNG
        output_type: str = "html"  # "html", "svg" o "png"
    ):
    """
    Genera una red de interacción entre términos GO usando R y rpy2.
    
    Parámetros:
    - gene_list: Lista de genes (símbolos o IDs).
    - organism: Base de datos en R (ej. "org.Hs.eg.db").
    - aspect: Tipo de ontología ('BP', 'MF', 'CC').
    - similarity_threshold: Umbral de similitud Wang.
    - save_path: Ruta para guardar (con extensión .html, .svg o .png).
    - convert_ids: Convertir a IDs de Entrez.
    - keytype: Tipo de ID de entrada.
    - width, height: Tamaño del gráfico (pulgadas).
    - res: Resolución para PNG.
    - output_type: Formato de salida: 'html', 'svg' o 'png'.
    """
    try:
        if not isinstance(gene_list, list) or len(gene_list) == 0:
            raise ValueError("Input must be a non-empty list of genes.")

        # Convertir símbolos si es necesario
        if convert_ids:
            entrez_ids = convert_symbols_to_entrez(gene_list, organism, keytype)
            if len(entrez_ids) == 0:
                raise ValueError("No se pudieron convertir los genes a IDs válidos.")
            print(f"Se convirtieron {len(gene_list)} genes a {len(entrez_ids)} IDs Entrez")
        else:
            entrez_ids = gene_list

        # Activar conversión pandas <-> R
        pandas2ri.activate()
        r_gene_list = robjects.StrVector(entrez_ids)

        # Código R embebido
        r_code = f"""
        function(gene_list, save_path, similarity_threshold, organism, aspect, width, height, res, output_type) {{
            library(clusterProfiler)
            library(enrichplot)
            library({organism})
            library(ggplot2)
            
            if (output_type == "html") {{
                library(visNetwork)
                library(htmlwidgets)
                library(dplyr)
                library(igraph)
                library(RColorBrewer)
                library(grDevices)
                library(htmltools)
            }} else if (output_type == "svg") {{
                library(svglite)
            }}

            tryCatch({{
                # Enriquecimiento GO con filtro de p-valor
                ego <- enrichGO(gene = gene_list, 
                            ont = aspect, 
                            OrgDb = get(organism),
                            keyType = "ENTREZID",
                            readable = TRUE
                        )
                
                if (nrow(ego@result) == 0) {{
                    stop("No se encontraron términos GO enriquecidos con p-valor ajustado < ", {p_value_cutoff})
                }}
                
                # Extraer términos relevantes de 'ego'
                relevant_terms <- ego$Description
                
                # Calcular similitud entre términos
                edox <- pairwise_termsim(ego)

                if (output_type == "html") {{
                
                    # Extraer datos para visNetwork
                    sim_matrix <- edox@termsim[rownames(edox@termsim) %in% relevant_terms, colnames(edox@termsim) %in% relevant_terms]
                    result_df <- edox@result[edox@result$Description %in% relevant_terms, ]
                    
                    # Definir esquema de colores para p-valores (del rojo al verde)
                    color_palette <- colorRampPalette(c("red", "yellow", "green"))(100)
                    # Escalar p-valores a índices de color
                    p_values <- result_df$p.adjust
                    p_value_range <- range(p_values)
                    color_indices <- 1 + floor(99 * (p_values - p_value_range[1]) / 
                                            max(1e-10, p_value_range[2] - p_value_range[1]))
                    node_colors <- color_palette[color_indices]
                    
                    # Calcular tamaños de nodos según recuentos
                    size_min <- 10
                    size_max <- 40
                    node_sizes <- size_min + (size_max - size_min) * (result_df$Count - min(result_df$Count)) / 
                                max(1, max(result_df$Count) - min(result_df$Count))
                    
                    # Crear nodos
                    nodes <- data.frame(
                        id = 1:nrow(result_df),
                        label = result_df$Description,
                        title = paste0(
                            "ID: ", result_df$ID, "<br>",
                            "Description: ", result_df$Description, "<br>",
                            "GeneRatio: ", result_df$GeneRatio, "<br>",
                            "Gene Count: ", result_df$Count, "<br>",
                            "p.adjust: ", signif(result_df$p.adjust, 3), "<br>",
                            "-log10(p.adjust): ", signif(-log10(result_df$p.adjust), 3)
                        ),
                        value = node_sizes,
                        shape = "dot",
                        color = node_colors,
                        borderWidth = 2
                    )
                    
                    # Crear bordes (conexiones) basadas en similitud
                    edges <- data.frame()
                    for (i in 1:nrow(sim_matrix)) {{
                        for (j in i:nrow(sim_matrix)) {{
                            if (i != j && sim_matrix[i,j] >= similarity_threshold) {{
                                edges <- rbind(
                                    edges,
                                    data.frame(
                                        from = i,
                                        to = j,
                                        width = sim_matrix[i,j] * 5,
                                        title = paste0("Similarity: ", round(sim_matrix[i,j], 3))
                                    )
                                )
                            }}
                        }}
                    }}
                    
                    # Crear leyenda de tamaño
                    size_breaks <- quantile(result_df$Count, probs = seq(0, 1, length.out = 5))
                    size_legend <- data.frame(
                        label = paste0("Count: ", round(size_breaks[1:4]), "-", round(size_breaks[2:5])),
                        shape = rep("dot", 4),
                        size = seq(size_min, size_max, length.out = 4),
                        color = rep("gray", 4)
                    )
                    
                    # Crear red visNetwork
                    network <- visNetwork(nodes, edges, width = "100%", height = "600px") %>%
                        visOptions(
                            highlightNearest = TRUE,
                            nodesIdSelection = TRUE
                        ) %>%
                        visPhysics(
                            solver = "forceAtlas2Based",
                            forceAtlas2Based = list(
                                gravitationalConstant = -100,
                                centralGravity = 0.01,
                                springLength = 150,
                                springConstant = 0.05
                            )
                        ) %>%
                        visLayout(randomSeed = 123) %>%
                        visLegend(
                            useGroups = FALSE,
                            addNodes = size_legend,
                            main = "Node Size",
                            position = "right",
                            width = 0.2
                        ) %>%
                        visInteraction(navigationButtons = TRUE)
                    
                    # Crear leyenda de color con barra de gradiente como en emapplot
                    # Generar HTML personalizado para la barra de colores
                    color_bar_html <- tags$div(
                        style = "padding: 10px; background-color: white; border: 1px solid #ddd; position: absolute; bottom: 10px; right: 10px; z-index: 999; width: 250px;",
                        tags$h3(style = "margin-top: 0; font-size: 14px; text-align: center;", "Color: -log10(p.adjust)"),
                        tags$div(
                            style = "display: flex; align-items: center;",
                            tags$div(
                                style = paste0("width: 200px; height: 20px; background: linear-gradient(to right, ", 
                                            paste(color_palette[seq(1, 100, length.out = 10)], collapse = ", "), 
                                            ");")
                            )
                        ),
                        tags$div(
                            style = "display: flex; justify-content: space-between; margin-top: 5px;",
                            tags$span(style = "font-size: 12px;", round(p_value_range[1], 2)),
                            tags$span(style = "font-size: 12px; text-align: center;", 
                                    round((p_value_range[1] + p_value_range[2])/2, 2)),
                            tags$span(style = "font-size: 12px; text-align: right;", round(p_value_range[2], 2))
                        )
                    )
                    
                    # Agregar título y barra de colores
                    network <- htmlwidgets::prependContent(network,
                        tags$div(
                            style = "text-align: center; font-weight: bold; font-size: 20px; margin-bottom: 10px;",
                            paste0("GO Term Interaction Network - ", aspect, " (p-adj < ", {p_value_cutoff}, ")")
                        )
                    )
                    
                    network <- htmlwidgets::appendContent(network, color_bar_html)
                    
                    # Guardar como HTML
                    saveWidget(network, save_path, selfcontained = FALSE)
                    
                }} else {{
                    # Para SVG y PNG, usar el método estándar
                    p <- emapplot(edox, showCategory = nrow(edox@result), node_label = "term", label_format = 100)
                    
                    if (output_type == "svg") {{
                        svglite::svglite(file = save_path, width = width, height = height)
                        print(p)
                        dev.off()
                    }} else if (output_type == "png") {{
                        png(filename = save_path, width = width * res, height = height * res, res = res)
                        print(p)
                        dev.off()
                    }} else {{
                        print(p)
                    }}
                }}
            }}, error = function(e) {{
                stop("Error en R: ", e$message)
            }})
        }}
        """

        # Crear y ejecutar función R
        r_func = robjects.r(r_code)
        r_func(
            r_gene_list,
            save_path,
            similarity_threshold,
            organism,
            aspect,
            width,
            height,
            res,
            output_type
        )

        print(f"Gráfico generado exitosamente{' y guardado en ' + save_path if save_path else ''}")

    except Exception as e:
        print(f"Error al generar la red GO: {str(e)}")

def create_go_tree_rpy2(
        df: pd.DataFrame, 
        aspect: str = 'BP', 
        max_nodes: int = 50, 
        save_path: str = None):
    """
    Genera un DAG de términos GO a partir de un DataFrame, usando rpy2 y visNetwork.

    Parámetros:
    - df: DataFrame con columnas 'ID', 'Description', 'p.adjust'.
    - aspect: ontología GO ('BP', 'MF', 'CC').
    - max_nodes: número máximo de nodos en el grafo.
    - save_path: ruta para guardar archivo HTML (si es None, se muestra en pantalla).
    """
    try:
        # Validación básica del DataFrame
        required_cols = {'ID', 'Description', 'p.adjust'}
        if df.empty or not required_cols.issubset(df.columns):
            raise ValueError("El DataFrame debe contener las columnas: 'ID', 'Description', 'p.adjust'.")

        # Preparar vectores R
        go_ids = r_vectors.StrVector(df['ID'].tolist())
        p_adjust = dict(zip(df['ID'], df['p.adjust']))
        descriptions = dict(zip(df['ID'], df['Description'].astype(str)))

        # Mapeo de aspectos
        aspect_map = {"BP": "GOBPPARENTS", "MF": "GOMFPARENTS", "CC": "GOCCPARENTS"}
        go_parents = aspect_map.get(aspect, "GOBPPARENTS")

        # Código R
        r_code = f"""
        function(go_ids, max_nodes, save_path, p_adjust_values, descriptions) {{
            library(GO.db)
            library(visNetwork)
            library(htmlwidgets)
            library(graph)
            library(RColorBrewer)
            library(GOstats)
            library(Rgraphviz)

            go_terms <- unique(go_ids)
            gograph <- GOGraph(go_terms, {go_parents})
            relevant_terms <- intersect(nodes(gograph), go_terms)
            gograph <- subGraph(relevant_terms, gograph)

            if (numNodes(gograph) > max_nodes) {{
                relevant_terms <- sample(relevant_terms, max_nodes)
                gograph <- subGraph(relevant_terms, gograph)
            }}

            # Colores discretos según p-value
            get_color <- function(p) {{
                if (is.null(p) || is.na(p)) return("#D3D3D3")  # gris si faltante
                if (p > 0.05) return("#FFFFFF")
                if (p > 5e-3) return("#F2E6D9")
                if (p > 5e-4) return("#E6CFA1")
                if (p > 5e-5) return("#F4A582")
                if (p > 5e-6) return("#D6604D")
                if (p > 5e-7) return("#B2182B")
                if (p > 5e-8) return("#A50026")
                if (p > 5e-9) return("#800026")
                return("#67001F")
            }}

            nodes <- data.frame(
                id = relevant_terms,
                label = sapply(relevant_terms, function(x) {{
                    desc <- descriptions[[x]]
                    if (is.null(desc) || is.na(desc)) {{
                        desc <- "Término desconocido"
                    }}
                    paste0("<b>", desc, "</b>\\n<i>", x, "</i>\\np.adj = ", signif(p_adjust_values[[x]], 3))
                }}),
                color = sapply(relevant_terms, function(x) get_color(p_adjust_values[[x]])),
                shape = "box",
                font = list(align = "left", multi = "html")
            )

            edges <- data.frame(
                from = unlist(sapply(relevant_terms, function(node) {{
                    neighbors <- edges(gograph)[[node]]
                    rep(node, length(neighbors))
                }})),
                to = unlist(sapply(relevant_terms, function(node) edges(gograph)[[node]])),
                width = 1,
                arrows = "to"
            )

            network <- visNetwork(nodes, edges) %>%
                visNodes(size = 30) %>%
                visEdges(smooth = TRUE) %>%
                visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE) %>%
                visPhysics(enabled = TRUE)

            if (!is.null(save_path)) {{
                saveWidget(network, save_path, selfcontained = FALSE)
            }} else {{
                print(network)
            }}

            return(relevant_terms)
        }}
        """

        r_func = robjects.r(r_code)

        # Convertir a listas R simples (como named lists)
        p_adjust_r = robjects.ListVector({k: robjects.FloatVector([v]) for k, v in p_adjust.items()})
        descriptions_r = robjects.ListVector({k: robjects.StrVector([v]) for k, v in descriptions.items()})

        used_nodes = r_func(go_ids, max_nodes, save_path, p_adjust_r, descriptions_r)

        print(f"GO DAG creado correctamente{' y guardado en ' + save_path if save_path else ''}")
        print(f"Número de términos utilizados: {len(used_nodes)}")

    except Exception as e:
        raise RuntimeError(f"Error al crear el DAG de términos GO: {str(e)}")