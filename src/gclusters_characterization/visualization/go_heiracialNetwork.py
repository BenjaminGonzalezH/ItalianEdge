"""
GoHierarchicalNetwork.py (refactor completo)

Objetivo:
- Construir una visualización jerárquica por niveles (DAG) de términos GO,
  similar al estilo de la imagen (cajas por capa).
- Enfocado a GO DAG (no red de similitud). Esto NO es GoNetwork.

Características:
- Separación por ontología: BP / MF / CC (namespace).
- SubDAG robusto: opción de incluir ancestros para preservar estructura.
- Layout determinista por niveles (layered layout).
- Color por -log10(p-value) con colorbar coherente.
- Hover muestra: nombre, GO id, p-value, #genes asociados.
- Click: resalta ancestros/descendientes (camino jerárquico). Doble click resetea.
- Export HTML sin "read-modify-write" (inyección JS directa).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

import json
import logging
import os
import urllib.request

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from goatools.obo_parser import GODag

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]
Ontology = Literal["BP", "MF", "CC"]


# ──────────────────────────────────────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GoHierarchyOptions:
    ontology: Ontology = "BP"
    max_terms: int = 120                       # términos "target" (antes de ancestros)
    min_genes_per_term: int = 1
    include_ancestors: bool = True             # clave para estructura tipo imagen
    max_total_nodes: int = 600                 # límite final (incluyendo ancestros)

    # Layout
    layer_gap_y: float = 1.0                   # distancia vertical entre niveles
    node_half_width: float = 0.14              # tamaño visual caja (x)
    node_half_height: float = 0.07             # tamaño visual caja (y)
    min_sep_x: float = 0.05                    # separación mínima entre cajas (misma capa)

    # Estética
    title: str = "GO Hierarchical DAG"
    cmap_colors: Tuple[str, str, str] = ("#ffffff", "#ffa500", "#ff0000")  # blanco→naranjo→rojo

    # OBO
    obo_path: PathLike = "go.obo"
    download_obo_if_missing: bool = True
    force_download_obo: bool = False

    # HTML
    include_plotlyjs: Union[Literal["cdn", "embed"], bool] = "cdn"
    full_html: bool = True
    verbose: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _log(msg: str, verbose: bool) -> None:
    logger.info(msg)
    if verbose:
        print(msg)


def download_go_obo(obo_path: PathLike = "go.obo", force_download: bool = False) -> Path:
    """
    Descarga go.obo para goatools.
    """
    p = Path(obo_path)
    url = "http://purl.obolibrary.org/obo/go/go.obo"

    if p.exists() and not force_download:
        return p

    p.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(p))
    return p


def _ensure_obo(options: GoHierarchyOptions) -> Path:
    p = Path(options.obo_path)
    if p.exists() and not options.force_download_obo:
        return p
    if options.download_obo_if_missing or options.force_download_obo:
        _log(f"[GO] Downloading OBO to: {p}", options.verbose)
        return download_go_obo(p, force_download=options.force_download_obo)
    raise FileNotFoundError(f"OBO not found at {p}. Set download_obo_if_missing=True.")


def _normalize_gene_terms_input(
    gene2terms_or_term2genes: Dict[str, Sequence[str]]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Acepta:
      A) gene -> [terms]
      B) term -> [genes]
    Retorna:
      gene2terms (gene -> terms)
      term2genes (term -> genes)
    """
    if not isinstance(gene2terms_or_term2genes, dict) or not gene2terms_or_term2genes:
        raise ValueError("Input must be a non-empty dict.")

    # Heurística: si la mayoría de keys parecen "GO:xxxxxxx", asumimos term->genes
    keys = list(gene2terms_or_term2genes.keys())
    go_like = sum(1 for k in keys if isinstance(k, str) and k.startswith("GO:"))
    assume_term2genes = go_like >= max(1, int(0.6 * len(keys)))

    gene2terms: Dict[str, List[str]] = {}
    term2genes: Dict[str, List[str]] = {}

    if assume_term2genes:
        for term, genes in gene2terms_or_term2genes.items():
            if genes is None:
                continue
            term = str(term)
            term2genes.setdefault(term, [])
            for g in genes:
                gs = str(g)
                term2genes[term].append(gs)
                gene2terms.setdefault(gs, []).append(term)
    else:
        for gene, terms in gene2terms_or_term2genes.items():
            if terms is None:
                continue
            gs = str(gene)
            gene2terms.setdefault(gs, [])
            for t in terms:
                ts = str(t)
                gene2terms[gs].append(ts)
                term2genes.setdefault(ts, []).append(gs)

    # De-dup conservando orden
    for g, ts in gene2terms.items():
        gene2terms[g] = list(dict.fromkeys(ts))
    for t, gs in term2genes.items():
        term2genes[t] = list(dict.fromkeys(gs))

    return gene2terms, term2genes


def _validate_pvalues(term_pvalues: Dict[str, float]) -> None:
    if not isinstance(term_pvalues, dict):
        raise TypeError("term_pvalues must be dict[str, float].")
    # permitimos términos faltantes: default=1.0


def _namespace_match(go_term_obj, ontology: Ontology) -> bool:
    # goatools: namespace suele ser 'biological_process', 'molecular_function', 'cellular_component'
    ns = getattr(go_term_obj, "namespace", "") or ""
    if ontology == "BP":
        return ns == "biological_process"
    if ontology == "MF":
        return ns == "molecular_function"
    if ontology == "CC":
        return ns == "cellular_component"
    return False


def _select_target_terms(
    go_dag: GODag,
    term2genes: Dict[str, List[str]],
    term_pvalues: Dict[str, float],
    options: GoHierarchyOptions,
) -> List[str]:
    """
    Selecciona términos a graficar (targets) antes de ancestros:
    - existen en OBO
    - matchean ontology
    - cumplen min_genes_per_term
    - se ordenan por p-value asc
    """
    candidates = []
    for term, genes in term2genes.items():
        if term not in go_dag:
            continue
        obj = go_dag[term]
        if not _namespace_match(obj, options.ontology):
            continue
        if len(genes) < options.min_genes_per_term:
            continue
        candidates.append(term)

    if not candidates:
        raise ValueError(f"No GO terms available after filtering for ontology={options.ontology}.")

    candidates.sort(key=lambda t: float(term_pvalues.get(t, 1.0)))
    return candidates[: int(options.max_terms)]


def _expand_with_ancestors(
    go_dag: GODag,
    terms: Sequence[str],
    options: GoHierarchyOptions,
) -> List[str]:
    """
    Incluye ancestros para mantener estructura jerárquica.
    Limita total a max_total_nodes.
    """
    keep = set(terms)

    if not options.include_ancestors:
        return list(keep)

    # Expandimos ancestros; stop si supera max_total_nodes
    stack = list(terms)
    while stack and len(keep) < options.max_total_nodes:
        t = stack.pop()
        if t not in go_dag:
            continue
        for parent in go_dag[t].parents:
            pid = parent.id
            if pid in keep:
                continue
            # Mantener misma ontología para el “panel” (BP/MF/CC)
            if pid in go_dag and _namespace_match(go_dag[pid], options.ontology):
                keep.add(pid)
                stack.append(pid)

            if len(keep) >= options.max_total_nodes:
                break

    return list(keep)


def _build_hierarchy_digraph(go_dag: GODag, terms_set: Sequence[str]) -> nx.DiGraph:
    """
    Construye DiGraph padre -> hijo dentro del conjunto.
    """
    terms = set(terms_set)
    G = nx.DiGraph()
    for t in terms:
        if t in go_dag:
            G.add_node(t, go_name=go_dag[t].name, namespace=getattr(go_dag[t], "namespace", ""))

    for child in list(G.nodes()):
        for parent in go_dag[child].parents:
            pid = parent.id
            if pid in terms:
                G.add_edge(pid, child)

    # Es DAG en GO; pero por robustez:
    if not nx.is_directed_acyclic_graph(G):
        # Si se diera algo raro, intentamos seguir igual (layout por BFS puede fallar)
        logger.warning("[GO] Graph is not a DAG (unexpected). Some layouts may be unstable.")

    return G


def _compute_levels_from_roots(G: nx.DiGraph) -> Dict[str, int]:
    """
    Niveles tipo imagen:
    - raíces: nodos sin padres dentro del subgrafo (in_degree == 0)
    - nivel = distancia mínima desde cualquier raíz (BFS multi-source)
    """
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
        # fallback: usar "mínimo topológico"
        roots = [next(iter(G.nodes()))]

    level = {r: 0 for r in roots}
    queue = list(roots)

    while queue:
        u = queue.pop(0)
        for v in G.successors(u):
            cand = level[u] + 1
            if (v not in level) or (cand < level[v]):
                level[v] = cand
                queue.append(v)

    # Asegurar todos:
    for n in G.nodes():
        level.setdefault(n, 0)

    return level


def _layered_layout(
    G: nx.DiGraph,
    level: Dict[str, int],
    *,
    layer_gap_y: float,
    node_half_width: float,
    min_sep_x: float,
    order_key,
) -> Dict[str, Tuple[float, float]]:
    """
    Layout determinista por capas:
    - y = -level * layer_gap_y
    - x distribuido uniformemente por capa con separación mínima

    order_key: función para ordenar nodos dentro de una capa (determinismo).
    """
    # agrupar por nivel
    layers: Dict[int, List[str]] = {}
    for n, lv in level.items():
        layers.setdefault(int(lv), []).append(n)

    pos: Dict[str, Tuple[float, float]] = {}

    max_lv = max(layers.keys()) if layers else 0

    for lv in range(0, max_lv + 1):
        nodes = layers.get(lv, [])
        if not nodes:
            continue

        nodes.sort(key=order_key)

        # separación:
        step = max(2 * node_half_width + min_sep_x, 0.01)
        # centrado:
        total_w = step * (len(nodes) - 1) if len(nodes) > 1 else 0.0
        start_x = -total_w / 2.0

        y = -lv * layer_gap_y
        for i, n in enumerate(nodes):
            x = start_x + i * step
            pos[n] = (float(x), float(y))

    return pos


def _color_mapper_from_terms(
    terms: Sequence[str],
    term_pvalues: Dict[str, float],
    colors: Tuple[str, str, str],
) -> Tuple[callable, float, float]:
    """
    Color por -log10(p).
    Retorna:
      get_color(term)->hex
      vmin, vmax (en espacio -log10)
    """
    vals = []
    for t in terms:
        p = float(term_pvalues.get(t, 1.0))
        vals.append(-np.log10(p + 1e-300))

    vmin = float(min(vals)) if vals else 0.0
    vmax = float(max(vals)) if vals else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    cmap = mcolors.LinearSegmentedColormap.from_list("pval", list(colors))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def get_color(term: str) -> str:
        p = float(term_pvalues.get(term, 1.0))
        logp = -np.log10(p + 1e-300)
        return mcolors.to_hex(cmap(norm(logp)))

    return get_color, vmin, vmax


def _build_figure(
    G: nx.DiGraph,
    pos: Dict[str, Tuple[float, float]],
    term2genes: Dict[str, List[str]],
    term_pvalues: Dict[str, float],
    options: GoHierarchyOptions,
) -> Tuple[go.Figure, Dict[str, dict]]:
    """
    Construye la figura Plotly y el graph_data para JS (ancestros/descendientes).
    """
    nodes = list(G.nodes())

    # Precompute paths sets (rápido y simple)
    graph_data: Dict[str, dict] = {}
    for n in nodes:
        graph_data[n] = {
            "ancestors": list(nx.ancestors(G, n)),
            "descendants": list(nx.descendants(G, n)),
        }

    get_color, vmin, vmax = _color_mapper_from_terms(nodes, term_pvalues, options.cmap_colors)

    # Shapes (rectángulos)
    shapes = []
    hover_x, hover_y, hover_text, customdata = [], [], [], []

    w = float(options.node_half_width)
    h = float(options.node_half_height)

    for n in nodes:
        x, y = pos[n]
        name = G.nodes[n].get("go_name", n)
        pval = float(term_pvalues.get(n, 1.0))
        genes = term2genes.get(n, [])
        n_genes = len(genes)

        shapes.append(
            dict(
                type="rect",
                x0=x - w, x1=x + w,
                y0=y - h, y1=y + h,
                line=dict(color="black", width=1),
                fillcolor=get_color(n),
                layer="below",
                name=f"rect_{n}",
            )
        )

        hover_x.append(x)
        hover_y.append(y)
        customdata.append(n)
        hover_text.append(
            f"<b>{name}</b>"
            f"<br>GO: {n}"
            f"<br>p-value: {pval:.2e}"
            f"<br>Genes: {n_genes}"
            f"<br><i>Click: highlight ancestors/descendants</i>"
        )

    # Edges como annotations flecha (manteniendo tu estilo)
    arrow_annotations = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        arrow_annotations.append(
            dict(
                ax=x0, ay=y0,
                x=x1, y=y1,
                xref="x", yref="y",
                axref="x", ayref="y",
                showarrow=True,
                arrowhead=3, arrowsize=1, arrowwidth=1.5, arrowcolor="gray",
                standoff=6,
                name=f"edge_{u}__{v}",  # separador seguro para split
            )
        )

    # Trace invisible grande para capturar click + hover
    node_trace = go.Scatter(
        x=hover_x,
        y=hover_y,
        mode="markers",
        marker=dict(size=30, color="rgba(0,0,0,0)"),
        hoverinfo="text",
        hovertext=hover_text,
        showlegend=False,
        customdata=customdata,
    )

    # Colorbar “fake” (scatter vacío con colorscale)
    colorbar_trace = go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(
            colorscale=[[0, options.cmap_colors[0]], [0.5, options.cmap_colors[1]], [1, options.cmap_colors[2]]],
            showscale=True,
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title=dict(text="-log10(p-value)"),
                tickmode="linear",
                tick0=vmin,
                dtick=(vmax - vmin) / 5,
                tickformat=".2f",
                x=1.02,
                xanchor="left",
                len=0.85,
            ),
        ),
        hoverinfo="skip",
        showlegend=False,
    )

    fig = go.Figure(data=[colorbar_trace, node_trace])

    fig.update_layout(
        title=f"{options.title} ({options.ontology})",
        shapes=shapes,
        showlegend=False,
        margin=dict(b=20, l=20, r=140, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )

    # Annotations
    fig.update_layout(annotations=tuple(fig.layout.annotations) + tuple(arrow_annotations))

    return fig, graph_data


def _inject_js(fig: go.Figure, graph_data: Dict[str, dict], options: GoHierarchyOptions) -> str:
    """
    Convierte figura a HTML e inyecta JS para highlight.
    """
    base_html = fig.to_html(include_plotlyjs=options.include_plotlyjs, full_html=options.full_html)

    # Extraer div id de plotly
    import re
    m = re.search(r'<div id="([^"]+)"', base_html)
    if not m:
        raise RuntimeError("Could not locate Plotly div id in generated HTML.")
    div_id = m.group(1)

    js = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
  var gd = document.getElementById('{div_id}');
  if (!gd) return;

  var graphData = {json.dumps(graph_data)};
  var originalShapes = null;
  var originalAnnotations = null;

  function snapshot() {{
    if (originalShapes === null) {{
      originalShapes = JSON.parse(JSON.stringify(gd.layout.shapes || []));
    }}
    if (originalAnnotations === null) {{
      originalAnnotations = JSON.parse(JSON.stringify(gd.layout.annotations || []));
    }}
  }}

  function reset() {{
    if (originalShapes && originalAnnotations) {{
      Plotly.relayout(gd, {{
        shapes: originalShapes,
        annotations: originalAnnotations
      }});
    }}
  }}

  function highlight(termId) {{
    snapshot();

    var pathNodes = new Set([termId]);
    if (graphData[termId]) {{
      (graphData[termId].ancestors || []).forEach(function(a) {{ pathNodes.add(a); }});
      (graphData[termId].descendants || []).forEach(function(d) {{ pathNodes.add(d); }});
    }}

    var newShapes = (gd.layout.shapes || []).map(function(s) {{
      if (s.name && s.name.startsWith('rect_')) {{
        var id = s.name.replace('rect_', '');
        if (pathNodes.has(id)) {{
          return Object.assign({{}}, s, {{ line: Object.assign({{}}, s.line || {{}}, {{width: 3, color: 'red'}}), opacity: 1.0 }});
        }}
        return Object.assign({{}}, s, {{ opacity: 0.25 }});
      }}
      return s;
    }});

    var newAnn = (gd.layout.annotations || []).map(function(a) {{
      if (a.name && a.name.startsWith('edge_')) {{
        var rest = a.name.replace('edge_', '');
        var parts = rest.split('__');
        var u = parts[0];
        var v = parts[1];
        if (pathNodes.has(u) && pathNodes.has(v)) {{
          return Object.assign({{}}, a, {{ arrowcolor: 'blue', arrowwidth: 3, opacity: 1.0 }});
        }}
        return Object.assign({{}}, a, {{ opacity: 0.25 }});
      }}
      return a;
    }});

    Plotly.relayout(gd, {{ shapes: newShapes, annotations: newAnn }});
  }}

  gd.on('plotly_click', function(evt) {{
    if (!evt || !evt.points || evt.points.length === 0) return;
    var termId = evt.points[0].customdata;
    if (!termId) return;
    highlight(termId);
  }});

  gd.on('plotly_doubleclick', function() {{
    reset();
  }});
}});
</script>
"""

    if "</body>" in base_html:
        return base_html.replace("</body>", js + "\n</body>")
    return base_html + "\n" + js


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def plot_go_hierarchy_html(
    gene2terms_or_term2genes: Dict[str, Sequence[str]],
    term_pvalues: Dict[str, float],
    *,
    options: GoHierarchyOptions = GoHierarchyOptions(),
    save_html_to: Optional[PathLike] = "go_hierarchy.html",
    return_fig: bool = False,
    return_html: bool = False,
) -> Union[None, go.Figure, str, Tuple[go.Figure, str]]:
    """
    Construye y exporta un DAG jerárquico por niveles para una ontología GO (BP/MF/CC).

    Parámetros
    ----------
    gene2terms_or_term2genes:
        Puede ser:
          - gene -> [GO terms]
          - GO term -> [genes]
    term_pvalues:
        dict[GO term] -> p-value. Si falta un término, se asume 1.0
    options:
        GoHierarchyOptions
    save_html_to:
        ruta para guardar HTML (None para no guardar)
    return_fig / return_html:
        modos de retorno.

    Retorna
    -------
      - None por defecto
      - fig si return_fig=True
      - html si return_html=True
      - (fig, html) si ambos True
    """
    _validate_pvalues(term_pvalues)
    _, term2genes = _normalize_gene_terms_input(gene2terms_or_term2genes)

    obo_path = _ensure_obo(options)
    _log(f"[GO] Loading OBO from: {obo_path}", options.verbose)
    go_dag = GODag(str(obo_path))

    # 1) Elegir términos target (p-value asc)
    targets = _select_target_terms(go_dag, term2genes, term_pvalues, options)

    # 2) Expandir con ancestros para no cortar jerarquía
    terms = _expand_with_ancestors(go_dag, targets, options)
    if len(terms) > options.max_total_nodes:
        terms = terms[: options.max_total_nodes]

    # 3) Construir DAG
    G = _build_hierarchy_digraph(go_dag, terms)
    if G.number_of_nodes() == 0:
        raise ValueError("Empty DAG after building. Check inputs/ontology.")

    # 4) Niveles reales desde raíces del subgrafo
    level = _compute_levels_from_roots(G)

    # 5) Layout determinista por capas
    def order_key(n: str):
        # orden estable dentro de capa: primero pvalue (asc), luego id
        return (float(term_pvalues.get(n, 1.0)), n)

    pos = _layered_layout(
        G,
        level,
        layer_gap_y=options.layer_gap_y,
        node_half_width=options.node_half_width,
        min_sep_x=options.min_sep_x,
        order_key=order_key,
    )

    # 6) Figura
    fig, graph_data = _build_figure(G, pos, term2genes, term_pvalues, options)

    # 7) HTML (con JS)
    html = None
    if save_html_to is not None or return_html:
        html = _inject_js(fig, graph_data, options)

        if save_html_to is not None:
            p = Path(save_html_to)
            if p.parent and str(p.parent) not in ("", "."):
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html, encoding="utf-8")
            _log(f"[GO] Saved interactive hierarchy HTML at: {p}", options.verbose)

    if return_fig and return_html:
        return fig, html
    if return_fig:
        return fig
    if return_html:
        return html
    return None