# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-11

### Added
- Consensus matrix computation.
- Jaccard similarity index for cluster comparison.
- Rand index implementation.
- Hierarchical clustering (manual cluster count) and automatic,
  inconsistency-coefficient-based hierarchical clustering.
- Hungarian algorithm for cluster label alignment.
- Gene Ontology enrichment utilities, including on-demand download of GAF /
  NCBI gene_info reference files per species.
- Gene ID mapping utilities.
- GO interaction network visualization.
- GO hierarchical DAG visualization.
- GO enrichment summary plots (gene ratio / q-score).
- Consensus-distance, gene-overlap, and semantic-structural discrepancy
  summary analyses across clustering solutions.
- Interactive clustered heatmap (HoloViews + Bokeh backend) with linked
  dendrograms.
- Core dependencies: numpy, pandas, networkx, plotly, matplotlib,
  scikit-learn, scipy, goatools, gprofiler-official, mygene, go3, pyarrow,
  holoviews.

### Changed
- Consolidated the clustering, visualization, and summary modules. As part
  of this consolidation, CSPA-based ensemble clustering, plurality voting,
  the GO chord diagram visualization, and the raincloud plot visualization
  were removed from the public API and are not currently planned for
  reintroduction.

### Fixed
- `plot_gene_ratio` / `plot_qscore` (`visualization.go_plots`) no longer
  crash when the input dataframe lacks an optional `source` column.

---

[1.0.0]: https://github.com/BenjaminGonzalezH/ItalianEdge/releases/tag/v1.0.0