# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-07-11

### Changed
- Consolidated the clustering, visualization, and summary modules. As part of
  this consolidation, the following experimental features previously listed
  under `[1.0.0]` are not currently part of the public API and have been
  removed from this changelog's `Added` section until they are reintroduced:
  CSPA-based ensemble clustering, plurality voting, the GO chord diagram
  visualization, and the raincloud plot visualization.
- Added `pyarrow` as a required dependency (needed by the Parquet export
  path in `utils.actions`).

## [0.7.0] - 2026-03-14

### Added
- Initial release of the GClusters Characterization toolkit.
- Consensus matrix computation.
- Jaccard similarity index for cluster comparison.
- Rand index implementation.
- Hungarian algorithm for cluster label alignment.
- Gene Ontology enrichment utilities.
- Gene ID mapping utilities.
- GO interaction network visualization.
- GO hierarchical DAG visualization.
- Heatmap visualization utilities.
- Core dependencies: numpy, pandas, networkx, plotly, matplotlib, scikit-learn, scipy, goatools, gprofiler-official, mygene, go3.

---

[Unreleased]: https://github.com/BenjaminGonzalezH/ItalianEdge/releases/tag/v1.0.0
[1.0.0]: https://github.com/BenjaminGonzalezH/ItalianEdge/releases/tag/v0.7.0
