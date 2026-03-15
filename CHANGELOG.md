# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog
and this project adheres to Semantic Versioning.

---

## [Unreleased]

### Added
- GPU support for clustering computations.

### Changed
- Improved GO network layout performance.

---

## [1.0.0] - 2026-03-14

### Added
- Initial release of the GClusters Characterization toolkit.
- Ensemble clustering framework.
- Consensus matrix computation.
- Jaccard similarity index for cluster comparison.
- Rand index implementation.
- Hungarian algorithm for cluster label alignment.
- Gene Ontology enrichment utilities.
- Gene ID mapping utilities.
- GO interaction network visualization.
- GO hierarchical DAG visualization.
- GO chord diagram visualization.
- Heatmap visualization utilities.
- Raincloud plot visualization.

### Dependencies
- numpy
- pandas
- networkx
- plotly
- matplotlib
- scikit-learn
- scipy
- goatools
- gprofiler-official
- mygene
- go3