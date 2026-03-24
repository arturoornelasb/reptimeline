# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-24

### Added

- **Lifecycle tracking**: birth, death, and connection events for discrete code elements across training
- **Phase transition detection**: automatic discovery of training regime changes via metric discontinuities
- **Bottom-up ontology discovery** (`BitDiscovery`): duals, dependencies, 3-way interactions, hierarchical structure
- **Auto-labeling** (`AutoLabeler`): embedding-based, contrastive, and LLM-based strategies
- **Causal verification** (`CausalVerifier`): intervention testing with bootstrap CIs, permutation p-values, BH-FDR correction
- **Theory reconciliation** (`Reconciler`): compare discovered structure against domain primitives
- **Primitive overlay** (`PrimitiveOverlay`): domain-specific primitive injection and analysis
- **Built-in extractors**: `SAEExtractor`, `VQVAEExtractor`, `FSQExtractor`, plus extensible `RepresentationExtractor` ABC
- **Visualizations**: swimlane, phase dashboard, churn heatmap, layer emergence, causal heatmap (matplotlib + Plotly)
- **Export**: JSON round-trip (`save_json`/`load_json`), CSV export (events, curves, codes, stability)
- **CLI**: `reptimeline --snapshots data.json --discover --plot`
- **Statistics**: bootstrap confidence intervals, permutation tests, Benjamini-Hochberg FDR, Cohen's d
- **Full test suite**: 224 tests across 18 modules (Python 3.10--3.13)

### Validated on

- MNIST Binary Autoencoder (32-bit): 100% decoder determinism, 9 dual pairs, 3 phase transitions
- Pythia-70M SAE (32K features): 8 causally selective features, 34 dual pairs, 12 checkpoints

[0.1.0]: https://github.com/arturoornelasb/reptimeline/releases/tag/v0.1.0
