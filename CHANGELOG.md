# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-05-04

Maintenance release. No source-code changes to the package. Bumps example dependencies and CI Action versions; adds a research-program note to the README.

### Changed

- **Examples**: `requirements-examples.txt` floors raised — `transformers>=4.35` → `>=5.7.0` (#6) and `torchvision>=0.15` → `>=0.26.0` (#7). Affects users who run `examples/`; the library itself has no torch/transformers dependency.
- **CI workflow versions** (no behaviour change, all upstream-compatible):
  - `actions/upload-artifact` 4 → 7 (#1)
  - `actions/upload-pages-artifact` 3 → 4 (#3)
  - `actions/setup-python` 5 → 6 (#2)
  - `actions/checkout` 4 → 6 (#5)
  - `actions/download-artifact` 4 → 8 (#4)

### Documentation

- README now opens with a one-line note linking the package to the wider research program at [github.com/arturoornelasb](https://github.com/arturoornelasb) (P1–P4 computational substrate; P11–P13 quaternionic-logic formal framework). Reptimeline is paper P3 in that program.

[0.2.2]: https://github.com/arturoornelasb/reptimeline/compare/v0.2.1...v0.2.2

## [0.2.1] - 2026-05-04

### Fixed

- **CI**: lint and typecheck jobs were failing on master since 2026-03-31 due to two issues in `BitDiscovery._compute_mcc` (`reptimeline/discovery.py`): unused local `n = len(col_i)` (ruff F841) and `np.sqrt(...)` returning `Any` into a `-> float` function (mypy `no-any-return`). Both fixed; no behavior change. 224/224 tests pass.

### Note

- Versions in `pyproject.toml` and `__init__.py` are bumped from `0.1.1` to `0.2.1`. The `v0.2.0` git tag at `b66b1f4` (2026-04-14) was created with the source files still reporting `0.1.1`, and was never published to PyPI because of the CI failure. `0.2.0` is therefore documented as a tag-only release below; `0.2.1` is the first 0.2.x with consistent version metadata and the first 0.2.x intended for PyPI.

[0.2.1]: https://github.com/arturoornelasb/reptimeline/compare/v0.2.0...v0.2.1

## [0.2.0] - 2026-04-14

Tagged release; not published to PyPI. See note in `[0.2.1]` above.

### Added

- **Null model for connections**: `connections_null_model()` in `tracker.py` computes a permutation-based null distribution for connection counts, enabling O/E ratio testing.
- **Null baseline for discovery**: discovery reports expected counts under a permutation null, supporting false-positive-rate estimation for duals, dependencies, and triadic interactions.
- **Example**: `examples/run_null_models.py` runs both null models against reference data; results in `results/null_model_results.json`.

### Fixed

- **Paper**: connections O/E=1.00 result extended to all three backends (TriadicGPT added alongside MNIST and Pythia) — V7 null model with 72 primitives, 1000 permutations.

### Changed

- **README**: HuggingFace V8 (`triadic-gpt2-medium-v8`) and V9 (`triadic-gptneo-125m-v9`) model badges added.
- **README and paper**: companion repo DOIs added (Engine `10.5281/zenodo.18748671`, triadic-microgpt `10.5281/zenodo.19207845`, Triadic Emergent Duality repo `10.5281/zenodo.19374914`); P2 paper DOI updated to `10.5281/zenodo.19375167`.
- **`.zenodo.json`**: version bumped to `0.2.0`.

[0.2.0]: https://github.com/arturoornelasb/reptimeline/compare/v0.1.1...v0.2.0

## [0.1.1] - 2026-03-25

### Fixed

- **README**: MNIST results showed TriadicGPT numbers (9 duals, 3 phases); corrected to actual values (65 duals, 179 dependencies, 0 phase transitions)
- **README**: "6 epochs" corrected to "10 epochs, 6 checkpoints"
- **README**: Pythia steps corrected from "step 1" to "step 0"
- **README**: `CausalVerifier` API example corrected to match actual signature
- **README**: Visualization counts clarified (5 static + 4 interactive)
- **README**: Origin DOI for parent paper corrected (was pointing to reptimeline preprint)
- **CHANGELOG**: Same MNIST number correction
- **ROADMAP**: PyPI publication marked as resolved
- **Migration guide**: `TriadicExtractor` import path corrected (it's in `examples/`, not a built-in)
- **Migration guide**: `dual_threshold` default corrected from -0.7 to -0.3
- **Migration guide**: `CausalVerifier` API example corrected

[0.1.1]: https://github.com/arturoornelasb/reptimeline/compare/v0.1.0...v0.1.1

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

- MNIST Binary Autoencoder (32-bit): 100% decoder determinism, 65 dual pairs, 179 dependencies, 0 phase transitions
- Pythia-70M SAE (32K features): 8 causally selective features, 34 dual pairs, 12 checkpoints

[0.1.0]: https://github.com/arturoornelasb/reptimeline/releases/tag/v0.1.0
