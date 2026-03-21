# reptimeline -- Agent Configuration

## Project Identity
**reptimeline** tracks how discrete representations evolve during neural network training. Backend-agnostic: works with triadic bits, VQ-VAE, FSQ, SAE, or any discrete bottleneck.

## Repository Structure
```
reptimeline/             # Python package
  __init__.py            # Public API
  __main__.py            # python -m reptimeline
  core.py                # ConceptSnapshot, Timeline, lifecycle events
  tracker.py             # TimelineTracker: births, deaths, connections, phases
  discovery.py           # BitDiscovery: bottom-up ontology discovery
  autolabel.py           # AutoLabeler: 3 strategies to name bits
  reconcile.py           # Reconciler: compare discovered vs theory
  cli.py                 # Command-line interface
  extractors/
    base.py              # RepresentationExtractor ABC
  overlays/
    primitive_overlay.py # Domain-specific overlay (e.g., triadic primitives)
  viz/                   # 4 visualization modules
tests/                   # pytest test suite
examples/                # Reference extractor implementations
```

## Key Commands
```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest

# Run CLI
reptimeline --snapshots data.json --discover --plot
```

## License
BUSL-1.1 (Business Source License). Converts to AGPL-3.0 on 2030-03-20.
Commercial production use requires a license from the author.

## Origin
Extracted from github.com/arturoornelasb/triadic-microgpt.
Paper: "Prime Factorization as a Neurosymbolic Bridge" (Ornelas Brand, 2026).
