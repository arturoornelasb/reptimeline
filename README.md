# reptimeline

**Track how discrete representations evolve during training. Then break the black box open.**

reptimeline detects when concepts "are born" (first become distinguishable), when they "die" (collapse), when relationships form between concept pairs, and where phase transitions occur. Then it discovers what each feature means, labels it, and proves it's causal.

Works with any discrete bottleneck: triadic bits, VQ-VAE codebooks, FSQ levels, sparse autoencoders, concept bottleneck models, or binary codes.

## Key results

Validated on two architecturally distinct backends:

### MNIST Binary Autoencoder (32-bit)

| Metric | Value |
|--------|-------|
| Decoder determinism | **100%** (decoder output fully determined by 32-bit code; n=100 swaps) |
| Dual pairs discovered | 9 anti-correlated |
| Phase transitions | 3 detected automatically |
| Lifecycle tracking | 6 epochs, 10 digit classes |

### Pythia-70M Sparse Autoencoder (32K features)

| Metric | Value |
|--------|-------|
| Causal selectivity (KL) | 8 features with finite selectivity (1.96x--98.4x, mean 26.8x L2); 8 features with zero cross-activation (SAE sparsity) |
| Features tested | **8/16** finite KL selectivity + 8 sparse-zero (see Limitations) |
| Dual pairs discovered | 34 anti-correlated |
| Lifecycle tracking | 12 checkpoints (step 1 to 143K) |

**What "26.8x selectivity" means:** for the 8 features with finite selectivity, removing a feature (zeroing its SAE activation) changes hidden-state L2 norms for labeled concepts 26.8x more than for unrelated concepts (best: 98.4x for "fast"). The other 8 features show zero cross-activation, which may reflect SAE sparsity (the feature simply never fires for non-labeled concepts) rather than proven causal selectivity. We report these groups separately. L1 ratios yield a mean of 19.5x on the same features.

<p align="center">
<img src="results/causal/sae_intervention_heatmap.png" width="800" alt="SAE causal intervention heatmap">
<br>
<em>Causal intervention on Pythia-70M SAE features. Yellow = no effect; dark red = strong effect. Features show selective causal patterns: "dog" primarily affects animals, "teacher" primarily affects social concepts.</em>
</p>

## Why this exists

Every system that quantizes neural representations into discrete codes faces the same blind spot: **you can see what the codes look like after training, but not how they got there.**

Current tools log scalar metrics (codebook utilization, perplexity, loss) to WandB/TensorBoard. These tell you *that* something changed, not *what* changed. When codebook collapse happens at step 27,500, you want to know: which codes died? Did they die together? Did a phase transition cause it? Which concepts lost their unique representations?

reptimeline answers these questions for any discrete representation system.

## What it does that nothing else does

| Capability | reptimeline | VQ-VAE tools | SAE-Track | CBM tools | RepE |
|---|---|---|---|---|---|
| Per-code lifecycle (birth/death) | Yes | No | Partial | No | No |
| Connection tracking between concepts | Yes | No | No | No | No |
| Automatic phase transition detection | Yes | No | Manual | No | No |
| 3-way interaction discovery | **Yes** | No | No | No | No |
| Bottom-up ontology discovery | **Yes** | No | No | No | No |
| Auto-labeling (3 strategies) | **Yes** | No | Manual | Manual | No |
| Causal intervention verification | **Yes** | No | No | No | No |
| Backend-agnostic | **Yes** | No | SAE-only | No | LLM-only |

*Note: This table compares tool categories. "VQ-VAE tools" refers to standard codebook monitoring (e.g., WandB utilization metrics). "SAE-Track" refers to SAE dashboards like Neuronpedia. Individual tools may offer partial overlapping capabilities not captured by a binary comparison.*

## Limitations and Known Issues

- **Prediction experiments did not improve over baseline.** Using discovered SAE features for next-token prediction produced -0.13% (embedding-based) and -4.20% (MLP-based) accuracy relative to an unmodified baseline. The causal selectivity results show features are *individually meaningful* but do not yet translate to prediction improvements.
- **Sentinel features.** 8 of 16 tested SAE features showed zero cross-activation. This could indicate perfect selectivity or simply that those SAE features are sparse enough to never fire for non-labeled concepts. We report these separately from the 8 features with measurable finite selectivity.
- **Statistical corrections.** Discovery of duals, dependencies, and triadic interactions now includes Bonferroni correction for multiple comparisons. Use `null_baseline()` to estimate expected false positive rates for your data dimensions.

## Installation

```bash
pip install reptimeline
```

Or from source:
```bash
git clone https://github.com/arturoornelasb/reptimeline.git
cd reptimeline
pip install -e ".[dev]"
```

## Quick start

### 1. Extract snapshots from your model

Implement a `RepresentationExtractor` for your backend (see `examples/`):

```python
from reptimeline.extractors.base import RepresentationExtractor
from reptimeline.core import ConceptSnapshot

class MyExtractor(RepresentationExtractor):
    def extract(self, checkpoint_path, concepts, device='cpu'):
        codes = {}
        for concept in concepts:
            codes[concept] = get_discrete_code(model, concept)  # List[int]
        return ConceptSnapshot(step=parse_step(checkpoint_path), codes=codes)

    def similarity(self, code_a, code_b):
        # Jaccard, Hamming, or domain-specific
        ...

    def shared_features(self, code_a, code_b):
        # Indices where both codes are active
        ...
```

### 2. Analyze representation evolution

```python
from reptimeline import TimelineTracker

extractor = MyExtractor()
snapshots = extractor.extract_sequence("checkpoints/", concepts)
tracker = TimelineTracker(extractor)
timeline = tracker.analyze(snapshots)
timeline.print_summary()
```

### 3. Discover what each code element means

```python
from reptimeline import BitDiscovery

discovery = BitDiscovery()
report = discovery.discover(snapshots[-1], timeline=timeline)
discovery.print_report(report)

# Auto-label with embeddings (no API needed)
from reptimeline import AutoLabeler
labeler = AutoLabeler()
labels = labeler.label_by_embedding(report, embeddings)
```

### 4. Command line

```bash
# Analyze pre-extracted snapshots
reptimeline --snapshots timeline_data.json --discover --plot
```

## Architecture

```
Your model checkpoints
        |
        v
RepresentationExtractor  (you implement this)
        |  produces ConceptSnapshot objects
        v
TimelineTracker          (backend-agnostic)
        |  births, deaths, connections, phase transitions
        v
BitDiscovery             (backend-agnostic)
        |  duals, dependencies, 3-way interactions, hierarchy
        v
AutoLabeler              (backend-agnostic)
        |  translates bits to words (embedding, contrastive, or LLM)
        v
Reconciler               (optional, needs domain overlay)
        |  compares discovered vs. expected structure
        v
Visualizations           (swimlane, phase dashboard, churn heatmap)
```

## Examples

- `examples/mnist_pipeline.py` -- Full MNIST Binary AE pipeline (train, extract, analyze, discover)
- `examples/pythia_sae_pipeline.py` -- Pythia-70M SAE pipeline (12 checkpoints, 60 concepts)
- `examples/causal_v2.py` -- Causal intervention experiments (SAE encode/modify/decode + KL measurement)
- `examples/semantic_analysis.py` -- Semantic analysis of discovered features

## Tests

```bash
pytest tests/ -v  # 26 tests, ~0.1s
```

## Requirements

- Python 3.10+
- numpy
- matplotlib
- torch (optional, only for extractors that load model checkpoints)

To run the examples (MNIST, Pythia SAE, causal experiments):
```bash
pip install -r requirements-examples.txt
```

## License

[Business Source License 1.1](LICENSE) (BUSL-1.1)

- **Free** for research, education, evaluation, development, and personal use
- **Commercial production use** requires a license -- contact arturoornelas62@gmail.com
- Converts to [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) on 2030-03-21

## Citation

```bibtex
@software{ornelas2026reptimeline,
  author = {Ornelas Brand, José Arturo},
  title = {reptimeline: Tracking Discrete Representation Evolution During Training},
  year = {2026},
  url = {https://github.com/arturoornelasb/reptimeline}
}
```

## Origin

Extracted from [triadic-microgpt](https://github.com/arturoornelasb/triadic-microgpt).
Paper: "Prime Factorization as a Neurosymbolic Bridge" (Ornelas Brand, J.A., 2026).
