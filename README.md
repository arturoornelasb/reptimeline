# reptimeline

**Track how discrete representations evolve during training.**

reptimeline detects when concepts "are born" (first become distinguishable), when they "die" (collapse), when relationships form between concept pairs, and where phase transitions occur in representation dynamics.

Works with any discrete bottleneck: triadic bits, VQ-VAE codebooks, FSQ levels, sparse autoencoders, concept bottleneck models, or binary codes.

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
| Theory vs. discovery reconciliation | **Yes** | No | No | Partial | No |
| Backend-agnostic | **Yes** | No | SAE-only | No | LLM-only |

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
from reptimeline import BitDiscovery, AutoLabeler

discovery = BitDiscovery()
report = discovery.discover(snapshots[-1], timeline=timeline)
discovery.print_report(report)

# Name them automatically (3 strategies: embedding, contrastive, LLM)
labeler = AutoLabeler()
labels = labeler.label_by_llm(report, llm_fn=my_api_call)
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
        |  translates bits to words
        v
Reconciler               (optional, needs domain overlay)
        |  compares discovered vs. expected structure
        v
Visualizations           (swimlane, phase dashboard, churn heatmap)
```

## Proof-of-concept results

Ran against TriadicGPT (40M params, 63 bits, 8 checkpoints):

```
REPRESENTATION TIMELINE
  Steps:              2,500 -> 50,000 (8 checkpoints)
  Concepts tracked:   53
  Code dimension:     63

  Bit births:         2,632
  Bit deaths:         1,732
  Connections formed: 1,378
  Phase transitions:  3

  Phase transitions:
    step 27,500  entropy          decrease delta=0.2626
    step  7,500  churn_rate       increase delta=1.0000
    step 27,500  utilization      decrease delta=0.1887
```

## Requirements

- Python 3.10+
- numpy
- matplotlib
- torch (optional, only for extractors that load model checkpoints)

## License

[Business Source License 1.1](LICENSE) (BUSL-1.1)

- **Free** for research, education, evaluation, development, and personal use
- **Commercial production use** requires a license -- contact arturoornelasb@gmail.com
- Converts to [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) on 2030-03-20

## Origin

Extracted from [triadic-microgpt](https://github.com/arturoornelasb/triadic-microgpt).
Paper: "Prime Factorization as a Neurosymbolic Bridge" (Ornelas Brand, 2026).
