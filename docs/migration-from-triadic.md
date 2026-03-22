# Migration Guide: triadic-microgpt → reptimeline

This guide helps you move from the inline analysis code in
[triadic-microgpt](https://github.com/arturoornelasb/triadic-microgpt)
to the standalone **reptimeline** package.

## Why migrate?

| Before (triadic-microgpt) | After (reptimeline) |
|---|---|
| Triadic bits only | Any discrete bottleneck (SAE, VQ-VAE, FSQ, triadic, custom) |
| Inline scripts in `playground/` | Installable package with CLI |
| Manual birth/death detection | Automated lifecycle tracking |
| Manual dual/dependency analysis | `BitDiscovery` with statistical tests |
| `primitivos.json` required | Unsupervised discovery + optional overlay |
| No causal testing | `CausalVerifier` with bootstrap CIs and BH-FDR |

## Import mapping

| Old (triadic-microgpt) | New (reptimeline) |
|---|---|
| `from src.triadic import PrimeMapper` | Hidden inside `TriadicExtractor` |
| `from src.triadic import prime_factors` | Hidden inside `TriadicExtractor.algebraic_similarity()` |
| `from src.evaluate import load_model` | Hidden inside `TriadicExtractor.extract()` |
| Manual `ConceptSnapshot` construction | `extractor.extract(checkpoint, concepts)` |
| Manual Hamming distance loops | `snapshot.hamming("king", "queen")` |
| Manual active-bit counting | `snapshot.active_indices("king")` |

## Architecture (4 layers)

```
1. Extractor       — checkpoint → ConceptSnapshot (backend-specific)
2. Tracker         — snapshots  → Timeline (births, deaths, connections, curves)
3. Discovery       — snapshot   → DiscoveryReport (duals, deps, triadic, hierarchy)
4. Overlay (opt.)  — timeline   → PrimitiveReport (requires primitivos.json)
```

Layers 2–3 are backend-agnostic. Layer 4 is triadic-specific.

## Step-by-step migration

### 1. Install reptimeline

```bash
pip install -e ".[dev]"
# If using triadic models:
pip install torch>=2.0
```

### 2. Replace checkpoint loading + extraction

**Before:**
```python
from src.evaluate import load_model
from src.triadic import PrimeMapper

model, tokenizer, config = load_model(ckpt_path, tok_path, device)
mapper = PrimeMapper(config.n_triadic_bits)

ids = tokenizer.encode(concept)
x = torch.tensor([ids], device=device)
proj = model(x)[1][0].mean(dim=0).cpu().numpy()
bits = mapper.get_bits(proj)
```

**After:**
```python
from reptimeline.extractors.triadic import TriadicExtractor

extractor = TriadicExtractor(n_bits=63, max_tokens=4)
snapshot = extractor.extract(ckpt_path, concepts=["king", "queen", "love"],
                             device="cpu")
# snapshot.codes["king"] → [0, 1, 1, 0, ...]
```

### 3. Replace manual timeline analysis

**Before:**
```python
# Loop through checkpoints manually, diff codes, track births/deaths
for ckpt in sorted(glob("checkpoints/model_step*.pt")):
    codes = extract_codes(ckpt, concepts)
    for concept in concepts:
        for bit in range(63):
            if codes[concept][bit] == 1 and prev_codes[concept][bit] == 0:
                print(f"Birth: {concept} bit {bit} at {step}")
```

**After:**
```python
from reptimeline import TimelineTracker

snapshots = extractor.extract_sequence("checkpoints/", concepts,
                                        max_checkpoints=10)
tracker = TimelineTracker(extractor, stability_window=3)
timeline = tracker.analyze(snapshots)
timeline.print_summary()

# Access structured data:
for birth in timeline.births:
    print(f"Birth: {birth.concept} bit {birth.code_index} at step {birth.step}")
```

### 4. Replace manual discovery

**Before:**
```python
# Manual dual detection via correlation
for i in range(63):
    for j in range(i+1, 63):
        corr = compute_correlation(codes, i, j)
        if corr < -0.7:
            print(f"Dual: bit {i} ↔ bit {j}")
```

**After:**
```python
from reptimeline import BitDiscovery

discovery = BitDiscovery(dead_threshold=0.02, dual_threshold=-0.7,
                          triadic_threshold=0.7)
report = discovery.discover(snapshots[-1], timeline=timeline, top_k=10)
discovery.print_report(report)

# Structured access:
for dual in report.discovered_duals:
    print(f"Dual: bit {dual.bit_a} ↔ bit {dual.bit_b} (r={dual.anti_correlation:.2f})")
```

### 5. Replace primitivos.json overlay (optional)

**Before:**
```python
# Manual mapping from bit → primitive name using primitivos.json
with open("primitivos.json") as f:
    prims = json.load(f)
for entry in prims["primitivos"]:
    if entry["bit"] == bit_index:
        print(f"bit {bit_index} = {entry['nombre']}")
```

**After:**
```python
from reptimeline import PrimitiveOverlay

overlay = PrimitiveOverlay("path/to/primitivos.json")
prim_report = overlay.analyze(timeline, concepts)
overlay.print_report(prim_report)

# Layer emergence, dependency completions, dual coherence — all computed
```

### 6. Add causal verification (new capability)

This didn't exist in triadic-microgpt:

```python
from reptimeline import CausalVerifier

verifier = CausalVerifier(labels, n_bootstrap=1000)
causal_report = verifier.verify(intervene_fn, concepts)
# → selectivity ratios, bootstrap CIs, permutation p-values, BH-FDR correction
```

### 7. Use the CLI instead of one-off scripts

```bash
# What used to be run_reptimeline_d_a18.py:
reptimeline --snapshots timeline_data.json --discover --plot

# With overlay:
reptimeline --snapshots timeline_data.json --overlay primitivos.json --output result.json
```

## Data format reference

### primitivos.json

```json
{
  "primitivos": [
    {
      "bit": 0,
      "primo": 2,
      "nombre": "vacío",
      "capa": 1,
      "deps": [],
      "def": "Ausencia como sustancia"
    }
  ],
  "ejes_duales": [["bien", "mal"], ["orden", "caos"]],
  "capas": {"1": {"nombre": "Punto (0D)"}, "2": {"nombre": "Línea (1D)"}}
}
```

### anclas.json

```json
{
  "frío": {
    "bits": ["tierra", "fuerza", "tacto", "orden", "control"],
    "razon": "Tierra: solidez. Fuerza: energía baja. ..."
  }
}
```

Anchors are training supervision data. reptimeline doesn't require them —
it discovers structure unsupervised. Use them with `Reconciler` to compare
discovered vs. expected structure.

### ConceptSnapshot JSON

```json
{
  "schema_version": "0.1",
  "step": 5000,
  "codes": {
    "king": [0, 1, 1, 0, ...],
    "queen": [1, 1, 0, 0, ...]
  }
}
```

## Key differences

1. **No more `src.*` imports.** Everything is `from reptimeline import ...`
2. **No more manual loops.** `TimelineTracker.analyze()` does births, deaths, connections, curves, stability, and phase transitions in one call.
3. **Discovery is unsupervised.** You don't need primitivos.json. `BitDiscovery` finds duals, dependencies, and triadic interactions from the data alone.
4. **Overlay is optional.** `PrimitiveOverlay` adds domain semantics on top of the generic analysis, but the core pipeline works without it.
5. **Extractors are pluggable.** Implement `RepresentationExtractor` for any discrete system — you're not locked into triadic bits.
6. **Statistical rigor.** Causal verification includes bootstrap CIs, permutation tests, and BH-FDR correction. Discovery includes Bonferroni correction.
