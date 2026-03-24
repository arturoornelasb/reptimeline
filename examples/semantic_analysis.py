#!/usr/bin/env python3
"""
Semantic analysis of Pythia-70M SAE results.

Determines whether the structure discovered by reptimeline (duals, dependencies,
triadic interactions, hierarchy) has genuine semantic meaning.

Usage:
    python examples/semantic_analysis.py
    python examples/semantic_analysis.py --output results/pythia_sae/semantic_report.json
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reptimeline.autolabel import AutoLabeler
from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery
from reptimeline.extractors.base import RepresentationExtractor
from reptimeline.tracker import TimelineTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Lightweight extractor for analysis (no model needed)
class _JaccardExtractor(RepresentationExtractor):
    def extract(self, *a, **k): raise NotImplementedError
    def similarity(self, a, b):
        sa = set(i for i, v in enumerate(a) if v == 1)
        sb = set(i for i, v in enumerate(b) if v == 1)
        u = sa | sb
        return len(sa & sb) / len(u) if u else 1.0
    def shared_features(self, a, b):
        return [i for i in range(min(len(a), len(b))) if a[i] == 1 and b[i] == 1]


def load_results(results_dir: str):
    """Load cached Pythia SAE results."""
    with open(os.path.join(results_dir, "snapshots.json")) as f:
        data = json.load(f)
    snapshots = [ConceptSnapshot.from_dict(s) for s in data["snapshots"]]
    concepts = data["concepts"]
    domains = data["domains"]
    return snapshots, concepts, domains


def build_concept_domain_map(domains: Dict[str, List[str]]) -> Dict[str, str]:
    """Map each concept to its domain."""
    return {c: domain for domain, words in domains.items() for c in words}


def extract_embeddings(concepts: List[str], cache_dir=None):
    """Extract Pythia-70M embeddings for concepts + broad vocabulary."""
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    logger.info("Loading Pythia-70M for embedding extraction...")
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", revision="step143000", cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", cache_dir=cache_dir)
    embed_matrix = model.gpt_neox.embed_in.weight.data.cpu().numpy()  # (50304, 512)

    # Extract concept embeddings
    embeddings = {}
    for concept in concepts:
        token_ids = tokenizer.encode(f" {concept}", add_special_tokens=False)
        if len(token_ids) == 1:
            embeddings[concept] = embed_matrix[token_ids[0]]

    logger.info(f"Extracted embeddings for {len(embeddings)}/{len(concepts)} concepts")

    # Build broad vocabulary: common single-token English words
    broad_vocab_words = [
        # Semantic categories
        "animal", "plant", "object", "person", "place", "thing", "action",
        "feeling", "emotion", "thought", "idea", "concept", "property",
        "nature", "body", "mind", "soul", "spirit", "life", "death",
        # Dimensions
        "physical", "abstract", "concrete", "mental", "social", "natural",
        "living", "dead", "human", "divine", "good", "evil", "positive", "negative",
        "active", "passive", "male", "female", "young", "ancient",
        # Domains
        "weather", "land", "sky", "sea", "creature", "beast", "predator", "prey",
        "emotion", "virtue", "vice", "knowledge", "power", "family", "war", "peace",
        "size", "speed", "temperature", "age", "color", "shape",
        "royalty", "authority", "danger", "safety", "wild", "domestic",
        # Relations
        "similar", "opposite", "related", "different", "same", "other",
        "cause", "effect", "part", "whole", "type", "instance",
    ]

    broad_embeddings = dict(embeddings)  # Start with concepts
    for word in broad_vocab_words:
        token_ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        if len(token_ids) == 1 and word not in broad_embeddings:
            broad_embeddings[word] = embed_matrix[token_ids[0]]

    logger.info(f"Broad vocabulary: {len(broad_embeddings)} words")

    del model
    return embeddings, broad_embeddings


def analyze_duals(report, embeddings, broad_embeddings, concept_domain, concepts):
    """Semantic analysis of dual pairs."""
    logger.info(f"Analyzing {len(report.discovered_duals)} dual pairs...")
    bit_semantics = {bs.bit_index: bs for bs in report.bit_semantics}
    results = []

    for dual in report.discovered_duals:
        bs_a = bit_semantics.get(dual.bit_a)
        bs_b = bit_semantics.get(dual.bit_b)
        if not bs_a or not bs_b:
            continue

        # Domain distribution for each bit
        domain_a = Counter(concept_domain.get(c, "unknown") for c in bs_a.top_concepts)
        domain_b = Counter(concept_domain.get(c, "unknown") for c in bs_b.top_concepts)
        total_a = sum(domain_a.values()) or 1
        total_b = sum(domain_b.values()) or 1
        domain_a_pct = {d: round(n / total_a, 3) for d, n in domain_a.most_common()}
        domain_b_pct = {d: round(n / total_b, 3) for d, n in domain_b.most_common()}

        # Dominant domains
        top_domain_a = domain_a.most_common(1)[0][0] if domain_a else "?"
        top_domain_b = domain_b.most_common(1)[0][0] if domain_b else "?"

        # Coherence: fraction of top concepts sharing the dominant domain
        coherence_a = domain_a_pct.get(top_domain_a, 0)
        coherence_b = domain_b_pct.get(top_domain_b, 0)
        coherence = (coherence_a + coherence_b) / 2

        # Separation direction in embedding space
        active_a_vecs = [embeddings[c] for c in bs_a.top_concepts if c in embeddings]
        active_b_vecs = [embeddings[c] for c in bs_b.top_concepts if c in embeddings]
        separation_word = "?"
        if active_a_vecs and active_b_vecs:
            centroid_a = np.mean(active_a_vecs, axis=0)
            centroid_b = np.mean(active_b_vecs, axis=0)
            direction = centroid_a - centroid_b
            d_norm = np.linalg.norm(direction)
            if d_norm > 1e-8:
                direction_normed = direction / d_norm
                # Find closest word in broad vocab
                best_sim = -1
                for word, vec in broad_embeddings.items():
                    if word in bs_a.top_concepts or word in bs_b.top_concepts:
                        continue
                    v_norm = np.linalg.norm(vec)
                    if v_norm < 1e-8:
                        continue
                    sim = float(np.dot(vec / v_norm, direction_normed))
                    if sim > best_sim:
                        best_sim = sim
                        separation_word = word

        # Axis label
        if top_domain_a != top_domain_b:
            axis_label = f"{top_domain_a}-vs-{top_domain_b}"
        else:
            axis_label = f"{top_domain_a}-internal"

        results.append({
            "bit_a": dual.bit_a,
            "bit_b": dual.bit_b,
            "corr": round(dual.anti_correlation, 3),
            "concepts_a": bs_a.top_concepts[:5],
            "concepts_b": bs_b.top_concepts[:5],
            "domain_a": domain_a_pct,
            "domain_b": domain_b_pct,
            "axis_label": axis_label,
            "separation_word": separation_word,
            "coherence": round(coherence, 3),
        })

    return results


def analyze_hierarchy(report, bit_labels_map):
    """Analyze whether hierarchy layers correspond to broad->fine features."""
    logger.info(f"Analyzing {len(report.discovered_hierarchy)} hierarchy entries...")
    bit_semantics = {bs.bit_index: bs for bs in report.bit_semantics}

    # Group by layer
    layers = defaultdict(list)
    for h in report.discovered_hierarchy:
        if h.layer > 0:
            layers[h.layer].append(h)

    results = []
    for layer_num in sorted(layers.keys()):
        entries = layers[layer_num]
        bits = [e.bit_index for e in entries]
        rates = [bit_semantics[b].activation_rate for b in bits if b in bit_semantics]
        labels = [bit_labels_map.get(b, f"bit_{b}") for b in bits]
        stable_steps = [e.first_stable_step for e in entries if e.first_stable_step]

        results.append({
            "layer": layer_num,
            "n_bits": len(bits),
            "bits": bits[:10],
            "labels": labels[:10],
            "mean_activation_rate": round(np.mean(rates), 4) if rates else 0,
            "stable_at_step": min(stable_steps) if stable_steps else None,
        })

    return results


def analyze_triadic(report, bit_labels_map, embeddings, concept_domain):
    """Analyze whether triadic interactions are semantic AND-gates."""
    logger.info(f"Analyzing {len(report.discovered_triadic_deps)} triadic interactions...")
    bit_semantics = {bs.bit_index: bs for bs in report.bit_semantics}
    results = []

    for t in report.discovered_triadic_deps[:20]:  # Top 20 by strength
        label_i = bit_labels_map.get(t.bit_i, f"bit_{t.bit_i}")
        label_j = bit_labels_map.get(t.bit_j, f"bit_{t.bit_j}")
        label_r = bit_labels_map.get(t.bit_r, f"bit_{t.bit_r}")

        # Find concepts where both i and j are active
        bs_i = bit_semantics.get(t.bit_i)
        bs_j = bit_semantics.get(t.bit_j)
        bs_r = bit_semantics.get(t.bit_r)
        if not bs_i or not bs_j or not bs_r:
            continue

        ij_concepts = set(bs_i.top_concepts) & set(bs_j.top_concepts)
        r_concepts = set(bs_r.top_concepts)

        # Domain distribution of the intersection
        ij_domains = Counter(concept_domain.get(c, "?") for c in ij_concepts)

        # Embedding tightness: how clustered are the i∧j concepts?
        ij_vecs = [embeddings[c] for c in ij_concepts if c in embeddings]
        tightness = 0.0
        if len(ij_vecs) >= 2:
            centroid = np.mean(ij_vecs, axis=0)
            dists = [np.linalg.norm(v - centroid) for v in ij_vecs]
            tightness = round(1.0 / (1.0 + np.mean(dists)), 4)

        results.append({
            "bit_i": t.bit_i, "bit_j": t.bit_j, "bit_r": t.bit_r,
            "label_i": label_i, "label_j": label_j, "label_r": label_r,
            "p_r_given_ij": round(t.p_r_given_ij, 3),
            "strength": round(t.interaction_strength, 3),
            "ij_concepts": sorted(ij_concepts)[:5],
            "r_concepts": sorted(r_concepts)[:5],
            "ij_domains": dict(ij_domains.most_common(3)),
            "tightness": tightness,
        })

    return results


def print_summary(dual_results, hierarchy_results, triadic_results, labels_embed, labels_contrast):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("  SEMANTIC ANALYSIS REPORT — Pythia-70M SAE Features")
    print("=" * 70)

    # -- Duals --
    print(f"\n  DUAL PAIRS ({len(dual_results)} total)")
    print("  " + "-" * 66)
    semantic_duals = [d for d in dual_results if d["coherence"] > 0.4]
    print(f"  Semantically coherent (coherence > 0.4): {len(semantic_duals)}/{len(dual_results)}")
    for d in sorted(dual_results, key=lambda x: -x["coherence"])[:10]:
        print(f"    bit {d['bit_a']:>3d} vs {d['bit_b']:>3d}  "
              f"corr={d['corr']:.2f}  axis={d['axis_label']:<25s}  "
              f"coh={d['coherence']:.2f}  sep={d['separation_word']}")

    # Axis distribution
    axes = Counter(d["axis_label"] for d in dual_results)
    print("\n  Axis types:")
    for axis, count in axes.most_common():
        print(f"    {axis}: {count}")

    # -- Hierarchy --
    print(f"\n  HIERARCHY ({len(hierarchy_results)} layers)")
    print("  " + "-" * 66)
    for h in hierarchy_results:
        labels_str = ", ".join(h["labels"][:5])
        step = h["stable_at_step"] or "?"
        print(f"    Layer {h['layer']}: {h['n_bits']:>3d} bits  "
              f"act_rate={h['mean_activation_rate']:.3f}  "
              f"stable@step {step}  [{labels_str}]")

    # Check monotonicity
    rates = [h["mean_activation_rate"] for h in hierarchy_results]
    if len(rates) >= 2:
        is_decreasing = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
        print(f"\n  Activation rate decreasing across layers: {'YES' if is_decreasing else 'NO'}")
        print(f"  (broad->fine hierarchy {'CONFIRMED' if is_decreasing else 'NOT confirmed'})")

    # -- Triadic --
    n_shown = min(10, len(triadic_results))
    print(f"\n  TRIADIC INTERACTIONS (top {n_shown} "
          f"of {len(triadic_results)})")
    print("  " + "-" * 66)
    for t in triadic_results[:10]:
        print(f"    {t['label_i']} + {t['label_j']} -> {t['label_r']}  "
              f"P={t['p_r_given_ij']:.2f}  strength={t['strength']:.2f}  "
              f"tight={t['tightness']:.2f}  [{', '.join(t['ij_concepts'][:3])}]")

    # -- Auto-labels summary --
    active_embed = [lb for lb in labels_embed if lb.label != "DEAD"]
    active_contrast = [lb for lb in labels_contrast if lb.label != "DEAD"]
    print("\n  AUTO-LABELS")
    print("  " + "-" * 66)
    print(f"  Embedding strategy: {len(active_embed)} active bits labeled")
    print(f"  Contrastive strategy: {len(active_contrast)} active bits labeled")

    # Agreement between strategies
    agreement = 0
    for le, lc in zip(labels_embed, labels_contrast):
        if le.label == lc.label and le.label != "DEAD":
            agreement += 1
    n_active = len(active_embed)
    print(f"  Strategy agreement: {agreement}/{n_active} "
          f"({100*agreement/n_active:.0f}%)" if n_active > 0 else "")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Semantic analysis of Pythia SAE results")
    parser.add_argument("--results", default="results/pythia_sae",
                        help="Directory with snapshots.json and discovery.json")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results_dir/semantic_report.json)")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.results, "semantic_report.json")

    # Step 1: Load cached results
    logger.info("Loading cached results...")
    snapshots, concepts, domains = load_results(args.results)
    concept_domain = build_concept_domain_map(domains)
    logger.info(
        f"Loaded {len(snapshots)} snapshots, "
        f"{len(concepts)} concepts, {len(domains)} domains"
    )

    # Step 2: Re-run discovery for full report
    logger.info("Re-running BitDiscovery for full report...")
    tracker = TimelineTracker(_JaccardExtractor())
    timeline = tracker.analyze(snapshots)
    discovery = BitDiscovery()
    report = discovery.discover(snapshots[-1], timeline=timeline, top_k=15)
    logger.info(f"Discovery: {len(report.discovered_duals)} duals, "
                f"{len(report.discovered_deps)} deps, "
                f"{len(report.discovered_triadic_deps)} triadic, "
                f"{len(report.discovered_hierarchy)} hierarchy")

    # Step 3: Extract embeddings
    embeddings, broad_embeddings = extract_embeddings(concepts, cache_dir=args.cache_dir)

    # Step 4: Auto-label
    logger.info("Auto-labeling bits...")
    labeler = AutoLabeler()
    labels_embed = labeler.label_by_embedding(report, broad_embeddings)
    labels_contrast = labeler.label_by_contrast(report, broad_embeddings)
    labeler.print_labels(labels_embed)
    labeler.print_labels(labels_contrast)

    # Build label map (contrastive preferred — usually more specific)
    bit_labels_map = {}
    for bl in labels_contrast:
        if bl.label != "DEAD":
            bit_labels_map[bl.bit_index] = bl.label

    # Step 5: Dual analysis
    dual_results = analyze_duals(report, embeddings, broad_embeddings, concept_domain, concepts)

    # Step 6: Hierarchy analysis
    hierarchy_results = analyze_hierarchy(report, bit_labels_map)

    # Step 7: Triadic analysis
    triadic_results = analyze_triadic(report, bit_labels_map, embeddings, concept_domain)

    # Step 8: Output
    print_summary(dual_results, hierarchy_results, triadic_results, labels_embed, labels_contrast)

    # Save JSON report
    report_json = {
        "metadata": {
            "model": "EleutherAI/pythia-70m",
            "sae": "EleutherAI/sae-pythia-70m-32k",
            "layer": 3,
            "n_concepts": len(concepts),
            "n_active_bits": report.n_active_bits,
            "n_dead_bits": report.n_dead_bits,
        },
        "bit_labels": {
            "embedding": [
                {"bit": lb.bit_index, "label": lb.label,
                 "confidence": round(lb.confidence, 3),
                 "active_concepts": lb.active_concepts[:5]}
                for lb in labels_embed if lb.label != "DEAD"
            ],
            "contrastive": [
                {"bit": lb.bit_index, "label": lb.label,
                 "confidence": round(lb.confidence, 3),
                 "active_concepts": lb.active_concepts[:5]}
                for lb in labels_contrast if lb.label != "DEAD"
            ],
        },
        "dual_analysis": dual_results,
        "hierarchy_analysis": hierarchy_results,
        "triadic_analysis": triadic_results,
        "summary": {
            "n_semantic_duals": len([d for d in dual_results if d["coherence"] > 0.4]),
            "n_total_duals": len(dual_results),
            "hierarchy_decreasing": all(
                hierarchy_results[i]["mean_activation_rate"]
                >= hierarchy_results[i+1]["mean_activation_rate"]
                for i in range(len(hierarchy_results)-1)
            ) if len(hierarchy_results) >= 2 else None,
            "n_triadic_analyzed": len(triadic_results),
        },
    }

    # Convert numpy types for JSON serialization
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(args.output, "w") as f:
        json.dump(report_json, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved semantic report to {args.output}")


if __name__ == "__main__":
    main()
