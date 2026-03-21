#!/usr/bin/env python3
"""
Causal v2: Break the Pythia SAE black box.

Part 1 — Embedding-based prediction:
    Predict holdout concept codes using embedding direction vectors
    (not coarse domain labels).

Part 2 — SAE intervention:
    Zero out individual SAE features, decode back to hidden states,
    inject into model, measure next-token logit change.

Usage:
    python examples/causal_v2.py --device cuda
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ======================================================================
# PART 1: EMBEDDING-BASED PREDICTION
# ======================================================================

def embedding_prediction(output_dir):
    """Predict holdout codes using per-bit direction vectors in embedding space."""
    from reptimeline.core import ConceptSnapshot
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    logger.info("=== PART 1: Embedding-Based Prediction ===")

    # Load data
    with open("results/pythia_sae/snapshots.json") as f:
        data = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), "pythia_concepts.json")) as f:
        cdata = json.load(f)

    final = ConceptSnapshot.from_dict(data["snapshots"][-1])
    domains = cdata["domains"]
    c2d = {c: d for d, ws in domains.items() for c in ws}
    all_concepts = data["concepts"]

    # Extract embeddings
    logger.info("Loading Pythia-70M embeddings...")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", revision="step143000")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    embed_matrix = model.gpt_neox.embed_in.weight.data.cpu().numpy()
    del model

    embeddings = {}
    for c in all_concepts:
        ids = tokenizer.encode(f" {c}", add_special_tokens=False)
        if len(ids) == 1:
            embeddings[c] = embed_matrix[ids[0]]

    # Active bits mask
    all_codes = np.array([final.codes[c] for c in all_concepts])
    active_mask = all_codes.mean(axis=0) > 0.02
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    logger.info(f"Active bits: {n_active}")

    # Stratified split
    train_concepts, test_concepts = [], []
    for domain, words in domains.items():
        train_concepts.extend(words[:-2])
        test_concepts.extend(words[-2:])

    # --- Learn per-bit direction vectors ---
    direction_vectors = {}  # bit_index -> normalized direction
    thresholds = {}

    for bit_idx in active_indices:
        active_cs = [c for c in train_concepts if c in final.codes and final.codes[c][bit_idx] == 1
                     and c in embeddings]
        inactive_cs = [c for c in train_concepts if c in final.codes and final.codes[c][bit_idx] == 0
                       and c in embeddings]

        if len(active_cs) < 2 or len(inactive_cs) < 2:
            continue

        centroid_a = np.mean([embeddings[c] for c in active_cs], axis=0)
        centroid_i = np.mean([embeddings[c] for c in inactive_cs], axis=0)
        direction = centroid_a - centroid_i
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction_vectors[bit_idx] = direction / norm

        # Threshold: midpoint of projections
        projs_a = [np.dot(embeddings[c], direction_vectors[bit_idx]) for c in active_cs]
        projs_i = [np.dot(embeddings[c], direction_vectors[bit_idx]) for c in inactive_cs]
        thresholds[bit_idx] = (np.mean(projs_a) + np.mean(projs_i)) / 2

    logger.info(f"Learned direction vectors for {len(direction_vectors)} bits")

    # --- Predict ---
    # 3 methods: embedding, domain, baseline
    emb_accs, dom_accs, base_accs = [], [], []
    emb_jacs, dom_jacs, base_jacs = [], [], []

    # Domain profiles (for comparison)
    domain_profiles = {}
    for domain in domains:
        dc = [c for c in train_concepts if c2d[c] == domain and c in final.codes]
        if dc:
            codes = np.array([final.codes[c] for c in dc])
            domain_profiles[domain] = codes[:, active_mask].mean(axis=0)

    overall = np.array([final.codes[c] for c in train_concepts if c in final.codes])[:, active_mask].mean(axis=0)

    results = []
    for c in test_concepts:
        if c not in final.codes or c not in embeddings:
            continue
        actual_full = np.array(final.codes[c])
        actual = actual_full[active_mask]
        domain = c2d[c]
        e = embeddings[c]

        # Embedding prediction
        emb_pred_full = np.zeros(len(actual_full))
        for bit_idx, dv in direction_vectors.items():
            proj = np.dot(e, dv)
            emb_pred_full[bit_idx] = 1 if proj > thresholds[bit_idx] else 0
        emb_pred = emb_pred_full[active_mask]

        # Domain prediction
        dom_pred = (domain_profiles[domain] > 0.5).astype(int)

        # Baseline
        base_pred = (overall > 0.5).astype(int)

        # Accuracies
        emb_acc = (emb_pred == actual).mean()
        dom_acc = (dom_pred == actual).mean()
        base_acc = (base_pred == actual).mean()
        emb_accs.append(emb_acc)
        dom_accs.append(dom_acc)
        base_accs.append(base_acc)

        # Jaccard
        def jaccard(p, a):
            ps, acs = set(np.where(p == 1)[0]), set(np.where(a == 1)[0])
            u = ps | acs
            return len(ps & acs) / len(u) if u else 1
        ej = jaccard(emb_pred, actual)
        dj = jaccard(dom_pred, actual)
        bj = jaccard(base_pred, actual)
        emb_jacs.append(ej)
        dom_jacs.append(dj)
        base_jacs.append(bj)

        results.append({"concept": c, "domain": domain,
                        "emb_acc": emb_acc, "dom_acc": dom_acc, "base_acc": base_acc,
                        "emb_jac": ej, "dom_jac": dj, "base_jac": bj})

    # Print results
    print("\n" + "=" * 75)
    print("  PART 1: EMBEDDING-BASED PREDICTION (active bits only)")
    print("=" * 75)
    for r in sorted(results, key=lambda x: -x["emb_acc"]):
        best = "EMB" if r["emb_acc"] >= max(r["dom_acc"], r["base_acc"]) else (
            "DOM" if r["dom_acc"] >= r["base_acc"] else "BASE")
        print(f"  {r['concept']:<12s} ({r['domain']:<10s})  "
              f"emb={r['emb_acc']:.1%}  dom={r['dom_acc']:.1%}  base={r['base_acc']:.1%}  "
              f"jac={r['emb_jac']:.2f}  [{best}]")

    me, md, mb = np.mean(emb_accs), np.mean(dom_accs), np.mean(base_accs)
    je, jd, jb = np.mean(emb_jacs), np.mean(dom_jacs), np.mean(base_jacs)
    print(f"\n  ACCURACY:  emb={me:.1%}  domain={md:.1%}  baseline={mb:.1%}")
    print(f"  JACCARD:   emb={je:.2f}  domain={jd:.2f}  baseline={jb:.2f}")
    print(f"  EMB vs baseline: acc {me - mb:+.1%}  jac {je - jb:+.2f}")
    print(f"  EMB vs domain:   acc {me - md:+.1%}  jac {je - jd:+.2f}")
    print("=" * 75)

    return results, embeddings


# ======================================================================
# PART 2: SAE CAUSAL INTERVENTION
# ======================================================================

def sae_intervention(device, output_dir):
    """Zero out SAE features and measure selective causal effect."""
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    from sparsify import Sae
    from reptimeline.core import ConceptSnapshot

    logger.info("\n=== PART 2: SAE Causal Intervention ===")

    # Load data
    with open("results/pythia_sae/snapshots.json") as f:
        data = json.load(f)
    with open("results/pythia_sae/primitives.json") as f:
        primitives = json.load(f)
    with open("results/pythia_sae/feature_selection.json") as f:
        feat_sel = json.load(f)

    concepts = data["concepts"]
    selected_features = feat_sel["features"]  # 256 SAE feature indices

    # Build label map: our bit index -> SAE feature index + label
    bit_labels = {}
    for p in primitives["primitivos"]:
        bit_labels[p["bit"]] = {
            "label": p["nombre"],
            "sae_feature": selected_features[p["bit"]],
            "top_concepts": p["top_concepts"][:5],
        }

    # Select top 10 most interesting bits (highest confidence, not universal)
    with open("results/pythia_sae/discovery.json") as f:
        disc = json.load(f)
    active_semantics = [bs for bs in disc["bit_semantics"]
                        if 0.1 < bs["activation_rate"] < 0.9]
    active_semantics.sort(key=lambda x: -abs(x["activation_rate"] - 0.5))
    test_bits = [bs["bit"] for bs in active_semantics[:10]]
    logger.info(f"Testing {len(test_bits)} bits for causal intervention")
    for b in test_bits:
        info = bit_labels.get(b, {"label": f"bit_{b}", "sae_feature": selected_features[b]})
        logger.info(f"  bit {b}: label={info['label']}, SAE feature={info['sae_feature']}")

    # Load model + SAE
    logger.info("Loading Pythia-70M + SAE...")
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", revision="step143000"
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sae = Sae.load_from_hub(
        "EleutherAI/sae-pythia-70m-32k", hookpoint="layers.3", device=device
    )

    logger.info(f"SAE: d_in={sae.d_in}, num_latents={sae.num_latents}, k={sae.cfg.k}")

    # --- Level 1: Hidden state perturbation ---
    logger.info("Level 1: Measuring hidden state perturbation...")
    context_template = "The word is: {concept}"

    # Collect hidden states for all concepts
    captured = {}
    def hook_fn(module, input, output):
        captured["h"] = output

    handle = model.gpt_neox.layers[3].mlp.register_forward_hook(hook_fn)
    concept_hidden = {}
    with torch.no_grad():
        for c in concepts:
            inputs = tokenizer(context_template.format(concept=c), return_tensors="pt").to(device)
            model(**inputs)
            concept_hidden[c] = captured["h"][0, -1, :].clone()
    handle.remove()

    # For each test bit, measure perturbation per concept
    perturbation_matrix = np.zeros((len(test_bits), len(concepts)))

    for i, bit_idx in enumerate(test_bits):
        sae_feat = selected_features[bit_idx]

        for j, c in enumerate(concepts):
            h = concept_hidden[c].unsqueeze(0)

            # Encode
            enc = sae.encode(h)

            # Decode original
            h_recon = sae.decode(enc.top_acts, enc.top_indices).squeeze(0)

            # Zero out the target feature
            mask = enc.top_indices[0] != sae_feat
            if mask.all():
                # Feature wasn't active for this concept
                perturbation_matrix[i, j] = 0.0
                continue

            acts_mod = enc.top_acts[0, mask].unsqueeze(0)
            idx_mod = enc.top_indices[0, mask].unsqueeze(0)
            h_modified = sae.decode(acts_mod, idx_mod).squeeze(0)

            # Perturbation = L2 distance
            perturbation_matrix[i, j] = torch.norm(h_recon - h_modified).item()

    # Compute selectivity: perturbation on labeled concepts / perturbation on others
    print("\n" + "=" * 75)
    print("  LEVEL 1: HIDDEN STATE PERTURBATION (zero one feature)")
    print("=" * 75)

    selectivity_results = []
    for i, bit_idx in enumerate(test_bits):
        info = bit_labels.get(bit_idx, {"label": f"bit_{bit_idx}", "top_concepts": []})
        labeled = set(info.get("top_concepts", []))
        perturbs = perturbation_matrix[i]

        if labeled:
            labeled_idx = [j for j, c in enumerate(concepts) if c in labeled]
            other_idx = [j for j, c in enumerate(concepts) if c not in labeled]
            mean_labeled = np.mean(perturbs[labeled_idx]) if labeled_idx else 0
            mean_other = np.mean(perturbs[other_idx]) if other_idx else 0
            selectivity = mean_labeled / mean_other if mean_other > 1e-8 else 0
        else:
            mean_labeled = mean_other = selectivity = 0

        # Top 5 most affected concepts
        top5_idx = np.argsort(perturbs)[::-1][:5]
        top5 = [(concepts[k], round(perturbs[k], 4)) for k in top5_idx]

        selectivity_results.append({
            "bit": bit_idx, "label": info["label"],
            "selectivity": round(selectivity, 2),
            "mean_labeled": round(mean_labeled, 4),
            "mean_other": round(mean_other, 4),
            "top5_affected": top5,
        })

        print(f"  bit {bit_idx:>2d} ({info['label']:<15s})  "
              f"sel={selectivity:.2f}  labeled={mean_labeled:.4f}  other={mean_other:.4f}  "
              f"top=[{', '.join(c for c, _ in top5)}]")

    selective_bits = [s for s in selectivity_results if s["selectivity"] > 1.5]
    print(f"\n  Selective bits (sel > 1.5): {len(selective_bits)} / {len(test_bits)}")

    # --- Level 2: Next-token logit change ---
    logger.info("\nLevel 2: Measuring next-token logit changes...")

    logit_kl_matrix = np.zeros((len(test_bits), len(concepts)))

    for i, bit_idx in enumerate(test_bits):
        sae_feat = selected_features[bit_idx]

        for j, c in enumerate(concepts):
            prompt = context_template.format(concept=c)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Original forward pass
            with torch.no_grad():
                outputs_orig = model(**inputs)
            logits_orig = outputs_orig.logits[0, -1, :].float()

            # Get hidden state and modify
            h = concept_hidden[c].unsqueeze(0)
            enc = sae.encode(h)

            mask = enc.top_indices[0] != sae_feat
            if mask.all():
                logit_kl_matrix[i, j] = 0.0
                continue

            acts_mod = enc.top_acts[0, mask].unsqueeze(0)
            idx_mod = enc.top_indices[0, mask].unsqueeze(0)
            h_modified = sae.decode(acts_mod, idx_mod).squeeze(0)

            # Hook: replace MLP output at layer 3 with modified version
            def make_replace_hook(h_mod):
                def hook(module, input, output):
                    out = output.clone()
                    out[0, -1, :] = h_mod
                    return out
                return hook

            hook_handle = model.gpt_neox.layers[3].mlp.register_forward_hook(
                make_replace_hook(h_modified))
            with torch.no_grad():
                outputs_mod = model(**inputs)
            hook_handle.remove()

            logits_mod = outputs_mod.logits[0, -1, :].float()

            # KL divergence (clamp to avoid log(0) -> NaN)
            p = F.softmax(logits_orig, dim=0).clamp(min=1e-10)
            q = F.softmax(logits_mod, dim=0).clamp(min=1e-10)
            kl = (p * (p.log() - q.log())).sum().item()
            if np.isfinite(kl):
                logit_kl_matrix[i, j] = kl

    print("\n" + "=" * 75)
    print("  LEVEL 2: NEXT-TOKEN LOGIT CHANGE (KL divergence)")
    print("=" * 75)

    logit_results = []
    for i, bit_idx in enumerate(test_bits):
        info = bit_labels.get(bit_idx, {"label": f"bit_{bit_idx}", "top_concepts": []})
        labeled = set(info.get("top_concepts", []))
        kls = logit_kl_matrix[i]

        if labeled:
            labeled_idx = [j for j, c in enumerate(concepts) if c in labeled]
            other_idx = [j for j, c in enumerate(concepts) if c not in labeled]
            mean_kl_labeled = np.mean(kls[labeled_idx]) if labeled_idx else 0
            mean_kl_other = np.mean(kls[other_idx]) if other_idx else 0
            kl_selectivity = mean_kl_labeled / mean_kl_other if mean_kl_other > 1e-8 else 0
        else:
            mean_kl_labeled = mean_kl_other = kl_selectivity = 0

        top5_idx = np.argsort(kls)[::-1][:5]
        top5 = [(concepts[k], round(kls[k], 4)) for k in top5_idx]

        logit_results.append({
            "bit": bit_idx, "label": info["label"],
            "kl_selectivity": round(kl_selectivity, 2),
            "mean_kl_labeled": round(mean_kl_labeled, 4),
            "mean_kl_other": round(mean_kl_other, 4),
            "top5_affected": top5,
        })

        print(f"  bit {bit_idx:>2d} ({info['label']:<15s})  "
              f"kl_sel={kl_selectivity:.2f}  labeled={mean_kl_labeled:.4f}  other={mean_kl_other:.4f}  "
              f"top=[{', '.join(c for c, _ in top5)}]")

    kl_selective = [r for r in logit_results if r["kl_selectivity"] > 1.5]
    print(f"\n  KL-selective bits (sel > 1.5): {len(kl_selective)} / {len(test_bits)}")

    # --- Visualization ---
    logger.info("Generating visualizations...")

    # Heatmap: perturbation matrix
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    im1 = ax1.imshow(perturbation_matrix, aspect="auto", cmap="YlOrRd")
    ax1.set_yticks(range(len(test_bits)))
    ax1.set_yticklabels([f"bit {b} ({bit_labels.get(b, {}).get('label', '?')})" for b in test_bits])
    ax1.set_xticks(range(len(concepts)))
    ax1.set_xticklabels(concepts, rotation=90, fontsize=6)
    ax1.set_title("Level 1: Hidden State Perturbation (L2 norm)")
    plt.colorbar(im1, ax=ax1, label="L2 distance")

    im2 = ax2.imshow(logit_kl_matrix, aspect="auto", cmap="YlOrRd")
    ax2.set_yticks(range(len(test_bits)))
    ax2.set_yticklabels([f"bit {b} ({bit_labels.get(b, {}).get('label', '?')})" for b in test_bits])
    ax2.set_xticks(range(len(concepts)))
    ax2.set_xticklabels(concepts, rotation=90, fontsize=6)
    ax2.set_title("Level 2: Next-Token KL Divergence")
    plt.colorbar(im2, ax=ax2, label="KL divergence")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sae_intervention_heatmap.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved sae_intervention_heatmap.png")

    del model
    torch.cuda.empty_cache()

    return selectivity_results, logit_results, perturbation_matrix, logit_kl_matrix


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Causal v2: break Pythia SAE black box")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="results/causal")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    os.makedirs(args.output, exist_ok=True)

    # Part 1
    pred_results, embeddings = embedding_prediction(args.output)

    # Part 2
    sel_results, logit_results, _, _ = sae_intervention(args.device, args.output)

    # Save report
    me = np.mean([r["emb_acc"] for r in pred_results])
    md = np.mean([r["dom_acc"] for r in pred_results])
    mb = np.mean([r["base_acc"] for r in pred_results])
    je = np.mean([r["emb_jac"] for r in pred_results])

    n_selective_l1 = len([s for s in sel_results if s["selectivity"] > 1.5])
    n_selective_l2 = len([s for s in logit_results if s["kl_selectivity"] > 1.5])

    report = {
        "prediction": {
            "embedding_accuracy": round(me, 4),
            "domain_accuracy": round(md, 4),
            "baseline_accuracy": round(mb, 4),
            "embedding_jaccard": round(je, 4),
            "improvement_over_baseline": round(me - mb, 4),
            "improvement_over_domain": round(me - md, 4),
            "per_concept": pred_results,
        },
        "intervention_l1": {
            "n_selective": n_selective_l1,
            "n_tested": len(sel_results),
            "results": sel_results,
        },
        "intervention_l2": {
            "n_selective": n_selective_l2,
            "n_tested": len(logit_results),
            "results": logit_results,
        },
    }

    with open(os.path.join(args.output, "causal_v2_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Verdict
    pred_pass = me > mb + 0.03
    intervention_pass = n_selective_l2 >= 3
    mean_kl_sel = np.mean([r["kl_selectivity"] for r in logit_results])

    print("\n" + "=" * 75)
    print("  FINAL VERDICT")
    print("=" * 75)
    print(f"  Prediction:    emb={me:.1%} vs baseline={mb:.1%} ({me-mb:+.1%})  "
          f"{'PASS' if pred_pass else 'FAIL'}")
    print(f"  Intervention:  {n_selective_l2}/{len(logit_results)} bits KL-selective  "
          f"(mean sel={mean_kl_sel:.1f}x)  {'PASS' if intervention_pass else 'FAIL'}")

    if intervention_pass:
        print(f"\n  BLACK BOX BROKEN (causal): features are identifiable, labelable,")
        print(f"  and causally selective (removing them changes output {mean_kl_sel:.0f}x more")
        print(f"  for semantically related concepts than unrelated ones)")
        if not pred_pass:
            print(f"\n  Note: prediction from embeddings alone fails — SAE features encode")
            print(f"  context-dependent information beyond token-level similarity")
    else:
        print(f"\n  NOT BROKEN: insufficient causal selectivity")
    print("=" * 75)


if __name__ == "__main__":
    main()
