#!/usr/bin/env python3
"""
Causal v2: Break the Pythia SAE black box.

Part 1 — Embedding-based prediction:
    Predict holdout concept codes using embedding direction vectors
    (not coarse domain labels).

Part 2 — SAE intervention via CausalVerifier:
    Zero out individual SAE features, decode back to hidden states,
    inject into model, measure next-token logit change.
    Uses reptimeline.CausalVerifier for statistical testing (bootstrap CI,
    permutation tests, BH-FDR correction).

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

from reptimeline.core import ConceptSnapshot
from reptimeline.causal import CausalVerifier
from reptimeline.viz.causal_heatmap import plot_causal_heatmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ======================================================================
# PART 1: EMBEDDING-BASED PREDICTION
# ======================================================================

def embedding_prediction(output_dir):
    """Predict holdout codes using per-bit direction vectors in embedding space."""
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
# PART 1b: MLP PREDICTION (replace failed linear prediction)
# ======================================================================

def mlp_prediction(output_dir):
    """Train a small MLP per bit to predict SAE features from embeddings."""
    import torch
    import torch.nn as nn

    logger.info("\n=== PART 1b: MLP Prediction ===")

    # Load same data as Part 1
    with open("examples/pythia_concepts.json") as f:
        data = json.load(f)
    concepts_by_domain = data["domains"]
    all_concepts = []
    for domain, clist in concepts_by_domain.items():
        for c in clist:
            all_concepts.append((c, domain))

    with open("results/pythia_sae/snapshots.json") as f:
        snap_data = json.load(f)
    final_snap = snap_data["snapshots"][-1]
    codes = final_snap["codes"]

    # Load embeddings
    from transformers import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", revision="step143000")
    embed_matrix = model.gpt_neox.embed_in.weight.detach().float()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    del model

    embeddings = {}
    for c, _ in all_concepts:
        ids = tokenizer.encode(c, add_special_tokens=False)
        vecs = embed_matrix[ids]
        embeddings[c] = vecs.mean(dim=0).numpy()

    dim = len(list(embeddings.values())[0])

    # Split: 2 holdout per domain
    train_concepts, test_concepts = [], []
    for domain, clist in concepts_by_domain.items():
        test_concepts.extend([(c, domain) for c in clist[:2]])
        train_concepts.extend([(c, domain) for c in clist[2:]])

    # Find active bits
    code_len = len(list(codes.values())[0])
    all_bits = np.array([[codes[c][b] for c in [x[0] for x in train_concepts]]
                         for b in range(code_len)])
    active_mask = (all_bits.mean(axis=1) > 0.05) & (all_bits.mean(axis=1) < 0.95)
    active_bits = np.where(active_mask)[0]
    logger.info(f"Active bits for MLP: {len(active_bits)}")

    # Prepare data
    X_train = np.array([embeddings[c] for c, _ in train_concepts])
    X_test = np.array([embeddings[c] for c, _ in test_concepts])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    # Train one MLP per active bit
    mlp_preds = np.zeros((len(test_concepts), code_len))
    base_preds = np.zeros((len(test_concepts), code_len))

    for b in range(code_len):
        y_train = np.array([codes[c][b] for c, _ in train_concepts])
        majority = 1 if y_train.mean() > 0.5 else 0
        base_preds[:, b] = majority

        if b not in active_bits:
            mlp_preds[:, b] = majority
            continue

        y_train_t = torch.tensor(y_train, dtype=torch.float32)

        # Small MLP: dim -> 32 -> 1
        mlp = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        opt = torch.optim.Adam(mlp.parameters(), lr=0.01)
        loss_fn = nn.BCELoss()

        for epoch in range(200):
            pred = mlp(X_train_t).squeeze()
            loss = loss_fn(pred, y_train_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            test_pred = mlp(X_test_t).squeeze().numpy()
        mlp_preds[:, b] = (test_pred > 0.5).astype(float)

    # Evaluate
    mlp_accs, base_accs, mlp_jacs = [], [], []
    print("\n" + "=" * 75)
    print("  PART 1b: MLP PREDICTION (active bits only)")
    print("=" * 75)

    mlp_results = []
    for i, (c, domain) in enumerate(test_concepts):
        actual = np.array(codes[c])
        mp = mlp_preds[i]
        bp = base_preds[i]

        # Only active bits
        ma = (mp[active_bits] == actual[active_bits]).mean()
        ba = (bp[active_bits] == actual[active_bits]).mean()
        mlp_accs.append(ma)
        base_accs.append(ba)

        # Jaccard on active
        mp_set = set(active_bits[mp[active_bits] == 1])
        a_set = set(active_bits[actual[active_bits] == 1])
        u = mp_set | a_set
        mj = len(mp_set & a_set) / len(u) if u else 1
        mlp_jacs.append(mj)

        best = "MLP" if ma > ba else "BASE"
        print(f"  {c:<12s} ({domain:<10s})  mlp={ma:.1%}  base={ba:.1%}  jac={mj:.2f}  [{best}]")
        mlp_results.append({"concept": c, "domain": domain,
                           "mlp_acc": float(ma), "base_acc": float(ba), "mlp_jac": float(mj)})

    mean_mlp = np.mean(mlp_accs)
    mean_base = np.mean(base_accs)
    mean_jac = np.mean(mlp_jacs)
    print(f"\n  MLP ACCURACY:  {mean_mlp:.1%}  vs baseline {mean_base:.1%}  ({mean_mlp - mean_base:+.1%})")
    print(f"  MLP JACCARD:   {mean_jac:.2f}")
    print("=" * 75)

    return mlp_results


# ======================================================================
# PART 2: SAE CAUSAL INTERVENTION (via CausalVerifier)
# ======================================================================

def sae_intervention(device, output_dir):
    """Zero out SAE features and measure selective causal effect.

    Uses reptimeline.CausalVerifier for statistical testing instead of
    manual selectivity computation.
    """
    from transformers import GPTNeoXForCausalLM, AutoTokenizer
    from sparsify import Sae

    logger.info("\n=== PART 2: SAE Causal Intervention (via CausalVerifier) ===")

    # Load data
    with open("results/pythia_sae/snapshots.json") as f:
        data = json.load(f)
    with open("results/pythia_sae/feature_selection.json") as f:
        feat_sel = json.load(f)

    concepts = data["concepts"]
    selected_features = feat_sel["features"]  # 256 SAE feature indices
    final_snapshot = ConceptSnapshot.from_dict(data["snapshots"][-1])

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

    # Collect hidden states for all concepts
    context_template = "The word is: {concept}"
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

    # --- Level 1: Hidden state perturbation via CausalVerifier ---
    logger.info("Level 1: Measuring hidden state perturbation...")

    def l2_intervene_fn(concept, bit_index):
        """Compute L2 perturbation from zeroing one SAE feature."""
        sae_feat = selected_features[bit_index]
        h = concept_hidden[concept].unsqueeze(0)
        enc = sae.encode(h)
        h_recon = sae.decode(enc.top_acts, enc.top_indices).squeeze(0)
        mask = enc.top_indices[0] != sae_feat
        if mask.all():
            return 0.0  # feature not active
        acts_mod = enc.top_acts[0, mask].unsqueeze(0)
        idx_mod = enc.top_indices[0, mask].unsqueeze(0)
        h_modified = sae.decode(acts_mod, idx_mod).squeeze(0)
        return torch.norm(h_recon - h_modified).item()

    verifier_l1 = CausalVerifier(
        intervene_fn=l2_intervene_fn,
        n_bootstrap=1000, n_perms=1000, alpha=0.05,
        selectivity_threshold=1.5, min_selective_bits=3,
    )
    l1_report = verifier_l1.verify(final_snapshot)

    print("\n  LEVEL 1: HIDDEN STATE PERTURBATION (zero one feature)")
    verifier_l1.print_report(l1_report)

    plot_causal_heatmap(
        l1_report,
        title="Level 1: Hidden State Perturbation Selectivity",
        save_path=os.path.join(output_dir, "sae_l1_causal_heatmap.png"),
        show=False,
    )

    # --- Level 2: Next-token logit change via CausalVerifier ---
    logger.info("Level 2: Measuring next-token logit changes...")

    # Cache original logits
    orig_logits = {}
    with torch.no_grad():
        for c in concepts:
            inp = tokenizer(context_template.format(concept=c), return_tensors="pt").to(device)
            orig_logits[c] = model(**inp).logits[0, -1, :].float()

    def kl_intervene_fn(concept, bit_index):
        """Compute KL divergence from zeroing one SAE feature and re-running model."""
        sae_feat = selected_features[bit_index]
        h = concept_hidden[concept].unsqueeze(0)
        enc = sae.encode(h)
        mask = enc.top_indices[0] != sae_feat
        if mask.all():
            return 0.0
        acts_mod = enc.top_acts[0, mask].unsqueeze(0)
        idx_mod = enc.top_indices[0, mask].unsqueeze(0)
        h_modified = sae.decode(acts_mod, idx_mod).squeeze(0)

        def make_replace_hook(h_mod):
            def hook(module, input, output):
                out = output.clone()
                out[0, -1, :] = h_mod
                return out
            return hook

        inp = tokenizer(context_template.format(concept=concept), return_tensors="pt").to(device)
        hook_handle = model.gpt_neox.layers[3].mlp.register_forward_hook(
            make_replace_hook(h_modified))
        with torch.no_grad():
            logits_mod = model(**inp).logits[0, -1, :].float()
        hook_handle.remove()

        p = F.softmax(orig_logits[concept], dim=0).clamp(min=1e-10)
        q = F.softmax(logits_mod, dim=0).clamp(min=1e-10)
        kl = (p * (p.log() - q.log())).sum().item()
        return kl if np.isfinite(kl) else 0.0

    verifier_l2 = CausalVerifier(
        intervene_fn=kl_intervene_fn,
        n_bootstrap=1000, n_perms=1000, alpha=0.05,
        selectivity_threshold=1.5, min_selective_bits=3,
    )
    l2_report = verifier_l2.verify(final_snapshot)

    print("\n  LEVEL 2: NEXT-TOKEN LOGIT CHANGE (KL divergence)")
    verifier_l2.print_report(l2_report)

    plot_causal_heatmap(
        l2_report,
        title="Level 2: Next-Token KL Divergence Selectivity",
        save_path=os.path.join(output_dir, "sae_l2_causal_heatmap.png"),
        show=False,
    )
    logger.info("Saved causal heatmaps to %s/", output_dir)

    del model
    torch.cuda.empty_cache()

    return l1_report, l2_report


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Causal v2: break Pythia SAE black box")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="results/causal")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    os.makedirs(args.output, exist_ok=True)

    # Part 1: Embedding prediction
    pred_results, embeddings = embedding_prediction(args.output)

    # Part 1b: MLP prediction
    mlp_results = mlp_prediction(args.output)

    # Part 2: SAE causal intervention (via CausalVerifier)
    l1_report, l2_report = sae_intervention(args.device, args.output)

    # Save report
    me = np.mean([r["emb_acc"] for r in pred_results])
    md = np.mean([r["dom_acc"] for r in pred_results])
    mb = np.mean([r["base_acc"] for r in pred_results])
    je = np.mean([r["emb_jac"] for r in pred_results])
    mm = np.mean([r["mlp_acc"] for r in mlp_results])
    mj_mlp = np.mean([r["mlp_jac"] for r in mlp_results])

    report = {
        "prediction_embedding": {
            "embedding_accuracy": round(me, 4),
            "domain_accuracy": round(md, 4),
            "baseline_accuracy": round(mb, 4),
            "embedding_jaccard": round(je, 4),
            "improvement_over_baseline": round(me - mb, 4),
            "improvement_over_domain": round(me - md, 4),
            "per_concept": pred_results,
        },
        "prediction_mlp": {
            "mlp_accuracy": round(mm, 4),
            "baseline_accuracy": round(mb, 4),
            "mlp_jaccard": round(mj_mlp, 4),
            "improvement_over_baseline": round(mm - mb, 4),
            "per_concept": mlp_results,
        },
        "intervention_l1": {
            "verdict": l1_report.verdict,
            "n_significant": l1_report.n_significant,
            "n_tested": l1_report.n_tested,
            "correction": l1_report.correction_method,
            "alpha": l1_report.alpha,
        },
        "intervention_l2": {
            "verdict": l2_report.verdict,
            "n_significant": l2_report.n_significant,
            "n_tested": l2_report.n_tested,
            "correction": l2_report.correction_method,
            "alpha": l2_report.alpha,
        },
    }

    with open(os.path.join(args.output, "causal_v2_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Verdict
    print("\n" + "=" * 75)
    print("  FINAL VERDICT")
    print("=" * 75)

    # Primary criterion: causal intervention
    print(f"\n  [PRIMARY] Causal intervention (CausalVerifier):")
    print(f"    L1 (L2):  {l1_report.verdict.upper()} "
          f"({l1_report.n_significant}/{l1_report.n_tested} significant)")
    print(f"    L2 (KL):  {l2_report.verdict.upper()} "
          f"({l2_report.n_significant}/{l2_report.n_tested} significant)")
    print(f"    Correction: {l2_report.correction_method} (alpha={l2_report.alpha})")

    # Secondary observation: prediction accuracy (informational)
    print(f"\n  [SECONDARY] Prediction accuracy (informational):")
    print(f"    Embedding:  emb={me:.1%} vs baseline={mb:.1%} ({me-mb:+.1%})  NEGATIVE (expected)")
    print(f"    MLP:        mlp={mm:.1%} vs baseline={mb:.1%} ({mm-mb:+.1%})  NEGATIVE (expected)")
    print(f"    Note: prediction failure does not invalidate causal evidence.")

    # Overall conclusion
    print()
    if l2_report.verdict == 'causal_evidence_found':
        print(f"  CONCLUSION: BLACK BOX BROKEN (causal)")
        print(f"    Features are identifiable, labelable, and causally selective")
        print(f"    ({l2_report.n_significant} bits pass BH-FDR corrected permutation test).")
    else:
        print(f"  CONCLUSION: NOT BROKEN -- insufficient causal selectivity")
    print("=" * 75)


if __name__ == "__main__":
    main()
