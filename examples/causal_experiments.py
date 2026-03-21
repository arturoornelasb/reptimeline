#!/usr/bin/env python3
"""
Causal experiments: Can we CONTROL and PREDICT representations?

Experiment A — MNIST Intervention:
    Flip specific bits in Binary AE codes and decode.
    If bit 0 = "even/odd", flipping it on digit "4" should produce
    a reconstruction that looks more like an odd digit.

Experiment B — Pythia Holdout Prediction:
    Train bit-domain profiles on 50 concepts, predict binary codes
    for 10 holdout concepts, measure accuracy.

Usage:
    python examples/causal_experiments.py --device cuda
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ======================================================================
# EXPERIMENT A: MNIST BIT INTERVENTION
# ======================================================================

def experiment_a(device="cuda", output_dir="results/causal"):
    """Flip bits in MNIST Binary AE codes and measure causal effect."""
    from mnist_binary_ae import BinaryAE
    from torchvision import datasets, transforms

    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== EXPERIMENT A: MNIST Bit Intervention ===")

    # Load model
    ckpt = "results/mnist_bae/checkpoints/model_step10.pt"
    model = BinaryAE(input_dim=784, hidden=256, bottleneck=32)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Load MNIST test set
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    test_data = datasets.MNIST("data/mnist", train=False, download=True, transform=transform)
    digit_images = {d: [] for d in range(10)}
    for img, label in test_data:
        digit_images[label].append(img)
    digit_images = {d: torch.stack(imgs) for d, imgs in digit_images.items()}

    # Get prototype codes per digit (majority vote)
    digit_codes = {}
    digit_mean_codes = {}  # continuous version for distance
    for d in range(10):
        imgs = digit_images[d].to(device)
        with torch.no_grad():
            z = model.encoder(imgs)
            z_sig = torch.sigmoid(z)
            z_bin = (z_sig > 0.5).float()
        digit_codes[d] = (z_bin.mean(dim=0) > 0.5).float()
        digit_mean_codes[d] = z_sig.mean(dim=0)

    # Compute prototype reconstructions for each digit
    digit_protos = {}
    for d in range(10):
        with torch.no_grad():
            digit_protos[d] = model.decoder(digit_codes[d].unsqueeze(0)).squeeze(0)

    # --- Identify interesting bits ---
    # Bit 0: even/odd discovered earlier
    # Also find the strongest duals from discovery
    with open("results/mnist_bae/discovery.json") as f:
        disc = json.load(f)

    # Top duals to test
    test_bits = []

    # Bit 0 (even=0,2,4,6,8 vs odd=1,3,5,7,9)
    code_matrix = torch.stack([digit_codes[d] for d in range(10)])  # (10, 32)
    for bit_idx in range(32):
        vals = code_matrix[:, bit_idx].cpu().numpy()
        even_rate = np.mean([vals[d] for d in [0, 2, 4, 6, 8]])
        odd_rate = np.mean([vals[d] for d in [1, 3, 5, 7, 9]])
        if abs(even_rate - odd_rate) > 0.6:
            test_bits.append({"bit": bit_idx, "name": f"bit{bit_idx}_even/odd",
                              "even_rate": even_rate, "odd_rate": odd_rate})

    # Top duals from discovery
    for dual in disc.get("duals", [])[:3]:
        test_bits.append({"bit": dual["bit_a"], "name": f"dual_a{dual['bit_a']}_b{dual['bit_b']}",
                          "corr": dual["corr"]})

    logger.info(f"Testing {len(test_bits)} bits for causal effect")
    for tb in test_bits:
        logger.info(f"  {tb}")

    # --- Run interventions ---
    results = []
    all_figures = []

    for tb in test_bits[:6]:  # Top 6 bits
        bit_idx = tb["bit"]
        bit_name = tb["name"]

        digit_results = []
        for source_digit in range(10):
            code = digit_codes[source_digit].clone()
            original_val = code[bit_idx].item()

            # Flip the bit
            code[bit_idx] = 1.0 - code[bit_idx]
            flipped_val = code[bit_idx].item()

            # Decode original and flipped
            with torch.no_grad():
                recon_orig = model.decoder(digit_codes[source_digit].unsqueeze(0)).squeeze(0)
                recon_flip = model.decoder(code.unsqueeze(0)).squeeze(0)

            # Classify: which digit prototype is the flipped reconstruction closest to?
            distances = {}
            for target_d in range(10):
                dist = torch.nn.functional.mse_loss(recon_flip, digit_protos[target_d]).item()
                distances[target_d] = dist

            closest = min(distances, key=distances.get)
            shifted = closest != source_digit

            digit_results.append({
                "source": source_digit,
                "original_bit": int(original_val),
                "flipped_bit": int(flipped_val),
                "closest_after_flip": closest,
                "shifted": shifted,
                "distances": {str(k): round(v, 6) for k, v in distances.items()},
            })

        # Count how many digits shifted class
        n_shifted = sum(1 for r in digit_results if r["shifted"])
        results.append({
            "bit": bit_idx,
            "name": bit_name,
            "n_shifted": n_shifted,
            "shift_rate": n_shifted / 10,
            "details": digit_results,
        })
        logger.info(f"  bit {bit_idx} ({bit_name}): {n_shifted}/10 digits shifted class")

    # --- Visualize top intervention ---
    # Find the bit with highest shift rate
    best = max(results, key=lambda r: r["shift_rate"])
    bit_idx = best["bit"]
    logger.info(f"\nBest causal bit: {best['bit']} ({best['name']}), shift rate: {best['shift_rate']:.0%}")

    fig, axes = plt.subplots(3, 10, figsize=(20, 6))
    fig.suptitle(f"Intervention: Flip bit {bit_idx} ({best['name']})", fontsize=14)

    for d in range(10):
        # Row 1: Original image (mean of class)
        orig_img = digit_images[d].mean(dim=0).cpu().numpy().reshape(28, 28)
        axes[0, d].imshow(orig_img, cmap="gray")
        axes[0, d].set_title(f"digit {d}")
        axes[0, d].axis("off")

        # Row 2: Original reconstruction
        with torch.no_grad():
            recon_orig = model.decoder(digit_codes[d].unsqueeze(0)).squeeze(0)
        axes[1, d].imshow(recon_orig.cpu().numpy().reshape(28, 28), cmap="gray")
        val = int(digit_codes[d][bit_idx].item())
        axes[1, d].set_title(f"b{bit_idx}={val}")
        axes[1, d].axis("off")

        # Row 3: Flipped reconstruction
        code_flip = digit_codes[d].clone()
        code_flip[bit_idx] = 1.0 - code_flip[bit_idx]
        with torch.no_grad():
            recon_flip = model.decoder(code_flip.unsqueeze(0)).squeeze(0)
        detail = best["details"][d]
        closest = detail["closest_after_flip"]
        color = "green" if detail["shifted"] else "gray"
        axes[2, d].imshow(recon_flip.cpu().numpy().reshape(28, 28), cmap="gray")
        axes[2, d].set_title(f"-> {closest}", color=color, fontweight="bold")
        axes[2, d].axis("off")

    axes[0, 0].set_ylabel("Mean input", fontsize=10)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=10)
    axes[2, 0].set_ylabel("Bit flipped", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mnist_intervention.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved intervention visualization")

    # --- Aggregate: flip EACH bit, measure average shift rate ---
    logger.info("\nScanning ALL 32 bits for causal effect...")
    bit_shift_rates = []
    for bit_idx in range(32):
        n_shifted = 0
        for source_digit in range(10):
            code = digit_codes[source_digit].clone()
            code[bit_idx] = 1.0 - code[bit_idx]
            with torch.no_grad():
                recon_flip = model.decoder(code.unsqueeze(0)).squeeze(0)
            distances = {td: torch.nn.functional.mse_loss(recon_flip, digit_protos[td]).item()
                         for td in range(10)}
            closest = min(distances, key=distances.get)
            if closest != source_digit:
                n_shifted += 1
        bit_shift_rates.append({"bit": bit_idx, "shift_rate": n_shifted / 10, "n_shifted": n_shifted})

    bit_shift_rates.sort(key=lambda x: -x["shift_rate"])
    causal_bits = [b for b in bit_shift_rates if b["shift_rate"] > 0.3]

    print("\n" + "=" * 60)
    print("  EXPERIMENT A: CAUSAL BIT SCAN")
    print("=" * 60)
    print(f"  Bits with shift_rate > 30%: {len(causal_bits)} / 32")
    print()
    for b in bit_shift_rates[:15]:
        bar = "#" * int(b["shift_rate"] * 20)
        print(f"    bit {b['bit']:>2d}  shift={b['shift_rate']:.0%}  "
              f"({b['n_shifted']}/10 digits changed)  {bar}")
    print("=" * 60)

    return results, bit_shift_rates


# ======================================================================
# EXPERIMENT B: PYTHIA HOLDOUT PREDICTION
# ======================================================================

def experiment_b(output_dir="results/causal"):
    """Predict binary codes for holdout concepts using domain profiles."""
    from reptimeline.core import ConceptSnapshot

    os.makedirs(output_dir, exist_ok=True)
    logger.info("\n=== EXPERIMENT B: Pythia Holdout Prediction ===")

    # Load data
    with open("results/pythia_sae/snapshots.json") as f:
        data = json.load(f)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "pythia_concepts.json")) as f:
        concepts_data = json.load(f)

    final_snap = ConceptSnapshot.from_dict(data["snapshots"][-1])
    domains = concepts_data["domains"]
    concept_to_domain = {c: d for d, words in domains.items() for c in words}

    # Stratified holdout: 2 concepts per domain (last 2)
    train_concepts = []
    test_concepts = []
    for domain, words in domains.items():
        train_concepts.extend(words[:-2])
        test_concepts.extend(words[-2:])

    logger.info(f"Train: {len(train_concepts)} concepts, Test: {len(test_concepts)} concepts")
    logger.info(f"Test concepts: {test_concepts}")

    # Get codes
    code_dim = len(final_snap.codes[train_concepts[0]])
    train_codes = {c: final_snap.codes[c] for c in train_concepts if c in final_snap.codes}
    test_codes = {c: final_snap.codes[c] for c in test_concepts if c in final_snap.codes}

    # --- Learn domain profiles ---
    # For each bit, compute P(bit=1 | domain) from training set
    domain_bit_profiles = {}  # domain -> array of P(bit=1)
    for domain in domains:
        domain_concepts = [c for c in train_concepts if concept_to_domain[c] == domain
                           and c in train_codes]
        if not domain_concepts:
            continue
        codes_matrix = np.array([train_codes[c] for c in domain_concepts])
        domain_bit_profiles[domain] = codes_matrix.mean(axis=0)  # P(bit=1) per bit

    # --- Predict test codes ---
    predictions = {}
    for concept in test_concepts:
        if concept not in test_codes:
            continue
        domain = concept_to_domain[concept]
        if domain not in domain_bit_profiles:
            continue
        # Predict: bit=1 if P(bit=1|domain) > 0.5
        profile = domain_bit_profiles[domain]
        predicted = [1 if p > 0.5 else 0 for p in profile]
        predictions[concept] = predicted

    # --- Evaluate ---
    bit_accuracies = []
    concept_results = []

    for concept in test_concepts:
        if concept not in predictions or concept not in test_codes:
            continue
        pred = predictions[concept]
        actual = test_codes[concept]

        # Bit-level accuracy
        matches = sum(1 for p, a in zip(pred, actual) if p == a)
        bit_acc = matches / len(pred)
        bit_accuracies.append(bit_acc)

        # Hamming distance
        hamming = sum(1 for p, a in zip(pred, actual) if p != a)

        # Active bit overlap (Jaccard)
        pred_active = set(i for i, v in enumerate(pred) if v == 1)
        actual_active = set(i for i, v in enumerate(actual) if v == 1)
        union = pred_active | actual_active
        jaccard = len(pred_active & actual_active) / len(union) if union else 1.0

        concept_results.append({
            "concept": concept,
            "domain": concept_to_domain[concept],
            "bit_accuracy": round(bit_acc, 4),
            "hamming_distance": hamming,
            "jaccard": round(jaccard, 4),
            "n_pred_active": len(pred_active),
            "n_actual_active": len(actual_active),
            "n_overlap": len(pred_active & actual_active),
        })

    # --- Random baseline ---
    # Predict using overall bit rate (not domain-specific)
    all_train_codes = np.array([train_codes[c] for c in train_codes])
    overall_rate = all_train_codes.mean(axis=0)
    random_accuracies = []
    for concept in test_concepts:
        if concept not in test_codes:
            continue
        actual = test_codes[concept]
        random_pred = [1 if p > 0.5 else 0 for p in overall_rate]
        matches = sum(1 for p, a in zip(random_pred, actual) if p == a)
        random_accuracies.append(matches / len(actual))

    mean_bit_acc = np.mean(bit_accuracies) if bit_accuracies else 0
    mean_random_acc = np.mean(random_accuracies) if random_accuracies else 0
    improvement = mean_bit_acc - mean_random_acc

    print("\n" + "=" * 60)
    print("  EXPERIMENT B: HOLDOUT PREDICTION")
    print("=" * 60)
    print(f"  Train: {len(train_concepts)} concepts | Test: {len(test_concepts)} concepts")
    print(f"  Code dimension: {code_dim} bits")
    print()
    print(f"  Domain-based prediction accuracy: {mean_bit_acc:.1%}")
    print(f"  Overall-rate baseline accuracy:   {mean_random_acc:.1%}")
    print(f"  Improvement over baseline:        +{improvement:.1%}")
    print()
    print("  Per-concept results:")
    print("  " + "-" * 56)
    for r in sorted(concept_results, key=lambda x: -x["bit_accuracy"]):
        print(f"    {r['concept']:<12s}  domain={r['domain']:<10s}  "
              f"bit_acc={r['bit_accuracy']:.1%}  jaccard={r['jaccard']:.2f}  "
              f"hamming={r['hamming_distance']}")

    # --- Per-domain accuracy ---
    print()
    print("  Per-domain accuracy:")
    print("  " + "-" * 56)
    domain_accs = defaultdict(list)
    for r in concept_results:
        domain_accs[r["domain"]].append(r["bit_accuracy"])
    for domain in sorted(domain_accs):
        mean_acc = np.mean(domain_accs[domain])
        print(f"    {domain:<12s}  {mean_acc:.1%}")

    print("=" * 60)

    return concept_results, mean_bit_acc, mean_random_acc


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Causal experiments for reptimeline")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="results/causal")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Experiment A
    intervention_results, bit_scan = experiment_a(device=args.device, output_dir=args.output)

    # Experiment B
    prediction_results, domain_acc, baseline_acc = experiment_b(output_dir=args.output)

    # Save all results
    report = {
        "experiment_a": {
            "description": "Flip bits in MNIST Binary AE, measure class shift",
            "bit_scan": bit_scan,
            "causal_bits": len([b for b in bit_scan if b["shift_rate"] > 0.3]),
            "total_bits": 32,
            "top_interventions": intervention_results[:3],
        },
        "experiment_b": {
            "description": "Predict Pythia SAE codes from domain membership",
            "domain_prediction_accuracy": round(domain_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "improvement": round(domain_acc - baseline_acc, 4),
            "per_concept": prediction_results,
        },
    }

    report_path = os.path.join(args.output, "causal_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nSaved full report to {report_path}")

    # Final verdict
    causal_rate = len([b for b in bit_scan if b["shift_rate"] > 0.3]) / 32
    print("\n" + "=" * 60)
    print("  VERDICT: CAN WE BREAK THE BLACK BOX?")
    print("=" * 60)
    print(f"  A. Intervention: {causal_rate:.0%} of bits have causal effect (>{30}% shift)")
    print(f"  B. Prediction:   {domain_acc:.1%} bit accuracy ({domain_acc - baseline_acc:+.1%} vs baseline)")
    both_pass = causal_rate > 0.2 and domain_acc > baseline_acc + 0.05
    print(f"\n  {'PASS' if both_pass else 'PARTIAL'}: "
          f"{'Primitives are causal AND predictive' if both_pass else 'More work needed'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
