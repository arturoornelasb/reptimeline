#!/usr/bin/env python3
"""
Causal intervention on TriadicGPT's triadic bits.

Tests whether flipping individual bits in hidden space selectively
affects the model's language output (next-token predictions).

Uses reptimeline.CausalVerifier for statistical testing (bootstrap CI,
permutation tests, BH-FDR correction).

Method:
  1. Run model to get hidden states x after final LayerNorm
  2. For bit i, direction = W_triadic[i, :] (row of triadic head)
  3. Flip bit: delta = -2 * z_i * w_i / ||w_i||^2  (exact sign flip)
  4. Modified logits = lm_head(x + delta)
  5. Measure KL divergence between original and modified logits
  6. CausalVerifier tests selectivity with BH-FDR correction
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

# TriadicGPT model lives in triadic-microgpt repo
MICROGPT_SRC = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'triadic-microgpt', 'src'))
sys.path.insert(0, MICROGPT_SRC)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reptimeline.core import ConceptSnapshot
from reptimeline.causal import CausalVerifier
from reptimeline.viz.causal_heatmap import plot_causal_heatmap

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(ckpt_dir, device):
    """Load DanzaTriadicGPT from checkpoint."""
    import torch
    from torch_transformer import TriadicGPT, TriadicGPTConfig
    from fast_tokenizer import FastBPETokenizer as BPETokenizer

    ckpt = torch.load(os.path.join(ckpt_dir, 'model_best.pt'),
                      map_location=device, weights_only=True)
    cfg = ckpt['config']
    config = TriadicGPTConfig(
        vocab_size=cfg['vocab_size'],
        block_size=cfg['block_size'],
        n_layer=cfg['n_layer'],
        n_embd=cfg['n_embd'],
        n_head=cfg['n_head'],
        n_triadic_bits=cfg['n_triadic_bits'],
    )
    model = TriadicGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tokenizer = BPETokenizer.load(os.path.join(ckpt_dir, 'tokenizer.json'))

    logger.info(f"Loaded model: {cfg['n_layer']}L/{cfg['n_embd']}D, {cfg['n_triadic_bits']} bits, "
                f"step {ckpt['step']}, bit_acc={ckpt.get('bit_accuracy_test', '?'):.1%}")
    return model, tokenizer, config


def causal_intervention(ckpt_dir, anchors_path, primitivos_path, device_str, output_dir):
    """Main causal intervention experiment using CausalVerifier."""
    import torch
    import torch.nn.functional as F

    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer, config = load_model_and_tokenizer(ckpt_dir, device)

    # Load primitives (bit names)
    with open(primitivos_path) as f:
        prim_data = json.load(f)
    bit_names = {}
    for p in prim_data['primitivos']:
        bit_names[p['bit']] = p['nombre']

    # Load anchors (concept -> expected bits)
    with open(anchors_path) as f:
        anchors = json.load(f)
    anchor_concepts = {k: v for k, v in anchors.items() if not k.startswith('_')}

    concept_list = list(anchor_concepts.keys())
    logger.info(f"Concepts: {len(concept_list)}")
    logger.info(f"Bits: {config.n_triadic_bits}")

    # --- Get hidden states for all concepts ---
    W_triadic = model.triadic_head.weight.detach()  # (n_bits, n_embd)
    b_triadic = None
    if model.triadic_head.bias is not None:
        b_triadic = model.triadic_head.bias.detach()

    hidden_states = {}
    orig_logits = {}
    orig_proj = {}

    for word in concept_list:
        ids = tokenizer.encode(word, add_special=False)[:4]
        if not ids:
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)

        with torch.no_grad():
            B, T = x.shape
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            tok_emb = model.wte(x)
            pos_emb = model.wpe(pos)
            h = model.drop(tok_emb + pos_emb)
            for block in model.blocks:
                h = block(h)
            h = model.ln_f(h)

            h_last = h[0, -1, :]
            logits_last = model.lm_head(h_last)

            z = W_triadic @ h_last
            if b_triadic is not None:
                z = z + b_triadic
            proj = torch.tanh(z)

        hidden_states[word] = h_last.clone()
        orig_logits[word] = logits_last.clone()
        orig_proj[word] = proj.cpu().numpy()

    valid_concepts = [c for c in concept_list if c in hidden_states]
    logger.info(f"Valid concepts (tokenizable): {len(valid_concepts)}")

    # --- Build ConceptSnapshot from triadic projections ---
    codes = {}
    for c in valid_concepts:
        codes[c] = [int(v > 0) for v in orig_proj[c]]
    snapshot = ConceptSnapshot(step=0, codes=codes)

    # --- Build intervene_fn: flip bit -> measure KL divergence ---
    def kl_intervene_fn(concept, bit_index):
        """Flip triadic bit and measure KL divergence in output logits."""
        h_last = hidden_states[concept]
        w_i = W_triadic[bit_index]
        w_norm_sq = (w_i * w_i).sum()

        z_i = (w_i * h_last).sum()
        if b_triadic is not None:
            z_i = z_i + b_triadic[bit_index]

        # Flip: delta = -2 * z_i * w_i / ||w_i||^2
        delta = -2.0 * z_i * w_i / w_norm_sq
        h_modified = h_last + delta

        with torch.no_grad():
            logits_mod = model.lm_head(h_modified)

        p = F.softmax(orig_logits[concept], dim=0).clamp(min=1e-10)
        q = F.softmax(logits_mod, dim=0).clamp(min=1e-10)
        kl = (p * (p.log() - q.log())).sum().item()
        return kl if np.isfinite(kl) else 0.0

    # --- Run CausalVerifier ---
    logger.info("Running CausalVerifier on triadic bits...")
    verifier = CausalVerifier(
        intervene_fn=kl_intervene_fn,
        n_bootstrap=1000, n_perms=1000, alpha=0.05,
        selectivity_threshold=1.5, min_selective_bits=3,
    )
    report = verifier.verify(snapshot)

    print("\n" + "=" * 75)
    print("  TRIADIC CAUSAL INTERVENTION (via CausalVerifier)")
    print("=" * 75)
    verifier.print_report(report)

    # --- Visualization via plot_causal_heatmap ---
    plot_causal_heatmap(
        report,
        title="TriadicGPT: Causal Selectivity by Bit (KL Divergence)",
        save_path=os.path.join(output_dir, "triadic_causal_heatmap.png"),
        show=False,
    )
    logger.info("Saved triadic_causal_heatmap.png")

    # --- Save report ---
    n_selective = sum(1 for br in report.bit_results
                      if br.significant and br.selectivity >= 1.5)
    report_dict = {
        "backend": "TriadicGPT (DanzaTriadicGPT, D-A14 v2)",
        "checkpoint": ckpt_dir,
        "n_concepts": len(valid_concepts),
        "n_bits_tested": report.n_tested,
        "n_significant": report.n_significant,
        "verdict": report.verdict,
        "correction_method": report.correction_method,
        "alpha": report.alpha,
        "results": [
            {
                "bit": br.bit_index,
                "name": bit_names.get(br.bit_index, f"bit_{br.bit_index}"),
                "selectivity": round(br.selectivity, 2),
                "p_value": round(br.p_value, 4),
                "significant": br.significant,
                "effect_size": round(br.effect_size, 2),
                "mean_labeled": round(br.mean_labeled, 4),
                "mean_other": round(br.mean_other, 4),
            }
            for br in report.bit_results
        ],
    }

    with open(os.path.join(output_dir, "triadic_causal_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    # Verdict
    print("\n" + "=" * 75)
    print("  VERDICT")
    print("=" * 75)
    print(f"  {report.verdict.replace('_', ' ').upper()}")
    print(f"  Significant bits: {report.n_significant}/{report.n_tested}")
    print(f"  Correction: {report.correction_method} (alpha={report.alpha})")
    if report.verdict == 'insufficient_evidence':
        print(f"\n  EXPECTED NEGATIVE: TriadicGPT has parallel architecture.")
        print(f"  The triadic head is a SIDE projection from hidden states,")
        print(f"  not in the LM computation path. Flipping a triadic bit")
        print(f"  direction perturbs the shared hidden space but changes")
        print(f"  LM output non-selectively (mean KL > 0 but sel ~1.0x).")
        print(f"\n  TriadicGPT causality evidence is STRUCTURAL, not interventional:")
        print(f"  reptimeline discovers duals (vida/muerte, placer/dolor) that")
        print(f"  match the gold semantic axes from the training ontology.")
    else:
        print(f"  PASS: triadic bits are causally selective")
    print("=" * 75)

    return report


def main():
    parser = argparse.ArgumentParser(description="Causal intervention on TriadicGPT triadic bits")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to TriadicGPT checkpoint directory")
    parser.add_argument("--anchors", required=True,
                        help="Path to anclas.json anchor file")
    parser.add_argument("--primitivos", required=True,
                        help="Path to primitivos.json primitives file")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="results/triadic_causal")
    args = parser.parse_args()

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    causal_intervention(args.checkpoint, args.anchors, args.primitivos,
                        args.device, args.output)


if __name__ == "__main__":
    main()
