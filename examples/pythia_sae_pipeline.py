#!/usr/bin/env python3
"""
Pythia-70M + SAE → reptimeline: End-to-end pipeline.

Downloads pre-trained Pythia-70M checkpoints and SAEs from HuggingFace,
extracts discrete binary codes for 60 concepts across 12 training steps,
and runs full reptimeline analysis (timeline + discovery + plots).

Zero training required — everything is pre-trained and public.

Usage:
    python examples/pythia_sae_pipeline.py --layer 3 --device cuda
    python examples/pythia_sae_pipeline.py --device cpu --skip-plots

Requirements:
    pip install transformers torch numpy matplotlib
    pip install "git+https://github.com/EleutherAI/sae.git"
"""

import argparse
import json
import logging
import os
import sys
import time

# Add parent directory so reptimeline is importable when running from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sae_extractor import PythiaSAEExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_concepts(path: str):
    """Load concepts and checkpoints from JSON config."""
    with open(path) as f:
        data = json.load(f)

    concepts = []
    for domain_words in data["domains"].values():
        concepts.extend(domain_words)

    return concepts, data["domains"], data["checkpoints"], data.get("context_template")


def verify_tokenization(concepts, model_name, cache_dir=None):
    """Check that all concepts are single tokens. Drop multi-token ones."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    single_token = []
    multi_token = []

    for concept in concepts:
        # Tokenize with space prefix (how it appears in context)
        tokens = tokenizer.encode(f" {concept}", add_special_tokens=False)
        if len(tokens) == 1:
            single_token.append(concept)
        else:
            multi_token.append((concept, len(tokens)))

    if multi_token:
        logger.warning(
            f"Dropping {len(multi_token)} multi-token concepts: "
            f"{[c for c, _ in multi_token]}"
        )

    logger.info(f"Using {len(single_token)} single-token concepts")
    return single_token


def run_pipeline(args):
    """Main pipeline: extract → analyze → discover → plot."""
    from reptimeline import BitDiscovery, TimelineTracker
    from reptimeline.core import ConceptSnapshot

    t_start = time.time()

    # -- Setup --------------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    concepts_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pythia_concepts.json"
    )

    concepts, domains, checkpoints, template = load_concepts(concepts_path)
    logger.info(f"Loaded {len(concepts)} concepts across {len(domains)} domains")
    first_step = checkpoints[0]['step']
    last_step = checkpoints[-1]['step']
    logger.info(
        f"Checkpoints: {len(checkpoints)} "
        f"(step {first_step} -> {last_step})"
    )

    # Verify tokenization
    concepts = verify_tokenization(
        concepts, PythiaSAEExtractor.MODEL_NAME, cache_dir=args.cache_dir
    )

    # -- Load or extract snapshots ------------------------------------
    snapshots_path = os.path.join(args.output, "snapshots.json")
    features_path = os.path.join(args.output, "feature_selection.json")

    if args.load_snapshots and os.path.exists(snapshots_path):
        logger.info(f"Loading cached snapshots from {snapshots_path}")
        with open(snapshots_path) as f:
            snap_data = json.load(f)
        snapshots = [ConceptSnapshot.from_dict(s) for s in snap_data["snapshots"]]
        logger.info(f"Loaded {len(snapshots)} snapshots")
    else:
        # Create extractor
        extractor = PythiaSAEExtractor(
            layer=args.layer,
            top_k=args.top_k,
            device=args.device,
            context_template=template or "The word is: {concept}",
            cache_dir=args.cache_dir,
        )

        # Load cached feature selection if available
        if os.path.exists(features_path):
            logger.info(f"Loading cached feature selection from {features_path}")
            extractor.load_feature_selection(features_path)

        # Extract all checkpoints
        snapshots = extractor.extract_checkpoints(
            concepts=concepts,
            checkpoints=checkpoints,
            select_revision=checkpoints[-1]["revision"],
        )

        # Save feature selection
        extractor.save_feature_selection(features_path)

        # Save snapshots
        snap_data = {
            "model": PythiaSAEExtractor.MODEL_NAME,
            "sae": PythiaSAEExtractor.SAE_NAME,
            "layer": args.layer,
            "top_k": args.top_k,
            "n_concepts": len(concepts),
            "concepts": concepts,
            "domains": domains,
            "snapshots": [s.to_dict() for s in snapshots],
        }
        with open(snapshots_path, "w") as f:
            json.dump(snap_data, f, indent=2)
        logger.info(f"Saved snapshots to {snapshots_path}")

    t_extract = time.time()
    logger.info(f"Extraction done in {t_extract - t_start:.1f}s")

    # -- Timeline analysis --------------------------------------------
    logger.info("=== Timeline Analysis ===")

    # Use a lightweight extractor for analysis (no model needed)
    from reptimeline.extractors.base import RepresentationExtractor

    class JaccardExtractor(RepresentationExtractor):
        def extract(self, checkpoint_path, concepts, device="cpu"):
            raise NotImplementedError

        def similarity(self, code_a, code_b):
            a = set(i for i, v in enumerate(code_a) if v == 1)
            b = set(i for i, v in enumerate(code_b) if v == 1)
            u = a | b
            return len(a & b) / len(u) if u else 1.0

        def shared_features(self, code_a, code_b):
            return [i for i in range(min(len(code_a), len(code_b)))
                    if code_a[i] == 1 and code_b[i] == 1]

    tracker = TimelineTracker(JaccardExtractor())
    timeline = tracker.analyze(snapshots)
    timeline.print_summary()

    # Save timeline
    timeline_path = os.path.join(args.output, "timeline.json")
    with open(timeline_path, "w") as f:
        json.dump(timeline.to_dict(), f, indent=2)
    logger.info(f"Saved timeline to {timeline_path}")

    t_timeline = time.time()

    # -- BitDiscovery -------------------------------------------------
    logger.info("=== BitDiscovery ===")
    discovery = BitDiscovery()
    report = discovery.discover(snapshots[-1], timeline=timeline, top_k=10)
    discovery.print_report(report)

    # Save discovery report
    discovery_path = os.path.join(args.output, "discovery.json")
    report_dict = {
        "n_concepts": len(snapshots[-1].codes),
        "code_dimension": args.top_k,
        "n_active_bits": report.n_active_bits,
        "n_dead_bits": report.n_dead_bits,
        "n_duals": len(report.discovered_duals),
        "n_deps": len(report.discovered_deps),
        "n_triadic": len(report.discovered_triadic_deps),
        "n_hierarchy": len(report.discovered_hierarchy),
        "duals": [
            {"bit_a": d.bit_a, "bit_b": d.bit_b, "corr": d.anti_correlation}
            for d in report.discovered_duals
        ],
        "deps": [
            {"parent": d.bit_parent, "child": d.bit_child, "confidence": d.confidence}
            for d in report.discovered_deps
        ],
        "triadic": [
            {"bit_i": t.bit_i, "bit_j": t.bit_j, "bit_r": t.bit_r}
            for t in report.discovered_triadic_deps
        ],
        "bit_semantics": [
            {
                "bit": bs.bit_index,
                "activation_rate": round(bs.activation_rate, 4),
                "top_concepts": bs.top_concepts[:5],
            }
            for bs in report.bit_semantics[:20]
        ],
    }
    with open(discovery_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    logger.info(f"Saved discovery to {discovery_path}")

    t_discovery = time.time()

    # -- Plots --------------------------------------------------------
    if not args.skip_plots:
        logger.info("=== Generating Plots ===")
        plots_dir = os.path.join(args.output, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        try:
            from reptimeline.viz.phase_dashboard import plot_phase_dashboard
            fig = plot_phase_dashboard(timeline)
            fig.savefig(
                os.path.join(plots_dir, "phase_dashboard.png"), dpi=150, bbox_inches="tight"
            )
            logger.info("Saved phase_dashboard.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Phase dashboard failed: {e}")

        try:
            from reptimeline.viz.swimlane import plot_swimlane
            fig = plot_swimlane(timeline, max_bits=30)
            fig.savefig(
                os.path.join(plots_dir, "swimlane.png"), dpi=150, bbox_inches="tight"
            )
            logger.info("Saved swimlane.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Swimlane failed: {e}")

        try:
            from reptimeline.viz.churn_heatmap import plot_churn_heatmap
            fig = plot_churn_heatmap(timeline)
            fig.savefig(
                os.path.join(plots_dir, "churn_heatmap.png"), dpi=150, bbox_inches="tight"
            )
            logger.info("Saved churn_heatmap.png")
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Churn heatmap failed: {e}")

    # -- Summary ------------------------------------------------------
    t_end = time.time()
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model:       {PythiaSAEExtractor.MODEL_NAME}")
    print(f"  SAE:         {PythiaSAEExtractor.SAE_NAME}")
    print(f"  Layer:       {args.layer}")
    print(f"  Code dim:    {args.top_k}")
    print(f"  Concepts:    {len(snapshots[-1].codes)}")
    print(f"  Checkpoints: {len(snapshots)}")
    print(f"  Births:      {len(timeline.births)}")
    print(f"  Deaths:      {len(timeline.deaths)}")
    print(f"  Connections: {len(timeline.connections)}")
    print(f"  Phase trans: {len(timeline.phase_transitions)}")
    print(f"  Duals:       {len(report.discovered_duals)}")
    print(f"  Dependencies:{len(report.discovered_deps)}")
    print(f"  Dead bits:   {report.n_dead_bits} / {args.top_k}")
    print(f"  Time:        {t_end - t_start:.1f}s total")
    print(f"    Extract:   {t_extract - t_start:.1f}s")
    print(f"    Timeline:  {t_timeline - t_extract:.1f}s")
    print(f"    Discovery: {t_discovery - t_timeline:.1f}s")
    print(f"  Output:      {os.path.abspath(args.output)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Pythia-70M + SAE → reptimeline analysis pipeline"
    )
    parser.add_argument(
        "--layer", type=int, default=3, help="Pythia layer for SAE (0-5, default: 3)"
    )
    parser.add_argument(
        "--top-k", type=int, default=256, help="Number of SAE features to track (default: 256)"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cuda", "cpu"], help="Device (default: cpu)"
    )
    parser.add_argument(
        "--output",
        default="results/pythia_sae",
        help="Output directory (default: results/pythia_sae)",
    )
    parser.add_argument(
        "--cache-dir", default=None, help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--load-snapshots",
        action="store_true",
        help="Load cached snapshots instead of re-extracting",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Skip plot generation"
    )
    args = parser.parse_args()

    # Validate CUDA
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            args.device = "cpu"
        else:
            gpu = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu.name}, {gpu.total_memory / 1e9:.1f} GB")

    run_pipeline(args)


if __name__ == "__main__":
    main()
