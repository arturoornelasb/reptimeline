#!/usr/bin/env python3
"""
MNIST Binary Autoencoder -> reptimeline: End-to-end pipeline.

Trains a binary autoencoder on MNIST (or loads cached checkpoints),
extracts discrete binary codes for 10 digit classes across 6 training
epochs, and runs full reptimeline analysis.

Usage:
    python examples/mnist_pipeline.py --device cuda
    python examples/mnist_pipeline.py --load-snapshots --skip-plots
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mnist_binary_ae import train_binary_ae
from mnist_extractor import MNISTBinaryAEExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_pipeline(args):
    """Main pipeline: train -> extract -> analyze -> discover -> plot."""
    from reptimeline import TimelineTracker, BitDiscovery
    from reptimeline.core import ConceptSnapshot

    t_start = time.time()
    os.makedirs(args.output, exist_ok=True)

    concepts = [str(d) for d in range(10)]
    checkpoint_dir = os.path.join(args.output, "checkpoints")
    snapshots_path = os.path.join(args.output, "snapshots.json")

    # -- Train or load snapshots -----------------------------------------
    if args.load_snapshots and os.path.exists(snapshots_path):
        logger.info(f"Loading cached snapshots from {snapshots_path}")
        with open(snapshots_path) as f:
            snap_data = json.load(f)
        snapshots = [ConceptSnapshot.from_dict(s) for s in snap_data["snapshots"]]
        logger.info(f"Loaded {len(snapshots)} snapshots")
    else:
        # Train if checkpoints don't exist
        if not os.path.exists(os.path.join(checkpoint_dir, "model_step0.pt")) or args.force_train:
            logger.info("=== Training Binary AE ===")
            train_binary_ae(
                output_dir=checkpoint_dir,
                epochs=args.epochs,
                save_every=2,
                bottleneck=args.bottleneck,
                device=args.device,
            )

        # Extract snapshots from checkpoints
        logger.info("=== Extracting snapshots ===")
        extractor = MNISTBinaryAEExtractor(
            bottleneck=args.bottleneck, device=args.device,
        )

        # Discover checkpoints
        checkpoint_files = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.startswith("model_step")],
            key=lambda f: int(f.replace("model_step", "").replace(".pt", "")),
        )
        logger.info(f"Found {len(checkpoint_files)} checkpoints")

        snapshots = []
        for fname in checkpoint_files:
            path = os.path.join(checkpoint_dir, fname)
            snap = extractor.extract(path, concepts)
            snapshots.append(snap)
            active = sum(1 for c in snap.codes.values() for b in c if b == 1)
            total = sum(len(c) for c in snap.codes.values())
            logger.info(f"  {fname}: step={snap.step}, "
                        f"active_bits={active}/{total}")

        # Save snapshots
        snap_data = {
            "model": "BinaryAE",
            "dataset": "MNIST",
            "bottleneck": args.bottleneck,
            "n_concepts": len(concepts),
            "concepts": concepts,
            "snapshots": [s.to_dict() for s in snapshots],
        }
        with open(snapshots_path, "w") as f:
            json.dump(snap_data, f, indent=2)
        logger.info(f"Saved snapshots to {snapshots_path}")

    t_extract = time.time()

    # -- Timeline analysis -----------------------------------------------
    logger.info("=== Timeline Analysis ===")
    extractor_for_analysis = MNISTBinaryAEExtractor(bottleneck=args.bottleneck)
    tracker = TimelineTracker(extractor_for_analysis)
    timeline = tracker.analyze(snapshots)
    timeline.print_summary()

    timeline_path = os.path.join(args.output, "timeline.json")
    with open(timeline_path, "w") as f:
        json.dump(timeline.to_dict(), f, indent=2)

    t_timeline = time.time()

    # -- BitDiscovery ----------------------------------------------------
    logger.info("=== BitDiscovery ===")
    discovery = BitDiscovery()
    report = discovery.discover(snapshots[-1], timeline=timeline, top_k=5)
    discovery.print_report(report)

    discovery_path = os.path.join(args.output, "discovery.json")
    report_dict = {
        "n_concepts": len(snapshots[-1].codes),
        "code_dimension": args.bottleneck,
        "n_active_bits": report.n_active_bits,
        "n_dead_bits": report.n_dead_bits,
        "n_duals": len(report.discovered_duals),
        "n_deps": len(report.discovered_deps),
        "n_triadic": len(report.discovered_triadic_deps),
        "duals": [
            {"bit_a": d.bit_a, "bit_b": d.bit_b, "corr": d.anti_correlation}
            for d in report.discovered_duals
        ],
        "deps": [
            {"parent": d.bit_parent, "child": d.bit_child, "confidence": d.confidence}
            for d in report.discovered_deps
        ],
        "bit_semantics": [
            {"bit": bs.bit_index, "activation_rate": round(bs.activation_rate, 4),
             "top_concepts": bs.top_concepts[:5]}
            for bs in report.bit_semantics if bs.activation_rate > 0.02
        ],
    }
    with open(discovery_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    t_discovery = time.time()

    # -- Plots -----------------------------------------------------------
    if not args.skip_plots:
        logger.info("=== Generating Plots ===")
        plots_dir = os.path.join(args.output, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        import matplotlib.pyplot as plt

        try:
            from reptimeline.viz.phase_dashboard import plot_phase_dashboard
            fig = plot_phase_dashboard(timeline)
            fig.savefig(os.path.join(plots_dir, "phase_dashboard.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved phase_dashboard.png")
        except Exception as e:
            logger.warning(f"Phase dashboard failed: {e}")

        try:
            from reptimeline.viz.swimlane import plot_swimlane
            fig = plot_swimlane(timeline, max_bits=args.bottleneck)
            fig.savefig(os.path.join(plots_dir, "swimlane.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved swimlane.png")
        except Exception as e:
            logger.warning(f"Swimlane failed: {e}")

        try:
            from reptimeline.viz.churn_heatmap import plot_churn_heatmap
            fig = plot_churn_heatmap(timeline)
            fig.savefig(os.path.join(plots_dir, "churn_heatmap.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved churn_heatmap.png")
        except Exception as e:
            logger.warning(f"Churn heatmap failed: {e}")

    # -- Summary ---------------------------------------------------------
    t_end = time.time()
    print("\n" + "=" * 60)
    print("MNIST PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model:       BinaryAE ({args.bottleneck}-bit)")
    print(f"  Dataset:     MNIST")
    print(f"  Concepts:    {len(snapshots[-1].codes)} digit classes")
    print(f"  Checkpoints: {len(snapshots)}")
    print(f"  Births:      {len(timeline.births)}")
    print(f"  Deaths:      {len(timeline.deaths)}")
    print(f"  Connections: {len(timeline.connections)}")
    print(f"  Phase trans: {len(timeline.phase_transitions)}")
    print(f"  Duals:       {len(report.discovered_duals)}")
    print(f"  Dependencies:{len(report.discovered_deps)}")
    print(f"  Dead bits:   {report.n_dead_bits} / {args.bottleneck}")
    print(f"  Time:        {t_end - t_start:.1f}s total")
    print(f"  Output:      {os.path.abspath(args.output)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MNIST Binary AE -> reptimeline analysis pipeline"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bottleneck", type=int, default=32, help="Number of binary bits")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="results/mnist_bae")
    parser.add_argument("--load-snapshots", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            args.device = "cpu"

    run_pipeline(args)


if __name__ == "__main__":
    main()
