"""
Command-line interface for reptimeline.

Usage:
    # Analyze pre-extracted snapshots from JSON
    reptimeline --snapshots timeline_snapshots.json

    # Analyze with plots
    reptimeline --snapshots timeline_snapshots.json --plot

    # Run discovery on the last snapshot
    reptimeline --snapshots timeline_snapshots.json --discover

    # Full pipeline with overlay (requires primitivos.json)
    reptimeline --snapshots timeline_snapshots.json --discover --overlay primitivos.json --plot
"""

import argparse
import json
import os
import sys

from reptimeline.core import ConceptSnapshot, Timeline
from reptimeline.extractors.base import RepresentationExtractor
from reptimeline.tracker import TimelineTracker


class _JaccardExtractor(RepresentationExtractor):
    """Default extractor using Jaccard similarity on active bits.

    Used when analyzing from JSON snapshots (no model loading needed).
    """

    def extract(self, checkpoint_path, concepts, device='cpu'):
        raise NotImplementedError("Use --snapshots to provide pre-extracted data")

    def similarity(self, code_a, code_b):
        active_a = set(i for i, v in enumerate(code_a) if v == 1)
        active_b = set(i for i, v in enumerate(code_b) if v == 1)
        union = active_a | active_b
        if not union:
            return 1.0
        return len(active_a & active_b) / len(union)

    def shared_features(self, code_a, code_b):
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]


def _load_snapshots(path: str):
    """Load snapshots from a JSON file.

    Expected format:
    {
        "snapshots": [
            {"step": 1000, "codes": {"king": [0,1,1,...], "queen": [1,0,1,...]}},
            {"step": 2000, "codes": {"king": [0,1,0,...], "queen": [1,1,1,...]}},
            ...
        ]
    }

    Or a list directly:
    [
        {"step": 1000, "codes": {"king": [0,1,1,...]}},
        ...
    ]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        raw_snapshots = data
    elif isinstance(data, dict) and 'snapshots' in data:
        raw_snapshots = data['snapshots']
    else:
        raise ValueError(
            f"Expected a list of snapshots or a dict with 'snapshots' key. "
            f"Got: {type(data)} with keys {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
        )

    snapshots = []
    for s in raw_snapshots:
        snapshots.append(ConceptSnapshot.from_dict(s))

    snapshots.sort(key=lambda s: s.step)
    return snapshots


def main():
    parser = argparse.ArgumentParser(
        description='reptimeline -- track discrete representation evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Analyze pre-extracted snapshots
  reptimeline --snapshots timeline_data.json

  # Analyze with discovery and plots
  reptimeline --snapshots timeline_data.json --discover --plot

  # Full pipeline with primitive overlay
  reptimeline --snapshots timeline_data.json --discover --overlay primitivos.json --plot

  # Save analysis results
  reptimeline --snapshots timeline_data.json --output analysis.json
""",
    )

    parser.add_argument('--snapshots', required=True,
                        help='Path to JSON file with pre-extracted snapshots')
    parser.add_argument('--discover', action='store_true',
                        help='Run BitDiscovery on the last snapshot')
    parser.add_argument('--overlay', default=None, metavar='PRIMITIVOS_JSON',
                        help='Run PrimitiveOverlay with the given primitivos.json')
    parser.add_argument('--stability-window', type=int, default=3,
                        help='Consecutive stable snapshots to count as stabilized')
    parser.add_argument('--output', default=None,
                        help='Save analysis results (Timeline JSON) to this path')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--plot-dir', default=None,
                        help='Directory to save plots (default: ./reptimeline_plots/)')
    parser.add_argument('--causal', default=None, metavar='EFFECTS_JSON',
                        help='Run CausalVerifier with pre-computed intervention effects')

    args = parser.parse_args()

    # Load snapshots
    print(f"Loading snapshots from: {args.snapshots}")
    snapshots = _load_snapshots(args.snapshots)
    print(f"  Loaded {len(snapshots)} snapshots "
          f"(steps {snapshots[0].step:,} to {snapshots[-1].step:,})")

    concepts = snapshots[-1].concepts
    print(f"  Concepts: {len(concepts)}")

    # Analyze
    extractor = _JaccardExtractor()
    tracker = TimelineTracker(extractor, stability_window=args.stability_window)
    timeline = tracker.analyze(snapshots)
    timeline.print_summary()

    # Overlay
    report = None
    if args.overlay:
        from reptimeline.overlays import PrimitiveOverlay
        overlay = PrimitiveOverlay(args.overlay)
        report = overlay.analyze(timeline, concepts)
        overlay.print_report(report)

    # Discovery
    if args.discover:
        from reptimeline.discovery import BitDiscovery
        discovery = BitDiscovery()
        disc_report = discovery.discover(snapshots[-1], timeline=timeline)
        discovery.print_report(disc_report)

    # Causal verification
    causal_report = None
    if args.causal:
        from reptimeline.causal import CausalVerifier
        effects_data = _load_effects(args.causal)

        def intervene_fn(concept, bit_index):
            bit_key = f"bit_{bit_index}"
            return effects_data.get(bit_key, {}).get(concept, 0.0)

        verifier = CausalVerifier(intervene_fn=intervene_fn)
        causal_report = verifier.verify(snapshots[-1])
        verifier.print_report(causal_report)

    # Plots
    if args.plot:
        plot_dir = args.plot_dir or 'reptimeline_plots'
        os.makedirs(plot_dir, exist_ok=True)
        _generate_plots(timeline, report, causal_report, concepts, plot_dir)
        print(f"\nPlots saved to {plot_dir}/")

    # Save
    if args.output:
        _save_timeline(timeline, args.output)
        print(f"\nTimeline saved to {args.output}")


def _generate_plots(timeline, report, causal_report, concepts, plot_dir):
    """Generate all visualization plots."""
    from reptimeline.viz import (
        plot_swimlane, plot_phase_dashboard, plot_churn_heatmap, plot_layer_emergence,
        plot_causal_heatmap,
    )

    print("\nGenerating plots...")

    plot_swimlane(
        timeline, concepts=concepts[:20],  # limit for readability
        save_path=os.path.join(plot_dir, 'swimlane.png'),
        show=False,
    )
    print("  swimlane.png")

    plot_phase_dashboard(
        timeline,
        save_path=os.path.join(plot_dir, 'phase_dashboard.png'),
        show=False,
    )
    print("  phase_dashboard.png")

    plot_churn_heatmap(
        timeline, concepts=concepts,
        save_path=os.path.join(plot_dir, 'churn_heatmap.png'),
        show=False,
    )
    print("  churn_heatmap.png")

    if report is not None:
        plot_layer_emergence(
            report,
            save_path=os.path.join(plot_dir, 'layer_emergence.png'),
            show=False,
        )
        print("  layer_emergence.png")

    if causal_report is not None:
        plot_causal_heatmap(
            causal_report,
            save_path=os.path.join(plot_dir, 'causal_heatmap.png'),
            show=False,
        )
        print("  causal_heatmap.png")


def _load_effects(path):
    """Load pre-computed intervention effects from JSON.

    Expected format: {"effects": {"bit_0": {"king": 0.5, ...}, ...}}
    Or a flat dict: {"bit_0": {"king": 0.5, ...}, ...}
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'effects' in data:
        return data['effects']
    return data


def _save_timeline(timeline, path):
    """Serialize Timeline to JSON."""
    data = timeline.to_dict()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
