#!/usr/bin/env python3
"""
Null baseline demo — compare discovered structure against random expectations.

Shows that the structure found by BitDiscovery (duals, dependencies, triadic
interactions) is statistically significant by comparing against random binary
codes of the same dimensions.

Usage:
    python examples/null_baseline_demo.py [--snapshots path/to/snapshots.json]
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery


def load_snapshots(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict) and 'snapshots' in data:
        raw = data['snapshots']
    else:
        raise ValueError(f"Invalid format: {type(data)}")
    snaps = [ConceptSnapshot.from_dict(s) for s in raw]
    snaps.sort(key=lambda s: s.step)
    return snaps


def main():
    parser = argparse.ArgumentParser(description="Null baseline comparison")
    parser.add_argument('--snapshots', default=None,
                        help='Path to snapshots JSON (default: results/mnist_bae/snapshots.json)')
    args = parser.parse_args()

    snap_path = args.snapshots or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'mnist_bae', 'snapshots.json'
    )

    if not os.path.exists(snap_path):
        print(f"Snapshots not found: {snap_path}")
        sys.exit(1)

    print(f"Loading: {snap_path}")
    snapshots = load_snapshots(snap_path)
    last = snapshots[-1]
    n_concepts = len(last.concepts)
    n_bits = last.code_dim

    print(f"  {n_concepts} concepts, {n_bits} bits, {len(snapshots)} snapshots")

    # Discover real structure
    discovery = BitDiscovery()
    report = discovery.discover(last)

    real_duals = len(report.discovered_duals)
    real_deps = len(report.discovered_deps)
    real_triadic = len(report.discovered_triadic_deps)

    # Run null baseline
    print(f"\nRunning null baseline (10 trials, {n_concepts}x{n_bits} random binary)...")
    baseline = discovery.null_baseline(n_concepts, n_bits, n_trials=10)

    # Print comparison
    print("\n" + "=" * 65)
    print("  REAL DATA vs NULL BASELINE")
    print("=" * 65)
    print(f"  {'Metric':<25s}  {'Real':>8s}  {'Random':>8s}  {'Ratio':>8s}")
    print("  " + "-" * 61)

    for label, real, rand_key in [
        ("Dual pairs", real_duals, 'mean_random_duals'),
        ("Dependencies", real_deps, 'mean_random_deps'),
        ("Triadic interactions", real_triadic, 'mean_random_triadic'),
    ]:
        rand = baseline[rand_key]
        ratio = real / rand if rand > 0 else float('inf')
        print(f"  {label:<25s}  {real:>8d}  {rand:>8.1f}  {ratio:>7.1f}x")

    print("  " + "-" * 61)
    print(f"  Active bits: {report.n_active_bits}/{n_bits}  "
          f"Dead bits: {report.n_dead_bits}/{n_bits}")
    print("=" * 65)

    # Interpretation
    if real_duals > baseline['mean_random_duals'] * 2:
        print("\n  Dual pairs are significantly above random expectation.")
    else:
        print("\n  Warning: dual pair count is close to random baseline.")

    if real_triadic > baseline['mean_random_triadic'] * 2:
        print("  Triadic interactions are significantly above random expectation.")
    else:
        print("  Warning: triadic interaction count is close to random baseline.")


if __name__ == "__main__":
    main()
