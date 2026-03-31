#!/usr/bin/env python3
"""
Run null models for connections + BitDiscovery on all available backends.

Sprint 1.6: Permutation null for connections (tracker.py)
Sprint 1.7: MCC alongside Pearson (discovery.py)
Sprint 1.8: Report sample sizes (discovery.py)

Usage:
    python examples/run_null_models.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery
from reptimeline.tracker import TimelineTracker
from reptimeline.extractors.base import RepresentationExtractor


class MinimalExtractor(RepresentationExtractor):
    """Lightweight extractor for null model — only implements shared_features."""
    def extract(self, checkpoint_path, concepts, device='cpu'):
        raise NotImplementedError("Not needed for null model")
    def similarity(self, code_a, code_b):
        shared = sum(1 for a, b in zip(code_a, code_b) if a == 1 and b == 1)
        total = sum(1 for a, b in zip(code_a, code_b) if a == 1 or b == 1)
        return shared / total if total > 0 else 0.0
    def shared_features(self, code_a, code_b):
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]
    def discover_checkpoints(self, directory):
        return []


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


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

BACKENDS = {
    'mnist_bae': os.path.join(RESULTS_DIR, 'mnist_bae', 'snapshots.json'),
    'pythia_sae': os.path.join(RESULTS_DIR, 'pythia_sae', 'snapshots.json'),
}


def run_backend(name, snap_path):
    print(f"\n{'='*68}")
    print(f"  BACKEND: {name}")
    print(f"{'='*68}")

    if not os.path.exists(snap_path):
        print(f"  SKIP — snapshots not found: {snap_path}")
        return None

    snapshots = load_snapshots(snap_path)
    last = snapshots[-1]
    n_concepts = len(last.concepts)
    n_bits = last.code_dim
    print(f"  {n_concepts} concepts, {n_bits} bits, {len(snapshots)} snapshots")

    results = {'backend': name, 'n_concepts': n_concepts, 'n_bits': n_bits,
               'n_snapshots': len(snapshots)}

    # --- 1. BitDiscovery with MCC (Sprint 1.7) ---
    print(f"\n[1/3] BitDiscovery (with MCC + 1000 permutations)...")
    discovery = BitDiscovery()
    report = discovery.discover(last)

    real_duals = len(report.discovered_duals)
    real_deps = len(report.discovered_deps)
    real_triadic = len(report.discovered_triadic_deps)

    print(f"  Duals: {real_duals}")
    print(f"  Dependencies: {real_deps}")
    print(f"  Triadic: {real_triadic}")

    # Report MCC for duals
    if report.discovered_duals:
        mcc_values = [d.mcc for d in report.discovered_duals if hasattr(d, 'mcc') and d.mcc != 0.0]
        if mcc_values:
            import numpy as np
            print(f"  Dual MCC: mean={np.mean(mcc_values):.3f}, "
                  f"min={np.min(mcc_values):.3f}, max={np.max(mcc_values):.3f}")
            results['dual_mcc_mean'] = float(np.mean(mcc_values))
        else:
            print(f"  Dual MCC: not computed (all 0.0)")

    results['discovery'] = {
        'duals': real_duals,
        'deps': real_deps,
        'triadic': real_triadic,
    }

    # --- 2. Null baseline for BitDiscovery ---
    print(f"\n[2/3] Null baseline for BitDiscovery (10 trials)...")
    baseline = discovery.null_baseline(n_concepts, n_bits, n_trials=10)

    for label, real, rand_key in [
        ("Dual pairs", real_duals, 'mean_random_duals'),
        ("Dependencies", real_deps, 'mean_random_deps'),
        ("Triadic", real_triadic, 'mean_random_triadic'),
    ]:
        rand = baseline[rand_key]
        ratio = real / rand if rand > 0 else float('inf')
        print(f"  {label:<20s}  real={real:>4d}  random={rand:>6.1f}  O/E={ratio:>6.1f}x")

    results['null_discovery'] = baseline

    # --- 3. Connections null model (Sprint 1.6) ---
    print(f"\n[3/3] Connections null model (1000 permutations)...")
    tracker = TimelineTracker(extractor=MinimalExtractor())
    conn_null = tracker.connections_null_model(snapshots, n_permutations=1000)

    print(f"  Observed connections: {conn_null['n_observed']}")
    print(f"  Expected (null):     {conn_null['n_expected']:.1f} +/- {conn_null['null_std']:.1f}")
    print(f"  O/E ratio:           {conn_null['oe_ratio']:.2f}")
    print(f"  p-value:             {conn_null['null_p_value']:.4f}")
    print(f"  Kill K3 (O/E<1.5):   {'FAIL' if conn_null['kill_k3'] else 'PASS'}")

    results['connections_null'] = conn_null

    return results


def main():
    all_results = {}
    for name, path in BACKENDS.items():
        result = run_backend(name, path)
        if result:
            all_results[name] = result

    # Save
    out_path = os.path.join(RESULTS_DIR, 'null_model_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    # Summary
    print(f"\n{'='*68}")
    print(f"  SUMMARY — NULL MODEL RESULTS")
    print(f"{'='*68}")
    for name, r in all_results.items():
        cn = r['connections_null']
        print(f"  {name}:")
        print(f"    Connections O/E = {cn['oe_ratio']:.2f} (p={cn['null_p_value']:.4f}) "
              f"{'PASS' if not cn['kill_k3'] else 'FAIL K3'}")
        nd = r['null_discovery']
        for metric in ['duals', 'deps', 'triadic']:
            real = r['discovery'][metric]
            rand = nd.get(f'mean_random_{metric}', 0)
            ratio = real / rand if rand > 0 else float('inf')
            print(f"    {metric}: real={real}, random={rand:.1f}, O/E={ratio:.1f}x")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
