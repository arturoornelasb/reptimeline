"""
TimelineTracker — Backend-agnostic analysis of representation evolution.

Consumes a sequence of ConceptSnapshot objects and computes lifecycle events:
births, deaths, connections, phase transitions, churn, stability.
"""

from typing import List, Optional, Tuple

import numpy as np

from reptimeline.core import (
    CodeEvent,
    ConceptSnapshot,
    ConnectionEvent,
    PhaseTransition,
    Timeline,
)
from reptimeline.extractors.base import RepresentationExtractor


class TimelineTracker:
    """Analyzes how discrete representations evolve across training snapshots."""

    def __init__(self, extractor: RepresentationExtractor,
                 stability_window: int = 3):
        """
        Args:
            extractor: Backend-specific extractor for similarity/shared features.
            stability_window: Number of consecutive snapshots a code element must
                remain unchanged to count as "stabilized".
        """
        self.extractor = extractor
        self.stability_window = stability_window

    def analyze(self, snapshots: List[ConceptSnapshot],
                concept_pairs: Optional[List[Tuple[str, str]]] = None,
                ) -> Timeline:
        """Run full timeline analysis on a sequence of snapshots.

        Args:
            snapshots: List of ConceptSnapshot sorted by step.
            concept_pairs: Optional pairs to track connections for.
                If None, tracks all pairs (can be slow for many concepts).

        Returns:
            Timeline with all lifecycle events and curves.
        """
        if not snapshots:
            raise ValueError("Need at least one snapshot")

        for snap in snapshots:
            snap.validate()

        steps = [s.step for s in snapshots]
        all_concepts = sorted(set().union(*(s.concepts for s in snapshots)))

        births = self._compute_births(snapshots, steps, all_concepts)
        deaths = self._compute_deaths(snapshots, steps, all_concepts)
        connections = self._compute_connections(snapshots, steps, concept_pairs)

        curves = {}
        curves['entropy'] = self._entropy_curve(snapshots)
        curves['churn_rate'] = self._churn_curve(snapshots)
        curves['utilization'] = self._utilization_curve(snapshots)

        stability = self._compute_stability(snapshots, all_concepts)

        phase_transitions = self._detect_phase_transitions(steps, curves)

        return Timeline(
            steps=steps,
            snapshots=snapshots,
            births=births,
            deaths=deaths,
            connections=connections,
            phase_transitions=phase_transitions,
            curves=curves,
            stability=stability,
        )

    # ------------------------------------------------------------------
    # Births: first step where a code element activates for a concept
    # ------------------------------------------------------------------

    def _compute_births(self, snapshots, steps, concepts):
        births = []
        seen = set()

        for t, snap in enumerate(snapshots):
            for concept in concepts:
                code = snap.codes.get(concept)
                if code is None:
                    continue
                for idx, val in enumerate(code):
                    if val == 1:
                        key = (concept, idx)
                        if key not in seen:
                            # Verify it wasn't active in earlier snapshots
                            was_active = False
                            for prev_t in range(t):
                                prev = snapshots[prev_t].codes.get(concept)
                                if prev and prev[idx] == 1:
                                    was_active = True
                                    break
                            if not was_active:
                                seen.add(key)
                                births.append(CodeEvent(
                                    event_type='birth',
                                    step=steps[t],
                                    concept=concept,
                                    code_index=idx,
                                ))
        return births

    # ------------------------------------------------------------------
    # Deaths: first step where a code element permanently deactivates
    # ------------------------------------------------------------------

    def _compute_deaths(self, snapshots, steps, concepts):
        deaths = []
        n_steps = len(snapshots)

        for concept in concepts:
            first_code = None
            for snap in snapshots:
                if concept in snap.codes:
                    first_code = snap.codes[concept]
                    break
            if first_code is None:
                continue

            n_bits = len(first_code)
            for idx in range(n_bits):
                for t in range(1, n_steps):
                    prev = snapshots[t - 1].codes.get(concept)
                    curr = snapshots[t].codes.get(concept)
                    if prev and curr and prev[idx] == 1 and curr[idx] == 0:
                        # Check permanence
                        stays_dead = all(
                            snapshots[ft].codes.get(concept, [0] * n_bits)[idx] == 0
                            for ft in range(t + 1, n_steps)
                        )
                        if stays_dead:
                            deaths.append(CodeEvent(
                                event_type='death',
                                step=steps[t],
                                concept=concept,
                                code_index=idx,
                            ))
                            break
        return deaths

    # ------------------------------------------------------------------
    # Connections: when two concepts first share a feature
    # ------------------------------------------------------------------

    def _compute_connections(self, snapshots, steps, pairs):
        connections = []
        if pairs is None:
            # All pairs from last snapshot
            concepts = snapshots[-1].concepts if snapshots else []
            pairs = [(a, b) for i, a in enumerate(concepts)
                     for b in concepts[i + 1:]]

        for a, b in pairs:
            for t, snap in enumerate(snapshots):
                code_a = snap.codes.get(a)
                code_b = snap.codes.get(b)
                if code_a is None or code_b is None:
                    continue

                shared = self.extractor.shared_features(code_a, code_b)
                if shared:
                    # Check if first time
                    was_connected = False
                    for prev_t in range(t):
                        prev_a = snapshots[prev_t].codes.get(a)
                        prev_b = snapshots[prev_t].codes.get(b)
                        if prev_a and prev_b:
                            if self.extractor.shared_features(prev_a, prev_b):
                                was_connected = True
                                break
                    if not was_connected:
                        connections.append(ConnectionEvent(
                            event_type='form',
                            step=steps[t],
                            concept_a=a,
                            concept_b=b,
                            shared_indices=shared,
                        ))
                    break
        return connections

    # ------------------------------------------------------------------
    # Curves
    # ------------------------------------------------------------------

    def _entropy_curve(self, snapshots):
        """Per-step mean entropy across all code elements."""
        curve = []
        for snap in snapshots:
            if not snap.codes:
                curve.append(0.0)
                continue
            n_bits = snap.code_dim
            codes = list(snap.codes.values())
            n = len(codes)
            entropies = []
            for bit_idx in range(n_bits):
                active = sum(1 for c in codes if bit_idx < len(c) and c[bit_idx] == 1)
                p = active / n if n > 0 else 0
                if 0 < p < 1:
                    entropies.append(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
                else:
                    entropies.append(0.0)
            curve.append(float(np.mean(entropies)))
        return curve

    def _churn_curve(self, snapshots):
        """Fraction of concepts whose code changed between consecutive steps."""
        curve = [0.0]  # No churn at first step
        for t in range(1, len(snapshots)):
            prev, curr = snapshots[t - 1], snapshots[t]
            common = set(prev.codes.keys()) & set(curr.codes.keys())
            if not common:
                curve.append(0.0)
                continue
            changed = sum(1 for c in common if prev.codes[c] != curr.codes[c])
            curve.append(changed / len(common))
        return curve

    def _utilization_curve(self, snapshots):
        """Fraction of unique codes out of total concepts at each step."""
        curve = []
        for snap in snapshots:
            if not snap.codes:
                curve.append(0.0)
                continue
            unique = len(set(tuple(v) for v in snap.codes.values()))
            curve.append(unique / len(snap.codes))
        return curve

    # ------------------------------------------------------------------
    # Stability: per code-element stability score
    # ------------------------------------------------------------------

    def _compute_stability(self, snapshots, concepts):
        """For each code index, compute stability = fraction of (concept, step)
        pairs where the code element didn't change from the previous step."""
        if len(snapshots) < 2:
            return {}

        n_bits = snapshots[-1].code_dim
        if n_bits == 0:
            return {}

        stable_counts = [0] * n_bits
        total_counts = [0] * n_bits

        for t in range(1, len(snapshots)):
            for concept in concepts:
                prev = snapshots[t - 1].codes.get(concept)
                curr = snapshots[t].codes.get(concept)
                if prev is None or curr is None:
                    continue
                for idx in range(min(n_bits, len(prev), len(curr))):
                    total_counts[idx] += 1
                    if prev[idx] == curr[idx]:
                        stable_counts[idx] += 1

        return {
            idx: stable_counts[idx] / max(total_counts[idx], 1)
            for idx in range(n_bits)
        }

    # ------------------------------------------------------------------
    # Phase transition detection
    # ------------------------------------------------------------------

    def _detect_phase_transitions(self, steps, curves, sigma_threshold=2.0):
        """Detect steps where any metric jumps by more than sigma_threshold * std."""
        transitions = []
        for metric_name, curve in curves.items():
            if len(curve) < 3:
                continue
            deltas = [curve[i] - curve[i - 1] for i in range(1, len(curve))]
            abs_deltas = [abs(d) for d in deltas]
            mean_d = np.mean(abs_deltas)
            std_d = np.std(abs_deltas)
            if std_d < 1e-8:
                continue
            threshold = mean_d + sigma_threshold * std_d
            for i, (d, ad) in enumerate(zip(deltas, abs_deltas)):
                if ad > threshold:
                    transitions.append(PhaseTransition(
                        step=steps[i + 1],
                        metric=metric_name,
                        delta=float(ad),
                        direction='increase' if d > 0 else 'decrease',
                    ))
        return transitions
