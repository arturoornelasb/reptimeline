"""
Reconciler — Compare discovered ontology vs manual ontology, suggest corrections.

Closes the loop:
  1. Train model (with or without supervision)
  2. BitDiscovery discovers what the model learned
  3. PrimitiveOverlay defines what the theory says
  4. Reconciler finds mismatches and suggests corrections

Corrections can go in BOTH directions:
  - Fix the model: generate corrected anchors for retraining
  - Fix the theory: suggest changes to primitivos.json
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from reptimeline.discovery import BitDiscovery, BitSemantics, DiscoveryReport
from reptimeline.overlays.primitive_overlay import PrimitiveOverlay, PrimitiveInfo


@dataclass
class BitMismatch:
    """A discovered incongruence between model and theory."""
    bit_index: int
    primitive_name: str
    mismatch_type: str  # 'semantic_drift', 'wrong_dual', 'missing_dep',
                        # 'extra_dep', 'dead_but_assigned', 'active_but_unassigned',
                        # 'swapped_bits'
    severity: str  # 'critical', 'warning', 'info'
    description: str
    suggestion: str
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DualMismatch:
    """A dual pair that doesn't match between discovery and theory."""
    mismatch_type: str  # 'missing_in_theory', 'missing_in_model', 'inverted'
    bit_a: int
    bit_b: int
    name_a: str
    name_b: str
    model_correlation: float
    description: str


@dataclass
class DepMismatch:
    """A dependency that doesn't match between discovery and theory."""
    mismatch_type: str  # 'missing_in_theory', 'missing_in_model', 'contradicted'
    parent_bit: int
    child_bit: int
    parent_name: str
    child_name: str
    confidence: float
    description: str


@dataclass
class ReconciliationReport:
    """Full comparison between discovered and theoretical ontology."""
    bit_mismatches: List[BitMismatch]
    dual_mismatches: List[DualMismatch]
    dep_mismatches: List[DepMismatch]
    agreement_score: float  # 0-1, how well model matches theory
    suggested_anchor_corrections: Dict[str, Any]
    suggested_theory_corrections: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class Reconciler:
    """Compares discovered structure with theoretical primitives.

    Finds mismatches and suggests corrections in both directions.
    """

    def __init__(self, overlay: PrimitiveOverlay,
                 semantic_drift_threshold: float = 0.3):
        """
        Args:
            overlay: PrimitiveOverlay with loaded primitivos.json.
            semantic_drift_threshold: How different a bit's actual usage
                can be from its theoretical meaning before flagging.
        """
        self.overlay = overlay
        self.drift_threshold = semantic_drift_threshold

    def reconcile(self, discovery_report: DiscoveryReport,
                  snapshot_codes: Dict[str, List[int]],
                  ) -> ReconciliationReport:
        """Compare discovered ontology with theoretical primitives.

        Args:
            discovery_report: Output from BitDiscovery.discover().
            snapshot_codes: The codes dict from the snapshot used for discovery.
        """
        bit_mismatches = self._check_bit_assignments(
            discovery_report, snapshot_codes
        )
        dual_mismatches = self._check_duals(discovery_report)
        dep_mismatches = self._check_dependencies(discovery_report)

        total_checks = (len(bit_mismatches) + len(dual_mismatches)
                        + len(dep_mismatches))
        critical = sum(1 for m in bit_mismatches if m.severity == 'critical')
        critical += sum(1 for m in dual_mismatches
                        if m.mismatch_type == 'missing_in_model')

        n_prims = len(self.overlay.primitives)
        agreement = 1.0 - (critical / max(n_prims, 1))

        anchor_corrections = self._suggest_anchor_corrections(
            bit_mismatches, dep_mismatches
        )
        theory_corrections = self._suggest_theory_corrections(
            bit_mismatches, dual_mismatches, dep_mismatches
        )

        return ReconciliationReport(
            bit_mismatches=bit_mismatches,
            dual_mismatches=dual_mismatches,
            dep_mismatches=dep_mismatches,
            agreement_score=max(0.0, agreement),
            suggested_anchor_corrections=anchor_corrections,
            suggested_theory_corrections=theory_corrections,
            metadata={
                'n_primitives': n_prims,
                'n_discovered_active': discovery_report.n_active_bits,
                'n_discovered_dead': discovery_report.n_dead_bits,
                'total_mismatches': total_checks,
                'critical_mismatches': critical,
            },
        )

    # ------------------------------------------------------------------
    # Bit assignment checks
    # ------------------------------------------------------------------

    def _check_bit_assignments(self, report: DiscoveryReport,
                               codes: Dict[str, List[int]],
                               ) -> List[BitMismatch]:
        """Check if each primitive's bit is used as the theory expects."""
        mismatches = []
        semantics = {bs.bit_index: bs for bs in report.bit_semantics}
        assigned_bits = {p.bit for p in self.overlay.primitives}

        for prim in self.overlay.primitives:
            bs = semantics.get(prim.bit)
            if bs is None:
                continue

            # Dead bit that should be active
            if bs.activation_rate < 0.02:
                mismatches.append(BitMismatch(
                    bit_index=prim.bit,
                    primitive_name=prim.name,
                    mismatch_type='dead_but_assigned',
                    severity='critical',
                    description=(f"Bit {prim.bit} ({prim.name}) is dead "
                                 f"(rate={bs.activation_rate:.3f}) but "
                                 f"theory assigns it as a primitive"),
                    suggestion=(f"Check if {prim.name} concepts are in "
                                f"training data. Consider reassigning to "
                                f"an active bit."),
                    evidence={'activation_rate': bs.activation_rate},
                ))

            # Very high activation = not discriminative
            elif bs.activation_rate > 0.95:
                mismatches.append(BitMismatch(
                    bit_index=prim.bit,
                    primitive_name=prim.name,
                    mismatch_type='semantic_drift',
                    severity='warning',
                    description=(f"Bit {prim.bit} ({prim.name}) activates "
                                 f"for {bs.activation_rate:.0%} of concepts — "
                                 f"not discriminative"),
                    suggestion=(f"This bit may have collapsed to 'always on'. "
                                f"Check discretization pressure."),
                    evidence={
                        'activation_rate': bs.activation_rate,
                        'top_concepts': bs.top_concepts[:5],
                    },
                ))

        # Active bits that have no primitive assigned
        for bs in report.bit_semantics:
            if bs.activation_rate > 0.05 and bs.bit_index not in assigned_bits:
                mismatches.append(BitMismatch(
                    bit_index=bs.bit_index,
                    primitive_name='(unassigned)',
                    mismatch_type='active_but_unassigned',
                    severity='info',
                    description=(f"Bit {bs.bit_index} is active "
                                 f"(rate={bs.activation_rate:.2f}) but has "
                                 f"no primitive assigned"),
                    suggestion=(f"Consider assigning a primitive to this bit. "
                                f"Top concepts: {bs.top_concepts[:5]}"),
                    evidence={
                        'activation_rate': bs.activation_rate,
                        'top_concepts': bs.top_concepts[:5],
                    },
                ))

        return mismatches

    # ------------------------------------------------------------------
    # Dual pair checks
    # ------------------------------------------------------------------

    def _check_duals(self, report: DiscoveryReport) -> List[DualMismatch]:
        """Compare discovered duals with theoretical duals."""
        mismatches = []

        # Theoretical duals
        theory_duals: Dict[Tuple[int, int], Tuple[str, str]] = {}
        for prim in self.overlay.primitives:
            if prim.dual:
                dual_info = self.overlay._name_to_info.get(prim.dual)
                if dual_info:
                    key = tuple(sorted([prim.bit, dual_info.bit]))
                    theory_duals[key] = (prim.name, prim.dual)

        # Discovered duals
        discovered_pairs: Dict[Tuple[int, int], float] = {}
        for dd in report.discovered_duals:
            key = tuple(sorted([dd.bit_a, dd.bit_b]))
            discovered_pairs[key] = dd.anti_correlation

        # Theory says dual but model doesn't show it
        for bits, names in theory_duals.items():
            if bits not in discovered_pairs:
                mismatches.append(DualMismatch(
                    mismatch_type='missing_in_model',
                    bit_a=bits[0], bit_b=bits[1],
                    name_a=names[0], name_b=names[1],
                    model_correlation=0.0,
                    description=(f"Theory says {names[0]}/{names[1]} "
                                 f"(bits {bits[0]},{bits[1]}) are duals, "
                                 f"but model shows no anti-correlation"),
                ))

        # Model shows dual but theory doesn't list it
        for bits, corr in discovered_pairs.items():
            if bits not in theory_duals and corr < -0.5:
                name_a = self.overlay._bit_to_name.get(bits[0], f"bit_{bits[0]}")
                name_b = self.overlay._bit_to_name.get(bits[1], f"bit_{bits[1]}")
                mismatches.append(DualMismatch(
                    mismatch_type='missing_in_theory',
                    bit_a=bits[0], bit_b=bits[1],
                    name_a=name_a, name_b=name_b,
                    model_correlation=corr,
                    description=(f"Model shows {name_a}/{name_b} "
                                 f"(bits {bits[0]},{bits[1]}) as duals "
                                 f"(corr={corr:.3f}), but theory doesn't "
                                 f"list them"),
                ))

        return mismatches

    # ------------------------------------------------------------------
    # Dependency checks
    # ------------------------------------------------------------------

    def _check_dependencies(self, report: DiscoveryReport) -> List[DepMismatch]:
        """Compare discovered deps with theoretical deps."""
        mismatches = []

        # Build discovered dep set
        discovered_deps: Dict[Tuple[int, int], float] = {}
        for dd in report.discovered_deps:
            discovered_deps[(dd.bit_parent, dd.bit_child)] = dd.confidence

        # Check each theoretical dependency
        for prim in self.overlay.primitives:
            child_bit = prim.bit
            for dep_name in prim.deps:
                dep_info = self.overlay._name_to_info.get(dep_name)
                if dep_info is None:
                    continue
                parent_bit = dep_info.bit
                key = (parent_bit, child_bit)

                if key not in discovered_deps:
                    mismatches.append(DepMismatch(
                        mismatch_type='missing_in_model',
                        parent_bit=parent_bit, child_bit=child_bit,
                        parent_name=dep_name, child_name=prim.name,
                        confidence=0.0,
                        description=(f"Theory says {prim.name} depends on "
                                     f"{dep_name}, but model doesn't show "
                                     f"this dependency"),
                    ))

        # Check for discovered deps not in theory
        assigned_bits = {p.bit: p.name for p in self.overlay.primitives}
        theory_deps: Set[Tuple[int, int]] = set()
        for prim in self.overlay.primitives:
            for dep_name in prim.deps:
                dep_info = self.overlay._name_to_info.get(dep_name)
                if dep_info:
                    theory_deps.add((dep_info.bit, prim.bit))

        for (parent, child), conf in discovered_deps.items():
            if (parent, child) not in theory_deps:
                parent_name = assigned_bits.get(parent, f"bit_{parent}")
                child_name = assigned_bits.get(child, f"bit_{child}")
                if conf > 0.95:  # only flag strong unexpected deps
                    mismatches.append(DepMismatch(
                        mismatch_type='missing_in_theory',
                        parent_bit=parent, child_bit=child,
                        parent_name=parent_name, child_name=child_name,
                        confidence=conf,
                        description=(f"Model shows {child_name} depends on "
                                     f"{parent_name} (conf={conf:.2f}), "
                                     f"but theory doesn't list this"),
                    ))

        return mismatches

    # ------------------------------------------------------------------
    # Correction suggestions
    # ------------------------------------------------------------------

    def _suggest_anchor_corrections(self, bit_mismatches, dep_mismatches):
        """Suggest corrections to anchor files for retraining."""
        corrections = {
            'add_anchors_for': [],
            'remove_anchors_for': [],
            'modify_bit_targets': [],
        }

        for m in bit_mismatches:
            if m.mismatch_type == 'dead_but_assigned':
                corrections['add_anchors_for'].append({
                    'primitive': m.primitive_name,
                    'bit': m.bit_index,
                    'reason': 'Bit is dead — add more training examples '
                              'that should activate this bit',
                })
            elif m.mismatch_type == 'semantic_drift':
                corrections['modify_bit_targets'].append({
                    'primitive': m.primitive_name,
                    'bit': m.bit_index,
                    'reason': 'Bit activates for too many concepts — '
                              'add negative examples to make it selective',
                })

        return corrections

    def _suggest_theory_corrections(self, bit_mismatches, dual_mismatches,
                                    dep_mismatches):
        """Suggest corrections to primitivos.json based on what model learned."""
        corrections = {
            'add_duals': [],
            'remove_duals': [],
            'add_dependencies': [],
            'remove_dependencies': [],
            'review_primitives': [],
        }

        for m in dual_mismatches:
            if m.mismatch_type == 'missing_in_theory':
                corrections['add_duals'].append({
                    'pair': [m.name_a, m.name_b],
                    'correlation': m.model_correlation,
                    'reason': m.description,
                })
            elif m.mismatch_type == 'missing_in_model':
                corrections['review_primitives'].append({
                    'pair': [m.name_a, m.name_b],
                    'reason': f"Listed as duals but model doesn't "
                              f"anti-correlate them",
                })

        for m in dep_mismatches:
            if m.mismatch_type == 'missing_in_theory':
                corrections['add_dependencies'].append({
                    'child': m.child_name,
                    'parent': m.parent_name,
                    'confidence': m.confidence,
                    'reason': m.description,
                })
            elif m.mismatch_type == 'missing_in_model':
                corrections['remove_dependencies'].append({
                    'child': m.child_name,
                    'parent': m.parent_name,
                    'reason': m.description,
                })

        return corrections

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_report(self, report: ReconciliationReport):
        """Print reconciliation results."""
        print()
        print("=" * 60)
        print("  RECONCILIATION REPORT")
        print("=" * 60)
        meta = report.metadata
        print(f"  Primitives:          {meta.get('n_primitives', 0)}")
        print(f"  Active bits (model): {meta.get('n_discovered_active', 0)}")
        print(f"  Dead bits (model):   {meta.get('n_discovered_dead', 0)}")
        print(f"  Agreement score:     {report.agreement_score:.1%}")
        print(f"  Total mismatches:    {meta.get('total_mismatches', 0)}")
        print(f"  Critical:            {meta.get('critical_mismatches', 0)}")
        print()

        # Bit mismatches
        if report.bit_mismatches:
            print("  BIT MISMATCHES")
            print("  " + "-" * 56)
            for m in sorted(report.bit_mismatches,
                            key=lambda x: {'critical': 0, 'warning': 1,
                                           'info': 2}[x.severity]):
                icon = {'critical': '!!', 'warning': '! ', 'info': '  '}
                print(f"    {icon[m.severity]} [{m.severity.upper()}] {m.description}")
                print(f"       -> {m.suggestion}")
            print()

        # Dual mismatches
        if report.dual_mismatches:
            print("  DUAL MISMATCHES")
            print("  " + "-" * 56)
            for m in report.dual_mismatches:
                print(f"    {m.description}")
            print()

        # Dep mismatches
        if report.dep_mismatches:
            n_model = sum(1 for m in report.dep_mismatches
                          if m.mismatch_type == 'missing_in_model')
            n_theory = sum(1 for m in report.dep_mismatches
                           if m.mismatch_type == 'missing_in_theory')
            print(f"  DEPENDENCY MISMATCHES "
                  f"({n_model} theory-only, {n_theory} model-only)")
            print("  " + "-" * 56)
            for m in report.dep_mismatches[:20]:
                print(f"    {m.description}")
            if len(report.dep_mismatches) > 20:
                print(f"    ... and {len(report.dep_mismatches) - 20} more")
            print()

        # Suggestions
        ac = report.suggested_anchor_corrections
        tc = report.suggested_theory_corrections

        has_suggestions = (ac.get('add_anchors_for')
                           or ac.get('modify_bit_targets')
                           or tc.get('add_duals')
                           or tc.get('add_dependencies'))

        if has_suggestions:
            print("  SUGGESTED CORRECTIONS")
            print("  " + "-" * 56)
            if ac.get('add_anchors_for'):
                print("  For retraining (fix anchors):")
                for s in ac['add_anchors_for']:
                    print(f"    + Add anchors for '{s['primitive']}' "
                          f"(bit {s['bit']}): {s['reason']}")
            if ac.get('modify_bit_targets'):
                for s in ac['modify_bit_targets']:
                    print(f"    ~ Modify targets for '{s['primitive']}' "
                          f"(bit {s['bit']}): {s['reason']}")
            if tc.get('add_duals'):
                print("  For theory (fix primitivos.json):")
                for s in tc['add_duals']:
                    print(f"    + Add dual pair: {s['pair']} "
                          f"(corr={s['correlation']:.3f})")
            if tc.get('add_dependencies'):
                for s in tc['add_dependencies'][:10]:
                    print(f"    + Add dep: {s['child']} -> {s['parent']} "
                          f"(conf={s['confidence']:.2f})")
            print()

        print("=" * 60)
