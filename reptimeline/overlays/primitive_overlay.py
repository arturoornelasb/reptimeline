"""
PrimitiveOverlay — Maps generic timeline events onto a set of named primitives.

This overlay adds domain-specific semantics on top of the backend-agnostic
Timeline produced by TimelineTracker:

  - Primitive activation epochs: when each primitive first activates
  - Dependency chain completion: when all deps of a primitive are satisfied
  - Layer emergence order: do lower-layer primitives stabilize before higher ones?
  - Dual axis coherence: when dual pairs become anti-correlated
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from reptimeline.core import Timeline
from reptimeline.exceptions import ConfigurationError


@dataclass
class PrimitiveInfo:
    """Parsed info for one primitive."""
    bit: int
    prime: int
    name: str
    layer: int
    deps: List[str]
    dual: Optional[str] = None
    definition: str = ""


@dataclass
class ActivationEpoch:
    """When a primitive first becomes active for a given concept."""
    primitive: str
    concept: str
    step: int
    bit_index: int


@dataclass
class DepsCompletion:
    """When all dependencies of a primitive are simultaneously active."""
    primitive: str
    concept: str
    step: int
    deps: List[str]
    deps_met_steps: Dict[str, int]  # dep_name -> step it was first active


@dataclass
class LayerEmergence:
    """Aggregate statistics for when a layer's primitives emerge."""
    layer: int
    layer_name: str
    n_primitives: int
    first_activation_step: Optional[int]
    median_activation_step: Optional[float]
    last_activation_step: Optional[int]
    primitives_activated: int


@dataclass
class DualCoherence:
    """Tracks whether dual pairs show expected anti-correlation."""
    primitive_a: str
    primitive_b: str
    steps_both_active: List[int]
    steps_exclusive: List[int]  # one active, other not
    coherence_score: float  # fraction of steps that are exclusive (anti-correlated)


@dataclass
class PrimitiveReport:
    """Full overlay analysis output."""
    activations: List[ActivationEpoch]
    deps_completions: List[DepsCompletion]
    layer_emergence: List[LayerEmergence]
    dual_coherence: List[DualCoherence]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrimitiveOverlay:
    """Interprets a Timeline through the lens of named primitives.

    This is not an extractor or tracker -- it takes an already-computed Timeline
    and overlays domain-specific analysis.
    """

    def __init__(self, primitivos_path: Optional[str] = None):
        """
        Args:
            primitivos_path: Path to a primitivos.json file defining the
                primitives. Required -- there is no auto-detection in the
                standalone package.
        """
        if primitivos_path is None:
            raise ConfigurationError(
                "primitivos_path is required. Pass the path to your "
                "primitivos.json file (e.g., PrimitiveOverlay('primitivos.json'))."
            )
        self.primitivos_path = primitivos_path
        self.primitives = self._load_primitives()
        self._name_to_bit = {p.name: p.bit for p in self.primitives}
        self._bit_to_name = {p.bit: p.name for p in self.primitives}
        self._name_to_info = {p.name: p for p in self.primitives}

    def _load_primitives(self) -> List[PrimitiveInfo]:
        with open(self.primitivos_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        prims = []
        for entry in data['primitivos']:
            prims.append(PrimitiveInfo(
                bit=entry['bit'],
                prime=entry.get('primo', 0),
                name=entry['nombre'],
                layer=entry.get('capa', 0),
                deps=entry.get('deps', []),
                dual=entry.get('dual'),
                definition=entry.get('def', ''),
            ))
        return prims

    def analyze(self, timeline: Timeline,
                concepts: Optional[List[str]] = None) -> PrimitiveReport:
        """Run full primitive overlay analysis on a Timeline.

        Args:
            timeline: A Timeline from TimelineTracker.analyze().
            concepts: Subset of concepts to analyze. If None, uses all
                concepts from the last snapshot.
        """
        if concepts is None:
            concepts = timeline.snapshots[-1].concepts if timeline.snapshots else []

        activations = self._compute_activations(timeline, concepts)
        deps_completions = self._compute_deps_completions(timeline, concepts, activations)
        layer_emergence = self._compute_layer_emergence(activations)
        dual_coherence = self._compute_dual_coherence(timeline, concepts)

        return PrimitiveReport(
            activations=activations,
            deps_completions=deps_completions,
            layer_emergence=layer_emergence,
            dual_coherence=dual_coherence,
            metadata={
                'n_primitives': len(self.primitives),
                'n_concepts': len(concepts),
                'n_steps': len(timeline.steps),
            },
        )

    # ------------------------------------------------------------------
    # Activation epochs
    # ------------------------------------------------------------------

    def _compute_activations(self, timeline: Timeline,
                             concepts: List[str]) -> List[ActivationEpoch]:
        """Find the first step where each primitive bit activates per concept."""
        activations = []
        for concept in concepts:
            for prim in self.primitives:
                bit_idx = prim.bit
                for snap in timeline.snapshots:
                    code = snap.codes.get(concept)
                    if code is None or bit_idx >= len(code):
                        continue
                    if code[bit_idx] == 1:
                        activations.append(ActivationEpoch(
                            primitive=prim.name,
                            concept=concept,
                            step=snap.step,
                            bit_index=bit_idx,
                        ))
                        break
        return activations

    # ------------------------------------------------------------------
    # Dependency chain completion
    # ------------------------------------------------------------------

    def _compute_deps_completions(self, timeline: Timeline,
                                  concepts: List[str],
                                  activations: List[ActivationEpoch],
                                  ) -> List[DepsCompletion]:
        """Find when all deps of a primitive are simultaneously active."""
        # Build activation lookup: (concept, primitive_name) -> step
        act_lookup: Dict[Tuple[str, str], int] = {}
        for act in activations:
            act_lookup[(act.concept, act.primitive)] = act.step

        completions = []
        for concept in concepts:
            for prim in self.primitives:
                if not prim.deps:
                    continue

                # Check if all deps have activated
                deps_steps: Dict[str, int] = {}
                all_met = True
                for dep_name in prim.deps:
                    key = (concept, dep_name)
                    if key in act_lookup:
                        deps_steps[dep_name] = act_lookup[key]
                    else:
                        all_met = False
                        break

                if not all_met:
                    continue

                # Find first step where all deps are simultaneously active
                completion_step = self._find_simultaneous_activation(
                    timeline, concept, prim.deps
                )
                if completion_step is not None:
                    completions.append(DepsCompletion(
                        primitive=prim.name,
                        concept=concept,
                        step=completion_step,
                        deps=prim.deps,
                        deps_met_steps=deps_steps,
                    ))

        return completions

    def _find_simultaneous_activation(self, timeline: Timeline,
                                      concept: str,
                                      dep_names: List[str]) -> Optional[int]:
        """Return first step where all dep bits are == 1 for a concept."""
        dep_bits = [self._name_to_bit[d] for d in dep_names if d in self._name_to_bit]
        if len(dep_bits) != len(dep_names):
            return None

        for snap in timeline.snapshots:
            code = snap.codes.get(concept)
            if code is None:
                continue
            if all(b < len(code) and code[b] == 1 for b in dep_bits):
                return snap.step
        return None

    # ------------------------------------------------------------------
    # Layer emergence
    # ------------------------------------------------------------------

    def _compute_layer_emergence(self, activations: List[ActivationEpoch],
                                 ) -> List[LayerEmergence]:
        """Aggregate activation epochs by layer."""
        # Group primitives by layer
        prims_per_layer: Dict[int, List[str]] = {}
        for p in self.primitives:
            prims_per_layer.setdefault(p.layer, []).append(p.name)

        # Earliest activation per primitive (across all concepts)
        prim_first_step: Dict[str, int] = {}
        for act in activations:
            if act.primitive not in prim_first_step:
                prim_first_step[act.primitive] = act.step
            else:
                prim_first_step[act.primitive] = min(
                    prim_first_step[act.primitive], act.step
                )

        results = []
        for layer in sorted(prims_per_layer.keys()):
            prims = prims_per_layer[layer]
            steps = [prim_first_step[p] for p in prims if p in prim_first_step]

            results.append(LayerEmergence(
                layer=layer,
                layer_name=f"Layer {layer}",
                n_primitives=len(prims),
                first_activation_step=min(steps) if steps else None,
                median_activation_step=sorted(steps)[len(steps) // 2] if steps else None,
                last_activation_step=max(steps) if steps else None,
                primitives_activated=len(steps),
            ))
        return results

    # ------------------------------------------------------------------
    # Dual coherence
    # ------------------------------------------------------------------

    def _compute_dual_coherence(self, timeline: Timeline,
                                concepts: List[str]) -> List[DualCoherence]:
        """Check whether dual pairs show anti-correlation across training."""
        dual_pairs = [(p.name, p.dual) for p in self.primitives if p.dual]
        # Deduplicate (a,b) and (b,a)
        seen = set()
        unique_pairs = []
        for a, b in dual_pairs:
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                unique_pairs.append((a, b))

        results = []
        for prim_a, prim_b in unique_pairs:
            bit_a = self._name_to_bit.get(prim_a)
            bit_b = self._name_to_bit.get(prim_b)
            if bit_a is None or bit_b is None:
                continue

            both_active = []
            exclusive = []

            for snap in timeline.snapshots:
                for concept in concepts:
                    code = snap.codes.get(concept)
                    if code is None:
                        continue
                    if bit_a >= len(code) or bit_b >= len(code):
                        continue

                    a_on = code[bit_a] == 1
                    b_on = code[bit_b] == 1

                    if a_on and b_on:
                        both_active.append(snap.step)
                    elif a_on or b_on:
                        exclusive.append(snap.step)

            total = len(both_active) + len(exclusive)
            coherence = len(exclusive) / total if total > 0 else 0.0

            results.append(DualCoherence(
                primitive_a=prim_a,
                primitive_b=prim_b,
                steps_both_active=both_active,
                steps_exclusive=exclusive,
                coherence_score=coherence,
            ))
        return results

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_report(self, report: PrimitiveReport):
        """Print a structured console summary."""
        print()
        print("=" * 60)
        print("  PRIMITIVE OVERLAY REPORT")
        print("=" * 60)
        meta = report.metadata
        print(f"  Primitives: {meta.get('n_primitives', 0)}")
        print(f"  Concepts:   {meta.get('n_concepts', 0)}")
        print(f"  Steps:      {meta.get('n_steps', 0)}")
        print()

        # Layer emergence
        print("  LAYER EMERGENCE ORDER")
        print("  " + "-" * 56)
        for le in report.layer_emergence:
            activated = f"{le.primitives_activated}/{le.n_primitives}"
            if le.first_activation_step is not None:
                print(f"    L{le.layer} {le.layer_name:<18s}  "
                      f"first={le.first_activation_step:>6,}  "
                      f"median={le.median_activation_step:>6,}  "
                      f"activated={activated}")
            else:
                print(f"    L{le.layer} {le.layer_name:<18s}  "
                      f"NOT YET ACTIVATED  ({activated})")
        print()

        # Dual coherence
        if report.dual_coherence:
            print("  DUAL AXIS COHERENCE")
            print("  " + "-" * 56)
            sorted_dc = sorted(report.dual_coherence,
                               key=lambda d: d.coherence_score, reverse=True)
            for dc in sorted_dc:
                n_both = len(dc.steps_both_active)
                n_excl = len(dc.steps_exclusive)
                print(f"    {dc.primitive_a:<16s} <-> {dc.primitive_b:<16s}  "
                      f"coherence={dc.coherence_score:.2f}  "
                      f"(excl={n_excl}, both={n_both})")
            print()

        # Activation summary by primitive
        if report.activations:
            print(f"  ACTIVATIONS: {len(report.activations)} "
                  f"(primitive x concept events)")
            # Show earliest activation per primitive
            earliest: Dict[str, int] = {}
            for act in report.activations:
                if act.primitive not in earliest:
                    earliest[act.primitive] = act.step
                else:
                    earliest[act.primitive] = min(earliest[act.primitive], act.step)
            sorted_prims = sorted(earliest.items(), key=lambda x: x[1])
            print("  First 10 primitives to activate:")
            for name, step in sorted_prims[:10]:
                info = self._name_to_info.get(name)
                layer = f"L{info.layer}" if info else "?"
                print(f"    {layer}  {name:<20s}  step {step:>6,}")
            print()

        # Deps completion
        if report.deps_completions:
            print(f"  DEPENDENCY COMPLETIONS: {len(report.deps_completions)}")
            print()

        print("=" * 60)
