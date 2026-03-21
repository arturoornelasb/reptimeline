"""
Core data structures for representation timeline tracking.

These dataclasses are backend-agnostic — they work with any discrete
representation system (triadic bits, VQ-VAE, FSQ, sparse autoencoders).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ConceptSnapshot:
    """What a model "thinks" about a set of concepts at a given training step.

    This is the universal exchange format between extractors and the tracker.
    Every backend (triadic, VQ-VAE, FSQ) produces these.
    """
    step: int
    codes: Dict[str, List[int]]
    continuous: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def concepts(self) -> List[str]:
        return list(self.codes.keys())

    @property
    def code_dim(self) -> int:
        if self.codes:
            return len(next(iter(self.codes.values())))
        return 0

    def hamming(self, concept_a: str, concept_b: str) -> int:
        """Hamming distance between two concept codes."""
        a, b = self.codes.get(concept_a), self.codes.get(concept_b)
        if a is None or b is None:
            return -1
        return sum(x != y for x, y in zip(a, b))

    def active_indices(self, concept: str) -> List[int]:
        """Indices where code == 1 for a concept."""
        code = self.codes.get(concept)
        if code is None:
            return []
        return [i for i, v in enumerate(code) if v == 1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            'step': self.step,
            'codes': self.codes,
            'continuous': self.continuous,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ConceptSnapshot':
        """Deserialize from a dict."""
        return cls(
            step=d['step'],
            codes=d['codes'],
            continuous=d.get('continuous'),
            metadata=d.get('metadata', {}),
        )


@dataclass
class CodeEvent:
    """A discrete event in a code element's lifecycle."""
    event_type: str  # 'birth', 'death', 'flip', 'stabilize'
    step: int
    concept: str
    code_index: int
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectionEvent:
    """When two concepts first share (or lose) a discrete feature."""
    event_type: str  # 'form', 'break'
    step: int
    concept_a: str
    concept_b: str
    shared_indices: List[int] = field(default_factory=list)


@dataclass
class PhaseTransition:
    """A detected discontinuity in a training metric."""
    step: int
    metric: str
    delta: float
    direction: str  # 'increase' or 'decrease'


@dataclass
class Timeline:
    """Complete analysis result from TimelineTracker."""
    steps: List[int]
    snapshots: List[ConceptSnapshot]
    births: List[CodeEvent]
    deaths: List[CodeEvent]
    connections: List[ConnectionEvent]
    phase_transitions: List[PhaseTransition]
    curves: Dict[str, List[float]]  # metric_name -> values per step
    stability: Dict[int, float]  # code_index -> stability score [0,1]

    def print_summary(self):
        """Print a concise console summary."""
        print()
        print("=" * 60)
        print("  REPRESENTATION TIMELINE")
        print("=" * 60)
        print(f"  Steps:              {self.steps[0]:,} -> {self.steps[-1]:,} ({len(self.steps)} checkpoints)")
        print(f"  Concepts tracked:   {len(self.snapshots[-1].concepts) if self.snapshots else 0}")
        print(f"  Code dimension:     {self.snapshots[-1].code_dim if self.snapshots else 0}")
        print()
        print(f"  Bit births:         {len(self.births)}")
        print(f"  Bit deaths:         {len(self.deaths)}")
        print(f"  Connections formed: {len([c for c in self.connections if c.event_type == 'form'])}")
        print(f"  Phase transitions:  {len(self.phase_transitions)}")
        print()

        if 'churn_rate' in self.curves:
            churn = self.curves['churn_rate']
            print(f"  Code churn:         {churn[0]:.3f} -> {churn[-1]:.3f}")
        if 'utilization' in self.curves:
            util = self.curves['utilization']
            print(f"  Code utilization:   {util[0]:.3f} -> {util[-1]:.3f}")
        if 'entropy' in self.curves:
            ent = self.curves['entropy']
            print(f"  Mean entropy:       {ent[0]:.4f} -> {ent[-1]:.4f}")

        if self.phase_transitions:
            print()
            print("  Phase transitions:")
            for pt in self.phase_transitions:
                print(f"    step {pt.step:>6,}  {pt.metric:<15s}  "
                      f"{pt.direction} delta={pt.delta:.4f}")

        # Stability: most and least stable code elements
        if self.stability:
            sorted_stab = sorted(self.stability.items(), key=lambda x: x[1])
            print()
            print(f"  Least stable bits:  {[s[0] for s in sorted_stab[:5]]}")
            print(f"  Most stable bits:   {[s[0] for s in sorted_stab[-5:]]}")

        print("=" * 60)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Timeline to a JSON-compatible dict."""
        return {
            'steps': self.steps,
            'snapshots': [s.to_dict() for s in self.snapshots],
            'births': [
                {'step': e.step, 'concept': e.concept, 'code_index': e.code_index,
                 'event_type': e.event_type}
                for e in self.births
            ],
            'deaths': [
                {'step': e.step, 'concept': e.concept, 'code_index': e.code_index,
                 'event_type': e.event_type}
                for e in self.deaths
            ],
            'connections': [
                {'step': e.step, 'concept_a': e.concept_a,
                 'concept_b': e.concept_b, 'shared_indices': e.shared_indices,
                 'event_type': e.event_type}
                for e in self.connections
            ],
            'phase_transitions': [
                {'step': pt.step, 'metric': pt.metric,
                 'delta': pt.delta, 'direction': pt.direction}
                for pt in self.phase_transitions
            ],
            'curves': self.curves,
            'stability': {str(k): v for k, v in self.stability.items()},
        }
