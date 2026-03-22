"""
Core data structures for representation timeline tracking.

These dataclasses are backend-agnostic — they work with any discrete
representation system (triadic bits, VQ-VAE, FSQ, sparse autoencoders).
"""

import csv
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from reptimeline.exceptions import SnapshotError

SCHEMA_VERSION = "0.1"


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

    def validate(self) -> None:
        """Raise ValueError if codes have inconsistent lengths."""
        if not self.codes:
            return
        lengths = {concept: len(code) for concept, code in self.codes.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            examples = {sz: [c for c, cl in lengths.items() if cl == sz][:3]
                        for sz in unique_lengths}
            raise SnapshotError(
                f"Inconsistent code lengths in snapshot at step {self.step}: "
                f"{examples}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            'schema_version': SCHEMA_VERSION,
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
        n_ckpt = len(self.steps)
        print(f"  Steps:              {self.steps[0]:,} -> {self.steps[-1]:,}"
              f" ({n_ckpt} checkpoints)")
        print(f"  Concepts tracked:   {len(self.snapshots[-1].concepts) if self.snapshots else 0}")
        print(f"  Code dimension:     {self.snapshots[-1].code_dim if self.snapshots else 0}")
        print()
        print(f"  Bit births:         {len(self.births)}")
        print(f"  Bit deaths:         {len(self.deaths)}")
        n_conn = len([c for c in self.connections if c.event_type == 'form'])
        print(f"  Connections formed: {n_conn}")
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
            'schema_version': SCHEMA_VERSION,
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Timeline':
        """Deserialize a Timeline from a dict."""
        snapshots = [ConceptSnapshot.from_dict(s) for s in d['snapshots']]
        births = [
            CodeEvent(event_type=e['event_type'], step=e['step'],
                      concept=e['concept'], code_index=e['code_index'])
            for e in d.get('births', [])
        ]
        deaths = [
            CodeEvent(event_type=e['event_type'], step=e['step'],
                      concept=e['concept'], code_index=e['code_index'])
            for e in d.get('deaths', [])
        ]
        connections = [
            ConnectionEvent(event_type=e['event_type'], step=e['step'],
                            concept_a=e['concept_a'], concept_b=e['concept_b'],
                            shared_indices=e.get('shared_indices', []))
            for e in d.get('connections', [])
        ]
        phase_transitions = [
            PhaseTransition(step=pt['step'], metric=pt['metric'],
                            delta=pt['delta'], direction=pt['direction'])
            for pt in d.get('phase_transitions', [])
        ]
        stability = {int(k): v for k, v in d.get('stability', {}).items()}
        return cls(
            steps=d['steps'],
            snapshots=snapshots,
            births=births,
            deaths=deaths,
            connections=connections,
            phase_transitions=phase_transitions,
            curves=d.get('curves', {}),
            stability=stability,
        )

    def save_json(self, path: str) -> None:
        """Save Timeline to a JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> 'Timeline':
        """Load Timeline from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def to_csv(self, directory: str) -> Dict[str, str]:
        """Export Timeline data as CSV files in the given directory.

        Creates:
            - events.csv: births, deaths (step, event_type, concept, code_index)
            - connections.csv: connection events
            - curves.csv: metric curves (step, entropy, churn_rate, utilization)
            - stability.csv: per-bit stability scores
            - codes.csv: all concept codes at each step

        Args:
            directory: Output directory (created if needed).

        Returns:
            Dict mapping filename to full path.
        """
        import os
        os.makedirs(directory, exist_ok=True)
        written = {}

        # events.csv
        path = os.path.join(directory, 'events.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['step', 'event_type', 'concept', 'code_index'])
            for e in self.births + self.deaths:
                w.writerow([e.step, e.event_type, e.concept, e.code_index])
        written['events.csv'] = path

        # connections.csv
        path = os.path.join(directory, 'connections.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['step', 'event_type', 'concept_a', 'concept_b', 'shared_indices'])
            for c in self.connections:
                w.writerow([c.step, c.event_type, c.concept_a, c.concept_b,
                            ';'.join(str(i) for i in c.shared_indices)])
        written['connections.csv'] = path

        # curves.csv
        path = os.path.join(directory, 'curves.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            metric_names = sorted(self.curves.keys())
            w.writerow(['step'] + metric_names)
            for i, step in enumerate(self.steps):
                row = [step] + [self.curves[m][i] for m in metric_names]
                w.writerow(row)
        written['curves.csv'] = path

        # stability.csv
        if self.stability:
            path = os.path.join(directory, 'stability.csv')
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['bit_index', 'stability'])
                for idx in sorted(self.stability.keys()):
                    w.writerow([idx, f"{self.stability[idx]:.6f}"])
            written['stability.csv'] = path

        # codes.csv
        path = os.path.join(directory, 'codes.csv')
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            n_bits = self.snapshots[-1].code_dim if self.snapshots else 0
            w.writerow(['step', 'concept'] + [f'bit_{i}' for i in range(n_bits)])
            for snap in self.snapshots:
                for concept in sorted(snap.codes.keys()):
                    w.writerow([snap.step, concept] + snap.codes[concept])
        written['codes.csv'] = path

        return written
