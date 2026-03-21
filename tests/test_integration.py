"""Integration tests using real pre-computed data from results/mnist_bae/."""

import os
import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.tracker import TimelineTracker
from reptimeline.discovery import BitDiscovery

SNAPSHOTS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'mnist_bae', 'snapshots.json'
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(SNAPSHOTS_PATH),
    reason="results/mnist_bae/snapshots.json not present"
)


def _load_snapshots():
    """Load real snapshots using the CLI loader."""
    from reptimeline.cli import _load_snapshots as load
    return load(SNAPSHOTS_PATH)


class _JaccardExtractor:
    """Minimal extractor for pre-extracted binary codes."""
    def extract(self, checkpoint_path, concepts, device='cpu'):
        raise NotImplementedError
    def similarity(self, code_a, code_b):
        a = set(i for i, v in enumerate(code_a) if v == 1)
        b = set(i for i, v in enumerate(code_b) if v == 1)
        union = a | b
        return len(a & b) / len(union) if union else 1.0
    def shared_features(self, code_a, code_b):
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]


class TestMNISTIntegration:

    def test_snapshots_load(self):
        snaps = _load_snapshots()
        assert len(snaps) >= 2
        assert all(isinstance(s, ConceptSnapshot) for s in snaps)

    def test_steps_monotonic(self):
        snaps = _load_snapshots()
        steps = [s.step for s in snaps]
        assert steps == sorted(steps)

    def test_timeline_analysis(self):
        snaps = _load_snapshots()
        tracker = TimelineTracker(_JaccardExtractor())
        timeline = tracker.analyze(snaps)

        assert len(timeline.steps) == len(snaps)
        assert len(timeline.births) > 0
        assert 'entropy' in timeline.curves
        assert 'churn_rate' in timeline.curves
        assert len(timeline.curves['entropy']) == len(snaps)

    def test_code_dimension(self):
        snaps = _load_snapshots()
        assert snaps[-1].code_dim == 32  # MNIST BAE is 32-bit

    def test_concepts_are_digits(self):
        snaps = _load_snapshots()
        concepts = snaps[-1].concepts
        assert len(concepts) == 10

    def test_discovery_on_last_snapshot(self):
        snaps = _load_snapshots()
        discovery = BitDiscovery()
        report = discovery.discover(snaps[-1])
        assert report.n_active_bits > 0
        assert report.n_active_bits + report.n_dead_bits == 32

    def test_stability_computed(self):
        snaps = _load_snapshots()
        tracker = TimelineTracker(_JaccardExtractor())
        timeline = tracker.analyze(snaps)
        assert len(timeline.stability) > 0
        for score in timeline.stability.values():
            assert 0.0 <= score <= 1.0
