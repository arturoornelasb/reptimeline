"""
Tests for AutoLabeler — embedding-based and contrastive labeling.
"""

import numpy as np
import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery
from reptimeline.autolabel import AutoLabeler
from tests.conftest import make_code


@pytest.fixture
def labeled_snapshot():
    """Snapshot where bit 0 = animals, bit 1 = abstract."""
    n_bits = 8
    return ConceptSnapshot(
        step=5000,
        codes={
            'dog': make_code(n_bits, [0, 2]),
            'cat': make_code(n_bits, [0, 3]),
            'wolf': make_code(n_bits, [0, 2, 4]),
            'lion': make_code(n_bits, [0, 3, 4]),
            'idea': make_code(n_bits, [1, 5]),
            'thought': make_code(n_bits, [1, 5, 6]),
            'logic': make_code(n_bits, [1, 6]),
            'reason': make_code(n_bits, [1, 5, 7]),
        },
    )


@pytest.fixture
def fake_embeddings():
    """Embeddings where animals cluster near [1,0] and abstract near [0,1]."""
    rng = np.random.RandomState(42)
    return {
        'dog': np.array([0.9, 0.1]) + rng.randn(2) * 0.05,
        'cat': np.array([0.85, 0.15]) + rng.randn(2) * 0.05,
        'wolf': np.array([0.95, 0.05]) + rng.randn(2) * 0.05,
        'lion': np.array([0.88, 0.12]) + rng.randn(2) * 0.05,
        'idea': np.array([0.1, 0.9]) + rng.randn(2) * 0.05,
        'thought': np.array([0.15, 0.85]) + rng.randn(2) * 0.05,
        'logic': np.array([0.05, 0.95]) + rng.randn(2) * 0.05,
        'reason': np.array([0.12, 0.88]) + rng.randn(2) * 0.05,
        # Candidate labels
        'animal': np.array([0.92, 0.08]),
        'abstract': np.array([0.08, 0.92]),
        'food': np.array([0.5, 0.5]),
    }


class TestAutoLabeler:

    def test_embedding_labels(self, labeled_snapshot, fake_embeddings):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)

        labeler = AutoLabeler()
        labels = labeler.label_by_embedding(
            report, fake_embeddings,
            candidate_labels=['animal', 'abstract', 'food'],
        )

        assert len(labels) > 0
        for lbl in labels:
            assert lbl.label in ('animal', 'abstract', 'food')
            assert 0 <= lbl.confidence <= 1
            assert lbl.method == 'embedding'

    def test_contrastive_labels(self, labeled_snapshot, fake_embeddings):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)

        labeler = AutoLabeler()
        labels = labeler.label_by_contrast(
            report, fake_embeddings,
            candidate_labels=['animal', 'abstract', 'food'],
        )

        assert len(labels) > 0
        for lbl in labels:
            assert lbl.method == 'contrastive'

    def test_label_to_dict(self, labeled_snapshot, fake_embeddings):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)

        labeler = AutoLabeler()
        labels = labeler.label_by_embedding(report, fake_embeddings)

        # Labels should be serializable
        for lbl in labels:
            d = {
                'bit': lbl.bit_index,
                'label': lbl.label,
                'confidence': lbl.confidence,
                'method': lbl.method,
            }
            assert isinstance(d['label'], str)
            assert isinstance(d['confidence'], float)


class TestImports:
    """Verify all public API imports work."""

    def test_core_imports(self):
        from reptimeline.core import ConceptSnapshot
        assert ConceptSnapshot is not None

    def test_tracker_imports(self):
        from reptimeline.tracker import TimelineTracker
        assert TimelineTracker is not None

    def test_discovery_imports(self):
        from reptimeline.discovery import BitDiscovery, DiscoveryReport
        assert BitDiscovery is not None
        assert DiscoveryReport is not None

    def test_autolabel_imports(self):
        from reptimeline.autolabel import AutoLabeler, BitLabel
        assert AutoLabeler is not None
        assert BitLabel is not None

    def test_top_level_imports(self):
        from reptimeline import TimelineTracker, BitDiscovery
        assert TimelineTracker is not None
        assert BitDiscovery is not None

    def test_extractor_base_import(self):
        from reptimeline.extractors.base import RepresentationExtractor
        assert RepresentationExtractor is not None
