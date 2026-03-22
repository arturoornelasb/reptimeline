"""
Tests for AutoLabeler — embedding-based and contrastive labeling.
"""

import numpy as np
import pytest

from reptimeline.autolabel import AutoLabeler
from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery
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


class TestLabelByLLM:

    def test_llm_labels(self, labeled_snapshot):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)

        def mock_llm(prompt):
            return "test_category"

        labeler = AutoLabeler()
        labels = labeler.label_by_llm(report, llm_fn=mock_llm)
        active = [lbl for lbl in labels if lbl.label != "DEAD"]
        assert len(active) > 0
        for lbl in active:
            assert lbl.method == 'llm'
            assert lbl.label == 'test_category'
            assert lbl.confidence == 0.8

    def test_llm_error_handling(self, labeled_snapshot):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)

        def failing_llm(prompt):
            raise RuntimeError("API down")

        labeler = AutoLabeler()
        labels = labeler.label_by_llm(report, llm_fn=failing_llm)
        active = [lbl for lbl in labels if lbl.label != "DEAD"]
        for lbl in active:
            assert "ERROR" in lbl.label


class TestPrintLabels:

    def test_output(self, labeled_snapshot, fake_embeddings, capsys):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)
        labeler = AutoLabeler()
        labels = labeler.label_by_embedding(report, fake_embeddings)
        labeler.print_labels(labels)
        captured = capsys.readouterr()
        assert "AUTO-LABELED BITS" in captured.out


class TestExportAsPrimitives:

    def test_export(self, labeled_snapshot, fake_embeddings, tmp_path):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)
        labeler = AutoLabeler()
        labels = labeler.label_by_embedding(report, fake_embeddings)

        out_path = str(tmp_path / "discovered.json")
        labeler.export_as_primitives(labels, out_path)

        import json
        with open(out_path) as f:
            data = json.load(f)
        assert 'primitivos' in data
        assert data['version'] == 'discovered_1.0'
        assert len(data['primitivos']) > 0


class TestZeroNormFallback:
    """Verify that zero-norm centroids produce fallback labels instead of being dropped."""

    def test_embedding_zero_norm_centroid(self, labeled_snapshot, fake_embeddings):
        """When all active concept embeddings cancel out, a fallback label is emitted."""
        # Create embeddings that produce a zero centroid for some bit
        zero_embeddings = dict(fake_embeddings)
        # Override so that active concepts for bit 0 (dog,cat,wolf,lion) cancel out
        zero_embeddings['dog'] = np.array([1.0, 0.0])
        zero_embeddings['cat'] = np.array([-1.0, 0.0])
        zero_embeddings['wolf'] = np.array([0.0, 1.0])
        zero_embeddings['lion'] = np.array([0.0, -1.0])

        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)

        labeler = AutoLabeler()
        labels = labeler.label_by_embedding(report, zero_embeddings,
                                            candidate_labels=['animal', 'abstract', 'food'])

        # Every bit_semantics entry must produce a label (no silent drops)
        assert len(labels) == len(report.bit_semantics)

    def test_contrastive_zero_norm_direction(self, labeled_snapshot, fake_embeddings):
        """When active/inactive centroids are identical, a fallback label is emitted."""
        # Make active and inactive embeddings identical so direction = 0
        uniform = np.array([0.5, 0.5])
        uniform_embeddings = {k: uniform.copy() for k in fake_embeddings}

        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(labeled_snapshot, top_k=4)

        labeler = AutoLabeler()
        labels = labeler.label_by_contrast(report, uniform_embeddings,
                                           candidate_labels=['animal', 'abstract', 'food'])

        assert len(labels) == len(report.bit_semantics)
        # Fallback labels should have confidence 0.0
        fallbacks = [lbl for lbl in labels if lbl.confidence == 0.0 and lbl.label != "DEAD"]
        assert len(fallbacks) > 0


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
        from reptimeline import BitDiscovery, TimelineTracker
        assert TimelineTracker is not None
        assert BitDiscovery is not None

    def test_extractor_base_import(self):
        from reptimeline.extractors.base import RepresentationExtractor
        assert RepresentationExtractor is not None
