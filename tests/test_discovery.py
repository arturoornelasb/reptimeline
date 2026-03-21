"""
Test BitDiscovery against synthetic data with known patterns.
"""

import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.discovery import BitDiscovery
from reptimeline.tracker import TimelineTracker
from tests.conftest import make_code, SyntheticExtractor


class TestDiscoverySynthetic:
    """Test discovery with synthetic data that has clear patterns."""

    @pytest.fixture
    def structured_snapshot(self):
        """Snapshot with intentional structure:
        - Bits 0,1 = "basic" (active in almost everything)
        - Bits 2,3 = dual pair (never both active)
        - Bit 4 depends on bit 0 (only active when 0 is active)
        - Bits 14,15 = dead (never active)
        - Triadic: bit 13 activates ONLY when bits 2 AND 5 are both active
        """
        n_bits = 16
        return ConceptSnapshot(
            step=5000,
            codes={
                'fire':    make_code(n_bits, [0, 1, 2, 5, 6, 13]),
                'water':   make_code(n_bits, [0, 1, 2, 7]),
                'earth':   make_code(n_bits, [0, 1, 2, 8]),
                'stone':   make_code(n_bits, [0, 1, 2, 8, 9]),
                'metal':   make_code(n_bits, [0, 1, 2, 8, 9, 10]),
                'wind':    make_code(n_bits, [0, 1, 2, 5, 13]),
                'storm':   make_code(n_bits, [0, 1, 2, 5, 6, 13]),
                'love':    make_code(n_bits, [0, 1, 3, 4, 11]),
                'hate':    make_code(n_bits, [0, 1, 3, 4, 12]),
                'truth':   make_code(n_bits, [0, 1, 3, 11]),
                'lie':     make_code(n_bits, [0, 1, 3, 12]),
                'justice': make_code(n_bits, [0, 1, 3, 4, 11]),
                'freedom': make_code(n_bits, [0, 1, 3, 4]),
                'spirit':  make_code(n_bits, [0, 1, 3, 5]),
                'soul':    make_code(n_bits, [0, 1, 3, 5, 11]),
                'dream':   make_code(n_bits, [0, 1, 3, 5, 12]),
                'void':    make_code(n_bits, [0]),
                'nothing': make_code(n_bits, []),
            },
        )

    def test_basic_activation_rates(self, structured_snapshot):
        discovery = BitDiscovery(dead_threshold=0.05, dual_threshold=-0.3,
                                 dep_confidence=0.85)
        report = discovery.discover(structured_snapshot, top_k=5)
        rates = {bs.bit_index: bs.activation_rate for bs in report.bit_semantics}

        assert rates[0] > 0.8, f"Bit 0 should be very active, got {rates[0]}"
        assert rates[1] > 0.8, f"Bit 1 should be very active, got {rates[1]}"

    def test_dead_bits(self, structured_snapshot):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(structured_snapshot, top_k=5)
        rates = {bs.bit_index: bs.activation_rate for bs in report.bit_semantics}

        assert rates[14] < 0.05, f"Bit 14 should be dead, got {rates[14]}"
        assert rates[15] < 0.05, f"Bit 15 should be dead, got {rates[15]}"
        assert report.n_dead_bits >= 2

    def test_dual_discovery(self, structured_snapshot):
        discovery = BitDiscovery(dead_threshold=0.05, dual_threshold=-0.3)
        report = discovery.discover(structured_snapshot, top_k=5)

        dual_pairs = {(d.bit_a, d.bit_b) for d in report.discovered_duals}
        assert (2, 3) in dual_pairs, f"Bits 2,3 should be duals. Found: {dual_pairs}"

    def test_dependency_discovery(self, structured_snapshot):
        discovery = BitDiscovery(dep_confidence=0.85)
        report = discovery.discover(structured_snapshot, top_k=5)

        has_dep = any(d.bit_child == 4 for d in report.discovered_deps)
        assert has_dep, "Bit 4 should have a dependency"

    def test_triadic_discovery(self, structured_snapshot):
        discovery = BitDiscovery(dead_threshold=0.05)
        report = discovery.discover(structured_snapshot, top_k=5)

        has_triadic = any(
            t.bit_r == 13 and set([t.bit_i, t.bit_j]) == {2, 5}
            for t in report.discovered_triadic_deps
        )
        assert has_triadic, "Should find triadic 2+5->13"


class TestDiscoveryWithTimeline:
    """Test hierarchy discovery with synthetic timeline."""

    def test_hierarchy_order(self):
        n_bits = 8
        extractor = SyntheticExtractor()

        snapshots = [
            ConceptSnapshot(step=0, codes={
                'a': make_code(n_bits, [0, 1]),
                'b': make_code(n_bits, [0]),
            }),
            ConceptSnapshot(step=1000, codes={
                'a': make_code(n_bits, [0, 1, 2]),
                'b': make_code(n_bits, [0, 3]),
            }),
            ConceptSnapshot(step=2000, codes={
                'a': make_code(n_bits, [0, 1, 2]),
                'b': make_code(n_bits, [0, 3]),
            }),
            ConceptSnapshot(step=3000, codes={
                'a': make_code(n_bits, [0, 1, 2]),
                'b': make_code(n_bits, [0, 3]),
            }),
            ConceptSnapshot(step=4000, codes={
                'a': make_code(n_bits, [0, 1, 2, 4]),
                'b': make_code(n_bits, [0, 3, 5]),
            }),
            ConceptSnapshot(step=5000, codes={
                'a': make_code(n_bits, [0, 1, 2, 4]),
                'b': make_code(n_bits, [0, 3, 5]),
            }),
            ConceptSnapshot(step=6000, codes={
                'a': make_code(n_bits, [0, 1, 2, 4]),
                'b': make_code(n_bits, [0, 3, 5]),
            }),
        ]

        tracker = TimelineTracker(extractor)
        timeline = tracker.analyze(snapshots)

        discovery = BitDiscovery()
        report = discovery.discover(snapshots[-1], timeline=timeline, top_k=3)

        hierarchy = {h.bit_index: h for h in report.discovered_hierarchy}
        if hierarchy.get(0) and hierarchy.get(4):
            if hierarchy[0].first_stable_step and hierarchy[4].first_stable_step:
                assert hierarchy[0].first_stable_step <= hierarchy[4].first_stable_step, \
                    "Bit 0 should stabilize before bit 4"
