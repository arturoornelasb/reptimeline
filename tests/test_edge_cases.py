"""Tests for edge cases and boundary conditions."""

import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.tracker import TimelineTracker
from tests.conftest import make_code, SyntheticExtractor


class TestEdgeCases:

    def test_single_concept(self):
        """Single concept should not crash."""
        extractor = SyntheticExtractor()
        snapshots = [
            ConceptSnapshot(step=0, codes={'only': make_code(8, [0, 1])}),
            ConceptSnapshot(step=1000, codes={'only': make_code(8, [0, 1, 2])}),
        ]
        tracker = TimelineTracker(extractor)
        timeline = tracker.analyze(snapshots)
        assert len(timeline.births) > 0
        assert len(timeline.steps) == 2

    def test_single_snapshot(self):
        """Single snapshot (no transitions possible)."""
        extractor = SyntheticExtractor()
        snapshots = [
            ConceptSnapshot(step=0, codes={
                'a': make_code(8, [0, 1]),
                'b': make_code(8, [2, 3]),
            }),
        ]
        tracker = TimelineTracker(extractor)
        timeline = tracker.analyze(snapshots)
        assert len(timeline.steps) == 1
        assert timeline.curves['churn_rate'] == [0.0]
        assert timeline.stability == {}

    def test_empty_codes(self):
        """Concept with all-zero code."""
        extractor = SyntheticExtractor()
        snapshots = [
            ConceptSnapshot(step=0, codes={
                'empty': make_code(8, []),
                'notempty': make_code(8, [0]),
            }),
            ConceptSnapshot(step=1000, codes={
                'empty': make_code(8, []),
                'notempty': make_code(8, [0, 1]),
            }),
        ]
        tracker = TimelineTracker(extractor)
        timeline = tracker.analyze(snapshots)
        assert timeline is not None
        # empty concept should have no births
        empty_births = [b for b in timeline.births if b.concept == 'empty']
        assert len(empty_births) == 0

    def test_all_zero_codes(self):
        """All concepts have all-zero codes."""
        extractor = SyntheticExtractor()
        snapshots = [
            ConceptSnapshot(step=0, codes={
                'a': [0, 0, 0, 0],
                'b': [0, 0, 0, 0],
            }),
        ]
        tracker = TimelineTracker(extractor)
        timeline = tracker.analyze(snapshots)
        assert len(timeline.births) == 0

    def test_all_one_codes(self):
        """All concepts have all-one codes."""
        extractor = SyntheticExtractor()
        snapshots = [
            ConceptSnapshot(step=0, codes={
                'a': [1, 1, 1, 1],
                'b': [1, 1, 1, 1],
            }),
        ]
        tracker = TimelineTracker(extractor)
        timeline = tracker.analyze(snapshots)
        assert len(timeline.births) > 0

    def test_mismatched_code_lengths_raises(self):
        """Mismatched code lengths should raise ValueError."""
        snap = ConceptSnapshot(step=0, codes={
            'short': [0, 1],
            'long': [0, 1, 0, 1],
        })
        with pytest.raises(ValueError, match="Inconsistent code lengths"):
            snap.validate()

    def test_mismatched_codes_rejected_by_tracker(self):
        """Tracker should reject snapshots with inconsistent code lengths."""
        extractor = SyntheticExtractor()
        snapshots = [
            ConceptSnapshot(step=0, codes={
                'short': [0, 1],
                'long': [0, 1, 0, 1],
            }),
        ]
        tracker = TimelineTracker(extractor)
        with pytest.raises(ValueError, match="Inconsistent code lengths"):
            tracker.analyze(snapshots)

    def test_consistent_codes_pass_validation(self):
        """Consistent code lengths should not raise."""
        snap = ConceptSnapshot(step=0, codes={
            'a': [0, 1, 0, 1],
            'b': [1, 0, 1, 0],
        })
        snap.validate()  # Should not raise
