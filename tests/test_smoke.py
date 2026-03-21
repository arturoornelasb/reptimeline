"""
Smoke tests -- verify reptimeline core works without model checkpoints.
"""

from reptimeline.tracker import TimelineTracker


class TestBasicPipeline:
    """Test tracker with synthetic evolving codes."""

    def test_timeline_steps(self, synthetic_extractor, basic_snapshots):
        tracker = TimelineTracker(synthetic_extractor, stability_window=2)
        timeline = tracker.analyze(basic_snapshots)

        assert len(timeline.steps) == 5
        assert timeline.steps == [0, 1000, 2000, 3000, 4000]

    def test_births_detected(self, synthetic_extractor, basic_snapshots):
        tracker = TimelineTracker(synthetic_extractor, stability_window=2)
        timeline = tracker.analyze(basic_snapshots)

        assert len(timeline.births) > 0

    def test_deaths_detected(self, synthetic_extractor, basic_snapshots):
        tracker = TimelineTracker(synthetic_extractor, stability_window=2)
        timeline = tracker.analyze(basic_snapshots)

        assert len(timeline.deaths) >= 1
        king_deaths = [d for d in timeline.deaths if d.concept == 'king']
        assert any(d.code_index == 0 for d in king_deaths), \
            "Expected king to lose bit 0"

    def test_connections_detected(self, synthetic_extractor, basic_snapshots):
        tracker = TimelineTracker(synthetic_extractor, stability_window=2)
        timeline = tracker.analyze(basic_snapshots)

        assert len(timeline.connections) > 0

    def test_curves_correct_length(self, synthetic_extractor, basic_snapshots):
        tracker = TimelineTracker(synthetic_extractor, stability_window=2)
        timeline = tracker.analyze(basic_snapshots)

        assert len(timeline.curves['entropy']) == 5
        assert len(timeline.curves['churn_rate']) == 5
        assert len(timeline.curves['utilization']) == 5

    def test_stability_computed(self, synthetic_extractor, basic_snapshots):
        tracker = TimelineTracker(synthetic_extractor, stability_window=2)
        timeline = tracker.analyze(basic_snapshots)

        assert len(timeline.stability) > 0

    def test_print_summary(self, synthetic_extractor, basic_snapshots, capsys):
        tracker = TimelineTracker(synthetic_extractor, stability_window=2)
        timeline = tracker.analyze(basic_snapshots)
        timeline.print_summary()

        captured = capsys.readouterr()
        assert "REPRESENTATION TIMELINE" in captured.out

    def test_to_dict_roundtrip(self, synthetic_extractor, basic_snapshots):
        tracker = TimelineTracker(synthetic_extractor, stability_window=2)
        timeline = tracker.analyze(basic_snapshots)

        data = timeline.to_dict()
        assert 'steps' in data
        assert 'births' in data
        assert 'curves' in data
        assert len(data['steps']) == 5


class TestConceptSnapshot:
    """Test ConceptSnapshot methods."""

    def test_hamming(self, basic_snapshots):
        snap = basic_snapshots[0]
        # king and queen have the same code at step 0
        assert snap.hamming('king', 'queen') == 0

    def test_active_indices(self, basic_snapshots):
        snap = basic_snapshots[0]
        assert snap.active_indices('king') == [0, 1]
        assert snap.active_indices('fire') == [2]

    def test_serialization(self, basic_snapshots):
        snap = basic_snapshots[0]
        d = snap.to_dict()
        restored = type(snap).from_dict(d)
        assert restored.step == snap.step
        assert restored.codes == snap.codes
