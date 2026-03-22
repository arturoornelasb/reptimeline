"""Tests for Timeline export (CSV, JSON round-trip)."""

import os

import pytest

from reptimeline.core import (
    CodeEvent,
    ConceptSnapshot,
    ConnectionEvent,
    PhaseTransition,
    Timeline,
)


@pytest.fixture
def sample_timeline():
    """A minimal Timeline for export tests."""
    snap0 = ConceptSnapshot(step=0, codes={'a': [0, 1, 0], 'b': [1, 0, 0]})
    snap1 = ConceptSnapshot(step=100, codes={'a': [1, 1, 0], 'b': [1, 0, 1]})
    return Timeline(
        steps=[0, 100],
        snapshots=[snap0, snap1],
        births=[
            CodeEvent(event_type='birth', step=0, concept='a', code_index=1),
            CodeEvent(event_type='birth', step=100, concept='a', code_index=0),
        ],
        deaths=[
            CodeEvent(event_type='death', step=100, concept='b', code_index=0),
        ],
        connections=[
            ConnectionEvent(event_type='form', step=100,
                            concept_a='a', concept_b='b', shared_indices=[0]),
        ],
        phase_transitions=[
            PhaseTransition(step=100, metric='entropy', delta=0.5, direction='increase'),
        ],
        curves={'entropy': [0.3, 0.8], 'churn_rate': [0.0, 0.5]},
        stability={0: 0.5, 1: 1.0, 2: 0.75},
    )


class TestJsonRoundTrip:

    def test_to_dict_from_dict(self, sample_timeline):
        d = sample_timeline.to_dict()
        restored = Timeline.from_dict(d)

        assert restored.steps == [0, 100]
        assert len(restored.snapshots) == 2
        assert restored.snapshots[0].codes['a'] == [0, 1, 0]
        assert len(restored.births) == 2
        assert len(restored.deaths) == 1
        assert len(restored.connections) == 1
        assert len(restored.phase_transitions) == 1
        assert restored.curves['entropy'] == [0.3, 0.8]
        assert restored.stability[1] == 1.0

    def test_save_load_json(self, sample_timeline, tmp_path):
        path = str(tmp_path / "timeline.json")
        sample_timeline.save_json(path)

        restored = Timeline.load_json(path)
        assert restored.steps == sample_timeline.steps
        assert len(restored.births) == len(sample_timeline.births)
        assert restored.snapshots[1].codes['b'] == [1, 0, 1]


class TestCsvExport:

    def test_creates_all_files(self, sample_timeline, tmp_path):
        out_dir = str(tmp_path / "export")
        written = sample_timeline.to_csv(out_dir)

        assert 'events.csv' in written
        assert 'connections.csv' in written
        assert 'curves.csv' in written
        assert 'stability.csv' in written
        assert 'codes.csv' in written

        for path in written.values():
            assert os.path.exists(path)

    def test_events_csv_content(self, sample_timeline, tmp_path):
        out_dir = str(tmp_path / "export")
        written = sample_timeline.to_csv(out_dir)

        with open(written['events.csv'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        assert lines[0].strip() == 'step,event_type,concept,code_index'
        # 2 births + 1 death = 3 data rows
        assert len(lines) == 4

    def test_curves_csv_content(self, sample_timeline, tmp_path):
        out_dir = str(tmp_path / "export")
        written = sample_timeline.to_csv(out_dir)

        with open(written['curves.csv'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header = lines[0].strip()
        assert 'step' in header
        assert 'entropy' in header
        assert 'churn_rate' in header
        # 2 steps = 2 data rows
        assert len(lines) == 3

    def test_codes_csv_content(self, sample_timeline, tmp_path):
        out_dir = str(tmp_path / "export")
        written = sample_timeline.to_csv(out_dir)

        with open(written['codes.csv'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        header = lines[0].strip()
        assert 'step' in header
        assert 'concept' in header
        assert 'bit_0' in header
        # 2 snapshots x 2 concepts = 4 data rows
        assert len(lines) == 5

    def test_stability_csv_content(self, sample_timeline, tmp_path):
        out_dir = str(tmp_path / "export")
        written = sample_timeline.to_csv(out_dir)

        with open(written['stability.csv'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        assert lines[0].strip() == 'bit_index,stability'
        assert len(lines) == 4  # 3 bits

    def test_empty_stability_skips_file(self, tmp_path):
        tl = Timeline(
            steps=[0], snapshots=[ConceptSnapshot(step=0, codes={'a': [1]})],
            births=[], deaths=[], connections=[], phase_transitions=[],
            curves={}, stability={},
        )
        written = tl.to_csv(str(tmp_path / "export"))
        assert 'stability.csv' not in written
