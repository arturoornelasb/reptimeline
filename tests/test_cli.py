"""Tests for the CLI module."""

import json
import os
import subprocess
import sys
import pytest

from reptimeline.cli import _load_snapshots, _save_timeline, main
from reptimeline.core import ConceptSnapshot, Timeline, CodeEvent, ConnectionEvent, PhaseTransition


class TestLoadSnapshots:

    def test_load_list_format(self, tmp_path):
        """Load snapshots from a bare list."""
        data = [
            {"step": 100, "codes": {"a": [1, 0], "b": [0, 1]}},
            {"step": 200, "codes": {"a": [1, 1], "b": [0, 1]}},
        ]
        path = tmp_path / "snaps.json"
        path.write_text(json.dumps(data), encoding='utf-8')
        snapshots = _load_snapshots(str(path))
        assert len(snapshots) == 2
        assert snapshots[0].step == 100
        assert snapshots[1].step == 200

    def test_load_dict_wrapper_format(self, tmp_path):
        """Load snapshots from a dict with 'snapshots' key."""
        data = {
            "snapshots": [
                {"step": 100, "codes": {"a": [1, 0]}},
                {"step": 200, "codes": {"a": [0, 1]}},
            ]
        }
        path = tmp_path / "snaps.json"
        path.write_text(json.dumps(data), encoding='utf-8')
        snapshots = _load_snapshots(str(path))
        assert len(snapshots) == 2

    def test_load_invalid_format_raises(self, tmp_path):
        """Invalid JSON format raises ValueError."""
        data = {"wrong_key": [1, 2, 3]}
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(data), encoding='utf-8')
        with pytest.raises((ValueError, KeyError)):
            _load_snapshots(str(path))

    def test_snapshots_sorted_by_step(self, tmp_path):
        """Snapshots should be sorted by step regardless of input order."""
        data = [
            {"step": 300, "codes": {"a": [1, 0]}},
            {"step": 100, "codes": {"a": [0, 1]}},
            {"step": 200, "codes": {"a": [1, 1]}},
        ]
        path = tmp_path / "unordered.json"
        path.write_text(json.dumps(data), encoding='utf-8')
        snapshots = _load_snapshots(str(path))
        assert [s.step for s in snapshots] == [100, 200, 300]


class TestSaveTimeline:

    def test_save_and_reload(self, tmp_path):
        timeline = Timeline(
            steps=[0, 1000],
            snapshots=[
                ConceptSnapshot(step=0, codes={'a': [1, 0]}),
                ConceptSnapshot(step=1000, codes={'a': [0, 1]}),
            ],
            births=[CodeEvent(event_type='birth', step=0, concept='a', code_index=0)],
            deaths=[],
            connections=[],
            phase_transitions=[],
            curves={'entropy': [0.5, 0.8]},
            stability={0: 0.5, 1: 0.5},
        )
        out = str(tmp_path / "timeline.json")
        _save_timeline(timeline, out)

        with open(out) as f:
            data = json.load(f)
        assert data['steps'] == [0, 1000]
        assert len(data['births']) == 1
        assert 'entropy' in data['curves']


class TestMainCLI:

    def test_basic_analysis(self, tmp_path, monkeypatch, capsys):
        """Test main() with minimal snapshots."""
        data = [
            {"step": 0, "codes": {"a": [1, 0], "b": [0, 1]}},
            {"step": 100, "codes": {"a": [1, 1], "b": [0, 1]}},
        ]
        snap_file = tmp_path / "snaps.json"
        snap_file.write_text(json.dumps(data), encoding='utf-8')

        monkeypatch.setattr(sys, 'argv',
                            ['reptimeline', '--snapshots', str(snap_file)])
        main()
        captured = capsys.readouterr()
        assert "REPRESENTATION TIMELINE" in captured.out

    def test_with_output(self, tmp_path, monkeypatch):
        """Test --output saves JSON."""
        data = [
            {"step": 0, "codes": {"a": [1, 0]}},
            {"step": 100, "codes": {"a": [0, 1]}},
        ]
        snap_file = tmp_path / "snaps.json"
        snap_file.write_text(json.dumps(data), encoding='utf-8')
        out_file = tmp_path / "result.json"

        monkeypatch.setattr(sys, 'argv',
                            ['reptimeline', '--snapshots', str(snap_file),
                             '--output', str(out_file)])
        main()
        assert out_file.exists()
        with open(out_file) as f:
            result = json.load(f)
        assert 'steps' in result

    def test_with_discover(self, tmp_path, monkeypatch, capsys):
        """Test --discover runs BitDiscovery."""
        data = [
            {"step": 0, "codes": {"a": [1, 0, 1], "b": [0, 1, 0], "c": [1, 1, 0]}},
            {"step": 100, "codes": {"a": [1, 0, 1], "b": [0, 1, 0], "c": [1, 1, 0]}},
        ]
        snap_file = tmp_path / "snaps.json"
        snap_file.write_text(json.dumps(data), encoding='utf-8')

        monkeypatch.setattr(sys, 'argv',
                            ['reptimeline', '--snapshots', str(snap_file), '--discover'])
        main()
        captured = capsys.readouterr()
        assert "DISCOVERY" in captured.out or "Active bits" in captured.out

    def test_with_overlay(self, tmp_path, monkeypatch, capsys):
        """Test --overlay loads PrimitiveOverlay and prints report."""
        data = [
            {"step": 0, "codes": {"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]}},
            {"step": 100, "codes": {"a": [1, 0, 1, 0], "b": [0, 1, 0, 1]}},
        ]
        snap_file = tmp_path / "snaps.json"
        snap_file.write_text(json.dumps(data), encoding='utf-8')

        prims = {
            "primitivos": [
                {"bit": 0, "nombre": "alpha", "primo": 2, "capa": 1, "deps": []},
                {"bit": 1, "nombre": "beta", "primo": 3, "capa": 1, "deps": []},
            ]
        }
        prim_file = tmp_path / "primitivos.json"
        prim_file.write_text(json.dumps(prims), encoding='utf-8')

        monkeypatch.setattr(sys, 'argv',
                            ['reptimeline', '--snapshots', str(snap_file),
                             '--overlay', str(prim_file)])
        main()
        captured = capsys.readouterr()
        assert "PRIMITIVE" in captured.out or "OVERLAY" in captured.out or "alpha" in captured.out

    def test_with_plot(self, tmp_path, monkeypatch, capsys):
        """Test --plot generates plot files."""
        import matplotlib
        matplotlib.use('Agg')

        data = [
            {"step": 0, "codes": {"a": [1, 0], "b": [0, 1]}},
            {"step": 100, "codes": {"a": [1, 1], "b": [0, 1]}},
        ]
        snap_file = tmp_path / "snaps.json"
        snap_file.write_text(json.dumps(data), encoding='utf-8')
        plot_dir = tmp_path / "plots"

        monkeypatch.setattr(sys, 'argv',
                            ['reptimeline', '--snapshots', str(snap_file),
                             '--plot', '--plot-dir', str(plot_dir)])
        main()
        assert plot_dir.exists()
        assert (plot_dir / "swimlane.png").exists()
        assert (plot_dir / "phase_dashboard.png").exists()


class TestCausalCLI:

    def test_with_causal(self, tmp_path, monkeypatch, capsys):
        """Test --causal loads effects and runs CausalVerifier."""
        data = [
            {"step": 0, "codes": {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}},
            {"step": 100, "codes": {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}},
        ]
        snap_file = tmp_path / "snaps.json"
        snap_file.write_text(json.dumps(data), encoding='utf-8')

        effects = {
            "effects": {
                "bit_0": {"a": 5.0, "b": 0.1, "c": 0.2},
                "bit_1": {"a": 0.1, "b": 4.0, "c": 0.1},
                "bit_2": {"a": 0.2, "b": 0.1, "c": 3.0},
            }
        }
        effects_file = tmp_path / "effects.json"
        effects_file.write_text(json.dumps(effects), encoding='utf-8')

        monkeypatch.setattr(sys, 'argv',
                            ['reptimeline', '--snapshots', str(snap_file),
                             '--causal', str(effects_file)])
        main()
        captured = capsys.readouterr()
        assert "CAUSAL VERIFICATION" in captured.out


class TestMainModule:

    def test_dunder_main_runs(self, tmp_path):
        """Test python -m reptimeline invokes the CLI."""
        data = [
            {"step": 0, "codes": {"x": [1, 0]}},
            {"step": 100, "codes": {"x": [0, 1]}},
        ]
        snap_file = tmp_path / "snaps.json"
        snap_file.write_text(json.dumps(data), encoding='utf-8')

        result = subprocess.run(
            [sys.executable, '-m', 'reptimeline',
             '--snapshots', str(snap_file)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "REPRESENTATION TIMELINE" in result.stdout
