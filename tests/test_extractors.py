"""Tests for extractors/base.py — discover_checkpoints and extract_sequence."""

import os
import pytest

from reptimeline.extractors.base import RepresentationExtractor
from reptimeline.core import ConceptSnapshot


class DummyExtractor(RepresentationExtractor):
    """Extractor that returns fixed codes based on step parsed from filename."""

    def extract(self, checkpoint_path, concepts, device='cpu'):
        step = int(os.path.basename(checkpoint_path).split('_step')[1].split('.')[0])
        return ConceptSnapshot(
            step=step,
            codes={c: [1, 0] for c in concepts},
        )

    def similarity(self, code_a, code_b):
        return 1.0

    def shared_features(self, code_a, code_b):
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]


class TestDiscoverCheckpoints:

    def test_model_step_pattern(self, tmp_path):
        (tmp_path / "model_step1000.pt").write_text("x")
        (tmp_path / "model_step2000.pt").write_text("x")
        (tmp_path / "model_step500.pt").write_text("x")

        ext = DummyExtractor()
        results = ext.discover_checkpoints(str(tmp_path))
        steps = [s for s, _ in results]
        assert steps == [500, 1000, 2000]

    def test_checkpoint_pattern(self, tmp_path):
        (tmp_path / "checkpoint-100.pt").write_text("x")
        (tmp_path / "checkpoint-200.pt").write_text("x")

        ext = DummyExtractor()
        results = ext.discover_checkpoints(str(tmp_path))
        assert len(results) == 2
        assert results[0][0] == 100

    def test_step_safetensors_pattern(self, tmp_path):
        (tmp_path / "step_50.safetensors").write_text("x")

        ext = DummyExtractor()
        results = ext.discover_checkpoints(str(tmp_path))
        assert len(results) == 1
        assert results[0][0] == 50

    def test_ignores_non_matching(self, tmp_path):
        (tmp_path / "model_best.pt").write_text("x")
        (tmp_path / "readme.txt").write_text("x")

        ext = DummyExtractor()
        results = ext.discover_checkpoints(str(tmp_path))
        assert len(results) == 0

    def test_empty_directory(self, tmp_path):
        ext = DummyExtractor()
        results = ext.discover_checkpoints(str(tmp_path))
        assert results == []

    def test_model_xl_step_pattern(self, tmp_path):
        (tmp_path / "model_xl_step300.pt").write_text("x")

        ext = DummyExtractor()
        results = ext.discover_checkpoints(str(tmp_path))
        assert len(results) == 1
        assert results[0][0] == 300


class TestExtractSequence:

    def test_extract_all(self, tmp_path):
        (tmp_path / "model_step100.pt").write_text("x")
        (tmp_path / "model_step200.pt").write_text("x")
        (tmp_path / "model_step300.pt").write_text("x")

        ext = DummyExtractor()
        snaps = ext.extract_sequence(str(tmp_path), ['a', 'b'])
        assert len(snaps) == 3
        assert all(isinstance(s, ConceptSnapshot) for s in snaps)

    def test_max_checkpoints(self, tmp_path):
        for i in range(10):
            (tmp_path / f"model_step{i * 100}.pt").write_text("x")

        ext = DummyExtractor()
        snaps = ext.extract_sequence(str(tmp_path), ['a'], max_checkpoints=3)
        assert len(snaps) == 3

    def test_no_checkpoints_raises(self, tmp_path):
        ext = DummyExtractor()
        with pytest.raises(FileNotFoundError):
            ext.extract_sequence(str(tmp_path), ['a'])
