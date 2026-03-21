"""Tests for the PrimitiveOverlay module."""

import json
import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.tracker import TimelineTracker
from reptimeline.overlays.primitive_overlay import (
    PrimitiveOverlay, PrimitiveReport, ActivationEpoch,
    DepsCompletion, LayerEmergence, DualCoherence,
)
from reptimeline.extractors.base import RepresentationExtractor


class SyntheticExtractor(RepresentationExtractor):
    def extract(self, checkpoint_path, concepts, device='cpu'):
        raise NotImplementedError
    def similarity(self, code_a, code_b):
        active_a = set(i for i, v in enumerate(code_a) if v == 1)
        active_b = set(i for i, v in enumerate(code_b) if v == 1)
        union = active_a | active_b
        return 1.0 if not union else len(active_a & active_b) / len(union)
    def shared_features(self, code_a, code_b):
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]


def make_code(n_bits, active_indices):
    code = [0] * n_bits
    for i in active_indices:
        code[i] = 1
    return code


# ── fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def primitivos_json(tmp_path):
    """Primitivos file mapping to bits 0-3 in 2 layers with duals and deps."""
    data = {
        "primitivos": [
            {"bit": 0, "nombre": "alive", "primo": 2, "capa": 1,
             "deps": [], "dual": "dead", "def": "entity is alive"},
            {"bit": 1, "nombre": "dead", "primo": 3, "capa": 1,
             "deps": [], "dual": "alive", "def": "entity is dead"},
            {"bit": 2, "nombre": "big", "primo": 5, "capa": 2,
             "deps": ["alive"], "def": "entity is big"},
            {"bit": 3, "nombre": "small", "primo": 7, "capa": 2,
             "deps": ["alive"], "dual": "big", "def": "entity is small"},
        ]
    }
    path = tmp_path / "primitivos.json"
    path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


@pytest.fixture
def overlay(primitivos_json):
    return PrimitiveOverlay(primitivos_json)


@pytest.fixture
def overlay_timeline():
    """Timeline where bits activate at known steps for controlled testing.

    Step 0:   king=[alive],         queen=[dead]
    Step 1000: king=[alive,big],     queen=[dead,small]
    Step 2000: king=[alive,big],     queen=[dead,small]  (stable)
    """
    n_bits = 8
    snaps = [
        ConceptSnapshot(step=0, codes={
            'king': make_code(n_bits, [0]),        # alive
            'queen': make_code(n_bits, [1]),        # dead
        }),
        ConceptSnapshot(step=1000, codes={
            'king': make_code(n_bits, [0, 2]),     # alive, big
            'queen': make_code(n_bits, [1, 3]),     # dead, small
        }),
        ConceptSnapshot(step=2000, codes={
            'king': make_code(n_bits, [0, 2]),     # alive, big (stable)
            'queen': make_code(n_bits, [1, 3]),     # dead, small (stable)
        }),
    ]
    extractor = SyntheticExtractor()
    tracker = TimelineTracker(extractor)
    return tracker.analyze(snaps)


# ── __init__ ──────────────────────────────────────────────────────

class TestInit:

    def test_raises_without_path(self):
        with pytest.raises(ValueError, match="primitivos_path is required"):
            PrimitiveOverlay(None)

    def test_loads_primitives(self, overlay):
        assert len(overlay.primitives) == 4
        assert overlay._name_to_bit['alive'] == 0
        assert overlay._name_to_bit['dead'] == 1
        assert overlay._bit_to_name[2] == 'big'

    def test_parses_deps(self, overlay):
        big = overlay._name_to_info['big']
        assert big.deps == ['alive']
        assert big.layer == 2


# ── analyze ───────────────────────────────────────────────────────

class TestAnalyze:

    def test_returns_report(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        assert isinstance(report, PrimitiveReport)
        assert report.metadata['n_primitives'] == 4
        assert report.metadata['n_concepts'] == 2
        assert report.metadata['n_steps'] == 3

    def test_with_concept_subset(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline, concepts=['king'])
        assert report.metadata['n_concepts'] == 1
        # Only king activations
        concepts_found = {a.concept for a in report.activations}
        assert concepts_found == {'king'}


# ── activations ───────────────────────────────────────────────────

class TestActivations:

    def test_activation_steps(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        # alive (bit 0) activates at step 0 for king
        king_alive = [a for a in report.activations
                      if a.primitive == 'alive' and a.concept == 'king']
        assert len(king_alive) == 1
        assert king_alive[0].step == 0

        # big (bit 2) activates at step 1000 for king
        king_big = [a for a in report.activations
                    if a.primitive == 'big' and a.concept == 'king']
        assert len(king_big) == 1
        assert king_big[0].step == 1000

    def test_no_activation_for_missing_bit(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        # queen never activates alive (bit 0)
        queen_alive = [a for a in report.activations
                       if a.primitive == 'alive' and a.concept == 'queen']
        assert len(queen_alive) == 0


# ── deps completions ─────────────────────────────────────────────

class TestDepsCompletions:

    def test_dep_completion_detected(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        # big depends on alive; king has both at step 1000
        king_big_deps = [d for d in report.deps_completions
                         if d.primitive == 'big' and d.concept == 'king']
        assert len(king_big_deps) == 1
        assert king_big_deps[0].deps == ['alive']

    def test_no_completion_when_dep_missing(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        # queen has dead but not alive, so big/small deps on alive never complete for queen
        queen_big = [d for d in report.deps_completions
                     if d.primitive == 'big' and d.concept == 'queen']
        assert len(queen_big) == 0


# ── layer emergence ──────────────────────────────────────────────

class TestLayerEmergence:

    def test_layer_order(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        layers = report.layer_emergence
        assert len(layers) == 2  # layer 1, layer 2
        assert layers[0].layer == 1
        assert layers[1].layer == 2

    def test_layer1_activates_first(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        l1 = report.layer_emergence[0]
        l2 = report.layer_emergence[1]
        assert l1.first_activation_step <= l2.first_activation_step

    def test_primitives_activated_count(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        l1 = report.layer_emergence[0]
        assert l1.primitives_activated == 2  # alive and dead both activate


# ── dual coherence ───────────────────────────────────────────────

class TestDualCoherence:

    def test_dual_pairs_detected(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        assert len(report.dual_coherence) >= 1
        pair_names = {(d.primitive_a, d.primitive_b) for d in report.dual_coherence}
        # alive-dead is a declared dual
        assert ('alive', 'dead') in pair_names or ('dead', 'alive') in pair_names

    def test_coherence_score_for_anti_correlated(self, overlay, overlay_timeline):
        report = overlay.analyze(overlay_timeline)
        # alive and dead are perfectly anti-correlated in our data
        alive_dead = [d for d in report.dual_coherence
                      if {d.primitive_a, d.primitive_b} == {'alive', 'dead'}]
        assert len(alive_dead) == 1
        # king has alive, queen has dead -- never both for same concept
        assert alive_dead[0].coherence_score > 0.5


# ── print_report ─────────────────────────────────────────────────

class TestPrintReport:

    def test_output(self, overlay, overlay_timeline, capsys):
        report = overlay.analyze(overlay_timeline)
        overlay.print_report(report)
        captured = capsys.readouterr()
        assert "PRIMITIVE OVERLAY REPORT" in captured.out
        assert "LAYER EMERGENCE" in captured.out
        assert "DUAL AXIS COHERENCE" in captured.out
