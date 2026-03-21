"""Tests for the Reconciler module."""

import json
import pytest

from reptimeline.discovery import (
    BitSemantics, DiscoveredDual, DiscoveredDependency,
    DiscoveryReport,
)
from reptimeline.overlays.primitive_overlay import PrimitiveOverlay
from reptimeline.reconcile import Reconciler


@pytest.fixture
def primitivos_json(tmp_path):
    """Create a minimal primitivos.json for testing."""
    data = {
        "primitivos": [
            {"bit": 0, "nombre": "alive", "primo": 2, "capa": 1,
             "deps": [], "dual": "dead"},
            {"bit": 1, "nombre": "dead", "primo": 3, "capa": 1,
             "deps": [], "dual": "alive"},
            {"bit": 2, "nombre": "big", "primo": 5, "capa": 2,
             "deps": ["alive"]},
            {"bit": 3, "nombre": "small", "primo": 7, "capa": 2,
             "deps": ["alive"], "dual": "big"},
        ]
    }
    path = tmp_path / "primitivos.json"
    path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


@pytest.fixture
def overlay(primitivos_json):
    return PrimitiveOverlay(primitivos_json)


def _make_report(bit_rates, duals=None, deps=None):
    """Helper to create a DiscoveryReport with given activation rates."""
    semantics = []
    for bit_idx, rate in enumerate(bit_rates):
        semantics.append(BitSemantics(
            bit_index=bit_idx, activation_rate=rate,
            top_concepts=["c1", "c2"], anti_concepts=["c3"],
            label=f"bit_{bit_idx}" if rate > 0.02 else f"bit_{bit_idx}_DEAD",
        ))
    return DiscoveryReport(
        bit_semantics=semantics,
        discovered_duals=duals or [],
        discovered_deps=deps or [],
        discovered_triadic_deps=[],
        discovered_hierarchy=[],
        n_active_bits=sum(1 for r in bit_rates if r > 0.02),
        n_dead_bits=sum(1 for r in bit_rates if r <= 0.02),
        metadata={'n_concepts': 10, 'n_bits': len(bit_rates)},
    )


class TestReconciler:

    def test_dead_bit_detected(self, overlay):
        """Bit assigned as primitive but actually dead -> critical."""
        report = _make_report([0.01, 0.5, 0.4, 0.3])
        codes = {'c1': [0, 1, 1, 0], 'c2': [0, 1, 0, 1]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        critical = [m for m in result.bit_mismatches
                    if m.mismatch_type == 'dead_but_assigned']
        assert len(critical) >= 1
        assert critical[0].bit_index == 0
        assert critical[0].severity == 'critical'

    def test_active_unassigned(self, overlay):
        """Active bit not in primitives -> info mismatch."""
        report = _make_report([0.5, 0.5, 0.4, 0.3, 0.6, 0.7])
        codes = {'c1': [1, 1, 1, 0, 1, 1]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        unassigned = [m for m in result.bit_mismatches
                      if m.mismatch_type == 'active_but_unassigned']
        unassigned_bits = {m.bit_index for m in unassigned}
        assert 4 in unassigned_bits or 5 in unassigned_bits

    def test_dual_mismatch_detection(self, overlay):
        """Theory declares dual alive-dead but model doesn't find it."""
        report = _make_report([0.5, 0.5, 0.4, 0.3])  # no duals discovered
        codes = {'c1': [1, 1, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        missing = [m for m in result.dual_mismatches
                   if m.mismatch_type == 'missing_in_model']
        assert len(missing) >= 1

    def test_dependency_mismatch(self, overlay):
        """Theory says big depends on alive, but model doesn't find it."""
        report = _make_report([0.5, 0.5, 0.4, 0.3])  # no deps discovered
        codes = {'c1': [1, 1, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        dep_missing = [m for m in result.dep_mismatches
                       if m.mismatch_type == 'missing_in_model']
        assert len(dep_missing) >= 1

    def test_agreement_score_with_matches(self, overlay):
        """Known matching structure -> good agreement score."""
        duals = [DiscoveredDual(bit_a=0, bit_b=1, anti_correlation=-0.9,
                                concepts_exclusive=8, concepts_both=0),
                 DiscoveredDual(bit_a=2, bit_b=3, anti_correlation=-0.8,
                                concepts_exclusive=7, concepts_both=1)]
        deps = [DiscoveredDependency(bit_parent=0, bit_child=2,
                                     confidence=0.95, support=10),
                DiscoveredDependency(bit_parent=0, bit_child=3,
                                     confidence=0.92, support=10)]
        report = _make_report([0.5, 0.5, 0.4, 0.3], duals=duals, deps=deps)
        codes = {'c1': [1, 0, 1, 0], 'c2': [0, 1, 0, 1]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        assert result.agreement_score > 0.5

    def test_print_report(self, overlay, capsys):
        """Verify print_report produces output."""
        report = _make_report([0.5, 0.5, 0.4, 0.3])
        codes = {'c1': [1, 1, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        rec.print_report(result)
        captured = capsys.readouterr()
        assert "RECONCILIATION" in captured.out or len(captured.out) > 0

    def test_suggestion_generation(self, overlay):
        """Verify anchor and theory corrections are produced."""
        report = _make_report([0.01, 0.5, 0.4, 0.3])
        codes = {'c1': [0, 1, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        assert isinstance(result.suggested_anchor_corrections, dict)
        assert isinstance(result.suggested_theory_corrections, dict)

    def test_semantic_drift_high_activation(self, overlay):
        """Bit with activation_rate > 0.95 -> semantic_drift warning."""
        report = _make_report([0.97, 0.5, 0.4, 0.3])
        codes = {'c1': [1, 1, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        drift = [m for m in result.bit_mismatches
                 if m.mismatch_type == 'semantic_drift']
        assert len(drift) >= 1
        assert drift[0].bit_index == 0
        assert drift[0].severity == 'warning'

    def test_missing_in_theory_duals(self, overlay):
        """Model discovers a dual pair not in theory -> missing_in_theory."""
        # Create duals between bits 4 and 5 (not in primitivos)
        duals = [DiscoveredDual(bit_a=4, bit_b=5, anti_correlation=-0.8,
                                concepts_exclusive=8, concepts_both=0)]
        report = _make_report([0.5, 0.5, 0.4, 0.3, 0.6, 0.6], duals=duals)
        codes = {'c1': [1, 1, 1, 0, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        missing = [m for m in result.dual_mismatches
                   if m.mismatch_type == 'missing_in_theory']
        assert len(missing) >= 1
        bits = {(m.bit_a, m.bit_b) for m in missing}
        assert (4, 5) in bits

    def test_missing_in_theory_deps(self, overlay):
        """Model discovers a strong dep not in theory -> missing_in_theory."""
        deps = [DiscoveredDependency(bit_parent=4, bit_child=5,
                                     confidence=0.98, support=15)]
        report = _make_report([0.5, 0.5, 0.4, 0.3, 0.6, 0.6], deps=deps)
        codes = {'c1': [1, 1, 1, 0, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        missing = [m for m in result.dep_mismatches
                   if m.mismatch_type == 'missing_in_theory']
        assert len(missing) >= 1
        assert missing[0].confidence == 0.98

    def test_anchor_correction_content(self, overlay):
        """Dead bit -> add_anchors_for suggestion with correct primitive name."""
        report = _make_report([0.01, 0.5, 0.4, 0.3])
        codes = {'c1': [0, 1, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        ac = result.suggested_anchor_corrections
        assert len(ac['add_anchors_for']) >= 1
        entry = ac['add_anchors_for'][0]
        assert entry['primitive'] == 'alive'
        assert entry['bit'] == 0

    def test_theory_correction_content(self, overlay):
        """Missing-in-theory dual -> add_duals suggestion."""
        duals = [DiscoveredDual(bit_a=4, bit_b=5, anti_correlation=-0.8,
                                concepts_exclusive=8, concepts_both=0)]
        report = _make_report([0.5, 0.5, 0.4, 0.3, 0.6, 0.6], duals=duals)
        codes = {'c1': [1, 1, 1, 0, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        tc = result.suggested_theory_corrections
        assert len(tc['add_duals']) >= 1
        pair = tc['add_duals'][0]
        assert 'pair' in pair
        assert 'correlation' in pair

    def test_semantic_drift_correction_content(self, overlay):
        """Semantic drift -> modify_bit_targets suggestion."""
        report = _make_report([0.97, 0.5, 0.4, 0.3])
        codes = {'c1': [1, 1, 1, 0]}
        rec = Reconciler(overlay)
        result = rec.reconcile(report, codes)
        ac = result.suggested_anchor_corrections
        assert len(ac['modify_bit_targets']) >= 1
        entry = ac['modify_bit_targets'][0]
        assert entry['primitive'] == 'alive'
        assert 'selective' in entry['reason'].lower() or 'negative' in entry['reason'].lower()
