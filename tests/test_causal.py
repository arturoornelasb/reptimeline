"""Tests for CausalVerifier with mock intervention functions."""

import numpy as np
import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.causal import CausalVerifier, CausalReport, BitCausalResult


def make_code(n_bits, active_indices):
    code = [0] * n_bits
    for i in active_indices:
        code[i] = 1
    return code


@pytest.fixture
def selective_snapshot():
    """Snapshot where bit 0 = group A (fire/water/earth), bit 1 = group B (love/hate),
    bit 2 = non-selective (active for everything)."""
    n = 6
    return ConceptSnapshot(step=5000, codes={
        'fire':  make_code(n, [0, 2]),
        'water': make_code(n, [0, 2]),
        'earth': make_code(n, [0, 2]),
        'love':  make_code(n, [1, 2]),
        'hate':  make_code(n, [1, 2]),
        'peace': make_code(n, []),     # no bits active
        'void':  make_code(n, []),     # no bits active
        'null':  make_code(n, []),     # no bits active
    })


@pytest.fixture
def selective_intervene_fn():
    """Mock intervene_fn: bit 0 is selective for fire/water/earth,
    bit 1 selective for love/hate, bit 2 non-selective."""
    rng = np.random.RandomState(42)
    concepts = ['fire', 'water', 'earth', 'love', 'hate', 'peace', 'void', 'null']
    noise = {c: rng.randn() * 0.01 for c in concepts}

    def fn(concept, bit_index):
        base = noise[concept]
        if bit_index == 0:
            if concept in ('fire', 'water', 'earth'):
                return 5.0 + base
            return 0.1 + base
        elif bit_index == 1:
            if concept in ('love', 'hate'):
                return 4.0 + base
            return 0.2 + base
        else:
            return 1.0 + base  # equal for all → non-selective
    return fn


class TestCausalVerifier:

    def test_selective_bit_found(self, selective_snapshot, selective_intervene_fn):
        verifier = CausalVerifier(
            intervene_fn=selective_intervene_fn,
            n_bootstrap=200, n_perms=200, seed=42,
            min_selective_bits=1,
        )
        report = verifier.verify(selective_snapshot, bit_indices=[0, 1, 2])
        bit_0 = next(br for br in report.bit_results if br.bit_index == 0)
        assert bit_0.selectivity > 5.0
        assert bit_0.mean_labeled > bit_0.mean_other

    def test_non_selective_bit_low_selectivity(self, selective_snapshot, selective_intervene_fn):
        verifier = CausalVerifier(
            intervene_fn=selective_intervene_fn,
            n_bootstrap=200, n_perms=200, seed=42,
        )
        report = verifier.verify(selective_snapshot, bit_indices=[2])
        bit_2 = report.bit_results[0]
        assert bit_2.selectivity < 2.0

    def test_verdict_causal_evidence(self, selective_snapshot, selective_intervene_fn):
        verifier = CausalVerifier(
            intervene_fn=selective_intervene_fn,
            n_bootstrap=200, n_perms=200, seed=42,
            alpha=0.20, min_selective_bits=1, selectivity_threshold=1.5,
        )
        report = verifier.verify(selective_snapshot, bit_indices=[0, 1])
        assert report.n_significant >= 1
        assert report.verdict == 'causal_evidence_found'

    def test_verdict_insufficient(self, selective_snapshot):
        """Non-selective fn → insufficient evidence."""
        def flat_fn(concept, bit_index):
            return 1.0
        verifier = CausalVerifier(
            intervene_fn=flat_fn,
            n_bootstrap=100, n_perms=100, seed=42,
            min_selective_bits=3,
        )
        report = verifier.verify(selective_snapshot, bit_indices=[0, 1, 2])
        assert report.verdict == 'insufficient_evidence'

    def test_auto_selects_non_trivial_bits(self, selective_snapshot, selective_intervene_fn):
        """When bit_indices=None, auto-selects bits with rate in (0.03, 0.97)."""
        verifier = CausalVerifier(
            intervene_fn=selective_intervene_fn,
            n_bootstrap=100, n_perms=100, seed=42,
        )
        report = verifier.verify(selective_snapshot)
        tested_bits = {br.bit_index for br in report.bit_results}
        # bits 3,4,5 have rate=0.0, should be excluded
        assert 3 not in tested_bits
        # bits 0,1 should be included (rate between 0.03 and 0.97)
        assert 0 in tested_bits
        assert 1 in tested_bits

    def test_specific_bit_indices(self, selective_snapshot, selective_intervene_fn):
        verifier = CausalVerifier(
            intervene_fn=selective_intervene_fn,
            n_bootstrap=100, n_perms=100, seed=42,
        )
        report = verifier.verify(selective_snapshot, bit_indices=[0])
        assert report.n_tested == 1
        assert report.bit_results[0].bit_index == 0

    def test_top_affected_recorded(self, selective_snapshot, selective_intervene_fn):
        verifier = CausalVerifier(
            intervene_fn=selective_intervene_fn,
            n_bootstrap=100, n_perms=100, seed=42, top_k=3,
        )
        report = verifier.verify(selective_snapshot, bit_indices=[0])
        br = report.bit_results[0]
        assert len(br.top_affected) <= 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in br.top_affected)

    def test_print_report_runs(self, selective_snapshot, selective_intervene_fn, capsys):
        verifier = CausalVerifier(
            intervene_fn=selective_intervene_fn,
            n_bootstrap=100, n_perms=100, seed=42, min_selective_bits=1,
        )
        report = verifier.verify(selective_snapshot, bit_indices=[0, 1])
        verifier.print_report(report)
        captured = capsys.readouterr()
        assert "CAUSAL VERIFICATION" in captured.out

    def test_bootstrap_ci_present(self, selective_snapshot, selective_intervene_fn):
        verifier = CausalVerifier(
            intervene_fn=selective_intervene_fn,
            n_bootstrap=100, n_perms=100, seed=42,
        )
        report = verifier.verify(selective_snapshot, bit_indices=[0])
        br = report.bit_results[0]
        assert br.bootstrap is not None
        assert br.bootstrap.ci_low <= br.bootstrap.observed <= br.bootstrap.ci_high


class TestCausalReportDataclass:

    def test_fields(self):
        report = CausalReport(
            bit_results=[], n_tested=0, n_significant=0,
            verdict='insufficient_evidence',
            correction_method='benjamini_hochberg', alpha=0.05,
        )
        assert report.verdict == 'insufficient_evidence'
        assert report.correction_method == 'benjamini_hochberg'
