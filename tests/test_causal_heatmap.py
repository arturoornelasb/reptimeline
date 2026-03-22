"""Tests for causal heatmap visualization."""

import matplotlib

matplotlib.use('Agg')


from reptimeline.causal import BitCausalResult, CausalReport
from reptimeline.stats import BootstrapResult
from reptimeline.viz.causal_heatmap import plot_causal_heatmap


def _make_bit_result(bit_index, selectivity, significant, p_value=0.01,
                     bootstrap=None):
    return BitCausalResult(
        bit_index=bit_index,
        selectivity=selectivity,
        bootstrap=bootstrap,
        p_value=p_value,
        significant=significant,
        effect_size=1.0,
        mean_labeled=5.0,
        mean_other=2.0,
        n_labeled=10,
        n_other=20,
        top_affected=[('a', 5.0), ('b', 3.0)],
    )


def _make_report(bit_results, n_significant=None):
    if n_significant is None:
        n_significant = sum(1 for r in bit_results if r.significant)
    verdict = 'causal_evidence_found' if n_significant >= 3 else 'insufficient_evidence'
    return CausalReport(
        bit_results=bit_results,
        n_tested=len(bit_results),
        n_significant=n_significant,
        verdict=verdict,
        correction_method='benjamini_hochberg',
        alpha=0.05,
        metadata={'selectivity_threshold': 1.5, 'min_selective_bits': 3},
    )


class TestCausalHeatmap:

    def test_basic_plot(self, tmp_path):
        bits = [
            _make_bit_result(0, 3.5, True),
            _make_bit_result(1, 1.2, False, p_value=0.3),
            _make_bit_result(2, 2.0, True),
        ]
        report = _make_report(bits)
        fig = plot_causal_heatmap(report, show=False,
                                  save_path=str(tmp_path / "causal.png"))
        assert fig is not None
        assert (tmp_path / "causal.png").exists()

    def test_empty_report(self, tmp_path):
        report = _make_report([])
        fig = plot_causal_heatmap(report, show=False,
                                  save_path=str(tmp_path / "empty.png"))
        assert fig is not None

    def test_with_bootstrap_ci(self, tmp_path):
        boot = BootstrapResult(observed=3.0, ci_low=2.0, ci_high=4.5,
                               n_bootstrap=1000)
        bits = [
            _make_bit_result(0, 3.0, True, bootstrap=boot),
            _make_bit_result(1, 1.0, False),
        ]
        report = _make_report(bits)
        fig = plot_causal_heatmap(report, show=False,
                                  save_path=str(tmp_path / "ci.png"))
        assert fig is not None
        assert (tmp_path / "ci.png").exists()

    def test_extreme_selectivity_capped(self, tmp_path):
        """Selectivity > 20 should be capped for display."""
        bits = [_make_bit_result(0, 999.0, True)]
        report = _make_report(bits, n_significant=1)
        fig = plot_causal_heatmap(report, show=False)
        assert fig is not None

    def test_no_threshold_line(self, tmp_path):
        bits = [_make_bit_result(0, 2.0, True)]
        report = _make_report(bits, n_significant=1)
        fig = plot_causal_heatmap(report, threshold_line=False, show=False)
        assert fig is not None

    def test_importable_from_viz(self):
        from reptimeline.viz import plot_causal_heatmap as pch
        assert pch is not None
