"""Tests for visualization modules (swimlane, phase_dashboard, churn_heatmap, layer_emergence)."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest

from reptimeline.core import ConceptSnapshot, Timeline, PhaseTransition
from reptimeline.tracker import TimelineTracker
from reptimeline.viz import plot_swimlane, plot_phase_dashboard, plot_churn_heatmap, plot_layer_emergence
from reptimeline.overlays.primitive_overlay import PrimitiveReport, LayerEmergence, DualCoherence
from reptimeline.extractors.base import RepresentationExtractor


def make_code(n_bits, active_indices):
    code = [0] * n_bits
    for i in active_indices:
        code[i] = 1
    return code


# ── fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def timeline(basic_snapshots, synthetic_extractor):
    """Build a real Timeline from the shared basic_snapshots fixture."""
    tracker = TimelineTracker(synthetic_extractor)
    return tracker.analyze(basic_snapshots)


@pytest.fixture
def single_step_timeline(synthetic_extractor):
    """Timeline with only one snapshot."""
    snaps = [ConceptSnapshot(step=0, codes={'a': make_code(8, [0, 1])})]
    tracker = TimelineTracker(synthetic_extractor)
    return tracker.analyze(snaps)


@pytest.fixture
def synthetic_report():
    """A PrimitiveReport with synthetic LayerEmergence data."""
    return PrimitiveReport(
        activations=[],
        deps_completions=[],
        layer_emergence=[
            LayerEmergence(layer=1, layer_name="Primes", n_primitives=3,
                           first_activation_step=0, median_activation_step=500,
                           last_activation_step=1000, primitives_activated=3),
            LayerEmergence(layer=2, layer_name="Composites", n_primitives=2,
                           first_activation_step=1000, median_activation_step=1500,
                           last_activation_step=2000, primitives_activated=2),
            LayerEmergence(layer=3, layer_name="Triadic", n_primitives=1,
                           first_activation_step=None, median_activation_step=None,
                           last_activation_step=None, primitives_activated=0),
        ],
        dual_coherence=[],
    )


# ── plot_swimlane ─────────────────────────────────────────────────

class TestPlotSwimlane:

    def test_smoke(self, timeline):
        fig = plot_swimlane(timeline, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_concept_subset(self, timeline):
        fig = plot_swimlane(timeline, concepts=['king', 'fire'], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_concepts_returns_none(self, timeline):
        result = plot_swimlane(timeline, concepts=[], show=False)
        assert result is None

    def test_save_path(self, timeline, tmp_path):
        path = str(tmp_path / "swimlane.png")
        fig = plot_swimlane(timeline, save_path=path, show=False)
        assert (tmp_path / "swimlane.png").exists()
        plt.close(fig)

    def test_single_step(self, single_step_timeline):
        fig = plot_swimlane(single_step_timeline, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ── plot_phase_dashboard ──────────────────────────────────────────

class TestPlotPhaseDashboard:

    def test_smoke(self, timeline):
        fig = plot_phase_dashboard(timeline, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, timeline, tmp_path):
        path = str(tmp_path / "dashboard.png")
        fig = plot_phase_dashboard(timeline, save_path=path, show=False)
        assert (tmp_path / "dashboard.png").exists()
        plt.close(fig)

    def test_single_step(self, single_step_timeline):
        fig = plot_phase_dashboard(single_step_timeline, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self, timeline):
        fig = plot_phase_dashboard(timeline, title="Custom Title", show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_transitions(self, synthetic_extractor):
        """Phase dashboard with no phase transitions should still render."""
        # Two identical snapshots -> no transitions detected
        snaps = [
            ConceptSnapshot(step=0, codes={'a': make_code(8, [0, 1]), 'b': make_code(8, [2])}),
            ConceptSnapshot(step=1000, codes={'a': make_code(8, [0, 1]), 'b': make_code(8, [2])}),
        ]
        tracker = TimelineTracker(synthetic_extractor)
        tl = tracker.analyze(snaps)
        assert len(tl.phase_transitions) == 0
        fig = plot_phase_dashboard(tl, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_phase_transitions(self, timeline):
        """Phase dashboard with injected transitions covers annotation code."""
        from reptimeline.core import Timeline
        # Inject a phase transition into the existing timeline
        tl = Timeline(
            steps=timeline.steps,
            snapshots=timeline.snapshots,
            births=timeline.births,
            deaths=timeline.deaths,
            connections=timeline.connections,
            phase_transitions=[
                PhaseTransition(step=1000, metric='churn_rate',
                                delta=0.5, direction='spike'),
                PhaseTransition(step=2000, metric='entropy',
                                delta=-0.3, direction='drop'),
            ],
            curves=timeline.curves,
            stability=timeline.stability,
        )
        fig = plot_phase_dashboard(tl, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ── plot_churn_heatmap ────────────────────────────────────────────

class TestPlotChurnHeatmap:

    def test_smoke(self, timeline):
        fig = plot_churn_heatmap(timeline, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_concept_subset(self, timeline):
        fig = plot_churn_heatmap(timeline, concepts=['queen'], show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, timeline, tmp_path):
        path = str(tmp_path / "churn.png")
        fig = plot_churn_heatmap(timeline, save_path=path, show=False)
        assert (tmp_path / "churn.png").exists()
        plt.close(fig)

    def test_single_step(self, single_step_timeline):
        fig = plot_churn_heatmap(single_step_timeline, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ── plot_layer_emergence ──────────────────────────────────────────

class TestPlotLayerEmergence:

    def test_smoke(self, synthetic_report):
        fig = plot_layer_emergence(synthetic_report, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_path(self, synthetic_report, tmp_path):
        path = str(tmp_path / "emergence.png")
        fig = plot_layer_emergence(synthetic_report, save_path=path, show=False)
        assert (tmp_path / "emergence.png").exists()
        plt.close(fig)

    def test_with_unactivated_layers(self, synthetic_report):
        """Layer 3 has no activations -- should render without error."""
        fig = plot_layer_emergence(synthetic_report, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_report(self):
        report = PrimitiveReport(
            activations=[], deps_completions=[],
            layer_emergence=[], dual_coherence=[],
        )
        fig = plot_layer_emergence(report, show=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
