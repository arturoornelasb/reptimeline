"""Tests for interactive Plotly visualizations."""

import pytest

from reptimeline.core import (
    CodeEvent,
    ConceptSnapshot,
    PhaseTransition,
    Timeline,
)

plotly = pytest.importorskip("plotly")


@pytest.fixture
def sample_timeline():
    snap0 = ConceptSnapshot(step=0, codes={'a': [0, 1, 0], 'b': [1, 0, 0]})
    snap1 = ConceptSnapshot(step=100, codes={'a': [1, 1, 0], 'b': [1, 0, 1]})
    return Timeline(
        steps=[0, 100],
        snapshots=[snap0, snap1],
        births=[CodeEvent('birth', 0, 'a', 1)],
        deaths=[],
        connections=[],
        phase_transitions=[
            PhaseTransition(step=100, metric='entropy', delta=0.5, direction='increase'),
        ],
        curves={'entropy': [0.3, 0.8], 'churn_rate': [0.0, 0.5],
                'utilization': [1.0, 1.0]},
        stability={0: 0.5, 1: 1.0, 2: 0.75},
    )


class TestPhaseDashboardInteractive:

    def test_returns_figure(self, sample_timeline):
        from reptimeline.viz.interactive import plot_phase_dashboard_interactive
        fig = plot_phase_dashboard_interactive(sample_timeline)
        assert fig is not None
        assert hasattr(fig, 'data')

    def test_save_html(self, sample_timeline, tmp_path):
        from reptimeline.viz.interactive import plot_phase_dashboard_interactive
        path = str(tmp_path / "dashboard.html")
        plot_phase_dashboard_interactive(sample_timeline, save_html=path)
        assert (tmp_path / "dashboard.html").exists()


class TestSwimlaneInteractive:

    def test_returns_figure(self, sample_timeline):
        from reptimeline.viz.interactive import plot_swimlane_interactive
        fig = plot_swimlane_interactive(sample_timeline, concepts=['a', 'b'])
        assert len(fig.data) == 2  # one heatmap per concept


class TestChurnHeatmapInteractive:

    def test_returns_figure(self, sample_timeline):
        from reptimeline.viz.interactive import plot_churn_heatmap_interactive
        fig = plot_churn_heatmap_interactive(sample_timeline)
        assert len(fig.data) == 1  # one heatmap
