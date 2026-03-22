"""
Interactive plots using Plotly.

Optional: requires `pip install plotly`. Falls back gracefully if not installed.

Provides interactive versions of the core visualizations:
  - plot_phase_dashboard_interactive: zoomable metric curves + phase transitions
  - plot_swimlane_interactive: per-concept bit activation heatmap
  - plot_causal_heatmap_interactive: selectivity bar chart with hover detail
  - plot_churn_heatmap_interactive: per-bit churn across training
"""

from typing import List, Optional

from reptimeline.exceptions import ConfigurationError

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def _require_plotly():
    if not HAS_PLOTLY:
        raise ConfigurationError(
            "plotly is required for interactive plots. "
            "Install it with: pip install plotly"
        )


def plot_phase_dashboard_interactive(timeline, title="Training Phase Dashboard",
                                     save_html: Optional[str] = None):
    """Interactive 3-panel dashboard of entropy, churn, utilization.

    Args:
        timeline: Timeline from tracker.analyze().
        title: Plot title.
        save_html: If set, saves to this HTML file path.

    Returns:
        plotly Figure.
    """
    _require_plotly()

    steps = timeline.steps
    curves = timeline.curves

    metrics = [
        ('entropy', 'Mean Bit Entropy', '#2196F3'),
        ('churn_rate', 'Code Churn Rate', '#FF5722'),
        ('utilization', 'Code Utilization', '#4CAF50'),
    ]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=[m[1] for m in metrics],
                        vertical_spacing=0.08)

    for i, (key, label, color) in enumerate(metrics, 1):
        if key not in curves:
            continue
        vals = curves[key]
        fig.add_trace(
            go.Scatter(x=steps, y=vals, mode='lines+markers', name=label,
                       line=dict(color=color, width=2),
                       marker=dict(size=5),
                       hovertemplate=f'{label}: %{{y:.4f}}<br>Step: %{{x:,}}<extra></extra>'),
            row=i, col=1,
        )
        fig.update_yaxes(title_text=label, row=i, col=1)

    # Phase transitions as vertical lines
    for pt in timeline.phase_transitions:
        for i in range(1, 4):
            fig.add_vline(x=pt.step, line_dash="dash", line_color="red",
                          opacity=0.6, row=i, col=1)
        # Annotate on the relevant panel
        row_map = {'entropy': 1, 'churn_rate': 2, 'utilization': 3}
        row = row_map.get(pt.metric, 1)
        # Plotly axis refs: first axis is 'x'/'y', subsequent are 'x2','y2', etc.
        x_ref = 'x' if row == 1 else f'x{row}'
        y_ref = 'y domain' if row == 1 else f'y{row} domain'
        fig.add_annotation(
            x=pt.step, y=0.9, xref=x_ref, yref=y_ref,
            text=f"{pt.direction} Δ={pt.delta:.3f}",
            showarrow=False, font=dict(size=10, color='red'),
            bgcolor='lightyellow', bordercolor='red', borderwidth=1,
        )

    fig.update_xaxes(title_text="Training Step", row=3, col=1)
    fig.update_layout(title=title, height=700, showlegend=False,
                      template='plotly_white')

    if save_html:
        fig.write_html(save_html)
    return fig


def plot_swimlane_interactive(timeline, concepts: Optional[List[str]] = None,
                              max_bits: int = 63,
                              title: str = "Representation Swimlane",
                              save_html: Optional[str] = None):
    """Interactive swimlane of bit activations per concept.

    Args:
        timeline: Timeline from tracker.analyze().
        concepts: Subset of concepts (default: all in last snapshot).
        max_bits: Max bits to show per concept.
        title: Plot title.
        save_html: Save to HTML path.

    Returns:
        plotly Figure.
    """
    _require_plotly()
    import numpy as np

    if concepts is None:
        concepts = timeline.snapshots[-1].concepts if timeline.snapshots else []
    if not concepts:
        return go.Figure()

    n_steps = len(timeline.steps)
    n_bits = min(max_bits, timeline.snapshots[-1].code_dim if timeline.snapshots else 63)

    fig = make_subplots(rows=len(concepts), cols=1, shared_xaxes=True,
                        subplot_titles=concepts, vertical_spacing=0.02)

    for row, concept in enumerate(concepts, 1):
        matrix = np.zeros((n_bits, n_steps))
        for t, snap in enumerate(timeline.snapshots):
            code = snap.codes.get(concept)
            if code is not None:
                for b in range(min(n_bits, len(code))):
                    matrix[b, t] = code[b]

        step_labels = [f"{s:,}" for s in timeline.steps]
        fig.add_trace(
            go.Heatmap(
                z=matrix, x=step_labels,
                y=[f"bit {b}" for b in range(n_bits)],
                colorscale=[[0, 'white'], [1, '#3366CC']],
                showscale=(row == 1),
                hovertemplate=(
                    f'{concept}<br>'
                    'Step: %{x}<br>%{y}<br>'
                    'Value: %{z}<extra></extra>'
                ),
            ),
            row=row, col=1,
        )

    fig.update_layout(title=title, height=max(300, len(concepts) * 150),
                      template='plotly_white')
    if save_html:
        fig.write_html(save_html)
    return fig


def plot_causal_heatmap_interactive(report, title="Causal Selectivity by Bit",
                                    save_html: Optional[str] = None):
    """Interactive horizontal bar chart of causal selectivity.

    Args:
        report: CausalReport from CausalVerifier.verify().
        title: Plot title.
        save_html: Save to HTML path.

    Returns:
        plotly Figure.
    """
    _require_plotly()

    if not report.bit_results:
        fig = go.Figure()
        fig.add_annotation(text="No bit results to display", x=0.5, y=0.5,
                           xref='paper', yref='paper', showarrow=False, font_size=16)
        return fig

    results = sorted(report.bit_results, key=lambda r: r.selectivity, reverse=True)
    max_display = 20.0

    labels = [f"bit {r.bit_index}" for r in results]
    selectivities = [min(r.selectivity, max_display) for r in results]
    colors = ['#2196F3' if r.significant else '#BDBDBD' for r in results]

    hover_texts = []
    for r in results:
        parts = [f"Selectivity: {r.selectivity:.2f}x"]
        if r.bootstrap is not None:
            parts.append(f"CI: [{r.bootstrap.ci_low:.2f}, {r.bootstrap.ci_high:.2f}]")
        if r.permutation is not None:
            parts.append(f"p-value: {r.permutation.p_value:.4f}")
        parts.append(f"Significant: {'Yes' if r.significant else 'No'}")
        hover_texts.append('<br>'.join(parts))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=selectivities, orientation='h',
        marker_color=colors, hovertext=hover_texts,
        hoverinfo='text+name', name='Selectivity',
    ))

    sel_threshold = report.metadata.get('selectivity_threshold', 1.5)
    fig.add_vline(x=sel_threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold ({sel_threshold}x)")

    fig.update_layout(
        title=title, xaxis_title="Selectivity Ratio",
        yaxis=dict(autorange='reversed'),
        height=max(400, len(results) * 25),
        template='plotly_white',
        annotations=[dict(
            x=0.98, y=0.02, xref='paper', yref='paper',
            text=f"Verdict: {report.verdict.replace('_', ' ').upper()}<br>"
                 f"{report.n_significant}/{report.n_tested} significant",
            showarrow=False, font=dict(size=11),
            bgcolor='lightyellow', bordercolor='gray', borderwidth=1,
        )],
    )

    if save_html:
        fig.write_html(save_html)
    return fig


def plot_churn_heatmap_interactive(timeline, concepts: Optional[List[str]] = None,
                                   max_bits: int = 63,
                                   title: str = "Bit Churn Heatmap",
                                   save_html: Optional[str] = None):
    """Interactive heatmap of per-bit churn rate across training.

    Args:
        timeline: Timeline from tracker.analyze().
        concepts: Subset of concepts (default: all).
        max_bits: Max bits to show.
        title: Plot title.
        save_html: Save to HTML path.

    Returns:
        plotly Figure.
    """
    _require_plotly()
    import numpy as np

    if concepts is None:
        concepts = timeline.snapshots[-1].concepts if timeline.snapshots else []

    n_steps = len(timeline.steps)
    n_bits = min(max_bits, timeline.snapshots[-1].code_dim if timeline.snapshots else 63)

    churn = np.zeros((n_bits, n_steps))
    for t in range(1, n_steps):
        for bit_idx in range(n_bits):
            changed = 0
            total = 0
            for concept in concepts:
                prev = timeline.snapshots[t - 1].codes.get(concept)
                curr = timeline.snapshots[t].codes.get(concept)
                if prev is None or curr is None:
                    continue
                if bit_idx < len(prev) and bit_idx < len(curr):
                    total += 1
                    if prev[bit_idx] != curr[bit_idx]:
                        changed += 1
            churn[bit_idx, t] = changed / max(total, 1)

    step_labels = [f"{s:,}" for s in timeline.steps]

    fig = go.Figure(data=go.Heatmap(
        z=churn, x=step_labels,
        y=[f"bit {i}" for i in range(n_bits)],
        colorscale='YlOrRd', zmin=0, zmax=1,
        colorbar=dict(title='Churn Rate'),
        hovertemplate='Step: %{x}<br>%{y}<br>Churn: %{z:.3f}<extra></extra>',
    ))

    fig.update_layout(title=title, xaxis_title="Training Step",
                      yaxis_title="Bit Index",
                      height=max(400, n_bits * 12),
                      template='plotly_white')

    if save_html:
        fig.write_html(save_html)
    return fig
