"""
Phase dashboard — multi-panel plot of entropy, churn, utilization + phase transitions.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from reptimeline.core import Timeline


def plot_phase_dashboard(timeline: Timeline,
                         figsize: tuple = (14, 8),
                         title: str = "Training Phase Dashboard",
                         save_path: Optional[str] = None,
                         show: bool = True):
    """Plot a 3-panel dashboard of training metrics with phase transitions marked.

    Panels:
        1. Entropy curve
        2. Churn rate curve
        3. Utilization curve

    Vertical lines mark detected phase transitions.
    """
    steps = timeline.steps
    curves = timeline.curves

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    metrics = [
        ('entropy', 'Mean Bit Entropy', '#2196F3'),
        ('churn_rate', 'Code Churn Rate', '#FF5722'),
        ('utilization', 'Code Utilization', '#4CAF50'),
    ]

    for ax, (key, label, color) in zip(axes, metrics):
        if key not in curves:
            continue
        vals = curves[key]
        ax.plot(steps, vals, color=color, linewidth=2, marker='o', markersize=4)
        ax.fill_between(steps, vals, alpha=0.15, color=color)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    # Mark phase transitions
    for pt in timeline.phase_transitions:
        for ax in axes:
            ax.axvline(pt.step, color='red', linestyle='--', alpha=0.6, linewidth=1)

    # Annotate transitions on the relevant panel
    metric_to_ax = {m[0]: ax for ax, m in zip(axes, metrics)}
    for pt in timeline.phase_transitions:
        ax = metric_to_ax.get(pt.metric)
        if ax is None:
            continue
        y_range = ax.get_ylim()
        y_pos = y_range[1] * 0.9
        ax.annotate(
            f"{pt.direction}\n\u0394={pt.delta:.3f}",
            xy=(pt.step, y_pos),
            fontsize=7, ha='center', color='red',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8),
        )

    axes[-1].set_xlabel("Training Step", fontsize=11)
    axes[-1].set_xticks(steps)
    axes[-1].set_xticklabels([f"{s:,}" for s in steps], rotation=45, ha='right')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
