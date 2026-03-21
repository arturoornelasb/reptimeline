"""
Churn heatmap — per-bit stability across training steps.

Rows are bit indices, columns are steps. Color shows whether each bit
changed from the previous step (aggregated across concepts).
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from reptimeline.core import Timeline


def plot_churn_heatmap(timeline: Timeline,
                       concepts: Optional[List[str]] = None,
                       max_bits: int = 63,
                       figsize: tuple = (14, 8),
                       title: str = "Bit Churn Heatmap",
                       save_path: Optional[str] = None,
                       show: bool = True):
    """Heatmap showing per-bit churn rate across training.

    Each cell (bit, step) shows the fraction of concepts where that bit
    changed from the previous step. Darker = more churn.

    Args:
        timeline: Timeline from tracker.analyze().
        concepts: Subset of concepts (default: all).
        max_bits: Max bits to show.
        figsize: Figure size.
        title: Plot title.
        save_path: Save figure path.
        show: Show figure.
    """
    if concepts is None:
        concepts = timeline.snapshots[-1].concepts if timeline.snapshots else []

    n_steps = len(timeline.steps)
    n_bits = min(max_bits, timeline.snapshots[-1].code_dim if timeline.snapshots else 63)

    # Build churn matrix: bits x steps
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

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(churn, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', vmin=0, vmax=1)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Bit Index", fontsize=11)
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f"{s:,}" for s in timeline.steps],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(0, n_bits, max(1, n_bits // 15)))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Churn Rate (fraction of concepts changed)", fontsize=9)

    # Annotate stability from timeline
    if timeline.stability:
        # Mark the 5 least stable bits
        sorted_stab = sorted(timeline.stability.items(), key=lambda x: x[1])
        for idx, score in sorted_stab[:5]:
            if idx < n_bits:
                ax.annotate(f"bit{idx}", xy=(n_steps - 1, idx),
                            xytext=(n_steps + 0.5, idx),
                            fontsize=7, color='red',
                            arrowprops=dict(arrowstyle='-', color='red', lw=0.5))

    ax.set_title(title, fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
