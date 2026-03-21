"""
Swimlane diagram — shows when each concept's bits activate/deactivate over training.

One row per concept, columns are training steps. Bits are color-coded by state.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reptimeline.core import Timeline


def plot_swimlane(timeline: Timeline,
                  concepts: Optional[List[str]] = None,
                  max_bits: int = 63,
                  figsize: Optional[tuple] = None,
                  title: str = "Representation Swimlane",
                  save_path: Optional[str] = None,
                  show: bool = True):
    """Plot a swimlane of bit activations per concept across training steps.

    Each row is a concept. Each column is a training step.
    Cells are colored: active=blue, inactive=white, changed=orange.

    Args:
        timeline: Timeline from tracker.analyze().
        concepts: Subset of concepts (default: all in last snapshot).
        max_bits: Max bits to show per concept (default: 63).
        figsize: Figure size tuple.
        title: Plot title.
        save_path: If set, saves figure to this path.
        show: If True, calls plt.show().
    """
    if concepts is None:
        concepts = timeline.snapshots[-1].concepts if timeline.snapshots else []
    if not concepts:
        return

    n_concepts = len(concepts)
    n_steps = len(timeline.steps)
    n_bits = min(max_bits, timeline.snapshots[-1].code_dim if timeline.snapshots else 63)

    if figsize is None:
        figsize = (max(12, n_steps * 1.5), max(4, n_concepts * 0.5))

    fig, axes = plt.subplots(n_concepts, 1, figsize=figsize,
                             sharex=True, squeeze=False)

    for row, concept in enumerate(concepts):
        ax = axes[row, 0]

        # Build matrix: steps x bits
        matrix = np.zeros((n_steps, n_bits))
        for t, snap in enumerate(timeline.snapshots):
            code = snap.codes.get(concept)
            if code is not None:
                for b in range(min(n_bits, len(code))):
                    matrix[t, b] = code[b]

        # Build change mask
        changes = np.zeros_like(matrix)
        for t in range(1, n_steps):
            changes[t] = (matrix[t] != matrix[t - 1]).astype(float)

        # Color: 0=white, 1=blue, changed=orange
        rgb = np.ones((n_steps, n_bits, 3))
        rgb[matrix == 1] = [0.2, 0.4, 0.8]  # blue for active
        for t in range(1, n_steps):
            for b in range(n_bits):
                if changes[t, b] == 1:
                    rgb[t, b] = [1.0, 0.6, 0.1]  # orange for changed

        ax.imshow(rgb.transpose(1, 0, 2), aspect='auto',
                  interpolation='nearest')
        ax.set_ylabel(concept, fontsize=8, rotation=0, ha='right', va='center')
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=7)

    axes[-1, 0].set_xticks(range(n_steps))
    axes[-1, 0].set_xticklabels([f"{s:,}" for s in timeline.steps],
                                 rotation=45, ha='right', fontsize=7)
    axes[-1, 0].set_xlabel("Training Step")

    # Legend
    patches = [
        mpatches.Patch(color=[0.2, 0.4, 0.8], label='Active'),
        mpatches.Patch(color=[1.0, 0.6, 0.1], label='Changed'),
        mpatches.Patch(facecolor='white', edgecolor='gray', label='Inactive'),
    ]
    fig.legend(handles=patches, loc='upper right', fontsize=8)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
