"""
Layer emergence plot — visualizes when each layer's primitives activate.

Specific to the triadic overlay: shows the 6-layer hierarchy and
whether lower layers stabilize before higher layers (as the theory predicts).
"""

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from reptimeline.overlays.primitive_overlay import PrimitiveReport

LAYER_COLORS = {
    1: '#1565C0',  # deep blue
    2: '#2E7D32',  # green
    3: '#F57F17',  # amber
    4: '#E65100',  # deep orange
    5: '#AD1457',  # pink
    6: '#6A1B9A',  # purple
}


def plot_layer_emergence(report: PrimitiveReport,
                         figsize: tuple = (12, 6),
                         title: str = "Layer Emergence Order",
                         save_path: Optional[str] = None,
                         show: bool = True):
    """Bar chart showing when each layer's primitives first activate.

    Args:
        report: PrimitiveReport from PrimitiveOverlay.analyze().
        figsize: Figure size.
        title: Plot title.
        save_path: Save path.
        show: Show figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    layers = report.layer_emergence
    y_positions = list(range(len(layers)))

    for i, le in enumerate(layers):
        color = LAYER_COLORS.get(le.layer, '#757575')
        if le.first_activation_step is not None:
            # Draw range bar from first to last
            width = (le.last_activation_step - le.first_activation_step) or 100
            ax.barh(i, width, left=le.first_activation_step, height=0.6,
                    color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

            # Mark median
            if le.median_activation_step is not None:
                ax.plot(le.median_activation_step, i, 'D', color='white',
                        markersize=6, markeredgecolor='black', markeredgewidth=0.5)

            # Label
            ax.text(le.last_activation_step + 50, i,
                    f"{le.primitives_activated}/{le.n_primitives}",
                    va='center', fontsize=9, color=color)
        else:
            ax.text(50, i, f"0/{le.n_primitives} activated",
                    va='center', fontsize=9, color='gray', style='italic')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"L{le.layer} {le.layer_name}" for le in layers], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, axis='x', alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(color=LAYER_COLORS[i], label=f'Layer {i}')
        for i in sorted(LAYER_COLORS.keys())
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
