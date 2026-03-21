"""
Causal heatmap — visualize selectivity per bit from CausalVerifier results.

Horizontal bar chart showing selectivity ratio per bit, with BH-significant
bits highlighted and confidence intervals as error bars.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from reptimeline.causal import CausalReport


def plot_causal_heatmap(report: CausalReport,
                        figsize: tuple = (12, 8),
                        title: str = "Causal Selectivity by Bit",
                        threshold_line: bool = True,
                        save_path: Optional[str] = None,
                        show: bool = True):
    """Horizontal bar chart of causal selectivity per bit.

    Significant bits (post BH-FDR) are colored differently. Bootstrap
    confidence intervals shown as error bars when available.

    Args:
        report: CausalReport from CausalVerifier.verify().
        figsize: Figure size.
        title: Plot title.
        threshold_line: Draw vertical line at selectivity threshold.
        save_path: Save figure path.
        show: Show figure.

    Returns:
        matplotlib Figure.
    """
    if not report.bit_results:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No bit results to display",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    # Sort by selectivity descending
    results = sorted(report.bit_results, key=lambda r: r.selectivity, reverse=True)

    # Cap extreme values for display
    max_display = 20.0
    selectivities = [min(r.selectivity, max_display) for r in results]
    labels = [f"bit {r.bit_index}" for r in results]
    colors = ['#2196F3' if r.significant else '#BDBDBD' for r in results]

    y_pos = np.arange(len(results))

    fig, ax = plt.subplots(figsize=figsize)

    # Error bars from bootstrap CI
    xerr_low = []
    xerr_high = []
    has_ci = any(r.bootstrap is not None for r in results)

    if has_ci:
        for r in results:
            if r.bootstrap is not None:
                ci_lo = max(0, min(r.bootstrap.ci_low, max_display))
                ci_hi = min(r.bootstrap.ci_high, max_display)
                sel = min(r.selectivity, max_display)
                xerr_low.append(max(0, sel - ci_lo))
                xerr_high.append(max(0, ci_hi - sel))
            else:
                xerr_low.append(0)
                xerr_high.append(0)
        ax.barh(y_pos, selectivities, color=colors, edgecolor='white',
                xerr=[xerr_low, xerr_high], error_kw={'elinewidth': 1, 'capsize': 3})
    else:
        ax.barh(y_pos, selectivities, color=colors, edgecolor='white')

    # Threshold line
    sel_threshold = report.metadata.get('selectivity_threshold', 1.5)
    if threshold_line:
        ax.axvline(x=sel_threshold, color='#E53935', linestyle='--',
                   linewidth=1.5, label=f'Threshold ({sel_threshold}x)')

    # Significance markers
    for i, r in enumerate(results):
        if r.significant and r.selectivity >= sel_threshold:
            ax.text(min(r.selectivity, max_display) + 0.1, i, '*',
                    va='center', fontsize=14, fontweight='bold', color='#E53935')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Selectivity Ratio (labeled / other)", fontsize=11)
    ax.set_title(title, fontsize=13)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label=f'Significant (BH, alpha={report.alpha})'),
        Patch(facecolor='#BDBDBD', label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Verdict annotation
    ax.text(0.98, 0.02,
            f"Verdict: {report.verdict.replace('_', ' ').upper()}\n"
            f"{report.n_significant}/{report.n_tested} significant",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig
