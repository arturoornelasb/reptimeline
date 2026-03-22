from reptimeline.viz.causal_heatmap import plot_causal_heatmap as plot_causal_heatmap
from reptimeline.viz.churn_heatmap import plot_churn_heatmap as plot_churn_heatmap
from reptimeline.viz.layer_emergence import plot_layer_emergence as plot_layer_emergence
from reptimeline.viz.phase_dashboard import plot_phase_dashboard as plot_phase_dashboard
from reptimeline.viz.swimlane import plot_swimlane as plot_swimlane

# Interactive plots (optional, requires plotly)
try:
    from reptimeline.viz.interactive import (
        plot_causal_heatmap_interactive as plot_causal_heatmap_interactive,
    )
    from reptimeline.viz.interactive import (
        plot_churn_heatmap_interactive as plot_churn_heatmap_interactive,
    )
    from reptimeline.viz.interactive import (
        plot_phase_dashboard_interactive as plot_phase_dashboard_interactive,
    )
    from reptimeline.viz.interactive import (
        plot_swimlane_interactive as plot_swimlane_interactive,
    )
except Exception:  # plotly not installed
    pass
