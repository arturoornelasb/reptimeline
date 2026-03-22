"""
reptimeline — Track how discrete representations evolve during training.

Detects when concepts "are born" (first become distinguishable), when they "die"
(collapse), when relationships form between concept pairs, and where phase
transitions occur.

Works with any discrete bottleneck: triadic bits, VQ-VAE codebooks, FSQ levels,
sparse autoencoders, or binary codes.

Usage:
    from reptimeline import TimelineTracker, ConceptSnapshot

    tracker = TimelineTracker(extractor)
    timeline = tracker.analyze(snapshots)
    timeline.print_summary()
"""

from reptimeline.autolabel import AutoLabeler as AutoLabeler
from reptimeline.causal import CausalReport as CausalReport
from reptimeline.causal import CausalVerifier as CausalVerifier
from reptimeline.core import CodeEvent as CodeEvent
from reptimeline.core import ConceptSnapshot as ConceptSnapshot
from reptimeline.core import ConnectionEvent as ConnectionEvent
from reptimeline.core import PhaseTransition as PhaseTransition
from reptimeline.core import Timeline as Timeline
from reptimeline.discovery import BitDiscovery as BitDiscovery
from reptimeline.exceptions import ConfigurationError as ConfigurationError
from reptimeline.exceptions import DiscoveryError as DiscoveryError
from reptimeline.exceptions import ExtractionError as ExtractionError
from reptimeline.exceptions import ReptimelineError as ReptimelineError
from reptimeline.exceptions import SnapshotError as SnapshotError
from reptimeline.overlays.primitive_overlay import PrimitiveOverlay as PrimitiveOverlay
from reptimeline.reconcile import Reconciler as Reconciler
from reptimeline.tracker import TimelineTracker as TimelineTracker

__version__ = "0.1.0"
