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

from reptimeline.core import (
    ConceptSnapshot, CodeEvent, ConnectionEvent, PhaseTransition, Timeline,
)
from reptimeline.tracker import TimelineTracker
from reptimeline.discovery import BitDiscovery
from reptimeline.autolabel import AutoLabeler
from reptimeline.reconcile import Reconciler
from reptimeline.overlays.primitive_overlay import PrimitiveOverlay
from reptimeline.causal import CausalVerifier, CausalReport

__version__ = "0.1.0"
