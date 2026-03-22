"""
Abstract base class for representation extractors.

Each backend (triadic, VQ-VAE, FSQ, sparse autoencoder) implements this
interface to produce standardized ConceptSnapshot objects.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from reptimeline.core import ConceptSnapshot

logger = logging.getLogger(__name__)


class RepresentationExtractor(ABC):
    """Extracts discrete concept representations from model checkpoints."""

    @abstractmethod
    def extract(self, checkpoint_path: str, concepts: List[str],
                device: str = 'cpu') -> ConceptSnapshot:
        """Extract a snapshot from a single checkpoint.

        Args:
            checkpoint_path: Path to the model checkpoint file.
            concepts: List of concept strings to extract.
            device: Torch device string.

        Returns:
            ConceptSnapshot with codes for each concept found.
        """
        ...

    @abstractmethod
    def similarity(self, code_a: List[int], code_b: List[int]) -> float:
        """Compute similarity between two concept codes.

        Returns a value in [0, 1] where 1 means identical.
        """
        ...

    @abstractmethod
    def shared_features(self, code_a: List[int], code_b: List[int]) -> List[int]:
        """Return indices of features shared by both codes."""
        ...

    def discover_checkpoints(self, directory: str) -> List[Tuple[int, str]]:
        """Find all step checkpoints in a directory, sorted by step.

        Handles common naming patterns:
          - model_step5000.pt
          - model_xl_step5000.pt
          - model_best.pt (excluded -- not a step checkpoint)
          - checkpoint-5000.pt
          - step_5000.safetensors

        Returns:
            List of (step, path) tuples sorted ascending.
        """
        patterns = [
            re.compile(r'model_.*?step(\d+)\.pt$'),
            re.compile(r'model_step(\d+)\.pt$'),
            re.compile(r'checkpoint[-_](\d+)\.pt$'),
            re.compile(r'step[-_](\d+)\.(pt|safetensors|bin)$'),
        ]
        results = []
        for fname in os.listdir(directory):
            for pattern in patterns:
                m = pattern.match(fname)
                if m:
                    step = int(m.group(1))
                    results.append((step, os.path.join(directory, fname)))
                    break
        results.sort(key=lambda x: x[0])
        return results

    def extract_sequence(self, directory: str, concepts: List[str],
                         device: str = 'cpu',
                         max_checkpoints: Optional[int] = None
                         ) -> List[ConceptSnapshot]:
        """Extract snapshots from all checkpoints in a directory.

        Args:
            directory: Checkpoint directory.
            concepts: Concepts to track.
            device: Torch device.
            max_checkpoints: Limit number of checkpoints (evenly spaced).

        Returns:
            List of ConceptSnapshot sorted by step.
        """
        checkpoints = self.discover_checkpoints(directory)
        if not checkpoints:
            raise FileNotFoundError(f"No step checkpoints found in {directory}")

        if max_checkpoints and len(checkpoints) > max_checkpoints:
            indices = [int(i * (len(checkpoints) - 1) / (max_checkpoints - 1))
                       for i in range(max_checkpoints)]
            checkpoints = [checkpoints[i] for i in indices]

        snapshots = []
        for i, (step, path) in enumerate(checkpoints):
            logger.info("[%d/%d] Extracting step %s...", i + 1, len(checkpoints), f"{step:,}")
            snapshot = self.extract(path, concepts, device)
            snapshots.append(snapshot)

        return snapshots
