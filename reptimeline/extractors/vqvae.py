"""
VQ-VAE extractor — converts codebook indices into binary indicator vectors.

This extractor works with any VQ-VAE architecture. The user provides:
  - n_codebook: size of the codebook (becomes the code dimension)
  - encode_fn: callable that maps input tensors to codebook indices
  - model_loader: optional callable that loads a model from a checkpoint path
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor


class VQVAEExtractor(RepresentationExtractor):
    """Extract discrete codes from VQ-VAE models.

    Converts VQ-VAE codebook indices to binary indicator vectors
    for reptimeline analysis. Each codebook entry becomes a bit position;
    the indicator is 1 if that entry was selected by the encoder.

    Args:
        n_codebook: Size of the VQ-VAE codebook.
        encode_fn: Function(input_tensor) -> index tensor of codebook indices.
            Can return shape (batch,), (batch, n_quantizers), or (n_quantizers,).
        model_loader: Optional function(checkpoint_path) -> model.
            Called before encode_fn when extracting from checkpoints.
        device: Torch device string.
    """

    def __init__(self, n_codebook: int,
                 encode_fn: Callable,
                 model_loader: Optional[Callable[[str], Any]] = None,
                 device: str = 'cpu'):
        self.n_codebook = n_codebook
        self.encode_fn = encode_fn
        self.model_loader = model_loader
        self.device = device

    def extract(self, checkpoint_path: str, concepts: Dict[str, Any],
                device: str = 'cpu') -> ConceptSnapshot:
        """Extract codes from a checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
            concepts: Dict mapping concept name to input data (tensors or arrays).
            device: Torch device.

        Returns:
            ConceptSnapshot with binary indicator codes.
        """
        if self.model_loader is not None:
            self.model_loader(checkpoint_path)

        codes = {}
        for name, input_data in concepts.items():
            indices = self.encode_fn(input_data)
            codes[name] = self._indices_to_binary(indices)

        step = self._parse_step(checkpoint_path)
        return ConceptSnapshot(step=step, codes=codes)

    def _indices_to_binary(self, indices) -> List[int]:
        """Convert codebook indices to a binary indicator vector.

        Handles numpy arrays, lists, or scalar indices.
        """
        arr = np.asarray(indices).flatten()
        binary = [0] * self.n_codebook
        for idx in arr:
            idx_int = int(idx)
            if 0 <= idx_int < self.n_codebook:
                binary[idx_int] = 1
        return binary

    def _parse_step(self, path: str) -> int:
        """Try to extract training step from filename, default to 0."""
        import re
        m = re.search(r'(\d+)', path.split('/')[-1].split('\\')[-1])
        return int(m.group(1)) if m else 0

    def similarity(self, code_a: List[int], code_b: List[int]) -> float:
        """Jaccard similarity between binary indicator codes."""
        a = set(i for i, v in enumerate(code_a) if v == 1)
        b = set(i for i, v in enumerate(code_b) if v == 1)
        union = a | b
        if not union:
            return 1.0
        return len(a & b) / len(union)

    def shared_features(self, code_a: List[int], code_b: List[int]) -> List[int]:
        """Indices where both codes are active (shared codebook entries)."""
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]
