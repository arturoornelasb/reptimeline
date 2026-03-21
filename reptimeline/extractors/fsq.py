"""FSQ (Finite Scalar Quantization) extractor for reptimeline.

Converts FSQ level vectors into binary codes by treating each non-zero
level as an active bit.

FSQ quantizes each dimension to a small set of discrete levels
(e.g., [-1, 0, 1] for 3 levels). This extractor binarizes:
  bit = 1 if level != 0 (or if level is in a specified active set).

Usage:
    fsq = FSQExtractor(
        n_levels=[3, 5, 3, 3],
        encode_fn=lambda x: my_model.encode(x),
    )
    snapshot = fsq.extract("checkpoint.pt", concept_data)
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor


class FSQExtractor(RepresentationExtractor):
    """Extract discrete codes from Finite Scalar Quantization models.

    FSQ produces a vector of discrete levels per input. This extractor
    converts each level into one or more binary bits depending on the mode:

    - 'nonzero' (default): 1 if level != 0, else 0. One bit per dimension.
    - 'onehot': One-hot encoding per dimension. Total bits = sum(n_levels).

    Args:
        n_levels: List of level counts per FSQ dimension (e.g., [3, 5, 3]).
        encode_fn: Callable(input) -> level vector (numpy array or list).
            Should return integer levels in range [0, n_levels[i]) or
            centered (e.g., [-1, 0, 1] for n_levels=3).
        model_loader: Optional Callable(checkpoint_path) -> model.
        binarize: 'nonzero' or 'onehot'.
        device: Device string.
    """

    def __init__(
        self,
        n_levels: List[int],
        encode_fn: Callable,
        model_loader: Optional[Callable[[str], Any]] = None,
        binarize: str = 'nonzero',
        device: str = 'cpu',
    ):
        if binarize not in ('nonzero', 'onehot'):
            raise ValueError(f"binarize must be 'nonzero' or 'onehot', got '{binarize}'")
        self.n_levels = n_levels
        self.n_dims = len(n_levels)
        self.encode_fn = encode_fn
        self.model_loader = model_loader
        self.binarize = binarize
        self.device = device

        if binarize == 'onehot':
            self._code_dim = sum(n_levels)
            self._offsets = []
            offset = 0
            for nl in n_levels:
                self._offsets.append(offset)
                offset += nl
        else:
            self._code_dim = self.n_dims

    @property
    def code_dim(self) -> int:
        return self._code_dim

    def extract(self, checkpoint_path: str, concepts: Dict[str, Any],
                device: str = 'cpu') -> ConceptSnapshot:
        """Extract binary codes from FSQ levels.

        Args:
            checkpoint_path: Path to model checkpoint.
            concepts: Dict mapping concept name to input data.
        """
        if self.model_loader is not None:
            self.model_loader(checkpoint_path)

        codes = {}
        for concept, input_data in concepts.items():
            levels = self.encode_fn(input_data)
            codes[concept] = self._levels_to_binary(levels)

        step = 0
        try:
            import re
            m = re.search(r'(\d+)', str(checkpoint_path).split('/')[-1].split('\\')[-1])
            if m:
                step = int(m.group(1))
        except (ValueError, AttributeError):
            pass

        return ConceptSnapshot(step=step, codes=codes)

    def _levels_to_binary(self, levels) -> List[int]:
        """Convert FSQ level vector to binary code.

        Args:
            levels: Array-like of quantized levels per dimension.
        """
        arr = np.asarray(levels).flatten()

        if self.binarize == 'nonzero':
            return [int(v != 0) for v in arr[:self.n_dims]]

        # onehot mode
        binary = [0] * self._code_dim
        for dim_idx, val in enumerate(arr[:self.n_dims]):
            level_int = int(val)
            nl = self.n_levels[dim_idx]
            # Center-to-index: if levels are centered (e.g., -1,0,1 for nl=3),
            # shift to 0-indexed
            if level_int < 0:
                level_int = level_int + nl // 2
            if 0 <= level_int < nl:
                binary[self._offsets[dim_idx] + level_int] = 1
        return binary

    def similarity(self, code_a: List[int], code_b: List[int]) -> float:
        """Jaccard similarity between two binary codes."""
        active_a = set(i for i, v in enumerate(code_a) if v == 1)
        active_b = set(i for i, v in enumerate(code_b) if v == 1)
        union = active_a | active_b
        if not union:
            return 1.0
        return len(active_a & active_b) / len(union)

    def shared_features(self, code_a: List[int], code_b: List[int]) -> List[int]:
        """Return indices where both codes are active."""
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]
