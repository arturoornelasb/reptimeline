"""SAE (Sparse Autoencoder) extractor for reptimeline.

Converts SAE feature activations into binary codes and provides
an intervention function compatible with CausalVerifier.

Usage:
    sae = SAEExtractor(
        n_features=32768,
        encode_fn=lambda h: my_sae.encode(h),
        decode_fn=lambda idx, act: my_sae.decode(idx, act),
        feature_indices=selected_features,
    )
    snapshot = sae.extract("checkpoint.pt", concept_data)
    intervene_fn = sae.make_intervene_fn(concept_hidden_states)
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor


class SAEExtractor(RepresentationExtractor):
    """Extract discrete codes from Sparse Autoencoder features.

    Binarizes SAE feature activations: 1 if the feature is in the
    active set, 0 otherwise.

    Args:
        n_features: Total number of SAE features.
        encode_fn: Callable(hidden_states) -> (top_indices, top_acts).
        decode_fn: Optional Callable(top_indices, top_acts) -> hidden_states.
            Required for intervention.
        feature_indices: Optional list of SAE feature indices to track.
            If provided, codes are len(feature_indices) bits long, mapping
            bit i to SAE feature feature_indices[i].
        model_loader: Optional Callable(checkpoint_path) -> model.
        device: Device string.
    """

    def __init__(
        self,
        n_features: int,
        encode_fn: Callable,
        decode_fn: Optional[Callable] = None,
        feature_indices: Optional[List[int]] = None,
        model_loader: Optional[Callable[[str], Any]] = None,
        device: str = 'cpu',
    ):
        self.n_features = n_features
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.feature_indices = feature_indices
        self.model_loader = model_loader
        self.device = device

        self._feat_to_bit: Optional[Dict[int, int]] = None
        if feature_indices is not None:
            self._feat_to_bit = {f: i for i, f in enumerate(feature_indices)}
            self._code_dim = len(feature_indices)
        else:
            self._code_dim = n_features

    def extract(self, checkpoint_path: str, concepts: Dict[str, Any],
                device: str = 'cpu') -> ConceptSnapshot:
        """Extract binary codes from SAE features.

        Args:
            checkpoint_path: Path to model checkpoint.
            concepts: Dict mapping concept name to hidden state input.
        """
        if self.model_loader is not None:
            self.model_loader(checkpoint_path)

        codes = {}
        for concept, hidden_state in concepts.items():
            top_indices, top_acts = self.encode_fn(hidden_state)
            codes[concept] = self._activation_to_binary(top_indices)

        step = 0
        try:
            import re
            m = re.search(r'(\d+)', str(checkpoint_path))
            if m:
                step = int(m.group(1))
        except (ValueError, AttributeError):
            pass

        return ConceptSnapshot(step=step, codes=codes)

    def _activation_to_binary(self, top_indices) -> List[int]:
        """Convert SAE top-k indices to binary code vector."""
        binary = [0] * self._code_dim
        for idx in np.asarray(top_indices).flatten():
            idx_int = int(idx)
            if self._feat_to_bit is not None:
                bit = self._feat_to_bit.get(idx_int)
                if bit is not None:
                    binary[bit] = 1
            else:
                if 0 <= idx_int < self._code_dim:
                    binary[idx_int] = 1
        return binary

    def intervene(self, hidden_state: Any, bit_index: int) -> float:
        """Zero one SAE feature and return L2 perturbation.

        Args:
            hidden_state: Input to encode_fn.
            bit_index: Which bit to zero. Maps to the actual SAE feature
                index via feature_indices if provided.

        Returns:
            L2 norm of (original_reconstruction - modified_reconstruction).
        """
        if self.decode_fn is None:
            raise ValueError("decode_fn required for intervention")

        sae_feature = (self.feature_indices[bit_index]
                       if self.feature_indices else bit_index)

        top_indices, top_acts = self.encode_fn(hidden_state)
        top_indices = np.asarray(top_indices).flatten()
        top_acts = np.asarray(top_acts).flatten()

        h_orig = self.decode_fn(top_indices, top_acts)

        mask = top_indices != sae_feature
        if mask.all():
            return 0.0  # Feature not active

        h_modified = self.decode_fn(top_indices[mask], top_acts[mask])

        return float(np.linalg.norm(
            np.asarray(h_orig).flatten() - np.asarray(h_modified).flatten()
        ))

    def make_intervene_fn(
        self, concept_hidden: Dict[str, Any],
    ) -> Callable[[str, int], float]:
        """Create an intervene_fn closure for CausalVerifier.

        Args:
            concept_hidden: Dict mapping concept name to hidden state.

        Returns:
            Callable(concept, bit_index) -> float.
        """
        def intervene_fn(concept: str, bit_index: int) -> float:
            return self.intervene(concept_hidden[concept], bit_index)
        return intervene_fn

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
