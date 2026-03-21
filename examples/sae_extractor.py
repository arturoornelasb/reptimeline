"""
Example: SAE Extractor for analyzing LLMs via Sparse Autoencoders.

This shows how to use reptimeline with large language models by converting
continuous SAE features into discrete binary codes.

Requirements:
    - A trained SAE on your LLM's hidden states
    - LLM checkpoints at different training steps
    - torch

Usage:
    from sae_extractor import SAEExtractor
    from reptimeline import TimelineTracker, BitDiscovery

    extractor = SAEExtractor(sae_path="sae_llama3_layer12.pt")
    snapshots = extractor.extract_sequence("llama3_checkpoints/", concepts)

    tracker = TimelineTracker(extractor)
    timeline = tracker.analyze(snapshots)
    timeline.print_summary()

    # Discover what each SAE feature means
    discovery = BitDiscovery()
    report = discovery.discover(snapshots[-1], timeline=timeline)
"""

import os
import re
from typing import List, Optional

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor


class SAEExtractor(RepresentationExtractor):
    """Extract discrete codes from an LLM via a Sparse Autoencoder.

    Override _load_model() and _load_sae() with your specific implementations.
    """

    def __init__(self, sae_path: str, threshold: float = 0.5,
                 layer: int = 12, max_tokens: int = 8):
        """
        Args:
            sae_path: Path to trained SAE checkpoint.
            threshold: Activation threshold for binarization.
            layer: Which LLM layer to extract from.
            max_tokens: Max tokens per concept.
        """
        self.sae_path = sae_path
        self.threshold = threshold
        self.layer = layer
        self.max_tokens = max_tokens

    def _load_model(self, checkpoint_path: str, device: str = 'cpu'):
        """Load your LLM. Override this method.

        Returns:
            (model, tokenizer) tuple
        """
        raise NotImplementedError(
            "Override _load_model() with your LLM loading code. Example:\n"
            "  from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "  model = AutoModelForCausalLM.from_pretrained(checkpoint_path)\n"
            "  tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)\n"
            "  return model.to(device), tokenizer"
        )

    def _load_sae(self, device: str = 'cpu'):
        """Load your SAE. Override this method.

        Returns:
            SAE model that has an encode() method returning feature activations.
        """
        raise NotImplementedError(
            "Override _load_sae() with your SAE loading code."
        )

    def _get_hidden_state(self, model, tokenizer, concept: str, device: str):
        """Get hidden state for a concept from the LLM.

        Override for custom behavior (e.g., specific layer, pooling strategy).
        """
        import torch
        inputs = tokenizer(concept, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        # Mean-pool the target layer's hidden states
        hidden = outputs.hidden_states[self.layer].mean(dim=1).squeeze(0)
        return hidden

    def extract(self, checkpoint_path: str, concepts: List[str],
                device: str = 'cpu') -> ConceptSnapshot:
        """Extract binary codes from LLM hidden states via SAE."""
        import torch

        model, tokenizer = self._load_model(checkpoint_path, device)
        sae = self._load_sae(device)

        codes = {}
        for concept in concepts:
            try:
                hidden = self._get_hidden_state(model, tokenizer, concept, device)
                features = sae.encode(hidden)
                if isinstance(features, torch.Tensor):
                    features = features.cpu().numpy()
                binary = [1 if f > self.threshold else 0 for f in features]
                codes[concept] = binary
            except Exception:
                continue

        # Parse step from path
        basename = os.path.basename(checkpoint_path.rstrip('/'))
        step = 0
        m = re.search(r'(\d+)', basename)
        if m:
            step = int(m.group(1))

        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ConceptSnapshot(step=step, codes=codes)

    def similarity(self, code_a: List[int], code_b: List[int]) -> float:
        """Jaccard similarity on active features."""
        a = set(i for i, v in enumerate(code_a) if v == 1)
        b = set(i for i, v in enumerate(code_b) if v == 1)
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    def shared_features(self, code_a: List[int], code_b: List[int]) -> List[int]:
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]
