"""
PythiaSAEExtractor: Extract discrete codes from Pythia-70M via EleutherAI SAEs.

Uses pre-trained Sparse Autoencoders (32K features) on Pythia-70M checkpoints
to produce binary concept codes for reptimeline analysis.

Requirements:
    pip install transformers torch
    pip install "git+https://github.com/EleutherAI/sae.git"

Usage:
    from sae_extractor import PythiaSAEExtractor
    from reptimeline import TimelineTracker, BitDiscovery

    extractor = PythiaSAEExtractor(layer=3, top_k=256, device='cuda')
    snapshots = extractor.extract_checkpoints(concepts, checkpoints)

    tracker = TimelineTracker(extractor)
    timeline = tracker.analyze(snapshots)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor

logger = logging.getLogger(__name__)


class PythiaSAEExtractor(RepresentationExtractor):
    """Extract binary codes from Pythia-70M hidden states via SAE encoding.

    Pipeline per concept:
        Pythia forward pass → hook layer N MLP output → SAE.encode() →
        top_indices → binarize against selected feature set → List[int]
    """

    MODEL_NAME = "EleutherAI/pythia-70m"
    SAE_NAME = "EleutherAI/sae-pythia-70m-32k"

    def __init__(
        self,
        layer: int = 3,
        top_k: int = 256,
        device: str = "cpu",
        context_template: str = "The word is: {concept}",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            layer: Pythia layer to extract from (0-5). Layer 3 recommended.
            top_k: Number of SAE features to keep (code dimension).
            device: 'cuda' or 'cpu'.
            context_template: Prompt template with {concept} placeholder.
            cache_dir: HuggingFace cache directory (None = default).
        """
        self.layer = layer
        self.top_k = top_k
        self.device = device
        self.context_template = context_template
        self.cache_dir = cache_dir

        # Loaded on demand
        self._tokenizer = None
        self._sae = None
        self._selected_features: Optional[np.ndarray] = None

    # -- Loading ----------------------------------------------------------

    def _load_tokenizer(self):
        """Load tokenizer once (shared across all checkpoints)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME, cache_dir=self.cache_dir
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _load_model(self, revision: str):
        """Load Pythia-70M at a specific training checkpoint.

        Args:
            revision: HuggingFace revision string (e.g., 'step1000').

        Returns:
            GPTNeoXForCausalLM model on self.device.
        """
        import torch
        from transformers import GPTNeoXForCausalLM

        logger.info(f"Loading {self.MODEL_NAME} revision={revision}")
        model = GPTNeoXForCausalLM.from_pretrained(
            self.MODEL_NAME,
            revision=revision,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float32,
        ).to(self.device)
        model.eval()
        return model

    def _load_sae(self):
        """Load SAE once (same SAE for all checkpoints)."""
        if self._sae is None:
            from sparsify import Sae

            hookpoint = f"layers.{self.layer}"
            logger.info(f"Loading SAE: {self.SAE_NAME} hookpoint={hookpoint}")
            self._sae = Sae.load_from_hub(
                self.SAE_NAME,
                hookpoint=hookpoint,
                device=self.device,
            )
        return self._sae

    # -- Hidden state extraction ------------------------------------------

    def _get_hidden_states(self, model, concepts: List[str]) -> Dict[str, "torch.Tensor"]:
        """Run all concepts through the model, return MLP outputs at target layer.

        Uses a forward hook on gpt_neox.layers[N].mlp to capture activations.
        Returns dict mapping concept -> hidden state tensor (d_model,).
        """
        import torch

        tokenizer = self._load_tokenizer()
        hidden_states = {}
        captured = {}

        # Register hook on target layer's MLP
        hook_layer = model.gpt_neox.layers[self.layer].mlp

        def hook_fn(module, input, output):
            captured["hidden"] = output

        handle = hook_layer.register_forward_hook(hook_fn)

        try:
            for concept in concepts:
                prompt = self.context_template.format(concept=concept)
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    model(**inputs)

                # Take the last token's hidden state (the concept token)
                h = captured["hidden"][0, -1, :].clone()  # (d_model,)
                hidden_states[concept] = h
        finally:
            handle.remove()

        return hidden_states

    # -- Feature selection ------------------------------------------------

    def select_features(
        self,
        concepts: List[str],
        revision: str = "step143000",
    ) -> np.ndarray:
        """Select top-K most activated SAE features across all concepts.

        Runs at the FINAL checkpoint to find which features are most used,
        then locks that set for all checkpoints to ensure consistent codes.

        Args:
            concepts: List of concept strings.
            revision: Checkpoint to use for selection (default: final).

        Returns:
            np.ndarray of shape (top_k,) with selected feature indices.
        """
        import torch

        model = self._load_model(revision)
        sae = self._load_sae()

        # Count how many concepts activate each feature
        n_features = sae.num_latents if hasattr(sae, "num_latents") else 32768
        feature_counts = np.zeros(n_features, dtype=np.int32)

        hidden_states = self._get_hidden_states(model, concepts)

        for concept, h in hidden_states.items():
            enc = sae.encode(h.unsqueeze(0))  # expects (batch, d_model)
            indices = enc.top_indices[0].cpu().numpy()  # (k,)
            for idx in indices:
                feature_counts[idx] += 1

        # Free model VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Select top-K most frequently activated features
        selected = np.argsort(feature_counts)[::-1][: self.top_k].copy()
        selected.sort()  # Sort for consistent ordering
        self._selected_features = selected

        logger.info(
            f"Selected {len(selected)} features. "
            f"Most activated: idx={selected[0]} ({feature_counts[selected[0]]} concepts), "
            f"least: idx={selected[-1]} ({feature_counts[selected[-1]]} concepts)"
        )
        return selected

    # -- Core extraction --------------------------------------------------

    def extract(
        self,
        checkpoint_path: str,
        concepts: List[str],
        device: str = "cpu",
    ) -> ConceptSnapshot:
        """Extract binary codes for all concepts at one checkpoint.

        Args:
            checkpoint_path: HuggingFace revision string (e.g., 'step1000').
            concepts: List of concept strings.
            device: Ignored (uses self.device).

        Returns:
            ConceptSnapshot with binary codes.
        """
        import torch

        if self._selected_features is None:
            raise RuntimeError(
                "Call select_features() first to choose which SAE features to track."
            )

        revision = checkpoint_path  # We use revision strings, not paths
        model = self._load_model(revision)
        sae = self._load_sae()

        # Build feature index set for O(1) lookup
        feature_set = set(self._selected_features.tolist())
        feature_to_idx = {f: i for i, f in enumerate(self._selected_features)}

        hidden_states = self._get_hidden_states(model, concepts)

        codes = {}
        for concept, h in hidden_states.items():
            enc = sae.encode(h.unsqueeze(0))
            active_indices = set(enc.top_indices[0].cpu().numpy().tolist())

            # Binary code: 1 if feature is in SAE's top-k activations, else 0
            code = [0] * self.top_k
            for feat_idx in active_indices & feature_set:
                code[feature_to_idx[feat_idx]] = 1
            codes[concept] = code

        # Parse step from revision
        step = 0
        if revision.startswith("step"):
            try:
                step = int(revision[4:])
            except ValueError:
                pass

        # Free model VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ConceptSnapshot(step=step, codes=codes)

    def extract_checkpoints(
        self,
        concepts: List[str],
        checkpoints: List[Dict],
        select_revision: str = "step143000",
    ) -> List[ConceptSnapshot]:
        """Extract snapshots for multiple checkpoints.

        Args:
            concepts: List of concept strings.
            checkpoints: List of dicts with 'step' and 'revision' keys.
            select_revision: Revision for feature selection.

        Returns:
            List of ConceptSnapshot, one per checkpoint.
        """
        # Step 1: Feature selection at final checkpoint
        if self._selected_features is None:
            logger.info(f"=== Feature selection at {select_revision} ===")
            self.select_features(concepts, revision=select_revision)

        # Step 2: Extract codes at each checkpoint
        snapshots = []
        for i, ckpt in enumerate(checkpoints):
            rev = ckpt["revision"]
            step = ckpt["step"]
            logger.info(f"=== Checkpoint {i+1}/{len(checkpoints)}: step {step} ({rev}) ===")
            snap = self.extract(rev, concepts)
            snapshots.append(snap)

        return snapshots

    # -- Similarity (RepresentationExtractor interface) -------------------

    def similarity(self, code_a: List[int], code_b: List[int]) -> float:
        """Jaccard similarity between two binary codes."""
        a = set(i for i, v in enumerate(code_a) if v == 1)
        b = set(i for i, v in enumerate(code_b) if v == 1)
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    def shared_features(self, code_a: List[int], code_b: List[int]) -> List[int]:
        """Indices where both codes are active."""
        return [
            i
            for i in range(min(len(code_a), len(code_b)))
            if code_a[i] == 1 and code_b[i] == 1
        ]

    # -- Serialization ----------------------------------------------------

    def save_feature_selection(self, path: str):
        """Save selected feature indices to JSON."""
        if self._selected_features is None:
            raise RuntimeError("No features selected yet.")
        with open(path, "w") as f:
            json.dump(
                {
                    "model": self.MODEL_NAME,
                    "sae": self.SAE_NAME,
                    "layer": self.layer,
                    "top_k": self.top_k,
                    "features": self._selected_features.tolist(),
                },
                f,
                indent=2,
            )
        logger.info(f"Saved feature selection to {path}")

    def load_feature_selection(self, path: str):
        """Load previously saved feature indices."""
        with open(path) as f:
            data = json.load(f)
        self._selected_features = np.array(data["features"])
        logger.info(f"Loaded {len(self._selected_features)} features from {path}")
