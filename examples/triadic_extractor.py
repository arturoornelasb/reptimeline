"""
Example: TriadicExtractor for triadic-microgpt checkpoints.

This is a concrete backend implementation that loads TriadicGPT checkpoints
and extracts 63-bit triadic codes. It serves as a reference for implementing
your own RepresentationExtractor.

Requirements:
    - triadic-microgpt repo (for src.evaluate, src.triadic)
    - torch
    - The checkpoint and tokenizer files

Usage:
    import sys
    sys.path.insert(0, '/path/to/triadic-microgpt')

    from triadic_extractor import TriadicExtractor
    from reptimeline import TimelineTracker

    extractor = TriadicExtractor()
    snapshots = extractor.extract_sequence('checkpoints/', concepts)
    tracker = TimelineTracker(extractor)
    timeline = tracker.analyze(snapshots)
    timeline.print_summary()
"""

import math
import os
import re
from typing import List, Optional

import torch

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor


class TriadicExtractor(RepresentationExtractor):
    """Extracts triadic bit representations from TriadicGPT checkpoints.

    Uses PrimeMapper for bit->prime mapping and Jaccard on active bits
    for similarity.
    """

    def __init__(self, tokenizer_path: Optional[str] = None,
                 n_bits: int = 63, max_tokens: int = 4,
                 project_root: Optional[str] = None):
        """
        Args:
            tokenizer_path: Path to tokenizer.json. If None, auto-detected
                from checkpoint directory.
            n_bits: Number of triadic bits (default: 63).
            max_tokens: Max tokens per concept (4 for custom BPE, 8 for GPT-2).
            project_root: Path to triadic-microgpt repo root. Required for
                importing src.evaluate and src.triadic.
        """
        self.tokenizer_path = tokenizer_path
        self.n_bits = n_bits
        self.max_tokens = max_tokens

        if project_root:
            import sys
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

    def extract(self, checkpoint_path: str, concepts: List[str],
                device: str = 'cpu') -> ConceptSnapshot:
        """Extract triadic snapshot from a TriadicGPT checkpoint."""
        from src.evaluate import load_model
        from src.triadic import PrimeMapper

        # Auto-detect tokenizer
        tok_path = self.tokenizer_path
        if tok_path is None:
            ckpt_dir = os.path.dirname(checkpoint_path)
            tok_path = os.path.join(ckpt_dir, 'tokenizer.json')

        model, tokenizer, config = load_model(checkpoint_path, tok_path, device)
        mapper = PrimeMapper(config.n_triadic_bits)

        codes = {}
        continuous = {}
        composites = {}

        for concept in concepts:
            try:
                ids = tokenizer.encode(concept, add_special=False)[:self.max_tokens]
            except Exception:
                continue
            if not ids:
                continue

            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                triadic_proj = model(x)[1]

            proj = triadic_proj[0].mean(dim=0).cpu().numpy()
            bits = mapper.get_bits(proj)
            composite = int(mapper.map(proj.tolist()))

            codes[concept] = bits
            continuous[concept] = proj.tolist()
            composites[concept] = composite

        # Parse step from filename
        basename = os.path.basename(checkpoint_path)
        step = 0
        m = re.search(r'step(\d+)', basename)
        if m:
            step = int(m.group(1))

        # Free GPU
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ConceptSnapshot(
            step=step,
            codes=codes,
            continuous=continuous,
            metadata={'composites': composites, 'n_bits': config.n_triadic_bits},
        )

    def similarity(self, code_a: List[int], code_b: List[int]) -> float:
        """Jaccard similarity on active bits."""
        active_a = set(i for i, v in enumerate(code_a) if v == 1)
        active_b = set(i for i, v in enumerate(code_b) if v == 1)
        if not active_a and not active_b:
            return 1.0
        union = active_a | active_b
        if not union:
            return 0.0
        return len(active_a & active_b) / len(union)

    def shared_features(self, code_a: List[int], code_b: List[int]) -> List[int]:
        """Indices where both codes are active (both == 1)."""
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]

    def algebraic_similarity(self, composite_a: int, composite_b: int) -> float:
        """GCD-based similarity using prime factorization.

        Triadic-specific: Jaccard on prime factors.
        """
        from src.triadic import prime_factors
        factors_a = set(prime_factors(composite_a))
        factors_b = set(prime_factors(composite_b))
        union = factors_a | factors_b
        if not union:
            return 0.0
        return len(factors_a & factors_b) / len(union)

    def are_connected(self, composite_a: int, composite_b: int) -> bool:
        """Two concepts are connected if they share at least one prime factor."""
        return (composite_a > 1 and composite_b > 1
                and math.gcd(composite_a, composite_b) > 1)
