"""
MNISTBinaryAEExtractor — reptimeline extractor for Binary Autoencoder on MNIST.

Concepts are digit classes (0-9). Code per digit = majority vote of binary
codes across all test samples of that class.
"""

import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor
from mnist_binary_ae import BinaryAE


class MNISTBinaryAEExtractor(RepresentationExtractor):
    """Extract binary codes from MNIST digits via a trained Binary Autoencoder."""

    def __init__(self, input_dim=784, hidden=256, bottleneck=32, device="cpu"):
        self.input_dim = input_dim
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.device = device
        self._digit_images: Optional[Dict[int, torch.Tensor]] = None

    def _load_digit_images(self) -> Dict[int, torch.Tensor]:
        """Load MNIST test set grouped by digit class."""
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        test_data = datasets.MNIST("data/mnist", train=False, download=True, transform=transform)

        digit_images = {d: [] for d in range(10)}
        for img, label in test_data:
            digit_images[label].append(img)

        return {d: torch.stack(imgs) for d, imgs in digit_images.items()}

    def extract(self, checkpoint_path: str, concepts: List[str],
                device: str = "cpu") -> ConceptSnapshot:
        """Extract binary codes for digit concepts from a checkpoint.

        Args:
            checkpoint_path: Path to model_stepN.pt checkpoint.
            concepts: List of digit strings ['0', '1', ..., '9'].
            device: Ignored (uses self.device).
        """
        # Load model
        model = BinaryAE(self.input_dim, self.hidden, self.bottleneck)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device,
                                         weights_only=True))
        model.to(self.device)
        model.eval()

        # Load digit images (cached)
        if self._digit_images is None:
            self._digit_images = self._load_digit_images()

        # Extract codes per digit class via majority vote
        codes = {}
        for concept in concepts:
            digit = int(concept)
            images = self._digit_images[digit].to(self.device)

            with torch.no_grad():
                binary_codes = model.encode_binary(images)  # (N, bottleneck)

            # Majority vote: if >50% of samples have bit active, set to 1
            mean_code = binary_codes.mean(dim=0)
            majority = (mean_code > 0.5).int().cpu().tolist()
            codes[concept] = majority

        # Parse step from filename
        step = 0
        m = re.search(r'step(\d+)', os.path.basename(checkpoint_path))
        if m:
            step = int(m.group(1))

        return ConceptSnapshot(step=step, codes=codes)

    def similarity(self, code_a: List[int], code_b: List[int]) -> float:
        """Jaccard similarity."""
        a = set(i for i, v in enumerate(code_a) if v == 1)
        b = set(i for i, v in enumerate(code_b) if v == 1)
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    def shared_features(self, code_a: List[int], code_b: List[int]) -> List[int]:
        """Indices where both codes are active."""
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]
