"""Shared test fixtures for reptimeline."""

import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.base import RepresentationExtractor


class SyntheticExtractor(RepresentationExtractor):
    """Dummy extractor for testing -- no model needed."""

    def extract(self, checkpoint_path, concepts, device='cpu'):
        raise NotImplementedError("Use synthetic snapshots directly")

    def similarity(self, code_a, code_b):
        active_a = set(i for i, v in enumerate(code_a) if v == 1)
        active_b = set(i for i, v in enumerate(code_b) if v == 1)
        union = active_a | active_b
        if not union:
            return 1.0
        return len(active_a & active_b) / len(union)

    def shared_features(self, code_a, code_b):
        return [i for i in range(min(len(code_a), len(code_b)))
                if code_a[i] == 1 and code_b[i] == 1]


def make_code(n_bits, active_indices):
    """Create a binary code with specific bits active."""
    code = [0] * n_bits
    for i in active_indices:
        code[i] = 1
    return code


@pytest.fixture
def synthetic_extractor():
    return SyntheticExtractor()


@pytest.fixture
def basic_snapshots():
    """5-step synthetic evolution for 3 concepts."""
    n_bits = 16
    return [
        ConceptSnapshot(
            step=0,
            codes={
                'king': make_code(n_bits, [0, 1]),
                'queen': make_code(n_bits, [0, 1]),
                'fire': make_code(n_bits, [2]),
            },
        ),
        ConceptSnapshot(
            step=1000,
            codes={
                'king': make_code(n_bits, [0, 1, 2, 5]),
                'queen': make_code(n_bits, [0, 1, 6]),
                'fire': make_code(n_bits, [2, 3, 7, 8]),
            },
        ),
        ConceptSnapshot(
            step=2000,
            codes={
                'king': make_code(n_bits, [0, 1, 2, 5, 9]),
                'queen': make_code(n_bits, [0, 1, 6, 10]),
                'fire': make_code(n_bits, [2, 3, 7, 8, 4]),
            },
        ),
        ConceptSnapshot(
            step=3000,
            codes={
                'king': make_code(n_bits, [0, 1, 2, 5, 9]),
                'queen': make_code(n_bits, [0, 1, 6, 10]),
                'fire': make_code(n_bits, [2, 3, 7, 8, 4]),
            },
        ),
        ConceptSnapshot(
            step=4000,
            codes={
                'king': make_code(n_bits, [1, 2, 5, 9]),  # lost bit 0
                'queen': make_code(n_bits, [0, 1, 6, 10]),
                'fire': make_code(n_bits, [2, 3, 7, 8, 4]),
            },
        ),
    ]
