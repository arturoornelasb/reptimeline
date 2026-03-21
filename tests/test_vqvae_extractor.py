"""Tests for the VQ-VAE extractor."""

import numpy as np
import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.vqvae import VQVAEExtractor


@pytest.fixture
def mock_encode_fn():
    """Returns known codebook indices for testing."""
    def encode(input_data):
        # Return indices based on input identity
        return np.array(input_data)
    return encode


@pytest.fixture
def extractor(mock_encode_fn):
    return VQVAEExtractor(n_codebook=8, encode_fn=mock_encode_fn)


class TestIndicesConversion:

    def test_single_index(self, extractor):
        binary = extractor._indices_to_binary(np.array([3]))
        assert binary == [0, 0, 0, 1, 0, 0, 0, 0]

    def test_multiple_indices(self, extractor):
        binary = extractor._indices_to_binary(np.array([0, 3, 7]))
        assert binary == [1, 0, 0, 1, 0, 0, 0, 1]

    def test_duplicate_indices(self, extractor):
        binary = extractor._indices_to_binary(np.array([2, 2, 5]))
        assert binary == [0, 0, 1, 0, 0, 1, 0, 0]

    def test_out_of_range_ignored(self, extractor):
        binary = extractor._indices_to_binary(np.array([1, 99]))
        assert binary == [0, 1, 0, 0, 0, 0, 0, 0]

    def test_empty_indices(self, extractor):
        binary = extractor._indices_to_binary(np.array([]))
        assert binary == [0] * 8

    def test_2d_input_flattened(self, extractor):
        binary = extractor._indices_to_binary(np.array([[1, 4], [2, 6]]))
        assert binary == [0, 1, 1, 0, 1, 0, 1, 0]


class TestSimilarity:

    def test_identical(self, extractor):
        a = [1, 0, 1, 0, 0, 0, 0, 0]
        assert extractor.similarity(a, a) == 1.0

    def test_disjoint(self, extractor):
        a = [1, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 1, 0, 0, 0, 0, 0, 0]
        assert extractor.similarity(a, b) == 0.0

    def test_overlap(self, extractor):
        a = [1, 1, 0, 0, 0, 0, 0, 0]
        b = [1, 0, 1, 0, 0, 0, 0, 0]
        assert extractor.similarity(a, b) == pytest.approx(1 / 3)

    def test_empty_codes(self, extractor):
        a = [0] * 8
        b = [0] * 8
        assert extractor.similarity(a, b) == 1.0


class TestSharedFeatures:

    def test_shared(self, extractor):
        a = [1, 1, 0, 1, 0, 0, 0, 0]
        b = [0, 1, 0, 1, 1, 0, 0, 0]
        assert extractor.shared_features(a, b) == [1, 3]

    def test_none_shared(self, extractor):
        a = [1, 0, 0, 0, 0, 0, 0, 0]
        b = [0, 1, 0, 0, 0, 0, 0, 0]
        assert extractor.shared_features(a, b) == []


class TestExtract:

    def test_extract_produces_snapshot(self, tmp_path):
        encode_fn = lambda x: np.array(x)
        extractor = VQVAEExtractor(n_codebook=8, encode_fn=encode_fn)

        ckpt = tmp_path / "model_step1000.pt"
        ckpt.write_text("dummy")

        concepts = {'cat': [0, 2], 'dog': [1, 3, 5]}
        snap = extractor.extract(str(ckpt), concepts)

        assert isinstance(snap, ConceptSnapshot)
        assert snap.step == 1000
        assert snap.codes['cat'] == [1, 0, 1, 0, 0, 0, 0, 0]
        assert snap.codes['dog'] == [0, 1, 0, 1, 0, 1, 0, 0]

    def test_model_loader_called(self, tmp_path):
        loaded = []
        def loader(path):
            loaded.append(path)

        encode_fn = lambda x: np.array([0])
        extractor = VQVAEExtractor(n_codebook=4, encode_fn=encode_fn,
                                    model_loader=loader)

        ckpt = tmp_path / "step_500.pt"
        ckpt.write_text("dummy")

        extractor.extract(str(ckpt), {'a': [0]})
        assert len(loaded) == 1


class TestImport:

    def test_importable_from_extractors(self):
        from reptimeline.extractors.vqvae import VQVAEExtractor
        assert VQVAEExtractor is not None

    def test_importable_from_package(self):
        from reptimeline.extractors import VQVAEExtractor
        assert VQVAEExtractor is not None
