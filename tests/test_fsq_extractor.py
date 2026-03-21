"""Tests for FSQExtractor."""

import numpy as np
import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.extractors.fsq import FSQExtractor


def mock_encode(input_data):
    """Mock FSQ encoder: returns pre-set levels."""
    return input_data['levels']


# -- Binarization (nonzero mode) ----------------------------------------

class TestLevelsToBinaryNonzero:

    def test_nonzero_basic(self):
        fsq = FSQExtractor(n_levels=[3, 3, 3], encode_fn=mock_encode)
        binary = fsq._levels_to_binary([0, 1, -1])
        assert binary == [0, 1, 1]

    def test_all_zero(self):
        fsq = FSQExtractor(n_levels=[3, 3, 3], encode_fn=mock_encode)
        binary = fsq._levels_to_binary([0, 0, 0])
        assert binary == [0, 0, 0]

    def test_all_nonzero(self):
        fsq = FSQExtractor(n_levels=[5, 5, 5], encode_fn=mock_encode)
        binary = fsq._levels_to_binary([2, -1, 1])
        assert binary == [1, 1, 1]

    def test_code_dim_equals_n_dims(self):
        fsq = FSQExtractor(n_levels=[3, 5, 3, 3], encode_fn=mock_encode)
        assert fsq.code_dim == 4


# -- Binarization (onehot mode) -----------------------------------------

class TestLevelsToBinaryOnehot:

    def test_onehot_basic(self):
        """3 dims with [3, 3, 3] levels -> 9 bits total."""
        fsq = FSQExtractor(n_levels=[3, 3, 3], encode_fn=mock_encode,
                           binarize='onehot')
        # level 1 in dim 0 -> bit 1, level 0 in dim 1 -> bit 3, level 2 in dim 2 -> bit 8
        binary = fsq._levels_to_binary([1, 0, 2])
        assert len(binary) == 9
        assert binary[1] == 1   # dim 0, level 1
        assert binary[3] == 1   # dim 1, level 0
        assert binary[8] == 1   # dim 2, level 2
        assert sum(binary) == 3  # exactly one per dimension

    def test_onehot_code_dim(self):
        fsq = FSQExtractor(n_levels=[3, 5, 7], encode_fn=mock_encode,
                           binarize='onehot')
        assert fsq.code_dim == 15  # 3 + 5 + 7

    def test_centered_levels(self):
        """Centered levels (e.g., -1 for n_levels=3) map correctly."""
        fsq = FSQExtractor(n_levels=[3, 3], encode_fn=mock_encode,
                           binarize='onehot')
        # -1 with n_levels=3: -1 + 3//2 = 0 -> bit at offset+0
        binary = fsq._levels_to_binary([-1, 1])
        assert binary[0] == 1   # dim 0: -1 -> index 0
        assert binary[4] == 1   # dim 1: 1 -> index 1 (offset 3 + 1)


# -- Extract -------------------------------------------------------------

class TestExtract:

    def test_basic_extraction(self):
        fsq = FSQExtractor(n_levels=[3, 3, 3], encode_fn=mock_encode)
        concepts = {
            'cat': {'levels': [1, 0, -1]},
            'dog': {'levels': [0, 1, 0]},
        }
        snap = fsq.extract("step_500.pt", concepts)
        assert isinstance(snap, ConceptSnapshot)
        assert snap.codes['cat'] == [1, 0, 1]
        assert snap.codes['dog'] == [0, 1, 0]
        assert snap.step == 500

    def test_model_loader_called(self, tmp_path):
        loaded = []
        def loader(path):
            loaded.append(path)

        fsq = FSQExtractor(n_levels=[3, 3], encode_fn=mock_encode,
                           model_loader=loader)
        concepts = {'a': {'levels': [0, 1]}}
        path = str(tmp_path / "model_step100.pt")
        fsq.extract(path, concepts)
        assert len(loaded) == 1

    def test_onehot_extraction(self):
        fsq = FSQExtractor(n_levels=[3, 3], encode_fn=mock_encode,
                           binarize='onehot')
        concepts = {'x': {'levels': [2, 0]}}
        snap = fsq.extract("step_0.pt", concepts)
        assert len(snap.codes['x']) == 6
        assert snap.codes['x'][2] == 1  # dim 0, level 2
        assert snap.codes['x'][3] == 1  # dim 1, level 0


# -- Similarity -----------------------------------------------------------

class TestSimilarity:

    def test_identical(self):
        fsq = FSQExtractor(n_levels=[3, 3], encode_fn=mock_encode)
        assert fsq.similarity([1, 0, 1], [1, 0, 1]) == 1.0

    def test_disjoint(self):
        fsq = FSQExtractor(n_levels=[3, 3], encode_fn=mock_encode)
        assert fsq.similarity([1, 0], [0, 1]) == 0.0

    def test_shared_features(self):
        fsq = FSQExtractor(n_levels=[3, 3, 3], encode_fn=mock_encode)
        shared = fsq.shared_features([1, 0, 1], [1, 1, 0])
        assert shared == [0]


# -- Validation -----------------------------------------------------------

class TestValidation:

    def test_invalid_binarize_raises(self):
        with pytest.raises(ValueError, match="binarize"):
            FSQExtractor(n_levels=[3], encode_fn=mock_encode, binarize='invalid')


# -- Imports --------------------------------------------------------------

class TestImport:

    def test_importable_from_extractors(self):
        from reptimeline.extractors import FSQExtractor as F
        assert F is not None

    def test_subclass_of_base(self):
        from reptimeline.extractors.base import RepresentationExtractor
        assert issubclass(FSQExtractor, RepresentationExtractor)
