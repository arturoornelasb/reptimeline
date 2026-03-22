"""Tests for SAEExtractor with mock encode/decode functions."""

import numpy as np
import pytest

from reptimeline.core import ConceptSnapshot
from reptimeline.exceptions import ConfigurationError
from reptimeline.extractors.sae import SAEExtractor


def mock_encode(hidden_state):
    """Mock SAE encoder: returns top_indices and top_acts from hidden state."""
    # hidden_state is a dict with 'indices' and 'acts' keys for testing
    return hidden_state['indices'], hidden_state['acts']


def mock_decode(top_indices, top_acts):
    """Mock SAE decoder: weighted sum of one-hot vectors."""
    result = np.zeros(8)
    for idx, act in zip(np.asarray(top_indices).flatten(),
                        np.asarray(top_acts).flatten()):
        if 0 <= int(idx) < 8:
            result[int(idx)] = act
    return result


# ── Binarization ─────────────────────────────────────────────────

class TestActivationToBinary:

    def test_active_features_set_to_one(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode)
        binary = sae._activation_to_binary(np.array([0, 3, 5]))
        assert binary == [1, 0, 0, 1, 0, 1, 0, 0]

    def test_inactive_features_stay_zero(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode)
        binary = sae._activation_to_binary(np.array([]))
        assert binary == [0] * 8

    def test_feature_indices_subset(self):
        """When feature_indices=[10, 20, 30], bit 0 → SAE feature 10, etc."""
        sae = SAEExtractor(n_features=100, encode_fn=mock_encode,
                           feature_indices=[10, 20, 30])
        binary = sae._activation_to_binary(np.array([10, 30, 99]))
        assert binary == [1, 0, 1]  # SAE 10→bit0, SAE 30→bit2, SAE 99→ignored

    def test_out_of_range_ignored(self):
        sae = SAEExtractor(n_features=4, encode_fn=mock_encode)
        binary = sae._activation_to_binary(np.array([0, 10]))
        assert binary == [1, 0, 0, 0]

    def test_2d_input_flattened(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode)
        binary = sae._activation_to_binary(np.array([[1, 3], [5, 7]]))
        assert binary == [0, 1, 0, 1, 0, 1, 0, 1]


# ── Extract ──────────────────────────────────────────────────────

class TestExtract:

    def test_basic_extraction(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode)
        concepts = {
            'cat': {'indices': np.array([0, 1]), 'acts': np.array([1.0, 0.5])},
            'dog': {'indices': np.array([2, 3]), 'acts': np.array([0.8, 0.3])},
        }
        snap = sae.extract("step_1000.pt", concepts)
        assert isinstance(snap, ConceptSnapshot)
        assert snap.codes['cat'] == [1, 1, 0, 0, 0, 0, 0, 0]
        assert snap.codes['dog'] == [0, 0, 1, 1, 0, 0, 0, 0]
        assert snap.step == 1000

    def test_model_loader_called(self, tmp_path):
        loaded = []
        def loader(path):
            loaded.append(path)

        sae = SAEExtractor(n_features=8, encode_fn=mock_encode,
                           model_loader=loader)
        concepts = {'a': {'indices': np.array([0]), 'acts': np.array([1.0])}}
        path = str(tmp_path / "model_step500.pt")
        sae.extract(path, concepts)
        assert len(loaded) == 1
        assert loaded[0] == path

    def test_feature_indices_mapping_in_extract(self):
        sae = SAEExtractor(n_features=100, encode_fn=mock_encode,
                           feature_indices=[10, 50])
        concepts = {
            'x': {'indices': np.array([10, 50]), 'acts': np.array([1.0, 0.5])},
        }
        snap = sae.extract("step_0.pt", concepts)
        assert snap.codes['x'] == [1, 1]


# ── Intervene ────────────────────────────────────────────────────

class TestIntervene:

    def test_active_feature_perturbs(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode,
                           decode_fn=mock_decode)
        hidden = {'indices': np.array([0, 3, 5]),
                  'acts': np.array([2.0, 1.0, 0.5])}
        perturbation = sae.intervene(hidden, bit_index=0)
        assert perturbation > 0  # zeroing feature 0 should change output

    def test_inactive_feature_returns_zero(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode,
                           decode_fn=mock_decode)
        hidden = {'indices': np.array([0, 3]),
                  'acts': np.array([2.0, 1.0])}
        perturbation = sae.intervene(hidden, bit_index=5)
        assert perturbation == 0.0  # feature 5 not active

    def test_no_decode_fn_raises(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode)
        with pytest.raises(ConfigurationError, match="decode_fn"):
            sae.intervene({'indices': np.array([0]), 'acts': np.array([1.0])},
                          bit_index=0)

    def test_feature_indices_maps_bit_to_sae(self):
        """bit_index=1 maps to SAE feature 3 when feature_indices=[0, 3]."""
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode,
                           decode_fn=mock_decode,
                           feature_indices=[0, 3])
        hidden = {'indices': np.array([0, 3]),
                  'acts': np.array([2.0, 1.0])}
        perturbation = sae.intervene(hidden, bit_index=1)
        assert perturbation > 0  # zeroing SAE feature 3


# ── make_intervene_fn ────────────────────────────────────────────

class TestMakeInterveneFn:

    def test_returns_callable(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode,
                           decode_fn=mock_decode)
        concept_hidden = {
            'cat': {'indices': np.array([0, 3]), 'acts': np.array([2.0, 1.0])},
        }
        fn = sae.make_intervene_fn(concept_hidden)
        assert callable(fn)

    def test_callable_invokes_intervene(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode,
                           decode_fn=mock_decode)
        concept_hidden = {
            'cat': {'indices': np.array([0, 3]), 'acts': np.array([2.0, 1.0])},
        }
        fn = sae.make_intervene_fn(concept_hidden)
        result = fn('cat', 0)
        assert result > 0


# ── Similarity ───────────────────────────────────────────────────

class TestSimilarity:

    def test_jaccard_identical(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode)
        assert sae.similarity([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_jaccard_disjoint(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode)
        assert sae.similarity([1, 0, 0, 0], [0, 0, 0, 1]) == 0.0

    def test_shared_features(self):
        sae = SAEExtractor(n_features=8, encode_fn=mock_encode)
        shared = sae.shared_features([1, 1, 0, 1], [1, 0, 0, 1])
        assert shared == [0, 3]


# ── Imports ──────────────────────────────────────────────────────

class TestImport:

    def test_importable_from_extractors(self):
        from reptimeline.extractors.sae import SAEExtractor as S
        assert S is not None

    def test_importable_from_base(self):
        from reptimeline.extractors.base import RepresentationExtractor
        assert issubclass(SAEExtractor, RepresentationExtractor)
