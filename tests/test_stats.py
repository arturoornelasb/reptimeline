"""Tests for reptimeline.stats — numpy-only statistical utilities."""

import numpy as np
import pytest

from reptimeline.stats import (
    BootstrapResult,
    bootstrap_ci,
    permutation_test,
    benjamini_hochberg,
    effect_size_cohens_d,
    selectivity_ratio,
)


# ── bootstrap_ci ─────────────────────────────────────────────────

class TestBootstrapCI:

    def test_identical_groups_ratio_near_one(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = bootstrap_ci(a, b, selectivity_ratio, seed=42)
        assert 0.5 < result.observed < 2.0
        assert isinstance(result, BootstrapResult)

    def test_different_groups_ci_excludes_one(self):
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 1.5, 2.0, 1.2, 0.8])
        result = bootstrap_ci(a, b, selectivity_ratio, seed=42)
        assert result.ci_low > 1.0
        assert result.observed > 5.0

    def test_deterministic_with_seed(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        r1 = bootstrap_ci(a, b, selectivity_ratio, seed=42)
        r2 = bootstrap_ci(a, b, selectivity_ratio, seed=42)
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_custom_statistic_fn(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([1.0, 2.0, 3.0])
        result = bootstrap_ci(a, b, lambda x, y: np.mean(x) - np.mean(y), seed=0)
        assert result.observed == pytest.approx(18.0)
        assert result.ci_low > 0

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]), np.array([1.0]), selectivity_ratio)
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([1.0]), np.array([]), selectivity_ratio)


# ── permutation_test ─────────────────────────────────────────────

class TestPermutationTest:

    def test_identical_groups_nonsignificant(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = permutation_test(a, b, lambda x, y: np.mean(x) - np.mean(y),
                             n_perms=500, seed=42)
        assert p > 0.3  # very non-significant

    def test_different_groups_significant(self):
        a = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = permutation_test(a, b, lambda x, y: np.mean(x) - np.mean(y),
                             n_perms=500, seed=42)
        assert p < 0.01

    def test_p_value_in_valid_range(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        p = permutation_test(a, b, lambda x, y: np.mean(x) - np.mean(y),
                             n_perms=100, seed=0)
        assert 0 < p <= 1.0

    def test_deterministic_with_seed(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        p1 = permutation_test(a, b, lambda x, y: np.mean(x) - np.mean(y), seed=42)
        p2 = permutation_test(a, b, lambda x, y: np.mean(x) - np.mean(y), seed=42)
        assert p1 == p2

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            permutation_test(np.array([]), np.array([1.0]),
                             lambda x, y: np.mean(x) - np.mean(y))


# ── benjamini_hochberg ───────────────────────────────────────────

class TestBenjaminiHochberg:

    def test_all_below_alpha(self):
        p = np.array([0.001, 0.002, 0.003])
        sig = benjamini_hochberg(p, alpha=0.05)
        assert sig.all()

    def test_all_above_alpha(self):
        p = np.array([0.5, 0.6, 0.7, 0.8])
        sig = benjamini_hochberg(p, alpha=0.05)
        assert not sig.any()

    def test_mixed_p_values(self):
        p = np.array([0.001, 0.01, 0.5, 0.9])
        sig = benjamini_hochberg(p, alpha=0.05)
        assert sig[0] and sig[1]
        assert not sig[3]

    def test_single_p_value_below(self):
        sig = benjamini_hochberg(np.array([0.01]), alpha=0.05)
        assert sig[0]

    def test_single_p_value_above(self):
        sig = benjamini_hochberg(np.array([0.1]), alpha=0.05)
        assert not sig[0]

    def test_empty_array(self):
        sig = benjamini_hochberg(np.array([]), alpha=0.05)
        assert len(sig) == 0

    def test_textbook_example(self):
        # BH at alpha=0.05 with 5 tests
        # Thresholds: 0.01, 0.02, 0.03, 0.04, 0.05
        p = np.array([0.005, 0.015, 0.025, 0.06, 0.1])
        sig = benjamini_hochberg(p, alpha=0.05)
        assert sig[0] and sig[1] and sig[2]
        assert not sig[3] and not sig[4]


# ── effect_size_cohens_d ─────────────────────────────────────────

class TestEffectSize:

    def test_identical_groups_zero(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        d = effect_size_cohens_d(a, b)
        assert d == pytest.approx(0.0)

    def test_known_effect(self):
        a = np.array([9.0, 10.0, 11.0, 10.0])
        b = np.array([0.0, 1.0, 0.5, 0.2])
        d = effect_size_cohens_d(a, b)
        assert d > 5.0  # very large effect

    def test_direction(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        d = effect_size_cohens_d(a, b)
        assert d < 0  # a < b → negative

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            effect_size_cohens_d(np.array([]), np.array([1.0]))


# ── selectivity_ratio ────────────────────────────────────────────

class TestSelectivityRatio:

    def test_equal_means(self):
        a = np.array([2.0, 4.0])
        b = np.array([2.0, 4.0])
        assert selectivity_ratio(a, b) == pytest.approx(1.0)

    def test_labeled_higher(self):
        a = np.array([10.0, 12.0])
        b = np.array([2.0, 3.0])
        r = selectivity_ratio(a, b)
        assert r > 3.0

    def test_zero_denominator(self):
        a = np.array([5.0, 6.0])
        b = np.array([0.0, 0.0])
        assert selectivity_ratio(a, b) == 999.0

    def test_both_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        assert selectivity_ratio(a, b) == 0.0
