"""Statistical utilities for reptimeline — numpy only, no scipy.

Provides:
  - bootstrap_ci: Bootstrap confidence intervals for any two-sample statistic
  - permutation_test: Permutation test for two-sample comparison
  - benjamini_hochberg: FDR correction for multiple comparisons
  - effect_size_cohens_d: Cohen's d effect size
  - selectivity_ratio: Mean ratio with division-by-zero protection
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BootstrapResult:
    """Result of a bootstrap confidence interval computation."""
    observed: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


def bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Compute bootstrap confidence interval for a two-sample statistic.

    Args:
        values_a: First sample (e.g., labeled group effects).
        values_b: Second sample (e.g., other group effects).
        statistic_fn: Function(a, b) -> float computing the statistic.
        n_bootstrap: Number of resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        BootstrapResult with observed value and CI bounds.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) == 0 or len(values_b) == 0:
        raise ValueError("Both samples must be non-empty")

    rng = np.random.RandomState(seed)
    observed = statistic_fn(values_a, values_b)

    stats = []
    for _ in range(n_bootstrap):
        a_sample = rng.choice(values_a, size=len(values_a), replace=True)
        b_sample = rng.choice(values_b, size=len(values_b), replace=True)
        stats.append(statistic_fn(a_sample, b_sample))

    stats_arr = np.array(stats)
    alpha = 1.0 - ci
    return BootstrapResult(
        observed=observed,
        ci_low=float(np.percentile(stats_arr, 100 * alpha / 2)),
        ci_high=float(np.percentile(stats_arr, 100 * (1 - alpha / 2))),
        n_bootstrap=n_bootstrap,
    )


def permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_perms: int = 1000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test.

    Pools values_a and values_b, reshuffles, recomputes statistic_fn,
    and returns the fraction of permuted statistics >= observed.

    Args:
        values_a: First sample.
        values_b: Second sample.
        statistic_fn: Function(a, b) -> float.
        n_perms: Number of permutations.
        seed: Random seed.

    Returns:
        p-value in [1/(n_perms+1), 1.0].
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) == 0 or len(values_b) == 0:
        raise ValueError("Both samples must be non-empty")

    rng = np.random.RandomState(seed)
    observed = abs(statistic_fn(values_a, values_b))

    pooled = np.concatenate([values_a, values_b])
    n_a = len(values_a)
    count_ge = 0

    for _ in range(n_perms):
        perm = rng.permutation(pooled)
        perm_stat = abs(statistic_fn(perm[:n_a], perm[n_a:]))
        if perm_stat >= observed:
            count_ge += 1

    return (count_ge + 1) / (n_perms + 1)


def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: Array of p-values.
        alpha: Significance level.

    Returns:
        Boolean array where True = significant after correction.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    thresholds = alpha * np.arange(1, m + 1) / m

    passing = sorted_p <= thresholds
    if not np.any(passing):
        return np.zeros(m, dtype=bool)

    max_k = int(np.max(np.where(passing)[0]))
    significant = np.zeros(m, dtype=bool)
    significant[sorted_idx[:max_k + 1]] = True
    return significant


def effect_size_cohens_d(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> float:
    """Cohen's d effect size for two independent samples.

    Uses pooled standard deviation.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) == 0 or len(values_b) == 0:
        raise ValueError("Both samples must be non-empty")

    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    n_a = len(values_a)
    n_b = len(values_b)

    var_a = np.var(values_a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(values_b, ddof=1) if n_b > 1 else 0.0

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std < 1e-12:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


def selectivity_ratio(
    labeled_values: np.ndarray,
    other_values: np.ndarray,
) -> float:
    """Compute selectivity ratio: mean(labeled) / mean(other).

    Handles division by zero:
      - Both near zero → 0.0
      - Only denominator near zero → 999.0 (large sentinel)
    """
    labeled_values = np.asarray(labeled_values, dtype=float)
    other_values = np.asarray(other_values, dtype=float)

    mean_l = float(np.mean(labeled_values)) if len(labeled_values) > 0 else 0.0
    mean_o = float(np.mean(other_values)) if len(other_values) > 0 else 0.0

    if abs(mean_o) < 1e-8:
        if abs(mean_l) < 1e-8:
            return 0.0
        return 999.0

    return mean_l / mean_o
