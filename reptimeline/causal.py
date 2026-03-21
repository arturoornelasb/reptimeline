"""CausalVerifier — Test whether discrete representations have causal effects.

The user provides an intervention function that perturbs one feature for one
concept and returns a scalar effect metric (e.g., L2 perturbation, KL divergence).
CausalVerifier runs it across all concepts, computes selectivity per bit,
and applies statistical testing with BH-FDR correction.

Usage:
    verifier = CausalVerifier(intervene_fn=my_fn)
    report = verifier.verify(snapshot)
    verifier.print_report(report)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from reptimeline.core import ConceptSnapshot
from reptimeline.stats import (
    BootstrapResult, bootstrap_ci, permutation_test,
    benjamini_hochberg, effect_size_cohens_d, selectivity_ratio,
)


@dataclass
class BitCausalResult:
    """Causal verification result for a single bit."""
    bit_index: int
    selectivity: float
    bootstrap: Optional[BootstrapResult]
    p_value: float
    significant: bool
    effect_size: float
    mean_labeled: float
    mean_other: float
    n_labeled: int
    n_other: int
    top_affected: List[Tuple[str, float]]


@dataclass
class CausalReport:
    """Complete causal verification result."""
    bit_results: List[BitCausalResult]
    n_tested: int
    n_significant: int
    verdict: str
    correction_method: str
    alpha: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalVerifier:
    """Tests whether bits in a discrete code have selective causal effects.

    Args:
        intervene_fn: Callable(concept: str, bit_index: int) -> float.
            Perturbs one feature for one concept, returns effect magnitude.
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
        n_perms: Number of permutations for permutation test.
        alpha: Significance level for BH-FDR correction.
        selectivity_threshold: Minimum selectivity ratio to count as selective.
        min_selective_bits: Minimum significant selective bits for positive verdict.
        seed: Random seed for reproducibility.
        top_k: Number of top-affected concepts to record per bit.
    """

    def __init__(
        self,
        intervene_fn: Callable[[str, int], float],
        n_bootstrap: int = 1000,
        n_perms: int = 1000,
        alpha: float = 0.05,
        selectivity_threshold: float = 1.5,
        min_selective_bits: int = 3,
        seed: int = 42,
        top_k: int = 5,
    ):
        self.intervene_fn = intervene_fn
        self.n_bootstrap = n_bootstrap
        self.n_perms = n_perms
        self.alpha = alpha
        self.selectivity_threshold = selectivity_threshold
        self.min_selective_bits = min_selective_bits
        self.seed = seed
        self.top_k = top_k

    def verify(
        self,
        snapshot: ConceptSnapshot,
        bit_indices: Optional[List[int]] = None,
    ) -> CausalReport:
        """Run causal verification on a snapshot.

        Args:
            snapshot: ConceptSnapshot with codes for all concepts.
            bit_indices: Which bits to test. If None, auto-selects bits
                with activation rate between 0.03 and 0.97.
        """
        concepts = list(snapshot.codes.keys())
        n_bits = snapshot.code_dim
        codes_matrix = np.array([snapshot.codes[c] for c in concepts])

        if bit_indices is None:
            rates = codes_matrix.mean(axis=0)
            bit_indices = [i for i in range(n_bits)
                           if 0.03 < rates[i] < 0.97]

        bit_results = []
        p_values = []

        for bit_idx in bit_indices:
            column = codes_matrix[:, bit_idx]
            labeled_idx = [i for i in range(len(concepts)) if column[i] == 1]
            other_idx = [i for i in range(len(concepts)) if column[i] == 0]

            # Compute intervention effects for all concepts
            effects = {}
            for i, concept in enumerate(concepts):
                effects[concept] = self.intervene_fn(concept, bit_idx)

            labeled_vals = np.array([effects[concepts[i]] for i in labeled_idx])
            other_vals = np.array([effects[concepts[i]] for i in other_idx])

            sel = selectivity_ratio(labeled_vals, other_vals)

            # Statistical tests (need at least 2 samples per group)
            boot = None
            p_val = 1.0
            eff_size = 0.0

            if len(labeled_vals) >= 2 and len(other_vals) >= 2:
                boot = bootstrap_ci(
                    labeled_vals, other_vals, selectivity_ratio,
                    n_bootstrap=self.n_bootstrap, seed=self.seed,
                )
                p_val = permutation_test(
                    labeled_vals, other_vals,
                    lambda a, b: np.mean(a) - np.mean(b),
                    n_perms=self.n_perms, seed=self.seed,
                )
                eff_size = effect_size_cohens_d(labeled_vals, other_vals)

            p_values.append(p_val)

            # Top affected concepts
            sorted_effects = sorted(effects.items(), key=lambda x: x[1], reverse=True)
            top_affected = sorted_effects[:self.top_k]

            bit_results.append(BitCausalResult(
                bit_index=bit_idx,
                selectivity=sel,
                bootstrap=boot,
                p_value=p_val,
                significant=False,  # set after BH
                effect_size=eff_size,
                mean_labeled=float(np.mean(labeled_vals)) if len(labeled_vals) > 0 else 0.0,
                mean_other=float(np.mean(other_vals)) if len(other_vals) > 0 else 0.0,
                n_labeled=len(labeled_vals),
                n_other=len(other_vals),
                top_affected=top_affected,
            ))

        # BH-FDR correction
        if p_values:
            significant = benjamini_hochberg(np.array(p_values), alpha=self.alpha)
            for br, sig in zip(bit_results, significant):
                br.significant = bool(sig)

        n_significant = sum(
            1 for br in bit_results
            if br.significant and br.selectivity >= self.selectivity_threshold
        )

        verdict = ('causal_evidence_found'
                    if n_significant >= self.min_selective_bits
                    else 'insufficient_evidence')

        return CausalReport(
            bit_results=bit_results,
            n_tested=len(bit_indices),
            n_significant=n_significant,
            verdict=verdict,
            correction_method='benjamini_hochberg',
            alpha=self.alpha,
            metadata={
                'n_concepts': len(concepts),
                'n_bits': n_bits,
                'selectivity_threshold': self.selectivity_threshold,
                'min_selective_bits': self.min_selective_bits,
            },
        )

    def print_report(self, report: CausalReport) -> None:
        """Print causal verification results."""
        print()
        print("=" * 60)
        print("  CAUSAL VERIFICATION REPORT")
        print("=" * 60)
        print(f"  Bits tested:      {report.n_tested}")
        print(f"  Significant:      {report.n_significant}")
        print(f"  Correction:       {report.correction_method} (alpha={report.alpha})")
        print(f"  Verdict:          {report.verdict.upper()}")
        print()

        selective = [br for br in report.bit_results
                     if br.significant and br.selectivity >= self.selectivity_threshold]
        if selective:
            print("  SELECTIVE BITS (significant + above threshold)")
            print("  " + "-" * 56)
            for br in sorted(selective, key=lambda x: x.selectivity, reverse=True):
                ci_str = ""
                if br.bootstrap:
                    ci_str = f"  CI=[{br.bootstrap.ci_low:.2f}, {br.bootstrap.ci_high:.2f}]"
                print(f"    bit {br.bit_index:>2d}  sel={br.selectivity:.1f}x"
                      f"  p={br.p_value:.4f}  d={br.effect_size:.2f}{ci_str}")
            print()

        non_selective = [br for br in report.bit_results if br not in selective]
        if non_selective:
            n_insig = sum(1 for br in non_selective if not br.significant)
            n_low_sel = sum(1 for br in non_selective
                           if br.significant and br.selectivity < self.selectivity_threshold)
            print(f"  Non-selective: {n_insig} not significant, "
                  f"{n_low_sel} below threshold")

        print("=" * 60)
