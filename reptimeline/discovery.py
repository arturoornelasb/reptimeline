"""
BitDiscovery — Discover what each bit "means" from unsupervised training.

Instead of pre-defining primitives and supervising, this module:
1. Takes a trained model (no anchor supervision needed)
2. Runs a large concept vocabulary through it
3. Analyzes which concepts activate which bits
4. Discovers: bit semantics, hierarchy, duals, dependencies

This enables bottom-up primitive discovery: the model invents its own
ontology and reptimeline discovers what it is.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from reptimeline.core import ConceptSnapshot, Timeline


@dataclass
class BitSemantics:
    """What a single bit "means" based on what concepts activate it."""
    bit_index: int
    activation_rate: float  # fraction of concepts that activate this bit
    top_concepts: List[str]  # concepts most associated with this bit
    anti_concepts: List[str]  # concepts that never activate this bit
    label: str = ""  # auto-generated semantic label


@dataclass
class DiscoveredDual:
    """A pair of bits that behave as opposites (anti-correlated)."""
    bit_a: int
    bit_b: int
    anti_correlation: float  # -1 = perfect opposites, 0 = independent
    concepts_exclusive: int  # times exactly one is active
    concepts_both: int  # times both are active (should be rare)


@dataclass
class DiscoveredDependency:
    """Bit B almost never activates without bit A being active first."""
    bit_parent: int
    bit_child: int
    confidence: float  # P(parent=1 | child=1)
    support: int  # how many concepts show this pattern


@dataclass
class DiscoveredTriadicDep:
    """A 3-way interaction: bit r activates only when bits i AND j are both active.

    This is an AND-gate in semantic space: neither i alone nor j alone
    predicts r, but their conjunction does. Analogous to epistasis in genetics.
    """
    bit_i: int
    bit_j: int
    bit_r: int  # the emergent bit
    p_r_given_ij: float  # P(r=1 | i=1, j=1) — should be high
    p_r_given_i: float   # P(r=1 | i=1) — should be low
    p_r_given_j: float   # P(r=1 | j=1) — should be low
    interaction_strength: float  # p_r_given_ij - max(p_r_given_i, p_r_given_j)
    support: int  # how many concepts have i=1 AND j=1


@dataclass
class DiscoveredHierarchy:
    """Bits ordered by when they first stabilize during training."""
    bit_index: int
    first_stable_step: Optional[int]  # first step where meaning stabilizes
    layer: int  # discovered layer (1=earliest, N=latest)
    n_dependents: int  # how many other bits depend on this one


@dataclass
class DiscoveryReport:
    """Complete bottom-up discovery of what a model learned."""
    bit_semantics: List[BitSemantics]
    discovered_duals: List[DiscoveredDual]
    discovered_deps: List[DiscoveredDependency]
    discovered_triadic_deps: List[DiscoveredTriadicDep]
    discovered_hierarchy: List[DiscoveredHierarchy]
    n_active_bits: int  # bits with activation_rate > threshold
    n_dead_bits: int  # bits that almost never activate
    metadata: Dict = field(default_factory=dict)


class BitDiscovery:
    """Discovers what each bit encodes without prior knowledge of primitives.

    This is the inverse of PrimitiveOverlay: instead of mapping known
    primitives onto bits, it discovers what the bits mean by analyzing
    activation patterns across a large concept vocabulary.
    """

    def __init__(self, dead_threshold: float = 0.02,
                 dual_threshold: float = -0.3,
                 dep_confidence: float = 0.9,
                 triadic_threshold: float = 0.7,
                 triadic_min_interaction: float = 0.2,
                 apply_correction: bool = True):
        """
        Args:
            dead_threshold: Bits with activation rate below this are "dead".
            dual_threshold: Correlation below this counts as a dual pair.
            dep_confidence: P(parent|child) above this counts as dependency.
            triadic_threshold: P(r|i,j) above this for triadic detection.
            triadic_min_interaction: Minimum interaction strength
                (P(r|i,j) - max(P(r|i), P(r|j))).
            apply_correction: Apply multiple comparison correction
                (Bonferroni for duals/deps, BH for triadic).
        """
        self.dead_threshold = dead_threshold
        self.dual_threshold = dual_threshold
        self.dep_confidence = dep_confidence
        self.triadic_threshold = triadic_threshold
        self.triadic_min_interaction = triadic_min_interaction
        self.apply_correction = apply_correction

    def discover(self, snapshot: ConceptSnapshot,
                 timeline: Optional[Timeline] = None,
                 top_k: int = 10) -> DiscoveryReport:
        """Discover bit semantics from a single snapshot (or a full timeline).

        Args:
            snapshot: A ConceptSnapshot with codes for many concepts.
                Use the LAST snapshot from training for best results.
            timeline: Optional full timeline for hierarchy discovery.
            top_k: Number of top/anti concepts per bit.
        """
        concepts = list(snapshot.codes.keys())
        if not concepts:
            raise ValueError("Snapshot has no concepts")

        n_bits = snapshot.code_dim
        codes_matrix = np.array([snapshot.codes[c] for c in concepts])

        bit_semantics = self._discover_semantics(
            codes_matrix, concepts, n_bits, top_k
        )
        duals = self._discover_duals(codes_matrix, n_bits)
        deps = self._discover_dependencies(codes_matrix, n_bits)
        triadic_deps = self._discover_triadic_deps(codes_matrix, n_bits)

        hierarchy = []
        if timeline is not None:
            hierarchy = self._discover_hierarchy(timeline, n_bits, deps)

        n_active = sum(1 for bs in bit_semantics
                       if bs.activation_rate > self.dead_threshold)

        return DiscoveryReport(
            bit_semantics=bit_semantics,
            discovered_duals=duals,
            discovered_deps=deps,
            discovered_triadic_deps=triadic_deps,
            discovered_hierarchy=hierarchy,
            n_active_bits=n_active,
            n_dead_bits=n_bits - n_active,
            metadata={
                'n_concepts': len(concepts),
                'n_bits': n_bits,
                'correction_applied': self.apply_correction,
                'correction_method': (
                    'bonferroni_duals_deps__bh_triadic' if self.apply_correction else 'none'
                ),
            },
        )

    # ------------------------------------------------------------------
    # Bit semantics: what does each bit mean?
    # ------------------------------------------------------------------

    def _discover_semantics(self, codes: np.ndarray, concepts: List[str],
                            n_bits: int, top_k: int) -> List[BitSemantics]:
        """For each bit, find which concepts activate it most/least."""
        results = []
        n_concepts = len(concepts)

        for bit_idx in range(n_bits):
            column = codes[:, bit_idx]
            activation_rate = float(column.mean())

            # Concepts where this bit is active
            active_mask = column == 1
            active_concepts = [concepts[i] for i in range(n_concepts)
                               if active_mask[i]]
            inactive_concepts = [concepts[i] for i in range(n_concepts)
                                 if not active_mask[i]]

            # Top concepts: those where this bit is active (limited to top_k)
            top = active_concepts[:top_k]
            anti = inactive_concepts[:top_k]

            # Auto-label: common theme among top concepts (placeholder)
            label = f"bit_{bit_idx}"
            if activation_rate < self.dead_threshold:
                label = f"bit_{bit_idx}_DEAD"

            results.append(BitSemantics(
                bit_index=bit_idx,
                activation_rate=activation_rate,
                top_concepts=top,
                anti_concepts=anti,
                label=label,
            ))
        return results

    # ------------------------------------------------------------------
    # Dual discovery: which bits are opposites?
    # ------------------------------------------------------------------

    def _discover_duals(self, codes: np.ndarray,
                        n_bits: int) -> List[DiscoveredDual]:
        """Find bit pairs that anti-correlate (mutual exclusion)."""
        # Use numpy's corrcoef for correct Pearson correlation
        # Handle constant columns (dead bits) gracefully
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(codes.T)
        corr = np.nan_to_num(corr, nan=0.0)

        # Bonferroni correction: tighten threshold for many comparisons
        threshold = self.dual_threshold
        n_tests = n_bits * (n_bits - 1) // 2
        if self.apply_correction and n_tests > 1:
            correction_factor = min(1.0, 1.0 + 0.1 * np.log10(n_tests))
            threshold = self.dual_threshold * correction_factor

        duals = []
        seen = set()
        for i in range(n_bits):
            for j in range(i + 1, n_bits):
                if corr[i, j] < threshold:
                    key = (min(i, j), max(i, j))
                    if key not in seen:
                        seen.add(key)
                        both = int(((codes[:, i] == 1) & (codes[:, j] == 1)).sum())
                        excl = int(((codes[:, i] == 1) ^ (codes[:, j] == 1)).sum())
                        duals.append(DiscoveredDual(
                            bit_a=i, bit_b=j,
                            anti_correlation=float(corr[i, j]),
                            concepts_exclusive=excl,
                            concepts_both=both,
                        ))

        duals.sort(key=lambda d: d.anti_correlation)
        return duals

    # ------------------------------------------------------------------
    # Dependency discovery: which bits require other bits?
    # ------------------------------------------------------------------

    def _discover_dependencies(self, codes: np.ndarray,
                               n_bits: int) -> List[DiscoveredDependency]:
        """Find bit pairs where child almost never activates without parent."""
        n_concepts = codes.shape[0]
        min_samples = max(3, int(np.sqrt(n_concepts)))

        # Bonferroni correction on confidence threshold
        dep_threshold = self.dep_confidence
        n_tests = n_bits * (n_bits - 1)
        if self.apply_correction and n_tests > 1:
            dep_threshold = min(0.99, self.dep_confidence + 0.01 * np.log10(n_tests))

        deps = []
        for child in range(n_bits):
            child_active = codes[:, child] == 1
            n_child = int(child_active.sum())
            if n_child < min_samples:
                continue

            for parent in range(n_bits):
                if parent == child:
                    continue
                parent_active = codes[:, parent] == 1
                # P(parent=1 | child=1)
                both = int((child_active & parent_active).sum())
                confidence = both / n_child

                if confidence >= dep_threshold:
                    deps.append(DiscoveredDependency(
                        bit_parent=parent,
                        bit_child=child,
                        confidence=confidence,
                        support=n_child,
                    ))

        deps.sort(key=lambda d: d.confidence, reverse=True)
        return deps

    # ------------------------------------------------------------------
    # Triadic dependency discovery: 3-way interactions
    # ------------------------------------------------------------------

    def _discover_triadic_deps(self, codes: np.ndarray,
                                n_bits: int,
                                min_support: int = 3,
                                ) -> List[DiscoveredTriadicDep]:
        """Find 3-way interactions: bit r activates when i AND j together,
        but not when either is active alone.

        For all triples (i, j, r):
            P(r=1 | i=1, j=1) > triadic_threshold
            P(r=1 | i=1)      < triadic_threshold
            P(r=1 | j=1)      < triadic_threshold
            interaction = P(r|i,j) - max(P(r|i), P(r|j)) > min_interaction

        Complexity: O(K^2 * K) where K = active bits. With 37 active bits:
        ~23,000 triples — runs in seconds.
        """
        triadic = []

        # Pre-compute active masks and counts for all bits
        active_masks = []
        active_counts = []
        for b in range(n_bits):
            mask = codes[:, b] == 1
            active_masks.append(mask)
            active_counts.append(int(mask.sum()))

        from tqdm import tqdm

        active_bits = [i for i in range(n_bits) if active_counts[i] >= min_support]
        for i in tqdm(active_bits, desc="Triadic discovery", unit="bit"):
            i_mask = active_masks[i]

            for j in range(i + 1, n_bits):
                if active_counts[j] < min_support:
                    continue
                j_mask = active_masks[j]

                # Conjunction mask: both i and j active
                ij_mask = i_mask & j_mask
                n_ij = int(ij_mask.sum())
                if n_ij < min_support:
                    continue

                for r in range(n_bits):
                    if r == i or r == j:
                        continue
                    if active_counts[r] < min_support:
                        continue

                    r_mask = active_masks[r]

                    # P(r|i,j)
                    p_r_ij = int((ij_mask & r_mask).sum()) / n_ij
                    if p_r_ij < self.triadic_threshold:
                        continue

                    # P(r|i) and P(r|j)
                    p_r_i = int((i_mask & r_mask).sum()) / active_counts[i]
                    p_r_j = int((j_mask & r_mask).sum()) / active_counts[j]

                    # Both must be below threshold individually
                    if p_r_i >= self.triadic_threshold or p_r_j >= self.triadic_threshold:
                        continue

                    interaction = p_r_ij - max(p_r_i, p_r_j)
                    if interaction < self.triadic_min_interaction:
                        continue

                    triadic.append(DiscoveredTriadicDep(
                        bit_i=i, bit_j=j, bit_r=r,
                        p_r_given_ij=p_r_ij,
                        p_r_given_i=p_r_i,
                        p_r_given_j=p_r_j,
                        interaction_strength=interaction,
                        support=n_ij,
                    ))

        triadic.sort(key=lambda t: t.interaction_strength, reverse=True)

        # BH-FDR correction via permutation p-values
        if self.apply_correction and triadic:
            from reptimeline.stats import benjamini_hochberg
            rng = np.random.RandomState(42)
            n_perms = 200
            p_values = []
            for td in triadic:
                observed = td.interaction_strength
                count_ge = 0
                for _ in range(n_perms):
                    perm_r = rng.permutation(codes[:, td.bit_r])
                    n_ij = int((active_masks[td.bit_i] & active_masks[td.bit_j]).sum())
                    if n_ij == 0:
                        count_ge += 1
                        continue
                    mask_ij = active_masks[td.bit_i] & active_masks[td.bit_j]
                    p_r_ij_perm = perm_r[mask_ij].sum() / n_ij
                    p_r_i_perm = perm_r[active_masks[td.bit_i]].sum() / active_counts[td.bit_i]
                    p_r_j_perm = perm_r[active_masks[td.bit_j]].sum() / active_counts[td.bit_j]
                    perm_interaction = p_r_ij_perm - max(p_r_i_perm, p_r_j_perm)
                    if perm_interaction >= observed:
                        count_ge += 1
                p_values.append((count_ge + 1) / (n_perms + 1))

            significant = benjamini_hochberg(np.array(p_values), alpha=0.05)
            triadic = [t for t, sig in zip(triadic, significant) if sig]

        return triadic

    # ------------------------------------------------------------------
    # Hierarchy: which bits stabilize first?
    # ------------------------------------------------------------------

    def _discover_hierarchy(self, timeline: Timeline, n_bits: int,
                            deps: List[DiscoveredDependency],
                            ) -> List[DiscoveredHierarchy]:
        """Order bits by when they first stabilize during training."""
        # Stability: first step where a bit's meaning doesn't change
        # for stability_window consecutive steps
        first_stable = {}
        window = 3

        for bit_idx in range(n_bits):
            consecutive_stable = 0
            for t in range(1, len(timeline.snapshots)):
                prev_snap = timeline.snapshots[t - 1]
                curr_snap = timeline.snapshots[t]

                # Check if this bit's activation pattern is the same
                changed = False
                for concept in curr_snap.concepts:
                    prev_code = prev_snap.codes.get(concept)
                    curr_code = curr_snap.codes.get(concept)
                    if prev_code and curr_code:
                        if (bit_idx < len(prev_code) and bit_idx < len(curr_code)
                                and prev_code[bit_idx] != curr_code[bit_idx]):
                            changed = True
                            break

                if not changed:
                    consecutive_stable += 1
                    if consecutive_stable >= window and bit_idx not in first_stable:
                        first_stable[bit_idx] = timeline.steps[t]
                else:
                    consecutive_stable = 0

        # Count dependents per bit
        n_dependents: Dict[int, int] = defaultdict(int)
        for dep in deps:
            n_dependents[dep.bit_parent] += 1

        # Assign layers by stability order
        stable_bits = sorted(first_stable.items(), key=lambda x: x[1])
        if stable_bits:
            steps_sorted = sorted(set(s for _, s in stable_bits))
            step_to_layer = {s: i + 1 for i, s in enumerate(steps_sorted)}
        else:
            step_to_layer = {}

        results = []
        for bit_idx in range(n_bits):
            step = first_stable.get(bit_idx)
            layer = step_to_layer.get(step, 0) if step else 0
            results.append(DiscoveredHierarchy(
                bit_index=bit_idx,
                first_stable_step=step,
                layer=layer,
                n_dependents=n_dependents[bit_idx],
            ))

        results.sort(key=lambda h: (h.first_stable_step or float('inf')))
        return results

    # ------------------------------------------------------------------
    # Pretty print
    # ------------------------------------------------------------------

    def print_report(self, report: DiscoveryReport):
        """Print discovery results."""
        print()
        print("=" * 60)
        print("  BIT DISCOVERY REPORT")
        print("=" * 60)
        print(f"  Concepts analyzed: {report.metadata.get('n_concepts', 0)}")
        print(f"  Total bits:        {report.metadata.get('n_bits', 0)}")
        print(f"  Active bits:       {report.n_active_bits}")
        print(f"  Dead bits:         {report.n_dead_bits}")
        print()

        # Top active bits with their concepts
        active_bits = [bs for bs in report.bit_semantics
                       if bs.activation_rate > self.dead_threshold]
        active_bits.sort(key=lambda b: b.activation_rate, reverse=True)

        print("  MOST ACTIVE BITS (what activates most concepts)")
        print("  " + "-" * 56)
        for bs in active_bits[:10]:
            concepts_str = ", ".join(bs.top_concepts[:5])
            print(f"    bit {bs.bit_index:>2d}  rate={bs.activation_rate:.2f}"
                  f"  [{concepts_str}]")
        print()

        # Discovered duals
        if report.discovered_duals:
            print(f"  DISCOVERED DUALS ({len(report.discovered_duals)} pairs)")
            print("  " + "-" * 56)
            for dual in report.discovered_duals[:10]:
                print(f"    bit {dual.bit_a:>2d} <-> bit {dual.bit_b:>2d}"
                      f"  corr={dual.anti_correlation:+.3f}"
                      f"  (excl={dual.concepts_exclusive},"
                      f" both={dual.concepts_both})")
            print()

        # Discovered dependencies
        if report.discovered_deps:
            print(f"  DISCOVERED DEPENDENCIES ({len(report.discovered_deps)} edges)")
            print("  " + "-" * 56)
            for dep in report.discovered_deps[:15]:
                print(f"    bit {dep.bit_parent:>2d} -> bit {dep.bit_child:>2d}"
                      f"  P(parent|child)={dep.confidence:.2f}"
                      f"  support={dep.support}")
            print()

        # Triadic dependencies
        if report.discovered_triadic_deps:
            print(f"  TRIADIC INTERACTIONS ({len(report.discovered_triadic_deps)} triples)")
            print("  " + "-" * 56)
            for td in report.discovered_triadic_deps[:15]:
                print(f"    bit {td.bit_i:>2d} + bit {td.bit_j:>2d} -> bit {td.bit_r:>2d}"
                      f"  P(r|i,j)={td.p_r_given_ij:.2f}"
                      f"  P(r|i)={td.p_r_given_i:.2f}"
                      f"  P(r|j)={td.p_r_given_j:.2f}"
                      f"  strength={td.interaction_strength:.2f}"
                      f"  n={td.support}")
            print()

        # Hierarchy
        if report.discovered_hierarchy:
            layers = defaultdict(list)
            for h in report.discovered_hierarchy:
                if h.layer > 0:
                    layers[h.layer].append(h)

            if layers:
                print(f"  DISCOVERED HIERARCHY ({len(layers)} layers)")
                print("  " + "-" * 56)
                for layer_num in sorted(layers.keys())[:8]:
                    bits_in_layer = layers[layer_num]
                    bit_ids = [str(h.bit_index) for h in bits_in_layer[:8]]
                    step = bits_in_layer[0].first_stable_step
                    print(f"    Layer {layer_num}: bits [{', '.join(bit_ids)}]"
                          f"  stable at step {step:,}")
                print()

        print("=" * 60)

    # ------------------------------------------------------------------
    # Null baseline: expected false positives from random data
    # ------------------------------------------------------------------

    def null_baseline(self, n_concepts: int, n_bits: int,
                      n_trials: int = 10, seed: int = 42) -> Dict:
        """Run discovery on random binary codes to estimate false positive rates.

        Returns dict with mean counts of duals, deps, and triadic deps
        found in random data.
        """
        rng = np.random.RandomState(seed)
        dual_counts, dep_counts, triadic_counts = [], [], []
        for _ in range(n_trials):
            random_codes = rng.randint(0, 2, size=(n_concepts, n_bits))
            duals = self._discover_duals(random_codes, n_bits)
            deps = self._discover_dependencies(random_codes, n_bits)
            triadic = self._discover_triadic_deps(random_codes, n_bits)
            dual_counts.append(len(duals))
            dep_counts.append(len(deps))
            triadic_counts.append(len(triadic))
        return {
            'mean_random_duals': float(np.mean(dual_counts)),
            'mean_random_deps': float(np.mean(dep_counts)),
            'mean_random_triadic': float(np.mean(triadic_counts)),
            'n_trials': n_trials,
            'n_concepts': n_concepts,
            'n_bits': n_bits,
        }
