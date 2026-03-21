"""
AutoLabeler — Automatically name discovered bits using semantic analysis.

Three strategies:
  1. Embedding-based: find the word closest to the centroid of active concepts
  2. Contrastive: find the word that best separates active vs inactive concepts
  3. LLM-based: ask an LLM "what do these concepts have in common?"

Strategy 1 and 2 work offline (no API needed). Strategy 3 is most accurate
but requires an LLM API call.
"""

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from reptimeline.discovery import BitSemantics, DiscoveryReport


@dataclass
class BitLabel:
    """A discovered label for a bit."""
    bit_index: int
    label: str
    confidence: float  # 0-1
    method: str  # 'embedding', 'contrastive', 'llm', 'manual'
    active_concepts: List[str]
    inactive_concepts: List[str]


class AutoLabeler:
    """Assigns human-readable labels to discovered bits."""

    def __init__(self):
        self._embeddings = None
        self._vocab = None

    # ------------------------------------------------------------------
    # Strategy 1: Embedding centroid
    # ------------------------------------------------------------------

    def label_by_embedding(self, report: DiscoveryReport,
                           embeddings: Dict[str, np.ndarray],
                           candidate_labels: Optional[List[str]] = None,
                           ) -> List[BitLabel]:
        """Name each bit by finding the word closest to the centroid
        of its active concepts in embedding space.

        Args:
            report: DiscoveryReport from BitDiscovery.
            embeddings: Dict mapping concept -> embedding vector.
            candidate_labels: Optional restricted vocabulary for labels.
                If None, uses all keys in embeddings.
        """
        if candidate_labels is None:
            candidate_labels = list(embeddings.keys())

        # Pre-compute candidate matrix
        candidates = [(w, embeddings[w]) for w in candidate_labels
                       if w in embeddings]
        if not candidates:
            return []
        cand_words, cand_vecs = zip(*candidates)
        cand_matrix = np.stack(cand_vecs)
        cand_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
        cand_norms = np.where(cand_norms < 1e-8, 1.0, cand_norms)
        cand_normed = cand_matrix / cand_norms

        labels = []
        for bs in report.bit_semantics:
            if bs.activation_rate < 0.02:
                labels.append(BitLabel(
                    bit_index=bs.bit_index, label="DEAD",
                    confidence=0.0, method='embedding',
                    active_concepts=[], inactive_concepts=[],
                ))
                continue

            # Centroid of active concepts
            active_vecs = [embeddings[c] for c in bs.top_concepts
                           if c in embeddings]
            if not active_vecs:
                labels.append(BitLabel(
                    bit_index=bs.bit_index, label=f"bit_{bs.bit_index}",
                    confidence=0.0, method='embedding',
                    active_concepts=bs.top_concepts,
                    inactive_concepts=bs.anti_concepts,
                ))
                continue

            centroid = np.mean(active_vecs, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-8:
                continue
            centroid_normed = centroid / centroid_norm

            # Cosine similarity with all candidates
            sims = cand_normed @ centroid_normed
            best_idx = int(np.argmax(sims))
            best_word = cand_words[best_idx]
            best_sim = float(sims[best_idx])

            labels.append(BitLabel(
                bit_index=bs.bit_index,
                label=best_word,
                confidence=best_sim,
                method='embedding',
                active_concepts=bs.top_concepts,
                inactive_concepts=bs.anti_concepts,
            ))

        return labels

    # ------------------------------------------------------------------
    # Strategy 2: Contrastive (active vs inactive)
    # ------------------------------------------------------------------

    def label_by_contrast(self, report: DiscoveryReport,
                          embeddings: Dict[str, np.ndarray],
                          candidate_labels: Optional[List[str]] = None,
                          ) -> List[BitLabel]:
        """Name each bit by finding the word that best separates
        active concepts from inactive concepts.

        The label is the word whose embedding is most similar to
        (centroid_active - centroid_inactive).
        """
        if candidate_labels is None:
            candidate_labels = list(embeddings.keys())

        candidates = [(w, embeddings[w]) for w in candidate_labels
                       if w in embeddings]
        if not candidates:
            return []
        cand_words, cand_vecs = zip(*candidates)
        cand_matrix = np.stack(cand_vecs)
        cand_norms = np.linalg.norm(cand_matrix, axis=1, keepdims=True)
        cand_norms = np.where(cand_norms < 1e-8, 1.0, cand_norms)
        cand_normed = cand_matrix / cand_norms

        labels = []
        for bs in report.bit_semantics:
            if bs.activation_rate < 0.02:
                labels.append(BitLabel(
                    bit_index=bs.bit_index, label="DEAD",
                    confidence=0.0, method='contrastive',
                    active_concepts=[], inactive_concepts=[],
                ))
                continue

            active_vecs = [embeddings[c] for c in bs.top_concepts
                           if c in embeddings]
            inactive_vecs = [embeddings[c] for c in bs.anti_concepts
                             if c in embeddings]

            if not active_vecs or not inactive_vecs:
                labels.append(BitLabel(
                    bit_index=bs.bit_index, label=f"bit_{bs.bit_index}",
                    confidence=0.0, method='contrastive',
                    active_concepts=bs.top_concepts,
                    inactive_concepts=bs.anti_concepts,
                ))
                continue

            centroid_active = np.mean(active_vecs, axis=0)
            centroid_inactive = np.mean(inactive_vecs, axis=0)
            direction = centroid_active - centroid_inactive
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-8:
                continue
            direction_normed = direction / direction_norm

            sims = cand_normed @ direction_normed
            best_idx = int(np.argmax(sims))

            labels.append(BitLabel(
                bit_index=bs.bit_index,
                label=cand_words[best_idx],
                confidence=float(sims[best_idx]),
                method='contrastive',
                active_concepts=bs.top_concepts,
                inactive_concepts=bs.anti_concepts,
            ))

        return labels

    # ------------------------------------------------------------------
    # Strategy 3: LLM-based
    # ------------------------------------------------------------------

    def label_by_llm(self, report: DiscoveryReport,
                     llm_fn: Callable[[str], str],
                     ) -> List[BitLabel]:
        """Name each bit by asking an LLM what the active concepts
        have in common.

        Args:
            report: DiscoveryReport.
            llm_fn: Function that takes a prompt string and returns
                the LLM's response string. User provides their own
                API wrapper.

        Example llm_fn:
            def my_llm(prompt):
                response = openai.chat.completions.create(
                    model="gpt-4", messages=[{"role":"user","content":prompt}])
                return response.choices[0].message.content
        """
        labels = []
        for bs in report.bit_semantics:
            if bs.activation_rate < 0.02:
                labels.append(BitLabel(
                    bit_index=bs.bit_index, label="DEAD",
                    confidence=0.0, method='llm',
                    active_concepts=[], inactive_concepts=[],
                ))
                continue

            prompt = (
                f"I have a neural network bit that activates for these "
                f"concepts: {', '.join(bs.top_concepts[:15])}\n\n"
                f"And does NOT activate for these concepts: "
                f"{', '.join(bs.anti_concepts[:10])}\n\n"
                f"What single abstract concept or property do the "
                f"activating concepts share that the non-activating "
                f"concepts lack? Answer with just one or two words."
            )

            try:
                label = llm_fn(prompt).strip().lower()
                # Clean up common LLM verbosity
                label = label.split('\n')[0].strip('"\'.,! ')
                if len(label) > 30:
                    label = label[:30]
            except Exception as e:
                label = f"bit_{bs.bit_index}_ERROR"

            labels.append(BitLabel(
                bit_index=bs.bit_index,
                label=label,
                confidence=0.8,  # LLM labels assumed high quality
                method='llm',
                active_concepts=bs.top_concepts,
                inactive_concepts=bs.anti_concepts,
            ))

        return labels

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def print_labels(self, labels: List[BitLabel]):
        """Print discovered labels."""
        print()
        print("=" * 60)
        print("  AUTO-LABELED BITS")
        print("=" * 60)
        active_labels = [l for l in labels if l.label != "DEAD"]
        dead_labels = [l for l in labels if l.label == "DEAD"]

        active_labels.sort(key=lambda l: l.confidence, reverse=True)
        for bl in active_labels:
            concepts = ", ".join(bl.active_concepts[:4])
            print(f"    bit {bl.bit_index:>2d} = {bl.label:<20s}"
                  f"  conf={bl.confidence:.2f}  [{concepts}]")

        if dead_labels:
            print(f"\n    ({len(dead_labels)} dead bits omitted)")
        print("=" * 60)

    def export_as_primitives(self, labels: List[BitLabel],
                             output_path: str):
        """Export discovered labels as a primitivos.json-compatible file.

        This allows using discovered primitives as if they were
        manually defined — closing the loop between discovery and
        supervision.
        """
        primitives = []
        for bl in labels:
            if bl.label == "DEAD":
                continue
            primitives.append({
                'bit': bl.bit_index,
                'nombre': bl.label,
                'discovered': True,
                'confidence': round(bl.confidence, 3),
                'method': bl.method,
                'top_concepts': bl.active_concepts[:10],
                'anti_concepts': bl.inactive_concepts[:5],
            })

        data = {
            'version': 'discovered_1.0',
            'total': len(primitives),
            'description': ('Primitives discovered by reptimeline AutoLabeler. '
                            'Not manually defined — learned from model behavior.'),
            'primitivos': primitives,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
