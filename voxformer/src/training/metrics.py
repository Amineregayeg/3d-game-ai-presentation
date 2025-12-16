"""
Evaluation Metrics for VoxFormer

Implements WER (Word Error Rate) calculation using edit distance.

WER = (S + D + I) / N

Where:
- S: Substitutions
- D: Deletions
- I: Insertions
- N: Total words in reference
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional
from collections import defaultdict
import torch


def normalize_text(text: str) -> str:
    """
    Normalize text for WER calculation.

    - Lowercase
    - Remove punctuation
    - Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_edit_distance(
    ref: List[str],
    hyp: List[str],
) -> Tuple[int, int, int, int]:
    """
    Compute word-level edit distance using dynamic programming.

    Args:
        ref: Reference word list
        hyp: Hypothesis word list

    Returns:
        Tuple of (substitutions, deletions, insertions, total_ref_words)
    """
    m, n = len(ref), len(hyp)

    # DP table: dp[i][j] = (edit_dist, substitutions, deletions, insertions)
    dp = [[None] * (n + 1) for _ in range(m + 1)]

    # Base cases
    dp[0][0] = (0, 0, 0, 0)
    for i in range(1, m + 1):
        dp[i][0] = (i, 0, i, 0)  # All deletions
    for j in range(1, n + 1):
        dp[0][j] = (j, 0, 0, j)  # All insertions

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                # Match: no operation needed
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Consider substitution, deletion, insertion
                sub = dp[i - 1][j - 1]
                delete = dp[i - 1][j]
                insert = dp[i][j - 1]

                # Choose minimum edit distance
                candidates = [
                    (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),      # Substitution
                    (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),  # Deletion
                    (insert[0] + 1, insert[1], insert[2], insert[3] + 1),  # Insertion
                ]
                dp[i][j] = min(candidates, key=lambda x: x[0])

    _, subs, dels, ins = dp[m][n]
    return subs, dels, ins, m


def compute_wer(
    references: List[str],
    hypotheses: List[str],
    normalize: bool = True,
) -> Tuple[float, dict]:
    """
    Compute Word Error Rate between reference and hypothesis transcriptions.

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions
        normalize: Whether to normalize text before comparison

    Returns:
        Tuple of:
        - WER as percentage
        - Dictionary with detailed metrics
    """
    assert len(references) == len(hypotheses), "Reference and hypothesis counts must match"

    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref = normalize_text(ref)
            hyp = normalize_text(hyp)

        ref_words = ref.split()
        hyp_words = hyp.split()

        subs, dels, ins, n_words = compute_edit_distance(ref_words, hyp_words)

        total_subs += subs
        total_dels += dels
        total_ins += ins
        total_words += n_words

    if total_words == 0:
        return 0.0, {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0}

    wer = 100.0 * (total_subs + total_dels + total_ins) / total_words

    metrics = {
        "wer": wer,
        "substitutions": total_subs,
        "deletions": total_dels,
        "insertions": total_ins,
        "total_words": total_words,
        "total_errors": total_subs + total_dels + total_ins,
    }

    return wer, metrics


class WERMetric:
    """
    Accumulating WER metric for tracking over batches/epochs.

    Example:
        wer_metric = WERMetric()
        for batch in dataloader:
            refs, hyps = model.transcribe(batch)
            wer_metric.update(refs, hyps)
        final_wer = wer_metric.compute()
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.total_subs = 0
        self.total_dels = 0
        self.total_ins = 0
        self.total_words = 0
        self.num_samples = 0

    def update(
        self,
        references: List[str],
        hypotheses: List[str],
    ):
        """
        Update metrics with new batch.

        Args:
            references: List of reference transcriptions
            hypotheses: List of hypothesis transcriptions
        """
        for ref, hyp in zip(references, hypotheses):
            if self.normalize:
                ref = normalize_text(ref)
                hyp = normalize_text(hyp)

            ref_words = ref.split()
            hyp_words = hyp.split()

            subs, dels, ins, n_words = compute_edit_distance(ref_words, hyp_words)

            self.total_subs += subs
            self.total_dels += dels
            self.total_ins += ins
            self.total_words += n_words
            self.num_samples += 1

    def compute(self) -> float:
        """Compute current WER."""
        if self.total_words == 0:
            return 0.0
        return 100.0 * (self.total_subs + self.total_dels + self.total_ins) / self.total_words

    def get_metrics(self) -> dict:
        """Get all accumulated metrics."""
        return {
            "wer": self.compute(),
            "substitutions": self.total_subs,
            "deletions": self.total_dels,
            "insertions": self.total_ins,
            "total_words": self.total_words,
            "total_errors": self.total_subs + self.total_dels + self.total_ins,
            "num_samples": self.num_samples,
        }


def compute_cer(
    references: List[str],
    hypotheses: List[str],
    normalize: bool = True,
) -> Tuple[float, dict]:
    """
    Compute Character Error Rate.

    Similar to WER but operates at character level.

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions
        normalize: Whether to normalize text

    Returns:
        Tuple of CER percentage and detailed metrics
    """
    total_subs = 0
    total_dels = 0
    total_ins = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref = normalize_text(ref)
            hyp = normalize_text(hyp)

        # Character-level comparison
        ref_chars = list(ref)
        hyp_chars = list(hyp)

        subs, dels, ins, n_chars = compute_edit_distance(ref_chars, hyp_chars)

        total_subs += subs
        total_dels += dels
        total_ins += ins
        total_chars += n_chars

    if total_chars == 0:
        return 0.0, {"cer": 0.0}

    cer = 100.0 * (total_subs + total_dels + total_ins) / total_chars

    return cer, {
        "cer": cer,
        "substitutions": total_subs,
        "deletions": total_dels,
        "insertions": total_ins,
        "total_chars": total_chars,
    }
