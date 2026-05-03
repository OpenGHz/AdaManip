"""Utilities for reasoning about language-condition chains.

The functions here intentionally operate only on the abstract chain tokens used
by language_template.json / language_expanded.json. They do not encode any
microwave-specific mechanism such as clock_wise.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple


_REPEAT_STAGE_RE = re.compile(r"^\s*(\d+)x(.+?)\s*$")


def normalize_chain(chain: Iterable[str]) -> List[str]:
    return [str(stage).strip() for stage in chain if str(stage).strip()]


def expand_stage_to_atomic(stage: str) -> List[str]:
    """Expand concrete repeat stages like ``2x旋转瓶盖`` into atomic tokens."""

    stage = str(stage).strip()
    match = _REPEAT_STAGE_RE.match(stage)
    if not match:
        return [stage] if stage else []

    count = int(match.group(1))
    if count <= 0:
        raise ValueError(f"repeat count must be positive in stage={stage!r}")
    operation = match.group(2).strip()
    if not operation:
        raise ValueError(f"missing operation after repeat count in stage={stage!r}")
    return [operation] * count


def expand_chain_to_atomic(chain: Iterable[str]) -> List[str]:
    atomic = []
    for stage in normalize_chain(chain):
        atomic.extend(expand_stage_to_atomic(stage))
    return atomic


def is_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    """Return whether all needle tokens appear in order inside haystack."""

    if not needle:
        return True
    cursor = 0
    for item in haystack:
        if item == needle[cursor]:
            cursor += 1
            if cursor == len(needle):
                return True
    return False


def chain_satisfies_ground_truth(language_chain: Iterable[str], ground_truth_chain: Iterable[str]) -> bool:
    """Whether the language-conditioned execution contains the ground-truth chain.

    The check is based on atomic-operation subsequence matching. This makes
    ``2x旋转`` sufficient for a ``1x旋转`` requirement while still treating
    opposite directions such as ``顺时针旋转`` and ``逆时针旋转`` as distinct.
    """

    language_atomic = expand_chain_to_atomic(language_chain)
    ground_truth_atomic = expand_chain_to_atomic(ground_truth_chain)
    return is_subsequence(ground_truth_atomic, language_atomic)


def infer_attempt_chain(language_chain: Iterable[str], ground_truth_chain: Iterable[str]) -> List[str]:
    """Infer the abstract attempt chain under the two-phase diagnostic model.

    First the policy executes ``language_chain``. If that chain already contains
    the ground-truth requirement, the observed attempt is just ``language_chain``.
    Otherwise the first attempt is insufficient, so the diagnostic abstraction
    appends the ground-truth recovery chain.
    """

    language_chain = normalize_chain(language_chain)
    ground_truth_chain = normalize_chain(ground_truth_chain)
    if chain_satisfies_ground_truth(language_chain, ground_truth_chain):
        return language_chain
    return language_chain + ground_truth_chain


def build_attempt_partitions(
    language_chain: Iterable[str],
    expanded_minimal_chains: Sequence[Iterable[str]],
) -> Dict[Tuple[str, ...], List[int]]:
    """Map each possible observed attempt to compatible ground-truth chain ids."""

    partitions: Dict[Tuple[str, ...], List[int]] = {}
    for chain_id, ground_truth_chain in enumerate(expanded_minimal_chains):
        attempt_chain = tuple(infer_attempt_chain(language_chain, ground_truth_chain))
        partitions.setdefault(attempt_chain, []).append(chain_id)
    return partitions


def score_language_chain_for_inference(
    language_chain: Iterable[str],
    expanded_minimal_chains: Sequence[Iterable[str]],
) -> Dict[str, float]:
    """Score how informative one language chain is for identifying ground truth."""

    partitions = build_attempt_partitions(language_chain, expanded_minimal_chains)
    num_ground_truth = len(expanded_minimal_chains)
    candidate_counts = []
    for chain_ids in partitions.values():
        candidate_counts.extend([len(chain_ids)] * len(chain_ids))

    unique_count = sum(1 for count in candidate_counts if count == 1)
    attempt_atomic_lengths = [
        len(expand_chain_to_atomic(infer_attempt_chain(language_chain, gt_chain)))
        for gt_chain in expanded_minimal_chains
    ]
    language_atomic_len = len(expand_chain_to_atomic(language_chain))

    return {
        "unique_ground_truth_count": float(unique_count),
        "unique_ground_truth_rate": (unique_count / num_ground_truth) if num_ground_truth else 0.0,
        "distinct_attempt_count": float(len(partitions)),
        "worst_case_candidate_count": float(max(candidate_counts) if candidate_counts else 0),
        "expected_candidate_count": (
            sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0
        ),
        "mean_attempt_atomic_length": (
            sum(attempt_atomic_lengths) / len(attempt_atomic_lengths)
            if attempt_atomic_lengths
            else 0.0
        ),
        "language_atomic_length": float(language_atomic_len),
    }


def rank_expanded_minimal_chain_ids(expanded_minimal_chains: Sequence[Iterable[str]]) -> List[int]:
    """Return chain ids sorted by diagnostic inference priority."""

    chains = [normalize_chain(chain) for chain in expanded_minimal_chains]

    def sort_key(index: int):
        score = score_language_chain_for_inference(chains[index], chains)
        return (
            -score["unique_ground_truth_count"],
            score["worst_case_candidate_count"],
            score["expected_candidate_count"],
            -score["distinct_attempt_count"],
            score["mean_attempt_atomic_length"],
            score["language_atomic_length"],
            index,
        )

    return sorted(range(len(chains)), key=sort_key)


def sort_expanded_minimal_chains_by_inference_priority(
    expanded_minimal_chains: Sequence[Iterable[str]],
) -> List[List[str]]:
    """Return chains sorted by diagnostic inference priority."""

    chains = [normalize_chain(chain) for chain in expanded_minimal_chains]
    return [chains[index] for index in rank_expanded_minimal_chain_ids(chains)]
