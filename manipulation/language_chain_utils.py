"""Utilities for reasoning about language-condition chains.

The functions here intentionally operate only on the abstract chain tokens used
by language_template.json / language_expanded.json. They do not encode any
microwave-specific mechanism such as clock_wise.
"""

from __future__ import annotations

import re
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


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


def extract_minimal_chain_from_attempt(
    attempt_chain: Iterable[str],
    stage_status: Iterable[bool],
) -> List[str]:
    """Aggregate ``attempt_chain`` into the shortest chain that, if executed
    straight through (no failures), would still successfully complete the
    task. Pure attempt-derived: it does NOT consult god's-eye-view env state.

    Algorithm (two passes):

    1. Drop every stage whose ``stage_status`` is ``False`` (those stages
       were detours that didn't push the task forward).
    2. Merge consecutive stages with the same operation:
       - For ``Nx{op}`` repeat stages, sum the ``N``: ``1x旋转 + 1x旋转
         → 2x旋转``.
       - For plain non-repeat stages, dedup adjacent identical strings:
         ``[拉门, 拉门] → [拉门]`` (rare in practice).

    Examples
    --------
    >>> # Bottle: 1 rotate, lift fails, 1 more rotate, lift succeeds.
    >>> # Cumulative 2 rotations were what actually worked.
    >>> extract_minimal_chain_from_attempt(
    ...     ["1x旋转瓶盖", "向上提起瓶盖", "1x旋转瓶盖", "向上提起瓶盖"],
    ...     [True, False, True, True],
    ... )
    ['2x旋转瓶盖', '向上提起瓶盖']

    >>> # Microwave: pull fails (locked door), then button + pull succeed.
    >>> extract_minimal_chain_from_attempt(
    ...     ["拉门", "按按钮", "拉门"], [False, True, True],
    ... )
    ['按按钮', '拉门']

    >>> # Already-clean trajectory passes through unchanged.
    >>> extract_minimal_chain_from_attempt(["拉门"], [True])
    ['拉门']

    Parameters
    ----------
    attempt_chain
        Sequence of stage strings as they appear in trajectory_language.jsonl.
    stage_status
        Boolean sequence aligned with ``attempt_chain``: ``True`` if the
        stage succeeded, ``False`` if it failed.

    Returns
    -------
    List of stage strings — the aggregated minimal_chain.

    Raises
    ------
    ValueError
        When ``attempt_chain`` and ``stage_status`` have different lengths,
        or when an Nx stage has a non-positive count.
    """

    attempt_list = [str(s) for s in attempt_chain]
    status_list = [bool(s) for s in stage_status]
    if len(attempt_list) != len(status_list):
        raise ValueError(
            "attempt_chain and stage_status must have the same length; "
            f"got {len(attempt_list)} vs {len(status_list)}"
        )

    # Pass 1: drop failed stages.
    successful_stages = [
        stage for stage, ok in zip(attempt_list, status_list) if ok
    ]

    # Pass 2: merge consecutive same-operation stages.
    merged: List[str] = []
    for stage in successful_stages:
        if not merged:
            merged.append(stage)
            continue
        prev = merged[-1]
        m_curr = _REPEAT_STAGE_RE.match(stage)
        m_prev = _REPEAT_STAGE_RE.match(prev)
        if (
            m_curr is not None
            and m_prev is not None
            and m_curr.group(2).strip() == m_prev.group(2).strip()
        ):
            n_curr = int(m_curr.group(1))
            n_prev = int(m_prev.group(1))
            if n_curr <= 0 or n_prev <= 0:
                raise ValueError(
                    f"repeat counts must be positive; got prev={prev!r} curr={stage!r}"
                )
            op = m_curr.group(2).strip()
            merged[-1] = f"{n_prev + n_curr}x{op}"
        elif m_curr is None and m_prev is None and stage == prev:
            # Adjacent identical non-repeat stages: dedup.
            continue
        else:
            merged.append(stage)
    return merged


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


def infer_attempt_chain(
    language_chain: Iterable[str],
    ground_truth_chain: Iterable[str],
    observed_attempt_chains: Optional[Iterable[Iterable[str]]] = None,
) -> List[str]:
    """Infer the attempt chain under the two-phase diagnostic model.

    First the policy executes ``language_chain``. If that chain already contains
    the ground-truth requirement (atomic subsequence), the observed attempt is just
    ``language_chain``. Otherwise the first attempt is insufficient, so the
    diagnostic abstraction appends the ground-truth recovery chain:
    ``candidate = language_chain + ground_truth_chain``.

    That bare concatenation is *physics-blind*: from abstract tokens alone it cannot
    know whether a wrong leading action (e.g. a wrong rotation direction) is
    mechanically blocked before a trailing action shared with the ground truth (so
    the shared 拉门 / 提起 never executes -> real attempt is shorter, e.g. door
    ``[顺,逆,拉]``), or whether that trailing action must be performed to even detect
    failure (so it executes -> real attempt equals the concat, e.g. microwave
    ``[拉门,按按钮,拉门]``, or a "must pull to detect" door variant ``[顺,拉,逆,拉]``).

    ``observed_attempt_chains`` (optional) lets the caller pass the recorded real
    attempts (collection ``attempt_chain_counts``) so the abstract result is snapped
    onto data — the only source that encodes the task physics:

    1. If ``candidate`` is itself a recorded attempt -> return it.
    2. Else keep the **first** language step and the **trailing** ground-truth steps
       fixed and drop a subset of the middle steps (preferring to remove as few as
       possible) until the result equals a recorded attempt.
    3. If nothing matches (or no observed attempts given) -> return ``candidate``.

    With no ``observed_attempt_chains`` the behaviour is the original abstract
    concat, which is all the diagnostic partitioning in §1.8 needs (it only cares
    which ground truths yield the *same* attempt, not the step count).
    """

    language_chain = normalize_chain(language_chain)
    ground_truth_chain = normalize_chain(ground_truth_chain)
    if chain_satisfies_ground_truth(language_chain, ground_truth_chain):
        return language_chain
    candidate = language_chain + ground_truth_chain
    if not observed_attempt_chains:
        return candidate
    observed = {tuple(normalize_chain(a)) for a in observed_attempt_chains}
    if tuple(candidate) in observed:
        return candidate
    # Snap onto recorded data: keep first language step + ground-truth tail fixed,
    # drop a subset of the middle steps (remove as few as possible) to match a
    # recorded attempt; covers the "wrong rotation blocked" physics.
    if language_chain:
        first, middle = language_chain[:1], language_chain[1:]
        for keep in range(len(middle), -1, -1):
            for idx in combinations(range(len(middle)), keep):
                cand = first + [middle[i] for i in idx] + ground_truth_chain
                if tuple(cand) in observed:
                    return cand
    return candidate


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


def infer_reasonable_prediction_chains(
    language_chain: Iterable[str],
    ground_truth_chain: Iterable[str] | None = None,
    expanded_minimal_chains: Sequence[Iterable[str]] | None = None,
) -> List[List[str]]:
    """Return chain predictions that are reasonable from the observed behavior.

    The result is about what an asker can reasonably report from the actual
    performed steps, not about all hidden ground-truth states compatible with the
    observation. This handles ambiguous cases such as:

    - language = ["按按钮", "拉门"]
    - ground truth = ["拉门"] or unknown
    - observed attempt = ["按按钮", "拉门"]

    The hidden ground truth may be either ["拉门"] or ["按按钮", "拉门"], but an
    asker that only sees the actual behavior can reasonably report
    ["按按钮", "拉门"].

    When ``ground_truth_chain`` is omitted, ``expanded_minimal_chains`` is
    required so every possible ground-truth chain can be enumerated.
    """

    language_chain = normalize_chain(language_chain)
    expanded_keys = set()
    if expanded_minimal_chains is not None:
        expanded_keys = {
            tuple(normalize_chain(chain))
            for chain in expanded_minimal_chains
        }

    if ground_truth_chain is None:
        if expanded_minimal_chains is None:
            raise ValueError(
                "expanded_minimal_chains is required when ground_truth_chain is None"
            )
        candidate_ground_truth_chains = [
            normalize_chain(chain)
            for chain in expanded_minimal_chains
        ]
    else:
        candidate_ground_truth_chains = [normalize_chain(ground_truth_chain)]

    reasonable_chains = []
    for normalized_gt_chain in candidate_ground_truth_chains:
        observed_attempt = infer_attempt_chain(language_chain, normalized_gt_chain)
        predicted_chain = (
            language_chain
            if chain_satisfies_ground_truth(language_chain, normalized_gt_chain)
            else normalized_gt_chain
        )
        reasonable_chains.append(predicted_chain)
        # If the full observed attempt itself is one of the known command chains,
        # keep it as an additional reasonable prediction. For the current templates
        # this normally only duplicates predicted_chain, but it preserves the generic
        # behavior for future task templates.
        if tuple(observed_attempt) in expanded_keys:
            reasonable_chains.append(observed_attempt)

    deduped = []
    seen = set()
    for chain in reasonable_chains:
        key = tuple(chain)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chain)
    return deduped


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
