#!/usr/bin/env python3
"""Print examples for language-chain inference utilities across all tasks.

This is a human-inspection script, not a pass/fail unit test. It reads
cfg/language_template.json, builds small example expanded_minimal_chains for
each task, then prints:

1. infer_attempt_chain(language_chain, ground_truth_chain) for every pair.
2. infer_reasonable_prediction_chains(...) for every pair.
3. sort_expanded_minimal_chains_by_inference_priority(expanded_minimal_chains).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


ADA_MANIP_ROOT = Path(__file__).resolve().parents[1]
if str(ADA_MANIP_ROOT) not in sys.path:
    sys.path.insert(0, str(ADA_MANIP_ROOT))

from manipulation.language_chain_utils import (  # noqa: E402
    infer_attempt_chain,
    infer_reasonable_prediction_chains,
    rank_expanded_minimal_chain_ids,
    score_language_chain_for_inference,
    sort_expanded_minimal_chains_by_inference_priority,
)


NX_STAGE_RE = re.compile(r"^\s*Nx(.+?)\s*$")


def parse_chain_text(chain_text: str) -> List[str]:
    return [stage.strip() for stage in chain_text.split("->") if stage.strip()]


def format_chain(chain: Iterable[str]) -> str:
    stages = list(chain)
    return " -> ".join(stages) if stages else "<empty>"


def expand_template_chain_for_examples(chain: Sequence[str], max_repeat: int) -> List[List[str]]:
    """Expand abstract Nx stages into a small concrete sample set.

    language_template.json uses abstract stages like "Nx旋转瓶盖"; the utilities
    operate on expanded_minimal_chains, so this script materializes examples such
    as "1x旋转瓶盖" and "2x旋转瓶盖".
    """

    stage_options = []
    for stage in chain:
        match = NX_STAGE_RE.match(stage)
        if not match:
            stage_options.append([stage])
            continue
        operation = match.group(1).strip()
        stage_options.append([f"{count}x{operation}" for count in range(1, max_repeat + 1)])

    return [list(stages) for stages in itertools.product(*stage_options)]


def build_example_expanded_minimal_chains(
    minimal_chains: Sequence[str],
    max_repeat: int,
) -> List[List[str]]:
    expanded = []
    seen = set()
    for chain_text in minimal_chains:
        chain = parse_chain_text(chain_text)
        for expanded_chain in expand_template_chain_for_examples(chain, max_repeat):
            key = tuple(expanded_chain)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(expanded_chain)
    return expanded


def print_task_examples(task_name: str, task_spec: dict, max_repeat: int) -> None:
    expanded_chains = build_example_expanded_minimal_chains(
        task_spec["minimal_chains"],
        max_repeat=max_repeat,
    )
    ranked_ids = rank_expanded_minimal_chain_ids(expanded_chains)
    sorted_chains = sort_expanded_minimal_chains_by_inference_priority(expanded_chains)

    print("=" * 100)
    print(f"task: {task_name}")
    print(f"command: {task_spec.get('command', '')}")
    print(f"operation_set: {task_spec.get('operation_set', [])}")
    print()

    print(f"expanded_minimal_chains example set (max_repeat={max_repeat}):")
    for chain_id, chain in enumerate(expanded_chains):
        print(f"  [{chain_id}] {format_chain(chain)}")
    print()

    print("sort_expanded_minimal_chains_by_inference_priority:")
    for rank, chain in enumerate(sorted_chains, start=1):
        chain_id = expanded_chains.index(chain)
        score = score_language_chain_for_inference(chain, expanded_chains)
        print(
            f"  rank {rank}: id={chain_id} {format_chain(chain)} "
            f"| unique={score['unique_ground_truth_rate']:.2f} "
            f"| worst={int(score['worst_case_candidate_count'])} "
            f"| expected={score['expected_candidate_count']:.2f}"
        )
    print(f"rank ids: {ranked_ids}")
    print()

    print("infer_attempt_chain + infer_reasonable_prediction_chains:")
    for language_id, language_chain in enumerate(expanded_chains):
        for ground_truth_id, ground_truth_chain in enumerate(expanded_chains):
            attempt_chain = infer_attempt_chain(language_chain, ground_truth_chain)
            reasonable_chains = infer_reasonable_prediction_chains(
                language_chain,
                ground_truth_chain=ground_truth_chain,
                expanded_minimal_chains=expanded_chains,
            )
            print(
                f"  L{language_id} + GT{ground_truth_id}: "
                f"{format_chain(language_chain)}  +  {format_chain(ground_truth_chain)}"
            )
            print(f"    => {format_chain(attempt_chain)}")
            print(
                "    reasonable predictions: "
                + ", ".join(format_chain(chain) for chain in reasonable_chains)
            )
    print()

    print("infer_reasonable_prediction_chains without ground_truth_chain:")
    for language_id, language_chain in enumerate(expanded_chains):
        possible_attempts = [
            f"GT{ground_truth_id}=>{format_chain(infer_attempt_chain(language_chain, ground_truth_chain))}"
            for ground_truth_id, ground_truth_chain in enumerate(expanded_chains)
        ]
        reasonable_chains = infer_reasonable_prediction_chains(
            language_chain,
            ground_truth_chain=None,
            expanded_minimal_chains=expanded_chains,
        )
        print(f"  L{language_id}: {format_chain(language_chain)}")
        print("    possible attempts: " + "; ".join(possible_attempts))
        print(
            "    reasonable predictions: "
            + ", ".join(format_chain(chain) for chain in reasonable_chains)
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show language-chain reasoning examples for all language_template tasks."
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ADA_MANIP_ROOT / "cfg" / "language_template.json",
        help="Path to language_template.json.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Only print selected task(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=2,
        help="Concrete repeat count used for abstract Nx stages.",
    )
    args = parser.parse_args()

    if args.max_repeat < 1:
        raise ValueError("--max-repeat must be >= 1")

    with args.template.open("r", encoding="utf-8") as f:
        template = json.load(f)

    tasks = template.get("tasks", {})
    selected_tasks = set(args.task or tasks.keys())
    missing_tasks = sorted(selected_tasks - set(tasks.keys()))
    if missing_tasks:
        raise KeyError(f"Unknown task(s): {missing_tasks}")

    for task_name, task_spec in tasks.items():
        if task_name not in selected_tasks:
            continue
        print_task_examples(task_name, task_spec, max_repeat=args.max_repeat)


if __name__ == "__main__":
    main()
