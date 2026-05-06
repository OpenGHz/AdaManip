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
    extract_minimal_chain_from_attempt,
    infer_attempt_chain,
    infer_reasonable_prediction_chains,
    rank_expanded_minimal_chain_ids,
    score_language_chain_for_inference,
    sort_expanded_minimal_chains_by_inference_priority,
)


# Hand-crafted (attempt_chain, stage_status, expected_minimal, comment)
# tuples covering the representative cases. ``expected_minimal`` documents
# the user-defined contract for the aggregate rule; the script prints both
# expected and actual so any drift is visible immediately.
EXTRACT_MINIMAL_CHAIN_EXAMPLES = [
    (
        ["拉门"],
        [True],
        ["拉门"],
        "microwave / cw=0 demo: 一次性拉门成功",
    ),
    (
        ["按按钮", "拉门"],
        [True, True],
        ["按按钮", "拉门"],
        "microwave / start_with_pull=False: 按按钮+拉门，没失败 → minimal == attempt"
        "（cw=0 时按按钮其实多余，但 attempt 视角看不出来——只在 ground_truth_chain 里反映）",
    ),
    (
        ["拉门", "按按钮", "拉门"],
        [False, True, True],
        ["按按钮", "拉门"],
        "microwave / cw=1, start_with_pull=True: 先误拉门(失败) → 按按钮 → 拉门(成功)",
    ),
    (
        ["1x旋转瓶盖", "向上提起瓶盖", "1x旋转瓶盖", "向上提起瓶盖"],
        [True, False, True, True],
        ["2x旋转瓶盖", "向上提起瓶盖"],
        "bottle: 1x rotate, lift fail, 再 1x rotate, lift succ。"
        "丢失败 lift → [1x, 1x, lift] → 合并相邻 Nx → [2x, lift]。"
        "（这与 data_collection.md §7 的示例 minimal=[1x旋转瓶盖, 向上提起瓶盖] 不一致，"
        "doc 中那个 1x 写错了 N，应当为 2x。）",
    ),
    (
        [
            "9x旋转笔盖", "向上提起笔盖",
            "1x旋转笔盖", "向上提起笔盖",
            "5x旋转笔盖", "向上提起笔盖",
        ],
        [True, False, True, False, True, True],
        ["15x旋转笔盖", "向上提起笔盖"],
        "pen: 9x→fail→1x→fail→5x→succ。"
        "丢两次失败 lift → [9x, 1x, 5x, lift] → 合并相邻 Nx → [15x, lift]",
    ),
    (
        ["12x旋转笔盖", "向上提起笔盖"],
        [True, True],
        ["12x旋转笔盖", "向上提起笔盖"],
        "pen: 一次成功 → minimal == attempt",
    ),
    (
        ["顺时针旋转把手", "拉开门"],
        [True, True],
        ["顺时针旋转把手", "拉开门"],
        "door: 一次成功 → minimal == attempt",
    ),
    (
        ["向上提起笔盖", "9x旋转笔盖", "向上提起笔盖"],
        [False, True, True],
        ["9x旋转笔盖", "向上提起笔盖"],
        "pen: 起手错误（直接抬升失败）→ 转 9x → lift 成功。"
        "丢首段失败 lift → [9x, lift]，无相邻 Nx 可合并",
    ),
    (
        ["1x旋转笔盖", "1x旋转笔盖", "1x旋转笔盖", "向上提起笔盖"],
        [True, True, True, True],
        ["3x旋转笔盖", "向上提起笔盖"],
        "pen 假想：连续 3 段独立 1x rotate 都成功 → 合并为 3x。"
        "（实际 demo 不会拆这么细；构造此例验证合并规则）",
    ),
    (
        ["按按钮", "按按钮", "拉门"],
        [True, True, True],
        ["按按钮", "拉门"],
        "microwave 假想：相邻同一个非 Nx stage 重复 → 去重。"
        "（实际 demo 不会出现这种 attempt；用例只是测合并规则的边界）",
    ),
]


def print_extract_minimal_chain_examples() -> None:
    print("=" * 100)
    print("extract_minimal_chain_from_attempt examples")
    print("rule: 1) drop stages where stage_status is False; "
          "2) merge consecutive same-op stages "
          "(Nx: sum counts; non-Nx: dedup adjacent)")
    print("=" * 100)
    for attempt, status, expected, comment in EXTRACT_MINIMAL_CHAIN_EXAMPLES:
        actual = extract_minimal_chain_from_attempt(attempt, status)
        ok = "✓" if actual == expected else "✗"
        print(f"  {ok} {comment}")
        print(f"    attempt : {format_chain(attempt)}")
        print(f"    status  : {status}")
        print(f"    expected: {format_chain(expected)}")
        print(f"    actual  : {format_chain(actual)}")
        print()


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

    print_extract_minimal_chain_examples()

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
