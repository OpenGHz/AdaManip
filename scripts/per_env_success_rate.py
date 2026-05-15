"""Aggregate per-env success rate from one or more eval runs, annotated
with each env's ground_truth_chain (resolved via language_expanded.json
when available).

Usage:
    python per_env_success_rate.py PATH [PATH ...]

PATH may be:
- an ``eval_metrics.json`` file → used directly
- an eval-run directory → ``eval_metrics.json`` inside it is used
- a parent directory → recursively scanned for ``eval_metrics.json``

Output: a small table per run (env_id, success_rate, gt_chain_id, gt_chain),
followed by an overall summary line. Pass ``--json`` to dump structured
data to stdout in addition (or use ``--json-only`` to suppress the table).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _load_chain_strings(run_dir: Path):
    """Return dict[chain_id -> list[str]] or {} if language_expanded.json
    is missing/malformed."""
    path = run_dir / "language_expanded.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    chains = data.get("expanded_minimal_chains")
    if not isinstance(chains, list):
        return {}
    return {i: list(c) for i, c in enumerate(chains) if isinstance(c, list)}


def _aggregate(metrics):
    """Walk episodes[].envs[], aggregate per env_id."""
    per_env = defaultdict(lambda: {
        "episode_count": 0,
        "success_count": 0,
        "gt_chain_ids": set(),
        "language_chain_ids": set(),
        "clock_wise_values": set(),
    })
    for ep in metrics.get("episodes", []):
        for env in ep.get("envs", []):
            env_id = env.get("env_id")
            if env_id is None:
                continue
            bucket = per_env[int(env_id)]
            bucket["episode_count"] += 1
            if env.get("success"):
                bucket["success_count"] += 1
            gt = env.get("ground_truth_chain_id")
            if gt is not None:
                bucket["gt_chain_ids"].add(int(gt))
            lcid = env.get("language_chain_id")
            if lcid is not None:
                bucket["language_chain_ids"].add(int(lcid))
            cw = env.get("clock_wise")
            if cw is not None:
                bucket["clock_wise_values"].add(float(cw))
    return per_env


def _fmt_chain(chain_ids, chain_strings):
    if not chain_ids:
        return "—"
    parts = []
    for cid in sorted(chain_ids):
        ops = chain_strings.get(cid)
        if ops is None:
            parts.append(f"id={cid}")
        else:
            parts.append(f"id={cid}:[{' → '.join(ops)}]")
    return " | ".join(parts)


def _fmt_set(values, formatter=str):
    if not values:
        return "—"
    return ",".join(formatter(v) for v in sorted(values))


def summarize_run(metrics_path: Path):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_dir = metrics_path.parent
    chain_strings = _load_chain_strings(run_dir)
    per_env = _aggregate(metrics)
    overall = metrics.get("overall", {})

    rows = []
    for env_id in sorted(per_env):
        b = per_env[env_id]
        n = b["episode_count"]
        s = b["success_count"]
        rate = (s / n) if n else 0.0
        rows.append({
            "env_id": env_id,
            "episode_count": n,
            "success_count": s,
            "success_rate": rate,
            "gt_chain_ids": sorted(b["gt_chain_ids"]),
            "gt_chain_strings": [
                chain_strings.get(cid, [f"<unknown chain_id={cid}>"])
                for cid in sorted(b["gt_chain_ids"])
            ],
            "language_chain_ids": sorted(b["language_chain_ids"]),
            "clock_wise": sorted(b["clock_wise_values"]),
        })

    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "overall_success_rate": overall.get("success_rate"),
        "mean_episode_success_rate": overall.get("mean_episode_success_rate"),
        "completed_episodes": overall.get("completed_episodes"),
        "rows": rows,
        "_chain_strings": chain_strings,
    }


def print_run_table(summary):
    print(f"=== {summary['run_dir']} ===")
    rate = summary.get("overall_success_rate")
    mean_rate = summary.get("mean_episode_success_rate")
    eps = summary.get("completed_episodes")
    extras = []
    if rate is not None:
        extras.append(f"overall_success_rate={rate:.4f}")
    if mean_rate is not None:
        extras.append(f"mean_episode_success_rate={mean_rate:.4f}")
    if eps is not None:
        extras.append(f"completed_episodes={eps}")
    if extras:
        print("  " + ", ".join(extras))

    chain_strings = summary["_chain_strings"]
    header = f"  {'env':>3} | {'eps':>4} | {'succ':>4} | {'rate':>6} | {'cw':>10} | gt_chain"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in summary["rows"]:
        cw = _fmt_set(r["clock_wise"], lambda v: f"{v:g}")
        chain_text = _fmt_chain(r["gt_chain_ids"], chain_strings)
        print(
            f"  {r['env_id']:>3} | {r['episode_count']:>4} | "
            f"{r['success_count']:>4} | {r['success_rate']:>6.2%} | "
            f"{cw:>10} | {chain_text}"
        )
    print()


def _collect_metrics_paths(roots):
    found = []
    for r in roots:
        if r.is_file() and r.name == "eval_metrics.json":
            found.append(r)
        elif r.is_dir():
            direct = r / "eval_metrics.json"
            if direct.exists():
                found.append(direct)
            else:
                found.extend(sorted(r.rglob("eval_metrics.json")))
        else:
            print(f"warning: {r} does not exist or is not a file/dir", file=sys.stderr)
    seen, uniq = set(), []
    for p in found:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--json", action="store_true",
        help="also dump structured JSON to stdout after the tables",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="dump JSON only; suppress the table output",
    )
    args = parser.parse_args()

    metrics_paths = _collect_metrics_paths(args.paths)
    if not metrics_paths:
        print("no eval_metrics.json found", file=sys.stderr)
        sys.exit(2)

    summaries = [summarize_run(p) for p in metrics_paths]

    if not args.json_only:
        for s in summaries:
            print_run_table(s)

    if args.json or args.json_only:
        out = []
        for s in summaries:
            s = dict(s)
            s.pop("_chain_strings", None)
            out.append(s)
        json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
