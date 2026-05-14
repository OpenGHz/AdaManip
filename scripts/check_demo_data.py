"""Validate AdaManip zarr ZipStore files (typically ``demo_data.zip``).

Usage:
    python check_demo_data.py PATH [PATH ...]

PATH may be:
- a ``.zip`` file → checked directly
- a directory → recursively scanned for files matching ``--pattern``
  (default: ``demo_data.zip``)

Exit code: 0 if all files OK, 1 if any file is bad, 2 if no targets matched.
"""
import argparse
import sys
import traceback
import zipfile
from pathlib import Path

import numpy as np
import zarr


REQUIRED_DATA_ARRAYS = ("pcs", "env_state", "action")


def check_one(path: Path):
    issues = []

    if not path.exists():
        return False, ["file does not exist"]

    size = path.stat().st_size
    if size == 0:
        return False, ["file is empty (0 bytes)"]

    # 1. Verify it's a structurally valid zip and CRCs match.
    try:
        with zipfile.ZipFile(path) as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                issues.append(f"zip CRC failure in member: {bad_member}")
            names = zf.namelist()
    except zipfile.BadZipFile as e:
        return False, [f"not a valid zip file: {e}"]
    except Exception as e:
        return False, [f"failed to open as zip: {e}"]

    if not names:
        return False, ["zip contains no entries"]

    # 2. Open as a zarr ZipStore.
    try:
        store = zarr.ZipStore(str(path), mode="r")
    except Exception as e:
        return False, issues + [f"failed to open as zarr ZipStore: {e}"]

    try:
        try:
            root = zarr.group(store=store)
        except Exception as e:
            return False, issues + [f"failed to open zarr root group: {e}"]

        # 3. Required groups.
        for group_name in ("data", "meta"):
            if group_name not in root:
                issues.append(f"missing group: {group_name!r}")

        # 4. meta/episode_ends.
        episode_ends = None
        if "meta" in root:
            if "episode_ends" not in root["meta"]:
                issues.append("missing meta/episode_ends")
            else:
                try:
                    episode_ends = np.asarray(root["meta"]["episode_ends"][:])
                except Exception as e:
                    issues.append(f"failed to read meta/episode_ends: {e}")

        # 5. Required data arrays + decompress sanity check.
        data_lengths = {}
        if "data" in root:
            data_group = root["data"]
            for key in REQUIRED_DATA_ARRAYS:
                if key not in data_group:
                    issues.append(f"missing data/{key}")
                    continue
                arr = data_group[key]
                shape = tuple(arr.shape)
                if not shape:
                    issues.append(f"data/{key} has empty shape")
                    continue
                data_lengths[key] = shape[0]
                # Touch first and last frames to surface decompress errors that
                # `zipfile.testzip()` won't catch on its own (corrupted blocks
                # only manifest when actually decoded by zarr).
                try:
                    if shape[0] > 0:
                        _ = arr[0]
                        _ = arr[shape[0] - 1]
                except Exception as e:
                    issues.append(f"failed to decode data/{key}: {e}")

        # 6. Cross-check shapes vs episode_ends.
        if episode_ends is not None:
            if episode_ends.size == 0:
                issues.append("episode_ends is empty")
            else:
                if not np.all(np.diff(episode_ends) > 0):
                    issues.append("episode_ends is not strictly increasing")
                last_end = int(episode_ends[-1])
                if last_end <= 0:
                    issues.append(f"episode_ends[-1]={last_end} is non-positive")
                for name, n in data_lengths.items():
                    if n != last_end:
                        issues.append(
                            f"data/{name}.shape[0]={n} != episode_ends[-1]={last_end}"
                        )

        if len(set(data_lengths.values())) > 1:
            issues.append(f"data array lengths disagree: {data_lengths}")
    finally:
        try:
            store.close()
        except Exception:
            pass

    return not issues, issues


def collect_targets(roots, pattern):
    targets = []
    for r in roots:
        if r.is_file():
            targets.append(r)
        elif r.is_dir():
            targets.extend(sorted(r.rglob(pattern)))
        else:
            print(f"warning: {r} does not exist", file=sys.stderr)
    seen = set()
    uniq = []
    for p in targets:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
    return uniq


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--pattern",
        default="demo_data.zip",
        help="filename to look for when scanning directories (default: demo_data.zip)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="also print [OK] entries (default prints only [BAD])",
    )
    parser.add_argument(
        "--traceback", action="store_true",
        help="print full traceback for unexpected exceptions",
    )
    args = parser.parse_args()

    targets = collect_targets(args.paths, args.pattern)
    if not targets:
        print("no files matched", file=sys.stderr)
        sys.exit(2)

    n_ok = 0
    n_bad = 0
    for path in targets:
        try:
            ok, issues = check_one(path)
        except Exception as e:
            ok = False
            issues = [f"unexpected exception: {e}"]
            if args.traceback:
                issues.append(traceback.format_exc())
        if ok:
            n_ok += 1
            if args.verbose:
                print(f"[OK]   {path}")
        else:
            n_bad += 1
            print(f"[BAD]  {path}")
            for msg in issues:
                print(f"        - {msg}")

    print(f"\nchecked {len(targets)} file(s): {n_ok} OK, {n_bad} BAD")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
