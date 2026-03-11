import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import zarr


def resolve_dataset_path(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "demo_data.zip")
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"No demo_data.zip found under directory: {path}")

    if os.path.isfile(path):
        return path

    raise FileNotFoundError(f"Dataset path does not exist: {path}")


def flatten_arrays(group: zarr.hierarchy.Group, prefix: str = "") -> List[Tuple[str, zarr.Array]]:
    arrays: List[Tuple[str, zarr.Array]] = []
    for key, value in group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, zarr.Array):
            arrays.append((full_key, value))
        else:
            arrays.extend(flatten_arrays(value, full_key))
    return arrays


def format_shape(shape: Iterable[int]) -> str:
    return "x".join(str(dim) for dim in shape)


def summarize_episode_lengths(episode_ends: np.ndarray) -> Dict[str, object]:
    if episode_ends.size == 0:
        return {
            "episode_count": 0,
            "total_steps": 0,
            "min_length": 0,
            "max_length": 0,
            "mean_length": 0.0,
            "first_lengths": [],
        }

    starts = np.concatenate(([0], episode_ends[:-1]))
    lengths = episode_ends - starts
    return {
        "episode_count": int(len(lengths)),
        "total_steps": int(episode_ends[-1]),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "first_lengths": lengths[: min(10, len(lengths))].tolist(),
    }


def print_dataset_summary(dataset_path: str, verbose: bool) -> None:
    dataset_root = zarr.open(dataset_path, mode="r")
    arrays = flatten_arrays(dataset_root)

    print(f"dataset: {dataset_path}")
    print(f"zarr format: {getattr(dataset_root, 'zarr_format', 'unknown')}")
    print(f"array count: {len(arrays)}")

    print("\narrays:")
    for array_path, array in arrays:
        chunk_desc = format_shape(array.chunks) if array.chunks is not None else "-"
        print(
            f"  - {array_path}: shape={tuple(array.shape)}, dtype={array.dtype}, chunks={chunk_desc}"
        )

    episode_ends = None
    if "meta" in dataset_root and "episode_ends" in dataset_root["meta"]:
        episode_ends = np.asarray(dataset_root["meta"]["episode_ends"][:])

    if episode_ends is not None:
        episode_stats = summarize_episode_lengths(episode_ends)
        print("\nepisodes:")
        print(f"  - count: {episode_stats['episode_count']}")
        print(f"  - total_steps: {episode_stats['total_steps']}")
        print(f"  - min_length: {episode_stats['min_length']}")
        print(f"  - max_length: {episode_stats['max_length']}")
        print(f"  - mean_length: {episode_stats['mean_length']:.2f}")
        print(f"  - first_lengths: {episode_stats['first_lengths']}")

    if "data" in dataset_root:
        print("\npreview:")
        for key in ("pcs", "env_state", "action"):
            if key not in dataset_root["data"]:
                continue
            array = dataset_root["data"][key]
            print(f"  - data/{key}: shape={tuple(array.shape)}, dtype={array.dtype}")
            if verbose and array.shape[0] > 0:
                sample = np.asarray(array[0])
                print(f"    first_sample_min={sample.min():.6f}")
                print(f"    first_sample_max={sample.max():.6f}")
                print(f"    first_sample_mean={sample.mean():.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an ada_manip demo_data zarr dataset.")
    parser.add_argument(
        "dataset_path",
        help="Path to demo_data.zip or a directory containing demo_data.zip.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print simple statistics for the first sample of each main array.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dataset_path = resolve_dataset_path(args.dataset_path)
    print_dataset_summary(dataset_path, verbose=args.verbose)


if __name__ == "__main__":
    main()