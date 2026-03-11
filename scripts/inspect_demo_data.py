import argparse
import os
import re
from collections import defaultdict

import numpy as np
import yaml
import zarr


def resolve_paths(input_path):
    normalized_path = os.path.abspath(input_path)
    if os.path.isdir(normalized_path):
        dataset_path = os.path.join(normalized_path, "demo_data.zip")
        save_dir = normalized_path
    else:
        dataset_path = normalized_path
        save_dir = os.path.dirname(normalized_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return dataset_path, save_dir


def load_cfg_num_envs(cfg_path):
    if not cfg_path:
        return None
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg.get("env", {}).get("numEnvs")


def summarize_dataset(dataset_path):
    dataset_root = zarr.open(dataset_path, mode="r")
    pcs = dataset_root["data"]["pcs"]
    env_state = dataset_root["data"]["env_state"]
    action = dataset_root["data"]["action"]
    episode_ends = np.asarray(dataset_root["meta"]["episode_ends"])

    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    frames_per_episode = episode_ends - episode_starts

    return {
        "pcs_shape": tuple(pcs.shape),
        "env_state_shape": tuple(env_state.shape),
        "action_shape": tuple(action.shape),
        "total_frames": int(pcs.shape[0]),
        "points_per_frame": int(pcs.shape[1]) if len(pcs.shape) > 1 else None,
        "point_dim": int(pcs.shape[2]) if len(pcs.shape) > 2 else None,
        "total_saved_episodes": int(len(episode_ends)),
        "frames_per_episode": frames_per_episode.astype(int).tolist(),
    }


def summarize_videos(save_dir):
    rgb_root = os.path.join(save_dir, "rgb_videos")
    if not os.path.isdir(rgb_root):
        return None

    env_pattern = re.compile(r"env_(\d+)_cam_(\d+)\.mp4$")
    collect_episode_dirs = []
    categorized_dirs = {}

    for entry in sorted(os.listdir(rgb_root)):
        entry_path = os.path.join(rgb_root, entry)
        if entry.startswith("episode_") and os.path.isdir(entry_path):
            collect_episode_dirs.append(entry_path)
        elif entry in {"success", "failure"} and os.path.isdir(entry_path):
            categorized_dirs[entry] = entry_path

    summary = {
        "collect_episode_count": len(collect_episode_dirs),
        "collect_env_counts": {},
        "success_env_counts": {},
        "failure_env_counts": {},
        "inferred_num_envs": None,
    }

    if collect_episode_dirs:
        collect_env_counts = defaultdict(int)
        env_ids = set()
        for episode_dir in collect_episode_dirs:
            env_ids_in_episode = set()
            for filename in os.listdir(episode_dir):
                match = env_pattern.match(filename)
                if match:
                    env_id = int(match.group(1))
                    env_ids_in_episode.add(env_id)
            for env_id in env_ids_in_episode:
                collect_env_counts[env_id] += 1
            env_ids.update(env_ids_in_episode)
        summary["collect_env_counts"] = dict(sorted(collect_env_counts.items()))
        summary["inferred_num_envs"] = len(env_ids)

    for result_name, result_root in categorized_dirs.items():
        result_counts = defaultdict(int)
        env_ids = set()
        for episode_name in sorted(os.listdir(result_root)):
            episode_dir = os.path.join(result_root, episode_name)
            if not os.path.isdir(episode_dir):
                continue
            env_ids_in_episode = set()
            for filename in os.listdir(episode_dir):
                match = env_pattern.match(filename)
                if match:
                    env_id = int(match.group(1))
                    env_ids_in_episode.add(env_id)
            for env_id in env_ids_in_episode:
                result_counts[env_id] += 1
            env_ids.update(env_ids_in_episode)
        summary[f"{result_name}_env_counts"] = dict(sorted(result_counts.items()))
        if summary["inferred_num_envs"] is None and env_ids:
            summary["inferred_num_envs"] = len(env_ids)

    return summary


def print_summary(input_path, cfg_path=None):
    dataset_path, save_dir = resolve_paths(input_path)
    dataset_summary = summarize_dataset(dataset_path)
    video_summary = summarize_videos(save_dir)
    cfg_num_envs = load_cfg_num_envs(cfg_path)

    print(f"Dataset path: {dataset_path}")
    print(f"Save dir: {save_dir}")
    print(f"pcs shape: {dataset_summary['pcs_shape']}")
    print(f"env_state shape: {dataset_summary['env_state_shape']}")
    print(f"action shape: {dataset_summary['action_shape']}")
    print(f"Total saved episodes: {dataset_summary['total_saved_episodes']}")
    print(f"Total frames: {dataset_summary['total_frames']}")
    print(f"Points per frame: {dataset_summary['points_per_frame']}")
    print(f"Point dimension: {dataset_summary['point_dim']}")

    if cfg_num_envs is not None:
        print(f"Configured num envs: {cfg_num_envs}")

    if video_summary is not None:
        if video_summary["inferred_num_envs"] is not None:
            print(f"Video-inferred num envs: {video_summary['inferred_num_envs']}")
        if video_summary["collect_episode_count"]:
            print(f"Collected rollout episodes from rgb_videos: {video_summary['collect_episode_count']}")
            print("Per-env rollout episode counts:")
            for env_id, count in video_summary["collect_env_counts"].items():
                print(f"  env_{env_id:02d}: {count}")
        if video_summary["success_env_counts"]:
            print("Per-env success episode counts:")
            for env_id, count in video_summary["success_env_counts"].items():
                print(f"  env_{env_id:02d}: {count}")
        if video_summary["failure_env_counts"]:
            print("Per-env failure episode counts:")
            for env_id, count in video_summary["failure_env_counts"].items():
                print(f"  env_{env_id:02d}: {count}")

    print("Frames and points per saved episode:")
    for episode_idx, frame_count in enumerate(dataset_summary["frames_per_episode"]):
        print(
            f"  episode_{episode_idx:04d}: frames={frame_count}, "
            f"points_per_frame={dataset_summary['points_per_frame']}"
        )

    print(
        "Note: demo_data.zip does not store env_id for each saved episode, "
        "so exact per-env saved-episode counts cannot be recovered from the zarr file alone."
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect ada_manip demo dataset statistics")
    parser.add_argument("input_path", help="Path to a demo_data directory or demo_data.zip file")
    parser.add_argument("--cfg", dest="cfg_path", help="Optional YAML config path to print configured numEnvs")
    args = parser.parse_args()
    print_summary(args.input_path, cfg_path=args.cfg_path)


if __name__ == "__main__":
    main()