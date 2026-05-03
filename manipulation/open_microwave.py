from manipulation.base_manipulation import BaseManipulation
from envs.base_env import BaseEnv
from manipulation.utils.transform import *
from manipulation.language_chain_utils import (
    infer_reasonable_prediction_chains,
    rank_expanded_minimal_chain_ids,
    score_language_chain_for_inference,
)
from logging import Logger
import numpy as np
import random
from dataset.dataset import Experience, Episode_Buffer, obs_wrapper
import os
import collections
import av
import shutil
import time
import json
import yaml
from pathlib import Path


class Mp4VideoWriter:

    def __init__(self, output_path, width, height, fps, codec, options=None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        if options:
            self.stream.options = {key: str(value) for key, value in options.items() if value is not None}

    def write(self, frame):
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def close(self):
        if self.container is None:
            return
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
        self.container = None


class EpisodeVideoRecorder:

    def __init__(self, save_dir, cfg, num_envs, width, height, num_fixed_cameras):
        video_cfg = cfg["env"].get("rgbVideo", {})
        self.save_dir = os.path.join(save_dir, "rgb_videos")
        self.camera_type = video_cfg.get("cameraType", "fixed")
        self.fps = int(video_cfg.get("fps", 10))
        self.codec = video_cfg.get("codec", "libx264")
        self.options = {
            "crf": video_cfg.get("crf", 18),
            "preset": video_cfg.get("preset", "fast"),
        }
        configured_camera_ids = video_cfg.get("cameraIds", [])
        if self.camera_type in ("fixed", "video"):
            self.camera_ids = list(configured_camera_ids) if configured_camera_ids else list(range(num_fixed_cameras))
        else:
            self.camera_ids = [0]
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.writers = {}
        self.current_episode_dir = None
        self.current_episode_idx = None
        self.temp_dir = os.path.join(self.save_dir, "_tmp")

    def start_episode(self, episode_idx):
        self.close_episode()
        self.current_episode_idx = episode_idx
        self.current_episode_dir = os.path.join(self.temp_dir, f"episode_{episode_idx:04d}")
        if os.path.isdir(self.current_episode_dir):
            shutil.rmtree(self.current_episode_dir)
        for env_id in range(self.num_envs):
            for camera_id in self.camera_ids:
                output_path = os.path.join(
                    self.current_episode_dir,
                    f"env_{env_id:02d}_cam_{camera_id:02d}.mp4",
                )
                self.writers[(env_id, camera_id)] = Mp4VideoWriter(
                    output_path=output_path,
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    codec=self.codec,
                    options=self.options,
                )

    def write_frames(self, frames_by_env):
        if not self.writers:
            return
        for env_id, env_frames in enumerate(frames_by_env):
            for frame_idx, frame in enumerate(env_frames):
                camera_id = self.camera_ids[frame_idx]
                self.writers[(env_id, camera_id)].write(frame)

    def close_episode(self):
        for writer in self.writers.values():
            writer.close()
        self.writers = {}

    def finish_episode(self, env_results=None):
        episode_dir = self.current_episode_dir
        episode_idx = self.current_episode_idx
        self.close_episode()

        if episode_dir is None or episode_idx is None:
            return

        if env_results is None:
            final_episode_dir = os.path.join(self.save_dir, f"episode_{episode_idx:04d}")
            if os.path.isdir(final_episode_dir):
                shutil.rmtree(final_episode_dir)
            os.makedirs(os.path.dirname(final_episode_dir), exist_ok=True)
            if os.path.isdir(episode_dir):
                shutil.move(episode_dir, final_episode_dir)
        else:
            for env_id, result in enumerate(env_results):
                result_dir = os.path.join(self.save_dir, result, f"episode_{episode_idx:04d}")
                os.makedirs(result_dir, exist_ok=True)
                for camera_id in self.camera_ids:
                    src_path = os.path.join(episode_dir, f"env_{env_id:02d}_cam_{camera_id:02d}.mp4")
                    if os.path.exists(src_path):
                        shutil.move(src_path, os.path.join(result_dir, os.path.basename(src_path)))
            if os.path.isdir(episode_dir):
                shutil.rmtree(episode_dir)

        self.current_episode_dir = None
        self.current_episode_idx = None

    def discard_episode(self):
        episode_dir = self.current_episode_dir
        self.close_episode()
        if episode_dir is not None and os.path.isdir(episode_dir):
            shutil.rmtree(episode_dir)
        self.current_episode_dir = None
        self.current_episode_idx = None

class OpenMicroWaveManipulation(BaseManipulation) :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        super().__init__(env, cfg, logger)
        self.video_recorder = None

    def _build_microwave_trajectory_label(self, env_id, start_with_pull, expanded_minimal_chains):
        clock_wise = int(self.env.clock_wise[env_id].item())
        if start_with_pull:
            if clock_wise == 0:
                attempt_chain = ["拉门"]
                stage_status = [True]
                minimal_chain = ["拉门"]
            else:
                attempt_chain = ["拉门", "按按钮", "拉门"]
                stage_status = [False, True, True]
                minimal_chain = ["按按钮", "拉门"]
        else:
            attempt_chain = ["按按钮", "拉门"]
            stage_status = [True, True]
            minimal_chain = ["按按钮", "拉门"]

        try:
            minimal_chain_id = expanded_minimal_chains.index(minimal_chain)
        except ValueError as exc:
            raise RuntimeError(f"minimal_chain {minimal_chain} not found in expanded_minimal_chains") from exc

        command_chains, command_chain_ids = self.match_command_chains(
            attempt_chain=attempt_chain,
            stage_status=stage_status,
            expanded_minimal_chains=expanded_minimal_chains,
        )

        return {
            "minimal_chain_id": minimal_chain_id,
            "minimal_chain": minimal_chain,
            "attempt_chain": attempt_chain,
            "stage_status": stage_status,
            "command_chains": command_chains,
            "command_chain_ids": command_chain_ids,
            "success": True,
        }

    def _build_collect_save_dir(self):
        dataset_path = "open_microwave" + "_" + self.cfg["task"]["policy"] + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+"_clock"+str(self.cfg["env"]["clockwise"])
        return './demo_data/'+ dataset_path

    def _build_eval_save_dir(self):
        dataset_path = "eval_open_microwave" + "_" + self.cfg["task"]["policy"] + "_" + str(self.cfg["env"]["asset"]["AssetNum"])+"_eps"+str(self.cfg["task"]["num_episode"])+"_clock"+str(self.cfg["env"]["clockwise"])
        run_ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        dataset_path = dataset_path + "_" + run_ts
        return './eval_data/'+ dataset_path

    def _write_eval_metrics(self, eval_save_dir, metrics):
        os.makedirs(eval_save_dir, exist_ok=True)
        metrics_path = os.path.join(eval_save_dir, "eval_metrics.json")
        tmp_path = metrics_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, metrics_path)

    def _copy_eval_config(self, eval_save_dir):
        os.makedirs(eval_save_dir, exist_ok=True)
        target_path = os.path.join(eval_save_dir, "eval_config.yaml")
        serializable_cfg = {
            key: value
            for key, value in self.cfg.items()
            if not str(key).startswith("_")
        }
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(serializable_cfg, f, allow_unicode=True, sort_keys=False)
        return target_path

    def _update_eval_metrics_summary(
        self,
        metrics,
        total_elapsed_sec,
        status,
        num_envs,
        expanded_minimal_chains=None,
    ):
        episodes = metrics.get("episodes", [])
        num_envs = int(num_envs)
        episode_rates = [float(ep["success_rate"]) for ep in episodes]
        episode_elapsed = [float(ep["elapsed_sec"]) for ep in episodes]
        rollout_elapsed = [float(ep["rollout_elapsed_sec"]) for ep in episodes]
        rollout_step_counts = [
            int(ep["rollout_step_count"])
            for ep in episodes
            if ep.get("rollout_step_count") is not None
        ]
        completion_steps = [
            int(ep["completion_step"])
            for ep in episodes
            if ep.get("completion_step") is not None
        ]
        env_success_steps = [
            int(env_record["success_step"])
            for ep in episodes
            for env_record in ep.get("envs", [])
            if env_record.get("success_step") is not None
        ]
        total_trials = len(episodes) * num_envs
        total_successes = int(sum(int(ep["success_count"]) for ep in episodes))

        per_env = []
        total_asker_prompt_predictions = 0
        total_asker_prompt_correct = 0
        total_asker_prompt_unknown = 0
        total_asker_prompt_incorrect = 0
        for env_id in range(num_envs):
            env_records = []
            for ep in episodes:
                for env_record in ep.get("envs", []):
                    if int(env_record["env_id"]) == env_id:
                        env_records.append(env_record)
                        break
            env_successes = int(sum(1 for item in env_records if item.get("success")))
            success_times = [
                float(item["time_to_success_sec"])
                for item in env_records
                if item.get("time_to_success_sec") is not None
            ]
            success_steps = [
                int(item["success_step"])
                for item in env_records
                if item.get("success_step") is not None
            ]
            asker_prompt_predictions = []
            for item in env_records:
                adaptive = item.get("adaptive") or {}
                if adaptive.get("skipped"):
                    continue
                asker_chain_id = adaptive.get("asker_chain_id")
                ground_truth_chain_id = item.get("ground_truth_chain_id")
                if asker_chain_id is None or ground_truth_chain_id is None:
                    continue
                language_chain_id = item.get("language_chain_id")
                reasonable_prediction_chain_ids = self._reasonable_prediction_chain_ids(
                    language_chain_id,
                    ground_truth_chain_id,
                    expanded_minimal_chains,
                )
                strict_ground_truth_correct = int(asker_chain_id) == int(ground_truth_chain_id)
                correctness = self._asker_prompt_correctness(
                    asker_chain_id,
                    ground_truth_chain_id,
                    reasonable_prediction_chain_ids,
                )
                asker_prompt_predictions.append({
                    "episode_id": int(item.get("episode_id", -1)) if "episode_id" in item else None,
                    "asker_chain_id": int(asker_chain_id),
                    "ground_truth_chain_id": int(ground_truth_chain_id),
                    "reasonable_prediction_chain_ids": reasonable_prediction_chain_ids,
                    "strict_ground_truth_correct": strict_ground_truth_correct,
                    "correct": correctness,
                })
            asker_prompt_correct_count = int(sum(1 for item in asker_prompt_predictions if item["correct"] is True))
            asker_prompt_unknown_count = int(sum(1 for item in asker_prompt_predictions if item["correct"] is None))
            asker_prompt_incorrect_count = int(sum(1 for item in asker_prompt_predictions if item["correct"] is False))
            asker_prompt_prediction_count = len(asker_prompt_predictions)
            total_asker_prompt_predictions += asker_prompt_prediction_count
            total_asker_prompt_correct += asker_prompt_correct_count
            total_asker_prompt_unknown += asker_prompt_unknown_count
            total_asker_prompt_incorrect += asker_prompt_incorrect_count
            last_asker_prediction = asker_prompt_predictions[-1] if asker_prompt_predictions else None

            per_env_record = {
                "env_id": env_id,
                "episode_count": len(env_records),
                "success_count": env_successes,
                "success_rate": env_successes / len(env_records) if env_records else 0.0,
                "mean_time_to_success_sec": float(np.mean(success_times)) if success_times else None,
                "min_time_to_success_sec": float(np.min(success_times)) if success_times else None,
                "max_time_to_success_sec": float(np.max(success_times)) if success_times else None,
                "mean_success_step": float(np.mean(success_steps)) if success_steps else None,
                "min_success_step": int(np.min(success_steps)) if success_steps else None,
                "max_success_step": int(np.max(success_steps)) if success_steps else None,
                "asker_prompt_prediction_count": asker_prompt_prediction_count,
                "asker_prompt_correct_count": asker_prompt_correct_count,
                "asker_prompt_unknown_count": asker_prompt_unknown_count,
                "asker_prompt_incorrect_count": asker_prompt_incorrect_count,
                "asker_prompt_accuracy": (
                    asker_prompt_correct_count / asker_prompt_prediction_count
                    if asker_prompt_prediction_count
                    else None
                ),
                "asker_prompt_correct": (
                    last_asker_prediction["correct"]
                    if last_asker_prediction is not None
                    else None
                ),
                "last_asker_chain_id": (
                    last_asker_prediction["asker_chain_id"]
                    if last_asker_prediction is not None
                    else None
                ),
                "last_ground_truth_chain_id": (
                    last_asker_prediction["ground_truth_chain_id"]
                    if last_asker_prediction is not None
                    else None
                ),
                "last_reasonable_prediction_chain_ids": (
                    last_asker_prediction["reasonable_prediction_chain_ids"]
                    if last_asker_prediction is not None
                    else None
                ),
                "last_strict_ground_truth_correct": (
                    bool(last_asker_prediction["strict_ground_truth_correct"])
                    if last_asker_prediction is not None
                    else None
                ),
            }
            per_env.append(per_env_record)

        metrics["overall"] = {
            "status": status,
            "completed_episodes": len(episodes),
            "total_trials": total_trials,
            "total_successes": total_successes,
            "success_rate": total_successes / total_trials if total_trials else 0.0,
            "mean_episode_success_rate": float(np.mean(episode_rates)) if episode_rates else 0.0,
            "std_episode_success_rate": float(np.std(episode_rates)) if episode_rates else 0.0,
            "total_elapsed_sec": float(total_elapsed_sec),
            "mean_episode_elapsed_sec": float(np.mean(episode_elapsed)) if episode_elapsed else 0.0,
            "mean_rollout_elapsed_sec": float(np.mean(rollout_elapsed)) if rollout_elapsed else 0.0,
            "mean_rollout_step_count": float(np.mean(rollout_step_counts)) if rollout_step_counts else 0.0,
            "mean_episode_completion_step": float(np.mean(completion_steps)) if completion_steps else None,
            "min_episode_completion_step": int(np.min(completion_steps)) if completion_steps else None,
            "max_episode_completion_step": int(np.max(completion_steps)) if completion_steps else None,
            "mean_env_success_step": float(np.mean(env_success_steps)) if env_success_steps else None,
            "std_env_success_step": float(np.std(env_success_steps)) if env_success_steps else None,
            "asker_prompt_prediction_count": total_asker_prompt_predictions,
            "asker_prompt_correct_count": total_asker_prompt_correct,
            "asker_prompt_unknown_count": total_asker_prompt_unknown,
            "asker_prompt_incorrect_count": total_asker_prompt_incorrect,
            "asker_prompt_accuracy": (
                total_asker_prompt_correct / total_asker_prompt_predictions
                if total_asker_prompt_predictions
                else None
            ),
            "per_env": per_env,
        }

    def _prepare_save_dir(self, save_dir, purpose):
        if not os.path.exists(save_dir):
            return

        prompt = f"{purpose} output directory '{save_dir}' already exists. Overwrite it? [y/N]: "
        try:
            answer = input(prompt).strip().lower()
        except EOFError as exc:
            raise RuntimeError(
                f"{purpose} output directory '{save_dir}' already exists and overwrite confirmation could not be read."
            ) from exc

        if answer not in {"y", "yes"}:
            raise RuntimeError(f"Aborted to avoid overwriting existing data at '{save_dir}'.")

        shutil.rmtree(save_dir)

    def _init_video_recorder(self, save_dir):
        if not self.cfg['env'].get('collectRGBVideo', False):
            self.video_recorder = None
            return
        video_cam_cfg = self.cfg["env"].get("videoCam")
        if video_cam_cfg is not None and self.cfg["env"].get("rgbVideo", {}).get("cameraType") == "video":
            vid_width = video_cam_cfg["width"]
            vid_height = video_cam_cfg["height"]
            num_video_cams = len(video_cam_cfg["cam_start"])
        else:
            vid_width = self.cfg["env"]["cam"]["width"]
            vid_height = self.cfg["env"]["cam"]["height"]
            num_video_cams = getattr(self.env, "num_cam", 1)
        self.video_recorder = EpisodeVideoRecorder(
            save_dir=save_dir,
            cfg=self.cfg,
            num_envs=self.env.num_envs,
            width=vid_width,
            height=vid_height,
            num_fixed_cameras=num_video_cams,
        )

    def _record_video_frame(self):
        if self.video_recorder is None:
            return
        rgb_frames = self.env.collect_rgb_frames(
            camera_type=self.video_recorder.camera_type,
            camera_ids=self.video_recorder.camera_ids,
        )
        self.video_recorder.write_frames(rgb_frames)

    def _load_eval_language_embedding_bank(self, diffusion):
        if not getattr(diffusion.args, 'use_language_conditioning', False):
            return None

        language_path = getattr(diffusion.args, 'language_embedding_dict_path', None)
        if language_path is None:
            ckpt_dir = Path(diffusion.args.ckpt_path).resolve().parent
            language_path = ckpt_dir / 'language_embedding_dict.json'
        else:
            language_path = Path(language_path)

        if not language_path.exists():
            raise FileNotFoundError(
                f'language_embedding_dict.json not found: {language_path}'
            )

        with language_path.open('r', encoding='utf-8') as f:
            payload = json.load(f)

        bank = np.asarray(payload['expanded_minimal_chains'], dtype=np.float32)
        if bank.ndim != 2:
            raise ValueError(f'expanded_minimal_chains must be 2D, got shape {bank.shape}')

        expected_dim = int(getattr(diffusion.args, 'language_input_dim', bank.shape[-1]))
        if bank.shape[-1] != expected_dim:
            raise ValueError(
                f'language embedding dim mismatch: expected {expected_dim}, got {bank.shape[-1]}'
            )

        print(f'loaded language embedding bank: {language_path}, size={bank.shape[0]}')
        return torch.from_numpy(bank).to(diffusion.device)

    def _sample_episode_language_embedding(self, embedding_bank, batch_size):
        if embedding_bank is None:
            return None, None
        bank_size = embedding_bank.shape[0]
        sampled_idx = int(np.random.randint(0, bank_size))
        sampled = embedding_bank[sampled_idx].unsqueeze(0).repeat(batch_size, 1)
        return sampled, sampled_idx

    def _microwave_ground_truth_chain_id(self, clock_wise):
        return 1 if int(round(float(clock_wise))) == 1 else 0

    def _reasonable_prediction_chain_ids(
        self,
        language_chain_id,
        ground_truth_chain_id,
        expanded_minimal_chains,
    ):
        if (
            expanded_minimal_chains is None
            or language_chain_id is None
            or not (0 <= int(language_chain_id) < len(expanded_minimal_chains))
        ):
            return [int(ground_truth_chain_id)] if ground_truth_chain_id is not None else []

        ground_truth_chain = None
        if (
            ground_truth_chain_id is not None
            and 0 <= int(ground_truth_chain_id) < len(expanded_minimal_chains)
        ):
            ground_truth_chain = expanded_minimal_chains[int(ground_truth_chain_id)]

        reasonable_chains = infer_reasonable_prediction_chains(
            expanded_minimal_chains[int(language_chain_id)],
            ground_truth_chain=ground_truth_chain,
            expanded_minimal_chains=expanded_minimal_chains,
        )
        reasonable_keys = {tuple(chain) for chain in reasonable_chains}
        return [
            int(chain_id)
            for chain_id, chain in enumerate(expanded_minimal_chains)
            if tuple(chain) in reasonable_keys
        ]

    def _asker_prompt_correctness(
        self,
        asker_chain_id,
        ground_truth_chain_id,
        reasonable_prediction_chain_ids,
    ):
        if asker_chain_id is None:
            return None
        asker_chain_id = int(asker_chain_id)
        if asker_chain_id in reasonable_prediction_chain_ids:
            return True
        if ground_truth_chain_id is not None:
            ground_truth_chain_id = int(ground_truth_chain_id)
            if asker_chain_id == ground_truth_chain_id:
                return None
        return False

    def _select_language_embedding_per_env(self, embedding_bank, chain_ids):
        """Build a (num_envs, embed_dim) tensor from a per-env list of chain ids."""
        if embedding_bank is None:
            return None
        import torch
        index = torch.as_tensor(list(chain_ids), dtype=torch.long, device=embedding_bank.device)
        return embedding_bank.index_select(0, index)

    def _adaptive_video_path(self, eval_save_dir, eps_idx, env_id, camera_id):
        return os.path.join(
            eval_save_dir,
            "rgb_videos",
            f"episode_{eps_idx:04d}",
            f"env_{env_id:02d}_cam_{camera_id:02d}.mp4",
        )

    def _adaptive_resolve_camera_id(self, asker_cfg):
        cam_id = getattr(asker_cfg, "camera_id", 0)
        if self.video_recorder is not None and self.video_recorder.camera_ids:
            if cam_id in self.video_recorder.camera_ids:
                return int(cam_id)
            return int(self.video_recorder.camera_ids[0])
        return int(cam_id)

    def _adaptive_recategorize_videos(self, eval_save_dir, eps_idx, env_results):
        """Move per-env videos from the flat episode dir into success|failure subdirs.

        Mirrors EpisodeVideoRecorder.finish_episode's split logic and is a no-op
        when the episode dir or camera files are absent. Best-effort: any
        IOError is logged and silently swallowed.
        """
        if self.video_recorder is None:
            return
        rgb_root = os.path.join(eval_save_dir, "rgb_videos")
        episode_dir = os.path.join(rgb_root, f"episode_{eps_idx:04d}")
        if not os.path.isdir(episode_dir):
            return
        try:
            for env_id, result in enumerate(env_results):
                if result not in ("success", "failure"):
                    continue
                result_dir = os.path.join(rgb_root, result, f"episode_{eps_idx:04d}")
                os.makedirs(result_dir, exist_ok=True)
                for camera_id in self.video_recorder.camera_ids:
                    src = os.path.join(episode_dir, f"env_{env_id:02d}_cam_{camera_id:02d}.mp4")
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(result_dir, os.path.basename(src)))
            if os.path.isdir(episode_dir) and not os.listdir(episode_dir):
                os.rmdir(episode_dir)
        except OSError as exc:
            print(f"[adaptive] video recategorization for episode {eps_idx} failed: {exc}")

    def _microwave_canonical_chain_for_clock_wise(self, clock_wise):
        """Return the canonical minimal_chain for the given clock_wise scalar.

        Mirrors AdaptiveLanguageAsker._gt_chain_for_clock_wise so dataset
        consumers (eval_video2prompt.py, Video2PromptGroundTruth) and the
        live asker agree on the truth chain.
        """
        return ["按按钮", "拉门"] if int(round(float(clock_wise))) == 1 else ["拉门"]

    def _adaptive_init_inference_dump(self, save_dir, task_spec, expanded_minimal_chains):
        """Initialize on-disk dump compatible with scripts/eval_video2prompt.py.

        Layout produced at run end (so an offline asker re-run only needs:
            scripts/eval_video2prompt.py --data_root <save_dir parent>
                                         --data_dir <save_dir base>
                                         --platform <pick> ...).

            <save_dir>/
                language_expanded.json
                trajectory_language.jsonl
                demo_data.zip          (zarr; meta/episode_ends + data/action)
                rgb_videos/episode_<eps>/env_<id>_cam_<cam>.mp4
        """
        os.makedirs(save_dir, exist_ok=True)
        language_expanded = {
            "schema_version": "v1",
            "generated_from": "manipulation/open_microwave.py::diffusion_evaluate "
            "(task.save_inference_data)",
            "task": "microwave",
            "command": task_spec["command"],
            "operation_set": task_spec.get("operation_set", []),
            "expanded_minimal_chains": [list(chain) for chain in expanded_minimal_chains],
        }
        for key in ("additional_prompt", "success_check_additional_prompt"):
            if key in task_spec:
                language_expanded[key] = task_spec[key]
        with open(os.path.join(save_dir, "language_expanded.json"), "w", encoding="utf-8") as f:
            json.dump(language_expanded, f, ensure_ascii=False, indent=2)
        return {
            "language_expanded": language_expanded,
            "records": [],          # one entry per (eps, env) tuple
            "action_arrays": [],    # parallel to records
        }

    def _adaptive_record_inference_episode(
        self,
        dump_state,
        eps_idx,
        env_id,
        clock_wise,
        chain_id_used,
        done_flag,
        adaptive_state,
        action_array,
        adaptive_asker_record,
        expanded_minimal_chains,
    ):
        if dump_state is None or action_array is None or len(action_array) == 0:
            return
        canonical_minimal_chain = self._microwave_canonical_chain_for_clock_wise(clock_wise)
        try:
            canonical_minimal_chain_id = expanded_minimal_chains.index(canonical_minimal_chain)
        except ValueError:
            canonical_minimal_chain_id = None
        chain_used_steps = (
            list(expanded_minimal_chains[chain_id_used])
            if chain_id_used is not None
            and 0 <= chain_id_used < len(expanded_minimal_chains)
            else []
        )

        # Append the per-env action array to the rolling buffer, then update
        # frame_range so consumers can slice the zarr correctly.
        prev_total = sum(arr.shape[0] for arr in dump_state["action_arrays"])
        dump_state["action_arrays"].append(np.asarray(action_array, dtype=np.float32))
        new_total = prev_total + dump_state["action_arrays"][-1].shape[0]

        record = {
            "minimal_chain_id": int(canonical_minimal_chain_id) if canonical_minimal_chain_id is not None else None,
            "minimal_chain": canonical_minimal_chain,
            # `attempt_chain` is what the policy was conditioned on this episode.
            # It groups eval_video2prompt.py's per-chain stats by language conditioning.
            "attempt_chain": chain_used_steps if chain_used_steps else canonical_minimal_chain,
            "stage_status": [True] * len(chain_used_steps) if chain_used_steps else [True],
            "command_chains": [chain_used_steps] if chain_used_steps else [canonical_minimal_chain],
            "command_chain_ids": [int(chain_id_used)] if chain_id_used is not None else [],
            "success": bool(done_flag),
            "episode_id": int(len(dump_state["records"])),
            "round_idx": int(eps_idx),
            "env_id": int(env_id),
            "frame_range": [int(prev_total), int(new_total)],
            # Inference-only metadata; eval_video2prompt.py ignores unknown fields.
            "clock_wise": float(clock_wise),
            "language_chain_id_used": int(chain_id_used) if chain_id_used is not None else None,
            "frozen_clock_wise": (
                float(adaptive_state.frozen_clock_wise)
                if adaptive_state is not None and adaptive_state.frozen_clock_wise is not None
                else None
            ),
            "tried_chain_ids": (
                sorted(int(item) for item in adaptive_state.tried_chain_ids)
                if adaptive_state is not None
                else []
            ),
            "locked_chain_id": (
                int(adaptive_state.locked_chain_id)
                if adaptive_state is not None and adaptive_state.locked_chain_id is not None
                else None
            ),
            "adaptive_asker": adaptive_asker_record,
        }
        dump_state["records"].append(record)

    def _adaptive_finalize_inference_dump(self, save_dir, dump_state):
        """Write trajectory_language.jsonl + demo_data.zip at run end."""
        if dump_state is None:
            return
        records = dump_state.get("records", [])
        action_arrays = dump_state.get("action_arrays", [])
        if not records:
            return

        traj_path = os.path.join(save_dir, "trajectory_language.jsonl")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[adaptive] wrote {traj_path} ({len(records)} entries)")

        try:
            import zarr
        except ImportError as exc:
            print(f"[adaptive] zarr not available; skipping demo_data.zip: {exc}")
            return
        all_actions = (
            np.concatenate(action_arrays, axis=0).astype(np.float32)
            if action_arrays
            else np.zeros((0, 10), dtype=np.float32)
        )
        episode_ends = np.cumsum([arr.shape[0] for arr in action_arrays], dtype=np.int64)
        zip_path = os.path.join(save_dir, "demo_data.zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        store = zarr.ZipStore(zip_path, mode="w")
        try:
            root = zarr.group(store=store, overwrite=True)
            data = root.create_group("data")
            data.create_dataset(
                "action",
                data=all_actions,
                chunks=(min(1024, max(1, all_actions.shape[0])), all_actions.shape[1]) if all_actions.size else (1, 10),
                dtype="float32",
            )
            meta = root.create_group("meta")
            meta.create_dataset("episode_ends", data=episode_ends, dtype="int64")
        finally:
            store.close()
        print(
            f"[adaptive] wrote {zip_path} (action shape={all_actions.shape}, "
            f"num_episodes={len(episode_ends)})"
        )

    '''
    test env
    '''
    def test_env(self, pose, eval=False):
        batch_size = pose.shape[0]
        handle_pos = pose[:,:7].clone()
        button_pos = pose[:,7:].clone()
        print(handle_pos)
        print(button_pos)
        '''
        two manipulation choice 1.pull handle open door 2.push button then pull handle open door
        '''
        self.env.reset()
        flag = False
        if flag:
            handle_pos[:, 0] += self.env.gripper_length*2
            for i in range(30):
                self.env.step(handle_pos)
            handle_pos[:, 0] -= self.env.gripper_length + 0.014
            for i in range(30):
                self.env.step(handle_pos)
            self.env.gripper = True
            for i in range(10):
                self.env.step(handle_pos)
            
            down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            step_size = 0.045
            for i in range(10):
                print("step_{}".format(i))
                handle_q = self.env.rigid_body_tensor[:, 3:7]
                open_dir = quat_axis(handle_q, axis=2)
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                pred_p = cur_p + open_dir * step_size
                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)
        else:
            hand_pose = self.env.hand_rigid_body_tensor[:,:7]
            for i in range(1000):
                self.env.step(hand_pose)
            init_handle_pose = handle_pos.clone()
            init_handle_pose[:, 0] += self.env.gripper_length*2
            for i in range(2):
                for j in range(15):
                    self.env.step(init_handle_pose)
            init_handle_pose[:, 0] -= self.env.gripper_length + 0.014
            for i in range(2):
                for j in range(15):
                    self.env.step(init_handle_pose)
            self.env.gripper = True
            for i in range(50):
                self.env.step(init_handle_pose)

            self.env.gripper = False
            for i in range(15):
                self.env.step(init_handle_pose)
            # push button
            button_pos[:, 0] += self.env.gripper_length*2 + 0.012
            for i in range(30):
                self.env.step(button_pos)
            button_pos[:, 0] -= self.env.gripper_length
            for i in range(30):
                self.env.step(button_pos)
            self.env.gripper = True
            for i in range(15):
                self.env.step(button_pos)
            button_pos[:, 0] -= 0.03
            for i in range(15):
                self.env.step(button_pos)
            self.env.gripper = False
            handle_pos[:, 0] += self.env.gripper_length*2
            for i in range(30):
                self.env.step(handle_pos)
            handle_pos[:, 0] -= self.env.gripper_length + 0.014
            for i in range(30):
                self.env.step(handle_pos)
            self.env.gripper = True
            for i in range(15):
                self.env.step(handle_pos)
            
            down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
            step_size = 0.045
            for i in range(10):
                print("step_{}".format(i))
                handle_q = self.env.rigid_body_tensor[:, 3:7]
                open_dir = quat_axis(handle_q, axis=2)
                cur_p = self.env.hand_rigid_body_tensor[:, :3]
                pred_p = cur_p + open_dir * step_size
                pred_q = quat_mul(handle_q, down_q)
                pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                for j in range(15):
                    self.env.step(pred_pose)
    
    def diffusion_evaluate(self, diffusion):
        eps_num = self.cfg["task"]["num_episode"]
        succ_cnt = 0
        succ_rate = []
        language_embedding_bank = self._load_eval_language_embedding_bank(diffusion)
        eval_save_dir = self._build_eval_save_dir()
        adaptive_cfg = self.cfg.get("task", {}).get("adaptive_language", {}) or {}
        adaptive_enable = bool(adaptive_cfg.get("enable", False))
        os.makedirs(eval_save_dir, exist_ok=True)
        eval_config_path = self._copy_eval_config(eval_save_dir)
        eval_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
        eval_timer_start = time.perf_counter()
        eval_metrics = {
            "schema_version": "v1",
            "run_dir": eval_save_dir,
            "config_path": eval_config_path,
            "started_at": eval_started_at,
            "finished_at": None,
            "episodes": [],
            "overall": {},
        }
        self._update_eval_metrics_summary(
            eval_metrics,
            0.0,
            status="running",
            num_envs=self.env.num_envs,
        )
        self._write_eval_metrics(eval_save_dir, eval_metrics)
        self._init_video_recorder(eval_save_dir)

        if adaptive_enable and language_embedding_bank is None:
            raise RuntimeError(
                "task.adaptive_language.enable is True but the policy does not provide a language embedding bank "
                "(set model.use_language_conditioning=True and supply language_embedding_dict.json)."
            )
        adaptive_states = None
        adaptive_asker = None
        adaptive_camera_id = 0
        adaptive_rng = None
        adaptive_expanded_chains = None
        adaptive_chain_priority_ids = None
        adaptive_dump_state = None
        # save_inference_data is independent of adaptive_language.enable: a pure
        # data-collection rollout (closed-loop Stage 2) wants the dump and skips
        # the asker.
        adaptive_save_inference_data = bool(
            self.cfg.get("task", {}).get("save_inference_data", False)
        )
        max_retry_rounds = int(adaptive_cfg.get("max_retry_rounds", 3))
        if adaptive_enable or adaptive_save_inference_data:
            template_path, task_spec = self.load_task_language_template("microwave")
            expanded_minimal_chains = self.build_expanded_minimal_chains(task_spec["minimal_chains"])
            adaptive_expanded_chains = expanded_minimal_chains
            if adaptive_enable:
                num_chains = int(language_embedding_bank.shape[0])
                if len(expanded_minimal_chains) != num_chains:
                    raise RuntimeError(
                        f"adaptive_language: expanded_minimal_chains length {len(expanded_minimal_chains)} "
                        f"!= embedding bank size {num_chains}; checkpoint and language_template are inconsistent."
                    )
        if adaptive_enable:
            from manipulation.adaptive_language_asker import (
                AdaptiveLanguageAsker,
                AdaptiveLanguageAskerConfig,
                AdaptiveLanguageState,
            )
            asker_cfg = AdaptiveLanguageAskerConfig(adaptive_cfg.get("asker", {}))
            adaptive_asker = AdaptiveLanguageAsker(
                asker_cfg, task_spec, expanded_minimal_chains
            )
            adaptive_chain_priority_ids = rank_expanded_minimal_chain_ids(
                expanded_minimal_chains
            )
            priority_parts = []
            for chain_id in adaptive_chain_priority_ids:
                chain = " -> ".join(expanded_minimal_chains[chain_id])
                score = score_language_chain_for_inference(
                    expanded_minimal_chains[chain_id],
                    expanded_minimal_chains,
                )
                priority_parts.append(
                    f"{chain_id}:{chain}|unique={score['unique_ground_truth_rate']:.2f}"
                    f"|worst={int(score['worst_case_candidate_count'])}"
                )
            adaptive_states = [
                AdaptiveLanguageState(num_chains=int(language_embedding_bank.shape[0]))
                for _ in range(self.env.num_envs)
            ]
            adaptive_camera_id = self._adaptive_resolve_camera_id(asker_cfg)
            seed = int(self.cfg.get("seed", 0)) if self.cfg.get("seed", None) is not None else 0
            adaptive_rng = random.Random(seed)
            print(
                f"[adaptive] enabled: platform={asker_cfg.platform}, model={asker_cfg.model}, "
                f"num_chains={int(language_embedding_bank.shape[0])}, "
                f"num_envs={self.env.num_envs}, camera_id={adaptive_camera_id}"
            )
            print(
                "[adaptive] chain inference priority: "
                + ", ".join(priority_parts)
            )
        if adaptive_save_inference_data:
            if not self.cfg.get("env", {}).get("collectRGBVideo", False):
                print(
                    "[dump] save_inference_data=true requires env.collectRGBVideo=true; "
                    "videos will be missing from the dump."
                )
            adaptive_dump_state = self._adaptive_init_inference_dump(
                eval_save_dir, task_spec, expanded_minimal_chains
            )
            print(
                f"[dump] save_inference_data: writing eval_video2prompt-compatible artifacts under {eval_save_dir} "
                f"(adaptive_language={'on' if adaptive_enable else 'off'})"
            )

        for eps in range(eps_num):
            print("eps_{}".format(eps+1))
            episode_started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
            episode_timer_start = time.perf_counter()
            done_flag = [False] * self.env.num_envs
            success_step = [None] * self.env.num_envs
            time_to_success = [None] * self.env.num_envs
            episode_completion_step = None
            step = 0
            episode_results = None
            sampled_language_idx = None
            chain_ids = None
            adaptive_asker_records = [None] * self.env.num_envs if adaptive_enable else None

            if adaptive_enable:
                chain_ids = []
                for s in adaptive_states:
                    if s.locked_chain_id is not None:
                        chain_ids.append(s.locked_chain_id)
                        s.current_chain_id = s.locked_chain_id
                    else:
                        cid = s.pick_next(
                            adaptive_rng,
                            priority_ids=adaptive_chain_priority_ids,
                        )
                        s.current_chain_id = cid
                        s.tried_chain_ids.add(cid)
                        chain_ids.append(cid)
                episode_language_embedding = self._select_language_embedding_per_env(
                    language_embedding_bank, chain_ids
                )
                print(
                    f"[adaptive] eps {eps + 1} chain ids per env: " +
                    ", ".join(
                        f"env{env_id}={'L' if s.locked_chain_id is not None else 'T'}{cid}"
                        for env_id, (s, cid) in enumerate(zip(adaptive_states, chain_ids))
                    )
                )
            else:
                episode_language_embedding, sampled_language_idx = self._sample_episode_language_embedding(
                    language_embedding_bank, self.env.num_envs
                )
                if sampled_language_idx is not None:
                    print(f'episode {eps + 1} language embedding id: {sampled_language_idx}')

            if adaptive_enable and eps > 0 and any(s.frozen_clock_wise is not None for s in adaptive_states):
                override = [float(s.frozen_clock_wise) for s in adaptive_states]
                self.env.reset(clock_wise_override=override)
            else:
                self.env.reset(clock_same=False)
            if adaptive_enable and eps == 0:
                cw = self.env.clock_wise.detach().cpu().numpy().tolist()
                for env_id, s in enumerate(adaptive_states):
                    s.frozen_clock_wise = float(cw[env_id])
                print(f"[adaptive] eps 1 frozen clock_wise per env: {[s.frozen_clock_wise for s in adaptive_states]}")
            episode_clock_wise = [
                float(value)
                for value in self.env.clock_wise.detach().cpu().numpy().tolist()
            ]

            self.env.gripper = torch.zeros((self.env.num_envs,1),device=self.env.device)
            if self.video_recorder is not None:
                self.video_recorder.start_episode(eps)
                self._record_video_frame()

            obs = self.env.collect_diff_data()
            pcs, env_state = obs_wrapper(obs)
            pcs_deque = collections.deque([pcs] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)
            env_state_deque = collections.deque([env_state] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon)

            episode_action_logs = (
                [[] for _ in range(self.env.num_envs)]
                if (adaptive_enable or adaptive_save_inference_data)
                else None
            )

            try:
                while step <= 32:
                    action = diffusion.infer_action_with_seg(
                        pcs_deque,
                        env_state_deque,
                        language_embedding=episode_language_embedding,
                    ).detach()
                    action = action[:, :diffusion.args.action_horizon, :]
                    step += diffusion.args.action_horizon
                    for act in range(action.shape[1]):
                        quat = self.rotate_6d_to_quat(action[:, act, 3:9])
                        pre_action = torch.cat([action[:, act, :3], quat], dim=-1)
                        self.env.gripper = (action[:, act, -1] > 0.5).unsqueeze(-1).int()
                        for j in range(15):
                            self.env.step(pre_action)

                        self._record_video_frame()
                        self.env.actions = action[:, act, :]
                        if episode_action_logs is not None:
                            action_step = action[:, act, :].detach().cpu().numpy()
                            for env_id in range(self.env.num_envs):
                                episode_action_logs[env_id].append(action_step[env_id])
                        obs = self.env.collect_diff_data()
                        pcs, env_state = obs_wrapper(obs)
                        pcs_deque.append(pcs)
                        env_state_deque.append(env_state)

                    for env_id in range(self.env.num_envs):
                        if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi/7).cpu().item() and not done_flag[env_id]:
                            done_flag[env_id] = True
                            success_step[env_id] = int(step)
                            time_to_success[env_id] = float(time.perf_counter() - episode_timer_start)
                            succ_cnt += 1
                            print(f"Env {env_id} Succeeded")
                    if episode_completion_step is None and all(done_flag):
                        episode_completion_step = int(step)

                episode_results = ["success" if success else "failure" for success in done_flag]
            finally:
                if self.video_recorder is not None:
                    if episode_results is None:
                        self.video_recorder.discard_episode()
                    elif adaptive_enable or adaptive_save_inference_data:
                        # Flatten to rgb_videos/episode_<eps>/env_<id>_cam_<cam>.mp4. The
                        # online asker (adaptive_enable) needs this layout to read the
                        # video by env_id; offline replay through eval_video2prompt.py
                        # (save_inference_data) likewise expects it via locate_video().
                        self.video_recorder.finish_episode(env_results=None)
                    else:
                        self.video_recorder.finish_episode(episode_results)

            rollout_elapsed_sec = float(time.perf_counter() - episode_timer_start)
            rollout_step_count = int(step)

            if adaptive_enable and episode_results is not None:
                recategorize = []
                lock_on_env_success = bool(adaptive_cfg.get("asker", {}).get("lock_on_env_success", False))
                for env_id, s in enumerate(adaptive_states):
                    if s.locked_chain_id is not None:
                        adaptive_asker_records[env_id] = {
                            "skipped": True,
                            "reason": "already_locked",
                            "locked_chain_id": int(s.locked_chain_id),
                            "current_chain_id": int(s.current_chain_id) if s.current_chain_id is not None else None,
                            "tried_chain_ids": sorted(int(item) for item in s.tried_chain_ids),
                            "sweep_count": int(s.sweep_count),
                        }
                        recategorize.append("success")
                        continue
                    video_path = self._adaptive_video_path(eval_save_dir, eps, env_id, adaptive_camera_id)
                    actions_arr = (
                        np.stack(episode_action_logs[env_id])
                        if episode_action_logs and episode_action_logs[env_id]
                        else None
                    )
                    asker_success, asker_chain_id = adaptive_asker.ask(
                        video_path=video_path if os.path.exists(video_path) else None,
                        action_array=actions_arr,
                        env_id=env_id,
                        done_flag=bool(done_flag[env_id]),
                        frozen_clock_wise=s.frozen_clock_wise,
                    )
                    ground_truth_chain_id = self._microwave_ground_truth_chain_id(
                        episode_clock_wise[env_id]
                    )
                    reasonable_prediction_chain_ids = self._reasonable_prediction_chain_ids(
                        s.current_chain_id,
                        ground_truth_chain_id,
                        adaptive_expanded_chains,
                    )
                    asker_prompt_correct = self._asker_prompt_correctness(
                        asker_chain_id,
                        ground_truth_chain_id,
                        reasonable_prediction_chain_ids,
                    )
                    print(
                        f"[adaptive] eps {eps + 1} env {env_id} done_flag={done_flag[env_id]} "
                        f"asker_success={asker_success} asker_chain_id={asker_chain_id} "
                        f"ground_truth_chain_id={ground_truth_chain_id} "
                        f"reasonable_prediction_chain_ids={reasonable_prediction_chain_ids} "
                        f"asker_prompt_correct={asker_prompt_correct} "
                        f"current_chain_id={s.current_chain_id} tried={sorted(s.tried_chain_ids)} "
                        f"sweep={s.sweep_count}"
                    )
                    if asker_success and asker_chain_id is not None:
                        s.locked_chain_id = asker_chain_id
                        recategorize.append("success")
                    elif lock_on_env_success and done_flag[env_id]:
                        s.locked_chain_id = s.current_chain_id
                        recategorize.append("success")
                    else:
                        if s.sweep_count >= max_retry_rounds:
                            fallback = asker_chain_id if asker_chain_id is not None else s.current_chain_id
                            print(
                                f"[adaptive] eps {eps + 1} env {env_id} max_retry_rounds={max_retry_rounds} reached; "
                                f"force-locking chain_id={fallback}"
                            )
                            s.locked_chain_id = fallback
                            recategorize.append("success" if done_flag[env_id] else "failure")
                        else:
                            recategorize.append("success" if done_flag[env_id] else "failure")
                    adaptive_asker_records[env_id] = {
                        "skipped": False,
                        "asker_success": bool(asker_success),
                        "asker_chain_id": int(asker_chain_id) if asker_chain_id is not None else None,
                        "ground_truth_chain_id": int(ground_truth_chain_id),
                        "reasonable_prediction_chain_ids": reasonable_prediction_chain_ids,
                        "asker_prompt_correct": asker_prompt_correct,
                        "asker_reasonable_prediction_correct": (
                            asker_prompt_correct is True
                            if asker_chain_id is not None
                            else None
                        ),
                        "asker_strict_ground_truth_correct": (
                            int(asker_chain_id) == int(ground_truth_chain_id)
                            if asker_chain_id is not None
                            else None
                        ),
                        "current_chain_id": int(s.current_chain_id) if s.current_chain_id is not None else None,
                        "locked_chain_id": int(s.locked_chain_id) if s.locked_chain_id is not None else None,
                        "tried_chain_ids": sorted(int(item) for item in s.tried_chain_ids),
                        "sweep_count": int(s.sweep_count),
                        "video_path": video_path if os.path.exists(video_path) else None,
                    }
                # When dumping inference data for offline replay through
                # scripts/eval_video2prompt.py, keep the flat
                # rgb_videos/episode_<eps>/ layout that locate_video expects.
                if (
                    not adaptive_save_inference_data
                    and bool(adaptive_cfg.get("asker", {}).get("recategorize_videos", True))
                ):
                    self._adaptive_recategorize_videos(eval_save_dir, eps, recategorize)

            # Dump per-(eps, env) record. This runs whenever save_inference_data is on,
            # regardless of whether the online asker (adaptive_language) was running.
            if (
                adaptive_save_inference_data
                and adaptive_dump_state is not None
                and episode_results is not None
            ):
                for env_id in range(self.env.num_envs):
                    actions_arr = (
                        np.stack(episode_action_logs[env_id])
                        if episode_action_logs and episode_action_logs[env_id]
                        else None
                    )
                    if chain_ids is not None:
                        chain_id_used = chain_ids[env_id]
                    elif sampled_language_idx is not None:
                        chain_id_used = sampled_language_idx
                    else:
                        chain_id_used = None
                    adaptive_state = (
                        adaptive_states[env_id] if adaptive_states is not None else None
                    )
                    asker_record = (
                        adaptive_asker_records[env_id]
                        if adaptive_asker_records is not None
                        else None
                    )
                    self._adaptive_record_inference_episode(
                        adaptive_dump_state,
                        eps_idx=eps,
                        env_id=env_id,
                        clock_wise=episode_clock_wise[env_id],
                        chain_id_used=chain_id_used,
                        done_flag=bool(done_flag[env_id]),
                        adaptive_state=adaptive_state,
                        action_array=actions_arr,
                        adaptive_asker_record=asker_record,
                        expanded_minimal_chains=adaptive_expanded_chains,
                    )

            episode_elapsed_sec = float(time.perf_counter() - episode_timer_start)
            final_open_dof = [
                float(value)
                for value in self.env.one_dof_tensor[:, 0].detach().cpu().numpy().tolist()
            ]
            env_metrics = []
            for env_id in range(self.env.num_envs):
                env_record = {
                    "env_id": int(env_id),
                    "episode_id": int(eps),
                    "success": bool(done_flag[env_id]),
                    "success_step": success_step[env_id],
                    "time_to_success_sec": time_to_success[env_id],
                    "episode_completion_step": episode_completion_step,
                    "episode_rollout_step_count": rollout_step_count,
                    "episode_elapsed_sec": episode_elapsed_sec,
                    "rollout_elapsed_sec": rollout_elapsed_sec,
                    "clock_wise": episode_clock_wise[env_id],
                    "ground_truth_chain_id": self._microwave_ground_truth_chain_id(
                        episode_clock_wise[env_id]
                    ),
                    "final_open_dof": final_open_dof[env_id],
                }
                if adaptive_enable:
                    env_record["language_chain_id"] = int(chain_ids[env_id]) if chain_ids is not None else None
                    env_record["adaptive"] = adaptive_asker_records[env_id]
                else:
                    env_record["language_chain_id"] = int(sampled_language_idx) if sampled_language_idx is not None else None
                env_metrics.append(env_record)

            episode_success_count = int(sum(1 for item in done_flag if item))
            cur_rate = episode_success_count/(self.env.num_envs)
            episode_record = {
                "episode_id": int(eps),
                "episode_number": int(eps + 1),
                "started_at": episode_started_at,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                "elapsed_sec": episode_elapsed_sec,
                "rollout_elapsed_sec": rollout_elapsed_sec,
                "rollout_step_count": rollout_step_count,
                "completion_step": episode_completion_step,
                "success_count": episode_success_count,
                "num_envs": int(self.env.num_envs),
                "success_rate": float(cur_rate),
                "sampled_language_id": int(sampled_language_idx) if sampled_language_idx is not None else None,
                "per_env_language_ids": [int(item) for item in chain_ids] if chain_ids is not None else None,
                "envs": env_metrics,
            }
            eval_metrics["episodes"].append(episode_record)
            self._update_eval_metrics_summary(
                eval_metrics,
                total_elapsed_sec=time.perf_counter() - eval_timer_start,
                status="running",
                num_envs=self.env.num_envs,
                expanded_minimal_chains=adaptive_expanded_chains,
            )
            self._write_eval_metrics(eval_save_dir, eval_metrics)

            print(f"Eps {eps+1}, current succ rate {cur_rate}")
            succ_rate.append(cur_rate)
            succ_cnt = 0
        print(f"Average Success rate: {np.mean(succ_rate)}")
        print(f"Success rate std: {np.std(succ_rate)}")
        if adaptive_enable:
            print(
                "[adaptive] final per-env locked_chain_id: " +
                ", ".join(
                    f"env{env_id}=cw={s.frozen_clock_wise}|locked={s.locked_chain_id}|tried={sorted(s.tried_chain_ids)}|sweep={s.sweep_count}"
                    for env_id, s in enumerate(adaptive_states)
                )
            )
        eval_metrics["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())
        self._update_eval_metrics_summary(
            eval_metrics,
            total_elapsed_sec=time.perf_counter() - eval_timer_start,
            status="completed",
            num_envs=self.env.num_envs,
            expanded_minimal_chains=adaptive_expanded_chains,
        )
        self._write_eval_metrics(eval_save_dir, eval_metrics)
        print(f"Saved eval metrics: {(Path(eval_save_dir) / 'eval_metrics.json').absolute()}")
        if adaptive_save_inference_data and adaptive_dump_state is not None:
            self._adaptive_finalize_inference_dump(eval_save_dir, adaptive_dump_state)
        self.video_recorder = None
        return

    def process_data(self, goal_pos):
        obs = self.env.collect_diff_data()
        pc, env_state = obs_wrapper(obs)
        goal_pos = self.action_process(goal_pos)
        # self.env.actions = goal_pos
        if self.env.gripper[0,0].cpu().item() == 1:
            temp = torch.ones((self.env.num_envs,1),device=self.env.device)
        else:
            temp = torch.zeros((self.env.num_envs,1),device=self.env.device)
        action_with_gripper = torch.cat([goal_pos, temp],dim=-1)
        self.env.actions = action_with_gripper
        for env_id in range(self.env.num_envs):
            self.eps_buffer[env_id].add(pc[env_id], env_state[env_id],action_with_gripper[env_id])
            self.append_frame_label(env_id)
        self._record_video_frame()

    def collect_manip_data(self):
        # move to the handle
        eps_num = self.cfg["task"]["num_episode"]
        policy = self.cfg["task"]["policy"]
        template_path, task_spec = self.load_task_language_template("microwave")
        expanded_minimal_chains = self.build_expanded_minimal_chains(task_spec["minimal_chains"])

        demo_buffer = Experience()
        trajectory_records = []
        frame_records = []
        saved_episode_id = 0

        save_dir = self._build_collect_save_dir()
        self._prepare_save_dir(save_dir, "Collection")
        self._init_video_recorder(save_dir)
        for eps in range(eps_num):
            self.eps_buffer = [Episode_Buffer() for _ in range(self.env.num_envs)]
            self.init_episode_frame_records(self.env.num_envs)
            print("eps_{}".format(eps+1))
            self.env.reset()
            if self.video_recorder is not None:
                self.video_recorder.start_episode(eps)
            ori_pose = self.env.get_adjust_hand_pose()
            pose = ori_pose.clone()
            handle_pos = pose[:,:7].clone()
            button_pos = pose[:,7:].clone()
            self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
            episode_start_with_pull = False
            try:
                if policy == "succ":
                    # succ policy under gt state
                    if self.env.clock_wise[0] == 1:
                        episode_start_with_pull = False
                        # cannot directly open door
                        # push button
                        self.set_current_step(0, "按按钮")
                        button_pos[:, 0] += self.env.gripper_length*2 + 0.012
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        button_pos[:, 0] -= self.env.gripper_length
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                        self.process_data(button_pos)
                        for j in range(15):
                            self.env.step(button_pos)
                        button_pos[:, 0] -= 0.03
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)

                        self.set_current_step(1, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length*2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)
                        
                        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                        step_size = 0.045
                        for i in range(8):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)
                    else:
                        episode_start_with_pull = True
                        # directly open door
                        self.set_current_step(0, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length*2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                        self.process_data(handle_pos)
                        for i in range(15):
                            self.env.step(handle_pos)
                        
                        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                        step_size = 0.045
                        for i in range(8):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)
                else:
                    # ada demo
                    down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                    step_size = 0.045
                    start_with_pull = np.random.rand() < 0.5
                    episode_start_with_pull = start_with_pull

                    if start_with_pull:
                        self.set_current_step(0, "拉门")
                        handle_pos[:, 0] += self.env.gripper_length*2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                        self.process_data(handle_pos)
                        for i in range(15):
                            self.env.step(handle_pos)

                        for i in range(2):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)
                    
                    if start_with_pull and self.env.clock_wise[0] == 0:
                        # continue open door
                        self.set_current_step(0, "拉门")
                        for i in range(6):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)
                    else:
                        button_step_idx = 1 if start_with_pull else 0
                        pull_step_idx = 2 if start_with_pull else 1

                        self.set_current_step(button_step_idx, "按按钮")
                        self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)
                        keep_pose = self.env.hand_rigid_body_tensor.clone()
                        self.process_data(keep_pose)
                        for i in range(15):
                            self.env.step(keep_pose)
                        
                        keep_pose[:, 0] += self.env.gripper_length
                        for i in range(2):
                            self.process_data(keep_pose)
                            for j in range(15):
                                self.env.step(keep_pose)
                        # push button
                        button_pos[:, 0] += self.env.gripper_length*2 + 0.012
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        button_pos[:, 0] -= self.env.gripper_length
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                        self.process_data(button_pos)
                        for j in range(15):
                            self.env.step(button_pos)
                        button_pos[:, 0] -= 0.03
                        for i in range(2):
                            self.process_data(button_pos)
                            for j in range(15):
                                self.env.step(button_pos)
                        self.env.gripper = torch.zeros((self.env.num_envs,1), device=self.env.device)

                        self.set_current_step(pull_step_idx, "拉门")
                        keep_pose = self.env.hand_rigid_body_tensor.clone()
                        keep_pose[:, 0] += self.env.gripper_length
                        self.process_data(keep_pose)
                        for j in range(15):
                            self.env.step(keep_pose)
                        handle_pos = pose[:,:7].clone()
                        handle_pos[:, 0] += self.env.gripper_length*2
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        handle_pos[:, 0] -= self.env.gripper_length + 0.014
                        for i in range(2):
                            self.process_data(handle_pos)
                            for j in range(15):
                                self.env.step(handle_pos)
                        self.env.gripper = torch.ones((self.env.num_envs,1), device=self.env.device)
                        self.process_data(handle_pos)
                        for j in range(15):
                            self.env.step(handle_pos)
                        
                        down_q = torch.stack(self.env.num_envs * [torch.tensor([0.7071068, 0.7071068, 0, 0])]).to(self.env.device).view((self.env.num_envs, 4))
                        step_size = 0.045
                        for i in range(8):
                            handle_q = self.env.rigid_body_tensor[:, 3:7]
                            open_dir = quat_axis(handle_q, axis=2)
                            cur_p = self.env.hand_rigid_body_tensor[:, :3]
                            pred_p = cur_p + open_dir * step_size
                            pred_q = quat_mul(handle_q, down_q)
                            pred_pose = torch.cat([pred_p, pred_q], dim=-1).float()
                            self.process_data(pred_pose)
                            for j in range(15):
                                self.env.step(pred_pose)

                for env_id in range(self.env.num_envs):
                    if (torch.abs(self.env.one_dof_tensor[env_id, 0]) > np.pi/7).cpu().item():
                        demo_buffer.append(self.eps_buffer[env_id])
                        traj_record = self._build_microwave_trajectory_label(
                            env_id=env_id,
                            start_with_pull=episode_start_with_pull,
                            expanded_minimal_chains=expanded_minimal_chains,
                        )
                        traj_record["episode_id"] = saved_episode_id
                        traj_record["round_idx"] = eps
                        traj_record["env_id"] = env_id
                        frame_start = len(frame_records)
                        env_frames = self._episode_frame_records[env_id]
                        frame_records.extend(env_frames)
                        frame_end = len(frame_records)
                        traj_record["frame_range"] = [frame_start, frame_end]
                        trajectory_records.append(traj_record)
                        saved_episode_id += 1
                        print(f"Env {env_id} Succeeded")
            finally:
                if self.video_recorder is not None:
                    self.video_recorder.finish_episode()
                self.clear_episode_frame_records()

        if self.cfg['env']['collectData']:
            save_path = save_dir + '/demo_data.zip'
            os.makedirs(save_dir, exist_ok=True)
            demo_buffer.save(save_path)
            self.save_language_sidecars(
                save_dir=save_dir,
                template_path=template_path,
                task_name="microwave",
                task_spec=task_spec,
                expanded_minimal_chains=expanded_minimal_chains,
                trajectory_records=trajectory_records,
                frame_records=frame_records,
            )
        self.video_recorder = None
        
    def action_choose(self,t,index,one_motion,two_motion):
        if "r" in self.env.action_chosen[index]:
            if one_motion > 0.0001:
                self.env.action_chosen[index,t] = "z"
                return "z"
            else:
                if two_motion > 0.05:
                    res = random.choice(["z","o"])
                    self.env.action_chosen[index,t] = res
                    return res
                else:
                    if "z" == self.env.action_chosen[index,t-1]:
                        self.env.action_chosen[index,t]= "o"
                        return "o"
                    else:
                        self.env.action_chosen[index,t]= "z"
                        return "z"
        else:
            if one_motion > 0.0001:
                # lift up is successful
                self.env.action_chosen[index,t] = "z"
                return "z"
            else:
                if two_motion > 0.05:
                    # did not lift up, but rotate is successful
                    res = random.choice(["z","o"])
                    self.env.action_chosen[index,t] = res
                    return res
                else:
                    # neither lift up, nor rotate
                    if "o" == self.env.action_chosen[index,t-1]:
                        self.env.action_chosen[index,t]= "z"
                        return "z"
                    else:
                        if t == 0:
                            res = random.choice(["z","o","r"])
                            print("random")
                            self.env.action_chosen[index,t] = res
                            return res
                        elif "z" in self.env.action_chosen[index,t-1]:
                            if "o" in self.env.action_chosen[index]:
                                self.env.action_chosen[index,t]= "o"
                                return "o"
                            else:
                                res = random.choice(["o","r"])
                                self.env.action_chosen[index,t] = res
                                return res
                            
    def pc_normalize(self, pc):
        print("normalize pc", pc.shape)
        print("max_test", torch.max(pc))
        print("min_test", torch.min(pc))
        center = torch.mean(pc, dim=-2, keepdim=True)
        pc = pc - center
        m = torch.max(torch.norm(pc, p=2, dim=-1)).unsqueeze(-1)
        pc = pc / m
        print("max_test", torch.max(torch.norm(pc, p=2, dim=-1)))
        print("min_test", torch.min(torch.norm(pc, p=2, dim=-1)))    
        return pc
