"""Task-agnostic manipulation base class.

Subclasses implement task-specific demonstrations (``collect_manip_data``,
``test_env``, etc.) and override the small set of hooks below to plug their
task semantics (success criterion, frozen-state freezing, canonical chain
mapping) into the shared eval/dump infrastructure.

The asker plumbing, video recorder, language-embedding bank loader and the
``diffusion_evaluate`` adaptive eval loop all live here so that adding a new
task is mostly a matter of writing the data-collection script + 4-5 short
hook overrides.
"""

from abc import abstractclassmethod
from envs.base_env import BaseEnv
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Optional

import collections
import json
import os
import random
import shutil
import time

import numpy as np
import pytorch3d.transforms as tf
import torch
import yaml

import av

from dataset.dataset import obs_wrapper
from manipulation.language_chain_utils import (
    infer_reasonable_prediction_chains,
    rank_expanded_minimal_chain_ids,
    score_language_chain_for_inference,
)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns ``torch.sqrt(torch.max(0, x))`` with zero subgradient at 0."""
    positive_mask = x > 0
    safe_x = torch.where(positive_mask, x, torch.ones_like(x))
    return torch.where(positive_mask, torch.sqrt(safe_x), torch.zeros_like(x))


# ---------------------------------------------------------------------------
# Video recording (task-agnostic)
# ---------------------------------------------------------------------------


class Mp4VideoWriter:
    """Single-file H.264 mp4 writer."""

    def __init__(self, output_path, width, height, fps, codec, options=None):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        if options:
            self.stream.options = {
                key: str(value) for key, value in options.items() if value is not None
            }

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
    """Per-(env, camera) mp4 writer with episode-level start/finish/discard."""

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
            self.camera_ids = (
                list(configured_camera_ids)
                if configured_camera_ids
                else list(range(num_fixed_cameras))
            )
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
                    src_path = os.path.join(
                        episode_dir, f"env_{env_id:02d}_cam_{camera_id:02d}.mp4"
                    )
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


# ---------------------------------------------------------------------------
# BaseManipulation
# ---------------------------------------------------------------------------


class BaseManipulation:

    def __init__(self, env: BaseEnv, cfg: dict, logger: Logger):
        self.env = env
        self.cfg = cfg
        self.logger = logger
        self._current_step_index = 0
        self._current_step_operation = ""
        self._episode_frame_records = None
        self.video_recorder = None

    # ----- Existing language template / chain helpers (preserved) -----

    def get_language_template_path(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(cur_dir, "..", "cfg", "language_template.json"))

    def parse_chain_text(self, chain_text):
        return [stage.strip() for stage in chain_text.split("->") if stage.strip()]

    def load_task_language_template(self, task_name, template_path=None):
        if template_path is None:
            template_path = self.get_language_template_path()
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Language template not found: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)

        tasks = template.get("tasks", {})
        if task_name not in tasks:
            raise KeyError(f"Task '{task_name}' missing in language template: {template_path}")
        task_spec = tasks[task_name]
        if (
            "command" not in task_spec
            or "operation_set" not in task_spec
            or "minimal_chains" not in task_spec
        ):
            raise KeyError(f"Task '{task_name}' missing required fields in language template")
        return template_path, task_spec

    def build_expanded_minimal_chains(self, minimal_chains):
        # Default strategy: parse concrete chains as-is. Tasks with Nx can override.
        return [self.parse_chain_text(chain_text) for chain_text in minimal_chains]

    def match_command_chains(self, attempt_chain, stage_status, expanded_minimal_chains):
        first_fail = None
        for i, ok in enumerate(stage_status):
            if not ok:
                first_fail = i
                break
        end_idx = first_fail if first_fail is not None else len(attempt_chain) - 1
        prefix = attempt_chain[:end_idx + 1]

        command_chains = []
        command_chain_ids = []
        for idx, chain in enumerate(expanded_minimal_chains):
            if len(chain) >= len(prefix) and chain[:len(prefix)] == prefix:
                command_chains.append(chain)
                command_chain_ids.append(idx)

        if len(command_chains) == 0:
            raise RuntimeError(
                f"No command_chains matched prefix={prefix}; task logic or labels are inconsistent"
            )
        return command_chains, command_chain_ids

    def relative_path_from(self, path, start_dir):
        return os.path.relpath(path, start=start_dir)

    def set_current_step(self, step_index, step_operation):
        self._current_step_index = int(step_index)
        self._current_step_operation = step_operation

    def init_episode_frame_records(self, num_envs):
        self._episode_frame_records = [[] for _ in range(num_envs)]

    def clear_episode_frame_records(self):
        self._episode_frame_records = None

    def append_frame_label(self, env_id):
        if self._episode_frame_records is None:
            return
        self._episode_frame_records[env_id].append({
            "step_index": self._current_step_index,
            "step_operation": self._current_step_operation,
        })

    def save_language_sidecars(self,
                               save_dir,
                               template_path,
                               task_name,
                               task_spec,
                               expanded_minimal_chains,
                               trajectory_records,
                               frame_records):
        relative_template_path = self.relative_path_from(template_path, save_dir)
        attempt_chain_count_map = {}
        attempt_chain_order = []
        for record in trajectory_records:
            attempt_chain = record.get("attempt_chain")
            if attempt_chain is None:
                continue
            chain_key = tuple(attempt_chain)
            if chain_key not in attempt_chain_count_map:
                attempt_chain_count_map[chain_key] = 0
                attempt_chain_order.append(chain_key)
            attempt_chain_count_map[chain_key] += 1

        attempt_chain_counts = [
            {
                "attempt_chain": list(chain_key),
                "count": attempt_chain_count_map[chain_key],
            }
            for chain_key in attempt_chain_order
        ]

        expanded_payload = {
            "schema_version": "v1",
            "generated_from": relative_template_path,
            "task": task_name,
            "command": task_spec["command"],
            "operation_set": task_spec["operation_set"],
            "expanded_minimal_chains": expanded_minimal_chains,
            "attempt_chain_counts": attempt_chain_counts,
        }
        for prompt_key in ("additional_prompt", "success_check_additional_prompt"):
            if prompt_key in task_spec:
                expanded_payload[prompt_key] = task_spec[prompt_key]

        with open(os.path.join(save_dir, "language_expanded.json"), "w", encoding="utf-8") as f:
            json.dump(expanded_payload, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_dir, "trajectory_language.jsonl"), "w", encoding="utf-8") as f:
            json.dump(trajectory_records, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_dir, "frame_language.jsonl"), "w", encoding="utf-8") as f:
            json.dump(frame_records, f, ensure_ascii=False, indent=2)

    @abstractclassmethod
    def collect_data(self, obs, eval=False):
        pass

    def action_process(self, pose):
        quat_isaac = pose[:, 3:7].float()
        quat_p3d = torch.cat([quat_isaac[:, 3:], quat_isaac[:, :3]], dim=-1)
        rotate_matix = tf.quaternion_to_matrix(quat_p3d)
        rotate_6d = tf.matrix_to_rotation_6d(rotate_matix)
        return torch.cat([pose[:, :3], rotate_6d], dim=-1)

    def rotate_6d_to_quat(self, rotate_6d):
        rotate_matix = tf.rotation_6d_to_matrix(rotate_6d)
        quat_p3d = self.matrix_to_quaternion(rotate_matix)
        quat_isaac = torch.cat([quat_p3d[:, 1:], quat_p3d[:, :1]], dim=-1)
        return quat_isaac

    @staticmethod
    def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = _sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        quat_by_rijk = torch.stack(
            [
                torch.stack(
                    [torch.square(q_abs[..., 0]), m21 - m12, m02 - m20, m10 - m01], dim=-1
                ),
                torch.stack(
                    [m21 - m12, torch.square(q_abs[..., 1]), m10 + m01, m02 + m20], dim=-1
                ),
                torch.stack(
                    [m02 - m20, m10 + m01, torch.square(q_abs[..., 2]), m12 + m21], dim=-1
                ),
                torch.stack(
                    [m10 - m01, m20 + m02, m21 + m12, torch.square(q_abs[..., 3])], dim=-1
                ),
            ],
            dim=-2,
        )

        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        indices = q_abs.argmax(dim=-1, keepdim=True)
        expand_dims = list(batch_dim) + [1, 4]
        gather_indices = indices.unsqueeze(-1).expand(expand_dims)
        out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
        return tf.standardize_quaternion(out)

    # =====================================================================
    # Task-specific HOOKS — subclasses override.
    # =====================================================================

    def task_name(self) -> str:
        """Cfg-level task name (e.g. ``"open_microwave"``)."""
        return self.cfg["task"]["task_name"]

    def language_template_task_name(self) -> str:
        """Key in ``language_template.json`` (e.g. ``"microwave"``).

        Default strips a leading ``"open_"`` from ``self.task_name()``;
        subclasses with non-conforming names should override.
        """
        name = self.task_name()
        return name[len("open_"):] if name.startswith("open_") else name

    def dataset_dir_suffix(self) -> str:
        """Optional task-specific suffix appended to eval/collect dir names.

        Microwave returns ``"clock<X>"``; tasks without per-task naming bits
        return ``""``.
        """
        return ""

    def reset_kwargs_initial(self) -> Dict[str, Any]:
        """kwargs for ``env.reset(...)`` on episode 1 / no-freeze episodes.

        Microwave returns ``{"clock_same": False}``; default is ``{}``.
        """
        return {}

    def capture_per_env_episode_state(self) -> List[Dict[str, Any]]:
        """Snapshot per-env episode state right after ``env.reset(...)``.

        Returned list has one dict per env. The same dict is used for:
        * canonical-chain lookup (``canonical_minimal_chain_for_state``),
        * freezing across episodes (``apply_frozen_states_to_reset_kwargs``),
        * extra log fields (``per_env_extra_log_fields``).

        Default returns empty dicts (no per-env state).
        """
        return [{} for _ in range(self.env.num_envs)]

    def apply_frozen_states_to_reset_kwargs(
        self, frozen_states: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Translate per-env frozen state dicts into ``env.reset(...)`` kwargs.

        Return ``None`` to skip the freeze branch entirely (each episode
        resets fresh). Default ``None``.
        """
        return None

    def canonical_minimal_chain_for_state(
        self, state: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Given per-env episode state, return the canonical minimal_chain.

        Used by the ground-truth asker and the inference dump's
        ``minimal_chain`` field. Default ``None`` (no oracle).
        """
        return None

    def task_success_for_env(self, env_id: int) -> bool:
        """Per-env task success criterion. Subclasses MUST override if they
        rely on ``diffusion_evaluate``."""
        raise NotImplementedError(
            f"{type(self).__name__}.task_success_for_env(env_id) not implemented"
        )

    def per_env_extra_log_fields(
        self, env_id: int, episode_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extra fields appended to env_record in eval_metrics.json.

        Microwave adds ``clock_wise`` + ``final_open_dof``. Default ``{}``.
        """
        return {}

    # =====================================================================
    # Save dirs
    # =====================================================================

    def _build_dir_name(self, prefix: str) -> str:
        parts = [
            f"{prefix}{self.task_name()}",
            self.cfg["task"]["policy"],
            str(self.cfg["env"]["asset"]["AssetNum"]),
            f"eps{self.cfg['task']['num_episode']}",
        ]
        suffix = self.dataset_dir_suffix()
        if suffix:
            parts.append(suffix)
        return "_".join(parts)

    def _build_collect_save_dir(self):
        return "./demo_data/" + self._build_dir_name(prefix="")

    def _build_eval_save_dir(self):
        run_ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return "./eval_data/" + self._build_dir_name(prefix="eval_") + "_" + run_ts

    def _prepare_save_dir(self, save_dir, purpose):
        if not os.path.exists(save_dir):
            return

        prompt_text = f"{purpose} output directory '{save_dir}' already exists. Overwrite it? [y/N]: "
        try:
            answer = input(prompt_text).strip().lower()
        except EOFError as exc:
            raise RuntimeError(
                f"{purpose} output directory '{save_dir}' already exists and overwrite confirmation could not be read."
            ) from exc

        if answer not in {"y", "yes"}:
            raise RuntimeError(f"Aborted to avoid overwriting existing data at '{save_dir}'.")

        shutil.rmtree(save_dir)

    # =====================================================================
    # Video recording wiring
    # =====================================================================

    def _init_video_recorder(self, save_dir):
        if not self.cfg["env"].get("collectRGBVideo", False):
            self.video_recorder = None
            return
        video_cam_cfg = self.cfg["env"].get("videoCam")
        if (
            video_cam_cfg is not None
            and self.cfg["env"].get("rgbVideo", {}).get("cameraType") == "video"
        ):
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

    # =====================================================================
    # Eval-metrics persistence
    # =====================================================================

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

    # =====================================================================
    # Language embedding bank
    # =====================================================================

    def _load_eval_language_embedding_bank(self, diffusion):
        if not getattr(diffusion.args, "use_language_conditioning", False):
            return None

        language_path = getattr(diffusion.args, "language_embedding_dict_path", None)
        if language_path is None:
            ckpt_dir = Path(diffusion.args.ckpt_path).resolve().parent
            language_path = ckpt_dir / "language_embedding_dict.json"
        else:
            language_path = Path(language_path)

        if not language_path.exists():
            raise FileNotFoundError(f"language_embedding_dict.json not found: {language_path}")

        with language_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        bank = np.asarray(payload["expanded_minimal_chains"], dtype=np.float32)
        if bank.ndim != 2:
            raise ValueError(f"expanded_minimal_chains must be 2D, got shape {bank.shape}")

        expected_dim = int(getattr(diffusion.args, "language_input_dim", bank.shape[-1]))
        if bank.shape[-1] != expected_dim:
            raise ValueError(
                f"language embedding dim mismatch: expected {expected_dim}, got {bank.shape[-1]}"
            )

        print(f"loaded language embedding bank: {language_path}, size={bank.shape[0]}")
        return torch.from_numpy(bank).to(diffusion.device)

    def _sample_episode_language_embedding(self, embedding_bank, batch_size):
        if embedding_bank is None:
            return None, None
        bank_size = embedding_bank.shape[0]
        sampled_idx = int(np.random.randint(0, bank_size))
        sampled = embedding_bank[sampled_idx].unsqueeze(0).repeat(batch_size, 1)
        return sampled, sampled_idx

    def _select_language_embedding_per_env(self, embedding_bank, chain_ids):
        if embedding_bank is None:
            return None
        index = torch.as_tensor(list(chain_ids), dtype=torch.long, device=embedding_bank.device)
        return embedding_bank.index_select(0, index)

    # =====================================================================
    # Asker correctness scoring
    # =====================================================================

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

    # =====================================================================
    # Adaptive video / dump infrastructure
    # =====================================================================

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

    def _adaptive_init_inference_dump(self, save_dir, task_spec, expanded_minimal_chains):
        os.makedirs(save_dir, exist_ok=True)
        language_expanded = {
            "schema_version": "v1",
            "generated_from": "manipulation/base_manipulation.py::diffusion_evaluate "
            "(task.save_inference_data)",
            "task": self.language_template_task_name(),
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
            "records": [],
            "action_arrays": [],
        }

    def _adaptive_record_inference_episode(
        self,
        dump_state,
        eps_idx,
        env_id,
        episode_state,
        chain_id_used,
        done_flag,
        adaptive_state,
        action_array,
        adaptive_asker_record,
        expanded_minimal_chains,
    ):
        """Append one (eps, env) record to the dump.

        ``episode_state`` is the per-env dict from
        ``capture_per_env_episode_state``; the canonical chain is derived via
        ``canonical_minimal_chain_for_state``. Records carry the raw dict in
        an ``episode_state`` field for offline tasks that need it.
        """
        if dump_state is None or action_array is None or len(action_array) == 0:
            return
        canonical_minimal_chain = self.canonical_minimal_chain_for_state(episode_state)
        if canonical_minimal_chain is not None:
            try:
                canonical_minimal_chain_id = expanded_minimal_chains.index(canonical_minimal_chain)
            except ValueError:
                canonical_minimal_chain_id = None
        else:
            canonical_minimal_chain_id = None
        chain_used_steps = (
            list(expanded_minimal_chains[chain_id_used])
            if chain_id_used is not None
            and 0 <= chain_id_used < len(expanded_minimal_chains)
            else []
        )
        fallback_chain = canonical_minimal_chain or chain_used_steps or []

        prev_total = sum(arr.shape[0] for arr in dump_state["action_arrays"])
        dump_state["action_arrays"].append(np.asarray(action_array, dtype=np.float32))
        new_total = prev_total + dump_state["action_arrays"][-1].shape[0]

        record = {
            "minimal_chain_id": int(canonical_minimal_chain_id) if canonical_minimal_chain_id is not None else None,
            "minimal_chain": canonical_minimal_chain or fallback_chain,
            "attempt_chain": chain_used_steps if chain_used_steps else fallback_chain,
            "stage_status": [True] * max(len(chain_used_steps or fallback_chain), 1),
            "command_chains": [chain_used_steps] if chain_used_steps else [fallback_chain],
            "command_chain_ids": [int(chain_id_used)] if chain_id_used is not None else [],
            "success": bool(done_flag),
            "episode_id": int(len(dump_state["records"])),
            "round_idx": int(eps_idx),
            "env_id": int(env_id),
            "frame_range": [int(prev_total), int(new_total)],
            "episode_state": dict(episode_state) if episode_state else {},
            "language_chain_id_used": int(chain_id_used) if chain_id_used is not None else None,
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
        if action_arrays:
            all_actions = np.concatenate(action_arrays, axis=0).astype(np.float32)
        else:
            all_actions = np.zeros((0, 10), dtype=np.float32)
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
                chunks=(min(1024, max(1, all_actions.shape[0])), all_actions.shape[1])
                if all_actions.size
                else (1, 10),
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

    # =====================================================================
    # Generic data-collection helper used by subclasses' collect_manip_data
    # =====================================================================

    def process_data(self, goal_pos):
        obs = self.env.collect_diff_data()
        pc, env_state = obs_wrapper(obs)
        goal_pos = self.action_process(goal_pos)
        if self.env.gripper[0, 0].cpu().item() == 1:
            temp = torch.ones((self.env.num_envs, 1), device=self.env.device)
        else:
            temp = torch.zeros((self.env.num_envs, 1), device=self.env.device)
        action_with_gripper = torch.cat([goal_pos, temp], dim=-1)
        self.env.actions = action_with_gripper
        for env_id in range(self.env.num_envs):
            self.eps_buffer[env_id].add(pc[env_id], env_state[env_id], action_with_gripper[env_id])
            self.append_frame_label(env_id)
        self._record_video_frame()

    @staticmethod
    def pc_normalize(pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    # =====================================================================
    # diffusion_evaluate — task-agnostic adaptive eval loop
    # =====================================================================

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
            eval_metrics, 0.0, status="running", num_envs=self.env.num_envs,
        )
        self._write_eval_metrics(eval_save_dir, eval_metrics)
        self._init_video_recorder(eval_save_dir)

        if adaptive_enable and language_embedding_bank is None:
            raise RuntimeError(
                "task.adaptive_language.enable is True but the policy does not provide a language embedding bank "
                "(set model.use_language_conditioning=True and supply language_embedding_dict.json)."
            )
        adaptive_states = None
        adaptive_frozen_states = None
        adaptive_asker = None
        adaptive_camera_id = 0
        adaptive_rng = None
        adaptive_expanded_chains = None
        adaptive_chain_priority_ids = None
        adaptive_dump_state = None
        adaptive_save_inference_data = bool(
            self.cfg.get("task", {}).get("save_inference_data", False)
        )
        max_retry_rounds = int(adaptive_cfg.get("max_retry_rounds", 3))
        task_spec = None
        expanded_minimal_chains = None
        if adaptive_enable or adaptive_save_inference_data:
            template_path, task_spec = self.load_task_language_template(
                self.language_template_task_name()
            )
            expanded_minimal_chains = self.build_expanded_minimal_chains(
                task_spec["minimal_chains"]
            )
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
                asker_cfg, task_spec, expanded_minimal_chains,
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
            adaptive_frozen_states = [None] * self.env.num_envs
            adaptive_camera_id = self._adaptive_resolve_camera_id(asker_cfg)
            seed = int(self.cfg.get("seed", 0)) if self.cfg.get("seed", None) is not None else 0
            adaptive_rng = random.Random(seed)
            print(
                f"[adaptive] enabled: platform={asker_cfg.platform}, model={asker_cfg.model}, "
                f"num_chains={int(language_embedding_bank.shape[0])}, "
                f"num_envs={self.env.num_envs}, camera_id={adaptive_camera_id}"
            )
            print(
                "[adaptive] chain inference priority: " + ", ".join(priority_parts)
            )
        if adaptive_save_inference_data:
            if not self.cfg.get("env", {}).get("collectRGBVideo", False):
                print(
                    "[dump] save_inference_data=true requires env.collectRGBVideo=true; "
                    "videos will be missing from the dump."
                )
            adaptive_dump_state = self._adaptive_init_inference_dump(
                eval_save_dir, task_spec, expanded_minimal_chains,
            )
            print(
                f"[dump] save_inference_data: writing eval_video2prompt-compatible artifacts under {eval_save_dir} "
                f"(adaptive_language={'on' if adaptive_enable else 'off'})"
            )

        for eps in range(eps_num):
            print("eps_{}".format(eps + 1))
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
                            adaptive_rng, priority_ids=adaptive_chain_priority_ids,
                        )
                        s.current_chain_id = cid
                        s.tried_chain_ids.add(cid)
                        chain_ids.append(cid)
                episode_language_embedding = self._select_language_embedding_per_env(
                    language_embedding_bank, chain_ids,
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
                    language_embedding_bank, self.env.num_envs,
                )
                if sampled_language_idx is not None:
                    print(f"episode {eps + 1} language embedding id: {sampled_language_idx}")

            # --- Reset env, optionally re-injecting frozen state from episode 1 ---
            reset_kwargs = self.reset_kwargs_initial()
            if (
                adaptive_enable
                and eps > 0
                and adaptive_frozen_states is not None
                and any(state is not None for state in adaptive_frozen_states)
            ):
                frozen_kwargs = self.apply_frozen_states_to_reset_kwargs(
                    adaptive_frozen_states
                )
                if frozen_kwargs is not None:
                    reset_kwargs = frozen_kwargs
            self.env.reset(**reset_kwargs)
            episode_states = self.capture_per_env_episode_state()
            if adaptive_enable and eps == 0:
                for env_id in range(self.env.num_envs):
                    adaptive_frozen_states[env_id] = dict(episode_states[env_id])
                print(
                    f"[adaptive] eps 1 frozen per-env state: {adaptive_frozen_states}"
                )

            self.env.gripper = torch.zeros((self.env.num_envs, 1), device=self.env.device)
            if self.video_recorder is not None:
                self.video_recorder.start_episode(eps)
                self._record_video_frame()

            obs = self.env.collect_diff_data()
            pcs, env_state = obs_wrapper(obs)
            pcs_deque = collections.deque(
                [pcs] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon,
            )
            env_state_deque = collections.deque(
                [env_state] * diffusion.args.obs_horizon, maxlen=diffusion.args.obs_horizon,
            )

            episode_action_logs = (
                [[] for _ in range(self.env.num_envs)]
                if (adaptive_enable or adaptive_save_inference_data)
                else None
            )

            try:
                while step <= 32:
                    action = diffusion.infer_action_with_seg(
                        pcs_deque, env_state_deque,
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
                            action_step_arr = action[:, act, :].detach().cpu().numpy()
                            for env_id in range(self.env.num_envs):
                                episode_action_logs[env_id].append(action_step_arr[env_id])
                        obs = self.env.collect_diff_data()
                        pcs, env_state = obs_wrapper(obs)
                        pcs_deque.append(pcs)
                        env_state_deque.append(env_state)

                    for env_id in range(self.env.num_envs):
                        if self.task_success_for_env(env_id) and not done_flag[env_id]:
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
                        # Flatten to rgb_videos/episode_<eps>/env_<id>_cam_<cam>.mp4. Online
                        # asker (adaptive_enable) reads by env_id; offline replay
                        # (save_inference_data) expects this layout in locate_video().
                        self.video_recorder.finish_episode(env_results=None)
                    else:
                        self.video_recorder.finish_episode(episode_results)

            rollout_elapsed_sec = float(time.perf_counter() - episode_timer_start)
            rollout_step_count = int(step)

            ground_truth_chain_ids: List[Optional[int]] = [None] * self.env.num_envs
            if adaptive_expanded_chains is not None:
                for env_id in range(self.env.num_envs):
                    canonical = self.canonical_minimal_chain_for_state(episode_states[env_id])
                    if canonical is not None:
                        try:
                            ground_truth_chain_ids[env_id] = adaptive_expanded_chains.index(canonical)
                        except ValueError:
                            ground_truth_chain_ids[env_id] = None

            if adaptive_enable and episode_results is not None:
                recategorize = []
                lock_on_env_success = bool(
                    adaptive_cfg.get("asker", {}).get("lock_on_env_success", False)
                )
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
                    canonical_chain = self.canonical_minimal_chain_for_state(
                        episode_states[env_id]
                    )
                    asker_success, asker_chain_id = adaptive_asker.ask(
                        video_path=video_path if os.path.exists(video_path) else None,
                        action_array=actions_arr,
                        env_id=env_id,
                        done_flag=bool(done_flag[env_id]),
                        ground_truth_chain=canonical_chain,
                    )
                    ground_truth_chain_id = ground_truth_chain_ids[env_id]
                    reasonable_prediction_chain_ids = self._reasonable_prediction_chain_ids(
                        s.current_chain_id, ground_truth_chain_id, adaptive_expanded_chains,
                    )
                    asker_prompt_correct = self._asker_prompt_correctness(
                        asker_chain_id, ground_truth_chain_id, reasonable_prediction_chain_ids,
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
                        "ground_truth_chain_id": int(ground_truth_chain_id) if ground_truth_chain_id is not None else None,
                        "reasonable_prediction_chain_ids": reasonable_prediction_chain_ids,
                        "asker_prompt_correct": asker_prompt_correct,
                        "asker_reasonable_prediction_correct": (
                            asker_prompt_correct is True
                            if asker_chain_id is not None
                            else None
                        ),
                        "asker_strict_ground_truth_correct": (
                            int(asker_chain_id) == int(ground_truth_chain_id)
                            if asker_chain_id is not None and ground_truth_chain_id is not None
                            else None
                        ),
                        "current_chain_id": int(s.current_chain_id) if s.current_chain_id is not None else None,
                        "locked_chain_id": int(s.locked_chain_id) if s.locked_chain_id is not None else None,
                        "tried_chain_ids": sorted(int(item) for item in s.tried_chain_ids),
                        "sweep_count": int(s.sweep_count),
                        "video_path": video_path if os.path.exists(video_path) else None,
                    }
                if (
                    not adaptive_save_inference_data
                    and bool(adaptive_cfg.get("asker", {}).get("recategorize_videos", True))
                ):
                    self._adaptive_recategorize_videos(eval_save_dir, eps, recategorize)

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
                        episode_state=episode_states[env_id],
                        chain_id_used=chain_id_used,
                        done_flag=bool(done_flag[env_id]),
                        adaptive_state=adaptive_state,
                        action_array=actions_arr,
                        adaptive_asker_record=asker_record,
                        expanded_minimal_chains=adaptive_expanded_chains,
                    )

            episode_elapsed_sec = float(time.perf_counter() - episode_timer_start)
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
                    "ground_truth_chain_id": ground_truth_chain_ids[env_id],
                }
                env_record.update(self.per_env_extra_log_fields(env_id, episode_states[env_id]))
                if adaptive_enable:
                    env_record["language_chain_id"] = (
                        int(chain_ids[env_id]) if chain_ids is not None else None
                    )
                    env_record["adaptive"] = adaptive_asker_records[env_id]
                else:
                    env_record["language_chain_id"] = (
                        int(sampled_language_idx) if sampled_language_idx is not None else None
                    )
                env_metrics.append(env_record)

            episode_success_count = int(sum(1 for item in done_flag if item))
            cur_rate = episode_success_count / (self.env.num_envs)
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

            print(f"Eps {eps + 1}, current succ rate {cur_rate}")
            succ_rate.append(cur_rate)
            succ_cnt = 0
        print(f"Average Success rate: {np.mean(succ_rate)}")
        print(f"Success rate std: {np.std(succ_rate)}")
        if adaptive_enable:
            print(
                "[adaptive] final per-env locked_chain_id: " +
                ", ".join(
                    f"env{env_id}=locked={s.locked_chain_id}|tried={sorted(s.tried_chain_ids)}|sweep={s.sweep_count}"
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
