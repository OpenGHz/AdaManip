"""Adaptive language-conditioning state machine and asker wrapper.

Used by ``BaseManipulation.diffusion_evaluate`` when
``cfg.task.adaptive_language.enable`` is true. Each env in the eval batch holds
its own ``AdaptiveLanguageState``: each unlocked episode picks a chain id from
the still-untried pool, optionally following a diagnostic priority order, and
calls the asker; subsequent episodes reuse the locked chain id (if the asker
confirmed success) or try the next chain. Task-specific environment state that
needs to be frozen across episodes (e.g. ``clock_wise`` for the microwave task)
is owned by the manipulation subclass via the
``capture_per_env_episode_state`` / ``apply_frozen_states_to_reset_kwargs``
hooks; this module is fully task-agnostic.
``AdaptiveLanguageAsker`` wraps the
``Video2Prompt`` / ``Video2PromptGroundTruth`` askers from ``try_to_remember``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class AdaptiveLanguageState:
    """Per-env state for the adaptive language-conditioning loop.

    Task-specific frozen environment state (e.g. ``clock_wise`` for the
    microwave task) is owned by the manipulation class via its
    ``capture_per_env_episode_state`` /
    ``apply_frozen_states_to_reset_kwargs`` hooks; this struct only tracks
    chain-id bookkeeping that's shared across tasks.
    """

    num_chains: int
    locked_chain_id: Optional[int] = None
    current_chain_id: Optional[int] = None
    tried_chain_ids: set = field(default_factory=set)
    sweep_count: int = 0

    def pick_next(self, rng: random.Random, priority_ids: Optional[List[int]] = None) -> int:
        """Pick a chain id excluding the already-tried ones.

        When ``priority_ids`` is provided, the first still-untried id in that
        order is returned. Resets ``tried_chain_ids`` and starts a new sweep
        when all ids have already been tried.
        """
        if self.num_chains <= 0:
            raise ValueError("num_chains must be positive")
        if priority_ids:
            ordered_ids = []
            seen = set()
            for item in priority_ids:
                chain_id = int(item)
                if 0 <= chain_id < self.num_chains and chain_id not in seen:
                    ordered_ids.append(chain_id)
                    seen.add(chain_id)
            if len(ordered_ids) != self.num_chains:
                ordered_ids.extend(i for i in range(self.num_chains) if i not in seen)
        else:
            ordered_ids = list(range(self.num_chains))

        candidates = [i for i in ordered_ids if i not in self.tried_chain_ids]
        if not candidates:
            self.tried_chain_ids = set()
            self.sweep_count += 1
            candidates = ordered_ids
        if priority_ids:
            return candidates[0]
        return rng.choice(candidates)


def _select_frames(frames: List[np.ndarray], max_frames: int, stride: int) -> List[np.ndarray]:
    """Mirror ``scripts/eval_video2prompt.py::select_frames`` (frames only)."""
    if stride < 1:
        raise ValueError("frame_stride must be >= 1")
    indexed = [frame for index, frame in enumerate(frames) if index % stride == 0]
    if max_frames > 0 and len(indexed) > max_frames:
        positions = sorted(
            set(int(round(pos)) for pos in np.linspace(0, len(indexed) - 1, max_frames))
        )
        indexed = [indexed[pos] for pos in positions]
    return indexed


def _coerce_str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


class AdaptiveLanguageAskerConfig:
    """Cfg holder mirroring ``ExperienceAbstractionConfig`` + frame/traj options.

    Constructed from the ``task.adaptive_language.asker`` cfg subtree (a plain
    dict). All fields have sensible defaults so the cfg only needs to specify
    overrides.
    """

    def __init__(self, raw: Optional[Dict[str, Any]] = None):
        raw = dict(raw or {})
        self.platform: str = _coerce_str(raw.get("platform", "ground-truth")).lower()
        self.model: str = _coerce_str(raw.get("model", "gpt-5.5"))
        self.mock: bool = bool(raw.get("mock", False))
        self.api_key: Optional[str] = raw.get("api_key")
        self.api_key_env: str = _coerce_str(raw.get("api_key_env", ""))
        self.base_url: str = _coerce_str(raw.get("base_url", ""))
        self.prompt_style: str = _coerce_str(raw.get("prompt_style", "structured"))
        self.check_success: str = _coerce_str(raw.get("check_success", "together")).lower()

        self.frame_max_count: int = int(raw.get("frame_max_count", 12))
        self.frame_stride: int = int(raw.get("frame_stride", 1))
        self.camera_id: int = int(raw.get("camera_id", 1))

        self.trajectory_context: bool = bool(raw.get("trajectory_context", True))
        self.trajectory_representation: str = _coerce_str(
            raw.get("trajectory_representation", "delta")
        )
        self.trajectory_sample_points: int = int(raw.get("trajectory_sample_points", 0))

        self.recategorize_videos: bool = bool(raw.get("recategorize_videos", True))
        self.lock_on_env_success: bool = bool(raw.get("lock_on_env_success", False))

        self.claude_cli_command: str = _coerce_str(raw.get("claude_cli_command", "claude"))
        self.claude_cli_skip_permissions: bool = bool(
            raw.get("claude_cli_skip_permissions", True)
        )
        self.claude_cli_timeout: int = int(raw.get("claude_cli_timeout", 900))

        self.codex_cli_command: str = _coerce_str(raw.get("codex_cli_command", "codex"))
        self.codex_cli_timeout: int = int(raw.get("codex_cli_timeout", 900))
        self.codex_cli_sandbox: str = _coerce_str(raw.get("codex_cli_sandbox", "read-only"))
        self.codex_cli_approval_policy: str = _coerce_str(
            raw.get("codex_cli_approval_policy", "never")
        )
        self.codex_cli_cd: str = _coerce_str(raw.get("codex_cli_cd", ""))
        self.codex_cli_ephemeral: bool = bool(raw.get("codex_cli_ephemeral", True))
        self.codex_cli_effort: str = _coerce_str(raw.get("codex_cli_effort", "xhigh"))

        self.gemini_temperature: float = float(raw.get("gemini_temperature", 1.0))
        self.gemini_thinking_budget: int = int(raw.get("gemini_thinking_budget", -1))
        self.gemini_image_thinking_budget: int = int(
            raw.get("gemini_image_thinking_budget", 0)
        )
        self.gemini_upload_poll_interval: float = float(
            raw.get("gemini_upload_poll_interval", 1.0)
        )
        self.gemini_upload_timeout: int = int(raw.get("gemini_upload_timeout", 600))

    @property
    def is_ground_truth(self) -> bool:
        return self.platform in {"ground-truth", "ground_truth", "gt"}


class AdaptiveLanguageAsker:
    """Wrap ``Video2Prompt`` / ``Video2PromptGroundTruth`` for per-env queries."""

    def __init__(
        self,
        asker_cfg: AdaptiveLanguageAskerConfig,
        task_spec: Dict[str, Any],
        expanded_minimal_chains: List[List[str]],
    ):
        # Lazy imports keep the ada-manip path importable even when
        # try_to_remember is missing in unrelated environments.
        from try_to_remember.experience_abstraction.video2prompt import (
            CheckSuccessMode,
            ExperienceAbstractionConfig,
            Video2Prompt,
            Video2PromptGroundTruth,
        )
        from try_to_remember.experience_abstraction.trajectory_context import (
            ActionTrajectoryContextBuilder,
        )
        from try_to_remember.chat import video_to_frames

        self.cfg = asker_cfg
        self.task_spec = task_spec
        self.expanded_minimal_chains = expanded_minimal_chains
        self._chain_to_id: Dict[Tuple[str, ...], int] = {
            tuple(chain): idx for idx, chain in enumerate(expanded_minimal_chains)
        }
        self._command = task_spec["command"]
        self._video_to_frames = video_to_frames
        self._CheckSuccessMode = CheckSuccessMode
        check_success_map = {
            "together": CheckSuccessMode.TOGETHER,
            "last": CheckSuccessMode.LAST,
            "all": CheckSuccessMode.ALL,
        }
        self._check_success = check_success_map.get(
            asker_cfg.check_success, CheckSuccessMode.TOGETHER
        )

        config = ExperienceAbstractionConfig(
            api_key=asker_cfg.api_key,
            api_key_env=asker_cfg.api_key_env,
            base_url=asker_cfg.base_url,
            model=asker_cfg.model,
            mock=asker_cfg.mock,
            platform=asker_cfg.platform,
            claude_cli_command=asker_cfg.claude_cli_command,
            claude_cli_skip_permissions=asker_cfg.claude_cli_skip_permissions,
            claude_cli_timeout=asker_cfg.claude_cli_timeout,
            codex_cli_command=asker_cfg.codex_cli_command,
            codex_cli_timeout=asker_cfg.codex_cli_timeout,
            codex_cli_sandbox=asker_cfg.codex_cli_sandbox,
            codex_cli_approval_policy=asker_cfg.codex_cli_approval_policy,
            codex_cli_cd=asker_cfg.codex_cli_cd,
            codex_cli_ephemeral=asker_cfg.codex_cli_ephemeral,
            codex_cli_effort=asker_cfg.codex_cli_effort,
            prompt_style=asker_cfg.prompt_style,
            gemini_temperature=asker_cfg.gemini_temperature,
            gemini_thinking_budget=asker_cfg.gemini_thinking_budget,
            gemini_image_thinking_budget=asker_cfg.gemini_image_thinking_budget,
            gemini_upload_poll_interval=asker_cfg.gemini_upload_poll_interval,
            gemini_upload_timeout=asker_cfg.gemini_upload_timeout,
        )
        if asker_cfg.is_ground_truth:
            self.video2prompt = Video2PromptGroundTruth(config)
        else:
            self.video2prompt = Video2Prompt(config)

        prior = dict(task_spec)
        prior["expanded_minimal_chains"] = expanded_minimal_chains
        self.video2prompt.set_prior(prior)

        if asker_cfg.trajectory_context:
            self.trajectory_builder = ActionTrajectoryContextBuilder(
                data_dir=None,
                max_rows=asker_cfg.trajectory_sample_points,
                representation=asker_cfg.trajectory_representation,
            )
        else:
            self.trajectory_builder = None

    @property
    def is_ground_truth(self) -> bool:
        return self.cfg.is_ground_truth

    def ask(
        self,
        video_path: Optional[Union[str, Path]],
        action_array: Optional[np.ndarray],
        env_id: int,
        done_flag: bool,
        ground_truth_chain: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[int]]:
        """Run the asker for one env. Returns (success, chain_id-or-None).

        ``ground_truth_chain`` is the canonical minimal chain (list[str]) for this
        env's current episode, computed by the caller from task-specific state.
        Required for the ground-truth platform; ignored by LLM platforms.
        """

        if self.is_ground_truth:
            if ground_truth_chain is None:
                logger.warning(
                    "AdaptiveLanguageAsker[gt]: env=%d missing ground_truth_chain; treating as failure",
                    env_id,
                )
                return False, None
            traj = {"success": bool(done_flag), "minimal_chain": list(ground_truth_chain)}
            self.video2prompt.set_ground_truth(traj)
            success = bool(self.video2prompt.is_success())
            if not success:
                return False, None
            pred_chain = list(self.video2prompt.to_prompt())
            chain_id = self._chain_to_id.get(tuple(pred_chain))
            return chain_id is not None, chain_id

        # LLM path
        if video_path is None:
            logger.warning("AdaptiveLanguageAsker[llm]: env=%d missing video_path", env_id)
            return False, None

        try:
            video_input: Union[str, List[np.ndarray]]
            if getattr(self.video2prompt.asker, "prefers_video_file", False):
                video_input = str(video_path)
            else:
                frames = self._video_to_frames(str(video_path))
                video_input = _select_frames(
                    frames, self.cfg.frame_max_count, self.cfg.frame_stride
                )

            traj_text: Optional[str] = None
            if self.trajectory_builder is not None and action_array is not None and len(action_array) > 0:
                payload = self.trajectory_builder.build_from_actions(
                    np.asarray(action_array), episode_id=env_id
                )
                traj_text = payload.get("text")

            pred_chain = self.video2prompt.to_prompt(
                video_input,
                check_success=self._check_success,
                trajectory_context=traj_text,
            )
        except Exception as exc:
            logger.exception(
                "AdaptiveLanguageAsker[llm]: env=%d ask failed: %s", env_id, exc
            )
            return False, None

        if not isinstance(pred_chain, list):
            return False, None
        if pred_chain == [self._command]:
            # Asker indicated failure (returned base command, not a chain).
            return False, None
        chain_id = self._chain_to_id.get(tuple(pred_chain))
        if chain_id is None:
            logger.warning(
                "AdaptiveLanguageAsker[llm]: env=%d unrecognized chain %r", env_id, pred_chain
            )
            return False, None
        return True, chain_id
