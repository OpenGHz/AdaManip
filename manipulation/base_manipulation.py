from abc import abstractclassmethod
from envs.base_env import BaseEnv
from logging import Logger
import pytorch3d.transforms as tf
import torch
import os
import json


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    positive_mask = x > 0
    safe_x = torch.where(positive_mask, x, torch.ones_like(x))
    return torch.where(positive_mask, torch.sqrt(safe_x), torch.zeros_like(x))

class BaseManipulation :

    def __init__(self, env : BaseEnv, cfg : dict, logger : Logger) :

        self.env = env
        self.cfg = cfg
        self.logger = logger
        self._current_step_index = 0
        self._current_step_operation = ""
        self._episode_frame_records = None

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
        if "command" not in task_spec or "operation_set" not in task_spec or "minimal_chains" not in task_spec:
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
            raise RuntimeError(f"No command_chains matched prefix={prefix}; task logic or labels are inconsistent")
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

        with open(os.path.join(save_dir, "language_expanded.json"), "w", encoding="utf-8") as f:
            json.dump(expanded_payload, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_dir, "trajectory_language.jsonl"), "w", encoding="utf-8") as f:
            json.dump(trajectory_records, f, ensure_ascii=False, indent=2)

        with open(os.path.join(save_dir, "frame_language.jsonl"), "w", encoding="utf-8") as f:
            json.dump(frame_records, f, ensure_ascii=False, indent=2)

    @abstractclassmethod
    def collect_data(self, obs, eval=False) :

        pass

    def action_process(self, pose):
        quat_isaac = pose[:,3:7].float()
        quat_p3d = torch.cat([quat_isaac[:,3:], quat_isaac[:,:3]], dim=-1)
        rotate_matix = tf.quaternion_to_matrix(quat_p3d)
        rotate_6d = tf.matrix_to_rotation_6d(rotate_matix)
        return torch.cat([pose[:,:3], rotate_6d], dim=-1)

    def rotate_6d_to_quat(self, rotate_6d):
        rotate_matix = tf.rotation_6d_to_matrix(rotate_6d)
        quat_p3d = self.matrix_to_quaternion(rotate_matix)
        quat_isaac = torch.cat([quat_p3d[:,1:], quat_p3d[:,:1]], dim=-1)
        return quat_isaac

    @staticmethod
    def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as rotation matrices to quaternions.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).

        Returns:
            quaternions with real part first, as tensor of shape (..., 4).
        """
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

        # we produce the desired quaternion multiplied by each of r, i, j, k
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

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)
        indices = q_abs.argmax(dim=-1, keepdim=True)
        expand_dims = list(batch_dim) + [1, 4]
        gather_indices = indices.unsqueeze(-1).expand(expand_dims)
        out = torch.gather(quat_candidates, -2, gather_indices).squeeze(-2)
        return tf.standardize_quaternion(out)
