from abc import abstractclassmethod
from envs.base_env import BaseEnv
from logging import Logger
import pytorch3d.transforms as tf
import torch


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
