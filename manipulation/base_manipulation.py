from abc import abstractclassmethod
from envs.base_env import BaseEnv
from logging import Logger
import pytorch3d.transforms as tf
import torch

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
        rotate_matix = tf.rotation_6d_to_matrix(rotate_6d.float()).float()
        quat_p3d = self._matrix_to_quaternion_wxyz(rotate_matix)
        quat_isaac = torch.cat([quat_p3d[:,1:], quat_p3d[:,:1]], dim=-1)
        return quat_isaac

    def _matrix_to_quaternion_wxyz(self, matrix):
        m00 = matrix[:, 0, 0]
        m01 = matrix[:, 0, 1]
        m02 = matrix[:, 0, 2]
        m10 = matrix[:, 1, 0]
        m11 = matrix[:, 1, 1]
        m12 = matrix[:, 1, 2]
        m20 = matrix[:, 2, 0]
        m21 = matrix[:, 2, 1]
        m22 = matrix[:, 2, 2]

        one = torch.ones_like(m00)
        qw = 0.5 * torch.sqrt(torch.clamp(one + m00 + m11 + m22, min=0.0))
        qx = 0.5 * torch.sign(m21 - m12) * torch.sqrt(torch.clamp(one + m00 - m11 - m22, min=0.0))
        qy = 0.5 * torch.sign(m02 - m20) * torch.sqrt(torch.clamp(one - m00 + m11 - m22, min=0.0))
        qz = 0.5 * torch.sign(m10 - m01) * torch.sqrt(torch.clamp(one - m00 - m11 + m22, min=0.0))
        return torch.stack([qw, qx, qy, qz], dim=-1)