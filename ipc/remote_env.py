from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import rpyc
import torch

from ipc.service import build_server_config


def _to_wire(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {key: _to_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_wire(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_wire(item) for item in value)
    return value


def _from_wire(value: Any, device: str) -> Any:
    if isinstance(value, np.ndarray):
        local_value = np.asarray(value)
        if local_value.dtype.kind in {"U", "S", "O"}:
            return local_value
        return torch.from_numpy(local_value).to(device=device)
    if isinstance(value, dict):
        return {key: _from_wire(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_wire(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_from_wire(item, device) for item in value)
    return value


class RemoteEnv:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 18861,
        connect_timeout_sec: float = 30.0,
        retry_interval_sec: float = 0.5,
    ):
        self._conn = self._connect_with_retry(
            host=host,
            port=port,
            connect_timeout_sec=connect_timeout_sec,
            retry_interval_sec=retry_interval_sec,
        )
        self._state: Dict[str, Any] = {}
        self.device = "cpu"
        self.refresh()

    def _connect_with_retry(
        self,
        host: str,
        port: int,
        connect_timeout_sec: float,
        retry_interval_sec: float,
    ):
        deadline = time.monotonic() + connect_timeout_sec
        last_error = None
        while time.monotonic() < deadline:
            try:
                return rpyc.connect(host, port, config=build_server_config())
            except Exception as error:
                last_error = error
                time.sleep(retry_interval_sec)
        raise RuntimeError(
            f"Failed to connect to AdaManip rpyc server at {host}:{port} within {connect_timeout_sec} seconds"
        ) from last_error

    def _decode_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        device = state.get("device", self.device)
        if isinstance(device, bytes):
            device = device.decode()
        self.device = device or self.device
        return {key: _from_wire(value, self.device) for key, value in state.items()}

    def _update_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        decoded = self._decode_state(state)
        self._state.update(decoded)
        return decoded

    def refresh(self) -> Dict[str, Any]:
        return self._update_state(dict(self._conn.root.get_state_snapshot()))

    def ping(self) -> Dict[str, Any]:
        return dict(self._conn.root.ping())

    def reset(self, to_reset: Any = "all", **kwargs: Any) -> Dict[str, Any]:
        if kwargs:
            return self._update_state(
                dict(self._conn.root.reset_with_kwargs(to_reset, kwargs))
            )
        return self._update_state(dict(self._conn.root.reset(to_reset)))

    def step(self, actions: Any) -> Dict[str, Any]:
        return self._update_state(dict(self._conn.root.step(_to_wire(actions))))

    def collect_diff_data(self, flag: Optional[bool] = None) -> Any:
        if flag is None:
            result = self._conn.root.collect_diff_data()
        else:
            result = self._conn.root.collect_diff_data(flag)
        return _from_wire(result, self.device)

    def collect_single_diff_data(self, env_id: int) -> Any:
        return _from_wire(self._conn.root.collect_single_diff_data(env_id), self.device)

    def collect_rgb_frames(
        self,
        camera_type: Optional[str] = None,
        camera_ids: Optional[Iterable[int]] = None,
    ) -> Any:
        if camera_type is None and camera_ids is None:
            result = self._conn.root.collect_rgb_frames()
        else:
            result = self._conn.root.collect_rgb_frames(camera_type, camera_ids)
        return result

    def get_adjust_hand_pose(self) -> Any:
        result = self._conn.root.get_adjust_hand_pose()
        pose = _from_wire(result, self.device)
        self._state["adjust_hand_pose"] = pose
        return pose

    def get_obj_dof_property_tensor(self) -> Dict[str, Any]:
        return self._update_state(dict(self._conn.root.get_obj_dof_property_tensor()))

    def close(self) -> None:
        self._conn.close()

    @property
    def gripper(self) -> Any:
        return self._state.get("gripper")

    @gripper.setter
    def gripper(self, value: Any) -> None:
        result = dict(self._conn.root.set_gripper(_to_wire(value)))
        self._state["gripper"] = _from_wire(result["gripper"], self.device)

    @property
    def actions(self) -> Any:
        return self._state.get("actions")

    @actions.setter
    def actions(self, value: Any) -> None:
        result = dict(self._conn.root.set_actions(_to_wire(value)))
        self._state["actions"] = _from_wire(result.get("actions"), self.device)

    def __getattr__(self, name: str) -> Any:
        if name in self._state:
            return self._state[name]
        raise AttributeError(f"RemoteEnv has no attribute '{name}'")

    def __del__(self) -> None:
        conn = self.__dict__.get("_conn")
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass