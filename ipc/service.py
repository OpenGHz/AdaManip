from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import rpyc
import torch
from rpyc.utils.server import ThreadedServer


DEFAULT_SNAPSHOT_FIELDS = (
    "num_envs",
    "num_actions",
    "device",
    "gripper",
    "actions",
    "gripper_length",
    "adjust_hand_pose",
    "hand_rigid_body_tensor",
    "part_rigid_body_tensor",
    "rigid_body_tensor",
    "one_dof_tensor",
    "two_dof_tensor",
    "open_bottle_stage",
    "open_door_stage",
    "clock_wise",
    "try_range",
    "action_chosen",
    "num_cam",
)


def build_server_config(allow_public_attrs: bool = True) -> Dict[str, Any]:
    return {
        "allow_all_attrs": True,
        "allow_public_attrs": allow_public_attrs,
        "allow_pickle": True,
        "sync_request_timeout": None,
    }


def _serialize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        serialized = [_serialize_value(item) for item in value]
        return serialized if isinstance(value, list) else tuple(serialized)
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _to_torch(value: Any, device: str) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.asarray(value)).to(device=device)
    if isinstance(value, dict):
        return {key: _to_torch(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_torch(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_torch(item, device) for item in value)
    return value


def _collect_snapshot(env: Any, snapshot_fields: Iterable[str]) -> Dict[str, Any]:
    snapshot = {}
    for field in snapshot_fields:
        if hasattr(env, field):
            snapshot[field] = _serialize_value(getattr(env, field))
    return snapshot


def _convert_actions(env: Any, actions: Any) -> Any:
    if isinstance(actions, torch.Tensor):
        return actions.to(device=env.device)
    return _to_torch(actions, env.device)


def _set_adjust_hand_pose_if_available(env: Any) -> None:
    if hasattr(env, "get_adjust_hand_pose"):
        try:
            env.adjust_hand_pose = env.get_adjust_hand_pose()
        except TypeError:
            pass


def create_service(env: Any, snapshot_fields: Optional[Iterable[str]] = None):
    fields = tuple(snapshot_fields or DEFAULT_SNAPSHOT_FIELDS)

    class AdaManipEnvService(rpyc.Service):
        def exposed_ping(self) -> Dict[str, Any]:
            return {
                "status": "ok",
                "task": type(env).__name__,
                "num_envs": getattr(env, "num_envs", None),
            }

        def exposed_reset(self, to_reset: Any = "all") -> Dict[str, Any]:
            env.reset(to_reset)
            _set_adjust_hand_pose_if_available(env)
            return self.exposed_get_state_snapshot()

        def exposed_reset_with_kwargs(
            self,
            to_reset: Any = "all",
            kwargs: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            env.reset(to_reset, **(kwargs or {}))
            _set_adjust_hand_pose_if_available(env)
            return self.exposed_get_state_snapshot()

        def exposed_step(self, actions: Any) -> Dict[str, Any]:
            env.step(_convert_actions(env, actions))
            _set_adjust_hand_pose_if_available(env)
            return self.exposed_get_state_snapshot()

        def exposed_collect_diff_data(self, flag: Optional[bool] = None) -> Any:
            if flag is None:
                result = env.collect_diff_data()
            else:
                result = env.collect_diff_data(flag=flag)
            return _serialize_value(result)

        def exposed_collect_single_diff_data(self, env_id: int) -> Any:
            return _serialize_value(env.collect_single_diff_data(env_id))

        def exposed_collect_rgb_frames(
            self,
            camera_type: Optional[str] = None,
            camera_ids: Optional[Iterable[int]] = None,
        ) -> Any:
            kwargs = {}
            if camera_type is not None:
                kwargs["camera_type"] = camera_type
            if camera_ids is not None:
                kwargs["camera_ids"] = list(camera_ids)
            return _serialize_value(env.collect_rgb_frames(**kwargs))

        def exposed_get_obj_dof_property_tensor(self) -> Dict[str, Any]:
            env.get_obj_dof_property_tensor()
            return self.exposed_get_state_snapshot()

        def exposed_set_gripper(self, value: bool) -> Dict[str, Any]:
            env.gripper = _convert_actions(env, value)
            return {"gripper": _serialize_value(env.gripper)}

        def exposed_set_actions(self, actions: Any) -> Dict[str, Any]:
            env.actions = _convert_actions(env, actions)
            return {"actions": _serialize_value(env.actions)}

        def exposed_get_state_snapshot(self) -> Dict[str, Any]:
            _set_adjust_hand_pose_if_available(env)
            return _collect_snapshot(env, fields)

        def exposed_get_adjust_hand_pose(self) -> Any:
            return _serialize_value(env.get_adjust_hand_pose())

        def exposed_shutdown(self) -> Dict[str, Any]:
            return {"status": "shutdown-requested"}

    return AdaManipEnvService


def serve_env(
    env: Any,
    host: str = "localhost",
    port: int = 18861,
    snapshot_fields: Optional[Iterable[str]] = None,
    protocol_config: Optional[Dict[str, Any]] = None,
) -> None:
    server = ThreadedServer(
        create_service(env, snapshot_fields=snapshot_fields),
        hostname=host,
        port=port,
        protocol_config=protocol_config or build_server_config(),
    )
    print(f"Starting AdaManip rpyc server on {host}:{port}")
    server.start()