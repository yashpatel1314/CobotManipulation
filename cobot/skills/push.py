from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Skill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule


_DIRECTION_VECTORS: dict[str, np.ndarray] = {
    "left":     np.array([0.0,  1.0, 0.0]),
    "right":    np.array([0.0, -1.0, 0.0]),
    "forward":  np.array([1.0,  0.0, 0.0]),
    "backward": np.array([-1.0, 0.0, 0.0]),
}

PUSH_DISTANCE = 0.12     # metres to push
PUSH_HEIGHT   = 0.840    # z-height for push sweep; matches cube centre (~0.835) plus margin
APPROACH_SIDE_OFFSET = 0.08  # metres behind the object before pushing


class PushSkill(Skill):
    """Contact-push an object along a cardinal direction without grasping.

    The end-effector approaches from the opposite side of the push direction,
    descends to table height, and sweeps through the object's centre.
    """

    NAME = "push"

    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        object_id: str,
        direction: str,
        **kwargs: Any,
    ) -> bool:
        if direction not in _DIRECTION_VECTORS:
            raise ValueError(
                f"Unknown direction '{direction}'. "
                f"Valid options: {list(_DIRECTION_VECTORS.keys())}"
            )

        if self._policy is not None:
            rgb   = env.get_scene_image()
            depth = env.get_depth_image()
            pose  = perception.get_object_pose(object_id, rgb, depth)
            obj_pos = pose.position()
            push_vec = _DIRECTION_VECTORS[direction]
            return self._run_policy(env, obj_pos + push_vec * PUSH_DISTANCE, np.array([1.0, 0.0, 0.0, 0.0]))

        obj_pos = env.get_object_pos(object_id)
        return self._scripted_push(env, obj_pos, direction, object_id)

    def _scripted_push(
        self, env: "CobotEnv", obj_pos: np.ndarray, direction: str,
        object_id: str = "",
    ) -> bool:
        push_vec  = _DIRECTION_VECTORS[direction]
        # Aim at the cube's centre height for solid mid-face contact.
        # A 5 mm lower bias helps the PD controller settle at or just below
        # centre rather than near the top edge.
        push_z = obj_pos[2] - 0.005

        # Approach from the back side of the push direction
        approach  = obj_pos - push_vec * APPROACH_SIDE_OFFSET
        approach[2] = push_z + 0.05

        # Descend to push height at approach position
        push_start = approach.copy()
        push_start[2] = push_z

        # Sweep through to push end
        push_end    = obj_pos + push_vec * PUSH_DISTANCE
        push_end[2] = push_z

        # Close gripper throughout: open fingers let the cube slip between them when
        # sweeping perpendicular to the finger-spread axis (e.g. forward direction).
        # Phase 1: move above approach (gripper closing)
        ok = self._move_to_target(env, approach, gripper_cmd=1.0)
        if not ok:
            return False

        # Phase 2: descend to push height (gripper closed)
        ok = self._move_to_target(env, push_start, tolerance=0.015, gripper_cmd=1.0)
        if not ok:
            return False

        # Phase 3: sweep push (gripper closed — solid face contacts cube)
        self._move_to_target(env, push_end, tolerance=0.02, max_steps=200, gripper_cmd=1.0)

        # Phase 4: retreat upward and reopen
        ee_pos = env.get_robot_state()["ee_pos"]
        self._move_to_target(env, ee_pos + np.array([0.0, 0.0, 0.10]), gripper_cmd=-1.0)

        # Verify: object must have moved at least 5 cm along the push axis
        if object_id:
            new_pos = env.get_object_pos(object_id)
            axis_displacement = np.dot(new_pos - obj_pos, push_vec)
            return bool(axis_displacement > 0.05)
        return True

    def is_precondition_met(self, scene: dict, object_id: str, direction: str, **kwargs: Any) -> bool:
        ids = [o.get("id", "") for o in scene.get("objects", [])]
        return object_id in ids and direction in _DIRECTION_VECTORS
