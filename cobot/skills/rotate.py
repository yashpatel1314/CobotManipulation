from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Skill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule

# Rotation directions and their yaw-delta signs
_ROTATE_DIRS = {"clockwise": -1.0, "counterclockwise": 1.0, "left": 1.0, "right": -1.0}
_ROTATE_STEPS = 30  # steps of yaw-delta application


class RotateSkill(Skill):
    """Rotate the held object around the vertical axis while keeping it gripped.

    Assumes GraspSkill has already run (object is in the gripper).

    action[5] = yaw-rate delta — the BASIC OSC_POSE controller maps this to
    end-effector angular velocity around Z.
    """

    NAME = "rotate"

    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        object_id: str,
        direction: str = "clockwise",
        **kwargs: Any,
    ) -> bool:
        sign = _ROTATE_DIRS.get(direction.lower(), -1.0)
        return self._scripted_rotate(env, object_id, sign)

    def _scripted_rotate(self, env: "CobotEnv", object_id: str, sign: float) -> bool:
        obj_pos_before = env.get_object_pos(object_id)

        # Phase 1: steady rotation — apply yaw delta while holding
        action = np.zeros(env.action_dim)
        action[5] = sign * 0.8   # yaw-rate command
        action[-1] = 1.0          # keep gripper closed
        for _ in range(_ROTATE_STEPS):
            env.step(action)

        # Phase 2: stop rotation and stabilise
        action_stop = np.zeros(env.action_dim)
        action_stop[-1] = 1.0
        for _ in range(15):
            env.step(action_stop)

        # Verify object stayed in gripper (didn't drop)
        obj_pos_after = env.get_object_pos(object_id)
        return bool(obj_pos_after[2] > obj_pos_before[2] - 0.05)

    def is_precondition_met(self, scene: dict, object_id: str, direction: str = "clockwise", **kwargs: Any) -> bool:
        return direction.lower() in _ROTATE_DIRS
