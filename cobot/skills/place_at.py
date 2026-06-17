from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Skill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule


# Named positions relative to table centre in world coordinates.
# Tuned for the robosuite Stack table layout; adjust if scene changes.
# Axes: x = forward/back (positive = far from robot), y = left/right (positive = left)
# Reachable zone: x in [-0.15, +0.15], y in [-0.25, +0.25]
_TABLE_POSITIONS: dict[str, np.ndarray] = {
    "center":       np.array([ 0.00,  0.00, 0.82]),
    "left":         np.array([ 0.00,  0.15, 0.82]),
    "right":        np.array([ 0.00, -0.15, 0.82]),
    "far_left":     np.array([ 0.00,  0.25, 0.82]),
    "far_right":    np.array([ 0.00, -0.25, 0.82]),
    "top":          np.array([ 0.10,  0.00, 0.82]),   # forward / far from robot
    "bottom":       np.array([-0.10,  0.00, 0.82]),   # near robot
    "top_left":     np.array([ 0.10,  0.20, 0.82]),   # far + left corner
    "top_right":    np.array([ 0.10, -0.20, 0.82]),   # far + right corner
    "bottom_left":  np.array([-0.10,  0.20, 0.82]),   # near + left corner
    "bottom_right": np.array([-0.10, -0.20, 0.82]),   # near + right corner
}

PLACE_HEIGHT_OFFSET = 0.03  # metres above table surface


class PlaceAtSkill(Skill):
    """Place the held object at a named position on the table."""

    NAME = "place_at"

    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        object_id: str,
        position: str,
        **kwargs: Any,
    ) -> bool:
        if position not in _TABLE_POSITIONS:
            raise ValueError(
                f"Unknown position '{position}'. "
                f"Valid options: {list(_TABLE_POSITIONS.keys())}"
            )
        target_pos = _TABLE_POSITIONS[position] + np.array([0.0, 0.0, PLACE_HEIGHT_OFFSET])

        if self._policy is not None:
            return self._run_policy(env, target_pos, np.array([1.0, 0.0, 0.0, 0.0]))

        return self._scripted_place_at(env, target_pos)

    def _scripted_place_at(self, env: "CobotEnv", target_pos: np.ndarray) -> bool:
        approach = target_pos + np.array([0.0, 0.0, 0.10])

        ok = self._move_to_target(env, approach, gripper_cmd=1.0)
        if not ok:
            return False

        ok = self._move_to_target(env, target_pos, tolerance=0.025, gripper_cmd=1.0, max_steps=300)
        if not ok:
            return False

        self._set_gripper(env, gripper_cmd=-1.0, steps=25)

        ee_pos = env.get_robot_state()["ee_pos"]
        self._move_to_target(env, ee_pos + np.array([0.0, 0.0, 0.10]), gripper_cmd=-1.0)
        return True

    def is_precondition_met(self, scene: dict, position: str, **kwargs: Any) -> bool:
        return position in _TABLE_POSITIONS
