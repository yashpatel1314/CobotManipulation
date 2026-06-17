from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Skill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule


class PlaceOnSkill(Skill):
    """Place the held object on top of a target object.

    Assumes the arm is already holding an object (i.e. GraspSkill succeeded).

    Scripted phases:
      1. Move above the target object
      2. Descend until the held object rests on the target
      3. Release gripper
      4. Retreat upward
    """

    NAME = "place_on"

    APPROACH_OFFSET = np.array([0.0, 0.0, 0.12])  # metres above target
    PLACE_OFFSET    = np.array([0.0, 0.0, 0.05])   # slight clearance above target top

    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        object_id: str,
        target_id: str,
        **kwargs: Any,
    ) -> bool:
        if self._policy is not None:
            rgb   = env.get_scene_image()
            depth = env.get_depth_image()
            target_pose = perception.get_object_pose(target_id, rgb, depth)
            return self._run_policy(env, target_pose.position(), np.array([1.0, 0.0, 0.0, 0.0]))

        target_pos = env.get_object_pos(target_id)
        return self._scripted_place_on(env, target_pos, object_id)

    def _scripted_place_on(self, env: "CobotEnv", target_pos: np.ndarray, object_id: str = "") -> bool:
        # Phase 1: move above target
        ok = self._move_to_target(
            env, target_pos + self.APPROACH_OFFSET, gripper_cmd=1.0
        )
        if not ok:
            return False

        # Phase 2: lower onto target
        ok = self._move_to_target(
            env, target_pos + self.PLACE_OFFSET, tolerance=0.015, gripper_cmd=1.0
        )
        if not ok:
            return False

        # Phase 3: release
        self._set_gripper(env, gripper_cmd=-1.0, steps=25)

        # Phase 4: retreat
        ee_pos = env.get_robot_state()["ee_pos"]
        retreat = np.array([ee_pos[0], ee_pos[1], ee_pos[2] + 0.10])
        self._move_to_target(env, retreat, gripper_cmd=-1.0)

        # Verify the held object is now resting on the target (above target top surface)
        if object_id:
            placed_pos = env.get_object_pos(object_id)
            target_top = target_pos[2] + 0.025  # cube half-height
            return bool(placed_pos[2] > target_top + 0.01)
        return True

    def is_precondition_met(
        self, scene: dict, object_id: str, target_id: str, **kwargs: Any
    ) -> bool:
        ids = [o.get("id", "") for o in scene.get("objects", [])]
        return target_id in ids
