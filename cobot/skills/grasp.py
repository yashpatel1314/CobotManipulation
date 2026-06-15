from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Skill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule


class GraspSkill(Skill):
    """Reach, grasp, and lift a target object.

    Execution phases (scripted fallback):
      1. Move end-effector to pre-grasp position (10 cm above object)
      2. Descend to grasp height (2 cm above object)
      3. Close gripper
      4. Lift to a safe carry height
    """

    NAME = "grasp"

    PRE_GRASP_OFFSET = np.array([0.0, 0.0, 0.10])   # metres above object centre
    GRASP_OFFSET     = np.array([0.0, 0.0, -0.01])  # slightly below object centre for secure grip
    LIFT_OFFSET      = np.array([0.0, 0.0, 0.15])   # metres above table

    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        object_id: str,
        **kwargs: Any,
    ) -> bool:
        if self._policy is not None:
            rgb   = env.get_scene_image()
            depth = env.get_depth_image()
            pose  = perception.get_object_pose(object_id, rgb, depth)
            obj_pos = pose.position()
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])
            return self._run_policy(env, obj_pos, target_quat)

        # Scripted fallback: use ground-truth sim position for precision
        obj_pos = env.get_object_pos(object_id)
        return self._scripted_grasp(env, obj_pos, object_id)

    def _scripted_grasp(self, env: "CobotEnv", obj_pos: np.ndarray, object_id: str = "") -> bool:
        # Phase 1: move to pre-grasp
        ok = self._move_to_target(env, obj_pos + self.PRE_GRASP_OFFSET, gripper_cmd=-1.0)
        if not ok:
            return False

        # Phase 2: descend
        ok = self._move_to_target(env, obj_pos + self.GRASP_OFFSET, tolerance=0.01, gripper_cmd=-1.0)
        if not ok:
            return False

        # Phase 3: close gripper
        self._set_gripper(env, gripper_cmd=1.0, steps=50)

        # Phase 4: lift
        ee_pos = env.get_robot_state()["ee_pos"]
        lift_target = np.array([ee_pos[0], ee_pos[1], obj_pos[2] + self.LIFT_OFFSET[2]])
        self._move_to_target(env, lift_target, gripper_cmd=1.0)

        # Check object actually rose with the arm (ground-truth position check)
        new_obj_pos = env.get_object_pos(object_id)
        return new_obj_pos[2] > obj_pos[2] + 0.03

    def is_precondition_met(self, scene: dict, object_id: str, **kwargs: Any) -> bool:
        ids = [o.get("id", "") for o in scene.get("objects", [])]
        return object_id in ids
