from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import Skill
from .grasp import GraspSkill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule

# Handover staging point: accessible to both arms
_HANDOVER_PLACE = np.array([0.00, 0.00, 0.83])   # on table, centred
_CARRY_HEIGHT   = np.array([0.00, 0.00, 0.98])    # above staging point


class HandoverSkill(Skill):
    """Transfer an object between arms via a place-and-pick sequence.

    Requires a DualArmCobotEnv.  Place-and-pick is more reliable than
    simultaneous grip exchange under OSC control.

    Phases:
      1. Arm `from_arm` grasps the object.
      2. Arm `from_arm` carries object above the staging point.
      3. Arm `from_arm` places object at the staging point.
      4. Arm `to_arm` grasps the object from the staging point.
      5. Arm `to_arm` lifts.
    """

    NAME = "handover"

    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        object_id: str,
        from_arm: int = 0,
        to_arm: int = 1,
        **kwargs: Any,
    ) -> bool:
        if not hasattr(env, "active_arm"):
            raise RuntimeError("HandoverSkill requires DualArmCobotEnv")

        grasp = GraspSkill(self._config)

        # Phase 1: from_arm grasps
        env.active_arm = from_arm
        if not grasp.execute(env, perception, object_id=object_id):
            return False

        # Phase 2: carry to above staging point
        self._move_to_target(env, _CARRY_HEIGHT, gripper_cmd=1.0)

        # Phase 3: lower to staging point and release
        half_h = env.get_object_half_height(object_id)
        place_pos = _HANDOVER_PLACE + np.array([0, 0, half_h])
        self._move_to_target(env, place_pos, tolerance=0.015, gripper_cmd=1.0)
        self._set_gripper(env, gripper_cmd=-1.0, steps=30)
        # Retract
        self._move_to_target(env, _CARRY_HEIGHT, gripper_cmd=-1.0)

        # Phase 4-5: to_arm grasps and lifts
        env.active_arm = to_arm
        ok = grasp.execute(env, perception, object_id=object_id)

        env.active_arm = 0  # reset after handover
        new_pos = env.get_object_pos(object_id)
        return ok and bool(new_pos[2] > 0.88)

    def is_precondition_met(self, scene: dict, object_id: str, **kwargs: Any) -> bool:
        return any(o.get("id") == object_id for o in scene.get("objects", []))
