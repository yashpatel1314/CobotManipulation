from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Skill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule

_PARKING = ["far_right", "far_left", "top_right", "top_left", "bottom_right"]


class ClearSkill(Skill):
    """Move every on-table task object to the edges of the workspace.

    Best-effort: continues even if one object fails, and returns False
    if any step failed (so the orchestrator can replan).
    """

    NAME = "clear"

    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        **kwargs: Any,
    ) -> bool:
        from .grasp import GraspSkill
        from .place_at import PlaceAtSkill

        _grasp = GraspSkill(self._config)
        _place = PlaceAtSkill(self._config)

        scene = env.get_sim_scene_description()
        catalog_colors = set(scene.get("catalog", {}).keys())
        objects = [
            o for o in scene["objects"]
            if o.get("color") in catalog_colors and o.get("id")
        ]

        all_ok = True
        for i, obj in enumerate(objects[: len(_PARKING)]):
            pos = _PARKING[i % len(_PARKING)]
            if not _grasp.execute(env, perception, object_id=obj["id"]):
                all_ok = False
                continue
            if not _place.execute(env, perception, object_id=obj["id"], position=pos):
                all_ok = False
        return all_ok

    def is_precondition_met(self, scene: dict, **kwargs: Any) -> bool:
        return len(scene.get("objects", [])) > 0
