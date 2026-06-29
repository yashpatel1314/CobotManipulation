from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Skill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule

_SORT_POSITIONS = ["far_left", "left", "center", "right", "far_right"]


class SortSkill(Skill):
    """Sort all on-table task objects left-to-right alphabetically by colour.

    Internally chains GraspSkill → PlaceAtSkill for each object.
    Returns True only if every object was successfully placed.
    """

    NAME = "sort"

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
        objects = sorted(
            [o for o in scene["objects"] if o.get("color") in catalog_colors and o.get("id")],
            key=lambda o: o.get("color", ""),
        )

        for i, obj in enumerate(objects[: len(_SORT_POSITIONS)]):
            if not _grasp.execute(env, perception, object_id=obj["id"]):
                return False
            if not _place.execute(env, perception, object_id=obj["id"], position=_SORT_POSITIONS[i]):
                return False
        return True

    def is_precondition_met(self, scene: dict, **kwargs: Any) -> bool:
        return len(scene.get("objects", [])) >= 2
