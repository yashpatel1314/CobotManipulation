from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Skill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule


class SpawnSkill(Skill):
    """Teleport an off-table cube onto the table surface.

    The cube must have been pre-loaded into the sim (listed under extra_colors
    in config.yaml).  Spawning is instantaneous — the cube appears at a random
    reachable position on the table.
    """

    NAME = "spawn"

    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        object_id: str,
        **kwargs: Any,
    ) -> bool:
        color = object_id.split("_")[0]
        ok = env.spawn_object(color)
        if not ok:
            import logging
            logging.getLogger(__name__).warning(
                "spawn: '%s' is not a pre-loaded extra colour or is already a base cube.",
                color,
            )
        return ok

    def is_precondition_met(self, scene: dict, object_id: str, **kwargs: Any) -> bool:
        return True
