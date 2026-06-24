from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .grasp import GraspSkill
from .place_at import PlaceAtSkill
from .place_on import PlaceOnSkill
from .push import PushSkill
from .rotate import RotateSkill
from .spawn import SpawnSkill

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule
    from cobot.planner.task_planner import SkillCall


_SKILL_CLASSES = {
    "grasp":    GraspSkill,
    "place_on": PlaceOnSkill,
    "place_at": PlaceAtSkill,
    "push":     PushSkill,
    "rotate":   RotateSkill,
    "spawn":    SpawnSkill,
}


class SkillLibrary:
    """Registry and dispatcher for manipulation primitives.

    Instantiates each skill once with the shared config so checkpoints are
    loaded only at startup.
    """

    def __init__(self, config: dict) -> None:
        self._skills = {
            name: cls(config) for name, cls in _SKILL_CLASSES.items()
        }

    def execute(
        self,
        call: "SkillCall",
        env: "CobotEnv",
        perception: "PerceptionModule",
    ) -> tuple[bool, str]:
        """Execute a skill call and return (success, failure_reason).

        failure_reason is an empty string on success.
        """
        skill = self._skills.get(call.skill)
        if skill is None:
            return False, f"Unknown skill '{call.skill}'"

        try:
            success = bool(skill.execute(env, perception, **call.args))
            return success, "" if success else f"{call.skill} failed (policy returned False)"
        except Exception as exc:
            return False, f"{call.skill} raised exception: {exc}"

    def check_preconditions(
        self, call: "SkillCall", scene: dict
    ) -> tuple[bool, str]:
        skill = self._skills.get(call.skill)
        if skill is None:
            return False, f"Unknown skill '{call.skill}'"
        ok = skill.is_precondition_met(scene, **call.args)
        return ok, "" if ok else f"Precondition not met for {call!r}"

    @property
    def available_skills(self) -> list[str]:
        return list(self._skills.keys())
