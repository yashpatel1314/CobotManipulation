from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class SkillCall:
    skill: str
    args: dict[str, Any]

    def __repr__(self) -> str:
        return f"{self.skill}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"


_SYSTEM_PROMPT = """\
You are a robot task planner. Given a JSON scene description and a user command, \
output a JSON array of skill calls to execute on the robot arm.

Available skills and their signatures:
- grasp(object_id: str)                       → pick up an object
- place_on(object_id: str, target_id: str)    → place held object on top of another
- place_at(object_id: str, position: str)     → place held object at a named position; \
position is one of: "left", "right", "center", "far_left", "far_right"
- push(object_id: str, direction: str)        → push an object; \
direction is one of: "left", "right", "forward", "backward"

Rules:
1. Use object_ids exactly as they appear in the scene description.
2. Match user color references to scene objects by color (e.g. "red block", "red cube", and "red one" all refer to the object whose color field is "red").
3. Skills must be in execution order (e.g. grasp before place_on).
4. Output ONLY valid JSON — a list of objects with "skill" and "args" keys. No explanation.
5. If the command is truly impossible given the scene (e.g. colour not present at all), output: {"error": "reason"}.
6. If the command is ambiguous, output: {"clarify": "one clarifying question"}.

Example: scene has {"id": "red_cube", "color": "red"} and command is "pick up the red block":
[
  {"skill": "grasp", "args": {"object_id": "red_cube"}}
]
"""

_REPLAN_PROMPT = """\
The previous plan failed at step: {failed_skill}
Failure reason: {reason}
Current scene: {scene_json}

Revise the plan to complete the original goal. Output only the remaining steps as JSON.
"""


class TaskPlanner:
    """LLM-based task planner that maps natural language + scene → skill sequence."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._max_replan = config.get("max_replan_attempts", 2)

        from openai import OpenAI
        provider = config.get("llm_provider", "openai")
        if provider == "groq":
            self._client = OpenAI(
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            )
            self._model = config.get("llm_model", "llama-3.3-70b-versatile")
        else:
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self._model = config.get("llm_model", "gpt-4o")
        self._temperature = config.get("temperature", 0.0)

    def plan(self, command: str, scene: dict) -> list[SkillCall]:
        """Decompose a natural language command into an ordered skill sequence."""
        user_msg = f"Scene: {json.dumps(scene)}\n\nCommand: {command}"
        raw = self._complete(user_msg)
        return self._parse_plan(raw)

    def replan(
        self,
        failed_call: SkillCall,
        reason: str,
        scene: dict,
        original_command: str,
    ) -> list[SkillCall]:
        """Re-plan after a skill failure, providing failure context."""
        user_msg = _REPLAN_PROMPT.format(
            failed_skill=repr(failed_call),
            reason=reason,
            scene_json=json.dumps(scene),
        ) + f"\nOriginal command: {original_command}"
        raw = self._complete(user_msg)
        return self._parse_plan(raw)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _complete(self, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=512,
            temperature=self._temperature,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _parse_plan(raw: str) -> list[SkillCall]:
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()

        data = json.loads(raw)

        if isinstance(data, dict):
            if "error" in data:
                raise ValueError(f"Planner error: {data['error']}")
            if "clarify" in data:
                raise ValueError(f"Planner needs clarification: {data['clarify']}")
            raise ValueError(f"Unexpected planner output: {data}")

        calls = []
        for item in data:
            calls.append(SkillCall(skill=item["skill"], args=item.get("args", {})))
        return calls
