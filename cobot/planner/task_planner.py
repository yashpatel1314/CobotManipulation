from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


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
- spawn(object_id: str)                       → add a new coloured cube to the scene; \
object_id is "<color>_cube" (e.g. "blue_cube"). Use this BEFORE manipulating a new object.
- grasp(object_id: str)                       → pick up an object already in the scene
- place_on(object_id: str, target_id: str)    → place held object on top of another
- place_at(object_id: str, position: str)     → place held object at a named position; \
position is one of: "left", "right", "center", "far_left", "far_right", \
"top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"
- push(object_id: str, direction: str)        → push an object; \
direction is one of: "left", "right", "forward", "backward"

Rules:
1. Use object_ids exactly as they appear in the scene description, or "<color>_cube" for spawn.
2. Match user color references to scene objects by color.
3. Skills must be in execution order (e.g. grasp before place_on; spawn before grasp of new object).
4. Output ONLY valid JSON — a list of objects with "skill" and "args" keys. No explanation.
5. If the command is truly impossible given the scene, output: {"error": "reason"}.
6. If the command is ambiguous, output: {"clarify": "one clarifying question"}.

Example 1: command "pick up the red block":
[{"skill": "grasp", "args": {"object_id": "red_cube"}}]

Example 2: command "put the red block at the top right corner":
[
  {"skill": "grasp", "args": {"object_id": "red_cube"}},
  {"skill": "place_at", "args": {"object_id": "red_cube", "position": "top_right"}}
]

Example 3: command "spawn a blue block then move it to the left":
[
  {"skill": "spawn",    "args": {"object_id": "blue_cube"}},
  {"skill": "grasp",    "args": {"object_id": "blue_cube"}},
  {"skill": "place_at", "args": {"object_id": "blue_cube", "position": "left"}}
]
"""

_REPLAN_PROMPT = """\
The previous plan failed at step: {failed_skill}
Failure reason: {reason}
Current scene: {scene_json}

Revise the plan to complete the original goal. Output only the remaining steps as JSON.
"""

# ---------------------------------------------------------------------------
# Rule-based planner (no API required)
# ---------------------------------------------------------------------------

# Colour names we recognise
_COLOURS = ("red", "green", "blue", "yellow", "orange", "purple")

# Default colour → object_id (scene dict overrides these at plan time)
_COLOUR_TO_ID: dict[str, str] = {c: f"{c}_cube" for c in _COLOURS}

# Direction synonyms
_DIRECTION_MAP: dict[str, str] = {
    "left":      "left",
    "right":     "right",
    "forward":   "forward",
    "front":     "forward",
    "ahead":     "forward",
    "backward":  "backward",
    "back":      "backward",
    "behind":    "backward",
}

# Position synonyms — longer/more-specific phrases MUST come before shorter ones
# so that "top right" is matched before "right" alone.
_POSITION_MAP: dict[str, str] = {
    # corners (must be before simple left/right/top/bottom entries)
    "top right corner":    "top_right",
    "upper right corner":  "top_right",
    "far right corner":    "top_right",
    "top left corner":     "top_left",
    "upper left corner":   "top_left",
    "far left corner":     "top_left",
    "bottom right corner": "bottom_right",
    "lower right corner":  "bottom_right",
    "bottom left corner":  "bottom_left",
    "lower left corner":   "bottom_left",
    # two-word combos without "corner"
    "top right":           "top_right",
    "upper right":         "top_right",
    "top left":            "top_left",
    "upper left":          "top_left",
    "bottom right":        "bottom_right",
    "near right":          "bottom_right",
    "lower right":         "bottom_right",
    "bottom left":         "bottom_left",
    "near left":           "bottom_left",
    "lower left":          "bottom_left",
    "far left":            "far_left",
    "far_left":            "far_left",
    "far right":           "far_right",
    "far_right":           "far_right",
    "far side":            "top",
    "near side":           "bottom",
    # single-word positions
    "top":                 "top",
    "forward":             "top",
    "back":                "top",
    "bottom":              "bottom",
    "front":               "bottom",
    "left":                "left",
    "right":               "right",
    "center":              "center",
    "centre":              "center",
    "middle":              "center",
}


def _find_colour(text: str) -> str | None:
    """Return the first colour word found in *text* (lowercased)."""
    for c in _COLOURS:
        if c in text:
            return c
    return None


def _find_two_colours(text: str) -> tuple[str | None, str | None]:
    """Return colours in the order they appear in *text* (left to right)."""
    hits: list[tuple[int, str]] = []
    for c in _COLOURS:
        idx = text.find(c)
        if idx >= 0:
            hits.append((idx, c))
    hits.sort(key=lambda x: x[0])
    first  = hits[0][1] if len(hits) > 0 else None
    second = hits[1][1] if len(hits) > 1 else None
    return first, second


class RuleBasedPlanner:
    """Deterministic regex/keyword planner for the fixed skill vocabulary.

    Handles all commands in the test suite without any API call.  Falls back
    gracefully (returns None) when it cannot parse the command.
    """

    def plan(self, command: str, scene: dict) -> list[SkillCall] | None:
        cmd = command.lower().strip()

        # Build scene colour→id map. Always normalise to "<color>_cube" since every
        # object in this simulation is a cube — VLM shape labels ("cylinder" etc.) are wrong.
        colour_to_id: dict[str, str] = dict(_COLOUR_TO_ID)
        for obj in scene.get("objects", []):
            colour = obj.get("color", "")
            if colour:
                colour_to_id[colour] = f"{colour}_cube"

        # ── Spawn ───────────────────────────────────────────────────────────
        spawn_m = re.search(r"\b(spawn|add|create|bring\s+in|introduce)\b", cmd)
        if spawn_m:
            colour = _find_colour(cmd)
            if colour:
                obj_id = f"{colour}_cube"
                calls: list[SkillCall] = [SkillCall("spawn", {"object_id": obj_id})]
                # If command also mentions a destination, add grasp+place_at
                position = None
                for kw, pos in sorted(_POSITION_MAP.items(), key=lambda x: -len(x[0])):
                    if kw in cmd:
                        position = pos
                        break
                if position:
                    calls += [
                        SkillCall("grasp",    {"object_id": obj_id}),
                        SkillCall("place_at", {"object_id": obj_id, "position": position}),
                    ]
                return calls

        # ── Push ────────────────────────────────────────────────────────────
        push_m = re.search(
            r"\b(push|slide|nudge|shove)\b", cmd
        )
        if push_m:
            colour = _find_colour(cmd)
            direction = None
            for kw, d in _DIRECTION_MAP.items():
                if kw in cmd:
                    direction = d
                    break
            if colour and direction:
                obj_id = colour_to_id.get(colour, f"{colour}_cube")
                return [SkillCall("push", {"object_id": obj_id, "direction": direction})]

        # ── Stack / place_on ────────────────────────────────────────────────
        stack_m = re.search(
            r"\b(stack|place\s+on\s+top|put\s+on\s+top|on\s+top\s+of)\b", cmd
        )
        pick_place_m = re.search(
            r"\b(pick\s+up|grab|grasp).{1,40}(place|put|set|drop)\b", cmd
        )
        if stack_m or pick_place_m:
            c1, c2 = _find_two_colours(cmd)
            if c1 and c2:
                obj_id    = colour_to_id.get(c1, f"{c1}_cube")
                target_id = colour_to_id.get(c2, f"{c2}_cube")
                return [
                    SkillCall("grasp",    {"object_id": obj_id}),
                    SkillCall("place_on", {"object_id": obj_id, "target_id": target_id}),
                ]

        # ── Place at named position ──────────────────────────────────────────
        place_at_m = re.search(
            r"\b(move|put|place|bring|send|go)\b", cmd
        )
        if place_at_m:
            colour   = _find_colour(cmd)
            position = None
            for kw, pos in sorted(_POSITION_MAP.items(), key=lambda x: -len(x[0])):
                if kw in cmd:
                    position = pos
                    break
            if colour and position:
                obj_id = colour_to_id.get(colour, f"{colour}_cube")
                return [
                    SkillCall("grasp",    {"object_id": obj_id}),
                    SkillCall("place_at", {"object_id": obj_id, "position": position}),
                ]

        # ── Simple grasp only ───────────────────────────────────────────────
        grasp_m = re.search(r"\b(pick\s+up|grab|grasp|lift)\b", cmd)
        if grasp_m:
            colour = _find_colour(cmd)
            if colour:
                obj_id = colour_to_id.get(colour, f"{colour}_cube")
                return [SkillCall("grasp", {"object_id": obj_id})]

        return None  # cannot parse

    def replan(
        self,
        failed_call: SkillCall,
        reason: str,
        scene: dict,
        original_command: str,
    ) -> list[SkillCall] | None:
        """Minimal replan: retry the original command."""
        return self.plan(original_command, scene)


# ---------------------------------------------------------------------------
# LLM-backed planner
# ---------------------------------------------------------------------------

class TaskPlanner:
    """Task planner: rule-based first, LLM fallback when API is available.

    The rule-based planner handles the full test suite without any network
    call.  The LLM fallback is used for open-ended or ambiguous commands and
    is skipped gracefully if the API is unreachable or rate-limited.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._max_replan = config.get("max_replan_attempts", 2)
        self._rule_planner = RuleBasedPlanner()

        self._client = None
        self._model  = ""
        provider = config.get("llm_provider", "openai")
        try:
            from openai import OpenAI
            if provider == "groq":
                self._client = OpenAI(
                    api_key=os.environ.get("GROQ_API_KEY", ""),
                    base_url="https://api.groq.com/openai/v1",
                )
                self._model = config.get("llm_model", "llama-3.3-70b-versatile")
            elif os.environ.get("OPENAI_API_KEY"):
                self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                self._model = config.get("llm_model", "gpt-4o")
        except Exception as exc:
            log.warning("LLM client init failed (%s); rule-based planner only.", exc)

        self._temperature = config.get("temperature", 0.0)

    def plan(self, command: str, scene: dict) -> list[SkillCall]:
        """Decompose a natural language command into an ordered skill sequence.

        Tries rule-based planner first; falls back to LLM when needed.
        """
        result = self._rule_planner.plan(command, scene)
        if result is not None:
            log.debug("[planner] rule-based plan for %r: %s", command, result)
            return result

        if self._client is None:
            raise ValueError(
                f"Rule-based planner could not parse command and no LLM client available: {command!r}"
            )

        log.debug("[planner] falling back to LLM for command: %r", command)
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
        result = self._rule_planner.replan(failed_call, reason, scene, original_command)
        if result is not None:
            return result

        if self._client is None:
            raise ValueError("Rule-based replan failed and no LLM client available.")

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
