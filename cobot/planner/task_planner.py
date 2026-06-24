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

The scene has a "catalog" (colour → {id, shape}) and "objects" listing on-table items.
Always use the object_id from the catalog, NOT a guessed "<color>_cube" form.

Skills:
- spawn(object_id)                      → add off-table object; REQUIRED before grasping blue/yellow/orange/purple
- grasp(object_id)                      → pick up an on-table object
- place_on(object_id, target_id)        → stack held object on top of target
- place_at(object_id, position)         → move to named position: left, right, center, far_left, far_right,
                                          top, bottom, top_left, top_right, bottom_left, bottom_right,
                                          adj_left, adj_right
- push(object_id, direction)            → push; direction: left, right, forward, backward
- rotate(object_id, direction)          → rotate held object; direction: clockwise, counterclockwise

Rules:
1. Use ids from catalog exactly. Red/green are always on-table; do NOT spawn them.
2. Sort/line-up: use place_at with spread positions (far_left → left → center → right → far_right).
3. adj_left/adj_right are for pyramid bases (~4 cm apart, touching).
4. Output ONLY valid JSON — a list with "skill" and "args" keys.

Example 1 — pick up red:
[{"skill":"grasp","args":{"object_id":"red_cube"}}]

Example 2 — spawn blue cylinder and place at top right:
[{"skill":"spawn","args":{"object_id":"blue_cylinder"}},
 {"skill":"grasp","args":{"object_id":"blue_cylinder"}},
 {"skill":"place_at","args":{"object_id":"blue_cylinder","position":"top_right"}}]

Example 3 — rotate red cube clockwise:
[{"skill":"grasp","args":{"object_id":"red_cube"}},
 {"skill":"rotate","args":{"object_id":"red_cube","direction":"clockwise"}}]

Example 4 — sort all 3 on-table objects (red, green, blue_cylinder) left to right:
[{"skill":"grasp","args":{"object_id":"red_cube"}},
 {"skill":"place_at","args":{"object_id":"red_cube","position":"far_left"}},
 {"skill":"grasp","args":{"object_id":"green_cube"}},
 {"skill":"place_at","args":{"object_id":"green_cube","position":"center"}},
 {"skill":"grasp","args":{"object_id":"blue_cylinder"}},
 {"skill":"place_at","args":{"object_id":"blue_cylinder","position":"far_right"}}]

Example 5 — pyramid with yellow and purple at bottom, green on top:
[{"skill":"spawn","args":{"object_id":"yellow_sphere"}},
 {"skill":"grasp","args":{"object_id":"yellow_sphere"}},
 {"skill":"place_at","args":{"object_id":"yellow_sphere","position":"adj_left"}},
 {"skill":"spawn","args":{"object_id":"purple_cube"}},
 {"skill":"grasp","args":{"object_id":"purple_cube"}},
 {"skill":"place_at","args":{"object_id":"purple_cube","position":"adj_right"}},
 {"skill":"grasp","args":{"object_id":"green_cube"}},
 {"skill":"place_on","args":{"object_id":"green_cube","target_id":"yellow_sphere"}}]

Example 6 — clear the table (park every object to the sides):
[{"skill":"grasp","args":{"object_id":"red_cube"}},
 {"skill":"place_at","args":{"object_id":"red_cube","position":"far_right"}},
 {"skill":"grasp","args":{"object_id":"green_cube"}},
 {"skill":"place_at","args":{"object_id":"green_cube","position":"far_left"}}]
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

# Default colour → object_id (catalog from scene overrides these at plan time)
_COLOUR_TO_ID: dict[str, str] = {c: f"{c}_cube" for c in _COLOURS}

# Shape synonyms — user may say "block", "ball", "prism", etc.
_SHAPE_SYNONYMS: dict[str, str] = {
    "block":    "cube",
    "cube":     "cube",
    "box":      "cube",
    "cylinder": "cylinder",
    "tube":     "cylinder",
    "pillar":   "cylinder",
    "sphere":   "sphere",
    "ball":     "sphere",
    "orb":      "sphere",
    "cone":     "cone",
    "pyramid":  "cone",
    "prism":    "cylinder",
}
_SHAPE_WORDS = frozenset(_SHAPE_SYNONYMS.keys())

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
    """Return the colour word that appears earliest in *text* by position."""
    hits: list[tuple[int, str]] = []
    for c in _COLOURS:
        idx = text.find(c)
        if idx >= 0:
            hits.append((idx, c))
    if not hits:
        return None
    return min(hits, key=lambda x: x[0])[1]


def _find_colour_after(text: str, pos: int) -> str | None:
    """Return the colour word that appears first at or after *pos* in *text*."""
    hits: list[tuple[int, str]] = []
    for c in _COLOURS:
        idx = text.find(c, pos)
        if idx >= 0:
            hits.append((idx, c))
    if not hits:
        return None
    return min(hits, key=lambda x: x[0])[1]


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


def _find_three_colours(text: str) -> tuple[str | None, str | None, str | None]:
    """Return up to three colours in the order they appear in *text*."""
    hits: list[tuple[int, str]] = []
    for c in _COLOURS:
        idx = text.find(c)
        if idx >= 0:
            hits.append((idx, c))
    hits.sort(key=lambda x: x[0])
    return (
        hits[0][1] if len(hits) > 0 else None,
        hits[1][1] if len(hits) > 1 else None,
        hits[2][1] if len(hits) > 2 else None,
    )


def _resolve_id(colour: str, scene: dict) -> str:
    """Return canonical object_id for a colour using the scene catalog.

    Falls back to "<colour>_cube" when catalog is absent (e.g. test fixtures).
    """
    catalog = scene.get("catalog", {})
    entry = catalog.get(colour)
    if entry:
        return entry["id"]
    return f"{colour}_cube"


# Colours that are not on the table at reset and must be spawned before use
_EXTRA_COLOURS: frozenset[str] = frozenset({"blue", "yellow", "orange", "purple"})


class RuleBasedPlanner:
    """Deterministic regex/keyword planner for the fixed skill vocabulary.

    Handles all commands in the test suite without any API call.  Falls back
    gracefully (returns None) when it cannot parse the command.
    """

    def plan(self, command: str, scene: dict) -> list[SkillCall] | None:
        cmd = command.lower().strip()

        # Build colour→id map: catalog takes priority, then scene objects, then defaults
        colour_to_id: dict[str, str] = dict(_COLOUR_TO_ID)
        for obj in scene.get("objects", []):
            colour = obj.get("color", "")
            oid    = obj.get("id", "")
            if colour and oid:
                colour_to_id[colour] = oid
        for colour, entry in scene.get("catalog", {}).items():
            if "id" in entry:
                colour_to_id[colour] = entry["id"]

        # ── Pyramid ─────────────────────────────────────────────────────────
        # "pyramid with X and Y at the bottom and Z at the top"
        # Strategy: place the two base cubes at adj_left / adj_right (4 cm
        # apart so they are touching), then stack the apex cube on the left
        # base. Cubes that are not initially on the table are spawned first.
        pyramid_m = re.search(r"\bpyramid\b", cmd)
        if pyramid_m:
            c1, c2, c3 = _find_three_colours(cmd)
            # Need exactly 3 colours; infer apex from "top" keyword position
            if c1 and c2 and c3:
                top_m = re.search(r"\btop\b", cmd)
                if top_m:
                    top_pos = top_m.start()
                    # The colour nearest to "top" is the apex
                    dists = {c: abs(cmd.find(c) - top_pos) for c in (c1, c2, c3)}
                    apex = min(dists, key=lambda c: dists[c])
                    bases = [c for c in (c1, c2, c3) if c != apex]
                else:
                    apex, bases = c3, [c1, c2]  # last colour by text position = top

                calls: list[SkillCall] = []
                base_positions = ["adj_left", "adj_right"]
                for base_colour, pos in zip(bases, base_positions):
                    base_id = colour_to_id.get(base_colour, f"{base_colour}_cube")
                    if base_colour in _EXTRA_COLOURS:
                        calls.append(SkillCall("spawn", {"object_id": base_id}))
                    calls += [
                        SkillCall("grasp",    {"object_id": base_id}),
                        SkillCall("place_at", {"object_id": base_id, "position": pos}),
                    ]
                apex_id  = colour_to_id.get(apex, f"{apex}_cube")
                base0_id = colour_to_id.get(bases[0], f"{bases[0]}_cube")
                if apex in _EXTRA_COLOURS:
                    calls.append(SkillCall("spawn", {"object_id": apex_id}))
                calls += [
                    SkillCall("grasp",    {"object_id": apex_id}),
                    SkillCall("place_on", {"object_id": apex_id,
                                           "target_id": base0_id}),
                ]
                return calls

        # ── Sort / line-up / arrange all objects ────────────────────────────
        # "sort", "line up", "arrange", "put in a line", "organize"
        sort_m = re.search(
            r"\b(sort|line\s+up|line\s+them\s+up|arrange|organize|organise|"
            r"put\s+(?:them\s+)?in\s+a\s+(?:straight\s+)?line)\b",
            cmd,
        )
        if sort_m:
            catalog_colors = set(scene.get("catalog", {}).keys())
            on_table = [o for o in scene.get("objects", [])
                        if o.get("color") in catalog_colors and o.get("id")]
            if on_table:
                # Sort alphabetically by colour for deterministic ordering
                on_table.sort(key=lambda o: o.get("color", ""))
                spread = ["far_left", "left", "center", "right", "far_right"]
                calls: list[SkillCall] = []
                for i, obj in enumerate(on_table[:5]):
                    oid = obj["id"]
                    calls += [
                        SkillCall("grasp",    {"object_id": oid}),
                        SkillCall("place_at", {"object_id": oid, "position": spread[i]}),
                    ]
                return calls

        # ── Clear table / clear area ─────────────────────────────────────────
        clear_m = re.search(
            r"\b(clear|clean\s+up|clean\s+off|remove\s+everything|tidy\s+up|"
            r"move\s+everything|sweep\s+(?:everything|all))\b",
            cmd,
        )
        if clear_m:
            catalog_colors = set(scene.get("catalog", {}).keys())
            on_table = [o for o in scene.get("objects", [])
                        if o.get("color") in catalog_colors and o.get("id")]
            if on_table:
                parking = ["far_right", "far_left", "top_right", "top_left", "bottom_right"]
                calls: list[SkillCall] = []
                for i, obj in enumerate(on_table[:5]):
                    oid = obj["id"]
                    calls += [
                        SkillCall("grasp",    {"object_id": oid}),
                        SkillCall("place_at", {"object_id": oid, "position": parking[i % len(parking)]}),
                    ]
                return calls

        # ── Rotate ──────────────────────────────────────────────────────────
        rotate_m = re.search(r"\b(rotate|spin|turn)\b", cmd)
        if rotate_m:
            colour = _find_colour(cmd)
            direction = "clockwise"
            if re.search(r"\b(counter\s*clockwise|anti\s*clockwise|left)\b", cmd):
                direction = "counterclockwise"
            elif re.search(r"\b(clockwise|right)\b", cmd):
                direction = "clockwise"
            if colour:
                obj_id = colour_to_id.get(colour, f"{colour}_cube")
                return [
                    SkillCall("grasp",  {"object_id": obj_id}),
                    SkillCall("rotate", {"object_id": obj_id, "direction": direction}),
                ]

        # ── Spawn ───────────────────────────────────────────────────────────
        spawn_m = re.search(r"\b(spawn|add|create|bring\s+in|introduce)\b", cmd)
        if spawn_m:
            # If manipulation verbs appear BEFORE the spawn keyword the command
            # is a multi-operation sequence we cannot fully parse (e.g. "move
            # red … then spawn blue … arrange in a line").  Return None so the
            # LLM handles the whole thing.
            pre_spawn = cmd[: spawn_m.start()]
            if re.search(r"\b(put|move|place|slide|push|grab|grasp)\b", pre_spawn):
                return None

            # Prefer the colour closest to (and after) the spawn verb so that
            # "put red … then spawn a blue block" correctly identifies blue.
            colour = _find_colour_after(cmd, spawn_m.end()) or _find_colour(cmd)
            if colour:
                obj_id = colour_to_id.get(colour, f"{colour}_cube")
                calls: list[SkillCall] = [SkillCall("spawn", {"object_id": obj_id})]

                # "on top of <colour>" / "onto" / "place/put it on the <colour>" → place_on
                _colour_alt = "|".join(_COLOURS)
                on_top_m = re.search(
                    r"\bon\s+top\s+of\b"
                    r"|\bonto\b"
                    rf"|\b(?:place|put)\s+(?:\w+\s+)?on\s+(?:the\s+)?(?:{_colour_alt})\b",
                    cmd,
                )
                if on_top_m:
                    c1, c2 = _find_two_colours(cmd)
                    target_colour = c2 if c1 == colour else c1
                    if target_colour:
                        target_id = colour_to_id.get(target_colour, f"{target_colour}_cube")
                        calls += [
                            SkillCall("grasp",    {"object_id": obj_id}),
                            SkillCall("place_on", {"object_id": obj_id,
                                                   "target_id": target_id}),
                        ]
                        return calls

                # Named table position → place_at (skip "top" when it's part of "on top of")
                position = None
                for kw, pos in sorted(_POSITION_MAP.items(), key=lambda x: -len(x[0])):
                    if kw == "top" and "on top of" in cmd:
                        continue
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
            r"\b(move|put|place|bring|send|go|take)\b", cmd
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
