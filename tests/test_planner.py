"""Unit tests for the LLM task planner.

These tests mock the OpenAI client so no API key is required and no
network calls are made.  They verify that the planner correctly parses
LLM output into SkillCall lists and handles error / clarification cases.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from cobot.planner.task_planner import SkillCall, TaskPlanner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SCENE_TWO_BLOCKS = {
    "objects": [
        {"id": "red_block",  "color": "red",  "shape": "cube",     "pixel_u": 120, "pixel_v": 130},
        {"id": "blue_block", "color": "blue", "shape": "cube",     "pixel_u": 200, "pixel_v": 180},
        {"id": "green_cyl",  "color": "green","shape": "cylinder", "pixel_u": 80,  "pixel_v": 100},
    ]
}


def _make_planner(mock_response: str) -> tuple[TaskPlanner, MagicMock]:
    """Return a TaskPlanner whose LLM client returns mock_response.

    The rule-based layer is bypassed (returns None) so every call hits the mock.
    """
    config = {
        "llm_model": "gpt-4o",
        "max_replan_attempts": 2,
        "temperature": 0.0,
    }
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        planner = TaskPlanner(config)

    # Bypass rule-based layer so LLM mock is always exercised
    planner._rule_planner.plan   = MagicMock(return_value=None)
    planner._rule_planner.replan = MagicMock(return_value=None)

    mock_client = MagicMock()
    choice = MagicMock()
    choice.message.content = mock_response
    mock_client.chat.completions.create.return_value = MagicMock(choices=[choice])
    planner._client = mock_client
    return planner, mock_client


# ---------------------------------------------------------------------------
# Parsing — valid outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("response,expected", [
    (
        json.dumps([
            {"skill": "grasp",    "args": {"object_id": "red_block"}},
            {"skill": "place_on", "args": {"object_id": "red_block", "target_id": "blue_block"}},
        ]),
        [
            SkillCall("grasp",    {"object_id": "red_block"}),
            SkillCall("place_on", {"object_id": "red_block", "target_id": "blue_block"}),
        ],
    ),
    (
        json.dumps([{"skill": "push", "args": {"object_id": "red_block", "direction": "left"}}]),
        [SkillCall("push", {"object_id": "red_block", "direction": "left"})],
    ),
    (
        json.dumps([
            {"skill": "grasp",    "args": {"object_id": "red_block"}},
            {"skill": "place_at", "args": {"object_id": "red_block", "position": "center"}},
        ]),
        [
            SkillCall("grasp",    {"object_id": "red_block"}),
            SkillCall("place_at", {"object_id": "red_block", "position": "center"}),
        ],
    ),
])
def test_plan_parses_valid_json(response, expected):
    planner, _ = _make_planner(response)
    result = planner.plan("stack the red block on the blue one", SCENE_TWO_BLOCKS)
    assert result == expected


def test_plan_strips_markdown_fences():
    raw = '```json\n[{"skill": "grasp", "args": {"object_id": "red_block"}}]\n```'
    planner, _ = _make_planner(raw)
    result = planner.plan("pick up the red block", SCENE_TWO_BLOCKS)
    assert result == [SkillCall("grasp", {"object_id": "red_block"})]


# ---------------------------------------------------------------------------
# Parsing — error cases
# ---------------------------------------------------------------------------

def test_plan_raises_on_error_response():
    planner, _ = _make_planner(json.dumps({"error": "red_block not in scene"}))
    with pytest.raises(ValueError, match="Planner error"):
        planner.plan("pick up the purple block", SCENE_TWO_BLOCKS)


def test_plan_raises_on_clarify_response():
    planner, _ = _make_planner(json.dumps({"clarify": "Which block do you mean?"}))
    with pytest.raises(ValueError, match="clarification"):
        planner.plan("move the block", SCENE_TWO_BLOCKS)


def test_plan_raises_on_invalid_json():
    planner, _ = _make_planner("this is not json at all")
    with pytest.raises(Exception):
        planner.plan("do something", SCENE_TWO_BLOCKS)


# ---------------------------------------------------------------------------
# Replan
# ---------------------------------------------------------------------------

def test_replan_calls_api_with_failure_context():
    replan_response = json.dumps([
        {"skill": "grasp", "args": {"object_id": "blue_block"}},
    ])
    planner, mock_client = _make_planner(replan_response)

    failed_call = SkillCall("grasp", {"object_id": "red_block"})
    result = planner.replan(
        failed_call=failed_call,
        reason="gripper missed object",
        scene=SCENE_TWO_BLOCKS,
        original_command="pick up the red block",
    )

    assert len(result) == 1
    assert result[0].skill == "grasp"
    # Verify failure context was included in the user message
    call_args = mock_client.chat.completions.create.call_args
    user_content = call_args.kwargs["messages"][1]["content"]
    assert "gripper missed object" in user_content


# ---------------------------------------------------------------------------
# SkillCall dataclass
# ---------------------------------------------------------------------------

def test_skill_call_repr():
    call = SkillCall("place_on", {"object_id": "red_block", "target_id": "blue_block"})
    r = repr(call)
    assert "place_on" in r
    assert "red_block" in r


def test_skill_call_equality():
    a = SkillCall("grasp", {"object_id": "red_block"})
    b = SkillCall("grasp", {"object_id": "red_block"})
    assert a == b


# ---------------------------------------------------------------------------
# 10 fixed command/scene pairs for regression
# ---------------------------------------------------------------------------

REGRESSION_CASES = [
    ("pick up the red block",
     [SkillCall("grasp", {"object_id": "red_block"})]),
    ("pick up the blue block",
     [SkillCall("grasp", {"object_id": "blue_block"})]),
    ("push the red block to the left",
     [SkillCall("push", {"object_id": "red_block", "direction": "left"})]),
    ("push the green cylinder forward",
     [SkillCall("push", {"object_id": "green_cyl", "direction": "forward"})]),
    ("move the red block to the center",
     [SkillCall("grasp", {"object_id": "red_block"}),
      SkillCall("place_at", {"object_id": "red_block", "position": "center"})]),
    ("stack the red block on the blue one",
     [SkillCall("grasp", {"object_id": "red_block"}),
      SkillCall("place_on", {"object_id": "red_block", "target_id": "blue_block"})]),
    ("stack the blue block on the red block",
     [SkillCall("grasp", {"object_id": "blue_block"}),
      SkillCall("place_on", {"object_id": "blue_block", "target_id": "red_block"})]),
    ("push the blue block to the right",
     [SkillCall("push", {"object_id": "blue_block", "direction": "right"})]),
    ("pick up the red block and place it to the right",
     [SkillCall("grasp", {"object_id": "red_block"}),
      SkillCall("place_at", {"object_id": "red_block", "position": "right"})]),
    ("move the blue block to the far left",
     [SkillCall("grasp", {"object_id": "blue_block"}),
      SkillCall("place_at", {"object_id": "blue_block", "position": "far_left"})]),
]


@pytest.mark.parametrize("command,expected_plan", REGRESSION_CASES)
def test_regression_plan_structure(command, expected_plan):
    """Verify the planner produces the expected skill sequence for known commands."""
    response = json.dumps([
        {"skill": c.skill, "args": c.args} for c in expected_plan
    ])
    planner, _ = _make_planner(response)
    result = planner.plan(command, SCENE_TWO_BLOCKS)
    assert result == expected_plan


# ---------------------------------------------------------------------------
# Rule-based planner — no LLM needed, no mock required
# These commands are intercepted before any API call.
# ---------------------------------------------------------------------------

_CATALOG = {
    "red":    {"id": "red_cube",   "shape": "cube"},
    "green":  {"id": "green_cube", "shape": "cube"},
    "blue":   {"id": "blue_cylinder", "shape": "cylinder"},
    "yellow": {"id": "yellow_sphere", "shape": "sphere"},
}

_SCENE_ON_TABLE = {
    "catalog": _CATALOG,
    "objects": [
        {"id": "red_cube",      "color": "red",   "shape": "cube"},
        {"id": "green_cube",    "color": "green", "shape": "cube"},
        {"id": "blue_cylinder", "color": "blue",  "shape": "cylinder"},
    ],
}

_SCENE_WITH_DISTRACTOR = {
    "catalog": _CATALOG,
    "objects": [
        {"id": "red_cube",    "color": "red",   "shape": "cube"},
        {"id": "green_cube",  "color": "green", "shape": "cube"},
        {"id": "distractor0", "color": "grey",  "shape": "cube"},  # not in catalog
    ],
}

_SCENE_EMPTY = {"catalog": _CATALOG, "objects": []}


def _rule_planner() -> TaskPlanner:
    """Planner with a dummy LLM config — rule-based commands never hit the API."""
    config = {"llm_model": "dummy", "temperature": 0.0}
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        return TaskPlanner(config)


@pytest.mark.parametrize("command,direction", [
    ("rotate the red cube clockwise",         "clockwise"),
    ("spin the green block counterclockwise", "counterclockwise"),
    ("turn the red block to the right",       "clockwise"),
    ("rotate the green cube to the left",     "counterclockwise"),
])
def test_rule_rotate(command, direction):
    p = _rule_planner()
    plan = p.plan(command, _SCENE_EMPTY)
    assert len(plan) == 2
    assert plan[0].skill == "grasp"
    assert plan[1].skill == "rotate"
    assert plan[1].args["direction"] == direction


def test_rule_sort_returns_atomic_skill():
    p = _rule_planner()
    for cmd in ("sort all the objects", "line up all objects", "arrange them in a line"):
        plan = p.plan(cmd, _SCENE_ON_TABLE)
        assert len(plan) == 1
        assert plan[0].skill == "sort"
        assert plan[0].args == {}


def test_rule_clear_returns_atomic_skill():
    p = _rule_planner()
    for cmd in ("clear the table", "clean up everything", "tidy up the workspace"):
        plan = p.plan(cmd, _SCENE_ON_TABLE)
        assert len(plan) == 1
        assert plan[0].skill == "clear"
        assert plan[0].args == {}


def test_rule_push_basic():
    p = _rule_planner()
    plan = p.plan("push the red block to the left", _SCENE_ON_TABLE)
    assert plan == [SkillCall("push", {"object_id": "red_cube", "direction": "left"})]


def test_rule_grasp_uses_catalog_id():
    """Planner should use 'blue_cylinder' from catalog, not 'blue_cube'."""
    p = _rule_planner()
    plan = p.plan("pick up the blue cylinder", _SCENE_ON_TABLE)
    assert any(c.skill == "grasp" and c.args.get("object_id") == "blue_cylinder" for c in plan)
