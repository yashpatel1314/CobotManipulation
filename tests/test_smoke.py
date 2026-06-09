"""Regression smoke tests — fast, no GPU, no API calls.

These tests verify that the core modules can be imported and instantiated
without crashing, and that the skill dispatch layer behaves correctly.
They do NOT run the MuJoCo simulation or make any network requests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import smoke tests
# ---------------------------------------------------------------------------

def test_import_cobot_package():
    import cobot
    assert cobot.__version__


def test_import_planner():
    from cobot.planner.task_planner import SkillCall, TaskPlanner
    assert SkillCall


def test_import_skill_classes():
    from cobot.skills.grasp import GraspSkill
    from cobot.skills.place_at import PlaceAtSkill
    from cobot.skills.place_on import PlaceOnSkill
    from cobot.skills.push import PushSkill
    assert GraspSkill and PlaceOnSkill and PlaceAtSkill and PushSkill


def test_import_policy():
    import torch
    from cobot.skills.policy import ManipulationPolicy
    policy = ManipulationPolicy(obs_dim=22, action_dim=7)
    x = torch.randn(4, 22)
    y = policy(x)
    assert y.shape == (4, 7)
    assert (y >= -1.0).all() and (y <= 1.0).all()


# ---------------------------------------------------------------------------
# SkillLibrary precondition checks (no simulation)
# ---------------------------------------------------------------------------

@pytest.fixture
def skill_library():
    from cobot.skills.skill_library import SkillLibrary
    config = {"checkpoint_dir": "checkpoints/", "scripted_fallback": True}
    return SkillLibrary(config)


SCENE = {
    "objects": [
        {"id": "cubeA", "color": "red",  "shape": "cube", "pixel_u": 100, "pixel_v": 120},
        {"id": "cubeB", "color": "blue", "shape": "cube", "pixel_u": 150, "pixel_v": 130},
    ]
}


def test_grasp_precondition_passes(skill_library):
    from cobot.planner.task_planner import SkillCall
    ok, _ = skill_library.check_preconditions(
        SkillCall("grasp", {"object_id": "cubeA"}), SCENE
    )
    assert ok


def test_grasp_precondition_fails_missing_object(skill_library):
    from cobot.planner.task_planner import SkillCall
    ok, reason = skill_library.check_preconditions(
        SkillCall("grasp", {"object_id": "purple_block"}), SCENE
    )
    assert not ok
    assert "Precondition" in reason


def test_place_on_precondition_passes(skill_library):
    from cobot.planner.task_planner import SkillCall
    ok, _ = skill_library.check_preconditions(
        SkillCall("place_on", {"object_id": "cubeA", "target_id": "cubeB"}), SCENE
    )
    assert ok


def test_place_at_invalid_position(skill_library):
    from cobot.planner.task_planner import SkillCall
    ok, _ = skill_library.check_preconditions(
        SkillCall("place_at", {"object_id": "cubeA", "position": "invalid"}), SCENE
    )
    assert not ok


def test_push_precondition_invalid_direction(skill_library):
    from cobot.planner.task_planner import SkillCall
    ok, _ = skill_library.check_preconditions(
        SkillCall("push", {"object_id": "cubeA", "direction": "up"}), SCENE
    )
    assert not ok


def test_unknown_skill_returns_false(skill_library):
    from cobot.planner.task_planner import SkillCall
    env  = MagicMock()
    perc = MagicMock()
    ok, reason = skill_library.execute(
        SkillCall("teleport", {"object_id": "cubeA"}), env, perc
    )
    assert not ok
    assert "Unknown skill" in reason


def test_available_skills(skill_library):
    skills = skill_library.available_skills
    assert set(skills) == {"grasp", "place_on", "place_at", "push"}


# ---------------------------------------------------------------------------
# VoiceInterface text mode
# ---------------------------------------------------------------------------

def test_voice_text_mode(monkeypatch):
    from cobot.voice.voice_interface import VoiceInterface

    monkeypatch.setattr("builtins.input", lambda _: "stack the red block on the blue one")
    vi = VoiceInterface({"mode": "text"})
    result = vi.listen()
    assert result == "stack the red block on the blue one"


# ---------------------------------------------------------------------------
# Pose6DOF helpers
# ---------------------------------------------------------------------------

def test_pose6dof_from_pos_quat():
    from cobot.env.cobot_env import Pose6DOF

    pos  = np.array([0.1, 0.2, 0.8])
    quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity (w, x, y, z)
    pose = Pose6DOF.from_pos_quat(pos, quat)
    assert pytest.approx(pose.x, abs=1e-5) == 0.1
    assert pytest.approx(pose.y, abs=1e-5) == 0.2
    assert pytest.approx(pose.z, abs=1e-5) == 0.8
    np.testing.assert_allclose(pose.euler(), [0, 0, 0], atol=1e-5)


def test_pose6dof_to_array():
    from cobot.env.cobot_env import Pose6DOF

    pose = Pose6DOF(0.1, 0.2, 0.8, 0.0, 0.0, 0.0)
    arr  = pose.to_array()
    assert arr.shape == (6,)
    np.testing.assert_allclose(arr[:3], [0.1, 0.2, 0.8])


# ---------------------------------------------------------------------------
# Perception backprojection (pure geometry, no sim)
# ---------------------------------------------------------------------------

def test_backproject_identity():
    from cobot.perception.perception_module import PerceptionModule

    # With identity extrinsics, a pixel at the principal point and depth d
    # should project to (0, 0, d) in world frame.
    K = np.array([[256., 0., 128.], [0., 256., 128.], [0., 0., 1.]])
    E = np.eye(4)  # identity: camera frame == world frame
    p = PerceptionModule._backproject(128, 128, 0.5, K, E)
    np.testing.assert_allclose(p, [0.0, 0.0, 0.5], atol=1e-6)
