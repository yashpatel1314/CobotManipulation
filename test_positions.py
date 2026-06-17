#!/usr/bin/env python3
"""Positional placement test suite — verifies all 11 named table positions.

For each command:
  1. Reset the environment.
  2. Run _execute_command.
  3. Check skill succeeded.
  4. Check actual cube x,y is within 5 cm of the target position.

Usage:
    python test_positions.py

Exit code 0 = all passed, 1 = some failed.
"""
from __future__ import annotations

import logging
import os
import sys
import time

os.environ.setdefault("DISPLAY", ":0")

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)

import numpy as np
import yaml
from cobot.orchestrator import CobotOrchestrator
from cobot.skills.place_at import _TABLE_POSITIONS

# ── Test cases ────────────────────────────────────────────────────────────────
# (expected_position_key, natural_language_command)
POSITION_COMMANDS = [
    ("left",         "put the red block on the left"),
    ("right",        "put the red block on the right"),
    ("far_left",     "move the red block to the far left"),
    ("far_right",    "move the red block to the far right"),
    ("center",       "put the red block in the center"),
    ("top",          "move the red block to the far side"),
    ("bottom",       "put the red block near the front"),
    ("top_right",    "put the red block at the top right corner"),
    ("top_left",     "put the red block at the top left corner"),
    ("bottom_right", "put the red block at the bottom right"),
    ("bottom_left",  "put the red block at the bottom left"),
]

POSITION_TOLERANCE = 0.05  # 5 cm


def check_position(env, obj_id: str, expected_pos_key: str) -> tuple[bool, float, float, float, float]:
    """Return (ok, actual_x, actual_y, target_x, target_y)."""
    target = _TABLE_POSITIONS[expected_pos_key]
    actual = env.get_object_pos(obj_id)
    dist_xy = np.linalg.norm(actual[:2] - target[:2])
    ok = dist_xy <= POSITION_TOLERANCE
    return ok, actual[0], actual[1], target[0], target[1]


def main() -> int:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    config["env"]["render"] = False  # headless

    orch = CobotOrchestrator(config)
    for noisy in ("robosuite_logs", "robosuite", "numba", "OpenGL", "mujoco"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    results: list[tuple[str, str, bool, bool, str]] = []
    # columns: pos_key, command, skill_ok, pos_ok, note

    for pos_key, cmd in POSITION_COMMANDS:
        print(f"\n{'─'*60}")
        print(f"[{pos_key}]  {cmd}")
        t0 = time.monotonic()
        orch._env.reset()
        orch._perception.clear_cache()
        result = orch._execute_command(cmd, render=False)
        elapsed = time.monotonic() - t0

        skill_ok = result["success"]
        pos_ok = False
        note = ""

        if skill_ok:
            pos_ok, ax, ay, tx, ty = check_position(orch._env, "red_cube", pos_key)
            dist = np.linalg.norm(np.array([ax, ay]) - np.array([tx, ty]))
            note = f"actual=({ax:.3f},{ay:.3f})  target=({tx:.3f},{ty:.3f})  dist={dist:.3f}m"
            status = "PASS" if pos_ok else "POS_FAIL"
        else:
            note = f"skills={[s['skill'] for s in result['skills']]}"
            if "error" in result:
                note += f"  error={result['error']}"
            status = "SKILL_FAIL"

        overall = skill_ok and pos_ok
        print(f"  {status}  ({elapsed:.1f}s)  {note}")
        results.append((pos_key, cmd, skill_ok, pos_ok, note))

    orch._env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, _, sk, pk, _ in results if sk and pk)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS  {passed}/{total} passed")
    print(f"{'='*60}")
    for pos_key, cmd, skill_ok, pos_ok, note in results:
        overall = skill_ok and pos_ok
        mark = "✓" if overall else "✗"
        detail = "skill_fail" if not skill_ok else ("pos_fail" if not pos_ok else "ok")
        print(f"  {mark}  [{pos_key:12s}]  {cmd[:45]}  ({detail})")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
