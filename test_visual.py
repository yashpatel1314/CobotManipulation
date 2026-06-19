#!/usr/bin/env python3
"""Visual test suite — runs 8 commands with screenshot capture at each step.

Usage:
    export DISPLAY=:0 && python test_visual.py 2>&1 | tee /tmp/visual_test_log.txt
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
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
# Suppress noisy 3rd-party loggers
for _n in ("robosuite_logs", "robosuite", "numba", "OpenGL", "mujoco", "httpx", "openai"):
    logging.getLogger(_n).setLevel(logging.ERROR)

log = logging.getLogger(__name__)

import yaml
import numpy as np
from cobot.orchestrator import CobotOrchestrator


def capture_screenshot(orch: "CobotOrchestrator", label: str) -> None:
    """Save the agentview camera image as a PNG for visual verification."""
    path = f"/tmp/shot_{label}.png"
    try:
        from PIL import Image
        img = orch._env.get_scene_image()  # HxWx3 uint8 numpy array
        Image.fromarray(img).save(path)
        print(f"  [screenshot] saved to {path}")
    except Exception as exc:
        print(f"  [screenshot] WARNING: could not save — {exc}")


def main() -> int:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Enable the passive viewer
    config["env"]["render"] = True

    print("\n" + "=" * 60)
    print("Initialising CobotOrchestrator (single instance)...")
    orch = CobotOrchestrator(config)

    # Commands to test: (label, command_string)
    TESTS = [
        ("stack_r_g",         "pick up the red block and place it on the green block"),
        ("push_left",         "push the green block to the left"),
        ("spawn_blue",        "spawn a blue block"),
        ("spawn_blue_on_green", "spawn blue block and put it on top of the green block"),
        ("spawn_yellow",      "add a yellow cube to the scene"),
        ("yellow_top_right",  "spawn a yellow block and put it at the top right corner"),
        ("push_forward",      "push the red block forward"),
        ("spawn_orange_left", "spawn an orange cube and move it to the left side of the table"),
    ]

    results: list[tuple[str, str, bool, dict]] = []

    for label, cmd in TESTS:
        print(f"\n{'─'*60}")
        print(f"[{label}]  {cmd}")

        # Reset env and clear perception cache before each command
        orch._env.reset()
        orch._perception.clear_cache()
        orch._env.render()

        t0 = time.monotonic()
        result = orch._execute_command(cmd, render=True)
        elapsed = time.monotonic() - t0

        # Capture screenshot after execution
        capture_screenshot(orch, label)

        ok = result["success"]
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  ({elapsed:.1f}s)")
        print(f"  Skills executed:")
        for s in result["skills"]:
            skill_status = "ok" if s["success"] else "FAIL"
            print(f"    {s['skill']}({s['args']}) → {skill_status}")
        if "error" in result:
            print(f"  ERROR: {result['error']}")

        results.append((label, cmd, ok, result))

    orch._env.close()

    # Summary
    passed = sum(1 for _, _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"VISUAL TEST RESULTS  {passed}/{total} passed")
    print(f"{'='*60}")
    for label, cmd, ok, _ in results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}]  [{label}]  {cmd[:55]}")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
