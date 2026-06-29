#!/usr/bin/env python3
"""Headless verification suite — runs diverse commands and reports pass/fail.

Usage:
    python test_commands.py

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

import yaml
from cobot.orchestrator import CobotOrchestrator

# ── Commands to test ─────────────────────────────────────────────────────────
# Each entry: (label, command_string)
COMMANDS = [
    # ── Existing primitives ───────────────────────────────────────────────────
    ("stack red on green",        "pick up the red block and place it on the green block"),
    ("stack green on red",        "put the green cube on top of the red cube"),
    ("place red at left",         "move the red block to the left side of the table"),
    ("place green at right",      "put the green cube on the right side of the table"),
    ("place red at center",       "put the red block in the center of the table"),
    ("push red left",             "push the red block to the left"),
    ("push green forward",        "push the green block forward"),
    ("push red right",            "slide the red block to the right"),
    # ── New skills / patterns ─────────────────────────────────────────────────
    ("rotate red clockwise",      "rotate the red cube clockwise"),
    ("rotate green ccw",          "spin the green block counterclockwise"),
    ("sort all objects",          "line up all the objects on the table from left to right"),
    ("clear table",               "clear the table"),
]


DUAL_ARM_COMMANDS = [
    ("handover red arm0→arm1", "hand the red cube to the other arm"),
]


def _run_suite(orch, commands, label_prefix=""):
    results: list[tuple[str, str, bool, dict]] = []
    for label, cmd in commands:
        full_label = f"{label_prefix}{label}" if label_prefix else label
        print(f"\n{'─'*60}")
        print(f"[{full_label}]  {cmd}")
        t0 = time.monotonic()
        orch._env.reset()
        orch._perception.clear_cache()
        result = orch._execute_command(cmd, render=False)
        elapsed = time.monotonic() - t0
        ok = result["success"]
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  ({elapsed:.1f}s)  skills={[s['skill'] for s in result['skills']]}")
        if not ok:
            for s in result["skills"]:
                if not s["success"]:
                    print(f"    failed skill: {s['skill']}({s['args']})")
            if "error" in result:
                print(f"    error: {result['error']}")
        results.append((full_label, cmd, ok, result))
    return results


def main() -> int:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    config["env"]["render"] = False  # headless — no viewer

    # ── Single-arm suite ──────────────────────────────────────────────────────
    orch = CobotOrchestrator(config)
    for noisy in ("robosuite_logs", "robosuite", "numba", "OpenGL", "mujoco"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    all_results = _run_suite(orch, COMMANDS)
    orch._env.close()

    # ── Dual-arm suite ────────────────────────────────────────────────────────
    dual_config = yaml.safe_load(open("config.yaml"))
    dual_config["env"]["render"] = False
    dual_config["env"]["dual_arm"] = True
    dual_config["env"]["distractors"] = {"count": 0}
    dual_orch = CobotOrchestrator(dual_config)
    for noisy in ("robosuite_logs", "robosuite", "numba", "OpenGL", "mujoco"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    all_results += _run_suite(dual_orch, DUAL_ARM_COMMANDS, label_prefix="dual-arm ")
    dual_orch._env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, _, ok, _ in all_results if ok)
    total = len(all_results)
    print(f"\n{'='*60}")
    print(f"RESULTS  {passed}/{total} passed")
    print(f"{'='*60}")
    for label, cmd, ok, _ in all_results:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  [{label}]  {cmd[:55]}")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
