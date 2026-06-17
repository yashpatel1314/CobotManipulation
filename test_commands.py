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
    ("stack red on green",        "pick up the red block and place it on the green block"),
    ("stack green on red",        "put the green cube on top of the red cube"),
    ("place red at left",         "move the red block to the left side of the table"),
    ("place green at right",      "put the green cube on the right side of the table"),
    ("place red at center",       "put the red block in the center of the table"),
    ("push red left",             "push the red block to the left"),
    ("push green forward",        "push the green block forward"),
    ("push red right",            "slide the red block to the right"),
]


def main() -> int:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    config["env"]["render"] = False  # headless — no viewer

    orch = CobotOrchestrator(config)
    for noisy in ("robosuite_logs", "robosuite", "numba", "OpenGL", "mujoco"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    results: list[tuple[str, str, bool, dict]] = []
    for label, cmd in COMMANDS:
        print(f"\n{'─'*60}")
        print(f"[{label}]  {cmd}")
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
        results.append((label, cmd, ok, result))

    orch._env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, _, ok, _ in results if ok)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"RESULTS  {passed}/{total} passed")
    print(f"{'='*60}")
    for label, cmd, ok, _ in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark}  [{label}]  {cmd[:55]}")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
