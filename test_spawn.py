#!/usr/bin/env python3
"""End-to-end test: spawn extra objects via LLM then manipulate them."""
import logging, os, sys
os.environ.setdefault("DISPLAY", ":0")
from dotenv import load_dotenv; load_dotenv()
logging.basicConfig(level=logging.WARNING)
for n in ("robosuite_logs","robosuite","numba","OpenGL","mujoco","httpx","openai"):
    logging.getLogger(n).setLevel(logging.ERROR)

import yaml, numpy as np
from cobot.orchestrator import CobotOrchestrator

with open("config.yaml") as f:
    config = yaml.safe_load(f)
config["env"]["render"] = False

orch = CobotOrchestrator(config)

def pos(color):
    return orch._env.get_object_pos(f"{color}_cube").copy()

def reset():
    orch._env.reset()
    orch._perception.clear_cache()

def run(cmd, label):
    result = orch._execute_command(cmd, render=False)
    ok = result["success"]
    print(f"  {'PASS' if ok else 'FAIL'}  [{label}]")
    print(f"    cmd: {cmd}")
    for s in result["skills"]:
        print(f"    skill {s['skill']}({s['args']}) → {'ok' if s['success'] else 'FAIL'}")
    return ok

results = []

# ── 1. Baseline: red/green still work ───────────────────────────────────────
print("\n── Baseline (red/green) ─────────────────────────────────────────────")
reset()
r = run("pick up the red block and place it on the green block", "stack_r_on_g")
results.append(r)

reset()
r = run("push the green block to the left", "push_green")
results.append(r)

# ── 2. Spawn blue then move it ───────────────────────────────────────────────
print("\n── Spawn + move blue ────────────────────────────────────────────────")
reset()
print(f"  blue z before spawn: {pos('blue')[2]:.3f}  (expect ≈ -5)")
r = run("spawn a blue block", "spawn_blue")
results.append(r)
print(f"  blue z after spawn:  {pos('blue')[2]:.3f}  (expect > 0.5)")

r = run("move the blue block to the right side of the table", "move_blue_right")
results.append(r)
print(f"  blue pos after move: {np.round(pos('blue'), 3)}")

# ── 3. Spawn yellow, stack blue on yellow ────────────────────────────────────
print("\n── Spawn yellow + stack blue on yellow ──────────────────────────────")
reset()
r = run("spawn a blue block", "spawn_blue_2")
results.append(r)

r = run("add a yellow cube to the scene", "spawn_yellow")
results.append(r)
print(f"  yellow z after spawn: {pos('yellow')[2]:.3f}")

blue_before = pos("blue").copy()
yellow_before = pos("yellow").copy()
r = run("put the blue block on top of the yellow block", "stack_blue_on_yellow")
results.append(r)
blue_after = pos("blue")
yellow_after = pos("yellow")
stacked = blue_after[2] > yellow_after[2] + 0.03
print(f"  blue  before={np.round(blue_before,3)}  after={np.round(blue_after,3)}")
print(f"  yellow before={np.round(yellow_before,3)}  after={np.round(yellow_after,3)}")
print(f"  stacked: {stacked}")

# ── 4. Spawn orange, push it ─────────────────────────────────────────────────
print("\n── Spawn orange + push ──────────────────────────────────────────────")
reset()
r = run("spawn an orange block", "spawn_orange")
results.append(r)
orange_before = pos("orange").copy()
r = run("push the orange block forward", "push_orange")
results.append(r)
print(f"  orange before={np.round(orange_before,3)}  after={np.round(pos('orange'),3)}")

orch._env.close()

passed = sum(results); total = len(results)
print(f"\n{'='*60}")
print(f"SPAWN TESTS  {passed}/{total} passed")
print(f"{'='*60}")
sys.exit(0 if passed == total else 1)
