#!/usr/bin/env python3
"""Viewer sync diagnostic — counts how many times sync fires during one skill execution."""
import os, time, logging
os.environ["DISPLAY"] = ":0"
from dotenv import load_dotenv; load_dotenv()
logging.basicConfig(level=logging.WARNING)
import yaml
from cobot.orchestrator import CobotOrchestrator

for n in ("robosuite_logs","robosuite","numba","OpenGL","mujoco","httpx","cobot"):
    logging.getLogger(n).setLevel(logging.ERROR)

with open("config.yaml") as f:
    config = yaml.safe_load(f)
config["env"]["render"] = True  # render ON

orch = CobotOrchestrator(config)
orch._env.reset()
orch._env.render()  # create viewer

print("viewer created:", orch._env._viewer is not None)
if orch._env._viewer is not None:
    print("viewer running:", orch._env._viewer.is_running())

# Patch viewer.sync directly to count calls and measure timing
sync_count = [0]
sync_times = []

if orch._env._viewer is not None:
    orig_sync = orch._env._viewer.sync
    def counted_sync(*args, **kwargs):
        sync_count[0] += 1
        t0 = time.monotonic()
        orig_sync(*args, **kwargs)
        dt = time.monotonic() - t0
        sync_times.append(dt * 1000)
        if sync_count[0] <= 3:
            print(f"  [sync #{sync_count[0]}] took {dt*1000:.1f}ms")
    orch._env._viewer.sync = counted_sync

orch._perception.clear_cache()
t_start = time.monotonic()
result = orch._execute_command("push the red block to the left", render=True)
t_total = time.monotonic() - t_start

print(f"\nsyncs fired: {sync_count[0]}")
print(f"total wall time: {t_total:.2f}s")
if sync_times:
    print(f"avg sync time: {sum(sync_times)/len(sync_times):.1f}ms")
    print(f"max sync time: {max(sync_times):.1f}ms")
print(f"success: {result['success']}")
orch._env.close()
