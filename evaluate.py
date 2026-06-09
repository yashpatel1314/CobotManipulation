"""Evaluation suite for primitives and end-to-end task success.

Usage:
  python evaluate.py --skill grasp              # 100-episode primitive benchmark
  python evaluate.py --suite full               # full 20-task end-to-end suite
  pytest tests/test_planner.py -v               # LLM planner unit tests (no sim)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed 20-task benchmark suite (1-step through 3-step)
# ---------------------------------------------------------------------------

BENCHMARK_TASKS = [
    # 1-step: single skill
    {"command": "pick up the red block",                         "steps": 1},
    {"command": "pick up the blue block",                        "steps": 1},
    {"command": "push the red block to the left",                "steps": 1},
    {"command": "push the blue block to the right",              "steps": 1},
    {"command": "move the red block to the center",              "steps": 1},
    {"command": "move the blue block to the left",               "steps": 1},
    # 2-step
    {"command": "pick up the red block and place it to the right", "steps": 2},
    {"command": "pick up the blue block and move it to the center","steps": 2},
    {"command": "stack the red block on the blue block",          "steps": 2},
    {"command": "stack the blue block on the red block",          "steps": 2},
    {"command": "push the red block left then pick it up",        "steps": 2},
    {"command": "grasp the blue block and put it on the far left","steps": 2},
    # 3-step
    {
        "command": "pick up the red block, place it to the right, then pick up the blue block",
        "steps": 3,
    },
    {
        "command": "stack the red block on the blue one, then move the stack to the center",
        "steps": 3,
    },
    {
        "command": "push the red block to the left, then stack the blue block on the red block",
        "steps": 3,
    },
    {
        "command": "pick up the red block, place it on the far right, then push the blue block forward",
        "steps": 3,
    },
    {
        "command": "move the blue block to the left, then stack the red block on top of the blue block",
        "steps": 3,
    },
    {
        "command": "push the red block backward, pick up the blue block, and place it on the red block",
        "steps": 3,
    },
    {
        "command": "pick up the red block, stack it on the blue block, then push the stack to the right",
        "steps": 3,
    },
    {
        "command": "place the red block in the center, then stack the blue block on top of the red block",
        "steps": 3,
    },
]


# ---------------------------------------------------------------------------
# Primitive benchmark
# ---------------------------------------------------------------------------

def run_primitive_benchmark(
    skill_name: str,
    env,
    perception,
    skill_library,
    episodes: int,
    seed: int,
) -> dict:
    from cobot.planner.task_planner import SkillCall

    np.random.seed(seed)
    successes, step_counts, failure_modes = [], [], []

    arg_map = {
        "grasp":    {"object_id": "cubeA"},
        "place_on": {"object_id": "cubeA", "target_id": "cubeB"},
        "place_at": {"object_id": "cubeA", "position": "center"},
        "push":     {"object_id": "cubeA", "direction": "left"},
    }
    call = SkillCall(skill=skill_name, args=arg_map[skill_name])

    for ep in range(episodes):
        env.reset()
        success, reason = skill_library.execute(call, env, perception)
        successes.append(int(success))
        if not success:
            failure_modes.append(reason)

        if (ep + 1) % 20 == 0:
            sr = np.mean(successes) * 100
            log.info("  %d/%d  success=%.1f%%", ep + 1, episodes, sr)

    sr = float(np.mean(successes))
    log.info("Skill '%s': success=%.1f%%  n=%d", skill_name, sr * 100, episodes)
    return {
        "skill":        skill_name,
        "episodes":     episodes,
        "success_rate": sr,
        "failure_modes": failure_modes[:10],  # keep up to 10 examples
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CobotManipulation")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--skill", choices=["grasp", "place_on", "place_at", "push"])
    group.add_argument("--suite", choices=["full"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get("evaluation", {})
    episodes = args.episodes or eval_cfg.get("episodes", 100)
    seed     = args.seed or eval_cfg.get("seed", 0)

    env_cfg = {**config["env"], "render": eval_cfg.get("render", False)}
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule
    from cobot.skills.skill_library import SkillLibrary

    env          = CobotEnv(env_cfg)
    skill_library = SkillLibrary(config["skills"])
    perception   = PerceptionModule(config["perception"], env)

    results_dir = Path("logs") / "eval"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.skill:
        result = run_primitive_benchmark(
            args.skill, env, perception, skill_library, episodes, seed
        )
        print(f"\n{'─'*40}")
        print(f"Skill     : {result['skill']}")
        print(f"Episodes  : {result['episodes']}")
        print(f"Success   : {result['success_rate']:.1%}")
        out = results_dir / f"{args.skill}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"Saved     → {out}")

    elif args.suite == "full":
        from cobot.orchestrator import CobotOrchestrator

        orch = CobotOrchestrator(config)
        suite_results = orch.run_benchmark(BENCHMARK_TASKS)

        by_steps: dict[int, list] = {}
        for r, t in zip(suite_results["results"], BENCHMARK_TASKS):
            s = t.get("steps", 1)
            by_steps.setdefault(s, []).append(r["success"])

        print(f"\n{'─'*50}")
        print(f"{'Task type':<30} {'Success':>10}")
        print(f"{'─'*50}")
        for steps in sorted(by_steps):
            sr = np.mean(by_steps[steps]) * 100
            label = f"{steps}-step tasks ({len(by_steps[steps])} tasks)"
            print(f"{label:<30} {sr:>9.1f}%")
        print(f"{'─'*50}")
        print(f"{'Overall':<30} {suite_results['success_rate']:>9.1%}")
        print(f"{'Replan rate':<30} {suite_results['replan_rate']:>9.2f} per task\n")

        out = results_dir / "suite_full.json"
        out.write_text(json.dumps(suite_results, indent=2))
        print(f"Saved → {out}")

    env.close()


if __name__ == "__main__":
    main()
