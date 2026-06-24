#!/usr/bin/env python3
"""Run the standard evaluation benchmark and save results.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --episodes 5 --output results/bench.json
"""
import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("DISPLAY", ":0")
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)
for n in ("robosuite_logs", "robosuite", "numba", "OpenGL", "mujoco", "httpx", "openai"):
    logging.getLogger(n).setLevel(logging.ERROR)

import yaml
from cobot.orchestrator import CobotOrchestrator
from cobot.evaluation.benchmark import EvaluationBenchmark, make_standard_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--episodes", type=int, default=None,
                        help="repetitions per task (overrides config)")
    parser.add_argument("--output",   default="results/benchmark.json")
    parser.add_argument("--render",   action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    config["env"]["render"] = args.render
    episodes = args.episodes or config["evaluation"].get("episodes", 1)

    orch  = CobotOrchestrator(config)
    tasks = make_standard_tasks(orch._env)

    print(f"\nRunning {len(tasks)} tasks × {episodes} episodes ...")
    bench   = EvaluationBenchmark(orch)
    summary = bench.run(tasks, episodes=episodes, render=args.render)

    EvaluationBenchmark.print_report(summary)
    EvaluationBenchmark.save_report(summary, args.output)

    orch._env.close()
    sys.exit(0 if summary["overall_success_rate"] > 0.5 else 1)


if __name__ == "__main__":
    main()
