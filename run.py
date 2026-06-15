"""Entry point for interactive and replay execution modes."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CobotManipulation")
    parser.add_argument(
        "--mode",
        choices=["interactive", "benchmark", "replay"],
        default="interactive",
        help="Execution mode (default: interactive)",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Use microphone input instead of keyboard",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Episode log directory for replay mode",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.mode == "replay":
        if args.log_dir is None:
            parser.error("--log-dir is required for replay mode")
        config["env"]["render"] = True

    if args.mode == "interactive":
        config["env"]["render"] = True

    from cobot.orchestrator import CobotOrchestrator

    orchestrator = CobotOrchestrator(config)

    # robosuite resets its own logger level during import; silence it now
    for _noisy in ("robosuite_logs", "robosuite", "numba", "OpenGL"):
        logging.getLogger(_noisy).setLevel(logging.ERROR)

    if args.mode == "interactive":
        orchestrator.run_interactive(voice=args.voice)

    elif args.mode == "benchmark":
        from evaluate import BENCHMARK_TASKS
        results = orchestrator.run_benchmark(BENCHMARK_TASKS)
        print(f"\nSuccess rate : {results['success_rate']:.1%}")
        print(f"Replan rate  : {results['replan_rate']:.2f} per task")

    elif args.mode == "replay":
        orchestrator.run_replay(args.log_dir)


if __name__ == "__main__":
    main()
