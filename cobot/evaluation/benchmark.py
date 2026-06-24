"""Task evaluation framework.

Usage:
    from cobot.evaluation.benchmark import EvaluationBenchmark, make_standard_tasks
    bench = EvaluationBenchmark(orchestrator)
    summary = bench.run(make_standard_tasks(orchestrator._env), episodes=3)
    EvaluationBenchmark.print_report(summary)
    EvaluationBenchmark.save_report(summary, "results/benchmark.json")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv


@dataclass
class TaskSpec:
    name: str
    command: str
    setup: Optional[Callable[["CobotEnv"], None]] = None
    validate: Optional[Callable[["CobotEnv", dict], bool]] = None
    expected_skills: list[str] = field(default_factory=list)
    category: str = "general"


@dataclass
class TaskResult:
    name: str
    command: str
    category: str
    success: bool
    validated: bool
    skill_results: list[dict]
    replans: int
    elapsed: float


class EvaluationBenchmark:
    """Runs a predefined task suite and reports per-task and aggregate metrics."""

    def __init__(self, orchestrator: Any) -> None:
        self._orch = orchestrator

    def run(
        self,
        tasks: list[TaskSpec],
        episodes: int = 1,
        render: bool = False,
    ) -> dict:
        """Run each task for *episodes* repetitions and return a summary dict."""
        all_results: list[TaskResult] = []

        for ep in range(episodes):
            print(f"\n── Episode {ep + 1}/{episodes} ──────────────────────────────────")
            for task in tasks:
                self._orch._env.reset()
                self._orch._perception.clear_cache()
                if task.setup:
                    task.setup(self._orch._env)

                t0 = time.monotonic()
                raw = self._orch._execute_command(task.command, render=render)
                elapsed = time.monotonic() - t0

                validated = task.validate(self._orch._env, raw) if task.validate else raw["success"]

                all_results.append(TaskResult(
                    name=task.name,
                    command=task.command,
                    category=task.category,
                    success=raw["success"],
                    validated=bool(validated),
                    skill_results=raw.get("skills", []),
                    replans=raw.get("replans", 0),
                    elapsed=elapsed,
                ))

        return self._summarize(all_results, tasks, episodes)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize(results: list[TaskResult], tasks: list[TaskSpec], episodes: int) -> dict:
        task_stats: dict[str, dict] = {}
        for task in tasks:
            task_results = [r for r in results if r.name == task.name]
            if not task_results:
                continue
            n = len(task_results)
            task_stats[task.name] = {
                "category":    task.category,
                "command":     task.command,
                "success_rate":  sum(r.success   for r in task_results) / n,
                "validated_rate": sum(r.validated for r in task_results) / n,
                "avg_elapsed": sum(r.elapsed     for r in task_results) / n,
                "avg_replans": sum(r.replans     for r in task_results) / n,
            }

        # Per-category rollup
        categories: dict[str, list[float]] = {}
        for stats in task_stats.values():
            cat = stats["category"]
            categories.setdefault(cat, []).append(stats["success_rate"])
        category_stats = {
            cat: sum(rates) / len(rates)
            for cat, rates in categories.items()
        }

        overall = sum(r.success for r in results) / len(results) if results else 0.0
        return {
            "episodes": episodes,
            "n_tasks":  len(tasks),
            "overall_success_rate": overall,
            "category_stats": category_stats,
            "task_stats": task_stats,
            "raw_results": [vars(r) for r in results],
        }

    @staticmethod
    def print_report(summary: dict) -> None:
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"BENCHMARK  {summary['n_tasks']} tasks × {summary['episodes']} episodes")
        print(f"Overall success: {summary['overall_success_rate']:.1%}")
        print()
        print("By category:")
        for cat, rate in summary["category_stats"].items():
            print(f"  {cat:<20} {rate:.1%}")
        print(f"{'─' * 68}")
        print(f"{'Task':<28} {'Cat':<12} {'Success':>8} {'Avg(s)':>8} {'Replan':>7}")
        print(f"{'─' * 68}")
        for name, stats in summary["task_stats"].items():
            print(
                f"{name:<28} {stats['category']:<12} "
                f"{stats['success_rate']:>7.0%}  "
                f"{stats['avg_elapsed']:>7.1f}  "
                f"{stats['avg_replans']:>6.1f}"
            )
        print(f"{sep}\n")

    @staticmethod
    def save_report(summary: dict, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Report saved to {path}")


# ------------------------------------------------------------------
# Standard task suite
# ------------------------------------------------------------------

def make_standard_tasks(env: "CobotEnv") -> list[TaskSpec]:
    """Return the standard benchmark task suite for this env configuration."""

    def on_table(color: str) -> Callable:
        def _check(e, _):
            try:
                return e.get_object_pos(e.object_id(color))[2] > 0.5
            except Exception:
                return False
        return _check

    def above(held: str, target: str, margin: float = 0.01) -> Callable:
        def _check(e, _):
            try:
                hp = e.get_object_pos(e.object_id(held))[2]
                tp = e.get_object_pos(e.object_id(target))[2]
                return hp > tp + margin
            except Exception:
                return False
        return _check

    def moved_to(color: str, region: str, tol: float = 0.06) -> Callable:
        from cobot.skills.place_at import _TABLE_POSITIONS
        import numpy as np
        target_xy = _TABLE_POSITIONS[region][:2]
        def _check(e, _):
            try:
                pos = e.get_object_pos(e.object_id(color))[:2]
                return bool(np.linalg.norm(pos - target_xy) < tol)
            except Exception:
                return False
        return _check

    def pushed(color: str, axis: int, min_disp: float = 0.05) -> Callable:
        import numpy as np
        snapshots: list[np.ndarray] = []
        def _setup(e):
            snapshots.clear()
            snapshots.append(e.get_object_pos(e.object_id(color)).copy())
        def _check(e, _):
            try:
                before = snapshots[0] if snapshots else None
                if before is None:
                    return False
                after = e.get_object_pos(e.object_id(color))
                return bool(abs(after[axis] - before[axis]) > min_disp)
            except Exception:
                return False
        return _setup, _check

    def spawn_setup(color: str) -> Callable:
        def _s(e):
            # Ensure object starts off-table for fair spawn test
            e.despawn_object(color)
        return _s

    tasks = []

    # ── Category: grasp ──────────────────────────────────────────
    tasks.append(TaskSpec(
        name="grasp_red",
        command="pick up the red cube",
        category="grasp",
        expected_skills=["grasp"],
    ))
    tasks.append(TaskSpec(
        name="grasp_green",
        command="pick up the green cube",
        category="grasp",
        expected_skills=["grasp"],
    ))

    # ── Category: push ───────────────────────────────────────────
    push_setups = {}
    for color, direction, ax in [("red", "forward", 0), ("green", "left", 1)]:
        setup_fn, validate_fn = pushed(color, ax)
        tasks.append(TaskSpec(
            name=f"push_{color}_{direction}",
            command=f"push the {color} cube {direction}",
            category="push",
            setup=setup_fn,
            validate=validate_fn,
            expected_skills=["push"],
        ))

    # ── Category: place_at ───────────────────────────────────────
    for color, position in [("red", "left"), ("green", "right"), ("red", "top_right")]:
        tasks.append(TaskSpec(
            name=f"place_{color}_at_{position}",
            command=f"move the {color} cube to the {position.replace('_', ' ')} corner"
                    if "top" in position or "bottom" in position
                    else f"move the {color} cube to the {position}",
            category="place_at",
            validate=moved_to(color, position),
            expected_skills=["grasp", "place_at"],
        ))

    # ── Category: stack ──────────────────────────────────────────
    tasks.append(TaskSpec(
        name="stack_red_on_green",
        command="put the red cube on top of the green cube",
        category="stack",
        validate=above("red", "green"),
        expected_skills=["grasp", "place_on"],
    ))
    tasks.append(TaskSpec(
        name="stack_green_on_red",
        command="place the green cube on top of the red cube",
        category="stack",
        validate=above("green", "red"),
        expected_skills=["grasp", "place_on"],
    ))

    # ── Category: spawn ──────────────────────────────────────────
    catalog = env.get_catalog()
    for color in ["blue", "yellow", "orange"]:
        entry = catalog.get(color, {})
        shape = entry.get("shape", "cube")
        tasks.append(TaskSpec(
            name=f"spawn_{color}_{shape}",
            command=f"spawn a {color} {shape}",
            category="spawn",
            setup=spawn_setup(color),
            validate=on_table(color),
            expected_skills=["spawn"],
        ))

    # ── Category: spawn + place ──────────────────────────────────
    blue_id = catalog.get("blue", {}).get("id", "blue_cube")
    blue_shape = catalog.get("blue", {}).get("shape", "cube")
    tasks.append(TaskSpec(
        name="spawn_blue_place_on_red",
        command=f"spawn the blue {blue_shape} and put it on the red cube",
        category="spawn+stack",
        setup=spawn_setup("blue"),
        validate=above("blue", "red"),
        expected_skills=["spawn", "grasp", "place_on"],
    ))

    # ── Category: sort ───────────────────────────────────────────
    tasks.append(TaskSpec(
        name="sort_all_objects",
        command="line up all the objects on the table from left to right",
        category="sort",
        expected_skills=["grasp", "place_at"],
    ))

    return tasks
