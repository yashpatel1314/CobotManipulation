from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from cobot.env.cobot_env import CobotEnv
from cobot.perception.perception_module import PerceptionModule
from cobot.planner.task_planner import SkillCall, TaskPlanner
from cobot.skills.skill_library import SkillLibrary
from cobot.voice.voice_interface import VoiceInterface

log = logging.getLogger(__name__)


class CobotOrchestrator:
    """Ties all modules together and drives the main execution loop.

    Modes:
      interactive — full loop with voice/text input and real-time rendering
      benchmark   — headless, runs a fixed task suite and reports metrics
      replay      — loads and renders a previously saved episode log
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._env    = CobotEnv(config["env"])
        self._voice  = VoiceInterface(config["voice"])
        self._planner = TaskPlanner(config["planner"])
        self._skills  = SkillLibrary(config["skills"])

        # Perception is constructed after env so it can access camera geometry
        self._perception = PerceptionModule(config["perception"], self._env)

        self._max_replan = config["planner"].get("max_replan_attempts", 2)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_interactive(self, voice: bool = False) -> None:
        """Run the interactive command loop."""
        mode = "voice" if voice else "text"
        log.info("Starting interactive loop (mode=%s). Type 'quit' to exit.", mode)
        print(f"\nCobotManipulation ready  [input: {mode}]")
        print("─" * 50)

        self._env.reset()
        if self._config["env"].get("render", False):
            self._env.render()

        while True:
            try:
                command = self._voice.listen(mode=mode)
            except (KeyboardInterrupt, EOFError):
                break

            if not command or command.lower() in ("quit", "exit", "q"):
                break

            self._execute_command(command, render=self._config["env"].get("render", False))

        self._env.close()

    def run_benchmark(self, tasks: list[dict]) -> dict:
        """Run a fixed task suite headlessly and return aggregated metrics."""
        results = []
        for i, task in enumerate(tasks):
            log.info("Task %d/%d: %s", i + 1, len(tasks), task["command"])
            self._env.reset()
            result = self._execute_command(task["command"], render=False)
            result["expected_skills"] = task.get("expected_skills", [])
            results.append(result)

        success_rate = sum(r["success"] for r in results) / len(results)
        replan_rate  = sum(r["replans"] for r in results) / len(results)
        log.info("Suite complete — success: %.1f%%  replan: %.2f/task", success_rate * 100, replan_rate)
        return {"results": results, "success_rate": success_rate, "replan_rate": replan_rate}

    def run_replay(self, log_dir: Path) -> None:
        """Replay a saved episode."""
        plan_path = log_dir / "plan.json"
        if not plan_path.exists():
            raise FileNotFoundError(f"No plan.json found in {log_dir}")

        with open(plan_path) as f:
            plan_data = json.load(f)

        self._env.reset()
        print(f"Replaying: {plan_data.get('command', '')}")
        plan = [SkillCall(skill=s["skill"], args=s["args"]) for s in plan_data["skills"]]

        for call in plan:
            print(f"  → {call}")
            self._skills.execute(call, self._env, self._perception)
            self._env.render()
            time.sleep(0.5)

        self._env.close()

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute_command(self, command: str, render: bool = False) -> dict:
        log_dir = self._make_log_dir(command)
        result  = {"command": command, "success": False, "replans": 0, "skills": []}

        try:
            rgb = self._env.get_scene_image()
            if int(rgb.max()) < 5:
                log.warning("Scene image appears blank; taking null step to refresh obs.")
                self._env.step(np.zeros(self._env.action_dim))
                rgb = self._env.get_scene_image()
            scene = self._perception.get_scene_description(rgb)
            plan  = self._planner.plan(command, scene)
        except Exception as exc:
            log.warning("Planning failed: %s", exc)
            result["error"] = str(exc)
            self._save_result(log_dir, result)
            return result

        self._save_plan(log_dir, command, plan)
        frames: list[np.ndarray] = []
        replan_count = 0

        for call in plan:
            log.info("Executing %r", call)
            print(f"  → {call}")
            success, reason = self._skills.execute(call, self._env, self._perception)
            result["skills"].append({"skill": call.skill, "args": call.args, "success": success})

            if render:
                self._env.render()
                frames.append(self._env.get_scene_image())

            if not success:
                if replan_count >= self._max_replan:
                    log.warning("Max replans reached. Halting.")
                    break

                log.info("Replanning after failure: %s", reason)
                rgb   = self._env.get_scene_image()
                scene = self._perception.get_scene_description(rgb)
                try:
                    plan = self._planner.replan(call, reason, scene, command)
                    replan_count += 1
                    result["replans"] += 1
                except Exception as exc:
                    log.warning("Replan failed: %s", exc)
                    break

        # Success = all skill steps succeeded without hitting max replans
        result["success"] = all(s["success"] for s in result["skills"])
        outcome = "SUCCESS" if result["success"] else "FAILED"
        print(f"  [{outcome}]  replans={replan_count}")

        if frames:
            self._save_video(log_dir, frames)
        self._save_result(log_dir, result)
        return result

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _make_log_dir(self, command: str) -> Path:
        ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tag = command[:30].replace(" ", "_").replace("/", "-")
        log_dir = Path("logs") / f"{ts}_{tag}"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    @staticmethod
    def _save_plan(log_dir: Path, command: str, plan: list[SkillCall]) -> None:
        data = {
            "command": command,
            "skills": [{"skill": c.skill, "args": c.args} for c in plan],
        }
        (log_dir / "plan.json").write_text(json.dumps(data, indent=2))

    @staticmethod
    def _save_result(log_dir: Path, result: dict) -> None:
        (log_dir / "result.json").write_text(json.dumps(result, indent=2))

    @staticmethod
    def _save_video(log_dir: Path, frames: list[np.ndarray]) -> None:
        try:
            import imageio
            path = log_dir / "episode.gif"
            imageio.mimsave(str(path), frames, fps=5, loop=0)
        except Exception as exc:
            log.warning("Could not save episode frames: %s", exc)
