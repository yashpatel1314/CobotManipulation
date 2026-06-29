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
from cobot.env.dual_arm_env import DualArmCobotEnv
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
        env_cfg = config["env"]
        if env_cfg.get("dual_arm", False):
            self._env = DualArmCobotEnv(env_cfg)
        else:
            self._env = CobotEnv(env_cfg)
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

        render = self._config["env"].get("render", False)
        self._env.reset()
        if render:
            self._env.render()

        while True:
            try:
                command = self._voice.listen(mode=mode)
            except (KeyboardInterrupt, EOFError):
                break

            if not command or command.lower() in ("quit", "exit", "q"):
                break

            # Fresh scene for every command — avoids state-dependent failures
            self._env.reset()
            self._perception.clear_cache()
            if render:
                self._env.render()

            self._execute_command(command, render=render)

        self._env.close()

    def run_benchmark(self, tasks: list[dict]) -> dict:
        """Run a fixed task suite headlessly and return aggregated metrics."""
        results = []
        for i, task in enumerate(tasks):
            log.info("Task %d/%d: %s", i + 1, len(tasks), task["command"])
            self._env.reset()
            self._perception.clear_cache()
            result = self._execute_command(task["command"], render=False)
            result["expected_skills"] = task.get("expected_skills", [])
            results.append(result)

        success_rate = sum(r["success"] for r in results) / len(results)
        replan_rate  = sum(r["replans"] for r in results) / len(results)
        log.info("Suite complete — success: %.1f%%  replan: %.2f/task", success_rate * 100, replan_rate)
        return {"results": results, "success_rate": success_rate, "replan_rate": replan_rate}

    def run_replay(self, log_dir: Path) -> None:
        """Replay a saved episode.

        If episode.gif exists, prints its path for external viewing.
        Always re-executes the saved plan in simulation for live review.
        """
        log_dir = Path(log_dir)
        gif_path = log_dir / "episode.gif"
        plan_path = log_dir / "plan.json"
        result_path = log_dir / "result.json"

        if result_path.exists():
            with open(result_path) as f:
                meta = json.load(f)
            outcome = "SUCCESS" if meta.get("success") else "FAILED"
            print(f"Original outcome: {outcome}  replans={meta.get('replans', 0)}")

        if gif_path.exists():
            print(f"Recording: {gif_path.resolve()}")

        if not plan_path.exists():
            raise FileNotFoundError(f"No plan.json found in {log_dir}")

        with open(plan_path) as f:
            plan_data = json.load(f)

        self._env.reset()
        render = self._config["env"].get("render", False)
        if render:
            self._env.render()

        print(f"\nReplaying: {plan_data.get('command', '')}")
        plan = [SkillCall(skill=s["skill"], args=s["args"]) for s in plan_data["skills"]]
        for call in plan:
            print(f"  → {call}")
            self._skills.execute(call, self._env, self._perception)
            if render:
                self._env.render()
                time.sleep(0.3)

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

            # Resolve ambiguous references ("that one", "the cube on the left")
            # before planning — the VLM looks at the camera and rewrites the
            # command with concrete colour names.
            available = ["red", "green"] + list(self._env._spawned_colors)
            command = self._perception.resolve_command(command, rgb, available)
            result["command"] = command  # log the resolved form

            try:
                scene = self._perception.get_scene_description(rgb)
            except Exception as vlm_exc:
                log.warning("VLM scene description failed (%s); using sim ground-truth fallback.", vlm_exc)
                scene = self._env.get_sim_scene_description()
            # Always inject the sim catalog so the planner knows each colour's shape/id
            if "catalog" not in scene:
                scene["catalog"] = self._env.get_catalog()
            plan  = self._planner.plan(command, scene)
        except Exception as exc:
            log.warning("Planning failed: %s", exc)
            result["error"] = str(exc)
            self._save_result(log_dir, result)
            return result

        self._save_plan(log_dir, command, plan)
        frames: list[np.ndarray] = [self._env.get_cached_image()]  # opening frame
        replan_count = 0

        for call in plan:
            log.info("Executing %r", call)
            print(f"  → {call}")
            success, reason = self._skills.execute(call, self._env, self._perception)
            result["skills"].append({"skill": call.skill, "args": call.args, "success": success})

            frames.append(self._env.get_cached_image())  # frame after each skill
            if render:
                self._env.render()

            if not success:
                if replan_count >= self._max_replan:
                    log.warning("Max replans reached. Halting.")
                    break

                log.info("Replanning after failure: %s", reason)
                rgb   = self._env.get_scene_image()
                try:
                    scene = self._perception.get_scene_description(rgb)
                except Exception as vlm_exc:
                    log.warning("VLM replan scene failed (%s); using sim ground-truth fallback.", vlm_exc)
                    scene = self._env.get_sim_scene_description()
                if "catalog" not in scene:
                    scene["catalog"] = self._env.get_catalog()
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
