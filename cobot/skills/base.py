from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from cobot.env.cobot_env import CobotEnv
    from cobot.perception.perception_module import PerceptionModule


class Skill(ABC):
    """Abstract base class for manipulation primitives.

    Each concrete skill owns a trained policy (MLP) loaded from a checkpoint
    and a scripted fallback controller used when no checkpoint exists.
    """

    # Sub-classes set this to their checkpoint sub-directory name
    NAME: str = ""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._policy: "ManipulationPolicy | None" = None
        self._scripted_fallback: bool = config.get("scripted_fallback", True)

        ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        ckpt_path = ckpt_dir / self.NAME / "policy.pt"
        if ckpt_path.exists():
            self._load_policy(ckpt_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(
        self,
        env: "CobotEnv",
        perception: "PerceptionModule",
        **kwargs: Any,
    ) -> bool:
        """Execute the skill.  Returns True on success, False on failure."""

    @abstractmethod
    def is_precondition_met(self, scene: dict, **kwargs: Any) -> bool:
        """Return True if the skill's preconditions are satisfied."""

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _load_policy(self, path: Path) -> None:
        import torch
        from cobot.skills.policy import ManipulationPolicy

        obs_dim = 22   # ee_pos(3) + ee_quat(4) + joint_pos(7) + target_pos(3) + target_quat(4) + gripper(1)
        act_dim = 7    # delta ee_pose (6) + gripper (1)
        self._policy = ManipulationPolicy(obs_dim, act_dim)
        state = torch.load(path, map_location="cpu")
        self._policy.load_state_dict(state)
        self._policy.eval()

    def _run_policy(
        self,
        env: "CobotEnv",
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_steps: int = 150,
    ) -> bool:
        import torch

        assert self._policy is not None

        for _ in range(max_steps):
            rs = env.get_robot_state()
            obs_vec = np.concatenate(
                [rs["ee_pos"], rs["ee_quat"], rs["joint_pos"], target_pos, target_quat, rs["gripper_qpos"][:1]]
            ).astype(np.float32)

            with torch.no_grad():
                action = self._policy(torch.from_numpy(obs_vec).unsqueeze(0)).squeeze(0).numpy()

            _, reward, done, _ = env.step(action)
            if done:
                return reward > 0.0

        return False

    # ------------------------------------------------------------------
    # Scripted helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _move_to_target(
        env: "CobotEnv",
        target_pos: np.ndarray,
        tolerance: float = 0.02,
        max_steps: int = 100,
        gripper_cmd: float = -1.0,
    ) -> bool:
        """PD controller that moves the end-effector to a target position."""
        kp = 10.0
        for _ in range(max_steps):
            ee_pos = env.get_robot_state()["ee_pos"]
            delta = target_pos - ee_pos
            if np.linalg.norm(delta) < tolerance:
                return True
            action = np.zeros(env.action_dim)
            action[:3] = np.clip(kp * delta, -1.0, 1.0)
            action[-1] = gripper_cmd
            env.step(action)
        return False

    @staticmethod
    def _set_gripper(
        env: "CobotEnv",
        gripper_cmd: float,
        steps: int = 20,
    ) -> None:
        action = np.zeros(env.action_dim)
        action[-1] = gripper_cmd
        for _ in range(steps):
            env.step(action)
