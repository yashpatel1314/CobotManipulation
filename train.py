"""Training pipeline for manipulation primitives.

Two-stage per skill:
  1. Behaviour Cloning (BC) on scripted oracle demonstrations
  2. SAC fine-tuning from the BC checkpoint

Usage:
  python train.py --skill grasp    --demos 200 --rl-steps 500000
  python train.py --skill place_on --demos 200 --rl-steps 500000
  python train.py --skill push     --demos 100 --rl-steps 300000
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VALID_SKILLS = ["grasp", "place_on", "place_at", "push"]


# ---------------------------------------------------------------------------
# Gymnasium wrapper for SB3
# ---------------------------------------------------------------------------

class SkillEnv:
    """Gymnasium-compatible wrapper around CobotEnv scoped to a single skill.

    Reward is provided by robosuite's shaping signal for the relevant task.
    """

    def __init__(self, cobot_env, skill_name: str, oracle_policy) -> None:
        import gymnasium as gym
        from gymnasium import spaces

        self.env = cobot_env
        self.skill_name = skill_name
        self.oracle = oracle_policy
        self._obs_dim = cobot_env.obs_dim
        self._act_dim = cobot_env.action_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._act_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.env.reset()
        return self.env.get_flat_obs().astype(np.float32), {}

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        obs = self.env.get_flat_obs().astype(np.float32)
        return obs, float(reward), bool(done), False, info

    def render(self):
        self.env.render()


# ---------------------------------------------------------------------------
# Oracle / scripted controllers for demonstration collection
# ---------------------------------------------------------------------------

def _get_oracle(skill_name: str, env):
    """Return a callable(obs) → action scripted oracle for the given skill."""

    def _grasp_oracle():
        states = env.get_object_states()
        obj_pos = states["cubeA"].position()
        phase = 0
        step = [0]

        def act(obs):
            nonlocal phase
            rs = env.get_robot_state()
            ee = rs["ee_pos"]
            action = np.zeros(env.action_dim)
            kp = 8.0

            targets = [
                obj_pos + np.array([0, 0, 0.10]),  # pre-grasp
                obj_pos + np.array([0, 0, 0.015]), # grasp height
                obj_pos + np.array([0, 0, 0.015]), # hold (close gripper)
                obj_pos + np.array([0, 0, 0.15]),  # lift
            ]
            gripper_cmds = [-1.0, -1.0, 1.0, 1.0]

            if phase < len(targets):
                delta = targets[phase] - ee
                if np.linalg.norm(delta) < 0.02:
                    phase += 1
                else:
                    action[:3] = np.clip(kp * delta, -1.0, 1.0)
            action[-1] = gripper_cmds[min(phase, len(gripper_cmds) - 1)]
            return action

        return act

    def _place_on_oracle():
        states = env.get_object_states()
        target_pos = states["cubeB"].position()
        phase = 0

        def act(obs):
            nonlocal phase
            rs = env.get_robot_state()
            ee = rs["ee_pos"]
            action = np.zeros(env.action_dim)
            kp = 8.0

            targets = [
                target_pos + np.array([0, 0, 0.12]),
                target_pos + np.array([0, 0, 0.05]),
            ]

            if phase < len(targets):
                delta = targets[phase] - ee
                if np.linalg.norm(delta) < 0.015:
                    phase += 1
                else:
                    action[:3] = np.clip(kp * delta, -1.0, 1.0)

            # Release once at place height, else keep closed
            action[-1] = -1.0 if phase >= len(targets) else 1.0
            return action

        return act

    oracles = {
        "grasp":    _grasp_oracle(),
        "place_on": _place_on_oracle(),
        "place_at": _grasp_oracle(),  # reuse grasp oracle as placeholder
        "push":     _grasp_oracle(),  # reuse grasp oracle as placeholder
    }
    return oracles[skill_name]


# ---------------------------------------------------------------------------
# Behaviour Cloning
# ---------------------------------------------------------------------------

def collect_demonstrations(env, oracle, n_demos: int, max_steps: int = 200) -> list[dict]:
    log.info("Collecting %d demonstrations...", n_demos)
    demos = []
    for i in range(n_demos):
        obs_list, act_list = [], []
        env.reset()
        for _ in range(max_steps):
            flat_obs = env.get_flat_obs()
            action = oracle(flat_obs)
            obs_list.append(flat_obs.copy())
            act_list.append(action.copy())
            _, _, done, _ = env.step(action)
            if done:
                break
        demos.append({
            "obs":     np.array(obs_list, dtype=np.float32),
            "actions": np.array(act_list, dtype=np.float32),
        })
        if (i + 1) % 20 == 0:
            log.info("  %d/%d demos collected", i + 1, n_demos)
    return demos


def train_bc(
    demos: list[dict],
    obs_dim: int,
    act_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    ckpt_path: Path,
) -> None:
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.tensorboard import SummaryWriter

    from cobot.skills.policy import ManipulationPolicy

    all_obs = np.concatenate([d["obs"] for d in demos])
    all_act = np.concatenate([d["actions"] for d in demos])

    dataset = TensorDataset(
        torch.from_numpy(all_obs),
        torch.from_numpy(all_act),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    policy = ManipulationPolicy(obs_dim, act_dim).to(device)
    optim  = torch.optim.Adam(policy.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=str(ckpt_path.parent / "tb_bc"))

    log.info("Training BC for %d epochs on %d transitions...", epochs, len(all_obs))
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            pred = policy(obs_b)
            loss = torch.nn.functional.mse_loss(pred, act_b)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        writer.add_scalar("bc/loss", avg_loss, epoch)
        if epoch % 10 == 0:
            log.info("  Epoch %3d/%d  loss=%.5f", epoch, epochs, avg_loss)

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), ckpt_path)
    log.info("BC checkpoint saved → %s", ckpt_path)
    writer.close()


# ---------------------------------------------------------------------------
# SAC Fine-tuning
# ---------------------------------------------------------------------------

def train_sac(
    skill_env: SkillEnv,
    bc_ckpt: Path,
    total_timesteps: int,
    lr: float,
    batch_size: int,
    seed: int,
    ckpt_path: Path,
) -> None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    log.info("SAC fine-tuning for %d timesteps...", total_timesteps)

    callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(ckpt_path.parent / "sac_checkpoints"),
        name_prefix="sac",
    )

    model = SAC(
        "MlpPolicy",
        skill_env,
        learning_rate=lr,
        batch_size=batch_size,
        seed=seed,
        verbose=1,
        tensorboard_log=str(ckpt_path.parent / "tb_sac"),
    )

    # Warm-start from BC policy weights
    if bc_ckpt.exists():
        import copy
        bc_state = torch.load(bc_ckpt, map_location="cpu")
        # SB3 actor is model.policy.actor; map BC weights to first two layers
        try:
            actor_state = model.policy.actor.state_dict()
            # Partial load: only matching keys
            for k in list(bc_state.keys()):
                sb3_key = k.replace("net.", "latent_pi.")
                if sb3_key in actor_state and actor_state[sb3_key].shape == bc_state[k].shape:
                    actor_state[sb3_key] = bc_state[k]
            model.policy.actor.load_state_dict(actor_state, strict=False)
            log.info("BC weights loaded into SAC actor.")
        except Exception as exc:
            log.warning("BC weight transfer failed (non-fatal): %s", exc)

    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=True)

    # Extract and save just the policy weights for inference
    torch.save(
        model.policy.actor.state_dict(),
        ckpt_path,
    )
    log.info("SAC policy saved → %s", ckpt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a manipulation primitive")
    parser.add_argument("--skill", required=True, choices=VALID_SKILLS)
    parser.add_argument("--demos",    type=int, default=200)
    parser.add_argument("--rl-steps", type=int, default=500_000)
    parser.add_argument("--config",   default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device_str = config["training"].get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device_str)

    # Prepare paths
    ckpt_dir = Path(config["skills"]["checkpoint_dir"]) / args.skill
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    bc_ckpt  = ckpt_dir / "bc_policy.pt"
    final_ckpt = ckpt_dir / "policy.pt"

    # Build env (headless for training)
    env_cfg = {**config["env"], "render": False}
    from cobot.env.cobot_env import CobotEnv
    env = CobotEnv(env_cfg)

    # Stage 1 — Behaviour Cloning
    if not bc_ckpt.exists():
        oracle = _get_oracle(args.skill, env)
        demos  = collect_demonstrations(env, oracle, n_demos=args.demos)
        train_bc(
            demos,
            obs_dim=env.obs_dim,
            act_dim=env.action_dim,
            epochs=config["training"]["bc_epochs"],
            batch_size=config["training"]["bc_batch_size"],
            lr=config["training"]["bc_lr"],
            device=device_str,
            ckpt_path=bc_ckpt,
        )
    else:
        log.info("BC checkpoint already exists at %s, skipping.", bc_ckpt)

    # Stage 2 — SAC fine-tuning
    oracle = _get_oracle(args.skill, env)
    skill_env = SkillEnv(env, args.skill, oracle)
    train_sac(
        skill_env,
        bc_ckpt=bc_ckpt,
        total_timesteps=args.rl_steps,
        lr=config["training"]["rl_learning_rate"],
        batch_size=config["training"]["rl_batch_size"],
        seed=seed,
        ckpt_path=final_ckpt,
    )

    env.close()
    log.info("Training complete for skill '%s'.", args.skill)


if __name__ == "__main__":
    main()
