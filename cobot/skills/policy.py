from __future__ import annotations

import torch
import torch.nn as nn


class ManipulationPolicy(nn.Module):
    """Shared MLP policy architecture for all manipulation primitives.

    Input:  concatenated state vector (ee_pos, ee_quat, joint_pos,
            target_pos, target_quat, gripper_state)
    Output: delta end-effector action (dx, dy, dz, dax, day, daz, gripper)
            scaled to [-1, 1] by tanh.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
