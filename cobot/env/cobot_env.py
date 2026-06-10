from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Pose6DOF:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def euler(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])

    @classmethod
    def from_pos_quat(cls, pos: np.ndarray, quat_wxyz: np.ndarray) -> "Pose6DOF":
        # MuJoCo uses (w, x, y, z); scipy expects (x, y, z, w)
        q = quat_wxyz[[1, 2, 3, 0]]
        roll, pitch, yaw = Rotation.from_quat(q).as_euler("xyz")
        return cls(pos[0], pos[1], pos[2], roll, pitch, yaw)


class CobotEnv:
    """Thin wrapper around robosuite Stack that exposes a clean interface.

    The Stack environment provides two coloured cubes on a tabletop and a
    Franka Panda arm.  All robosuite-specific details are contained here so
    the rest of the codebase never imports robosuite directly.
    """

    # Object IDs present in the Stack scene
    OBJECT_IDS = ("cubeA", "cubeB")

    def __init__(self, config: dict) -> None:
        import robosuite as suite
        from robosuite import load_composite_controller_config

        controller_config = load_composite_controller_config(
            controller=config.get("controller", "BASIC"),
            robot=config.get("robot", "Panda"),
        )

        camera_names = config.get("cameras", ["agentview", "robot0_eye_in_hand"])
        h = config.get("camera_height", 256)
        w = config.get("camera_width", 256)

        self._env = suite.make(
            "Stack",
            robots=config.get("robot", "Panda"),
            has_renderer=config.get("render", False),
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_heights=h,
            camera_widths=w,
            camera_depths=True,
            controller_configs=controller_config,
            control_freq=config.get("control_freq", 20),
            horizon=config.get("horizon", 500),
            reward_shaping=True,
        )

        self.config = config
        self._obs: dict[str, Any] = {}
        self._camera_height = h
        self._camera_width = w

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        self._obs = self._env.reset()
        return self._obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        self._obs, reward, done, info = self._env.step(action)
        return self._obs, reward, done, info

    def render(self) -> None:
        self._env.render()

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------
    # Camera observations
    # ------------------------------------------------------------------

    def get_scene_image(self) -> np.ndarray:
        """RGB uint8 (H, W, 3) from the overhead camera."""
        # MuJoCo stores images bottom-up; flip to standard top-down
        return self._obs["agentview_image"][::-1].copy()

    def get_depth_image(self) -> np.ndarray:
        """Linearised depth in metres (H, W) from the overhead camera."""
        import robosuite.utils.camera_utils as cu

        depth_raw = self._obs["agentview_depth"][::-1, :, 0]
        return cu.get_real_depth_map(self._env.sim, depth_raw)

    def get_wrist_image(self) -> np.ndarray:
        """RGB uint8 (H, W, 3) from the wrist camera."""
        return self._obs["robot0_eye_in_hand_image"][::-1].copy()

    # ------------------------------------------------------------------
    # Camera geometry
    # ------------------------------------------------------------------

    def get_camera_intrinsics(self, camera_name: str = "agentview") -> np.ndarray:
        """3×3 intrinsic matrix K for the named camera."""
        import robosuite.utils.camera_utils as cu

        return cu.get_camera_intrinsic_matrix(
            self._env.sim, camera_name, self._camera_height, self._camera_width
        )

    def get_camera_extrinsics(self, camera_name: str = "agentview") -> np.ndarray:
        """4×4 world-to-camera extrinsic matrix for the named camera."""
        import robosuite.utils.camera_utils as cu

        return cu.get_camera_extrinsic_matrix(self._env.sim, camera_name)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_robot_state(self) -> dict[str, np.ndarray]:
        return {
            "joint_pos": self._obs["robot0_joint_pos"],
            "joint_vel": self._obs["robot0_joint_vel"],
            "ee_pos": self._obs["robot0_eef_pos"],
            "ee_quat": self._obs["robot0_eef_quat"],
            "gripper_qpos": self._obs["robot0_gripper_qpos"],
        }

    def get_object_states(self) -> dict[str, Pose6DOF]:
        """Ground-truth object poses from the simulator.

        This is only used during training and evaluation.  The perception
        module uses camera observations and should never call this method
        during a live run.
        """
        states: dict[str, Pose6DOF] = {}
        for obj_id in self.OBJECT_IDS:
            pos = self._obs[f"{obj_id}_pos"]
            quat = self._obs[f"{obj_id}_quat"]  # (w, x, y, z)
            states[obj_id] = Pose6DOF.from_pos_quat(pos, quat)
        return states

    def get_flat_obs(self) -> np.ndarray:
        """Flat state vector used as policy input."""
        rs = self.get_robot_state()
        return np.concatenate(
            [rs["ee_pos"], rs["ee_quat"], rs["joint_pos"], rs["gripper_qpos"]]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        low, _ = self._env.action_spec
        return low.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.get_flat_obs().shape[0]

    @property
    def sim(self):
        return self._env.sim
