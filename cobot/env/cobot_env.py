from __future__ import annotations

import time
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
        q = quat_wxyz[[1, 2, 3, 0]]
        roll, pitch, yaw = Rotation.from_quat(q).as_euler("xyz")
        return cls(pos[0], pos[1], pos[2], roll, pitch, yaw)


# Slots for extra spawnable cubes (filled in order as extra_colors list grows)
_EXTRA_SLOTS = ["cubeC", "cubeD", "cubeE", "cubeF"]
_OFF_TABLE_Z = -5.0   # z-height for unspawned cubes


class CobotEnv:
    """Thin wrapper around a robosuite environment that exposes a clean interface.

    Supports a dynamic set of coloured cubes: red and green are always present
    (cubeA / cubeB from Stack); additional colours are pre-loaded but off-table
    until spawn_object() teleports them onto the surface.
    """

    _VIEWER_CAM = {
        "lookat": [0, 0, 1],
        "distance": 2,
        "azimuth": 180,
        "elevation": -20,
    }

    def __init__(self, config: dict) -> None:
        from robosuite import load_composite_controller_config
        from cobot.env.multi_object_env import MultiObjectStack

        controller_config = load_composite_controller_config(
            controller=config.get("controller", "BASIC"),
            robot=config.get("robot", "Panda"),
        )

        camera_names = config.get("cameras", ["agentview", "robot0_eye_in_hand"])
        h = config.get("camera_height", 256)
        w = config.get("camera_width", 256)

        extra_colors: list[str] = config.get("extra_colors", [])[:4]

        self._env = MultiObjectStack(
            robots=config.get("robot", "Panda"),
            extra_colors=extra_colors,
            has_renderer=False,
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

        # Build dynamic color → sim-name mapping
        self._color_to_sim: dict[str, str] = {"red": "cubeA", "green": "cubeB"}
        for i, color in enumerate(extra_colors):
            self._color_to_sim[color] = _EXTRA_SLOTS[i]

        self._spawned_colors: set[str] = set()

        self.config = config
        self._obs: dict[str, Any] = {}
        self._camera_height = h
        self._camera_width = w
        self._want_render = config.get("render", False)
        self._viewer = None
        self._last_viewer_sync: float = 0.0
        # Real-time pacing: step wall-clock target when viewer is active
        self._control_freq: float = float(config.get("control_freq", 20))
        self._step_dt: float = 1.0 / self._control_freq  # 0.05 s @ 20 Hz
        self._last_step_wall: float = 0.0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        self._obs = self._env.reset()
        self._spawned_colors.clear()
        self._last_step_wall = time.monotonic()
        if self._want_render and self._viewer is not None:
            try:
                sim = self._viewer._sim()
                if sim is not None:
                    sim.load(
                        self._env.sim.model._model,
                        self._env.sim.data._data,
                        "",
                    )
            except Exception:
                self._viewer.close()
                self._viewer = None
        return self._obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        self._obs, reward, done, info = self._env.step(action)
        if self._want_render and self._viewer is not None and self._viewer.is_running():
            # Sync viewer on every step so motion is continuous
            self._viewer.sync()
            self._last_viewer_sync = time.monotonic()
            # Real-time pacing: sleep for the remainder of the step period so
            # the viewer displays each frame at natural speed instead of
            # blasting through 200 steps in under a second.
            elapsed = time.monotonic() - self._last_step_wall
            remaining = self._step_dt - elapsed
            if remaining > 0.002:  # don't bother for <2 ms leftovers
                time.sleep(remaining)
        self._last_step_wall = time.monotonic()
        return self._obs, reward, done, info

    def render(self) -> None:
        if not self._want_render:
            return
        from mujoco import viewer as mj_viewer
        if self._viewer is None:
            self._viewer = mj_viewer.launch_passive(
                self._env.sim.model._model,
                self._env.sim.data._data,
                show_left_ui=False,
                show_right_ui=False,
            )
            cam = self._VIEWER_CAM
            self._viewer.cam.lookat[:] = cam["lookat"]
            self._viewer.cam.distance = cam["distance"]
            self._viewer.cam.azimuth = cam["azimuth"]
            self._viewer.cam.elevation = cam["elevation"]
        if self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._env.close()

    # ------------------------------------------------------------------
    # Object spawning
    # ------------------------------------------------------------------

    def spawn_object(self, color: str) -> bool:
        """Teleport an off-table cube onto the table surface."""
        sim_name = self._color_to_sim.get(color)
        if sim_name in (None, "cubeA", "cubeB"):
            return False
        cube = self._env._extra_cubes.get(sim_name)
        if cube is None:
            return False

        # table top z: table_offset[2] + half_thickness + cube_half_size + margin
        z = self._env.table_offset[2] + 0.025 + 0.020 + 0.010

        # Pick a position that avoids other on-table cubes (min 8cm separation)
        rng = np.random.default_rng()
        for _ in range(20):
            x = rng.uniform(-0.06, 0.06)
            y = rng.uniform(-0.10, 0.10)
            candidate = np.array([x, y, z])
            collision = False
            for other_color, other_sim in self._color_to_sim.items():
                if other_color == color:
                    continue
                try:
                    other_pos = self.get_object_pos(f"{other_color}_cube")
                    if other_pos[2] > 0.5 and np.linalg.norm(candidate[:2] - other_pos[:2]) < 0.08:
                        collision = True
                        break
                except Exception:
                    pass
            if not collision:
                break

        # Zero velocity first so there are no physics explosions from old motion
        self._env.sim.data.set_joint_qvel(cube.joints[0], np.zeros(6))
        self._env.sim.data.set_joint_qpos(
            cube.joints[0],
            np.array([x, y, z, 1.0, 0.0, 0.0, 0.0]),
        )
        self._env.sim.forward()
        # Let the cube settle for a few steps
        for _ in range(5):
            self._obs, _, _, _ = self._env.step(np.zeros(self.action_dim))
        self._spawned_colors.add(color)
        return True

    def despawn_object(self, color: str) -> bool:
        """Return an on-table cube to the off-table position."""
        sim_name = self._color_to_sim.get(color)
        if sim_name in (None, "cubeA", "cubeB"):
            return False
        cube = self._env._extra_cubes.get(sim_name)
        if cube is None:
            return False
        self._env.sim.data.set_joint_qpos(
            cube.joints[0],
            np.array([0.0, 0.0, _OFF_TABLE_Z, 1.0, 0.0, 0.0, 0.0]),
        )
        self._env.sim.forward()
        self._spawned_colors.discard(color)
        return True

    # ------------------------------------------------------------------
    # Camera observations
    # ------------------------------------------------------------------

    def get_scene_image(self) -> np.ndarray:
        # Take a zero-action step to flush the offscreen renderer so the image
        # reflects the current physics state rather than a stale obs buffer.
        self._obs, _, _, _ = self._env.step(np.zeros(self.action_dim))
        return self._obs["agentview_image"][::-1].copy()

    def get_depth_image(self) -> np.ndarray:
        import robosuite.utils.camera_utils as cu
        depth_raw = self._obs["agentview_depth"][::-1, :, 0]
        return cu.get_real_depth_map(self._env.sim, depth_raw)

    def get_wrist_image(self) -> np.ndarray:
        return self._obs["robot0_eye_in_hand_image"][::-1].copy()

    # ------------------------------------------------------------------
    # Camera geometry
    # ------------------------------------------------------------------

    def get_camera_intrinsics(self, camera_name: str = "agentview") -> np.ndarray:
        import robosuite.utils.camera_utils as cu
        return cu.get_camera_intrinsic_matrix(
            self._env.sim, camera_name, self._camera_height, self._camera_width
        )

    def get_camera_extrinsics(self, camera_name: str = "agentview") -> np.ndarray:
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

    def get_object_pos(self, obj_id: str) -> np.ndarray:
        """Ground-truth 3-D position for any named object (e.g. 'blue_cube')."""
        color = obj_id.split("_")[0]
        sim_name = self._color_to_sim.get(color, obj_id)

        # cubeA / cubeB: available in obs dict
        obs_key = f"{sim_name}_pos"
        if obs_key in self._obs:
            return np.array(self._obs[obs_key])

        # Extra cubes: read directly from sim body positions
        try:
            body_id = self._env.sim.model.body_name2id(f"{sim_name}_main")
            return np.array(self._env.sim.data.body_xpos[body_id])
        except Exception:
            pass

        # Direct sim-name fallback
        direct_key = f"{obj_id}_pos"
        if direct_key in self._obs:
            return np.array(self._obs[direct_key])

        raise ValueError(f"Cannot resolve object '{obj_id}' to a sim position")

    def get_sim_scene_description(self) -> dict:
        """Scene description from ground-truth sim state (VLM fallback)."""
        objects = []
        for color, sim_name in self._color_to_sim.items():
            try:
                pos = self.get_object_pos(f"{color}_cube")
            except ValueError:
                continue
            if pos[2] > 0.5:   # on-table check
                objects.append({
                    "id": f"{color}_cube",
                    "color": color,
                    "shape": "cube",
                    "pixel_u": 128,
                    "pixel_v": 128,
                })
        return {"objects": objects}

    def get_object_states(self) -> dict[str, Pose6DOF]:
        states: dict[str, Pose6DOF] = {}
        for color, sim_name in self._color_to_sim.items():
            obs_key = f"{sim_name}_pos"
            if obs_key in self._obs:
                pos  = self._obs[obs_key]
                quat = self._obs.get(f"{sim_name}_quat", np.array([1,0,0,0]))
                states[sim_name] = Pose6DOF.from_pos_quat(pos, quat)
        return states

    def get_flat_obs(self) -> np.ndarray:
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
