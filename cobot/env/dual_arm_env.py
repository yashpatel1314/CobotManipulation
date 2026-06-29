"""Dual-arm environment: two Franka Pandas sharing a table.

DualArmCobotEnv mirrors the CobotEnv API but exposes both robots via:
  - active_arm (int, 0=left / 1=right): routes step() and get_robot_state()
  - step_arm(action, arm): move a specific arm without changing active_arm
  - get_robot_state(arm): arm-specific EE / joint state

All existing single-arm skills work unchanged against DualArmCobotEnv because
step() defaults to active_arm=0 and get_robot_state() does the same.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from robosuite import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from cobot.env.cobot_env import Pose6DOF, _EXTRA_SLOTS, _OFF_TABLE_Z
from cobot.env.multi_object_env import _make_object, _DISTRACTOR_RGBA, OFF_TABLE, OFF_QUAT


# ---------------------------------------------------------------------------
# Inner robosuite env — ManipulationEnv with two Pandas (no TwoArmLift)
# ---------------------------------------------------------------------------

_TABLE_FULL_SIZE = (0.8, 0.8, 0.05)
_TABLE_OFFSET    = (0, 0, 0.82)


class _DualArmStack(ManipulationEnv):
    """ManipulationEnv with two Pandas positioned identically to single-arm mode,
    offset laterally so both can reach the shared table without collision."""

    def __init__(self, extra_objects=None, distractor_count=0, **kwargs):
        self._extra_objects_spec: list[dict] = extra_objects or []
        self._distractor_count: int = distractor_count
        self._extra_cubes: dict[str, Any] = {}
        self._distractors: list[Any] = []
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()

        # Both arms on same side, offset in y — same orientation as single-arm.
        # Use robot0's base_xpos_offset for both so action axes match single-arm.
        base_xpos = np.array(
            self.robots[0].robot_model.base_xpos_offset["table"](_TABLE_FULL_SIZE[0])
        )
        for robot, y_off in zip(self.robots, (-0.15, 0.15)):
            robot.robot_model.set_base_xpos(base_xpos + np.array([0, y_off, 0]))
            robot.robot_model.set_base_ori(np.zeros(3))

        arena = TableArena(
            table_full_size=_TABLE_FULL_SIZE,
            table_friction=(1, 0.005, 0.0001),
            table_offset=_TABLE_OFFSET,
        )
        arena.set_origin([0, 0, 0])

        self.cubeA = BoxObject(name="cubeA", size_min=[0.02]*3, size_max=[0.02]*3, rgba=[1,0,0,1])
        self.cubeB = BoxObject(name="cubeB", size_min=[0.025]*3, size_max=[0.025]*3, rgba=[0,1,0,1])

        self._extra_cubes = {}
        for i, spec in enumerate(self._extra_objects_spec[:4]):
            slot = _EXTRA_SLOTS[i]
            self._extra_cubes[slot] = _make_object(slot, spec["color"], spec.get("shape", "cube"))

        self._distractors = []
        for i in range(self._distractor_count):
            self._distractors.append(BoxObject(
                name=f"distractor{i}",
                size_min=[0.022]*3, size_max=[0.022]*3,
                rgba=_DISTRACTOR_RGBA,
            ))

        all_objects = (
            [self.cubeA, self.cubeB]
            + list(self._extra_cubes.values())
            + self._distractors
        )
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[self.cubeA, self.cubeB],
            x_range=[-0.10, 0.10],
            y_range=[-0.12, 0.12],
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=_TABLE_OFFSET,
            z_offset=0.01,
            rng=self.rng,
        )
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=all_objects,
        )

    def _setup_references(self):
        ManipulationEnv._setup_references(self)
        self.cubeA_body_id = self.sim.model.body_name2id("cubeA_main")
        self.cubeB_body_id = self.sim.model.body_name2id("cubeB_main")
        for slot, obj in self._extra_cubes.items():
            setattr(self, f"{slot}_body_id", self.sim.model.body_name2id(obj.root_body))
        self._distractor_body_ids: list[int] = [
            self.sim.model.body_name2id(d.root_body) for d in self._distractors
        ]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            object_placements = self.placement_initializer.sample()
            for obj_pos, obj_quat, obj in object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )
        for obj in self._extra_cubes.values():
            self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([OFF_TABLE, OFF_QUAT]))
        z = _TABLE_OFFSET[2] + 0.025 + 0.022 + 0.005
        corners = [
            (self.rng.uniform(-0.14, -0.08), self.rng.uniform(-0.22, -0.14)),
            (self.rng.uniform( 0.08,  0.14), self.rng.uniform( 0.14,  0.22)),
        ]
        for i, d in enumerate(self._distractors):
            x, y = corners[i % len(corners)]
            self.sim.data.set_joint_qpos(d.joints[0], np.array([x, y, z, 1, 0, 0, 0]))

    def _setup_observables(self):
        return ManipulationEnv._setup_observables(self)

    def reward(self, action=None): return 0.0
    def _check_success(self): return False


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class DualArmCobotEnv:
    """Two-arm version of CobotEnv.

    Exposes the same public API as CobotEnv so all existing single-arm skills
    work unchanged.  Dual-arm capabilities are added via:

        env.active_arm = 1           # switch to right arm
        env.step_arm(action, arm)    # move a specific arm explicitly
        env.get_robot_state(arm)     # EE state for a specific arm
    """

    _VIEWER_CAM = {
        "lookat": [0, 0, 1],
        "distance": 2.5,
        "azimuth": 180,
        "elevation": -20,
    }

    def __init__(self, config: dict) -> None:
        camera_names = config.get("cameras", ["agentview", "robot0_eye_in_hand"])
        h = config.get("camera_height", 256)
        w = config.get("camera_width", 256)

        extra_objects_cfg: list[dict] = config.get("extra_objects", [])
        extra_objects_cfg = extra_objects_cfg[:4]
        dist_cfg = config.get("distractors", {})

        cc = [
            load_composite_controller_config(controller=config.get("controller", "BASIC"), robot="Panda"),
            load_composite_controller_config(controller=config.get("controller", "BASIC"), robot="Panda"),
        ]

        self._env = _DualArmStack(
            extra_objects=extra_objects_cfg,
            distractor_count=dist_cfg.get("count", 0),
            robots=["Panda", "Panda"],
            controller_configs=cc,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_heights=h,
            camera_widths=w,
            camera_depths=True,
            control_freq=config.get("control_freq", 20),
            horizon=config.get("horizon", 500),
        )

        self._color_to_sim:   dict[str, str] = {"red": "cubeA", "green": "cubeB"}
        self._color_to_shape: dict[str, str] = {"red": "cube",  "green": "cube"}
        for i, spec in enumerate(extra_objects_cfg):
            color = spec["color"]
            self._color_to_sim[color]   = _EXTRA_SLOTS[i]
            self._color_to_shape[color] = spec.get("shape", "cube")

        self._spawned_colors: set[str] = set()
        self.config = config
        self.active_arm: int = 0

        self._obs: dict[str, Any] = {}
        self._camera_height = h
        self._camera_width  = w
        self._want_render   = config.get("render", False)
        self._viewer        = None
        self._control_freq  = float(config.get("control_freq", 20))
        self._step_dt       = 1.0 / self._control_freq
        self._last_step_wall: float = 0.0

    # ------------------------------------------------------------------
    # Core interface (matches CobotEnv)
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        self._obs = self._env.reset()
        self._spawned_colors.clear()
        self.active_arm = 0
        self._last_step_wall = time.monotonic()
        return self._obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Step using active_arm (default 0 = left)."""
        return self.step_arm(action, self.active_arm)

    def step_arm(self, action: np.ndarray, arm: int) -> tuple[dict, float, bool, dict]:
        """Send a 7-dim action to one arm; zero-hold the other."""
        full = np.zeros(14)
        full[arm * 7: arm * 7 + 7] = action
        self._obs, reward, done, info = self._env.step(full)
        if self._want_render and self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            elapsed = time.monotonic() - self._last_step_wall
            remaining = self._step_dt - elapsed
            if remaining > 0.002:
                time.sleep(remaining)
        self._last_step_wall = time.monotonic()
        return self._obs, reward, done, info

    def render(self) -> None:
        if not self._want_render:
            return
        from mujoco import viewer as mj_viewer
        if self._viewer is None:
            self._viewer = mj_viewer.launch_passive(
                self._env.sim.model._model, self._env.sim.data._data,
                show_left_ui=False, show_right_ui=False,
            )
            cam = self._VIEWER_CAM
            self._viewer.cam.lookat[:] = cam["lookat"]
            self._viewer.cam.distance  = cam["distance"]
            self._viewer.cam.azimuth   = cam["azimuth"]
            self._viewer.cam.elevation = cam["elevation"]
            self._viewer.opt.geomgroup[0] = 0
        if self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._env.close()

    # ------------------------------------------------------------------
    # Robot state (arm-aware)
    # ------------------------------------------------------------------

    def get_robot_state(self, arm: int | None = None) -> dict[str, np.ndarray]:
        a = self.active_arm if arm is None else arm
        p = f"robot{a}"
        return {
            "joint_pos":    self._obs[f"{p}_joint_pos"],
            "joint_vel":    self._obs[f"{p}_joint_vel"],
            "ee_pos":       self._obs[f"{p}_eef_pos"],
            "ee_quat":      self._obs[f"{p}_eef_quat"],
            "gripper_qpos": self._obs[f"{p}_gripper_qpos"],
        }

    # ------------------------------------------------------------------
    # Object interface (same as CobotEnv)
    # ------------------------------------------------------------------

    _SHAPE_HALF_HEIGHTS: dict[str, float] = {
        "cube": 0.020, "cylinder": 0.025, "sphere": 0.025, "cone": 0.020,
    }

    def object_id(self, color: str) -> str:
        shape = self._color_to_shape.get(color, "cube")
        return f"{color}_{shape}"

    def get_object_half_height(self, obj_id: str) -> float:
        color = obj_id.split("_")[0]
        shape = self._color_to_shape.get(color, "cube")
        return self._SHAPE_HALF_HEIGHTS.get(shape, 0.020)

    def get_catalog(self) -> dict[str, dict]:
        return {
            color: {"shape": self._color_to_shape.get(color, "cube"), "id": self.object_id(color)}
            for color in self._color_to_sim
        }

    def get_object_pos(self, obj_id: str) -> np.ndarray:
        color = obj_id.split("_")[0]
        sim_name = self._color_to_sim.get(color, obj_id)
        obs_key = f"{sim_name}_pos"
        if obs_key in self._obs:
            return np.array(self._obs[obs_key])
        try:
            body_id = self._env.sim.model.body_name2id(f"{sim_name}_main")
            return np.array(self._env.sim.data.body_xpos[body_id])
        except Exception:
            pass
        raise ValueError(f"Cannot resolve object '{obj_id}' to a sim position")

    def get_distractor_positions(self) -> list[np.ndarray]:
        return [
            np.array(self._env.sim.data.body_xpos[bid])
            for bid in getattr(self._env, "_distractor_body_ids", [])
        ]

    def spawn_object(self, color: str) -> bool:
        sim_name = self._color_to_sim.get(color)
        if sim_name in (None, "cubeA", "cubeB"):
            return False
        cube = self._env._extra_cubes.get(sim_name)
        if cube is None:
            return False
        z = _TABLE_OFFSET[2] + 0.025 + 0.020 + 0.010
        rng = np.random.default_rng()
        for _ in range(20):
            x = rng.uniform(-0.06, 0.06)
            y = rng.uniform(-0.10, 0.10)
            candidate = np.array([x, y, z])
            if all(
                np.linalg.norm(candidate[:2] - self.get_object_pos(self.object_id(c))[:2]) >= 0.08
                for c in self._color_to_sim if c != color
                if self.get_object_pos(self.object_id(c))[2] > 0.5
            ):
                break
        self._env.sim.data.set_joint_qvel(cube.joints[0], np.zeros(6))
        self._env.sim.data.set_joint_qpos(cube.joints[0], np.array([x, y, z, 1, 0, 0, 0]))
        self._env.sim.forward()
        for _ in range(5):
            self._obs, _, _, _ = self._env.step(np.zeros(14))
        self._spawned_colors.add(color)
        return True

    def get_scene_image(self) -> np.ndarray:
        try:
            self._obs, _, _, _ = self._env.step(np.zeros(14))
        except ValueError:
            pass
        return self._obs["agentview_image"][::-1].copy()

    def get_cached_image(self) -> np.ndarray:
        return self._obs["agentview_image"][::-1].copy()

    def get_depth_image(self) -> np.ndarray:
        import robosuite.utils.camera_utils as cu
        depth_raw = self._obs["agentview_depth"][::-1, :, 0]
        return cu.get_real_depth_map(self._env.sim, depth_raw)

    def get_wrist_image(self) -> np.ndarray:
        return self._obs["robot0_eye_in_hand_image"][::-1].copy()

    def get_sim_scene_description(self) -> dict:
        objects = []
        for color in self._color_to_sim:
            oid = self.object_id(color)
            shape = self._color_to_shape.get(color, "cube")
            try:
                pos = self.get_object_pos(oid)
            except ValueError:
                continue
            if pos[2] > 0.5:
                objects.append({"id": oid, "color": color, "shape": shape, "pixel_u": 128, "pixel_v": 128})
        distractors = [
            {"id": f"distractor{i}", "color": "grey", "shape": "cube",
             "pos": pos.tolist(), "pixel_u": 128, "pixel_v": 128}
            for i, pos in enumerate(self.get_distractor_positions())
        ]
        return {"objects": objects, "distractors": distractors, "catalog": self.get_catalog()}

    def get_object_states(self) -> dict[str, Pose6DOF]:
        states: dict[str, Pose6DOF] = {}
        for color, sim_name in self._color_to_sim.items():
            obs_key = f"{sim_name}_pos"
            if obs_key in self._obs:
                pos  = self._obs[obs_key]
                quat = self._obs.get(f"{sim_name}_quat", np.array([1, 0, 0, 0]))
                states[sim_name] = Pose6DOF.from_pos_quat(pos, quat)
        return states

    def get_flat_obs(self) -> np.ndarray:
        rs = self.get_robot_state()
        return np.concatenate([rs["ee_pos"], rs["ee_quat"], rs["joint_pos"], rs["gripper_qpos"]])

    def get_camera_intrinsics(self, camera_name: str = "agentview") -> np.ndarray:
        import robosuite.utils.camera_utils as cu
        return cu.get_camera_intrinsic_matrix(self._env.sim, camera_name, self._camera_height, self._camera_width)

    def get_camera_extrinsics(self, camera_name: str = "agentview") -> np.ndarray:
        import robosuite.utils.camera_utils as cu
        return cu.get_camera_extrinsic_matrix(self._env.sim, camera_name)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def obs_dim(self) -> int:
        return self.get_flat_obs().shape[0]

    @property
    def sim(self):
        return self._env.sim
