"""MultiObjectStack — Stack environment extended with up to 4 extra shaped objects.

Extra objects (blue cylinder, yellow sphere, orange cone, purple cube by default) are
pre-loaded into the sim but start off-table at z=-5 until explicitly spawned.
"""
from __future__ import annotations

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, ConeObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat

OFF_TABLE = np.array([0.0, 0.0, -5.0])
OFF_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])

_SLOT_NAMES = ["cubeC", "cubeD", "cubeE", "cubeF"]

_COLOR_RGBA: dict[str, list[float]] = {
    "blue":   [0.10, 0.40, 1.00, 1.0],
    "yellow": [1.00, 0.90, 0.00, 1.0],
    "orange": [1.00, 0.50, 0.00, 1.0],
    "purple": [0.60, 0.00, 0.80, 1.0],
}


def _make_object(sim_name: str, color: str, shape: str):
    """Create a robosuite primitive object of the given shape."""
    rgba = _COLOR_RGBA.get(color, [0.5, 0.5, 0.5, 1.0])
    if shape == "cylinder":
        return CylinderObject(
            name=sim_name,
            size=[0.020, 0.025],   # radius=2cm, half-height=2.5cm → 5cm tall
            rgba=rgba,
        )
    if shape == "sphere":
        return BallObject(
            name=sim_name,
            size=[0.025],          # radius=2.5cm → 5cm diameter
            rgba=rgba,
        )
    if shape == "cone":
        return ConeObject(
            name=sim_name,
            outer_radius=0.025,
            inner_radius=0.001,    # near-solid cone
            height=0.05,
            ngeoms=8,
            rgba=rgba,
        )
    # Default: cube
    return BoxObject(
        name=sim_name,
        size_min=[0.020, 0.020, 0.020],
        size_max=[0.020, 0.020, 0.020],
        rgba=rgba,
    )


class MultiObjectStack(Stack):
    """Stack env with configurable extra shaped objects that can be spawned at runtime."""

    def __init__(
        self,
        extra_objects: list[dict] | None = None,
        extra_colors: list[str] | None = None,
        **kwargs,
    ):
        # Backward-compat: accept old extra_colors=[str,...] as all-cube extra_objects
        if extra_objects is None:
            extra_colors = (extra_colors or [])[:4]
            extra_objects = [{"color": c, "shape": "cube"} for c in extra_colors]
        self.extra_objects: list[dict] = extra_objects[:4]
        self._extra_cubes: dict[str, object] = {}
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        ManipulationEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # cubeA (red) and cubeB (green) are always present
        self.cubeA = BoxObject(
            name="cubeA",
            size_min=[0.020, 0.020, 0.020],
            size_max=[0.020, 0.020, 0.020],
            rgba=[1.0, 0.0, 0.0, 1.0],
        )
        self.cubeB = BoxObject(
            name="cubeB",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0.0, 1.0, 0.0, 1.0],
        )

        self._extra_cubes = {}
        for i, obj_spec in enumerate(self.extra_objects):
            sim_name = _SLOT_NAMES[i]
            color = obj_spec.get("color", "blue")
            shape = obj_spec.get("shape", "cube")
            self._extra_cubes[sim_name] = _make_object(sim_name, color, shape)

        all_objects = [self.cubeA, self.cubeB] + list(self._extra_cubes.values())

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[self.cubeA, self.cubeB],
            x_range=[-0.08, 0.08],
            y_range=[-0.08, 0.08],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=all_objects,
        )

    # ------------------------------------------------------------------
    # References
    # ------------------------------------------------------------------

    def _setup_references(self) -> None:
        super()._setup_references()
        for sim_name, obj in self._extra_cubes.items():
            body_id = self.sim.model.body_name2id(obj.root_body)
            setattr(self, f"{sim_name}_body_id", body_id)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_internal(self) -> None:
        super()._reset_internal()
        for obj in self._extra_cubes.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0],
                np.concatenate([OFF_TABLE, OFF_QUAT]),
            )

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()
        modality = "object"
        for sim_name, obj in self._extra_cubes.items():
            body_id = getattr(self, f"{sim_name}_body_id")

            @sensor(modality=modality)
            def _pos(obs_cache, _bid=body_id):
                return np.array(self.sim.data.body_xpos[_bid])

            @sensor(modality=modality)
            def _quat(obs_cache, _bid=body_id):
                return convert_quat(np.array(self.sim.data.body_xquat[_bid]), to="xyzw")

            _pos.__name__  = f"{sim_name}_pos"
            _quat.__name__ = f"{sim_name}_quat"

            for s in (_pos, _quat):
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables
