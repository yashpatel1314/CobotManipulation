"""MultiObjectStack — Stack environment extended with up to 4 extra coloured cubes.

Extra cubes (blue, yellow, orange, purple) are pre-loaded into the sim but start
off-table at z=-5 so they are invisible/out-of-the-way until explicitly spawned.
"""
from __future__ import annotations

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.environments.manipulation.stack import Stack
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat

OFF_TABLE = np.array([0.0, 0.0, -5.0])
OFF_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])

# Ordered slot names and their default colours
_SLOT_COLORS = {
    "cubeC": {"blue":   [0.10, 0.40, 1.00, 1.0]},
    "cubeD": {"yellow": [1.00, 0.90, 0.00, 1.0]},
    "cubeE": {"orange": [1.00, 0.50, 0.00, 1.0]},
    "cubeF": {"purple": [0.60, 0.00, 0.80, 1.0]},
}
# All extra rgba by color name
_COLOR_RGBA = {
    "blue":   [0.10, 0.40, 1.00, 1.0],
    "yellow": [1.00, 0.90, 0.00, 1.0],
    "orange": [1.00, 0.50, 0.00, 1.0],
    "purple": [0.60, 0.00, 0.80, 1.0],
}


class MultiObjectStack(Stack):
    """Stack env with configurable extra coloured cubes that can be spawned at runtime."""

    def __init__(self, extra_colors: list[str] | None = None, **kwargs):
        self.extra_colors: list[str] = (extra_colors or [])[:4]
        # dict built during _load_model; kept as instance attr across resets
        self._extra_cubes: dict[str, BoxObject] = {}
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Model loading — add extra cubes to the scene
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        # Set up robot models (ManipulationEnv level only, skip Stack's object setup)
        ManipulationEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # cubeA (red) and cubeB (green) — identical to Stack defaults
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

        # Extra cubes: cubeC, cubeD, cubeE, cubeF
        slot_names = list(_SLOT_COLORS.keys())  # ["cubeC","cubeD","cubeE","cubeF"]
        self._extra_cubes = {}
        for i, color in enumerate(self.extra_colors):
            sim_name = slot_names[i]
            rgba = _COLOR_RGBA.get(color, [0.5, 0.5, 0.5, 1.0])
            cube = BoxObject(
                name=sim_name,
                size_min=[0.020, 0.020, 0.020],
                size_max=[0.020, 0.020, 0.020],
                rgba=rgba,
            )
            self._extra_cubes[sim_name] = cube

        all_cubes = [self.cubeA, self.cubeB] + list(self._extra_cubes.values())

        # Only cubeA and cubeB participate in random placement
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
            mujoco_objects=all_cubes,
        )

    # ------------------------------------------------------------------
    # References — look up body IDs for extra cubes
    # ------------------------------------------------------------------

    def _setup_references(self) -> None:
        super()._setup_references()
        for sim_name, cube in self._extra_cubes.items():
            body_id = self.sim.model.body_name2id(cube.root_body)
            setattr(self, f"{sim_name}_body_id", body_id)

    # ------------------------------------------------------------------
    # Reset — put extra cubes off-table
    # ------------------------------------------------------------------

    def _reset_internal(self) -> None:
        super()._reset_internal()
        for cube in self._extra_cubes.values():
            self.sim.data.set_joint_qpos(
                cube.joints[0],
                np.concatenate([OFF_TABLE, OFF_QUAT]),
            )

    # ------------------------------------------------------------------
    # Observables — add position sensors for extra cubes
    # ------------------------------------------------------------------

    def _setup_observables(self):
        observables = super()._setup_observables()
        modality = "object"
        for sim_name, cube in self._extra_cubes.items():
            body_id = getattr(self, f"{sim_name}_body_id")

            @sensor(modality=modality)
            def _pos(obs_cache, _bid=body_id):
                return np.array(self.sim.data.body_xpos[_bid])

            @sensor(modality=modality)
            def _quat(obs_cache, _bid=body_id):
                return convert_quat(np.array(self.sim.data.body_xquat[_bid]), to="xyzw")

            _pos.__name__ = f"{sim_name}_pos"
            _quat.__name__ = f"{sim_name}_quat"

            for s in (_pos, _quat):
                observables[s.__name__] = Observable(
                    name=s.__name__,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables
