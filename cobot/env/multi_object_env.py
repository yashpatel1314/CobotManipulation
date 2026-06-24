"""MultiObjectStack — Stack environment with shaped extra objects, distractors, and domain randomisation.

Extra objects (blue cylinder, yellow sphere, orange cone, purple cube by default) start
off-table at z=-5 until spawned.  Distractor cubes are always on the table as obstacles.
Domain randomisation optionally perturbs initial positions and friction each reset.
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

# Distractor: grey cube, visually distinct from task objects
_DISTRACTOR_RGBA = [0.55, 0.55, 0.55, 1.0]


def _make_object(sim_name: str, color: str, shape: str):
    rgba = _COLOR_RGBA.get(color, [0.5, 0.5, 0.5, 1.0])
    if shape == "cylinder":
        return CylinderObject(name=sim_name, size=[0.020, 0.025], rgba=rgba)
    if shape == "sphere":
        return BallObject(name=sim_name, size=[0.025], rgba=rgba)
    if shape == "cone":
        return ConeObject(name=sim_name, outer_radius=0.025, inner_radius=0.001,
                          height=0.05, ngeoms=8, rgba=rgba)
    return BoxObject(name=sim_name, size_min=[0.020, 0.020, 0.020],
                     size_max=[0.020, 0.020, 0.020], rgba=rgba)


class MultiObjectStack(Stack):
    """Stack env with configurable extra shaped objects, distractors, and domain randomisation."""

    def __init__(
        self,
        extra_objects: list[dict] | None = None,
        extra_colors: list[str] | None = None,
        distractor_count: int = 0,
        domain_randomization: bool = False,
        randomization_config: dict | None = None,
        **kwargs,
    ):
        if extra_objects is None:
            extra_colors = (extra_colors or [])[:4]
            extra_objects = [{"color": c, "shape": "cube"} for c in extra_colors]
        self.extra_objects: list[dict]  = extra_objects[:4]
        self._extra_cubes: dict[str, object] = {}

        self._distractor_count: int = distractor_count
        self._distractors: list[BoxObject] = []

        self._domain_rand: bool = domain_randomization
        self._rand_cfg: dict = randomization_config or {}

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

        # Extra spawnable objects
        self._extra_cubes = {}
        for i, obj_spec in enumerate(self.extra_objects):
            sim_name = _SLOT_NAMES[i]
            color = obj_spec.get("color", "blue")
            shape = obj_spec.get("shape", "cube")
            self._extra_cubes[sim_name] = _make_object(sim_name, color, shape)

        # Distractor obstacles (grey cubes, always on table)
        self._distractors = []
        for i in range(self._distractor_count):
            d = BoxObject(
                name=f"distractor{i}",
                size_min=[0.022, 0.022, 0.022],
                size_max=[0.022, 0.022, 0.022],
                rgba=_DISTRACTOR_RGBA,
            )
            self._distractors.append(d)

        all_objects = (
            [self.cubeA, self.cubeB]
            + list(self._extra_cubes.values())
            + self._distractors
        )

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[self.cubeA, self.cubeB],
            x_range=[-0.10, 0.10],   # wider range for domain diversity
            y_range=[-0.12, 0.12],
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
        # Distractor body IDs
        self._distractor_body_ids: list[int] = []
        for d in self._distractors:
            bid = self.sim.model.body_name2id(d.root_body)
            self._distractor_body_ids.append(bid)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_internal(self) -> None:
        super()._reset_internal()

        # Extra objects: start off-table
        for obj in self._extra_cubes.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([OFF_TABLE, OFF_QUAT])
            )

        # Distractors: place at random positions on table, away from workspace centre
        table_top_z = self.table_offset[2] + 0.025 + 0.022 + 0.005
        rng = self.rng
        # Place distractors in the corners of the workspace to maximise obstruction
        distractor_zones = [
            (rng.uniform(-0.14, -0.08), rng.uniform(-0.22, -0.14)),  # near-right corner
            (rng.uniform( 0.08,  0.14), rng.uniform( 0.14,  0.22)),  # far-left corner
            (rng.uniform(-0.14, -0.08), rng.uniform( 0.14,  0.22)),  # near-left corner
            (rng.uniform( 0.08,  0.14), rng.uniform(-0.22, -0.14)),  # far-right corner
        ]
        for i, d in enumerate(self._distractors):
            x, y = distractor_zones[i % len(distractor_zones)]
            self.sim.data.set_joint_qpos(
                d.joints[0],
                np.array([x, y, table_top_z, 1.0, 0.0, 0.0, 0.0]),
            )

        # Domain randomisation: add position noise to red/green cubes
        if self._domain_rand:
            noise_std = self._rand_cfg.get("position_noise", 0.0)
            if noise_std > 0:
                for cube in [self.cubeA, self.cubeB]:
                    qpos = self.sim.data.get_joint_qpos(cube.joints[0]).copy()
                    qpos[0] += rng.uniform(-noise_std, noise_std)
                    qpos[1] += rng.uniform(-noise_std, noise_std)
                    self.sim.data.set_joint_qpos(cube.joints[0], qpos)

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
