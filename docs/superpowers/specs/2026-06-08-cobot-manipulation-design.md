# CobotManipulation — System Design

**Date:** 2026-06-08
**Status:** Approved

## Overview

CobotManipulation is a language-conditioned tabletop robotic manipulation system running entirely in simulation. A user speaks or types a natural language command; the system perceives the scene, plans a sequence of learned skill primitives, and executes them in a MuJoCo physics simulation.

The project combines three areas: manipulation + perception (A), human-robot interaction via natural language (C), and LLM-driven task and motion planning (D). It is scoped to a tabletop scene as Phase 1, with the architecture designed to upgrade to richer environments incrementally.

**Tagline:** Speak a command. Watch the arm do it.

---

## Architecture Overview

Six loosely coupled Python modules, each with a single responsibility and a clean interface:

```
[Voice/Text Input]
        ↓
[LLM Task Planner]  ←→  [Skill API (primitive descriptions)]
        ↓
[Skill Dispatcher]
        ↓  ↑ (feedback loop: re-plan on failure)
[Skill Library]  ←→  [MuJoCo/robosuite Environment]
        ↑
[Perception Module]  ←→  [VLM (GPT-4o Vision / LLaVA)]
                    ←→  [Pose Estimator (depth + segmentation)]
```

**Runtime data flow:**
1. User speaks or types: *"stack the red block on top of the blue block"*
2. Whisper (STT) converts speech → text
3. LLM Planner receives text + scene JSON → emits ordered skill calls
4. Skill Dispatcher executes each primitive in sequence, querying Perception for object poses
5. Each primitive runs its learned policy inside MuJoCo
6. On failure, Dispatcher signals Planner to re-plan (up to 2 retries)

**Key decisions:**
- All modules are importable Python classes — no microservices, no ROS. Runs on a laptop.
- The Skill API is a typed Python interface the LLM is prompted with — makes planner output deterministic and executable.
- Perception runs on demand (queried before each primitive), not as a continuous stream.

---

## Module 1: Simulation Environment

**Simulator:** MuJoCo + robosuite
**Robot:** Franka Panda
**Scene:** Multi-object tabletop built on `robosuite.environments.manipulation.lift`

**Scene contents:**
- Franka Panda arm mounted at table edge
- 4–6 colored geometric objects (blocks, cylinders) with randomized positions at episode start
- Overhead + wrist-mounted RGB-D cameras
- Table with defined workspace bounds

**CobotEnv wrapper interface:**
```python
env.reset() → obs
env.step(action) → obs, reward, done, info
env.get_scene_image() → np.ndarray       # RGB frame for VLM
env.get_depth_image() → np.ndarray       # depth frame for pose estimation
env.get_object_states() → dict           # ground-truth poses (training/eval only)
```

**Observation space:** Joint positions/velocities + end-effector pose + RGB-D frames
**Action space:** Delta end-effector pose (6-DOF) + gripper open/close

**Randomization (for robust primitive training):**
- Object positions randomized within workspace bounds each episode
- Object colors/textures randomized for visual generalization
- Lighting variation for perception robustness

---

## Module 2: Perception Pipeline

**Goal:** Given an RGB-D frame, return a scene description (for the planner) and 6-DOF object poses (for primitive execution).

### Stage 1 — VLM Semantic Grounding
- Input: RGB frame from overhead camera
- Model: GPT-4o Vision (default) or LLaVA-1.6 (local fallback, no API cost)
- Output: structured JSON scene description

```json
{
  "objects": [
    {"id": "red_block", "color": "red", "shape": "cube", "position_2d": [312, 240]},
    {"id": "blue_block", "color": "blue", "shape": "cube", "position_2d": [198, 310]}
  ]
}
```

### Stage 2 — Geometric Pose Estimation
- Input: depth frame + 2D pixel coordinates from Stage 1
- Method: back-project depth pixel → 3D point in camera frame → transform to world frame via known camera extrinsics
- Output: `(x, y, z, roll, pitch, yaw)` per object
- Fallback: ground-truth pose from `env.get_object_states()` during training only (clearly flagged in code)

**Interface:**
```python
class PerceptionModule:
    def get_scene_description(self, rgb: np.ndarray) -> dict
    def get_object_pose(self, obj_id: str, rgb: np.ndarray, depth: np.ndarray) -> Pose6DOF
```

**Cost management:** VLM calls are cached per scene state hash — re-queries only when objects have visibly moved.

---

## Module 3: LLM Task Planner

**Goal:** Translate a natural language command + scene description into an ordered, executable sequence of skill calls.

**Model:** GPT-4o (default) or Claude Sonnet — swappable via config
**Approach:** Code-as-policy style — LLM is given the Skill API and asked to produce a valid JSON skill sequence

**System prompt structure:**
```
You are a robot task planner. Given a scene description and a user command,
output a JSON list of skill calls from the available skill API.

Available skills:
- grasp(object_id: str) → bool
- place_on(object_id: str, target_id: str) → bool
- place_at(object_id: str, position: str) → bool
- push(object_id: str, direction: str) → bool
- move_to(position: str) → bool

Rules:
- Only use object_ids present in the scene description
- Output valid JSON only — no explanation
- If the command is ambiguous, ask one clarifying question
```

**Example output** for *"stack the red block on the blue one"*:
```json
[
  {"skill": "grasp", "args": {"object_id": "red_block"}},
  {"skill": "place_on", "args": {"object_id": "red_block", "target_id": "blue_block"}}
]
```

**Re-planning loop:**
- If a skill returns `False`, Dispatcher sends failure reason back to Planner
- Planner gets one re-plan attempt with failure context appended to prompt
- After 2 consecutive failures on the same task, execution halts and reports to user

**Interface:**
```python
class TaskPlanner:
    def plan(self, command: str, scene: dict) -> list[SkillCall]
    def replan(self, failed_call: SkillCall, reason: str, scene: dict) -> list[SkillCall]
```

---

## Module 4: Skill Library (Learned Primitives)

**Goal:** A library of low-level manipulation primitives, each a trained policy that executes reliably given an object pose target.

### Phase 1 Primitives

| Skill | Input | Description |
|---|---|---|
| `grasp` | object 6-DOF pose | Move to pre-grasp, close gripper, lift |
| `place_on` | object pose + target pose | Move held object above target, lower, release |
| `place_at` | named position | Move held object to workspace region, release |
| `push` | object pose + direction | Contact push without grasping |

### Training — Two Stages Per Primitive

**Stage 1 — Behavior Cloning (BC):**
- Collect ~200 demonstrations per primitive using robosuite's scripted oracle controller
- Train a small MLP policy on (observation → action) pairs
- Achieves ~70% success rate

**Stage 2 — RL Fine-tuning:**
- Initialize from BC checkpoint, fine-tune with SAC or TD3
- Reward: sparse task success + shaped distance-to-target
- Achieves ~90%+ success on seen object positions

**Policy architecture:**
```
Input: [ee_pose (7), joint_pos (7), target_pose (7), gripper_state (1)]
     → MLP (256, 256) → delta_action (7)
```
Optional: wrist camera RGB features via CNN encoder for visual policies.

**Skill interface:**
```python
class Skill:
    def execute(self, env: CobotEnv, perception: PerceptionModule, **kwargs) -> bool
    def is_precondition_met(self, scene: dict, **kwargs) -> bool
```

Skills are registered in a `SkillLibrary` dict keyed by name. Each skill owns its trained checkpoint. Each primitive has a standalone 100-episode benchmark.

---

## Module 5: Voice Interface

**Goal:** Convert spoken commands to text (or accept text directly) and feed them to the Task Planner.

**STT model:** OpenAI Whisper (local, `whisper` Python package) — no API cost, runs on CPU
**VAD:** `webrtcvad` — detects speech onset/offset, auto-stops recording when user finishes speaking

**Input modes (config flag):**
- `voice` — record from mic via VAD → transcribe via Whisper → text
- `text` — terminal input (for demos, CI, testing)

**Pipeline:**
```
mic → webrtcvad (VAD) → wav buffer → Whisper → text → TaskPlanner
```

**Interface:**
```python
class VoiceInterface:
    def listen(self, mode: Literal["voice", "text"] = "text") -> str
```

Stateless — one utterance = one task. No wake word detection or multi-turn dialogue.

---

## Module 6: Orchestrator

**Goal:** Single entry point that owns all module instances and drives the main execution loop.

**Class:** `CobotOrchestrator`

**Runtime loop:**
```python
while True:
    command = voice.listen()
    scene = perception.get_scene_description(rgb)
    plan = planner.plan(command, scene)
    for skill_call in plan:
        success = skill_library.execute(skill_call, env, perception)
        if not success:
            new_plan = planner.replan(skill_call, reason, scene)
    env.render()
```

**Execution modes (config flag):**
- `interactive` — full loop with voice/text input, real-time MuJoCo rendering
- `benchmark` — headless, runs task suite, logs success rates
- `replay` — loads saved episode and replays (for demos/recordings)

**Logging:** Every run logs to `logs/YYYY-MM-DD_HH-MM/`:
- `command.txt` — raw user input
- `plan.json` — LLM skill sequence
- `episode.mp4` — rendered video via MuJoCo offscreen renderer
- `result.json` — per-skill success/failure + total outcome

**Config:** Single `config.yaml` at project root. Controls LLM model, voice mode, robot model, randomization seeds, checkpoint paths. No hardcoded values.

---

## Testing & Evaluation

### Tier 1 — Per-Primitive Benchmarks
- 100 randomized episodes per skill, headless
- Reports: success rate, mean steps, failure mode breakdown
- `python evaluate.py --skill grasp --episodes 100`
- Targets: grasp ≥ 90%, place_on ≥ 85%, push ≥ 80%

### Tier 2 — End-to-End Task Suite
- 20 fixed tasks of increasing complexity (1-step → 3-step sequences)
- Tests full pipeline: voice → plan → execution
- Reports: task success rate, re-plan rate, average steps
- `python evaluate.py --suite full`

### Tier 3 — Planner Unit Tests
- 50 command/scene pairs with ground-truth expected plans
- Checks: valid JSON output, correct object IDs, reasonable skill ordering
- No simulation needed
- `pytest tests/test_planner.py`

### Regression Tests (CI-friendly)
- Smoke test: one episode of each primitive completes without crash
- Planner test: 10 fixed commands produce valid JSON plans
- ~30s total, no GPU, runnable in GitHub Actions

### Portfolio Deliverables
- Benchmark table in README (success rates per primitive + end-to-end)
- Demo video (`replay` mode) of a 3-step task completing successfully
- Training curves (reward vs. timesteps) for each RL fine-tuned primitive

---

## Technology Stack

| Component | Library |
|---|---|
| Physics simulation | MuJoCo + robosuite |
| RL training | Stable-Baselines3 (SAC/TD3) |
| Behavior cloning | PyTorch |
| LLM planner | OpenAI Python SDK (GPT-4o) |
| VLM perception | OpenAI Vision API / LLaVA |
| Speech-to-text | openai-whisper |
| Voice activity detection | webrtcvad |
| Config management | PyYAML |
| Logging / video | Python logging + imageio |
| Testing | pytest |

---

## Project Structure

```
CobotManipulation/
├── config.yaml
├── evaluate.py
├── train.py
├── run.py
├── cobot/
│   ├── env/
│   │   └── cobot_env.py          # CobotEnv wrapper
│   ├── perception/
│   │   └── perception_module.py  # VLM + pose estimation
│   ├── planner/
│   │   └── task_planner.py       # LLM task planner
│   ├── skills/
│   │   ├── skill_library.py      # registry + dispatcher
│   │   ├── grasp.py
│   │   ├── place_on.py
│   │   ├── place_at.py
│   │   └── push.py
│   ├── voice/
│   │   └── voice_interface.py    # Whisper + VAD
│   └── orchestrator.py           # CobotOrchestrator
├── tests/
│   ├── test_planner.py
│   └── test_smoke.py
├── logs/
├── checkpoints/
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-08-cobot-manipulation-design.md
```

---

## Phase Roadmap

| Phase | Scope |
|---|---|
| 1 (current) | Tabletop with colored blocks, 4 primitives, text input |
| 2 | Voice input, visual policy primitives, more objects |
| 3 | Kitchen/household scene, more complex multi-step tasks |
| 4 | Gesture HRI layer, imitation learning from human demos |
