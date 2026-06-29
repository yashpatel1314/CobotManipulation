# CobotManipulation

Language-conditioned tabletop robotic manipulation using a Franka Panda arm in MuJoCo simulation. Type a natural language command and the system perceives the scene, plans a skill sequence, and executes it in real time.

---

## Architecture

```
 Text / Voice command
         │
         ▼
┌─────────────────────┐
│    Task Planner     │  Llama 3.3 70B (Groq) + Rule-based layer
│                     │  command → JSON skill sequence (no API call for common patterns)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Perception Module  │  Llama 4 Scout (Groq) — overhead RGB → object list + pixel coords
│                     │  VLM reference resolution for ambiguous commands
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Skill Library     │  grasp · place_on · place_at · push · rotate · spawn
│                     │  sort · clear   (composite: chains grasp + place_at internally)
│                     │  Scripted PD fallback (no checkpoint needed)
│                     │  Trained MLP policy (behaviour cloning + optional SAC RL)
└──────────┬──────────┘
           │  (re-plan on failure, up to 2 retries)
           ▼
┌─────────────────────┐
│  MuJoCo / robosuite │  Franka Panda, Stack env
│                     │  Objects: red/green cubes · blue cylinder · yellow sphere
│                     │           orange cone · purple cube · grey distractor obstacles
│                     │  Domain randomisation · Distractor obstacles
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Evaluation Framework│  Per-task and per-category success rates
│                     │  JSON + GIF episode recording
└─────────────────────┘
```

**Example commands**

```
pick up the red block and place it on the green block
push the green cube to the left
spawn the blue cylinder and put it on the red cube
rotate the red cube clockwise
sort all the objects left to right
clear the table
make a pyramid with yellow and purple at the base, green on top
```

---

## Stack

| Layer | Technology |
|---|---|
| Physics sim | [MuJoCo 3](https://mujoco.org/) via [robosuite 1.5](https://robosuite.ai/) |
| Robot | Franka Panda — BASIC composite controller (OSC_POSE) |
| Task planner | Groq API — `llama-3.3-70b-versatile` + rule-based fast path |
| Visual perception | Groq API — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Fallback policy | Scripted PD controllers (works out of the box, no training needed) |
| Trained policy | Behaviour cloning MLP → optional SAC RL fine-tune |
| Evaluation | Custom benchmark suite with per-task/per-category metrics |
| Voice input | Whisper (optional) |
| Config | PyYAML |
| Testing | pytest |

---

## Setup

### 1. Create conda environment

```bash
conda create -n cobot python=3.11 -y
conda activate cobot
pip install -e .
```

### 2. API keys

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

```
GROQ_API_KEY=gsk_...   # free at console.groq.com — no credit card required
```

### 3. Display (WSL2 / headless Linux)

```bash
export DISPLAY=:0   # WSL2 with WSLg, or configure a virtual framebuffer
```

---

## Running

```bash
conda activate cobot
export DISPLAY=:0
python run.py
```

Type a command at the prompt. The MuJoCo viewer opens automatically and shows the arm executing the plan.

```
Command> pick up the red block and place it on the green block
  → grasp(object_id='red_cube')
  → place_on(object_id='red_cube', target_id='green_cube')
  [SUCCESS]  replans=0
```

### Modes

```bash
python run.py                            # interactive text input (default)
python run.py --voice                    # microphone input via Whisper
python run.py --mode benchmark           # headless task suite (uses evaluate.py logic)
python run.py --mode replay --log-dir logs/<episode>
```

### Benchmark suite

```bash
python scripts/run_benchmark.py
python scripts/run_benchmark.py --episodes 5 --output results/bench.json
```

Runs all standard task categories (grasp, push, place_at, stack, spawn, sort) and prints per-category success rates.

---

## Project structure

```
CobotManipulation/
├── config.yaml                   # all hyperparameters and provider settings
├── run.py                        # entry point — interactive / benchmark / replay
├── train.py                      # behaviour cloning + RL training
├── evaluate.py                   # benchmark task suite (standalone)
├── scripts/
│   └── run_benchmark.py          # CLI benchmark runner
├── cobot/
│   ├── env/
│   │   ├── cobot_env.py          # CobotEnv — thin robosuite wrapper
│   │   └── multi_object_env.py   # MultiObjectStack — shapes, distractors, domain rand
│   ├── perception/
│   │   └── perception_module.py  # VLM scene description + reference resolution
│   ├── planner/
│   │   └── task_planner.py       # Rule-based + LLM planner, replan logic
│   ├── skills/
│   │   ├── base.py               # abstract Skill (scripted helpers + policy loader)
│   │   ├── grasp.py              # reach, descend, close gripper, lift
│   │   ├── place_on.py           # stack held object on a target
│   │   ├── place_at.py           # move to named table position
│   │   ├── push.py               # scripted push without grasping
│   │   ├── rotate.py             # yaw-rate rotation while holding object
│   │   ├── sort.py               # atomically sort all on-table objects L→R
│   │   ├── clear.py              # atomically move all on-table objects to edges
│   │   ├── spawn.py              # teleport off-table object onto surface
│   │   └── skill_library.py      # registry and dispatcher
│   ├── evaluation/
│   │   └── benchmark.py          # EvaluationBenchmark, TaskSpec, make_standard_tasks
│   ├── voice/
│   │   └── voice_interface.py    # Whisper STT + VAD
│   └── orchestrator.py           # perceive → plan → execute loop, episode logging
├── tests/
│   ├── test_planner.py           # planner unit tests (LLM mock + rule-based)
│   └── test_smoke.py             # import + env smoke tests
├── checkpoints/                  # trained skill checkpoints (optional)
└── logs/                         # per-episode logs: plan.json, result.json, episode.gif
```

---

## Configuration

All settings live in `config.yaml`. Key options:

```yaml
env:
  render: true          # open MuJoCo viewer window
  extra_objects:        # spawnable off-table objects
    - {color: blue,   shape: cylinder}
    - {color: yellow, shape: sphere}
    - {color: orange, shape: cone}
    - {color: purple, shape: cube}
  distractors:
    count: 2            # grey obstacle cubes always on table
  randomization:
    enabled: false      # perturb red/green start positions each reset
    position_noise: 0.03

perception:
  vlm_provider: groq    # groq | openai | local
  vlm_model: meta-llama/llama-4-scout-17b-16e-instruct

planner:
  llm_provider: groq    # groq | openai
  llm_model: llama-3.3-70b-versatile
  max_replan_attempts: 2

skills:
  scripted_fallback: true   # use scripted PD controller when no checkpoint exists

evaluation:
  episodes: 3           # repetitions per task for the benchmark suite
```

---

## Objects

| Colour | Shape | Always on table? |
|---|---|---|
| Red | Cube | Yes |
| Green | Cube | Yes |
| Blue | Cylinder | No — use `spawn` first |
| Yellow | Sphere | No — use `spawn` first |
| Orange | Cone | No — use `spawn` first |
| Purple | Cube | No — use `spawn` first |
| Grey | Cube | Yes (distractor obstacles) |

---

## Skills

| Skill | Description |
|---|---|
| `grasp(object_id)` | Reach, grip, and lift an object |
| `place_on(object_id, target_id)` | Stack held object on top of target |
| `place_at(object_id, position)` | Move to a named position on the table |
| `push(object_id, direction)` | Push without grasping |
| `rotate(object_id, direction)` | Yaw-rotate while holding (`clockwise` / `counterclockwise`) |
| `sort()` | Sort all on-table objects left-to-right alphabetically by colour |
| `clear()` | Move all on-table objects to table edges |
| `spawn(object_id)` | Teleport an off-table object onto the surface |

Named positions for `place_at`: `left`, `right`, `center`, `far_left`, `far_right`, `top`, `bottom`, `top_left`, `top_right`, `bottom_left`, `bottom_right`, `adj_left`, `adj_right`

---

## Training (optional)

The scripted fallback controllers work immediately with no training. To train an MLP policy for each skill:

```bash
python train.py --skill grasp    --demos 200 --rl-steps 500000
python train.py --skill place_on --demos 200 --rl-steps 500000
python train.py --skill rotate   --demos 100 --rl-steps 300000
```

Checkpoints are saved to `checkpoints/<skill>/policy.pt` and loaded automatically at startup when present.

---

## Tests

```bash
pytest                  # all tests
pytest tests/           # unit tests only (no sim required for planner tests)
python test_commands.py # end-to-end headless run (requires GROQ_API_KEY)
python test_spawn.py    # spawn + stack integration tests
```

---

## Roadmap

- [x] Phase 1 — Task evaluation framework, domain randomisation
- [x] Phase 2 — Diverse object shapes: cylinder, sphere, cone, cube (6 colours)
- [x] Phase 3 — Distractor obstacle cubes, shape-aware catalog system
- [x] Phase 4 — New skills: rotate, sort, clear; longer-horizon planner patterns
- [x] Phase 5 — Multi-step rule-based planner, LLM examples for all new skills
- [ ] Phase 6 — Dual-arm manipulation
