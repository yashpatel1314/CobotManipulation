# CobotManipulation

Language-conditioned tabletop robotic manipulation. Speak a command — the arm plans and executes it.

A research-grade simulation system combining VLM perception, LLM task planning, and learned manipulation primitives, running entirely in MuJoCo simulation.

---

## Architecture

```
[Voice / Text Input]
        │
        ▼
[LLM Task Planner]  ◄──  [Skill API Definitions]
        │
        ▼
[Skill Dispatcher] ──────────────────────────────┐
        │  ▲ (re-plan on failure)                 │
        ▼  │                                      ▼
[Skill Library]  ◄──►  [MuJoCo / robosuite]  ◄──► [Gymnasium Wrapper]
        ▲
        │
[Perception Module]  ◄──►  [GPT-4o Vision / LLaVA]
                    ◄──►  [Depth Pose Estimator]
```

**Runtime flow:**
1. User speaks or types: *"stack the red block on top of the blue block"*
2. Whisper STT converts speech → text
3. LLM Planner receives text + scene JSON → emits skill sequence
4. Dispatcher executes each primitive, querying Perception for object poses
5. Each skill runs its trained policy (BC + SAC fine-tuned) in MuJoCo
6. On failure, Planner re-plans with failure context (up to 2 retries)

---

## Features

- **Natural language control** — type or speak commands like *"stack the red block on the blue one"*
- **VLM scene understanding** — GPT-4o Vision (or local LLaVA-1.6) identifies objects and 2D positions from overhead camera
- **LLM task planning** — GPT-4o decomposes commands into executable JSON skill sequences with automatic re-planning
- **Learned manipulation primitives** — behavior cloning + SAC fine-tuning for `grasp`, `place_on`, `place_at`, and `push`
- **Scripted fallback controllers** — deterministic execution for testing without trained checkpoints
- **Full evaluation suite** — per-primitive benchmarks + end-to-end task success metrics
- **Logging** — every run saves command, plan, rendered video, and result JSON

---

## Demo

> *Train the primitives and add your demo video here.*

| Command | Result |
|---------|--------|
| "Stack the red block on the blue one" | `grasp(red_block)` → `place_on(red_block, blue_block)` |
| "Push the green block to the left" | `push(green_block, left)` |
| "Move the red block to the center" | `grasp(red_block)` → `place_at(red_block, center)` |

---

## Setup

### Prerequisites

- Python 3.10+
- MuJoCo 3.x (installed automatically via pip)
- OpenAI API key for GPT-4o (planner + perception), **or** local LLaVA setup

### Install

```bash
git clone https://github.com/yashpatel1314/CobotManipulation
cd CobotManipulation
pip install -e .
```

For local VLM inference (no API cost after download):
```bash
pip install -e ".[local-vlm]"
```

For development:
```bash
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Set OPENAI_API_KEY=sk-... in .env
```

Edit `config.yaml` to change models, rendering settings, camera resolution, etc.

---

## Train Primitives

Collect demonstrations and train each manipulation skill. Each command runs BC collection followed by SAC fine-tuning:

```bash
python train.py --skill grasp    --demos 200 --rl-steps 500000
python train.py --skill place_on --demos 200 --rl-steps 500000
python train.py --skill place_at --demos 200 --rl-steps 500000
python train.py --skill push     --demos 100 --rl-steps 300000
```

Checkpoints are saved to `checkpoints/<skill>/`. Training curves are logged to `logs/training/`.

---

## Run

Interactive mode with text input:
```bash
python run.py
```

Interactive mode with voice input (requires microphone):
```bash
python run.py --voice
```

Headless benchmark (no rendering):
```bash
python run.py --mode benchmark
```

Replay a saved episode:
```bash
python run.py --mode replay --log-dir logs/2026-06-08_12-00/
```

---

## Evaluate

Per-primitive benchmarks (100 randomized episodes each):
```bash
python evaluate.py --skill grasp
python evaluate.py --skill place_on
python evaluate.py --skill place_at
python evaluate.py --skill push
```

Full end-to-end task suite:
```bash
python evaluate.py --suite full
```

LLM planner unit tests (no simulation required):
```bash
pytest tests/test_planner.py -v
```

Regression smoke tests:
```bash
pytest tests/test_smoke.py -v
```

---

## Benchmark Results

> *Update after training run.*

### Primitive Success Rates (100 episodes, randomized object positions)

| Skill | Success Rate | Mean Steps | Notes |
|-------|-------------|------------|-------|
| `grasp` | — | — | BC + SAC |
| `place_on` | — | — | BC + SAC |
| `place_at` | — | — | BC + SAC |
| `push` | — | — | BC + SAC |

### End-to-End Task Suite (20 tasks)

| Task Type | Success Rate | Re-plan Rate |
|-----------|-------------|-------------|
| 1-step (single skill) | — | — |
| 2-step | — | — |
| 3-step | — | — |

---

## Tech Stack

| Component | Library / Tool |
|-----------|---------------|
| Physics simulation | [MuJoCo 3](https://mujoco.org/) + [robosuite 1.4](https://robosuite.ai/) |
| Robot model | Franka Panda |
| RL training | [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SAC) |
| Neural networks | [PyTorch 2.x](https://pytorch.org/) |
| LLM task planner | GPT-4o (OpenAI API) |
| VLM perception | GPT-4o Vision / LLaVA-1.6 |
| Speech recognition | [openai-whisper](https://github.com/openai/whisper) |
| Voice activity detection | [webrtcvad](https://github.com/wiseman/py-webrtcvad) |
| RL environments | [Gymnasium](https://gymnasium.farama.org/) |
| Config | PyYAML |
| Testing | pytest |

---

## Project Structure

```
CobotManipulation/
├── config.yaml               # All configuration
├── run.py                    # Interactive / replay entry point
├── train.py                  # Primitive training (BC + SAC)
├── evaluate.py               # Benchmarking + task suite
├── cobot/
│   ├── env/
│   │   └── cobot_env.py      # CobotEnv — robosuite wrapper
│   ├── perception/
│   │   └── perception_module.py  # VLM + depth pose estimation
│   ├── planner/
│   │   └── task_planner.py   # LLM task planner
│   ├── voice/
│   │   └── voice_interface.py    # Whisper STT + VAD
│   ├── skills/
│   │   ├── base.py           # Abstract Skill interface
│   │   ├── grasp.py
│   │   ├── place_on.py
│   │   ├── place_at.py
│   │   ├── push.py
│   │   └── skill_library.py  # Registry + dispatcher
│   └── orchestrator.py       # CobotOrchestrator — main loop
├── tests/
│   ├── test_planner.py       # LLM planner unit tests
│   └── test_smoke.py         # Regression smoke tests
├── checkpoints/              # Trained skill checkpoints
├── logs/                     # Episode logs + videos
└── docs/
    └── superpowers/specs/    # Design documentation
```

---

## Roadmap

- [x] Phase 1 — Tabletop blocks, 4 primitives, text/voice input, full eval suite
- [ ] Phase 2 — Visual policy primitives (wrist camera), richer object set, more skills
- [ ] Phase 3 — Kitchen/household scene, longer-horizon tasks
- [ ] Phase 4 — Gesture HRI layer, imitation learning from human demonstrations

---

## License

MIT
