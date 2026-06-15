# CobotManipulation

Language-conditioned tabletop robotic manipulation using a Franka Panda arm in MuJoCo simulation. Type a natural language command and the system perceives the scene, plans a skill sequence, and executes it in real time.

---

## Architecture

```
 Text / Voice command
         │
         ▼
┌─────────────────────┐
│    Task Planner     │  Llama 3.3 70B (Groq) — command → JSON skill sequence
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Perception Module  │  Llama 4 Scout (Groq) — overhead RGB → object list + pixel coords
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Skill Library     │  grasp · place_on · place_at · push
│                     │  Scripted PD fallback (no checkpoint needed)
│                     │  Trained MLP policy (behaviour cloning + optional RL)
└──────────┬──────────┘
           │  (re-plan on failure, up to 2 retries)
           ▼
┌─────────────────────┐
│  MuJoCo / robosuite │  Franka Panda, Stack env — two coloured cubes on a table
└─────────────────────┘
```

**Example commands**

```
pick up the red block and place it on the green block
push the green cube to the left
move the red one to the right side
```

---

## Stack

| Layer | Technology |
|---|---|
| Physics sim | [MuJoCo 3](https://mujoco.org/) via [robosuite 1.5](https://robosuite.ai/) |
| Robot | Franka Panda — BASIC composite controller (OSC_POSE) |
| Task planner | Groq API — `llama-3.3-70b-versatile` |
| Visual perception | Groq API — `meta-llama/llama-4-scout-17b-16e-instruct` |
| Fallback policy | Scripted PD controllers (works out of the box, no training needed) |
| Trained policy | Behaviour cloning MLP → optional SAC RL fine-tune |
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
python run.py --mode benchmark           # headless task suite
python run.py --mode replay --log-dir logs/<episode>
```

---

## Project structure

```
CobotManipulation/
├── config.yaml                   # all hyperparameters and provider settings
├── run.py                        # entry point — interactive / benchmark / replay
├── train.py                      # behaviour cloning + RL training
├── evaluate.py                   # benchmark task suite
├── cobot/
│   ├── env/
│   │   └── cobot_env.py          # CobotEnv — thin robosuite wrapper
│   ├── perception/
│   │   └── perception_module.py  # VLM scene description + depth backprojection
│   ├── planner/
│   │   └── task_planner.py       # LLM task planner + replan logic
│   ├── skills/
│   │   ├── base.py               # abstract Skill (scripted helpers + policy loader)
│   │   ├── grasp.py
│   │   ├── place_on.py
│   │   ├── place_at.py
│   │   ├── push.py
│   │   └── skill_library.py      # registry and dispatcher
│   ├── voice/
│   │   └── voice_interface.py    # Whisper STT + VAD
│   └── orchestrator.py           # perceive → plan → execute loop
├── tests/
│   ├── test_planner.py           # LLM planner unit tests
│   └── test_smoke.py             # import + env smoke tests
├── checkpoints/                  # trained skill checkpoints (optional)
└── logs/                         # episode logs and GIF replays
```

---

## Configuration

All settings live in `config.yaml`. Key options:

```yaml
env:
  render: true          # open MuJoCo viewer window

perception:
  vlm_provider: groq    # groq | openai | local
  vlm_model: meta-llama/llama-4-scout-17b-16e-instruct

planner:
  llm_provider: groq    # groq | openai
  llm_model: llama-3.3-70b-versatile
  max_replan_attempts: 2

skills:
  scripted_fallback: true   # use scripted PD controller when no checkpoint exists
```

---

## Training (optional)

The scripted fallback controllers work immediately with no training. To train an MLP policy for each skill:

```bash
python train.py
```

Checkpoints are saved to `checkpoints/<skill>/policy.pt` and loaded automatically at startup when present.

---

## Tests

```bash
pytest
```

---

## Roadmap

- [x] Phase 1 — MuJoCo tabletop, 4 primitives, Groq LLM/VLM, text + voice input, eval suite
- [ ] Phase 2 — Trained MLP policies, wrist-camera visual control, richer object set
- [ ] Phase 3 — Longer-horizon tasks, kitchen/household scene
- [ ] Phase 4 — Gesture HRI layer, imitation learning from human demos

---

## License

MIT
