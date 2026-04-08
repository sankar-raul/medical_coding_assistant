---
title: Medical Coding Assistant Environment Server
emoji: 🏥
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Medical Coding Assistant

Medical Coding Assistant is an OpenEnv environment for evaluating agents on a real clinical-operations workflow: assigning ICD-10 diagnosis codes from short chart summaries. The environment is intentionally scoped to a closed, offline code set so that grading is deterministic, reproducible, and safe for a hackathon setting.

This environment targets coding workflow automation, not diagnosis or treatment. The agent receives curated chart excerpts and must choose supported diagnosis codes, optionally add supporting secondary codes, and escalate ambiguous encounters for human review when required.

## Motivation

Medical coding is a real-world task performed by revenue-cycle and documentation teams. It has the right properties for an RL environment:

- real human workflow rather than a toy game
- typed actions and observations
- deterministic grading against gold codes
- dense feedback through intermediate draft quality
- clear penalties for unsupported or destructive actions

## Environment Overview

Each episode loads one fixed task and exposes a small allowed code set. The agent can iteratively refine a coding draft over multiple steps before finalizing.

### Action Space

`MedicalCodingAction`

- `primary_code`: proposed primary ICD-10 code
- `secondary_codes`: proposed supporting ICD-10 codes
- `needs_review`: whether a human coder should review the chart
- `request_hint`: reveal the next deterministic hint
- `finalize`: end the task and score the current draft

### Observation Space

`MedicalCodingObservation`

- `task_id`: current task key
- `difficulty`: `easy`, `medium`, or `hard`
- `objective`: task objective
- `encounter_text`: offline chart excerpt
- `allowed_codes`: closed label set for the task
- `revealed_hints`: hints shown so far
- `current_primary_code`: current draft primary code
- `current_secondary_codes`: current draft secondary codes
- `current_needs_review`: current review draft flag
- `attempts_remaining`: remaining step budget
- `progress_score`: best deterministic grader score reached so far
- `grader_feedback`: reproducible textual feedback from the grader
- `reward_breakdown`: typed `RewardBreakdown` model with score delta and penalties

### State

`MedicalCodingState`

- episode metadata and step count
- current task and difficulty
- current draft codes and review flag
- best score reached so far
- hints used and repeated-action count
- completion flag

## Tasks

The environment ships with three tasks that increase in difficulty.

### Easy: `easy_t2dm_followup`

- Objective: code uncomplicated type 2 diabetes and capture long-term oral hypoglycemic use.
- Gold answer: primary `E11.9`, secondary `Z79.84`
- Difficulty driver: basic specificity and status-code capture.

### Medium: `medium_hypertensive_ckd`

- Objective: use the hypertensive CKD combination diagnosis and add the documented CKD stage.
- Gold answer: primary `I12.9`, secondary `N18.30`
- Difficulty driver: choosing the combination code instead of coding isolated hypertension.

### Hard: `hard_chest_pain_review`

- Objective: code documented symptoms while escalating an unresolved acute coronary syndrome workup for review.
- Gold answer: primary `R07.9`, secondary `R94.31`, `needs_review=True`
- Difficulty driver: avoiding unsupported definitive diagnoses and recognizing when human review is necessary.

## Grading

Each task has a deterministic grader in [grading.py](/D:/projects/hackathon/Learn/Build%20My%20RL%20Agent/medical_coding_assistant/grading.py).

Scoring rules:

- exact primary code: `+0.60`
- accepted but less specific primary alternate: `+0.50`
- right code family, wrong specific code: `+0.30`
- correct secondary support codes: up to `+0.25`
- correct review flag: `+0.15`
- unsupported extra secondary codes: penalty up to `-0.15`

The grader always returns a score in `[0.0, 1.0]`.

## Reward Function

The environment gives feedback throughout the trajectory instead of only at completion.

- reward equals improvement over the best grader score achieved so far
- `request_hint=True` applies a small penalty
- invalid codes outside the allowed set apply a penalty
- repeating the same draft applies a penalty
- timing out by exhausting the step budget applies a penalty

This encourages incremental progress toward the objective while discouraging infinite loops and unsupported edits.

## OpenEnv Interface Compliance

This environment is compliant with the OpenEnv server interface used by `openenv validate`.

- Typed models:
  - action: `MedicalCodingAction`
  - observation: `MedicalCodingObservation`
  - reward details: `RewardBreakdown` (embedded in observation)
  - state: `MedicalCodingState`
- API shape:
  - `reset(...) -> observation`
  - `step(action, ...) -> observation`
  - `state` property exposes current typed state
- Gym-style tuple mapping:
  - OpenEnv transports `observation`, `reward`, and `done` in the step response payload.
  - `info` is provided in `observation.metadata["info"]`.

Validation status in this workspace:

```bash
openenv validate .
# [OK] : Ready for multi-mode deployment
```

## Setup

```bash
cd medical_coding_assistant
uv sync
```

## Run Locally

```bash
uv run --project . server
```

Or:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Validate

```bash
openenv validate .
```

If `openenv` is not on your shell PATH (common on Windows), run the venv executable directly:

```powershell
& "..\.venv\Scripts\openenv.exe" validate .
```

## Docker

Build:

```bash
docker build -t medical-coding-assistant:latest -f server/Dockerfile .
```

Run:

```bash
docker run --rm -p 8000:8000 medical-coding-assistant:latest
```

## Usage Example

```python
from medical_coding_assistant import MedicalCodingAction, MedicalCodingAssistantEnv

client = MedicalCodingAssistantEnv(base_url="http://localhost:8000").sync()

with client:
    reset_result = client.reset(task_id="easy_t2dm_followup")
    print(reset_result.observation.encounter_text)

    result = client.step(
        MedicalCodingAction(
            primary_code="E11.9",
            secondary_codes=["Z79.84"],
            finalize=True,
        )
    )
    print(result.reward, result.done, result.observation.progress_score)
```

## Baselines

Reference heuristic baseline executed locally:

- easy: `1.00`
- medium: `1.00`
- hard: `1.00`
- macro average: `1.00`

LLM baseline reproduction script:

- file: [baseline_inference.py](/D:/projects/hackathon/Learn/Build%20My%20RL%20Agent/medical_coding_assistant/baseline_inference.py)
- client: OpenAI Python SDK
- auth: `HF_TOKEN`
- transport: Hugging Face Inference Router OpenAI-compatible API

`HF_TOKEN` was not available in this workspace, so the OpenAI-compatible baseline was added but not executed here.

Run the offline reference baseline:

```bash
python baseline_inference.py --mode heuristic
```

## Dataset Learning Simulation

You can simulate incremental learning directly from [diagnoses.csv](/D:/projects/hackathon/Learn/Build%20My%20RL%20Agent/medical_coding_assistant/data/diagnoses.csv):

```bash
python -m medical_coding_assistant.simulate_learning --csv medical_coding_assistant/data/diagnoses.csv --warmup 1000
```

Sample result in this workspace:

- rows_total: `274592`
- rows_warmup: `1000`
- rows_evaluated: `273592`
- accuracy: `0.2352`

## Project Structure

```text
medical_coding_assistant/
├── __init__.py
├── baseline_inference.py
├── client.py
├── grading.py
├── models.py
├── openenv.yaml
├── outputs/
├── pyproject.toml
├── README.md
├── tasks.py
└── server/
    ├── __init__.py
    ├── app.py
    ├── medical_coding_environment.py
    ├── Dockerfile
    └── requirements.txt
```
