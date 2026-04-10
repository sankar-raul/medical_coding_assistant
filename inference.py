"""OpenAI-compatible baseline evaluation for Medical Coding Assistant."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

from openai import OpenAI

if __package__ in (None, ""):
    # Support direct script execution from inside the package directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medical_coding_assistant.grading import Submission, grade_submission
from medical_coding_assistant.models import MedicalCodingAction
from medical_coding_assistant.server.medical_coding_environment import MedicalCodingEnvironment
from medical_coding_assistant.tasks import TASK_SEQUENCE, TASKS
from dotenv import load_dotenv

load_dotenv()

HF_ROUTER_BASE_URL = os.environ.get("API_BASE_URL")
DEFAULT_MODEL = os.environ.get("MODEL_NAME")


def build_prompt(task_id: str) -> str:
    task = TASKS[task_id]
    return (
        "You are acting as a medical coding assistant for an offline benchmark. "
        "Return only valid JSON with keys primary_code, secondary_codes, needs_review, "
        "request_hint, finalize. Use only codes from allowed_codes.\n\n"
        f"Task ID: {task.task_id}\n"
        f"Difficulty: {task.difficulty}\n"
        f"Objective: {task.objective}\n"
        f"Encounter: {task.encounter_text}\n"
        f"Allowed codes: {list(task.allowed_codes)}\n"
        "Set finalize to true."
    )


def parse_action(raw_text: str) -> MedicalCodingAction:
    def _coerce_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "on"}:
                return True
            if normalized in {"false", "0", "no", "n", "off", ""}:
                return False
        return default

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Model response is not valid JSON: {raw_text}")
    payload = json.loads(raw_text[start : end + 1])
    payload["request_hint"] = _coerce_bool(payload.get("request_hint"), default=False)
    payload["needs_review"] = _coerce_bool(payload.get("needs_review"), default=False)
    payload["finalize"] = _coerce_bool(payload.get("finalize"), default=False)
    secondary_codes = payload.get("secondary_codes")
    payload["secondary_codes"] = secondary_codes if isinstance(secondary_codes, list) else []
    primary_code = payload.get("primary_code")
    payload["primary_code"] = primary_code if isinstance(primary_code, str) else ""
    return MedicalCodingAction(**payload)


def run_openai_baseline(model: str) -> list[dict[str, object]]:
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN") or hf_token
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for the OpenAI-compatible baseline.")

    client = OpenAI(base_url=HF_ROUTER_BASE_URL, api_key=hf_token)
    env = MedicalCodingEnvironment()
    results: list[dict[str, object]] = []

    for task_id in TASK_SEQUENCE:
        reset_obs = env.reset(task_id=task_id)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You output only JSON."},
                {"role": "user", "content": build_prompt(task_id)},
            ],
        )
        raw_message = response.choices[0].message.content or ""
        action = parse_action(raw_message)
        step_obs = env.step(action)
        grade = grade_submission(
            TASKS[task_id],
            Submission(
                primary_code=step_obs.current_primary_code,
                secondary_codes=tuple(step_obs.current_secondary_codes),
                needs_review=step_obs.current_needs_review,
            ),
        )
        results.append(
            {
                "task_id": task_id,
                "difficulty": reset_obs.difficulty,
                "score": grade.score,
                "reward": step_obs.reward,
                "reward_breakdown": step_obs.reward_breakdown.model_dump(),
                "done": step_obs.done,
                "feedback": list(grade.feedback),
                "action": action.model_dump(),
            }
        )

    return results


def run_heuristic_reference() -> list[dict[str, object]]:
    env = MedicalCodingEnvironment()
    results: list[dict[str, object]] = []
    for task_id in TASK_SEQUENCE:
        task = TASKS[task_id]
        env.reset(task_id=task_id)
        result = env.step(
            MedicalCodingAction(
                primary_code=task.gold_primary,
                secondary_codes=list(task.gold_secondary),
                needs_review=task.should_review,
                finalize=True,
            )
        )
        results.append(
            {
                "task_id": task_id,
                "difficulty": task.difficulty,
                "score": result.metadata["grader"]["score"],
                "reward": result.reward,
                "reward_breakdown": result.reward_breakdown.model_dump(),
                "done": result.done,
                "feedback": result.grader_feedback,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode",
        choices=("openai", "heuristic"),
        default="openai",
        help="Use the OpenAI-compatible HF router baseline or the offline reference heuristic.",
    )
    args = parser.parse_args()

    results = (
        run_openai_baseline(args.model)
        if args.mode == "openai"
        else run_heuristic_reference()
    )
    macro_average = mean(float(result["score"]) for result in results)
    payload = {
        "mode": args.mode,
        "model": args.model if args.mode == "openai" else "reference-heuristic",
        "macro_average": round(macro_average, 4),
        "results": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
