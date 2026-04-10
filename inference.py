"""OpenAI-compatible baseline evaluation for Medical Coding Assistant."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Callable

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
PROXY_API_KEY = os.environ.get("API_KEY")
DEFAULT_MODEL = os.environ.get("MODEL_NAME")


def emit_log(tag: str, payload: dict[str, object]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'))}", flush=True)


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


def run_openai_baseline(
    model: str, on_step: Callable[[dict[str, object]], None] | None = None
) -> list[dict[str, object]]:
    load_dotenv()
    api_key = os.environ.get("API_KEY") or PROXY_API_KEY
    client: OpenAI | None = None
    config_error = ""
    if not api_key:
        config_error = "RuntimeError: API_KEY is required for the OpenAI-compatible baseline."
    elif not HF_ROUTER_BASE_URL:
        config_error = "RuntimeError: API_BASE_URL is required for the OpenAI-compatible baseline."
    elif not model:
        config_error = "RuntimeError: Model name is required. Set MODEL_NAME or pass --model."
    else:
        client = OpenAI(base_url=HF_ROUTER_BASE_URL, api_key=api_key)
    env = MedicalCodingEnvironment()
    results: list[dict[str, object]] = []

    for task_id in TASK_SEQUENCE:
        reset_obs = env.reset(task_id=task_id)
        action = MedicalCodingAction(
            primary_code="",
            secondary_codes=[],
            needs_review=True,
            request_hint=False,
            finalize=True,
        )
        error_message = ""
        if config_error:
            error_message = config_error
        else:
            try:
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
            except Exception as exc:
                # Keep evaluation running even if one API call or parse fails.
                error_message = f"{type(exc).__name__}: {exc}"
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
        if error_message:
            results[-1]["error"] = error_message
        if on_step:
            on_step(results[-1])

    return results


def run_heuristic_reference(
    on_step: Callable[[dict[str, object]], None] | None = None
) -> list[dict[str, object]]:
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
        if on_step:
            on_step(results[-1])
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
    resolved_model = args.model if args.mode == "openai" else "reference-heuristic"

    emit_log(
        "START",
        {
            "mode": args.mode,
            "model": resolved_model,
            "task_count": len(TASK_SEQUENCE),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    step_index = 0

    def on_step(result: dict[str, object]) -> None:
        nonlocal step_index
        step_index += 1
        payload: dict[str, object] = {
            "index": step_index,
            "task_id": result["task_id"],
            "difficulty": result["difficulty"],
            "score": round(float(result["score"]), 4),
            "reward": round(float(result["reward"]), 4),
            "done": bool(result["done"]),
        }
        if "error" in result:
            payload["error"] = result["error"]
        emit_log("STEP", payload)

    try:
        results = (
            run_openai_baseline(args.model, on_step=on_step)
            if args.mode == "openai"
            else run_heuristic_reference(on_step=on_step)
        )
        fatal_error = ""
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        results = run_heuristic_reference(on_step=on_step)
    macro_average = mean(float(result["score"]) for result in results) if results else 0.0
    end_payload: dict[str, object] = {
        "mode": args.mode,
        "model": resolved_model,
        "status": "ok" if not fatal_error else "degraded",
        "tasks_completed": len(results),
        "macro_average": round(macro_average, 4),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if fatal_error:
        end_payload["fatal_error"] = fatal_error
    emit_log("END", end_payload)


if __name__ == "__main__":
    main()
