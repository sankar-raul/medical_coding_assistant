"""Core OpenEnv environment for Medical Coding Assistant."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from medical_coding_assistant.grading import Submission, grade_submission
    from medical_coding_assistant.models import (
        MedicalCodingAction,
        MedicalCodingObservation,
        RewardBreakdown,
        MedicalCodingState,
    )
    from medical_coding_assistant.tasks import TASK_SEQUENCE, TASKS, TaskCase
except ModuleNotFoundError:
    from grading import Submission, grade_submission
    from models import MedicalCodingAction, MedicalCodingObservation, MedicalCodingState, RewardBreakdown
    from tasks import TASK_SEQUENCE, TASKS, TaskCase


class MedicalCodingEnvironment(
    Environment[MedicalCodingAction, MedicalCodingObservation, MedicalCodingState]
):
    """A deterministic clinical coding workflow benchmark."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 6

    def __init__(self) -> None:
        super().__init__()
        self._task_index = 0
        self._task: TaskCase = TASKS[TASK_SEQUENCE[0]]
        self._last_signature = ""
        self._state = MedicalCodingState(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> MedicalCodingObservation:
        task_id = kwargs.get("task_id")
        if task_id is None:
            task_id = TASK_SEQUENCE[self._task_index % len(TASK_SEQUENCE)]
            self._task_index += 1
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Available tasks: {list(TASKS)}")

        self._task = TASKS[task_id]
        self._last_signature = ""
        self._state = MedicalCodingState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            current_task_id=self._task.task_id,
            difficulty=self._task.difficulty,  # type: ignore[arg-type]
            current_primary_code="",
            current_secondary_codes=[],
            current_needs_review=False,
            best_score=0.0,
            hints_used=0,
            repeated_actions=0,
            completed=False,
        )
        return self._build_observation(
            reward=0.0,
            done=False,
            feedback=["Environment reset. Start by drafting codes or requesting a hint."],
            reward_breakdown=RewardBreakdown(total=0.0),
        )

    def step(
        self,
        action: MedicalCodingAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> MedicalCodingObservation:
        if self._state.completed:
            return self._build_observation(
                reward=0.0,
                done=True,
                feedback=["Task is already complete. Call reset() for a new task."],
            )

        self._state.step_count += 1
        feedback: list[str] = []
        score_delta = 0.0
        hint_penalty = 0.0
        invalid_code_penalty = 0.0
        loop_penalty = 0.0
        timeout_penalty = 0.0

        if action.request_hint:
            if self._state.hints_used < len(self._task.hints):
                self._state.hints_used += 1
                hint_penalty -= 0.02
                feedback.append("Hint revealed with a small efficiency penalty.")
            else:
                hint_penalty -= 0.05
                feedback.append("No hints remain.")

        proposed_primary = action.primary_code.strip().upper() or self._state.current_primary_code
        proposed_secondary = [
            code.strip().upper()
            for code in (action.secondary_codes or self._state.current_secondary_codes)
            if code.strip()
        ]
        proposed_secondary = list(dict.fromkeys(proposed_secondary))
        proposed_review = action.needs_review

        invalid_codes = [
            code
            for code in [proposed_primary, *proposed_secondary]
            if code and code not in self._task.allowed_codes
        ]
        if invalid_codes:
            invalid_code_penalty -= min(0.2, 0.1 * len(invalid_codes))
            feedback.append(f"Unsupported code(s): {invalid_codes}.")

        signature = "|".join(
            [
                proposed_primary,
                ",".join(proposed_secondary),
                str(proposed_review),
                str(action.request_hint),
                str(action.finalize),
            ]
        )
        if signature == self._last_signature:
            self._state.repeated_actions += 1
            loop_penalty -= 0.05
            feedback.append("Repeated the same draft; small loop penalty applied.")
        self._last_signature = signature

        self._state.current_primary_code = proposed_primary
        self._state.current_secondary_codes = proposed_secondary
        self._state.current_needs_review = proposed_review

        grade = grade_submission(
            self._task,
            Submission(
                primary_code=proposed_primary,
                secondary_codes=tuple(proposed_secondary),
                needs_review=proposed_review,
            ),
        )
        delta = round(grade.score - self._state.best_score, 4)
        if delta > 0:
            score_delta = delta
            feedback.append(f"Draft improved by {delta:.2f}.")
        self._state.best_score = max(self._state.best_score, grade.score)
        feedback.extend(list(grade.feedback))

        done = False
        if action.finalize or grade.score >= 1.0:
            done = True
            self._state.completed = True
            feedback.append("Task finalized.")
        elif self._state.step_count >= self.MAX_STEPS:
            done = True
            self._state.completed = True
            timeout_penalty -= 0.1
            feedback.append("Step budget exhausted.")

        total_reward = round(
            score_delta
            + hint_penalty
            + invalid_code_penalty
            + loop_penalty
            + timeout_penalty,
            4,
        )
        reward_breakdown = RewardBreakdown(
            score_delta=score_delta,
            hint_penalty=hint_penalty,
            invalid_code_penalty=invalid_code_penalty,
            loop_penalty=loop_penalty,
            timeout_penalty=timeout_penalty,
            total=total_reward,
        )

        return self._build_observation(
            reward=total_reward,
            done=done,
            feedback=feedback,
            grader_score=grade.score,
            reward_breakdown=reward_breakdown,
        )

    def _build_observation(
        self,
        reward: float,
        done: bool,
        feedback: list[str],
        grader_score: float | None = None,
        reward_breakdown: RewardBreakdown | None = None,
    ) -> MedicalCodingObservation:
        score = self._state.best_score if grader_score is None else grader_score
        breakdown = reward_breakdown or RewardBreakdown(total=reward)
        info = {
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
            "hints_remaining": max(0, len(self._task.hints) - self._state.hints_used),
            "grader_score": score,
            "reward_breakdown": breakdown.model_dump(),
        }
        return MedicalCodingObservation(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,  # type: ignore[arg-type]
            objective=self._task.objective,
            encounter_text=self._task.encounter_text,
            allowed_codes=list(self._task.allowed_codes),
            revealed_hints=list(self._task.hints[: self._state.hints_used]),
            current_primary_code=self._state.current_primary_code,
            current_secondary_codes=list(self._state.current_secondary_codes),
            current_needs_review=self._state.current_needs_review,
            attempts_remaining=max(0, self.MAX_STEPS - self._state.step_count),
            progress_score=self._state.best_score,
            grader_feedback=feedback,
            reward_breakdown=breakdown,
            done=done,
            reward=reward,
            metadata={
                "info": info,
                "grader": {"score": score},
            },
        )

    @property
    def state(self) -> MedicalCodingState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="MedicalCodingEnvironment",
            description="Deterministic ICD-10 coding workflow benchmark with graded real-world tasks.",
            version="0.1.0",
        )
