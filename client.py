"""Client for the Medical Coding Assistant environment."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import MedicalCodingAction, MedicalCodingObservation, MedicalCodingState


class MedicalCodingAssistantEnv(
    EnvClient[MedicalCodingAction, MedicalCodingObservation, MedicalCodingState]
):
    """Persistent client for the Medical Coding Assistant environment."""

    def _step_payload(self, action: MedicalCodingAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[MedicalCodingObservation]:
        obs_data = payload.get("observation", {})
        observation = MedicalCodingObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            objective=obs_data.get("objective", ""),
            encounter_text=obs_data.get("encounter_text", ""),
            allowed_codes=obs_data.get("allowed_codes", []),
            revealed_hints=obs_data.get("revealed_hints", []),
            current_primary_code=obs_data.get("current_primary_code", ""),
            current_secondary_codes=obs_data.get("current_secondary_codes", []),
            current_needs_review=obs_data.get("current_needs_review", False),
            attempts_remaining=obs_data.get("attempts_remaining", 0),
            progress_score=obs_data.get("progress_score", 0.0),
            grader_feedback=obs_data.get("grader_feedback", []),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> MedicalCodingState:
        return MedicalCodingState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task_id=payload.get("current_task_id", ""),
            difficulty=payload.get("difficulty", "easy"),
            current_primary_code=payload.get("current_primary_code", ""),
            current_secondary_codes=payload.get("current_secondary_codes", []),
            current_needs_review=payload.get("current_needs_review", False),
            best_score=payload.get("best_score", 0.0),
            hints_used=payload.get("hints_used", 0),
            repeated_actions=payload.get("repeated_actions", 0),
            completed=payload.get("completed", False),
        )
