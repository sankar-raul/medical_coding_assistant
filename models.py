"""Typed models for the Medical Coding Assistant environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


Difficulty = Literal["easy", "medium", "hard"]


class RewardBreakdown(BaseModel):
    """Typed reward components for deterministic per-step feedback."""

    score_delta: float = Field(
        default=0.0,
        description="Positive reward from improvement in deterministic grader score.",
    )
    hint_penalty: float = Field(
        default=0.0,
        description="Penalty applied when requesting hints.",
    )
    invalid_code_penalty: float = Field(
        default=0.0,
        description="Penalty for codes outside the task's allowed set.",
    )
    loop_penalty: float = Field(
        default=0.0,
        description="Penalty for repeating identical actions.",
    )
    timeout_penalty: float = Field(
        default=0.0,
        description="Penalty when the episode ends by step-budget exhaustion.",
    )
    total: float = Field(
        default=0.0,
        description="Final per-step reward after all components.",
    )


class MedicalCodingAction(Action):
    """Action submitted by the agent on each environment step."""

    primary_code: str = Field(
        default="",
        description="Proposed primary ICD-10 code. Leave blank to keep the current draft.",
    )
    secondary_codes: list[str] = Field(
        default_factory=list,
        description="Proposed secondary ICD-10 codes for supporting findings or status codes.",
    )
    needs_review: bool = Field(
        default=False,
        description="Whether the chart should be escalated for human review.",
    )
    request_hint: bool = Field(
        default=False,
        description="Ask the environment for an additional deterministic hint.",
    )
    finalize: bool = Field(
        default=False,
        description="End the current task and score the current draft.",
    )


class MedicalCodingObservation(Observation):
    """Observation returned after reset and each environment step."""

    task_id: str = Field(..., description="Current task identifier.")
    difficulty: Difficulty = Field(..., description="Task difficulty level.")
    objective: str = Field(..., description="Task objective for the agent.")
    encounter_text: str = Field(..., description="Clinical note excerpt to code.")
    allowed_codes: list[str] = Field(
        default_factory=list,
        description="Closed code set allowed for this task.",
    )
    revealed_hints: list[str] = Field(
        default_factory=list,
        description="Hints revealed so far by the environment.",
    )
    current_primary_code: str = Field(
        default="",
        description="The agent's current draft primary code.",
    )
    current_secondary_codes: list[str] = Field(
        default_factory=list,
        description="The agent's current draft secondary codes.",
    )
    current_needs_review: bool = Field(
        default=False,
        description="The agent's current draft review flag.",
    )
    attempts_remaining: int = Field(
        default=0,
        description="How many steps remain in the current task.",
    )
    progress_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Best deterministic grader score achieved so far.",
    )
    grader_feedback: list[str] = Field(
        default_factory=list,
        description="Deterministic feedback strings explaining the current draft.",
    )
    reward_breakdown: RewardBreakdown = Field(
        default_factory=RewardBreakdown,
        description="Typed per-step reward details.",
    )


class MedicalCodingState(State):
    """Internal state exposed through the OpenEnv state endpoint."""

    current_task_id: str = Field(default="", description="Active task identifier.")
    difficulty: Difficulty = Field(default="easy", description="Task difficulty.")
    current_primary_code: str = Field(default="", description="Draft primary code.")
    current_secondary_codes: list[str] = Field(
        default_factory=list,
        description="Draft secondary codes.",
    )
    current_needs_review: bool = Field(
        default=False,
        description="Current review flag draft.",
    )
    best_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Best grader score seen in this episode.",
    )
    hints_used: int = Field(default=0, ge=0, description="Number of hints revealed.")
    repeated_actions: int = Field(
        default=0,
        ge=0,
        description="How many times the agent repeated the same draft.",
    )
    completed: bool = Field(default=False, description="Whether the task has finished.")
