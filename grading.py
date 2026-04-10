"""Deterministic graders for the Medical Coding Assistant environment."""

from dataclasses import dataclass

from .tasks import TaskCase

SCORE_EPSILON = 1e-4


@dataclass(frozen=True)
class Submission:
    """Normalized submission shape for scoring."""

    primary_code: str
    secondary_codes: tuple[str, ...]
    needs_review: bool


@dataclass(frozen=True)
class GradeResult:
    """Programmatic grade for a single task."""

    task_id: str
    score: float
    feedback: tuple[str, ...]


def _normalize_code(code: str) -> str:
    return code.strip().upper()


def _code_family(code: str) -> str:
    normalized = _normalize_code(code)
    if "." in normalized:
        return normalized.split(".", 1)[0]
    if len(normalized) > 3:
        return normalized[:3]
    return normalized


def _unique_codes(codes: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized_codes: list[str] = []
    for code in codes:
        normalized = _normalize_code(code)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_codes.append(normalized)
    return tuple(normalized_codes)


def grade_submission(task: TaskCase, submission: Submission) -> GradeResult:
    """Return a deterministic score strictly in the range (0.0, 1.0)."""

    primary = _normalize_code(submission.primary_code)
    secondary = _unique_codes(submission.secondary_codes)
    gold_secondary = _unique_codes(task.gold_secondary)
    feedback: list[str] = []
    score = 0.0

    if primary == task.gold_primary:
        score += 0.6
        feedback.append("Primary diagnosis code is exact.")
    elif primary in task.accepted_primary_alternates:
        score += 0.5
        feedback.append("Primary diagnosis is broadly correct but less specific than the gold code.")
    elif _code_family(primary) == _code_family(task.gold_primary):
        score += 0.3
        feedback.append("Primary diagnosis is in the right family but not the right code.")
    else:
        feedback.append("Primary diagnosis code is incorrect.")

    if gold_secondary:
        matched_secondary = len(set(secondary) & set(gold_secondary))
        secondary_weight = 0.25 / len(gold_secondary)
        if matched_secondary:
            score += matched_secondary * secondary_weight
            feedback.append(
                f"Matched {matched_secondary} supporting code(s) out of {len(gold_secondary)}."
            )
        else:
            feedback.append("Missing the supported secondary code(s).")

        extra_secondary = len(set(secondary) - set(gold_secondary))
        if extra_secondary:
            penalty = min(0.15, extra_secondary * 0.05)
            score -= penalty
            feedback.append("Included unsupported secondary code(s).")

    if submission.needs_review == task.should_review:
        score += 0.15
        feedback.append("Review/escalation flag is correct.")
    else:
        feedback.append("Review/escalation flag is incorrect.")

    score = max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, round(score, 4)))
    return GradeResult(task_id=task.task_id, score=score, feedback=tuple(feedback))
