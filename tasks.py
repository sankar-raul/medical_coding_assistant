"""Offline task set for the Medical Coding Assistant environment."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskCase:
    """A deterministic coding task with gold answers and hints."""

    task_id: str
    difficulty: str
    objective: str
    encounter_text: str
    allowed_codes: tuple[str, ...]
    gold_primary: str
    gold_secondary: tuple[str, ...]
    should_review: bool
    accepted_primary_alternates: tuple[str, ...] = ()
    accepted_secondary_alternates: tuple[str, ...] = ()
    hints: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


TASKS: dict[str, TaskCase] = {
    "easy_t2dm_followup": TaskCase(
        task_id="easy_t2dm_followup",
        difficulty="easy",
        objective="Assign the most specific primary diagnosis code and any supported status code.",
        encounter_text=(
            "Outpatient follow-up: 58-year-old with type 2 diabetes mellitus without "
            "documented complications. She takes metformin chronically. No insulin use. "
            "No hypertension documented at this visit."
        ),
        allowed_codes=("E11.9", "E11.65", "Z79.84", "I10", "R73.03"),
        gold_primary="E11.9",
        gold_secondary=("Z79.84",),
        should_review=False,
        accepted_primary_alternates=("E11",),
        hints=(
            "The note supports diabetes without complications, not prediabetes.",
            "Long-term oral hypoglycemic use can be coded separately when explicitly documented.",
        ),
    ),
    "medium_hypertensive_ckd": TaskCase(
        task_id="medium_hypertensive_ckd",
        difficulty="medium",
        objective="Use the combination diagnosis when hypertension and CKD are documented together.",
        encounter_text=(
            "Clinic visit: 71-year-old with longstanding hypertension and chronic kidney "
            "disease stage 3a. Provider documents blood-pressure management for hypertensive "
            "chronic kidney disease. No diabetes is documented."
        ),
        allowed_codes=("I12.9", "I10", "N18.30", "N18.9", "E11.22", "Z79.899"),
        gold_primary="I12.9",
        gold_secondary=("N18.30",),
        should_review=False,
        hints=(
            "Plain essential hypertension is not the best primary code when CKD is linked.",
            "The documented CKD stage should still be captured as an additional code.",
        ),
    ),
    "hard_chest_pain_review": TaskCase(
        task_id="hard_chest_pain_review",
        difficulty="hard",
        objective="Code the documented symptoms and flag the chart for human review when the final diagnosis remains uncertain.",
        encounter_text=(
            "Emergency department note: patient presents with acute chest pain and an "
            "abnormal ECG. Clinician documents 'rule out acute coronary syndrome' and "
            "requests cardiology review. Final diagnosis is not established in the note."
        ),
        allowed_codes=("R07.9", "R94.31", "I20.9", "I21.9", "Z03.89"),
        gold_primary="R07.9",
        gold_secondary=("R94.31",),
        should_review=True,
        hints=(
            "Do not code a definitive acute coronary diagnosis that is only being ruled out.",
            "The chart explicitly says a human specialist review is needed before a final diagnosis.",
        ),
    ),
}

TASK_SEQUENCE: tuple[str, ...] = (
    "easy_t2dm_followup",
    "medium_hypertensive_ckd",
    "hard_chest_pain_review",
)
