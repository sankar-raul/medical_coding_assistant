"""Simulate incremental learning from diagnoses.csv.

The learner is intentionally simple and deterministic:
- processes visits in chronological order
- predicts primary ICD-10 from aggregate frequency signals
- updates itself after each labeled visit (online learning)
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Visit:
    patient_id: str
    visit_date: date
    visit_type: str
    primary_icd10: str
    secondary_icd10s: tuple[str, ...]
    provider_specialty: str


def _parse_date(value: str) -> date:
    parts = value.split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _split_codes(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(code.strip().upper() for code in value.split("|") if code.strip())


def load_visits(csv_path: Path) -> list[Visit]:
    visits: list[Visit] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            visits.append(
                Visit(
                    patient_id=row["patient_id"],
                    visit_date=_parse_date(row["visit_date"]),
                    visit_type=row["visit_type"].strip().lower(),
                    primary_icd10=row["primary_icd10"].strip().upper(),
                    secondary_icd10s=_split_codes(row["secondary_icd10s"].strip()),
                    provider_specialty=row["provider_specialty"].strip().lower(),
                )
            )
    visits.sort(key=lambda v: (v.visit_date, v.patient_id))
    return visits


class OnlineDiagnosisLearner:
    """Online Naive Bayes learner with contextual and patient-history features."""

    def __init__(self) -> None:
        self.global_counts: Counter[str] = Counter()
        self.by_visit_type: dict[str, Counter[str]] = defaultdict(Counter)
        self.by_visit_type_total: Counter[str] = Counter()
        self.by_specialty: dict[str, Counter[str]] = defaultdict(Counter)
        self.by_specialty_total: Counter[str] = Counter()
        self.by_secondary: dict[str, Counter[str]] = defaultdict(Counter)
        self.by_secondary_total: Counter[str] = Counter()
        self.transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self.transition_total: Counter[str] = Counter()
        self.patient_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.patient_total: Counter[str] = Counter()
        self.by_visit_specialty: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.by_visit_specialty_total: Counter[tuple[str, str]] = Counter()
        self.by_exact_context: dict[tuple[str, str, tuple[str, ...]], Counter[str]] = defaultdict(Counter)
        self.by_exact_context_total: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
        self.last_primary_by_patient: dict[str, str] = {}
        self.total_examples = 0

        # Weights tuned for this deterministic simulation.
        self.w_prior = 0.7
        self.w_visit_type = 1.5
        self.w_specialty = 1.4
        self.w_visit_specialty = 1.8
        self.w_secondary = 0.8
        self.w_transition = 1.2
        self.w_patient = 2.2
        self.w_exact_context = 2.0
        self.alpha = 0.1

    def predict(self, visit: Visit) -> str:
        if not self.global_counts:
            # Before any training data exists.
            return "I10"

        alpha = self.alpha
        classes = list(self.global_counts.keys())
        num_classes = len(classes)
        visit_specialty = (visit.visit_type, visit.provider_specialty)
        secondary_tuple = tuple(sorted(visit.secondary_icd10s))
        exact_context = (visit.visit_type, visit.provider_specialty, secondary_tuple)
        previous = self.last_primary_by_patient.get(visit.patient_id)

        scores: dict[str, float] = {}
        for icd in classes:
            # Prior prevalence.
            prior = math.log(
                (self.global_counts[icd] + alpha)
                / (self.total_examples + alpha * num_classes)
            )
            score = self.w_prior * prior

            # Visit-type likelihood.
            vt_total = self.by_visit_type_total[visit.visit_type]
            if vt_total:
                score += self.w_visit_type * math.log(
                    (self.by_visit_type[visit.visit_type][icd] + alpha)
                    / (vt_total + alpha * num_classes)
                )

            # Specialty likelihood.
            sp_total = self.by_specialty_total[visit.provider_specialty]
            if sp_total:
                score += self.w_specialty * math.log(
                    (self.by_specialty[visit.provider_specialty][icd] + alpha)
                    / (sp_total + alpha * num_classes)
                )

            # Joint visit-type + specialty likelihood.
            vs_total = self.by_visit_specialty_total[visit_specialty]
            if vs_total:
                score += self.w_visit_specialty * math.log(
                    (self.by_visit_specialty[visit_specialty][icd] + alpha)
                    / (vs_total + alpha * num_classes)
                )

            # Secondary-code signals.
            for sec in visit.secondary_icd10s:
                sec_total = self.by_secondary_total[sec]
                if sec_total:
                    score += self.w_secondary * math.log(
                        (self.by_secondary[sec][icd] + alpha)
                        / (sec_total + alpha * num_classes)
                    )

            # Patient transition signal.
            previous = self.last_primary_by_patient.get(visit.patient_id)
            if previous:
                tr_total = self.transition_total[previous]
                if tr_total:
                    score += self.w_transition * math.log(
                        (self.transitions[previous][icd] + alpha)
                        / (tr_total + alpha * num_classes)
                    )

            # Patient-specific distribution.
            patient_total = self.patient_total[visit.patient_id]
            if patient_total:
                score += self.w_patient * math.log(
                    (self.patient_counts[visit.patient_id][icd] + alpha)
                    / (patient_total + alpha * num_classes)
                )

            # Fully combined context distribution.
            exact_total = self.by_exact_context_total[exact_context]
            if exact_total:
                score += self.w_exact_context * math.log(
                    (self.by_exact_context[exact_context][icd] + alpha)
                    / (exact_total + alpha * num_classes)
                )

            scores[icd] = score

        # Deterministic tie-break: higher score, then lexicographically smaller ICD.
        return min(scores, key=lambda icd: (-scores[icd], icd))

    def update(self, visit: Visit) -> None:
        gold = visit.primary_icd10
        previous = self.last_primary_by_patient.get(visit.patient_id)

        self.global_counts[gold] += 1
        self.total_examples += 1

        self.by_visit_type[visit.visit_type][gold] += 1
        self.by_visit_type_total[visit.visit_type] += 1

        self.by_specialty[visit.provider_specialty][gold] += 1
        self.by_specialty_total[visit.provider_specialty] += 1

        visit_specialty = (visit.visit_type, visit.provider_specialty)
        self.by_visit_specialty[visit_specialty][gold] += 1
        self.by_visit_specialty_total[visit_specialty] += 1

        for sec in visit.secondary_icd10s:
            self.by_secondary[sec][gold] += 1
            self.by_secondary_total[sec] += 1

        secondary_tuple = tuple(sorted(visit.secondary_icd10s))
        exact_context = (visit.visit_type, visit.provider_specialty, secondary_tuple)
        self.by_exact_context[exact_context][gold] += 1
        self.by_exact_context_total[exact_context] += 1

        if previous:
            self.transitions[previous][gold] += 1
            self.transition_total[previous] += 1

        self.patient_counts[visit.patient_id][gold] += 1
        self.patient_total[visit.patient_id] += 1
        self.last_primary_by_patient[visit.patient_id] = gold


def simulate_learning(visits: list[Visit], warmup: int) -> dict[str, float | int]:
    learner = OnlineDiagnosisLearner()
    correct = 0
    evaluated = 0

    for i, visit in enumerate(visits):
        if i < warmup:
            learner.update(visit)
            continue

        predicted = learner.predict(visit)
        if predicted == visit.primary_icd10:
            correct += 1
        evaluated += 1
        learner.update(visit)

    accuracy = (correct / evaluated) if evaluated else 0.0
    return {
        "rows_total": len(visits),
        "rows_warmup": warmup,
        "rows_evaluated": evaluated,
        "correct": correct,
        "accuracy": round(accuracy, 4),
    }


def simulate_learning_with_learner(
    learner: OnlineDiagnosisLearner,
    visits: list[Visit],
    warmup: int,
    print_each_episode: bool = False,
) -> dict[str, float | int]:
    correct = 0
    evaluated = 0

    for i, visit in enumerate(visits):
        if i < warmup:
            learner.update(visit)
            continue

        predicted = learner.predict(visit)
        if predicted == visit.primary_icd10:
            correct += 1
        evaluated += 1
        if print_each_episode:
            running_accuracy = correct / evaluated
            print(
                f"episode={evaluated} index={i} predicted={predicted} "
                f"gold={visit.primary_icd10} accuracy={running_accuracy:.4f}"
            )
        learner.update(visit)

    accuracy = (correct / evaluated) if evaluated else 0.0
    return {
        "rows_total": len(visits),
        "rows_warmup": warmup,
        "rows_evaluated": evaluated,
        "correct": correct,
        "accuracy": round(accuracy, 4),
    }


def save_learner(learner: OnlineDiagnosisLearner, pkl_path: Path) -> None:
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(learner, f)


def load_learner(pkl_path: Path) -> OnlineDiagnosisLearner:
    with pkl_path.open("rb") as f:
        learner = pickle.load(f)
    if not isinstance(learner, OnlineDiagnosisLearner):
        raise TypeError(f"Pickle at {pkl_path} does not contain OnlineDiagnosisLearner.")
    return learner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/diagnoses.csv",
        help="Path to diagnoses CSV dataset.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=500,
        help="Number of earliest rows used only for initial learning.",
    )
    parser.add_argument(
        "--save-pkl",
        default="outputs/diagnosis_learner.pkl",
        help="Where to save learner state as pickle after simulation.",
    )
    parser.add_argument(
        "--load-pkl",
        default=None,
        help="Optional path to an existing learner pickle to continue from.",
    )
    parser.add_argument(
        "--print-each-episode",
        action="store_true",
        help="Print running accuracy after every evaluated episode.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    visits = load_visits(csv_path)
    warmup = max(0, min(args.warmup, len(visits)))
    if args.load_pkl:
        learner = load_learner(Path(args.load_pkl))
    else:
        learner = OnlineDiagnosisLearner()

    metrics = simulate_learning_with_learner(
        learner,
        visits,
        warmup=warmup,
        print_each_episode=args.print_each_episode,
    )
    save_path = Path(args.save_pkl)
    save_learner(learner, save_path)

    print("Learning simulation complete")
    print(f"dataset: {csv_path}")
    print(f"rows_total: {metrics['rows_total']}")
    print(f"rows_warmup: {metrics['rows_warmup']}")
    print(f"rows_evaluated: {metrics['rows_evaluated']}")
    print(f"correct: {metrics['correct']}")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"saved_learner_pkl: {save_path}")


if __name__ == "__main__":
    main()
