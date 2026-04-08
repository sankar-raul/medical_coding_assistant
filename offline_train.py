"""Offline memorization-first trainer for diagnoses.csv.

This model is designed to maximize offline fit and supports pkl save/load.
It reports both train and holdout accuracy so results are transparent.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Visit:
    patient_id: str
    visit_date: str
    visit_type: str
    primary_icd10: str
    secondary_icd10s: tuple[str, ...]
    provider_specialty: str


def _split_codes(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(sorted(code.strip().upper() for code in value.split("|") if code.strip()))


def load_visits(csv_path: Path) -> list[Visit]:
    visits: list[Visit] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            visits.append(
                Visit(
                    patient_id=row["patient_id"].strip(),
                    visit_date=row["visit_date"].strip(),
                    visit_type=row["visit_type"].strip().lower(),
                    primary_icd10=row["primary_icd10"].strip().upper(),
                    secondary_icd10s=_split_codes(row["secondary_icd10s"].strip()),
                    provider_specialty=row["provider_specialty"].strip().lower(),
                )
            )
    return visits


class OfflineDiagnosisModel:
    """Hierarchical lookup model with deterministic backoff."""

    def __init__(self) -> None:
        self.maps: list[dict[tuple[object, ...], Counter[str]]] = [
            defaultdict(Counter) for _ in range(7)
        ]
        self.level_names = [
            "patient+visit+specialty+secondary",
            "patient",
            "visit+specialty+secondary",
            "visit+specialty",
            "visit",
            "specialty",
            "global",
        ]

    def _keys(self, visit: Visit) -> list[tuple[object, ...]]:
        return [
            (
                visit.patient_id,
                visit.visit_type,
                visit.provider_specialty,
                visit.secondary_icd10s,
            ),
            (visit.patient_id,),
            (visit.visit_type, visit.provider_specialty, visit.secondary_icd10s),
            (visit.visit_type, visit.provider_specialty),
            (visit.visit_type,),
            (visit.provider_specialty,),
            ("__global__",),
        ]

    def fit(self, visits: list[Visit]) -> None:
        for visit in visits:
            for model_map, key in zip(self.maps, self._keys(visit)):
                model_map[key][visit.primary_icd10] += 1

    def predict_with_level(self, visit: Visit) -> tuple[str, str]:
        for level_name, model_map, key in zip(self.level_names, self.maps, self._keys(visit)):
            counts = model_map.get(key)
            if counts:
                pred = min(counts, key=lambda icd: (-counts[icd], icd))
                return pred, level_name
        # Unreachable because global map always has values after fit.
        return "I10", "fallback"

    def predict(self, visit: Visit) -> str:
        pred, _ = self.predict_with_level(visit)
        return pred


def split_visits(
    visits: list[Visit], split_ratio: float, seed: int, split_mode: str
) -> tuple[list[Visit], list[Visit]]:
    ratio = max(0.01, min(split_ratio, 0.99))
    cut = int(len(visits) * ratio)
    if split_mode == "random":
        shuffled = visits[:]
        random.Random(seed).shuffle(shuffled)
        return shuffled[:cut], shuffled[cut:]
    if split_mode == "chronological":
        ordered = sorted(visits, key=lambda v: (v.visit_date, v.patient_id))
        return ordered[:cut], ordered[cut:]
    raise ValueError(f"Unsupported split mode: {split_mode}")


def evaluate(
    model: OfflineDiagnosisModel,
    visits: list[Visit],
    split_name: str,
    print_each_episode: bool = False,
) -> dict[str, float | int]:
    correct = 0
    for i, visit in enumerate(visits, start=1):
        predicted, level = model.predict_with_level(visit)
        if predicted == visit.primary_icd10:
            correct += 1
        if print_each_episode:
            running = correct / i
            print(
                f"split={split_name} episode={i} predicted={predicted} "
                f"gold={visit.primary_icd10} source={level} accuracy={running:.4f}"
            )
    total = len(visits)
    accuracy = (correct / total) if total else 0.0
    return {"rows": total, "correct": correct, "accuracy": round(accuracy, 4)}


def save_model(model: OfflineDiagnosisModel, pkl_path: Path) -> None:
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(model, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/diagnoses.csv", help="Path to CSV dataset.")
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train split ratio in (0, 1).",
    )
    parser.add_argument(
        "--split-mode",
        choices=["random", "chronological"],
        default="random",
        help="How to split train/holdout.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random split.")
    parser.add_argument(
        "--save-pkl",
        default="outputs/offline_diagnosis_model.pkl",
        help="Where to save trained model pickle.",
    )
    parser.add_argument(
        "--print-each-episode",
        action="store_true",
        help="Print running accuracy after each episode on both train and holdout splits.",
    )
    args = parser.parse_args()

    visits = load_visits(Path(args.csv))
    train_visits, holdout_visits = split_visits(
        visits, split_ratio=args.split_ratio, seed=args.seed, split_mode=args.split_mode
    )

    model = OfflineDiagnosisModel()
    model.fit(train_visits)

    train_metrics = evaluate(
        model, train_visits, split_name="train", print_each_episode=args.print_each_episode
    )
    holdout_metrics = evaluate(
        model, holdout_visits, split_name="holdout", print_each_episode=args.print_each_episode
    )

    save_path = Path(args.save_pkl)
    save_model(model, save_path)

    print("Offline training complete")
    print(f"dataset: {args.csv}")
    print(f"split_mode: {args.split_mode}")
    print(f"rows_total: {len(visits)}")
    print(f"rows_train: {train_metrics['rows']}")
    print(f"rows_holdout: {holdout_metrics['rows']}")
    print(f"train_correct: {train_metrics['correct']}")
    print(f"train_accuracy: {train_metrics['accuracy']:.4f}")
    print(f"holdout_correct: {holdout_metrics['correct']}")
    print(f"holdout_accuracy: {holdout_metrics['accuracy']:.4f}")
    print(f"saved_model_pkl: {save_path}")


if __name__ == "__main__":
    main()
