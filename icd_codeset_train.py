"""Build a searchable ICD codebook model from ICDCodeSet.csv."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "in",
    "for",
    "with",
    "due",
    "other",
    "unspecified",
}


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ICDEntry:
    code: str
    description: str


class ICDCodebookModel:
    """Simple lexical retrieval model for ICD descriptions."""

    def __init__(self) -> None:
        self.entries: list[ICDEntry] = []
        self.code_to_desc: dict[str, str] = {}
        self.token_to_codes: dict[str, Counter[str]] = defaultdict(Counter)
        self.code_token_counts: dict[str, Counter[str]] = {}
        self.doc_freq: Counter[str] = Counter()
        self.num_docs = 0

    def fit(self, entries: list[ICDEntry]) -> None:
        self.entries = entries
        self.code_to_desc = {e.code: e.description for e in entries}
        self.num_docs = len(entries)
        for entry in entries:
            tokens = [t for t in _tokenize(entry.description) if t not in STOPWORDS]
            token_counts = Counter(tokens)
            self.code_token_counts[entry.code] = token_counts
            for token in token_counts:
                self.doc_freq[token] += 1
            for token, count in token_counts.items():
                self.token_to_codes[token][entry.code] += count

    def _idf(self, token: str) -> float:
        # Smoothed IDF.
        return math.log((1 + self.num_docs) / (1 + self.doc_freq[token])) + 1.0

    def search(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        query_tokens = [t for t in _tokenize(text) if t not in STOPWORDS]
        if not query_tokens:
            return []
        query_counts = Counter(query_tokens)
        scores: Counter[str] = Counter()
        for token, q_count in query_counts.items():
            idf = self._idf(token)
            for code, tf in self.token_to_codes.get(token, {}).items():
                scores[code] += (q_count * idf) * (tf * idf)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return [(code, float(score)) for code, score in ranked[:top_k]]


def load_codeset(csv_path: Path) -> list[ICDEntry]:
    entries: list[ICDEntry] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row["ICDCode"].strip().upper()
            desc = row["Description"].strip()
            if code and desc:
                entries.append(ICDEntry(code=code, description=desc))
    return entries


def save_model(model: ICDCodebookModel, pkl_path: Path) -> None:
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as f:
        pickle.dump(model, f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/ICDCodeSet.csv", help="Path to ICDCodeSet.csv")
    parser.add_argument(
        "--save-pkl",
        default="outputs/icd_codeset_model.pkl",
        help="Where to save the trained ICD codebook model.",
    )
    parser.add_argument(
        "--sample-query",
        default="cholera due to vibrio",
        help="Optional query to test retrieval output.",
    )
    args = parser.parse_args()

    entries = load_codeset(Path(args.csv))
    model = ICDCodebookModel()
    model.fit(entries)
    save_model(model, Path(args.save_pkl))

    print("ICD codebook training complete")
    print(f"dataset: {args.csv}")
    print(f"rows_total: {len(entries)}")
    print(f"saved_model_pkl: {args.save_pkl}")

    if args.sample_query:
        results = model.search(args.sample_query, top_k=5)
        print(f"sample_query: {args.sample_query}")
        for i, (code, score) in enumerate(results, start=1):
            print(f"top_{i}: code={code} score={score:.2f} desc={model.code_to_desc[code]}")


if __name__ == "__main__":
    main()
