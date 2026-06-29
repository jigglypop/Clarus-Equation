"""Blend response-level hash scores with claim/span detector scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.pre_eq.train_ragtruth_claim_detector import (  # noqa: E402
    best_threshold,
    load_model as load_claim_model,
    metrics,
    predict_response_scores,
    read_rows,
)
from examples.pre_eq.train_ragtruth_hash_detector import (  # noqa: E402
    load_model as load_hash_model,
    predict_rows as predict_hash_rows,
)


def split_fit_validation(rows: tuple[dict[str, Any], ...], validation_fraction: float) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    split = max(1, int(len(rows) * (1.0 - validation_fraction)))
    return rows[:split], rows[split:] or rows[:split]


def response_hash_scores(rows: tuple[dict[str, Any], ...], model: dict[str, Any]) -> tuple[tuple[str, float, bool], ...]:
    values = predict_hash_rows(rows, model)
    return tuple(
        (
            str(row.get("id", idx)),
            value,
            bool(row.get("is_hallucinated")),
        )
        for idx, (row, value) in enumerate(zip(rows, values))
    )


def align_scores(
    hash_scores: tuple[tuple[str, float, bool], ...],
    claim_scores: tuple[tuple[str, float, bool], ...],
) -> tuple[tuple[str, float, float, bool], ...]:
    claim_by_id = {row_id: (score, label) for row_id, score, label in claim_scores}
    aligned = []
    for row_id, hash_score, label in hash_scores:
        claim_score, claim_label = claim_by_id[row_id]
        aligned.append((row_id, hash_score, claim_score, label or claim_label))
    return tuple(aligned)


def blend_score(hash_score: float, claim_score: float, alpha: float) -> float:
    return alpha * hash_score + (1.0 - alpha) * claim_score


def tune_alpha(aligned: tuple[tuple[str, float, float, bool], ...]) -> tuple[float, float, float]:
    best = (0.0, 0.0, 0.0)
    for idx in range(21):
        alpha = idx / 20.0
        values = tuple(blend_score(hash_score, claim_score, alpha) for _, hash_score, claim_score, _ in aligned)
        labels = tuple(label for _, _, _, label in aligned)
        threshold, f1 = best_threshold(values, labels)
        if f1 > best[2]:
            best = (alpha, threshold, f1)
    return best


def evaluate(aligned: tuple[tuple[str, float, float, bool], ...], *, alpha: float, threshold: float) -> dict[str, float]:
    values = tuple(blend_score(hash_score, claim_score, alpha) for _, hash_score, claim_score, _ in aligned)
    labels = tuple(label for _, _, _, label in aligned)
    return metrics(values, labels, threshold)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--hash-model", type=Path, required=True)
    parser.add_argument("--claim-model", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    args = parser.parse_args()

    _, validation_rows = split_fit_validation(read_rows(args.train), args.validation_fraction)
    test_rows = read_rows(args.test)
    hash_model = load_hash_model(args.hash_model)
    claim_model = load_claim_model(args.claim_model)

    validation = align_scores(
        response_hash_scores(validation_rows, hash_model),
        predict_response_scores(validation_rows, claim_model),
    )
    alpha, threshold, validation_f1 = tune_alpha(validation)
    test = align_scores(
        response_hash_scores(test_rows, hash_model),
        predict_response_scores(test_rows, claim_model),
    )
    result = evaluate(test, alpha=alpha, threshold=threshold)
    print("# RAGTruth hash + claim/span ensemble")
    print(f"validation_examples {len(validation)}")
    print(f"test_examples {len(test)}")
    print(f"alpha {alpha:.6f}")
    print(f"threshold {threshold:.6f}")
    print(f"validation_f1 {validation_f1:.6f}")
    print(f"weights {json.dumps({'hash': alpha, 'claim': 1.0 - alpha}, sort_keys=True)}")
    for key, value in result.items():
        print(f"test_{key} {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
