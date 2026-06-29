"""Tune CE claim residual evidence feature weights on labeled JSONL data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLARUS_ROOT = ROOT / "reality_stone" / "python" / "reality_stone"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CLARUS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLARUS_ROOT))

from examples.pre_eq.claim_residual_benchmark import (  # noqa: E402
    BenchmarkRecord,
    content_tokens,
    extract_entities,
    extract_negations,
    extract_numbers,
    load_jsonl,
    missing_fraction,
    retrieve_evidence_sentences,
    token_f1,
)


@dataclass(frozen=True)
class FeatureRow:
    record_id: str
    y: bool
    full_support_loss: float
    support_loss: float
    novelty: float
    number_mismatch: float
    entity_mismatch: float
    negation_mismatch: float


@dataclass(frozen=True)
class FeatureWeights:
    full_support_loss: float
    support_loss: float
    novelty: float
    number_mismatch: float
    entity_mismatch: float
    negation_mismatch: float


def record_features(record: BenchmarkRecord) -> FeatureRow:
    contexts = tuple(context[:2000] for context in record.contexts)
    claim = record.answer[:3000]
    evidence_sentences = retrieve_evidence_sentences(claim, contexts)
    evidence_text = " ".join(evidence_sentences) if evidence_sentences else " ".join(contexts)
    full_context = " ".join(contexts)
    support = token_f1(content_tokens(claim), content_tokens(evidence_text))
    full_support = token_f1(content_tokens(claim), content_tokens(full_context))
    claim_negations = extract_negations(claim)
    context_negations = extract_negations(full_context)
    negation_mismatch = 1.0 if claim_negations != context_negations and (claim_negations or context_negations) else 0.0
    return FeatureRow(
        record_id=record.record_id,
        y=record.is_hallucinated,
        full_support_loss=(1.0 - full_support) ** 2,
        support_loss=(1.0 - support) ** 2,
        novelty=missing_fraction(content_tokens(claim), content_tokens(evidence_text)) ** 2,
        number_mismatch=missing_fraction(extract_numbers(claim), extract_numbers(full_context)),
        entity_mismatch=missing_fraction(extract_entities(claim), extract_entities(full_context)),
        negation_mismatch=negation_mismatch,
    )


def load_features(path: Path, *, limit: int | None = None) -> tuple[FeatureRow, ...]:
    return tuple(record_features(record) for record in load_jsonl(path, limit=limit))


def score(row: FeatureRow, weights: FeatureWeights) -> float:
    return (
        weights.full_support_loss * row.full_support_loss
        + weights.support_loss * row.support_loss
        + weights.novelty * row.novelty
        + weights.number_mismatch * row.number_mismatch
        + weights.entity_mismatch * row.entity_mismatch
        + weights.negation_mismatch * row.negation_mismatch
    )


def best_threshold(scores: tuple[float, ...], labels: tuple[bool, ...]) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    tp = 0
    fp = 0
    best_t = max(scores) + 1e-9
    best_f1 = 0.0
    for idx, (value, label) in enumerate(ordered):
        if label:
            tp += 1
        else:
            fp += 1
        next_value = ordered[idx + 1][0] if idx + 1 < len(ordered) else value - 1e-9
        if next_value == value:
            continue
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = (value + next_value) / 2.0
    return best_t, best_f1


def random_weights(rng: random.Random) -> FeatureWeights:
    return FeatureWeights(
        full_support_loss=rng.uniform(0.0, 3.0),
        support_loss=rng.uniform(0.0, 3.0),
        novelty=rng.uniform(0.0, 3.0),
        number_mismatch=rng.uniform(0.0, 5.0),
        entity_mismatch=rng.uniform(0.0, 3.0),
        negation_mismatch=rng.uniform(0.0, 5.0),
    )


def tune(features: tuple[FeatureRow, ...], *, trials: int, seed: int) -> tuple[FeatureWeights, float, float]:
    rng = random.Random(seed)
    labels = tuple(row.y for row in features)
    best: tuple[float, FeatureWeights, float] | None = None
    candidates = [
        FeatureWeights(0.0, 0.70, 0.45, 1.25, 0.70, 1.00),
        FeatureWeights(1.60, 0.00, 0.00, 0.00, 0.00, 0.00),
    ]
    candidates.extend(random_weights(rng) for _ in range(trials))
    for weights in candidates:
        scores = tuple(score(row, weights) for row in features)
        threshold, f1 = best_threshold(scores, labels)
        if best is None or f1 > best[0]:
            best = (f1, weights, threshold)
    assert best is not None
    return best[1], best[2], best[0]


def evaluate(features: tuple[FeatureRow, ...], weights: FeatureWeights, threshold: float) -> dict[str, float]:
    scores = tuple(score(row, weights) for row in features)
    labels = tuple(row.y for row in features)
    tp = fp = tn = fn = 0
    for value, label in zip(scores, labels):
        pred = value > threshold
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(features) if features else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--trials", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-weights", type=Path)
    args = parser.parse_args()

    train_features = load_features(args.train, limit=args.train_limit)
    test_features = load_features(args.test)
    weights, threshold, train_f1 = tune(train_features, trials=args.trials, seed=args.seed)
    metrics = evaluate(test_features, weights, threshold)
    print("# CE Claim Residual feature tuning")
    print(f"train_examples {len(train_features)}")
    print(f"test_examples {len(test_features)}")
    print(f"train_best_f1 {train_f1:.6f}")
    print(f"threshold {threshold:.6f}")
    print(f"weights {json.dumps(weights.__dict__, sort_keys=True)}")
    for key, value in metrics.items():
        print(f"test_{key} {value:.6f}")
    if args.output_weights is not None:
        args.output_weights.parent.mkdir(parents=True, exist_ok=True)
        args.output_weights.write_text(
            json.dumps({"threshold": threshold, "weights": weights.__dict__}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
