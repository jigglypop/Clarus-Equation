"""Train a sparse hashed detector for RAGTruth response-level hallucination."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.pre_eq.claim_residual_benchmark import content_tokens, semantic_feature_summary, token_f1


def read_rows(path: Path, *, limit: int | None = None) -> tuple[dict, ...]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return tuple(rows)


def bucket(name: str, size: int) -> str:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return f"h:{int.from_bytes(digest, 'little') % size}"


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def lexical_action(row: dict) -> float:
    support = token_f1(content_tokens(row.get("answer", "")), content_tokens(row.get("context", "")))
    return 1.6 * (1.0 - support) ** 2


@lru_cache(maxsize=50000)
def cached_semantic_feature_summary(answer: str, context: str) -> tuple[tuple[str, float], ...]:
    return tuple(sorted(semantic_feature_summary(answer, (context,)).items()))


def features(row: dict, *, hash_size: int, semantic_features: bool = False) -> dict[str, float]:
    answer = row.get("answer", "")
    context = row.get("context", "")
    feats = {
        "bias": 1.0,
        "lexical_action": lexical_action(row),
        f"model={row.get('model', '')}": 1.0,
        f"task={row.get('task_type', '')}": 1.0,
        f"model_task={row.get('model', '')}::{row.get('task_type', '')}": 1.0,
    }
    if semantic_features:
        semantic_summary = dict(cached_semantic_feature_summary(answer, context))
        feats.update(
            {
                "max_contradiction_score": semantic_summary["max_contradiction_score"],
                "mean_entailment_score": semantic_summary["mean_entailment_score"],
                "neutral_claim_fraction": semantic_summary["neutral_claim_fraction"],
                "unsupported_span_fraction": semantic_summary["unsupported_span_fraction"],
                "reranker_margin": semantic_summary["reranker_margin"],
                "claim_action_p90": semantic_summary["claim_action_p90"],
            }
        )
    answer_tokens = [token for token in re.findall(r"[A-Za-z0-9가-힣_]+", answer.lower()) if len(token) > 2]
    context_tokens = content_tokens(context)
    for token in answer_tokens[:160]:
        feats[bucket(f"answer:{token}", hash_size)] = feats.get(bucket(f"answer:{token}", hash_size), 0.0) + 1.0
        if token not in context_tokens:
            key = bucket(f"novel:{token}", hash_size)
            feats[key] = feats.get(key, 0.0) + 1.0
    scale = max(1.0, math.sqrt(sum(value * value for value in feats.values())))
    return {key: value / scale for key, value in feats.items()}


def dot(weights: dict[str, float], feats: dict[str, float]) -> float:
    return sum(weights.get(key, 0.0) * value for key, value in feats.items())


def train(
    rows: tuple[dict, ...],
    *,
    hash_size: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
    semantic_features: bool = False,
) -> dict[str, float]:
    rng = random.Random(seed)
    weights: dict[str, float] = defaultdict(float)
    data = list(rows)
    for _ in range(epochs):
        rng.shuffle(data)
        for row in data:
            feats = features(row, hash_size=hash_size, semantic_features=semantic_features)
            y = 1.0 if row.get("is_hallucinated") else 0.0
            pred = sigmoid(dot(weights, feats))
            grad = pred - y
            for key, value in feats.items():
                weights[key] = (1.0 - lr * l2) * weights.get(key, 0.0) - lr * grad * value
    return dict(weights)


def scores(
    rows: tuple[dict, ...],
    weights: dict[str, float],
    *,
    hash_size: int,
    semantic_features: bool = False,
) -> tuple[float, ...]:
    return tuple(
        sigmoid(dot(weights, features(row, hash_size=hash_size, semantic_features=semantic_features)))
        for row in rows
    )


def predict_rows(rows: tuple[dict, ...], model: dict) -> tuple[float, ...]:
    return scores(
        rows,
        {str(key): float(value) for key, value in model["weights"].items()},
        hash_size=int(model["hash_size"]),
        semantic_features=bool(model.get("semantic_features", False)),
    )


def save_model(
    path: Path,
    *,
    threshold: float,
    hash_size: int,
    weights: dict[str, float],
    semantic_features: bool = False,
    test_metrics: dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "threshold": threshold,
                "hash_size": hash_size,
                "semantic_features": semantic_features,
                "weights": weights,
                "test_metrics": test_metrics or {},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_model(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": int(data.get("schema_version", 1)),
        "threshold": float(data["threshold"]),
        "hash_size": int(data["hash_size"]),
        "semantic_features": bool(data.get("semantic_features", False)),
        "weights": {str(key): float(value) for key, value in data["weights"].items()},
        "test_metrics": data.get("test_metrics", {}),
    }


def evaluate_model(path: Path, rows: tuple[dict, ...]) -> dict[str, float]:
    model = load_model(path)
    values = predict_rows(rows, model)
    labels = tuple(bool(row.get("is_hallucinated")) for row in rows)
    return metrics(values, labels, float(model["threshold"]))


def best_threshold(values: tuple[float, ...], labels: tuple[bool, ...]) -> tuple[float, float]:
    ordered = sorted(zip(values, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    tp = fp = 0
    best_t = 1.0
    best_f1 = 0.0
    for idx, (value, label) in enumerate(ordered):
        if label:
            tp += 1
        else:
            fp += 1
        next_value = ordered[idx + 1][0] if idx + 1 < len(ordered) else value - 1e-12
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


def metrics(values: tuple[float, ...], labels: tuple[bool, ...], threshold: float) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for value, label in zip(values, labels):
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
        "accuracy": (tp + tn) / len(labels),
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
    parser.add_argument("--train", type=Path)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--hash-size", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--semantic-features", action="store_true")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-model", type=Path)
    args = parser.parse_args()

    if args.model is not None:
        test_rows = read_rows(args.test)
        result = evaluate_model(args.model, test_rows)
        print("# RAGTruth sparse hashed detector")
        print(f"test_examples {len(test_rows)}")
        for key, value in result.items():
            print(f"test_{key} {value:.6f}")
        return 0

    if args.train is None:
        parser.error("--train is required unless --model is provided")

    train_rows = read_rows(args.train, limit=args.train_limit)
    split = max(1, int(len(train_rows) * 0.8))
    fit_rows = train_rows[:split]
    validation_rows = train_rows[split:] or fit_rows
    test_rows = read_rows(args.test)
    weights = train(
        fit_rows,
        hash_size=args.hash_size,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        semantic_features=args.semantic_features,
    )
    validation_scores = scores(
        validation_rows,
        weights,
        hash_size=args.hash_size,
        semantic_features=args.semantic_features,
    )
    validation_labels = tuple(bool(row.get("is_hallucinated")) for row in validation_rows)
    threshold, validation_f1 = best_threshold(validation_scores, validation_labels)
    test_scores = scores(
        test_rows,
        weights,
        hash_size=args.hash_size,
        semantic_features=args.semantic_features,
    )
    test_labels = tuple(bool(row.get("is_hallucinated")) for row in test_rows)
    result = metrics(test_scores, test_labels, threshold)
    print("# RAGTruth hashed detector")
    print(f"fit_examples {len(fit_rows)}")
    print(f"validation_examples {len(validation_rows)}")
    print(f"test_examples {len(test_rows)}")
    print(f"validation_f1 {validation_f1:.6f}")
    print(f"threshold {threshold:.6f}")
    for key, value in result.items():
        print(f"test_{key} {value:.6f}")
    if args.output_model is not None:
        save_model(
            args.output_model,
            threshold=threshold,
            hash_size=args.hash_size,
            weights=weights,
            semantic_features=args.semantic_features,
            test_metrics=result,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
