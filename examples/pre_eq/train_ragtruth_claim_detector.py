"""Train a claim/span-supervised detector from RAGTruth labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.pre_eq.claim_residual_benchmark import (  # noqa: E402
    content_tokens,
    deterministic_nli_scores,
    evidence_candidates,
    extract_entities,
    extract_numbers,
    semantic_claim_action,
    split_claims,
    token_f1,
)


@dataclass(frozen=True)
class ClaimExample:
    row_id: str
    claim_index: int
    answer_label: bool
    claim_label: bool
    model: str
    task_type: str
    features: dict[str, float]


def read_rows(path: Path, *, limit: int | None = None) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
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


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def dot(weights: dict[str, float], features: Mapping[str, float]) -> float:
    return sum(weights.get(key, 0.0) * value for key, value in features.items())


def overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def label_spans(row: Mapping[str, Any]) -> tuple[tuple[int, int, str], ...]:
    spans = []
    for label in row.get("labels", ()):
        if not isinstance(label, Mapping):
            continue
        start = label.get("start")
        end = label.get("end")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append((start, end, str(label.get("label_type", ""))))
    return tuple(spans)


def claim_offsets(answer: str, claims: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    offsets = []
    cursor = 0
    for claim in claims:
        start = answer.find(claim, cursor)
        if start < 0:
            start = answer.find(claim)
        if start < 0:
            offsets.append((-1, -1))
            continue
        end = start + len(claim)
        offsets.append((start, end))
        cursor = end
    return tuple(offsets)


def claim_is_labeled(start: int, end: int, spans: tuple[tuple[int, int, str], ...]) -> bool:
    if start < 0 or end <= start:
        return False
    return any(overlap(start, end, span_start, span_end) for span_start, span_end, _ in spans)


def claim_features(claim: str, context: str, row: Mapping[str, Any], *, hash_size: int) -> dict[str, float]:
    claim_tokens = content_tokens(claim)
    context_tokens = content_tokens(context)
    support = token_f1(claim_tokens, context_tokens)
    action, accepted = semantic_claim_action(claim, (context,))
    candidates = evidence_candidates(claim, (context,))
    top = candidates[0] if candidates else None
    nli = top.nli if top is not None else deterministic_nli_scores(claim, context)
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    feats: dict[str, float] = {
        "bias": 1.0,
        "lexical_action": 1.6 * (1.0 - support) ** 2,
        "semantic_action": action,
        "semantic_rejected": 1.0 - accepted,
        "entailment": nli.entailment,
        "contradiction": nli.contradiction,
        "neutral": nli.neutral,
        "reranker_margin": max(0.0, (top.score if top else 0.0) - second_score),
        "entity_count": float(len(extract_entities(claim))),
        "number_count": float(len(extract_numbers(claim))),
        f"model={row.get('model', '')}": 1.0,
        f"task={row.get('task_type', '')}": 1.0,
    }
    for token in re.findall(r"[A-Za-z0-9가-힣_]+", claim.lower())[:80]:
        if len(token) <= 2:
            continue
        feats[bucket(f"claim:{token}", hash_size)] = feats.get(bucket(f"claim:{token}", hash_size), 0.0) + 1.0
        if token not in context_tokens:
            key = bucket(f"claim_novel:{token}", hash_size)
            feats[key] = feats.get(key, 0.0) + 1.0
    scale = max(1.0, math.sqrt(sum(value * value for value in feats.values())))
    return {key: value / scale for key, value in feats.items()}


def build_claim_examples(rows: tuple[dict[str, Any], ...], *, hash_size: int) -> tuple[ClaimExample, ...]:
    examples = []
    for row in rows:
        answer = str(row.get("answer", ""))
        context = str(row.get("context", ""))
        claims = split_claims(answer)
        offsets = claim_offsets(answer, claims)
        spans = label_spans(row)
        answer_label = bool(row.get("is_hallucinated"))
        for idx, (claim, (start, end)) in enumerate(zip(claims, offsets)):
            examples.append(
                ClaimExample(
                    row_id=str(row.get("id", "")),
                    claim_index=idx,
                    answer_label=answer_label,
                    claim_label=claim_is_labeled(start, end, spans),
                    model=str(row.get("model", "")),
                    task_type=str(row.get("task_type", "")),
                    features=claim_features(claim, context, row, hash_size=hash_size),
                )
            )
    return tuple(examples)


def train(
    examples: tuple[ClaimExample, ...],
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    data = list(examples)
    weights: dict[str, float] = defaultdict(float)
    for _ in range(epochs):
        rng.shuffle(data)
        for example in data:
            y = 1.0 if example.claim_label else 0.0
            pred = sigmoid(dot(weights, example.features))
            grad = pred - y
            for key, value in example.features.items():
                weights[key] = (1.0 - lr * l2) * weights.get(key, 0.0) - lr * grad * value
    return dict(weights)


def claim_scores(examples: tuple[ClaimExample, ...], weights: dict[str, float]) -> tuple[float, ...]:
    return tuple(sigmoid(dot(weights, example.features)) for example in examples)


def aggregate_response_scores(
    examples: tuple[ClaimExample, ...],
    scores: tuple[float, ...],
) -> tuple[tuple[str, float, bool], ...]:
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for example, score in zip(examples, scores):
        grouped[example.row_id].append((score, example.answer_label))
    response_scores = []
    for row_id, items in grouped.items():
        max_score = max(score for score, _ in items)
        label = any(label for _, label in items)
        response_scores.append((row_id, max_score, label))
    return tuple(response_scores)


def best_threshold(values: tuple[float, ...], labels: tuple[bool, ...]) -> tuple[float, float]:
    ordered = sorted(zip(values, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    tp = fp = 0
    best_t = max(values) + 1e-9 if values else 0.0
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
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
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
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def save_model(path: Path, *, threshold: float, hash_size: int, weights: dict[str, float], result: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "threshold": threshold,
                "hash_size": hash_size,
                "weights": weights,
                "test_metrics": result,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def load_model(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": int(data.get("schema_version", 1)),
        "threshold": float(data["threshold"]),
        "hash_size": int(data["hash_size"]),
        "weights": {str(key): float(value) for key, value in data["weights"].items()},
        "test_metrics": data.get("test_metrics", {}),
    }


def predict_response_scores(rows: tuple[dict[str, Any], ...], model: Mapping[str, Any]) -> tuple[tuple[str, float, bool], ...]:
    examples = build_claim_examples(rows, hash_size=int(model["hash_size"]))
    scores = claim_scores(examples, {str(key): float(value) for key, value in model["weights"].items()})
    return aggregate_response_scores(examples, scores)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--hash-size", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-model", type=Path)
    args = parser.parse_args()

    train_rows = read_rows(args.train)
    split = max(1, int(len(train_rows) * 0.8))
    fit_examples = build_claim_examples(train_rows[:split], hash_size=args.hash_size)
    validation_examples = build_claim_examples(train_rows[split:], hash_size=args.hash_size)
    test_examples = build_claim_examples(read_rows(args.test), hash_size=args.hash_size)
    weights = train(
        fit_examples,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )
    validation_responses = aggregate_response_scores(
        validation_examples,
        claim_scores(validation_examples, weights),
    )
    threshold, validation_f1 = best_threshold(
        tuple(score for _, score, _ in validation_responses),
        tuple(label for _, _, label in validation_responses),
    )
    test_responses = aggregate_response_scores(test_examples, claim_scores(test_examples, weights))
    result = metrics(
        tuple(score for _, score, _ in test_responses),
        tuple(label for _, _, label in test_responses),
        threshold,
    )
    print("# RAGTruth claim/span detector")
    print(f"fit_claims {len(fit_examples)}")
    print(f"validation_claims {len(validation_examples)}")
    print(f"test_claims {len(test_examples)}")
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
            result=result,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
