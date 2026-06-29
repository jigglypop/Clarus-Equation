"""Train a lightweight supervised RAGTruth detector over CE residual features."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CLARUS_ROOT = ROOT / "reality_stone" / "python" / "reality_stone"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CLARUS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLARUS_ROOT))

from examples.pre_eq.claim_residual_benchmark import content_tokens, load_jsonl, token_f1  # noqa: E402


@dataclass(frozen=True)
class Row:
    y: bool
    lexical_action: float
    model: str
    task_type: str
    model_task: str


@dataclass(frozen=True)
class Weights:
    lexical: float
    model_prior: float
    task_prior: float
    model_task_prior: float
    bias: float


def rows(path: Path, *, limit: int | None = None) -> tuple[Row, ...]:
    out = []
    for record in load_jsonl(path, limit=limit):
        context = " ".join(record.contexts)
        support = token_f1(content_tokens(record.answer), content_tokens(context))
        out.append(
            Row(
                y=record.is_hallucinated,
                lexical_action=1.6 * (1.0 - support) ** 2,
                model=getattr(record, "model", ""),
                task_type=getattr(record, "task_type", ""),
                model_task=f"{getattr(record, 'model', '')}::{getattr(record, 'task_type', '')}",
            )
        )
    return tuple(out)


def raw_rows(path: Path, *, limit: int | None = None) -> tuple[dict, ...]:
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return tuple(out)


def build_rows(path: Path, *, limit: int | None = None) -> tuple[Row, ...]:
    out = []
    for item in raw_rows(path, limit=limit):
        support = token_f1(content_tokens(item.get("answer", "")), content_tokens(item.get("context", "")))
        out.append(
            Row(
                y=bool(item.get("is_hallucinated")),
                lexical_action=1.6 * (1.0 - support) ** 2,
                model=str(item.get("model", "")),
                task_type=str(item.get("task_type", "")),
                model_task=f"{item.get('model', '')}::{item.get('task_type', '')}",
            )
        )
    return tuple(out)


def category_rates(train: tuple[Row, ...], attr: str) -> dict[str, float]:
    counts: dict[str, Counter[bool]] = defaultdict(Counter)
    global_rate = sum(row.y for row in train) / len(train)
    for row in train:
        counts[getattr(row, attr)][row.y] += 1
    rates = {}
    for key, counter in counts.items():
        total = counter[True] + counter[False]
        rates[key] = (counter[True] + 2.0 * global_rate) / (total + 2.0)
    return rates


def score(
    row: Row,
    weights: Weights,
    model_rates: dict[str, float],
    task_rates: dict[str, float],
    model_task_rates: dict[str, float],
    global_rate: float,
) -> float:
    return (
        weights.bias
        + weights.lexical * row.lexical_action
        + weights.model_prior * model_rates.get(row.model, global_rate)
        + weights.task_prior * task_rates.get(row.task_type, global_rate)
        + weights.model_task_prior * model_task_rates.get(row.model_task, global_rate)
    )


def metrics(scores: tuple[float, ...], labels: tuple[bool, ...], threshold: float) -> dict[str, float]:
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


def metrics_from_predictions(predictions: tuple[bool, ...], labels: tuple[bool, ...]) -> dict[str, float]:
    tp = fp = tn = fn = 0
    for pred, label in zip(predictions, labels):
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


def best_threshold(scores: tuple[float, ...], labels: tuple[bool, ...]) -> tuple[float, float]:
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(labels)
    tp = 0
    fp = 0
    best_f1 = 0.0
    best_t = max(scores) + 1e-9 if scores else 0.0
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


def calibrate_task_thresholds(
    scores: tuple[float, ...],
    rows: tuple[Row, ...],
    *,
    global_threshold: float,
    min_examples: int = 3,
) -> dict[str, float]:
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for value, row in zip(scores, rows):
        grouped[row.task_type].append((value, row.y))
    thresholds = {}
    for task_type, items in grouped.items():
        labels = tuple(label for _, label in items)
        if len(items) < min_examples or len(set(labels)) < 2:
            thresholds[task_type] = global_threshold
            continue
        task_scores = tuple(value for value, _ in items)
        thresholds[task_type] = best_threshold(task_scores, labels)[0]
    return thresholds


def threshold_for_row(row: Row, thresholds: dict[str, float], global_threshold: float) -> float:
    return thresholds.get(row.task_type, global_threshold)


def metrics_with_task_thresholds(
    scores: tuple[float, ...],
    rows: tuple[Row, ...],
    *,
    task_thresholds: dict[str, float],
    global_threshold: float,
) -> dict[str, float]:
    predictions = tuple(
        value > threshold_for_row(row, task_thresholds, global_threshold)
        for value, row in zip(scores, rows)
    )
    labels = tuple(row.y for row in rows)
    return metrics_from_predictions(predictions, labels)


def random_weights(rng: random.Random) -> Weights:
    return Weights(
        lexical=rng.uniform(0.0, 3.0),
        model_prior=rng.uniform(-2.0, 4.0),
        task_prior=rng.uniform(-2.0, 4.0),
        model_task_prior=rng.uniform(-2.0, 4.0),
        bias=rng.uniform(-2.0, 1.0),
    )


def split_fit_validation(
    train: tuple[Row, ...],
    *,
    rng: random.Random,
    validation_fraction: float,
) -> tuple[tuple[Row, ...], tuple[Row, ...]]:
    shuffled = list(train)
    rng.shuffle(shuffled)
    split_at = max(1, int(len(shuffled) * (1.0 - validation_fraction)))
    fit = tuple(shuffled[:split_at])
    validation = tuple(shuffled[split_at:]) or fit
    return fit, validation


def tune(
    train: tuple[Row, ...],
    *,
    trials: int,
    seed: int,
    validation_fraction: float = 0.2,
) -> tuple[
    Weights,
    float,
    dict[str, float],
    dict[str, float],
    dict[str, float],
    float,
]:
    rng = random.Random(seed)
    fit, validation = split_fit_validation(
        train,
        rng=rng,
        validation_fraction=validation_fraction,
    )
    fit_labels = tuple(row.y for row in fit)
    validation_labels = tuple(row.y for row in validation)
    global_rate = sum(fit_labels) / len(fit_labels)
    model_rates = category_rates(fit, "model")
    task_rates = category_rates(fit, "task_type")
    model_task_rates = category_rates(fit, "model_task")
    best: tuple[float, Weights, float] | None = None
    for _ in range(trials):
        weights = random_weights(rng)
        fit_scores = tuple(
            score(row, weights, model_rates, task_rates, model_task_rates, global_rate)
            for row in fit
        )
        threshold, _ = best_threshold(fit_scores, fit_labels)
        validation_scores = tuple(
            score(row, weights, model_rates, task_rates, model_task_rates, global_rate)
            for row in validation
        )
        f1 = metrics(validation_scores, validation_labels, threshold)["f1"]
        if best is None or f1 > best[0]:
            best = (f1, weights, threshold)
    assert best is not None
    return best[1], best[2], model_rates, task_rates, model_task_rates, global_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--task-thresholds", action="store_true")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--output-model", type=Path)
    args = parser.parse_args()

    train = build_rows(args.train, limit=args.train_limit)
    test = build_rows(args.test)
    weights, threshold, model_rates, task_rates, model_task_rates, global_rate = tune(
        train,
        trials=args.trials,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )
    test_scores = tuple(
        score(row, weights, model_rates, task_rates, model_task_rates, global_rate)
        for row in test
    )
    test_labels = tuple(row.y for row in test)
    task_thresholds: dict[str, float] = {}
    if args.task_thresholds:
        fit, validation = split_fit_validation(
            train,
            rng=random.Random(args.seed),
            validation_fraction=args.validation_fraction,
        )
        validation_scores = tuple(
            score(row, weights, model_rates, task_rates, model_task_rates, global_rate)
            for row in validation
        )
        task_thresholds = calibrate_task_thresholds(
            validation_scores,
            validation,
            global_threshold=threshold,
        )
        result = metrics_with_task_thresholds(
            test_scores,
            test,
            task_thresholds=task_thresholds,
            global_threshold=threshold,
        )
    else:
        result = metrics(test_scores, test_labels, threshold)
    print("# RAGTruth supervised residual detector")
    print(f"train_examples {len(train)}")
    print(f"test_examples {len(test)}")
    print(f"threshold {threshold:.6f}")
    if task_thresholds:
        print(f"task_thresholds {json.dumps(task_thresholds, sort_keys=True)}")
    print(f"weights {json.dumps(weights.__dict__, sort_keys=True)}")
    for key, value in result.items():
        print(f"test_{key} {value:.6f}")
    if args.output_model is not None:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        args.output_model.write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "task_thresholds": task_thresholds,
                    "weights": weights.__dict__,
                    "model_rates": model_rates,
                    "task_rates": task_rates,
                    "model_task_rates": model_task_rates,
                    "global_rate": global_rate,
                    "test_metrics": result,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
