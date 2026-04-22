"""Benchmark the topology-aware intent classifier on standard datasets.

Compares against published SOTA on SNIPS (7 intents) and BANKING77
(77 intents) using the official train/test splits. Both full-shot and
few-shot regimes are reported.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clarus.text_intent import (  # noqa: E402
    IntentPrototype,
    LabeledIntentExample,
    TopologyIntentClassifier,
)
from clarus.text_lexical import LexicalEncoder  # noqa: E402


CACHE_DIR = ROOT / "data" / "hf_cache"


SOTA: dict[str, dict[str, float]] = {
    "snips_full": {"sota": 98.6, "model": "Joint BERT-base (Chen et al. 2019)"},
    "snips_10shot": {"sota": 96.0, "model": "BERT few-shot (approx)"},
    "banking_full": {"sota": 94.42, "model": "DistilRoBERTa fine-tune"},
    "banking_10shot": {"sota": 85.19, "model": "USE+CONVERT 10-shot"},
    "banking_5shot": {"sota": 84.01, "model": "ICDA 5-shot"},
}


def _derive_prototype(label: str) -> IntentPrototype:
    parts = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", label)
    keywords: list[str] = []
    for part in parts:
        piece = part.lower()
        if len(piece) >= 2 and piece not in keywords:
            keywords.append(piece)
    return IntentPrototype(name=label, keywords=tuple(keywords))


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return list(vec)
    inv = 1.0 / norm
    return [x * inv for x in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = 0.0
    for x, y in zip(a, b):
        dot += x * y
    return dot


class HybridCentroidBenchmark:
    """Benchmark-only topology + lexical centroid wrapper.

    This stays inside the benchmark script on purpose: dataset-driven label
    parsing and lexical fusion are evaluation utilities, not part of the
    topology-first core classifier.
    """

    def __init__(
        self,
        base: TopologyIntentClassifier,
        labels: list[str],
        topology_centroids: dict[str, list[float]],
        lexical_encoder: LexicalEncoder | None,
        lexical_centroids: dict[str, list[float]],
        *,
        lexical_alpha: float,
        temperature: float,
        topology_scatter: float,
    ) -> None:
        self.base = base
        self.labels = labels
        self.topology_centroids = topology_centroids
        self.lexical_encoder = lexical_encoder
        self.lexical_centroids = lexical_centroids
        self.lexical_alpha = lexical_alpha if lexical_encoder is not None else 0.0
        self.temperature = temperature
        self.topology_scatter = max(topology_scatter, 1e-6)

    def predict(self, text: str):
        feat = self.base.feature_vector(text)
        topology_scores: dict[str, float] = {}
        for label in self.labels:
            centroid = self.topology_centroids[label]
            dist2 = sum((a - b) * (a - b) for a, b in zip(feat, centroid))
            topology_scores[label] = -dist2 / self.topology_scatter

        scores = topology_scores
        if self.lexical_alpha > 0.0 and self.lexical_encoder is not None:
            lex_vec = self.lexical_encoder.encode(text)
            scores = {}
            for label in self.labels:
                lex_cos = _cosine(lex_vec, self.lexical_centroids[label])
                scores[label] = (
                    self.lexical_alpha * lex_cos
                    + (1.0 - self.lexical_alpha) * topology_scores[label]
                )

        prediction = self.base.predict(text)
        probabilities = {}
        tau = max(self.temperature, 1e-6)
        max_score = max(scores.values())
        exp_values = {
            label: math.exp((value - max_score) / tau)
            for label, value in scores.items()
        }
        denom = sum(exp_values.values()) or 1.0
        probabilities = {label: value / denom for label, value in exp_values.items()}
        best_label = max(scores, key=scores.__getitem__)
        prediction.label = best_label
        prediction.scores = scores
        prediction.probabilities = probabilities
        prediction.confidence = probabilities[best_label]
        return prediction


def _mean_intra_class_distance(
    grouped: dict[str, list[list[float]]],
    centroids: dict[str, list[float]],
) -> float:
    total = 0.0
    counted = 0
    for label, rows in grouped.items():
        centroid = centroids[label]
        for row in rows:
            total += sum((a - b) * (a - b) for a, b in zip(row, centroid))
            counted += 1
    if counted == 0:
        return 1.0
    return total / counted


def _load_snips() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    from datasets import load_dataset

    raw = load_dataset("benayas/snips", cache_dir=str(CACHE_DIR))
    train = [(row["text"], row["category"]) for row in raw["train"]]
    test = [(row["text"], row["category"]) for row in raw["test"]]
    labels = sorted({label for _, label in train})
    return train, test, labels


def _load_banking() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    from datasets import load_dataset

    raw = load_dataset("legacy-datasets/banking77", cache_dir=str(CACHE_DIR))
    label_names = raw["train"].features["label"].names
    train = [(row["text"], label_names[row["label"]]) for row in raw["train"]]
    test = [(row["text"], label_names[row["label"]]) for row in raw["test"]]
    return train, test, list(label_names)


def _subsample_per_label(
    items: list[tuple[str, str]], k: int, seed: int = 0
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for text, label in items:
        grouped[label].append((text, label))
    out: list[tuple[str, str]] = []
    for label, rows in grouped.items():
        rng.shuffle(rows)
        out.extend(rows[:k])
    rng.shuffle(out)
    return out


def _build_classifier(
    labels: list[str], lexical_dim: int, lexical_alpha: float
) -> TopologyIntentClassifier:
    prototypes = tuple(
        _derive_prototype(label) for label in labels
    )
    return TopologyIntentClassifier(
        dim=24,
        prototypes=prototypes,
    )


def _train_and_eval(
    train: list[tuple[str, str]],
    test: list[tuple[str, str]],
    labels: list[str],
    *,
    lexical_dim: int = 0,
    lexical_alpha: float = 0.7,
    seed: int = 0,
    log_prefix: str = "",
) -> dict[str, float]:
    classifier = _build_classifier(labels, lexical_dim, lexical_alpha)
    examples = [LabeledIntentExample(text=text, label=label) for text, label in train]

    t0 = time.perf_counter()
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    grouped_texts: dict[str, list[str]] = defaultdict(list)
    for example in examples:
        grouped[example.label].append(classifier.feature_vector(example.text))
        grouped_texts[example.label].append(example.text)

    topology_centroids: dict[str, list[float]] = {}
    for label in sorted(grouped):
        rows = grouped[label]
        width = len(rows[0])
        topology_centroids[label] = [
            sum(row[idx] for row in rows) / len(rows) for idx in range(width)
        ]
    topology_scatter = max(
        _mean_intra_class_distance(grouped, topology_centroids),
        1e-3,
    )

    lexical_encoder: LexicalEncoder | None = None
    lexical_centroids: dict[str, list[float]] = {}
    if lexical_dim > 0 and lexical_alpha > 0.0:
        lexical_encoder = LexicalEncoder(dim=lexical_dim, ngram_range=(1, 2))
        lexical_encoder.fit(text for text, _ in train)
        for label in sorted(grouped_texts):
            vectors = [lexical_encoder.encode(text) for text in grouped_texts[label]]
            width = len(vectors[0])
            mean_vec = [
                sum(v[idx] for v in vectors) / len(vectors) for idx in range(width)
            ]
            lexical_centroids[label] = _l2_normalise(mean_vec)

    if lexical_encoder is not None:
        temperature = max(0.05, topology_scatter / max(topology_scatter + 1.0, 1.0))
    else:
        temperature = topology_scatter

    fitted = HybridCentroidBenchmark(
        classifier,
        labels=sorted(grouped),
        topology_centroids=topology_centroids,
        lexical_encoder=lexical_encoder,
        lexical_centroids=lexical_centroids,
        lexical_alpha=lexical_alpha,
        temperature=temperature,
        topology_scatter=topology_scatter,
    )
    fit_time = time.perf_counter() - t0

    correct = 0
    confidences: list[float] = []
    t0 = time.perf_counter()
    for text, true_label in test:
        prediction = fitted.predict(text)
        confidences.append(prediction.confidence)
        if prediction.label == true_label:
            correct += 1
    eval_time = time.perf_counter() - t0

    accuracy = correct / max(len(test), 1)
    return {
        "accuracy_pct": accuracy * 100.0,
        "n_train": float(len(train)),
        "n_test": float(len(test)),
        "n_labels": float(len(labels)),
        "fit_time_s": fit_time,
        "eval_time_s": eval_time,
        "predict_us_per_call": eval_time / max(len(test), 1) * 1e6,
        "confidence_mean": statistics.mean(confidences) if confidences else 0.0,
        "confidence_median": statistics.median(confidences) if confidences else 0.0,
        "temperature": fitted.temperature,
    }


def _print_result(name: str, result: dict[str, float]) -> None:
    sota = SOTA.get(name, {})
    sota_acc = sota.get("sota")
    sota_model = sota.get("model", "n/a")
    print(f"== {name} ==")
    print(
        f"  labels={int(result['n_labels'])} "
        f"train={int(result['n_train'])} "
        f"test={int(result['n_test'])}"
    )
    print(f"  accuracy           {result['accuracy_pct']:6.2f} %")
    if sota_acc is not None:
        delta = result["accuracy_pct"] - sota_acc
        print(f"  SOTA reference     {sota_acc:6.2f} % ({sota_model})")
        print(f"  delta vs SOTA      {delta:+6.2f} pp")
    print(f"  confidence mean    {result['confidence_mean']:.3f}")
    print(f"  confidence median  {result['confidence_median']:.3f}")
    print(f"  fit time           {result['fit_time_s']:8.2f} s")
    print(
        f"  predict throughput {1e6 / result['predict_us_per_call']:8.1f} call/s "
        f"({result['predict_us_per_call']:.1f} us/call)"
    )
    print(f"  temperature        {result['temperature']:.4f}")
    print("")


def _run(
    name: str,
    train: list[tuple[str, str]],
    test: list[tuple[str, str]],
    labels: list[str],
    *,
    lexical_dim: int,
    lexical_alpha: float,
) -> None:
    result = _train_and_eval(
        train, test, labels, lexical_dim=lexical_dim, lexical_alpha=lexical_alpha
    )
    _print_result(name, result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Standard intent benchmark")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["snips", "banking"],
        choices=["snips", "banking"],
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["full", "10shot", "5shot"],
        choices=["full", "10shot", "5shot"],
    )
    parser.add_argument(
        "--lexical-dim",
        type=int,
        default=0,
        help="Hashing TF-IDF dimension (0 disables the lexical channel).",
    )
    parser.add_argument(
        "--lexical-alpha",
        type=float,
        default=0.7,
        help="Convex weight on the lexical cosine score in the hybrid combination.",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR))

    if "snips" in args.datasets:
        train, test, labels = _load_snips()
        if "full" in args.regimes:
            _run("snips_full", train, test, labels,
                 lexical_dim=args.lexical_dim, lexical_alpha=args.lexical_alpha)
        if "10shot" in args.regimes:
            sub = _subsample_per_label(train, k=10)
            _run("snips_10shot", sub, test, labels,
                 lexical_dim=args.lexical_dim, lexical_alpha=args.lexical_alpha)
        if "5shot" in args.regimes:
            sub = _subsample_per_label(train, k=5)
            _run("snips_5shot", sub, test, labels,
                 lexical_dim=args.lexical_dim, lexical_alpha=args.lexical_alpha)

    if "banking" in args.datasets:
        train, test, labels = _load_banking()
        if "full" in args.regimes:
            _run("banking_full", train, test, labels,
                 lexical_dim=args.lexical_dim, lexical_alpha=args.lexical_alpha)
        if "10shot" in args.regimes:
            sub = _subsample_per_label(train, k=10)
            _run("banking_10shot", sub, test, labels,
                 lexical_dim=args.lexical_dim, lexical_alpha=args.lexical_alpha)
        if "5shot" in args.regimes:
            sub = _subsample_per_label(train, k=5)
            _run("banking_5shot", sub, test, labels,
                 lexical_dim=args.lexical_dim, lexical_alpha=args.lexical_alpha)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
