"""Topology-aware lightweight intent classifier.

This module provides two paths:

  1. Rule-based scoring with topology features
  2. Few-shot centroid fitting on the same feature space

Both paths are zero-dependency and designed for fast experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .text_topology import TextTopologyEngine


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")


@dataclass(frozen=True)
class IntentPrototype:
    name: str
    keywords: tuple[str, ...]
    question_bias: float = 0.0
    imperative_bias: float = 0.0
    topology_bias: float = 0.0


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    scores: dict[str, float]
    matched_keywords: dict[str, list[str]]
    topology_snapshot: dict[str, float]


@dataclass
class LabeledIntentExample:
    text: str
    label: str


class FittedTopologyIntentClassifier:
    """Few-shot centroid classifier over topology-aware text features."""

    def __init__(self, labels: list[str], centroids: dict[str, list[float]], engine: TextTopologyEngine) -> None:
        self.labels = labels
        self.centroids = centroids
        self.engine = engine

    def _feature_vector(self, text: str) -> tuple[list[float], dict[str, float]]:
        raise RuntimeError("feature extraction is delegated by the training-time builder")

    def predict(self, text: str) -> IntentPrediction:
        feat, topo_snapshot = self._feature_vector(text)
        scores: dict[str, float] = {}
        for label in self.labels:
            centroid = self.centroids[label]
            dist2 = sum((a - b) * (a - b) for a, b in zip(feat, centroid))
            scores[label] = -dist2

        best_label = max(scores, key=scores.__getitem__)
        vals = list(scores.values())
        max_score = max(vals)
        exp_vals = [math.exp(v - max_score) for v in vals]
        denom = sum(exp_vals) if exp_vals else 1.0
        confidence = exp_vals[self.labels.index(best_label)] / denom

        return IntentPrediction(
            label=best_label,
            confidence=float(confidence),
            scores=scores,
            matched_keywords={label: [] for label in self.labels},
            topology_snapshot=topo_snapshot,
        )


class TopologyIntentClassifier:
    """Intent classifier using keyword overlap + topology features."""

    def __init__(self, dim: int = 32) -> None:
        self.engine = TextTopologyEngine(dim=dim)
        self.prototypes = (
            IntentPrototype(
                name="explain",
                keywords=("설명", "의미", "개념", "왜", "원리", "어떻게", "해석", "what", "why", "how"),
                question_bias=0.18,
                topology_bias=0.05,
            ),
            IntentPrototype(
                name="implement",
                keywords=("구현", "만들", "작성", "추가", "설계", "코드", "prototype", "build", "implement"),
                imperative_bias=0.26,
                topology_bias=0.08,
            ),
            IntentPrototype(
                name="analyze",
                keywords=("분석", "검토", "정합", "리뷰", "구조", "파악", "analyze", "review", "consistency"),
                topology_bias=0.18,
            ),
            IntentPrototype(
                name="compare",
                keywords=("비교", "추천", "어느", "가장", "뭐가", "해볼만", "versus", "compare", "best"),
                question_bias=0.12,
                topology_bias=0.10,
            ),
            IntentPrototype(
                name="debug",
                keywords=("에러", "오류", "안돼", "고쳐", "수정", "버그", "깨짐", "fail", "fix", "bug"),
                imperative_bias=0.16,
                topology_bias=-0.04,
            ),
        )

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def _keyword_matches(self, text: str, tokens: list[str], keywords: tuple[str, ...]) -> list[str]:
        haystack = text.lower()
        token_set = set(tokens)
        hits: list[str] = []
        for keyword in keywords:
            key = keyword.lower()
            if key in token_set or key in haystack:
                hits.append(keyword)
        return hits

    def _question_score(self, text: str) -> float:
        score = 0.0
        if "?" in text or "？" in text:
            score += 0.35
        lowered = text.lower()
        for marker in ("왜", "무엇", "뭐", "어떻게", "어느", "가능", "인가", "인가요", "할까", "what", "why", "how"):
            if marker in lowered:
                score += 0.08
        return min(score, 0.8)

    def _imperative_score(self, text: str) -> float:
        score = 0.0
        lowered = text.lower()
        for marker in ("해봐", "해줘", "만들어", "구현", "작성", "추가", "고쳐", "수정", "돌려봐", "implement", "build", "fix"):
            if marker in lowered:
                score += 0.12
        return min(score, 0.8)

    def _topology_features(self, text: str) -> dict[str, float]:
        topo = self.engine.analyze(text)
        return {
            "token_density": topo.token_summary.edge_density,
            "sentence_density": topo.sentence_summary.edge_density,
            "paragraph_density": topo.paragraph_summary.edge_density,
            "token_sentence_alignment": topo.token_sentence_alignment,
            "sentence_paragraph_alignment": topo.sentence_paragraph_alignment,
            "bridge_energy": topo.bridge_energy,
            "sentence_components": float(topo.sentence_summary.components),
            "paragraph_components": float(topo.paragraph_summary.components),
        }

    def feature_vector(self, text: str) -> list[float]:
        topo = self._topology_features(text)
        tokens = self._tokenize(text)
        token_count = max(len(tokens), 1)
        q_score = self._question_score(text)
        i_score = self._imperative_score(text)
        keyword_counts: list[float] = []
        rule_scores = self._rule_scores(text, topo=topo, tokens=tokens, q_score=q_score, i_score=i_score)[0]
        for proto in self.prototypes:
            keyword_counts.append(float(len(self._keyword_matches(text, tokens, proto.keywords))))
        return [
            topo["token_density"],
            topo["sentence_density"],
            topo["paragraph_density"],
            topo["token_sentence_alignment"],
            topo["sentence_paragraph_alignment"],
            topo["bridge_energy"],
            topo["sentence_components"],
            topo["paragraph_components"],
            q_score,
            i_score,
            min(token_count / 20.0, 1.0),
            *keyword_counts,
            *(rule_scores[proto.name] for proto in self.prototypes),
        ]

    def feature_snapshot(self, text: str) -> dict[str, float]:
        topo = self._topology_features(text)
        tokens = self._tokenize(text)
        topo["length_score"] = min(max(len(tokens), 1) / 20.0, 1.0)
        return topo

    def _rule_scores(
        self,
        text: str,
        *,
        topo: dict[str, float] | None = None,
        tokens: list[str] | None = None,
        q_score: float | None = None,
        i_score: float | None = None,
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        tokens = tokens if tokens is not None else self._tokenize(text)
        q_score = self._question_score(text) if q_score is None else q_score
        i_score = self._imperative_score(text) if i_score is None else i_score
        topo = self._topology_features(text) if topo is None else topo
        scores: dict[str, float] = {}
        matches: dict[str, list[str]] = {}

        for proto in self.prototypes:
            hits = self._keyword_matches(text, tokens, proto.keywords)
            matches[proto.name] = hits
            keyword_score = 0.22 * len(hits)
            topology_score = proto.topology_bias * (
                topo["sentence_density"]
                + 0.5 * topo["paragraph_density"]
                + 0.25 * topo["bridge_energy"]
            )
            component_penalty = 0.0
            if proto.name == "analyze":
                component_penalty += 0.08 * max(topo["sentence_components"] - 1.0, 0.0)
            if proto.name == "compare":
                component_penalty += 0.04 * max(topo["sentence_components"] - 1.0, 0.0)
            if proto.name == "implement":
                component_penalty -= 0.05 * max(topo["sentence_components"] - 1.0, 0.0)
            if proto.name == "debug":
                component_penalty += 0.06 * max(topo["paragraph_components"] - 1.0, 0.0)

            scores[proto.name] = (
                keyword_score
                + proto.question_bias * q_score
                + proto.imperative_bias * i_score
                + topology_score
                + component_penalty
            )
        return scores, matches

    def fit_centroids(self, examples: list[LabeledIntentExample]) -> FittedTopologyIntentClassifier:
        grouped: dict[str, list[list[float]]] = {}
        for example in examples:
            grouped.setdefault(example.label, []).append(self.feature_vector(example.text))

        labels = sorted(grouped)
        centroids: dict[str, list[float]] = {}
        for label in labels:
            rows = grouped[label]
            width = len(rows[0])
            centroids[label] = [
                sum(row[idx] for row in rows) / len(rows)
                for idx in range(width)
            ]
        model = FittedTopologyIntentClassifier(labels=labels, centroids=centroids, engine=self.engine)
        model._feature_vector = lambda text: (self.feature_vector(text), self.feature_snapshot(text))
        return model

    def predict(self, text: str) -> IntentPrediction:
        topo = self._topology_features(text)
        tokens = self._tokenize(text)
        q_score = self._question_score(text)
        i_score = self._imperative_score(text)
        scores, matches = self._rule_scores(text, topo=topo, tokens=tokens, q_score=q_score, i_score=i_score)

        best_label = max(scores, key=scores.__getitem__)
        values = list(scores.values())
        max_score = max(values)
        exp_values = [math.exp(v - max_score) for v in values]
        denom = sum(exp_values) if exp_values else 1.0
        confidence = exp_values[list(scores.keys()).index(best_label)] / denom

        return IntentPrediction(
            label=best_label,
            confidence=float(confidence),
            scores=scores,
            matched_keywords=matches,
            topology_snapshot=topo,
        )


__all__ = [
    "FittedTopologyIntentClassifier",
    "IntentPrediction",
    "IntentPrototype",
    "LabeledIntentExample",
    "TopologyIntentClassifier",
]
