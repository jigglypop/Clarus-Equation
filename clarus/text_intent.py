"""Topology-aware lightweight intent classifier.

This module builds on ``TextTopologyEngine`` and adds a zero-dependency
intent classifier suitable for fast experiments in Korean/English mixed
developer-assistant traffic.
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
        if "?" in text or "?" in text.replace("？", "?"):
            score += 0.35
        for marker in ("왜", "무엇", "뭐", "어떻게", "어느", "가능", "인가", "인가요", "할까", "what", "why", "how"):
            if marker in text.lower():
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

    def predict(self, text: str) -> IntentPrediction:
        tokens = self._tokenize(text)
        q_score = self._question_score(text)
        i_score = self._imperative_score(text)
        topo = self._topology_features(text)

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

            score = (
                keyword_score
                + proto.question_bias * q_score
                + proto.imperative_bias * i_score
                + topology_score
                + component_penalty
            )
            scores[proto.name] = score

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
    "IntentPrediction",
    "IntentPrototype",
    "TopologyIntentClassifier",
]
