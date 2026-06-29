"""JSONL benchmark adapter for the CE Claim Residual Verifier.

The adapter intentionally starts with a transparent lexical evidence mapper.
It is not the final SOTA path; it provides the stable measurement harness needed
before swapping in stronger retrievers, NLI models, or span classifiers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CLARUS_ROOT = ROOT / "reality_stone" / "python" / "reality_stone"
if str(CLARUS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLARUS_ROOT))

from clarus.llm_pre_eq import (  # noqa: E402
    ClaimAxisEvidence,
    ClaimResidualVerifier,
    ResidualAnswerCandidate,
    ResidualClaim,
)


ANSWER_KEYS = ("answer", "response", "output", "summary", "generation", "model_output")
CONTEXT_KEYS = (
    "context",
    "contexts",
    "evidence",
    "evidences",
    "reference",
    "references",
    "document",
    "documents",
    "source",
    "sources",
)
LABEL_KEYS = (
    "is_hallucinated",
    "hallucinated",
    "has_hallucination",
    "label",
    "labels",
    "factuality",
    "faithfulness",
)
HALLUCINATED_LABELS = {
    "hallucinated",
    "unsupported",
    "contradicted",
    "false",
    "unfaithful",
    "inconsistent",
    "not_supported",
    "incorrect",
    "1",
}
SUPPORTED_LABELS = {
    "supported",
    "faithful",
    "true",
    "consistent",
    "not_hallucinated",
    "correct",
    "0",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
}
NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "none",
    "without",
    "neither",
    "nor",
    "cannot",
    "can't",
    "won't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "doesn't",
    "don't",
    "didn't",
}


@dataclass(frozen=True)
class BenchmarkRecord:
    answer: str
    contexts: tuple[str, ...]
    is_hallucinated: bool
    record_id: str
    model: str = ""
    task_type: str = ""
    labels: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class BinaryMetrics:
    total: int
    predicted_hallucinated: int
    actual_hallucinated: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denom if denom else 0.0

    @property
    def specificity(self) -> float:
        denom = self.true_negative + self.false_positive
        return self.true_negative / denom if denom else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.recall + self.specificity)


@dataclass(frozen=True)
class BenchmarkPrediction:
    record_id: str
    score: float
    action: float
    accepted_fraction: float
    predicted_hallucinated: bool
    actual_hallucinated: bool


@dataclass(frozen=True)
class ClaimPrediction:
    record_id: str
    claim_index: int
    claim: str
    action: float
    accepted: bool
    evidence: str = ""
    entailment: float = 0.0
    contradiction: float = 0.0
    neutral: float = 1.0
    span_start: int | None = None
    span_end: int | None = None
    span_label: str = ""


@dataclass(frozen=True)
class NliScores:
    entailment: float
    contradiction: float
    neutral: float


@dataclass(frozen=True)
class EvidenceCandidate:
    sentence_index: int
    text: str
    lexical_score: float
    entity_coverage: float
    number_coverage: float
    nli: NliScores
    score: float


@dataclass(frozen=True)
class SpanMetrics:
    predicted_positive: int
    actual_positive: int
    true_positive: int

    @property
    def precision(self) -> float:
        return self.true_positive / self.predicted_positive if self.predicted_positive else 0.0

    @property
    def recall(self) -> float:
        return self.true_positive / self.actual_positive if self.actual_positive else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denom if denom else 0.0


NliScoreMap = dict[tuple[str, int], NliScores]


@dataclass(frozen=True)
class CalibrationResult:
    action_threshold: float
    accepted_fraction_threshold: float
    metrics: BinaryMetrics
    auroc: float
    auprc: float


def _first_present(record: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _flatten_text(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        parts: list[str] = []
        for nested in value.values():
            parts.extend(_flatten_text(nested))
        return tuple(parts)
    if isinstance(value, Iterable):
        parts = []
        for item in value:
            parts.extend(_flatten_text(item))
        return tuple(parts)
    return (str(value),)


def _label_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return bool(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        values = [_label_to_bool(item) for item in value]
        return any(values)
    label = str(value).strip().lower()
    if label in HALLUCINATED_LABELS:
        return True
    if label in SUPPORTED_LABELS:
        return False
    raise ValueError(f"unknown hallucination label: {value!r}")


def parse_record(record: Mapping[str, Any], idx: int) -> BenchmarkRecord:
    answer = _first_present(record, ANSWER_KEYS)
    context = _first_present(record, CONTEXT_KEYS)
    label = _first_present(record, LABEL_KEYS)
    if answer is None:
        raise ValueError(f"record {idx} has no answer field")
    if context is None:
        raise ValueError(f"record {idx} has no context/evidence field")
    if label is None:
        raise ValueError(f"record {idx} has no hallucination label field")
    record_id = str(record.get("id", record.get("record_id", idx)))
    return BenchmarkRecord(
        answer=" ".join(_flatten_text(answer)).strip(),
        contexts=tuple(text.strip() for text in _flatten_text(context) if text.strip()),
        is_hallucinated=_label_to_bool(label),
        record_id=record_id,
        model=str(record.get("model", "")),
        task_type=str(record.get("task_type", "")),
        labels=tuple(
            label_item
            for label_item in record.get("labels", ())
            if isinstance(label_item, Mapping)
        ),
    )


def load_jsonl(path: Path, *, limit: int | None = None) -> tuple[BenchmarkRecord, ...]:
    records: list[BenchmarkRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            records.append(parse_record(json.loads(line), idx))
            if limit is not None and len(records) >= limit:
                break
    return tuple(records)


@lru_cache(maxsize=20000)
def tokenize(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[A-Za-z0-9가-힣_]+", text.lower())
        if len(token) > 1
    )


def split_claims(answer: str) -> tuple[str, ...]:
    claims = tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+|\n+", answer)
        if part.strip()
    )
    return claims or (answer.strip(),)


def split_sentences(text: str) -> tuple[str, ...]:
    sentences = tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+|\n+", text)
        if part.strip()
    )
    return sentences or (text.strip(),)


def token_f1(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / (precision + recall)


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in tokenize(text) if token not in STOPWORDS)


def extract_numbers(text: str) -> frozenset[str]:
    return frozenset(
        match.group(0).replace(",", "").lower()
        for match in re.finditer(
            r"(?:\$|€|£)?\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|st|nd|rd|th)?\b|"
            r"\b(?:19|20)\d{2}\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def extract_negations(text: str) -> frozenset[str]:
    tokens = tokenize(text)
    return frozenset(token for token in tokens if token in NEGATION_WORDS)


def extract_entities(text: str) -> frozenset[str]:
    entities = set()
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*\b", text):
        value = match.group(0).strip()
        if len(value) > 2 and value.lower() not in STOPWORDS:
            entities.add(value.lower())
    return frozenset(entities)


def claim_support_score(claim: str, contexts: tuple[str, ...]) -> float:
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return 0.0
    return max((token_f1(claim_tokens, tokenize(context)) for context in contexts), default=0.0)


def coverage(items: frozenset[str], evidence: frozenset[str]) -> float:
    if not items:
        return 1.0
    return len(items & evidence) / len(items)


def deterministic_nli_scores(claim: str, evidence: str) -> NliScores:
    claim_tokens = content_tokens(claim)
    evidence_tokens = content_tokens(evidence)
    lexical = token_f1(claim_tokens, evidence_tokens)
    entity_cov = coverage(extract_entities(claim), extract_entities(evidence))
    number_cov = coverage(extract_numbers(claim), extract_numbers(evidence))
    claim_negations = extract_negations(claim)
    evidence_negations = extract_negations(evidence)
    negation_conflict = (
        1.0 if claim_negations != evidence_negations and (claim_negations or evidence_negations) else 0.0
    )
    number_conflict = 1.0 - number_cov
    entity_conflict = 1.0 - entity_cov
    contradiction = min(1.0, max(number_conflict, negation_conflict, 0.65 * entity_conflict))
    entailment = min(1.0, max(lexical, 0.55 * entity_cov + 0.35 * number_cov + 0.10 * lexical))
    entailment *= 1.0 - 0.85 * contradiction
    neutral = max(0.0, 1.0 - entailment - contradiction)
    total = entailment + contradiction + neutral
    if total <= 0.0:
        return NliScores(entailment=0.0, contradiction=0.0, neutral=1.0)
    return NliScores(
        entailment=entailment / total,
        contradiction=contradiction / total,
        neutral=neutral / total,
    )


def load_nli_scores_jsonl(path: Path | None) -> NliScoreMap:
    if path is None:
        return {}
    scores: NliScoreMap = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["record_id"])
            claim_index = int(row["claim_index"])
            entailment = float(row.get("entailment", 0.0))
            contradiction = float(row.get("contradiction", 0.0))
            neutral = float(row.get("neutral", max(0.0, 1.0 - entailment - contradiction)))
            total = entailment + contradiction + neutral
            if total <= 0.0:
                scores[(record_id, claim_index)] = NliScores(0.0, 0.0, 1.0)
            else:
                scores[(record_id, claim_index)] = NliScores(
                    entailment / total,
                    contradiction / total,
                    neutral / total,
                )
    return scores


def evidence_candidates(
    claim: str,
    contexts: tuple[str, ...],
) -> tuple[EvidenceCandidate, ...]:
    claim_tokens = content_tokens(claim)
    claim_entities = extract_entities(claim)
    claim_numbers = extract_numbers(claim)
    candidates: list[EvidenceCandidate] = []
    sentence_index = 0
    for context in contexts:
        for sentence in split_sentences(context):
            sentence_tokens = content_tokens(sentence)
            lexical = token_f1(claim_tokens, sentence_tokens)
            entity_cov = coverage(claim_entities, extract_entities(sentence))
            number_cov = coverage(claim_numbers, extract_numbers(sentence))
            nli = deterministic_nli_scores(claim, sentence)
            score = (
                0.45 * lexical
                + 0.20 * entity_cov
                + 0.15 * number_cov
                + 0.25 * nli.entailment
                - 0.30 * nli.contradiction
            )
            if lexical > 0.0 or entity_cov < 1.0 or number_cov < 1.0:
                candidates.append(
                    EvidenceCandidate(
                        sentence_index=sentence_index,
                        text=sentence,
                        lexical_score=lexical,
                        entity_coverage=entity_cov,
                        number_coverage=number_cov,
                        nli=nli,
                        score=score,
                    )
                )
            sentence_index += 1
    candidates.sort(key=lambda item: item.score, reverse=True)
    return tuple(candidates)


def retrieve_evidence_sentences(
    claim: str,
    contexts: tuple[str, ...],
    *,
    top_k: int = 3,
) -> tuple[str, ...]:
    return tuple(candidate.text for candidate in evidence_candidates(claim, contexts)[:top_k])


def missing_fraction(items: frozenset[str], evidence: frozenset[str]) -> float:
    if not items:
        return 0.0
    return len(items - evidence) / len(items)


def enhanced_claim_action(claim: str, contexts: tuple[str, ...]) -> tuple[float, float]:
    evidence_sentences = retrieve_evidence_sentences(claim, contexts)
    evidence_text = " ".join(evidence_sentences) if evidence_sentences else " ".join(contexts)
    full_context_text = " ".join(contexts)
    claim_content = content_tokens(claim)
    evidence_content = content_tokens(evidence_text)
    support = token_f1(claim_content, evidence_content)
    novelty = missing_fraction(claim_content, evidence_content)
    number_mismatch = missing_fraction(extract_numbers(claim), extract_numbers(full_context_text))
    entity_mismatch = missing_fraction(extract_entities(claim), extract_entities(full_context_text))
    claim_negations = extract_negations(claim)
    evidence_negations = extract_negations(full_context_text)
    negation_mismatch = 1.0 if claim_negations != evidence_negations and (claim_negations or evidence_negations) else 0.0

    action = (
        0.70 * (1.0 - support) ** 2
        + 0.45 * novelty**2
        + 1.25 * number_mismatch
        + 0.70 * entity_mismatch
        + 1.00 * negation_mismatch
    )
    accepted = (
        support >= 0.50
        and novelty <= 0.45
        and number_mismatch == 0.0
        and entity_mismatch <= 0.35
        and negation_mismatch == 0.0
    )
    return action, 1.0 if accepted else 0.0


def semantic_claim_action(
    claim: str,
    contexts: tuple[str, ...],
    *,
    nli_override: NliScores | None = None,
) -> tuple[float, float]:
    """Claim-level semantic residual using retrieved evidence and hard mismatch axes."""
    candidates = evidence_candidates(claim, contexts)
    evidence_sentences = tuple(candidate.text for candidate in candidates[:5])
    evidence_text = " ".join(evidence_sentences) if evidence_sentences else " ".join(contexts)
    full_context_text = " ".join(contexts)

    claim_tokens = content_tokens(claim)
    evidence_tokens = content_tokens(evidence_text)
    full_tokens = content_tokens(full_context_text)

    local_support = token_f1(claim_tokens, evidence_tokens)
    global_support = token_f1(claim_tokens, full_tokens)
    support = max(local_support, 0.75 * global_support)

    local_novelty = missing_fraction(claim_tokens, evidence_tokens)
    global_novelty = missing_fraction(claim_tokens, full_tokens)
    unsupported = min(local_novelty, 0.75 * global_novelty)

    claim_numbers = extract_numbers(claim)
    context_numbers = extract_numbers(full_context_text)
    number_mismatch = missing_fraction(claim_numbers, context_numbers)

    claim_entities = extract_entities(claim)
    context_entities = extract_entities(full_context_text)
    entity_mismatch = missing_fraction(claim_entities, context_entities)

    claim_negations = extract_negations(claim)
    context_negations = extract_negations(full_context_text)
    negation_mismatch = (
        1.0 if claim_negations != context_negations and (claim_negations or context_negations) else 0.0
    )

    best_nli = nli_override or (candidates[0].nli if candidates else deterministic_nli_scores(claim, full_context_text))
    contradiction = max(number_mismatch, negation_mismatch, 0.5 * entity_mismatch, best_nli.contradiction)
    action = (
        0.35 * (1.0 - support) ** 2
        + 0.90 * unsupported**2
        + 2.40 * contradiction
        + 0.75 * best_nli.neutral
        - 0.35 * best_nli.entailment
    )
    action = max(0.0, action)
    accepted = support >= 0.38 and unsupported <= 0.62 and contradiction <= 0.15 and best_nli.entailment >= 0.35
    return action, 1.0 if accepted else 0.0


def semantic_feature_summary(answer: str, contexts: tuple[str, ...]) -> dict[str, float]:
    claims = split_claims(answer)
    if not claims:
        claims = (answer,)
    actions = []
    entailments = []
    contradictions = []
    neutrals = []
    margins = []
    unsupported = 0
    for claim in claims:
        candidates = evidence_candidates(claim, contexts)
        top = candidates[0] if candidates else None
        action, accepted = semantic_claim_action(claim, contexts)
        actions.append(action)
        if accepted == 0.0:
            unsupported += 1
        if top is None:
            nli = deterministic_nli_scores(claim, " ".join(contexts))
            entailments.append(nli.entailment)
            contradictions.append(nli.contradiction)
            neutrals.append(nli.neutral)
            margins.append(0.0)
        else:
            entailments.append(top.nli.entailment)
            contradictions.append(top.nli.contradiction)
            neutrals.append(top.nli.neutral)
            second_score = candidates[1].score if len(candidates) > 1 else 0.0
            margins.append(max(0.0, top.score - second_score))
    ordered_actions = sorted(actions)
    p90_idx = min(len(ordered_actions) - 1, int(0.9 * (len(ordered_actions) - 1))) if ordered_actions else 0
    return {
        "max_contradiction_score": max(contradictions, default=0.0),
        "mean_entailment_score": sum(entailments) / len(entailments) if entailments else 0.0,
        "neutral_claim_fraction": sum(1 for value in neutrals if value >= 0.34) / len(neutrals) if neutrals else 0.0,
        "unsupported_span_fraction": unsupported / len(claims) if claims else 0.0,
        "reranker_margin": sum(margins) / len(margins) if margins else 0.0,
        "claim_action_p90": ordered_actions[p90_idx] if ordered_actions else 0.0,
    }


def span_overlaps(start: int, end: int, span_start: int | None, span_end: int | None) -> bool:
    if span_start is None or span_end is None:
        return False
    return max(start, span_start) < min(end, span_end)


def span_label_for_claim(record: BenchmarkRecord, claim: str, cursor: int) -> tuple[int | None, int | None, str]:
    start = record.answer.find(claim, cursor)
    if start < 0:
        start = record.answer.find(claim)
    if start < 0:
        return None, None, ""
    end = start + len(claim)
    for label in record.labels:
        span_start = label.get("start")
        span_end = label.get("end")
        if isinstance(span_start, int) and isinstance(span_end, int) and span_overlaps(start, end, span_start, span_end):
            return span_start, span_end, str(label.get("label_type", label.get("type", "")))
    return None, None, ""


def candidate_from_record(
    record: BenchmarkRecord,
    *,
    max_context_chars: int | None = 6000,
) -> ResidualAnswerCandidate:
    claims = []
    contexts = record.contexts
    if max_context_chars is not None and max_context_chars > 0:
        contexts = tuple(context[:max_context_chars] for context in contexts)
    for claim_idx, claim in enumerate(split_claims(record.answer)):
        score = claim_support_score(claim, contexts)
        claims.append(
            ResidualClaim(
                claim,
                (
                    ClaimAxisEvidence(
                        axis="lexical_support",
                        value=score,
                        reference=1.0,
                        sigma=0.25,
                        source_reliability=1.0,
                        source_family=f"context-{claim_idx % 2}",
                    ),
                    ClaimAxisEvidence(
                        axis="lexical_support",
                        value=score,
                        reference=1.0,
                        sigma=0.25,
                        source_reliability=1.0,
                        source_family=f"context-{(claim_idx + 1) % 2}",
                    ),
                ),
            )
        )
    return ResidualAnswerCandidate(
        text=record.answer,
        claims=tuple(claims),
        required_slots=max(1, len(claims)),
        covered_slots=len(claims),
    )


def predict_record(
    verifier: ClaimResidualVerifier,
    record: BenchmarkRecord,
    *,
    action_threshold: float,
    accepted_fraction_threshold: float,
    max_context_chars: int | None = 6000,
) -> BenchmarkPrediction:
    state = verifier.answer_state(
        candidate_from_record(record, max_context_chars=max_context_chars)
    )
    predicted = (
        state.action > action_threshold
        or state.accepted_fraction < accepted_fraction_threshold
    )
    score = state.action + max(0.0, accepted_fraction_threshold - state.accepted_fraction)
    return BenchmarkPrediction(
        record_id=record.record_id,
        score=score,
        action=state.action,
        accepted_fraction=state.accepted_fraction,
        predicted_hallucinated=predicted,
        actual_hallucinated=record.is_hallucinated,
    )


def predict_record_fast_lexical(
    record: BenchmarkRecord,
    *,
    action_threshold: float,
    accepted_fraction_threshold: float,
    max_context_chars: int | None = 6000,
    response_level: bool = False,
) -> BenchmarkPrediction:
    contexts = record.contexts
    if max_context_chars is not None and max_context_chars > 0:
        contexts = tuple(context[:max_context_chars] for context in contexts)
    claims = (record.answer,) if response_level else split_claims(record.answer)
    scores = [claim_support_score(claim, contexts) for claim in claims]
    if not scores:
        scores = [0.0]
    residual_actions = [1.6 * (1.0 - score) ** 2 for score in scores]
    action = sum(residual_actions) / len(residual_actions)
    accepted_fraction = sum(1 for score in scores if score >= 0.875) / len(scores)
    predicted = action > action_threshold or accepted_fraction < accepted_fraction_threshold
    score = action + max(0.0, accepted_fraction_threshold - accepted_fraction)
    return BenchmarkPrediction(
        record_id=record.record_id,
        score=score,
        action=action,
        accepted_fraction=accepted_fraction,
        predicted_hallucinated=predicted,
        actual_hallucinated=record.is_hallucinated,
    )


def predict_record_enhanced_evidence(
    record: BenchmarkRecord,
    *,
    action_threshold: float,
    accepted_fraction_threshold: float,
    max_context_chars: int | None = 6000,
    response_level: bool = False,
) -> BenchmarkPrediction:
    contexts = record.contexts
    if max_context_chars is not None and max_context_chars > 0:
        contexts = tuple(context[:max_context_chars] for context in contexts)
    claims = (record.answer,) if response_level else split_claims(record.answer)
    claim_results = [enhanced_claim_action(claim, contexts) for claim in claims]
    if not claim_results:
        claim_results = [(2.0, 0.0)]
    action = sum(result[0] for result in claim_results) / len(claim_results)
    accepted_fraction = sum(result[1] for result in claim_results) / len(claim_results)
    predicted = action > action_threshold or accepted_fraction < accepted_fraction_threshold
    score = action + max(0.0, accepted_fraction_threshold - accepted_fraction)
    return BenchmarkPrediction(
        record_id=record.record_id,
        score=score,
        action=action,
        accepted_fraction=accepted_fraction,
        predicted_hallucinated=predicted,
        actual_hallucinated=record.is_hallucinated,
    )


def predict_record_semantic_evidence(
    record: BenchmarkRecord,
    *,
    action_threshold: float,
    accepted_fraction_threshold: float,
    max_context_chars: int | None = 6000,
    response_level: bool = False,
    nli_scores: NliScoreMap | None = None,
) -> BenchmarkPrediction:
    contexts = record.contexts
    if max_context_chars is not None and max_context_chars > 0:
        contexts = tuple(context[:max_context_chars] for context in contexts)
    claims = (record.answer,) if response_level else split_claims(record.answer)
    claim_results = [
        semantic_claim_action(
            claim,
            contexts,
            nli_override=(nli_scores or {}).get((record.record_id, claim_idx)),
        )
        for claim_idx, claim in enumerate(claims)
    ]
    if not claim_results:
        claim_results = [(2.0, 0.0)]
    action = sum(result[0] for result in claim_results) / len(claim_results)
    accepted_fraction = sum(result[1] for result in claim_results) / len(claim_results)
    predicted = action > action_threshold or accepted_fraction < accepted_fraction_threshold
    score = action + max(0.0, accepted_fraction_threshold - accepted_fraction)
    return BenchmarkPrediction(
        record_id=record.record_id,
        score=score,
        action=action,
        accepted_fraction=accepted_fraction,
        predicted_hallucinated=predicted,
        actual_hallucinated=record.is_hallucinated,
    )


def claim_predictions_for_record(
    record: BenchmarkRecord,
    *,
    max_context_chars: int | None = 6000,
    response_level: bool = False,
    fast_lexical: bool = False,
    enhanced_evidence: bool = False,
    semantic_evidence: bool = False,
    nli_scores: NliScoreMap | None = None,
) -> tuple[ClaimPrediction, ...]:
    contexts = record.contexts
    if max_context_chars is not None and max_context_chars > 0:
        contexts = tuple(context[:max_context_chars] for context in contexts)
    claims = (record.answer,) if response_level else split_claims(record.answer)
    rows: list[ClaimPrediction] = []
    cursor = 0
    for idx, claim in enumerate(claims):
        candidates = evidence_candidates(claim, contexts)
        top = candidates[0] if candidates else None
        nli = (nli_scores or {}).get(
            (record.record_id, idx),
            top.nli if top is not None else deterministic_nli_scores(claim, " ".join(contexts)),
        )
        if semantic_evidence:
            action, accepted_float = semantic_claim_action(claim, contexts, nli_override=nli)
        elif enhanced_evidence:
            action, accepted_float = enhanced_claim_action(claim, contexts)
        elif fast_lexical:
            score = claim_support_score(claim, contexts)
            action = 1.6 * (1.0 - score) ** 2
            accepted_float = 1.0 if score >= 0.875 else 0.0
        else:
            score = claim_support_score(claim, contexts)
            action = 1.6 * (1.0 - score) ** 2
            accepted_float = 1.0 if score >= 0.875 else 0.0
        span_start, span_end, span_label = span_label_for_claim(record, claim, cursor)
        found = record.answer.find(claim, cursor)
        if found >= 0:
            cursor = found + len(claim)
        rows.append(
            ClaimPrediction(
                record_id=record.record_id,
                claim_index=idx,
                claim=claim,
                action=action,
                accepted=bool(accepted_float),
                evidence=top.text if top is not None else "",
                entailment=nli.entailment,
                contradiction=nli.contradiction,
                neutral=nli.neutral,
                span_start=span_start,
                span_end=span_end,
                span_label=span_label,
            )
        )
    return tuple(rows)


def with_threshold(
    prediction: BenchmarkPrediction,
    *,
    action_threshold: float,
    accepted_fraction_threshold: float,
) -> BenchmarkPrediction:
    predicted = (
        prediction.action > action_threshold
        or prediction.accepted_fraction < accepted_fraction_threshold
    )
    score = prediction.action + max(0.0, accepted_fraction_threshold - prediction.accepted_fraction)
    return BenchmarkPrediction(
        record_id=prediction.record_id,
        score=score,
        action=prediction.action,
        accepted_fraction=prediction.accepted_fraction,
        predicted_hallucinated=predicted,
        actual_hallucinated=prediction.actual_hallucinated,
    )


def evaluate_predictions(predictions: Iterable[BenchmarkPrediction]) -> BinaryMetrics:
    total = 0
    predicted_hallucinated = 0
    actual_hallucinated = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for prediction in predictions:
        total += 1
        predicted_hallucinated += int(prediction.predicted_hallucinated)
        actual_hallucinated += int(prediction.actual_hallucinated)
        if prediction.predicted_hallucinated and prediction.actual_hallucinated:
            true_positive += 1
        elif prediction.predicted_hallucinated and not prediction.actual_hallucinated:
            false_positive += 1
        elif not prediction.predicted_hallucinated and prediction.actual_hallucinated:
            false_negative += 1
        else:
            true_negative += 1
    return BinaryMetrics(
        total=total,
        predicted_hallucinated=predicted_hallucinated,
        actual_hallucinated=actual_hallucinated,
        true_positive=true_positive,
        false_positive=false_positive,
        true_negative=true_negative,
        false_negative=false_negative,
    )


def auroc(predictions: Sequence[BenchmarkPrediction]) -> float:
    positives = [prediction.score for prediction in predictions if prediction.actual_hallucinated]
    negatives = [prediction.score for prediction in predictions if not prediction.actual_hallucinated]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total if total else 0.0


def auprc(predictions: Sequence[BenchmarkPrediction]) -> float:
    positives = sum(1 for prediction in predictions if prediction.actual_hallucinated)
    if positives == 0:
        return 0.0
    ordered = sorted(predictions, key=lambda prediction: prediction.score, reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for prediction in ordered:
        if prediction.actual_hallucinated:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def raw_predictions(
    path: Path,
    *,
    accepted_fraction_threshold: float = 1.0,
    max_context_chars: int | None = 6000,
    limit: int | None = None,
    fast_lexical: bool = False,
    response_level: bool = False,
    enhanced_evidence: bool = False,
    semantic_evidence: bool = False,
    nli_evidence: bool = False,
    nli_scores_jsonl: Path | None = None,
) -> tuple[BenchmarkPrediction, ...]:
    verifier = ClaimResidualVerifier()
    nli_scores = load_nli_scores_jsonl(nli_scores_jsonl)
    if semantic_evidence or nli_evidence:
        return tuple(
            predict_record_semantic_evidence(
                record,
                action_threshold=math.inf,
                accepted_fraction_threshold=accepted_fraction_threshold,
                max_context_chars=max_context_chars,
                response_level=response_level,
                nli_scores=nli_scores,
            )
            for record in load_jsonl(path, limit=limit)
        )
    if enhanced_evidence:
        return tuple(
            predict_record_enhanced_evidence(
                record,
                action_threshold=math.inf,
                accepted_fraction_threshold=accepted_fraction_threshold,
                max_context_chars=max_context_chars,
                response_level=response_level,
            )
            for record in load_jsonl(path, limit=limit)
        )
    if fast_lexical:
        return tuple(
            predict_record_fast_lexical(
                record,
                action_threshold=math.inf,
                accepted_fraction_threshold=accepted_fraction_threshold,
                max_context_chars=max_context_chars,
                response_level=response_level,
            )
            for record in load_jsonl(path, limit=limit)
        )
    return tuple(
        predict_record(
            verifier,
            record,
            action_threshold=math.inf,
            accepted_fraction_threshold=accepted_fraction_threshold,
            max_context_chars=max_context_chars,
        )
        for record in load_jsonl(path, limit=limit)
    )


def threshold_candidates(predictions: Sequence[BenchmarkPrediction]) -> tuple[float, ...]:
    actions = sorted({prediction.action for prediction in predictions})
    if not actions:
        return (1.0,)
    candidates = [0.0]
    candidates.extend(actions)
    candidates.extend((left + right) / 2.0 for left, right in zip(actions, actions[1:]))
    candidates.append(max(actions) + 1e-9)
    return tuple(sorted(set(candidates)))


def apply_thresholds(
    predictions: Sequence[BenchmarkPrediction],
    *,
    action_threshold: float,
    accepted_fraction_threshold: float,
) -> tuple[BenchmarkPrediction, ...]:
    return tuple(
        with_threshold(
            prediction,
            action_threshold=action_threshold,
            accepted_fraction_threshold=accepted_fraction_threshold,
        )
        for prediction in predictions
    )


def calibrate_thresholds(
    predictions: Sequence[BenchmarkPrediction],
    *,
    accepted_fraction_threshold: float = 1.0,
) -> CalibrationResult:
    best: CalibrationResult | None = None
    ranking_auroc = auroc(predictions)
    ranking_auprc = auprc(predictions)
    for action_threshold in threshold_candidates(predictions):
        adjusted = apply_thresholds(
            predictions,
            action_threshold=action_threshold,
            accepted_fraction_threshold=accepted_fraction_threshold,
        )
        metrics = evaluate_predictions(adjusted)
        result = CalibrationResult(
            action_threshold=action_threshold,
            accepted_fraction_threshold=accepted_fraction_threshold,
            metrics=metrics,
            auroc=ranking_auroc,
            auprc=ranking_auprc,
        )
        key = (metrics.f1, metrics.balanced_accuracy, metrics.accuracy, -action_threshold)
        if best is None:
            best = result
            best_key = key
        elif key > best_key:
            best = result
            best_key = key
    if best is None:
        empty_metrics = BinaryMetrics(0, 0, 0, 0, 0, 0, 0)
        return CalibrationResult(
            action_threshold=1.0,
            accepted_fraction_threshold=accepted_fraction_threshold,
            metrics=empty_metrics,
            auroc=0.0,
            auprc=0.0,
        )
    return best


def evaluate_jsonl(
    path: Path,
    *,
    action_threshold: float = 1.0,
    accepted_fraction_threshold: float = 1.0,
    max_context_chars: int | None = 6000,
    limit: int | None = None,
    fast_lexical: bool = False,
    response_level: bool = False,
    enhanced_evidence: bool = False,
    semantic_evidence: bool = False,
    nli_evidence: bool = False,
    nli_scores_jsonl: Path | None = None,
) -> tuple[BinaryMetrics, tuple[BenchmarkPrediction, ...]]:
    predictions = apply_thresholds(
        raw_predictions(
            path,
            accepted_fraction_threshold=accepted_fraction_threshold,
            max_context_chars=max_context_chars,
            limit=limit,
            fast_lexical=fast_lexical,
            response_level=response_level,
            enhanced_evidence=enhanced_evidence,
            semantic_evidence=semantic_evidence,
            nli_evidence=nli_evidence,
            nli_scores_jsonl=nli_scores_jsonl,
        ),
        action_threshold=action_threshold,
        accepted_fraction_threshold=accepted_fraction_threshold,
    )
    return evaluate_predictions(predictions), predictions


def write_error_csv(path: Path, predictions: Sequence[BenchmarkPrediction]) -> None:
    errors = [
        prediction
        for prediction in predictions
        if prediction.predicted_hallucinated != prediction.actual_hallucinated
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_id",
                "score",
                "action",
                "accepted_fraction",
                "predicted_hallucinated",
                "actual_hallucinated",
            ]
        )
        for prediction in errors:
            writer.writerow(
                [
                    prediction.record_id,
                    f"{prediction.score:.12g}",
                    f"{prediction.action:.12g}",
                    f"{prediction.accepted_fraction:.12g}",
                    int(prediction.predicted_hallucinated),
                    int(prediction.actual_hallucinated),
                ]
            )


def write_claim_csv(path: Path, predictions: Sequence[ClaimPrediction]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_id",
                "claim_index",
                "action",
                "accepted",
                "entailment",
                "contradiction",
                "neutral",
                "span_start",
                "span_end",
                "span_label",
                "evidence",
                "claim",
            ]
        )
        for prediction in predictions:
            writer.writerow(
                [
                    prediction.record_id,
                    prediction.claim_index,
                    f"{prediction.action:.12g}",
                    int(prediction.accepted),
                    f"{prediction.entailment:.12g}",
                    f"{prediction.contradiction:.12g}",
                    f"{prediction.neutral:.12g}",
                    "" if prediction.span_start is None else prediction.span_start,
                    "" if prediction.span_end is None else prediction.span_end,
                    prediction.span_label,
                    prediction.evidence,
                    prediction.claim,
                ]
            )


def evaluate_span_predictions(predictions: Sequence[ClaimPrediction]) -> SpanMetrics:
    predicted_positive = sum(1 for prediction in predictions if not prediction.accepted)
    actual_positive = sum(1 for prediction in predictions if prediction.span_label)
    true_positive = sum(
        1
        for prediction in predictions
        if not prediction.accepted and prediction.span_label
    )
    return SpanMetrics(
        predicted_positive=predicted_positive,
        actual_positive=actual_positive,
        true_positive=true_positive,
    )


def collect_nli_pairs(
    path: Path,
    *,
    max_context_chars: int | None = 6000,
    limit: int | None = None,
    response_level: bool = False,
    top_k: int = 3,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for record in load_jsonl(path, limit=limit):
        contexts = record.contexts
        if max_context_chars is not None and max_context_chars > 0:
            contexts = tuple(context[:max_context_chars] for context in contexts)
        claims = (record.answer,) if response_level else split_claims(record.answer)
        for claim_idx, claim in enumerate(claims):
            candidates = evidence_candidates(claim, contexts)
            evidence = " ".join(candidate.text for candidate in candidates[:top_k]) if candidates else " ".join(contexts)
            rows.append(
                {
                    "record_id": record.record_id,
                    "claim_index": claim_idx,
                    "claim": claim,
                    "evidence": evidence,
                }
            )
    return tuple(rows)


def write_nli_pairs_jsonl(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(dict(pair), ensure_ascii=False) + "\n")


def collect_claim_predictions(
    path: Path,
    *,
    max_context_chars: int | None = 6000,
    limit: int | None = None,
    fast_lexical: bool = False,
    response_level: bool = False,
    enhanced_evidence: bool = False,
    semantic_evidence: bool = False,
    nli_evidence: bool = False,
    nli_scores_jsonl: Path | None = None,
) -> tuple[ClaimPrediction, ...]:
    rows: list[ClaimPrediction] = []
    nli_scores = load_nli_scores_jsonl(nli_scores_jsonl)
    for record in load_jsonl(path, limit=limit):
        rows.extend(
            claim_predictions_for_record(
                record,
                max_context_chars=max_context_chars,
                response_level=response_level,
                fast_lexical=fast_lexical,
                enhanced_evidence=enhanced_evidence,
                semantic_evidence=semantic_evidence or nli_evidence,
                nli_scores=nli_scores,
            )
        )
    return tuple(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--action-threshold", type=float, default=1.0)
    parser.add_argument("--accepted-fraction-threshold", type=float, default=1.0)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--export-errors", type=Path)
    parser.add_argument("--export-claims", type=Path)
    parser.add_argument("--export-nli-pairs", type=Path)
    parser.add_argument("--nli-scores-jsonl", type=Path)
    parser.add_argument("--nli-pair-top-k", type=int, default=3)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fast-lexical", action="store_true")
    parser.add_argument("--response-level", action="store_true")
    parser.add_argument("--enhanced-evidence", action="store_true")
    parser.add_argument("--semantic-evidence", action="store_true")
    parser.add_argument("--nli-evidence", action="store_true")
    parser.add_argument("--show-errors", type=int, default=5)
    args = parser.parse_args()

    if args.calibrate:
        raw = raw_predictions(
            args.jsonl,
            accepted_fraction_threshold=args.accepted_fraction_threshold,
            max_context_chars=args.max_context_chars,
            limit=args.limit,
            fast_lexical=args.fast_lexical,
            response_level=args.response_level,
            enhanced_evidence=args.enhanced_evidence,
            semantic_evidence=args.semantic_evidence,
            nli_evidence=args.nli_evidence,
            nli_scores_jsonl=args.nli_scores_jsonl,
        )
        calibration = calibrate_thresholds(
            raw,
            accepted_fraction_threshold=args.accepted_fraction_threshold,
        )
        metrics = calibration.metrics
        predictions = apply_thresholds(
            raw,
            action_threshold=calibration.action_threshold,
            accepted_fraction_threshold=calibration.accepted_fraction_threshold,
        )
        action_threshold = calibration.action_threshold
        print("# CE Claim Residual benchmark calibration")
        print(f"best_action_threshold {calibration.action_threshold:.6f}")
        print(f"accepted_fraction_threshold {calibration.accepted_fraction_threshold:.6f}")
        print(f"auroc {calibration.auroc:.6f}")
        print(f"auprc {calibration.auprc:.6f}")
    else:
        metrics, predictions = evaluate_jsonl(
            args.jsonl,
            action_threshold=args.action_threshold,
            accepted_fraction_threshold=args.accepted_fraction_threshold,
            max_context_chars=args.max_context_chars,
            limit=args.limit,
            fast_lexical=args.fast_lexical,
            response_level=args.response_level,
            enhanced_evidence=args.enhanced_evidence,
            semantic_evidence=args.semantic_evidence,
            nli_evidence=args.nli_evidence,
            nli_scores_jsonl=args.nli_scores_jsonl,
        )
        action_threshold = args.action_threshold
    print("# CE Claim Residual benchmark")
    print(f"action_threshold {action_threshold:.6f}")
    print(f"accepted_fraction_threshold {args.accepted_fraction_threshold:.6f}")
    print(f"records {metrics.total}")
    print(f"predicted_hallucinated {metrics.predicted_hallucinated}")
    print(f"actual_hallucinated {metrics.actual_hallucinated}")
    print(f"accuracy {metrics.accuracy:.6f}")
    print(f"balanced_accuracy {metrics.balanced_accuracy:.6f}")
    print(f"precision {metrics.precision:.6f}")
    print(f"recall {metrics.recall:.6f}")
    print(f"f1 {metrics.f1:.6f}")
    print(f"tp {metrics.true_positive}")
    print(f"fp {metrics.false_positive}")
    print(f"tn {metrics.true_negative}")
    print(f"fn {metrics.false_negative}")
    errors = [
        prediction
        for prediction in predictions
        if prediction.predicted_hallucinated != prediction.actual_hallucinated
    ]
    if errors:
        print()
        print("record_id,action,accepted_fraction,predicted,actual")
        for prediction in errors[: args.show_errors]:
            print(
                f"{prediction.record_id},{prediction.action:.6f},"
                f"{prediction.accepted_fraction:.6f},"
                f"{int(prediction.predicted_hallucinated)},"
                f"{int(prediction.actual_hallucinated)}"
            )
    if args.export_errors is not None:
        write_error_csv(args.export_errors, predictions)
    if args.export_nli_pairs is not None:
        write_nli_pairs_jsonl(
            args.export_nli_pairs,
            collect_nli_pairs(
                args.jsonl,
                max_context_chars=args.max_context_chars,
                limit=args.limit,
                response_level=args.response_level,
                top_k=args.nli_pair_top_k,
            ),
        )
    if args.export_claims is not None:
        write_claim_csv(
            args.export_claims,
            collect_claim_predictions(
                args.jsonl,
                max_context_chars=args.max_context_chars,
                limit=args.limit,
                fast_lexical=args.fast_lexical,
                response_level=args.response_level,
                enhanced_evidence=args.enhanced_evidence,
                semantic_evidence=args.semantic_evidence,
                nli_evidence=args.nli_evidence,
                nli_scores_jsonl=args.nli_scores_jsonl,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
