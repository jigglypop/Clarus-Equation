"""Topology-aware intent classifier with optional AGI-style stateful updates."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re

try:
    from .neuromod import (
        ModulationEffect,
        NeuromodulatorState,
        apply_modulation,
        step_neuromodulators,
    )
    from .stdp import compute_learning_gate
    from .text_topology import TextTopologyEngine, TextTopologyResult
except ImportError:
    from clarus.neuromod import (
        ModulationEffect,
        NeuromodulatorState,
        apply_modulation,
        step_neuromodulators,
    )
    from clarus.stdp import compute_learning_gate
    from clarus.text_topology import TextTopologyEngine, TextTopologyResult


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")
_RUNTIME_HASH_SALT = "intent-runtime-signal"


def _softmax_with_temperature(
    scores: dict[str, float], temperature: float
) -> dict[str, float]:
    if not scores:
        return {}
    tau = max(float(temperature), 1e-6)
    values = list(scores.values())
    max_score = max(values)
    exp_values = [math.exp((v - max_score) / tau) for v in values]
    denom = sum(exp_values) or 1.0
    return {label: exp_values[idx] / denom for idx, label in enumerate(scores)}


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    denom = max(_norm(a) * _norm(b), 1e-8)
    return _dot(a, b) / denom


def _l2_normalise(a: list[float]) -> list[float]:
    denom = _norm(a)
    if denom <= 1e-12:
        return list(a)
    inv = 1.0 / denom
    return [inv * x for x in a]


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), lo), hi)


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
    probabilities: dict[str, float] = field(default_factory=dict)
    replay_scores: dict[str, float] = field(default_factory=dict)
    trace_scores: dict[str, float] = field(default_factory=dict)
    effective_temperature: float = 0.0
    state_report: dict[str, float] = field(default_factory=dict)


@dataclass
class LabeledIntentExample:
    text: str
    label: str


@dataclass
class IntentReplayItem:
    feature_vector: list[float]
    label: str
    confidence: float
    salience: float
    priority: float
    reward: float = 0.0


@dataclass
class IntentSessionState:
    replay_buffer: list[IntentReplayItem] = field(default_factory=list)
    active_ratio_ema: float = 0.125
    label_bias: dict[str, float] = field(default_factory=dict)
    label_traces: dict[str, list[float]] = field(default_factory=dict)
    label_trace_strength: dict[str, float] = field(default_factory=dict)
    mode_occupancy: dict[str, int] = field(
        default_factory=lambda: {"WAKE": 0, "NREM": 0, "REM": 0}
    )
    prev_critic_score: float = 0.0
    last_label: str = ""
    last_confidence: float = 0.0
    step_count: int = 0
    neuromod: NeuromodulatorState = field(default_factory=NeuromodulatorState)
    last_phase_carrier: dict[str, float] = field(default_factory=dict)


class FittedTopologyIntentClassifier:
    """Few-shot centroid classifier over topology-aware text features."""

    def __init__(
        self,
        labels: list[str],
        centroids: dict[str, list[float]],
        parent: "TopologyIntentClassifier",
        temperature: float = 1.0,
    ) -> None:
        self.labels = list(labels)
        self.centroids = dict(centroids)
        self.parent = parent
        self.temperature = float(temperature)

    @property
    def engine(self) -> TextTopologyEngine:
        return self.parent.engine

    def _feature_vector(self, text: str) -> tuple[list[float], dict[str, float]]:
        analysis = self.parent.analyze(text)
        feat = self.parent._feature_vector_from_analysis(text, analysis)
        snapshot = self.parent._topology_features_from_analysis(analysis)
        snapshot["length_score"] = self.parent._length_score(text)
        return feat, snapshot

    def predict(self, text: str) -> IntentPrediction:
        feat, topo_snapshot = self._feature_vector(text)
        scores: dict[str, float] = {}
        for label in self.labels:
            centroid = self.centroids[label]
            dist2 = sum((a - b) * (a - b) for a, b in zip(feat, centroid))
            scores[label] = -dist2

        probabilities = _softmax_with_temperature(scores, self.temperature)
        best_label = max(scores, key=scores.__getitem__)
        return IntentPrediction(
            label=best_label,
            confidence=float(probabilities.get(best_label, 0.0)),
            scores=scores,
            matched_keywords={label: [] for label in self.labels},
            topology_snapshot=topo_snapshot,
            probabilities=probabilities,
            effective_temperature=self.temperature,
        )


_QUESTION_MARKERS = (
    "왜",
    "무엇",
    "뭐",
    "어떻게",
    "어느",
    "가능",
    "인가",
    "인가요",
    "할까",
    "할까요",
    "맞나",
    "맞나요",
    "what",
    "why",
    "how",
    "which",
)
_IMPERATIVE_MARKERS = (
    "해봐",
    "해줘",
    "만들어",
    "구현",
    "작성",
    "추가",
    "고쳐",
    "수정",
    "돌려봐",
    "리팩토",
    "정리",
    "implement",
    "build",
    "fix",
    "refactor",
    "add",
    "write",
)


class TopologyIntentClassifier:
    """Intent classifier using keyword overlap + topology features.

    `predict()` stays stateless for compatibility.
    `predict_step()` updates replay / self-measure / neuromod / STDP-lite state.
    """

    def __init__(
        self,
        dim: int = 32,
        temperature: float = 0.35,
        prototypes: tuple[IntentPrototype, ...] | None = None,
        *,
        replay_capacity: int = 16,
        replay_topk: int = 4,
        self_measure_target: float = 0.125,
        self_measure_alpha: float = 0.1,
        self_measure_strength: float = 0.35,
        replay_bias_scale: float = 0.18,
        trace_bias_scale: float = 0.22,
        stdp_lr: float = 0.08,
        stdp_trace_decay: float = 0.92,
        stdp_bias_clip: float = 1.5,
        consolidation_decay: float = 0.94,
    ) -> None:
        self.engine = TextTopologyEngine(dim=dim)
        self.temperature = float(temperature)
        self.replay_capacity = max(1, int(replay_capacity))
        self.replay_topk = max(1, int(replay_topk))
        self.self_measure_target = _clamp(self_measure_target, 0.0, 1.0)
        self.self_measure_alpha = _clamp(self_measure_alpha, 0.0, 1.0)
        self.self_measure_strength = max(float(self_measure_strength), 0.0)
        self.replay_bias_scale = max(float(replay_bias_scale), 0.0)
        self.trace_bias_scale = max(float(trace_bias_scale), 0.0)
        self.stdp_lr = max(float(stdp_lr), 0.0)
        self.stdp_trace_decay = _clamp(stdp_trace_decay, 0.0, 1.0)
        self.stdp_bias_clip = max(float(stdp_bias_clip), 0.0)
        self.consolidation_decay = _clamp(consolidation_decay, 0.0, 1.0)
        self.state = IntentSessionState(active_ratio_ema=self.self_measure_target)
        if prototypes is not None:
            self.prototypes = tuple(prototypes)
            self._ensure_state_labels()
            return
        self.prototypes = (
            IntentPrototype(
                name="explain",
                keywords=(
                    "설명",
                    "의미",
                    "개념",
                    "왜",
                    "원리",
                    "어떻게",
                    "해석",
                    "증명",
                    "유도",
                    "정의",
                    "이유",
                    "what",
                    "why",
                    "how",
                    "explain",
                    "derive",
                    "proof",
                ),
                question_bias=0.22,
                topology_bias=0.05,
            ),
            IntentPrototype(
                name="implement",
                keywords=(
                    "구현",
                    "만들",
                    "작성",
                    "추가",
                    "설계",
                    "코드",
                    "스크립트",
                    "함수",
                    "클래스",
                    "prototype",
                    "build",
                    "implement",
                    "write",
                    "create",
                    "add",
                ),
                imperative_bias=0.30,
                topology_bias=0.08,
            ),
            IntentPrototype(
                name="analyze",
                keywords=(
                    "분석",
                    "검토",
                    "정합",
                    "리뷰",
                    "구조",
                    "파악",
                    "평가",
                    "완성도",
                    "audit",
                    "analyze",
                    "review",
                    "consistency",
                    "evaluate",
                ),
                topology_bias=0.20,
            ),
            IntentPrototype(
                name="compare",
                keywords=(
                    "비교",
                    "추천",
                    "어느",
                    "가장",
                    "뭐가",
                    "해볼만",
                    "versus",
                    "compare",
                    "best",
                    "vs",
                    "쪽이",
                    "낫",
                    "선택",
                ),
                question_bias=0.14,
                topology_bias=0.10,
            ),
            IntentPrototype(
                name="debug",
                keywords=(
                    "에러",
                    "오류",
                    "안돼",
                    "안 돼",
                    "고쳐",
                    "버그",
                    "깨짐",
                    "왜안",
                    "fail",
                    "fix",
                    "bug",
                    "broken",
                    "crash",
                ),
                imperative_bias=0.18,
                topology_bias=-0.04,
            ),
            IntentPrototype(
                name="plan",
                keywords=(
                    "계획",
                    "로드맵",
                    "단계",
                    "우선순위",
                    "다음",
                    "전략",
                    "plan",
                    "roadmap",
                    "step",
                    "design",
                ),
                topology_bias=0.06,
            ),
            IntentPrototype(
                name="refactor",
                keywords=(
                    "리팩토링",
                    "리팩터링",
                    "정리",
                    "개선",
                    "다듬",
                    "고도화",
                    "refactor",
                    "cleanup",
                    "polish",
                    "improve",
                ),
                imperative_bias=0.20,
                topology_bias=0.04,
            ),
            IntentPrototype(
                name="research",
                keywords=(
                    "조사",
                    "탐색",
                    "찾",
                    "리서치",
                    "문헌",
                    "논문",
                    "research",
                    "investigate",
                    "survey",
                    "explore",
                ),
                topology_bias=0.08,
            ),
        )
        self._ensure_state_labels()

    def _ensure_state_labels(self) -> None:
        for proto in self.prototypes:
            self.state.label_bias.setdefault(proto.name, 0.0)
            self.state.label_trace_strength.setdefault(proto.name, 0.0)

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text)]

    def _length_score(self, text: str) -> float:
        return min(max(len(self._tokenize(text)), 1) / 20.0, 1.0)

    def analyze(self, text: str) -> TextTopologyResult:
        return self.engine.analyze(text)

    def _keyword_matches(
        self, text: str, tokens: list[str], keywords: tuple[str, ...]
    ) -> list[str]:
        haystack = text.lower()
        token_set = set(tokens)
        hits: list[str] = []
        for keyword in keywords:
            key = keyword.lower()
            if key in token_set:
                hits.append(keyword)
                continue
            if len(key) >= 2 and key in haystack:
                hits.append(keyword)
        return hits

    def _question_score(self, text: str) -> float:
        score = 0.0
        if "?" in text or "？" in text:
            score += 0.40
        lowered = text.lower()
        for marker in _QUESTION_MARKERS:
            if marker in lowered:
                score += 0.08
        return min(score, 0.95)

    def _imperative_score(self, text: str) -> float:
        score = 0.0
        lowered = text.lower()
        for marker in _IMPERATIVE_MARKERS:
            if marker in lowered:
                score += 0.12
        return min(score, 0.95)

    def _topology_features_from_analysis(
        self, topo: TextTopologyResult
    ) -> dict[str, float]:
        sent_count = max(topo.sentence_summary.count, 1)
        para_count = max(topo.paragraph_summary.count, 1)
        return {
            "token_density": topo.token_summary.edge_density,
            "sentence_density": topo.sentence_summary.edge_density,
            "paragraph_density": topo.paragraph_summary.edge_density,
            "token_sentence_alignment": topo.token_sentence_alignment,
            "sentence_paragraph_alignment": topo.sentence_paragraph_alignment,
            "bridge_energy": topo.bridge_energy,
            "token_sentence_bridge": topo.token_sentence_bridge,
            "sentence_paragraph_bridge": topo.sentence_paragraph_bridge,
            "sentence_fragmentation": (topo.sentence_summary.components - 1)
            / sent_count,
            "paragraph_fragmentation": (topo.paragraph_summary.components - 1)
            / para_count,
            "sentence_faces": float(topo.sentence_summary.faces),
            "sentence_fiedler": topo.sentence_summary.algebraic_connectivity,
            "paragraph_fiedler": topo.paragraph_summary.algebraic_connectivity,
            "phase_carrier_alignment": topo.phase_carrier_alignment,
            "phase_carrier_decay": topo.phase_carrier_decay,
        }

    def _topology_features(self, text: str) -> dict[str, float]:
        return self._topology_features_from_analysis(self.analyze(text))

    def _feature_vector_from_analysis(
        self, text: str, topo: TextTopologyResult
    ) -> list[float]:
        topo_features = self._topology_features_from_analysis(topo)
        tokens = self._tokenize(text)
        q_score = self._question_score(text)
        i_score = self._imperative_score(text)
        rule_scores, _ = self._rule_scores_from_features(
            text,
            tokens=tokens,
            topo_features=topo_features,
            q_score=q_score,
            i_score=i_score,
        )
        keyword_counts = [
            float(len(self._keyword_matches(text, tokens, proto.keywords)))
            for proto in self.prototypes
        ]
        return [
            topo_features["token_density"],
            topo_features["sentence_density"],
            topo_features["paragraph_density"],
            topo_features["token_sentence_alignment"],
            topo_features["sentence_paragraph_alignment"],
            topo_features["bridge_energy"],
            topo_features["token_sentence_bridge"],
            topo_features["sentence_paragraph_bridge"],
            topo_features["sentence_fragmentation"],
            topo_features["paragraph_fragmentation"],
            topo_features["sentence_fiedler"],
            topo_features["paragraph_fiedler"],
            topo_features["phase_carrier_alignment"],
            topo_features["phase_carrier_decay"],
            q_score,
            i_score,
            self._length_score(text),
            *keyword_counts,
            *(rule_scores[proto.name] for proto in self.prototypes),
        ]

    def feature_vector(self, text: str) -> list[float]:
        return self._feature_vector_from_analysis(text, self.analyze(text))

    def feature_snapshot(self, text: str) -> dict[str, float]:
        snapshot = self._topology_features(text)
        snapshot["length_score"] = self._length_score(text)
        return snapshot

    def _rule_scores_from_features(
        self,
        text: str,
        *,
        tokens: list[str],
        topo_features: dict[str, float],
        q_score: float,
        i_score: float,
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        scores: dict[str, float] = {}
        matches: dict[str, list[str]] = {}
        sent_frag = topo_features["sentence_fragmentation"]
        para_frag = topo_features["paragraph_fragmentation"]

        for proto in self.prototypes:
            hits = self._keyword_matches(text, tokens, proto.keywords)
            matches[proto.name] = hits
            keyword_score = 0.30 * len(hits)
            topology_score = proto.topology_bias * (
                topo_features["sentence_density"]
                + 0.5 * topo_features["paragraph_density"]
                + 0.25 * topo_features["bridge_energy"]
            )
            component_penalty = 0.0
            if proto.name == "analyze":
                component_penalty += 0.20 * sent_frag
            elif proto.name == "compare":
                component_penalty += 0.10 * sent_frag
            elif proto.name == "implement":
                component_penalty -= 0.10 * sent_frag
            elif proto.name == "debug":
                component_penalty += 0.15 * para_frag
            elif proto.name == "plan":
                component_penalty += 0.08 * sent_frag
            elif proto.name == "research":
                component_penalty += 0.06 * para_frag

            scores[proto.name] = (
                keyword_score
                + proto.question_bias * q_score
                + proto.imperative_bias * i_score
                + topology_score
                + component_penalty
            )
        return scores, matches

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
        topo_features = topo if topo is not None else self._topology_features(text)
        return self._rule_scores_from_features(
            text,
            tokens=tokens,
            topo_features=topo_features,
            q_score=q_score,
            i_score=i_score,
        )

    def _effective_temperature(self, modulation: ModulationEffect | None) -> float:
        temperature = self.temperature
        if modulation is not None:
            temperature = modulation.temperature_scale
        deviation = self.state.active_ratio_ema - self.self_measure_target
        temperature *= 1.0 + self.self_measure_strength * deviation
        return _clamp(temperature, 0.05, 4.0)

    def _compute_replay_bias(
        self, feature_vector: list[float]
    ) -> tuple[dict[str, float], float]:
        if not self.state.replay_buffer:
            return {proto.name: 0.0 for proto in self.prototypes}, 1.0
        scored: list[tuple[float, IntentReplayItem]] = []
        for item in self.state.replay_buffer:
            similarity = max(_cosine(feature_vector, item.feature_vector), 0.0)
            scored.append((similarity * item.priority, item))
        scored.sort(key=lambda row: row[0], reverse=True)
        top = scored[: self.replay_topk]
        if not top:
            return {proto.name: 0.0 for proto in self.prototypes}, 1.0
        replay_scores = {proto.name: 0.0 for proto in self.prototypes}
        novelty = 1.0
        for score, item in top:
            novelty = min(novelty, 1.0 - max(score, 0.0))
            replay_scores[item.label] = replay_scores.get(item.label, 0.0) + score
        denom = sum(top_score for top_score, _ in top) or 1.0
        for label in replay_scores:
            replay_scores[label] = self.replay_bias_scale * replay_scores[label] / denom
        return replay_scores, _clamp(novelty, 0.0, 1.0)

    def _compute_trace_bias(self, feature_vector: list[float]) -> dict[str, float]:
        out = {proto.name: 0.0 for proto in self.prototypes}
        for proto in self.prototypes:
            trace = self.state.label_traces.get(proto.name)
            if trace is None:
                continue
            strength = self.state.label_trace_strength.get(proto.name, 0.0)
            out[proto.name] = (
                self.trace_bias_scale * strength * max(_cosine(feature_vector, trace), 0.0)
            )
        return out

    def _predict_from_analysis(
        self,
        text: str,
        analysis: TextTopologyResult,
        *,
        effective_temperature: float,
        replay_scores: dict[str, float] | None = None,
        trace_scores: dict[str, float] | None = None,
    ) -> tuple[IntentPrediction, list[float]]:
        topo_features = self._topology_features_from_analysis(analysis)
        tokens = self._tokenize(text)
        q_score = self._question_score(text)
        i_score = self._imperative_score(text)
        scores, matches = self._rule_scores_from_features(
            text,
            tokens=tokens,
            topo_features=topo_features,
            q_score=q_score,
            i_score=i_score,
        )
        feature_vector = self._feature_vector_from_analysis(text, analysis)
        replay_scores = replay_scores or {proto.name: 0.0 for proto in self.prototypes}
        trace_scores = trace_scores or {proto.name: 0.0 for proto in self.prototypes}
        for proto in self.prototypes:
            scores[proto.name] += self.state.label_bias.get(proto.name, 0.0)
            scores[proto.name] += replay_scores.get(proto.name, 0.0)
            scores[proto.name] += trace_scores.get(proto.name, 0.0)

        probabilities = _softmax_with_temperature(scores, effective_temperature)
        best_label = max(scores, key=scores.__getitem__)
        prediction = IntentPrediction(
            label=best_label,
            confidence=float(probabilities.get(best_label, 0.0)),
            scores=scores,
            matched_keywords=matches,
            topology_snapshot=topo_features,
            probabilities=probabilities,
            replay_scores=replay_scores,
            trace_scores=trace_scores,
            effective_temperature=effective_temperature,
        )
        return prediction, feature_vector

    def _active_ratio_from_prediction(self, prediction: IntentPrediction) -> float:
        if not prediction.probabilities:
            return self.self_measure_target
        values = sorted(prediction.probabilities.values(), reverse=True)
        top_mass = values[0]
        second = values[1] if len(values) > 1 else 0.0
        return _clamp(0.5 * (top_mass + second), 0.0, 1.0)

    def _update_self_measure(self, prediction: IntentPrediction) -> None:
        p_emp = self._active_ratio_from_prediction(prediction)
        alpha = self.self_measure_alpha
        self.state.active_ratio_ema = (
            (1.0 - alpha) * self.state.active_ratio_ema + alpha * p_emp
        )

    def _store_replay(
        self,
        feature_vector: list[float],
        prediction: IntentPrediction,
        *,
        salience: float,
        reward: float,
    ) -> None:
        priority = _clamp(
            0.5 * prediction.confidence + 0.5 * salience + max(reward, 0.0),
            1e-4,
            2.0,
        )
        self.state.replay_buffer.append(
            IntentReplayItem(
                feature_vector=list(feature_vector),
                label=prediction.label,
                confidence=prediction.confidence,
                salience=salience,
                priority=priority,
                reward=reward,
            )
        )
        if len(self.state.replay_buffer) > self.replay_capacity:
            self.state.replay_buffer.sort(key=lambda item: item.priority, reverse=True)
            del self.state.replay_buffer[self.replay_capacity :]

    def _stdp_update(
        self,
        feature_vector: list[float],
        prediction: IntentPrediction,
        *,
        reward: float = 0.0,
        correct_label: str | None = None,
    ) -> None:
        critic = prediction.confidence + reward
        gate = compute_learning_gate(
            critic_score=critic,
            prev_critic_score=self.state.prev_critic_score,
            active_ratio=self.state.active_ratio_ema,
        )
        self.state.prev_critic_score = critic
        if abs(gate) <= 1e-8:
            return

        target_label = correct_label or prediction.label
        for proto in self.prototypes:
            label = proto.name
            old_bias = self.state.label_bias.get(label, 0.0)
            target = 1.0 if label == target_label else -0.25
            activation = prediction.probabilities.get(label, 0.0)
            eligibility = self.stdp_trace_decay * activation + target * prediction.confidence
            update = self.stdp_lr * gate * eligibility
            new_bias = _clamp(old_bias + update, -self.stdp_bias_clip, self.stdp_bias_clip)
            self.state.label_bias[label] = new_bias

        label_trace = self.state.label_traces.get(target_label)
        if label_trace is None:
            self.state.label_traces[target_label] = list(feature_vector)
        else:
            mix = _clamp(self.stdp_lr * max(prediction.confidence + reward, 0.05), 0.01, 0.5)
            merged = [
                (1.0 - mix) * old + mix * new
                for old, new in zip(label_trace, feature_vector)
            ]
            self.state.label_traces[target_label] = _l2_normalise(merged)
        strength = self.state.label_trace_strength.get(target_label, 0.0)
        self.state.label_trace_strength[target_label] = _clamp(
            strength + self.stdp_lr * max(gate, 0.0), 0.0, 1.0
        )

    def predict(self, text: str) -> IntentPrediction:
        analysis = self.analyze(text)
        modulation = apply_modulation(self.state.neuromod, base_temperature=self.temperature)
        effective_temperature = self._effective_temperature(modulation)
        prediction, _ = self._predict_from_analysis(
            text,
            analysis,
            effective_temperature=effective_temperature,
        )
        prediction.state_report = self.bridge_gate_report()
        return prediction

    def predict_step(
        self,
        text: str,
        *,
        mode: str = "WAKE",
        reward: float = 0.0,
        correct_label: str | None = None,
    ) -> IntentPrediction:
        analysis = self.analyze(text)
        feature_vector = self._feature_vector_from_analysis(text, analysis)
        replay_scores, novelty = self._compute_replay_bias(feature_vector)
        trace_scores = self._compute_trace_bias(feature_vector)
        base_salience = _clamp(
            0.5 * self._length_score(text)
            + 0.3 * max(
                self._question_score(text),
                self._imperative_score(text),
            )
            + 0.2 * self._topology_features_from_analysis(analysis)["bridge_energy"],
            0.0,
            1.0,
        )

        modulation = apply_modulation(
            self.state.neuromod,
            base_temperature=self.temperature,
            base_encode_threshold=0.30,
        )
        effective_temperature = self._effective_temperature(modulation)
        prediction, feature_vector = self._predict_from_analysis(
            text,
            analysis,
            effective_temperature=effective_temperature,
            replay_scores=replay_scores,
            trace_scores=trace_scores,
        )

        self.state.step_count += 1
        self.state.mode_occupancy[mode] = self.state.mode_occupancy.get(mode, 0) + 1
        self._update_self_measure(prediction)
        self._store_replay(
            feature_vector,
            prediction,
            salience=base_salience,
            reward=reward,
        )

        self.state.neuromod = step_neuromodulators(
            self.state.neuromod,
            c_pred=reward if correct_label is not None and correct_label == prediction.label else prediction.confidence,
            c_nov=novelty,
            discount=1.0 - prediction.confidence,
            salience=base_salience,
        )
        self._stdp_update(
            feature_vector,
            prediction,
            reward=reward,
            correct_label=correct_label,
        )
        self.state.last_label = prediction.label
        self.state.last_confidence = prediction.confidence
        self.state.last_phase_carrier = {
            "alignment": float(analysis.phase_carrier_alignment),
            "decay": float(analysis.phase_carrier_decay),
        }
        prediction.state_report = self.bridge_gate_report()
        prediction.state_report["novelty"] = novelty
        prediction.state_report["salience"] = base_salience
        return prediction

    def observe_feedback(
        self,
        label: str,
        *,
        reward: float = 1.0,
        text: str | None = None,
    ) -> None:
        if text is not None:
            analysis = self.analyze(text)
            feature_vector = self._feature_vector_from_analysis(text, analysis)
        elif self.state.replay_buffer:
            feature_vector = list(self.state.replay_buffer[-1].feature_vector)
        else:
            return
        dummy = IntentPrediction(
            label=label,
            confidence=max(self.state.last_confidence, 0.5),
            scores={proto.name: self.state.label_bias.get(proto.name, 0.0) for proto in self.prototypes},
            matched_keywords={},
            topology_snapshot={},
            probabilities={},
            effective_temperature=self.temperature,
        )
        self._stdp_update(feature_vector, dummy, reward=reward, correct_label=label)

    def consolidate(self, mode: str = "NREM") -> dict[str, float]:
        if not self.state.replay_buffer:
            return {"replay_items": 0.0, "consolidated": 0.0, "mode": 0.0}
        strength = 1.25 if mode == "REM" else 1.0
        top_items = sorted(
            self.state.replay_buffer,
            key=lambda item: item.priority,
            reverse=True,
        )[: self.replay_topk]
        for item in top_items:
            self._stdp_update(
                item.feature_vector,
                IntentPrediction(
                    label=item.label,
                    confidence=item.confidence,
                    scores={proto.name: 0.0 for proto in self.prototypes},
                    matched_keywords={},
                    topology_snapshot={},
                    probabilities={},
                    effective_temperature=self.temperature,
                ),
                reward=item.reward * strength,
                correct_label=item.label,
            )
            item.priority *= self.consolidation_decay
        for item in self.state.replay_buffer:
            item.priority *= self.consolidation_decay
        return {
            "replay_items": float(len(self.state.replay_buffer)),
            "consolidated": float(len(top_items)),
            "mode": 1.0 if mode == "REM" else 0.0,
        }

    def bridge_gate_report(self) -> dict[str, float]:
        total_modes = sum(self.state.mode_occupancy.values()) or 1
        wake_ratio = self.state.mode_occupancy.get("WAKE", 0) / total_modes
        last = self.state.last_phase_carrier
        return {
            "F1_active_ratio_ema": float(self.state.active_ratio_ema),
            "F1_active_ratio_target": float(self.self_measure_target),
            "F1_deviation": float(self.state.active_ratio_ema - self.self_measure_target),
            "F3_wake_ratio": float(wake_ratio),
            "replay_size": float(len(self.state.replay_buffer)),
            "neuromod_da": float(self.state.neuromod.da),
            "neuromod_ne": float(self.state.neuromod.ne),
            "neuromod_5ht": float(self.state.neuromod.sht),
            "neuromod_ach": float(self.state.neuromod.ach),
            "Pcarrier_alignment": float(last.get("alignment", 0.0)),
            "Pcarrier_decay": float(last.get("decay", 0.0)),
        }

    def session_state_dict(self) -> dict[str, object]:
        return {
            "replay_buffer": [
                {
                    "feature_vector": list(item.feature_vector),
                    "label": item.label,
                    "confidence": float(item.confidence),
                    "salience": float(item.salience),
                    "priority": float(item.priority),
                    "reward": float(item.reward),
                }
                for item in self.state.replay_buffer
            ],
            "active_ratio_ema": float(self.state.active_ratio_ema),
            "label_bias": dict(self.state.label_bias),
            "label_traces": {
                key: list(value) for key, value in self.state.label_traces.items()
            },
            "label_trace_strength": dict(self.state.label_trace_strength),
            "mode_occupancy": dict(self.state.mode_occupancy),
            "prev_critic_score": float(self.state.prev_critic_score),
            "last_label": self.state.last_label,
            "last_confidence": float(self.state.last_confidence),
            "step_count": int(self.state.step_count),
            "neuromod": {
                "da": float(self.state.neuromod.da),
                "ne": float(self.state.neuromod.ne),
                "sht": float(self.state.neuromod.sht),
                "ach": float(self.state.neuromod.ach),
            },
            "last_phase_carrier": dict(self.state.last_phase_carrier),
        }

    def load_session_state_dict(self, state: dict[str, object]) -> None:
        replay_buffer: list[IntentReplayItem] = []
        for row in state.get("replay_buffer", []):
            if not isinstance(row, dict):
                continue
            replay_buffer.append(
                IntentReplayItem(
                    feature_vector=list(row.get("feature_vector", [])),
                    label=str(row.get("label", "")),
                    confidence=float(row.get("confidence", 0.0)),
                    salience=float(row.get("salience", 0.0)),
                    priority=float(row.get("priority", 0.0)),
                    reward=float(row.get("reward", 0.0)),
                )
            )
        neuro_row = state.get("neuromod", {})
        if not isinstance(neuro_row, dict):
            neuro_row = {}
        self.state = IntentSessionState(
            replay_buffer=replay_buffer,
            active_ratio_ema=float(state.get("active_ratio_ema", self.self_measure_target)),
            label_bias={str(k): float(v) for k, v in dict(state.get("label_bias", {})).items()},
            label_traces={
                str(k): list(v) for k, v in dict(state.get("label_traces", {})).items()
            },
            label_trace_strength={
                str(k): float(v)
                for k, v in dict(state.get("label_trace_strength", {})).items()
            },
            mode_occupancy={
                str(k): int(v) for k, v in dict(state.get("mode_occupancy", {})).items()
            }
            or {"WAKE": 0, "NREM": 0, "REM": 0},
            prev_critic_score=float(state.get("prev_critic_score", 0.0)),
            last_label=str(state.get("last_label", "")),
            last_confidence=float(state.get("last_confidence", 0.0)),
            step_count=int(state.get("step_count", 0)),
            neuromod=NeuromodulatorState(
                da=float(neuro_row.get("da", 0.0)),
                ne=float(neuro_row.get("ne", 0.0)),
                sht=float(neuro_row.get("sht", 0.0)),
                ach=float(neuro_row.get("ach", 0.0)),
            ),
            last_phase_carrier={
                str(k): float(v)
                for k, v in dict(state.get("last_phase_carrier", {})).items()
            },
        )
        self._ensure_state_labels()

    def runtime_signal_from_prediction(
        self,
        text: str,
        prediction: IntentPrediction,
        *,
        dim: int,
    ) -> list[float]:
        feat = self.feature_vector(text)
        out = [0.0] * max(int(dim), 1)
        scale = max(prediction.confidence, 1e-6)
        for idx, value in enumerate(feat):
            key = f"{_RUNTIME_HASH_SALT}:{idx}".encode("utf-8")
            digest = hashlib.blake2b(key, digest_size=8).digest()
            bucket = int.from_bytes(digest[:6], "little") % len(out)
            sign = 1.0 if (digest[6] & 1) else -1.0
            out[bucket] += sign * scale * value
        label_digest = hashlib.blake2b(
            f"{_RUNTIME_HASH_SALT}:{prediction.label}".encode("utf-8"),
            digest_size=8,
        ).digest()
        bucket = int.from_bytes(label_digest[:6], "little") % len(out)
        out[bucket] += prediction.confidence
        return _l2_normalise(out)

    def fit_centroids(
        self,
        examples: list[LabeledIntentExample],
        *,
        temperature: float | None = None,
    ) -> FittedTopologyIntentClassifier:
        grouped: dict[str, list[list[float]]] = {}
        for example in examples:
            grouped.setdefault(example.label, []).append(
                self.feature_vector(example.text)
            )

        labels = sorted(grouped)
        centroids: dict[str, list[float]] = {}
        for label in labels:
            rows = grouped[label]
            width = len(rows[0])
            centroids[label] = [
                sum(row[idx] for row in rows) / len(rows) for idx in range(width)
            ]
        if temperature is None:
            scatter = self._mean_intra_class_distance(grouped, centroids)
            temperature = max(scatter, 1e-3)
        return FittedTopologyIntentClassifier(
            labels=labels,
            centroids=centroids,
            parent=self,
            temperature=float(temperature),
        )

    @staticmethod
    def _mean_intra_class_distance(
        grouped: dict[str, list[list[float]]],
        centroids: dict[str, list[float]],
    ) -> float:
        total = 0.0
        counted = 0
        for label, rows in grouped.items():
            centroid = centroids[label]
            for row in rows:
                dist2 = sum((a - b) * (a - b) for a, b in zip(row, centroid))
                total += dist2
                counted += 1
        if counted == 0:
            return 1.0
        return total / counted


__all__ = [
    "FittedTopologyIntentClassifier",
    "IntentPrediction",
    "IntentPrototype",
    "IntentReplayItem",
    "IntentSessionState",
    "LabeledIntentExample",
    "TopologyIntentClassifier",
]
