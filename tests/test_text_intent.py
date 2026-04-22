"""Regression tests for the topology-aware intent classifier."""

from __future__ import annotations

import importlib.util
import pickle
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def intent_module():
    sys.modules.setdefault("clarus", types.ModuleType("clarus"))
    _load("clarus.constants", ROOT / "clarus" / "constants.py")
    _load("clarus.neuromod", ROOT / "clarus" / "neuromod.py")
    _load("clarus.stdp", ROOT / "clarus" / "stdp.py")
    _load("clarus.text_topology", ROOT / "clarus" / "text_topology.py")
    return _load("clarus.text_intent", ROOT / "clarus" / "text_intent.py")


@pytest.fixture
def classifier(intent_module):
    return intent_module.TopologyIntentClassifier(dim=24)


@pytest.fixture
def training_set(intent_module):
    Example = intent_module.LabeledIntentExample
    return [
        Example("이게 왜 맞는지 설명해줘.", "explain"),
        Example("개념과 원리를 설명해줘.", "explain"),
        Example("증명을 유도해줘.", "explain"),
        Example("프로토타입을 하나 구현해봐.", "implement"),
        Example("새 모듈 하나 만들어줘.", "implement"),
        Example("스크립트 작성해줘.", "implement"),
        Example("최신 구조를 분석해줘.", "analyze"),
        Example("정합한지 검토해줘.", "analyze"),
        Example("완성도 평가해줘.", "analyze"),
        Example("둘 중 뭐가 나은지 비교해줘.", "compare"),
        Example("가장 해볼만한 걸 추천해줘.", "compare"),
        Example("에러나는 부분 고쳐줘.", "debug"),
        Example("버그를 수정해줘.", "debug"),
        Example("왜 이게 안 돼?", "debug"),
        Example("다음 단계 계획 세워줘.", "plan"),
        Example("우선순위와 로드맵을 짜줘.", "plan"),
        Example("이 모듈 리팩토링해줘.", "refactor"),
        Example("코드 정리하고 개선해줘.", "refactor"),
        Example("관련 논문 조사해줘.", "research"),
        Example("주제 관련 리서치해봐.", "research"),
    ]


def test_rule_based_predictions(classifier):
    cases = [
        ("의도분류 프로토타입 하나 구현해봐.", "implement"),
        ("최신 커밋 기준으로 bitfield 부분 분석해줘.", "analyze"),
        ("3d, 강화학습, OCR 중 뭐가 가장 해볼만한지 비교해줘.", "compare"),
        ("이 코드 에러나는 부분 고쳐봐.", "debug"),
        ("리만 가설 증명 좀 해줘.", "explain"),
        ("다음 단계 계획 세워줘.", "plan"),
        ("이 모듈 리팩토링하고 다듬어줘.", "refactor"),
        ("관련 논문들 좀 조사해봐.", "research"),
    ]
    correct = 0
    for text, expected in cases:
        prediction = classifier.predict(text)
        if prediction.label == expected:
            correct += 1
    assert correct >= 6


def test_softmax_probabilities_sum_to_one(classifier):
    prediction = classifier.predict("이 모듈 리팩토링해줘.")
    total = sum(prediction.probabilities.values())
    assert pytest.approx(total, rel=1e-6) == 1.0
    assert prediction.confidence == pytest.approx(prediction.probabilities[prediction.label])


def test_centroid_classifier_perfect_on_training(classifier, training_set):
    fitted = classifier.fit_centroids(training_set)
    correct = 0
    for example in training_set:
        prediction = fitted.predict(example.text)
        if prediction.label == example.label:
            correct += 1
    assert correct == len(training_set)


def test_centroid_classifier_temperature_calibrated(classifier, training_set):
    fitted = classifier.fit_centroids(training_set)
    assert fitted.temperature > 0.0
    prediction = fitted.predict(training_set[0].text)
    assert 0.0 < prediction.confidence <= 1.0


def test_centroid_classifier_is_pickleable(classifier, training_set):
    fitted = classifier.fit_centroids(training_set)
    blob = pickle.dumps(fitted)
    restored = pickle.loads(blob)
    sample = "이 코드 에러나는 부분 고쳐봐."
    a = fitted.predict(sample)
    b = restored.predict(sample)
    assert a.label == b.label
    assert a.confidence == pytest.approx(b.confidence, rel=1e-6)


def test_feature_vector_uses_single_topology_pass(classifier, monkeypatch):
    calls = {"count": 0}
    real_analyze = classifier.engine.analyze

    def counting(text):
        calls["count"] += 1
        return real_analyze(text)

    monkeypatch.setattr(classifier.engine, "analyze", counting)
    classifier.predict("이 구조를 분석해줘.")
    assert calls["count"] == 1


def test_topology_features_include_new_axes(classifier):
    snapshot = classifier.feature_snapshot("Alpha bridges concept. Beta closes it.")
    expected = {
        "token_sentence_bridge",
        "sentence_paragraph_bridge",
        "sentence_fragmentation",
        "paragraph_fragmentation",
        "sentence_fiedler",
        "paragraph_fiedler",
        "length_score",
    }
    assert expected.issubset(snapshot.keys())
