"""Regression tests for the hashing TF-IDF encoder."""

from __future__ import annotations

import importlib.util
import math
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
def lexical_module():
    sys.modules.setdefault("clarus", types.ModuleType("clarus"))
    return _load("clarus.text_lexical", ROOT / "clarus" / "text_lexical.py")


def test_unfitted_encoder_raises(lexical_module):
    encoder = lexical_module.LexicalEncoder(dim=64)
    assert not encoder.is_fitted
    with pytest.raises(RuntimeError):
        encoder.encode("hello world")


def test_encoded_vector_is_unit_norm(lexical_module):
    encoder = lexical_module.LexicalEncoder(dim=128).fit(
        [
            "alpha beta gamma",
            "beta gamma delta",
            "gamma delta epsilon",
            "epsilon zeta eta",
        ]
    )
    vec = encoder.encode("alpha beta gamma")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, rel=1e-6)


def test_idf_downweights_common_terms(lexical_module):
    corpus = [
        "alpha beta gamma",
        "alpha delta epsilon",
        "alpha eta theta",
        "alpha iota kappa",
        "rare unique term",
    ]
    encoder = lexical_module.LexicalEncoder(dim=512, ngram_range=(1, 1)).fit(corpus)
    idf = encoder.idf
    alpha_bucket, _ = encoder._hash("alpha")
    rare_bucket, _ = encoder._hash("rare")
    assert idf[rare_bucket] > idf[alpha_bucket]


def test_empty_text_returns_zero_vector(lexical_module):
    encoder = lexical_module.LexicalEncoder(dim=32).fit(["alpha", "beta"])
    vec = encoder.encode("")
    assert all(x == 0.0 for x in vec)


def test_signed_buckets_can_cancel(lexical_module):
    encoder = lexical_module.LexicalEncoder(dim=32, sublinear_tf=False).fit(
        ["a a a"]
    )
    raw = encoder._signed_counts("a a a")
    bucket, sign = encoder._hash("a")
    assert raw[bucket] == 3.0 * sign


def test_encoder_is_deterministic(lexical_module):
    encoder = lexical_module.LexicalEncoder(dim=64).fit(["hello there friend"])
    a = encoder.encode("hello there")
    b = encoder.encode("hello there")
    assert a == b
