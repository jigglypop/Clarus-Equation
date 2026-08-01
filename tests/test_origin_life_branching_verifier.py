from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from reality_stone.clarus.origin_life_branching import (
    build_branching_certificate,
)
from reality_stone.clarus.origin_life_branching_verifier import (
    independently_verified,
    verify_branching_certificate,
)


def _certificate() -> dict[str, object]:
    return build_branching_certificate()


def test_independent_verifier_recomputes_all_eight_obligations() -> None:
    report = verify_branching_certificate(_certificate())

    assert report.verified, report.errors
    assert len(report.checks) == 8


@pytest.mark.parametrize(
    "tamper",
    [
        lambda certificate: certificate["model"]["semantics"].__setitem__(
            "slow_age_transition", "slow newborn divides immediately"
        ),
        lambda certificate: certificate["proof_obligations"][
            "partition_mutation_kernel"
        ]["complete_daughter_distribution"]["P_X_2"].__setitem__(
            "exact", "1/2"
        ),
        lambda certificate: certificate["proof_obligations"][
            "age_structured_mean_operator"
        ]["one_tick_mean_matrix"][2][1].__setitem__("exact", "0"),
        lambda certificate: certificate["proof_obligations"][
            "total_sample_path_extinction"
        ]["slow_founder_extinction_probability"].__setitem__(
            "exact", "1/2"
        ),
        lambda certificate: certificate["proof_obligations"][
            "persistent_slow_sublineage"
        ].__setitem__(
            "strict_fast_fixation_from_slow_founder_almost_sure", True
        ),
        lambda certificate: certificate["proof_obligations"][
            "persistent_slow_sublineage"
        ].__setitem__("initial_condition", "one_fast_newborn_founder"),
        lambda certificate: certificate["proof_obligations"][
            "total_sample_path_extinction"
        ].__setitem__("embedded_generation_argument", "trust the builder"),
        lambda certificate: certificate["proof_obligations"][
            "persistent_slow_sublineage"
        ].__setitem__("strict_fast_fixation_definition", "frequency exceeds half"),
        lambda certificate: certificate.__setitem__("unexpected_top_level", True),
        lambda certificate: certificate["proof_obligations"].__setitem__(
            "fabricated_obligation", {"passed": True}
        ),
        lambda certificate: certificate["proof_obligations"][
            "total_sample_path_extinction"
        ]["positive_survival_probability"].__setitem__("decimal", float("nan")),
        lambda certificate: certificate["claim_scope"].__setitem__(
            "relative_frequency_fixation_proven", True
        ),
        lambda certificate: certificate["model"]["parameters"].__setitem__(
            "forward_mutation_probability", 0.0625
        ),
    ],
)
def test_independent_verifier_rejects_numeric_semantic_and_scope_tampering(
    tamper,
) -> None:
    certificate = deepcopy(_certificate())
    tamper(certificate)

    assert not independently_verified(certificate)


def test_independent_verifier_fails_closed_on_missing_negative_claim() -> None:
    certificate = deepcopy(_certificate())
    del certificate["claim_scope"]["finite_resource_survival_proven"]

    report = verify_branching_certificate(certificate)
    assert not report.verified
    assert any("claim_scope" in error for error in report.errors)


def test_committed_artifact_matches_builder_and_independent_verifier() -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "biology"
        / "origin_life_branching_certificate.json"
    )
    observed = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert observed == _certificate()
    report = verify_branching_certificate(observed)
    assert report.verified, report.errors
