from __future__ import annotations

from copy import deepcopy

import pytest

from reality_stone.clarus.origin_life_coupled import build_coupled_certificate
from reality_stone.clarus.origin_life_coupled_verifier import (
    independently_verified,
    verify_coupled_certificate,
)


def _certificate() -> dict[str, object]:
    return build_coupled_certificate()


def test_independent_verifier_recomputes_all_eight_obligations() -> None:
    report = verify_coupled_certificate(_certificate())

    assert report.verified, report.errors
    assert len(report.checks) == 8


@pytest.mark.parametrize(
    "tamper",
    [
        lambda certificate: certificate["model"]["semantics"].__setitem__(
            "division_comparator", ">"
        ),
        lambda certificate: certificate["proof_obligations"][
            "open_parameter_plateau"
        ]["fast_one_tick_lower_margin"].__setitem__("exact", "1/32"),
        lambda certificate: certificate["proof_obligations"][
            "division_gated_mutation_selection"
        ]["two_tick_matrix"][0][1].__setitem__("exact", "1/16"),
        lambda certificate: certificate["proof_obligations"][
            "stochastic_partition_threshold"
        ]["rows"][3]["both_daughters_complete_probability"].__setitem__(
            "exact", "29192926025390625/72057594037927936"
        ),
        lambda certificate: certificate["claim_scope"].__setitem__(
            "sample_path_fixation_proven", True
        ),
        lambda certificate: certificate["model"]["parameters"].__setitem__(
            "growth_slope", 0.5
        ),
    ],
)
def test_independent_verifier_rejects_semantic_numeric_and_scope_tampering(
    tamper,
) -> None:
    certificate = deepcopy(_certificate())
    tamper(certificate)

    assert not independently_verified(certificate)


def test_independent_verifier_fails_closed_on_missing_negative_claim() -> None:
    certificate = deepcopy(_certificate())
    del certificate["claim_scope"]["endogenous_mutation_chemistry_proven"]

    report = verify_coupled_certificate(certificate)
    assert not report.verified
    assert any("claim_scope" in error for error in report.errors)
