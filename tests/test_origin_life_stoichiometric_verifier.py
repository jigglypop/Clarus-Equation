from __future__ import annotations

from copy import deepcopy

import reality_stone.clarus.origin_life_stoichiometric as builder_module
from reality_stone.clarus.origin_life_stoichiometric import (
    build_stoichiometric_certificate,
)
from reality_stone.clarus.origin_life_stoichiometric_verifier import (
    verify_stoichiometric_certificate,
)


def test_independent_verifier_recomputes_canonical_v2_certificate() -> None:
    certificate = build_stoichiometric_certificate()

    report = verify_stoichiometric_certificate(certificate)

    assert certificate["artifact_version"] == 2
    assert certificate["all_stoichiometric_gates_passed"]
    assert report.verified, report.errors
    assert report.checks == (
        "schema_and_finite_numbers",
        "artifact_type",
        "artifact_version",
        "arithmetic",
        "model",
        "gates",
        "claim_scope",
        "all_stoichiometric_gates_passed",
    )


def test_independent_verifier_does_not_delegate_to_builder(monkeypatch) -> None:
    certificate = build_stoichiometric_certificate()
    monkeypatch.setattr(
        builder_module,
        "build_stoichiometric_certificate",
        lambda: {"artifact_version": -1},
    )

    report = verify_stoichiometric_certificate(certificate)

    assert report.verified, report.errors


def test_independent_verifier_rejects_version_and_heat_ledger_tampering() -> None:
    certificate = build_stoichiometric_certificate()

    stale = deepcopy(certificate)
    stale["artifact_version"] = 1
    stale_report = verify_stoichiometric_certificate(stale)

    changed_heat = deepcopy(certificate)
    changed_heat["gates"]["closed_batch_ledger"]["heat_quanta"] += 1
    heat_report = verify_stoichiometric_certificate(changed_heat)

    assert not stale_report.verified
    assert stale_report.errors == ("artifact_version differs from independent recomputation",)
    assert not heat_report.verified
    assert heat_report.errors == ("gates differs from independent recomputation",)


def test_independent_verifier_rejects_nonfinite_geometry_readout() -> None:
    certificate = build_stoichiometric_certificate()
    changed = deepcopy(certificate)
    changed["gates"]["division_geometry_and_conservation"]["parent_geometry"]["volume_m3"] = float(
        "nan"
    )

    report = verify_stoichiometric_certificate(changed)

    assert not report.verified
    assert report.errors == (
        "certificate.gates.division_geometry_and_conservation.parent_geometry.volume_m3 "
        "contains a non-finite number",
    )
