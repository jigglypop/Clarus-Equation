from __future__ import annotations

import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from reality_stone.clarus.origin_life_branching import (
    complete_daughter_distribution,
)
from reality_stone.clarus.origin_life_finite_resource import (
    CONDITIONS,
    FiniteResourceParameters,
    build_finite_resource_certificate,
    finite_generation_survival_probability,
    sample_complete_daughter_count,
    simulate_finite_resource_lineage,
    validate_finite_resource_certificate,
)
from reality_stone.clarus.origin_life_finite_resource_verifier import (
    independently_verified,
    verify_finite_resource_certificate,
)


@pytest.fixture(scope="module")
def certificate() -> dict[str, object]:
    return build_finite_resource_certificate()


def test_explicit_copy_runner_preserves_integer_material_and_lineage_events() -> None:
    first = simulate_finite_resource_lineage(seed=3, include_events=True)
    second = simulate_finite_resource_lineage(seed=3, include_events=True)
    different = simulate_finite_resource_lineage(seed=4, include_events=True)

    assert first == second
    assert first != different
    assert first["mass_balance_error_max"] == 0
    assert first["template_ledger_residual"] == 0
    assert first["boundary_ledger_residual"] == 0
    assert first["module_partition_residual_max"] == 0
    assert first["template_synthesis_events"] > 0
    assert first["boundary_synthesis_events"] > 0
    assert first["initial_total_mass"] == first["final_total_mass"]
    assert first["max_population"] <= first["population_mass_bound"]
    assert first["division_events"] > 0
    division_events = [event for event in first["events"] if event["kind"] == "division"]
    assert len(division_events) == first["division_events"]
    assert all(event["cell_id"] is not None for event in division_events)
    assert all("daughter_ids" in event for event in division_events)
    assert all("parent_module_copies" in event for event in division_events)
    assert all("daughter_module_copies" in event for event in division_events)
    assert all("partition_rng_key" in event for event in division_events)


def test_counter_based_lineage_partition_is_invariant_to_unrelated_founders() -> None:
    for seed in range(8):
        isolated = simulate_finite_resource_lineage(
            seed=seed,
            founder_count=1,
            include_events=True,
        )
        paired = simulate_finite_resource_lineage(
            seed=seed,
            founder_count=2,
            include_events=True,
        )

        def root_division(result: dict[str, object]) -> dict[str, object]:
            return next(
                event
                for event in result["events"]
                if event["kind"] == "division"
                and event["lineage_id"] == 0
                and event["lineage_path"] == [0]
            )

        isolated_event = root_division(isolated)
        paired_event = root_division(paired)
        for field in (
            "partition_rng_key",
            "parent_module_copies",
            "daughter_module_copies",
            "daughter_complete",
        ):
            assert paired_event[field] == isolated_event[field]


def test_partition_oracle_recovers_the_existing_exact_branching_kernel() -> None:
    existing = complete_daughter_distribution(7, 4)
    certificate_survival = finite_generation_survival_probability(3)
    p0, p1, p2 = existing
    extinct = Fraction(0)
    for _ in range(3):
        extinct = p0 + p1 * extinct + p2 * extinct**2

    assert certificate_survival == 1 - extinct
    assert 0 < certificate_survival < 1
    assert all(sample_complete_daughter_count(seed) in (0, 1, 2) for seed in range(64))


@pytest.mark.parametrize(
    "condition_name",
    ["no_uptake", "no_template_copying", "no_boundary_synthesis"],
)
def test_paired_term_ablation_prevents_division(condition_name: str) -> None:
    result = simulate_finite_resource_lineage(
        seed=0,
        condition=CONDITIONS[condition_name],
    )
    assert result["division_events"] == 0
    assert result["max_generation"] == 0


def test_zero_resource_causes_registered_starvation_death() -> None:
    parameters = FiniteResourceParameters(
        initial_external_resource=0,
        reservoir_resource=0,
        inflow_per_tick=0,
        horizon_ticks=8,
    )
    result = simulate_finite_resource_lineage(seed=0, parameters=parameters)

    assert result["final_population"] == 0
    assert result["starvation_deaths"] == 1
    assert result["mass_balance_error_max"] == 0


def test_certificate_passes_all_engineering_gates_without_overclaim(
    certificate: dict[str, object],
) -> None:
    assert certificate["all_engineering_gates_passed"]
    assert all(gate["passed"] for gate in certificate["gates"].values())
    conservation = certificate["gates"]["integer_material_conservation"]
    assert conservation["max_balance_error"] == 0
    assert conservation["max_template_ledger_residual"] == 0
    assert conservation["max_boundary_ledger_residual"] == 0
    partition = certificate["gates"]["exact_partition_bridge"]
    assert partition["sampler_draws"] == 8192
    assert sum(partition["sampler_counts_P0_P1_P2"]) == 8192
    assert partition["sampler_max_absolute_error"] <= partition["sampler_tolerance"]
    scope = certificate["claim_scope"]
    assert scope["finite_resource_accounting_proven_for_declared_simulator"]
    assert scope["explicit_token_consuming_copy_bookkeeping_implemented"]
    assert not scope["indefinite_survival_proven"]
    assert not scope["autonomous_metabolism_proven"]
    assert not scope["empirical_autonomous_protocell_proven"]
    assert validate_finite_resource_certificate(certificate)


def test_independent_verifier_recomputes_registered_seed_panels(
    certificate: dict[str, object],
) -> None:
    report = verify_finite_resource_certificate(certificate)

    assert report.verified, report.errors
    assert len(report.checks) == 8


@pytest.mark.parametrize(
    "tamper",
    [
        lambda value: value["model"]["parameters"].__setitem__(
            "reservoir_resource", 999999
        ),
        lambda value: value["gates"]["integer_material_conservation"].__setitem__(
            "max_balance_error", 1
        ),
        lambda value: value["gates"]["density_competition"]["high_density"].__setitem__(
            "mean_divisions_per_founder", 0.0
        ),
        lambda value: value["claim_scope"].__setitem__(
            "empirical_autonomous_protocell_proven", True
        ),
        lambda value: value.__setitem__("unexpected", True),
    ],
)
def test_independent_verifier_rejects_numeric_scope_and_schema_tampering(
    certificate: dict[str, object],
    tamper,
) -> None:
    changed = deepcopy(certificate)
    tamper(changed)

    assert not independently_verified(changed)


def test_independent_verifier_rejects_nonfinite_fields(
    certificate: dict[str, object],
) -> None:
    changed = deepcopy(certificate)
    changed["gates"]["explicit_copy_recurrence"][
        "observed_generation_3_survival"
    ] = float("nan")

    report = verify_finite_resource_certificate(changed)
    assert not report.verified
    assert any("non-finite" in error for error in report.errors)


def test_committed_finite_resource_artifact_matches_builder(
    certificate: dict[str, object],
) -> None:
    artifact_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "biology"
        / "origin_life_finite_resource_certificate.json"
    )
    observed = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert observed == certificate
    assert independently_verified(observed)


def test_input_guards_reject_invalid_resource_schemas() -> None:
    with pytest.raises(ValueError, match="founder_count"):
        simulate_finite_resource_lineage(seed=0, founder_count=0)
    with pytest.raises(ValueError, match="initial_external_resource"):
        simulate_finite_resource_lineage(
            seed=0,
            parameters=FiniteResourceParameters(
                initial_external_resource=65,
                external_capacity=64,
            ),
        )
    with pytest.raises(ValueError, match="two-daughter"):
        simulate_finite_resource_lineage(
            seed=0,
            parameters=FiniteResourceParameters(target_boundary_units=3),
        )
