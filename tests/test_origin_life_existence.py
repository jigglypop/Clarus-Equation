from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

from reality_stone.clarus.origin_life_existence import (
    WITNESSES,
    build_existence_certificate,
    exact_hybrid_step,
    validate_existence_certificate,
)
from reality_stone.clarus.origin_life_existence_verifier import (
    verify_existence_certificate,
)


def test_two_exact_witnesses_are_distinct_dividing_fixed_lineages() -> None:
    next_states = []
    for witness in WITNESSES:
        next_state, divided, predivision_mass = exact_hybrid_step(witness)
        assert divided
        assert predivision_mass == 1
        assert next_state == witness
        next_states.append(next_state)

    assert next_states[1][2] - next_states[0][2] == Fraction(1, 2)


def test_certificate_separates_base_existence_from_conditional_ablations() -> None:
    certificate = build_existence_certificate()

    assert certificate["invariant_box"]["passed"]
    assert certificate["local_stability"]["passed"]
    assert certificate["structural_robustness"]["open_parameter_neighborhood_exists"]
    assert certificate["conditional_core_ablation_suite_passed"]
    assert certificate["base_existence_theorem_proven"]
    assert certificate["all_exact_model_theorems_passed"]
    assert certificate["existence_theorem_proven"]
    assert not certificate["universal_necessity_proven"]
    assert not certificate["empirical_autonomous_protocell_proven"]


def test_global_lineage_basins_and_reproductive_rectangle_are_certified() -> None:
    certificate = build_existence_certificate()
    heredity = certificate["global_heredity_dynamics"]
    basin = certificate["certified_reproductive_basin"]

    assert heredity["passed"]
    assert heredity["stable_lineage_count"] == 2
    assert heredity["log2_stable_label_count"] == 1
    assert basin["passed"]
    assert basin["division_every_generation"]
    assert basin["certified_basin_volume_lower_bound"]["exact"] == "2/99"
    contraction = basin["contraction_rectangle_R1"]
    assert [row["exact"] for row in contraction["jacobian_row_bounds"]] == [
        "175/176",
        "659/660",
    ]


def test_fixed_point_classification_and_extinction_wedge_are_exact() -> None:
    certificate = build_existence_certificate()
    classification = certificate["fixed_point_classification"]
    extinction = certificate["extinction_boundary"]

    assert classification["passed"]
    assert [row["exact"] for row in classification["divided_mass_roots"]] == [
        "-2/9",
        "1/2",
    ]
    assert classification["locally_stable_reproductive_fixed_states"] == 2
    assert classification["full_fixed_state_count"] == 6
    assert extinction["passed"]
    assert extinction["boundary_zero_threshold"]["exact"] == "2/3"
    assert extinction["mass_boundary_area"]["exact"] == "1/10"
    assert not extinction["global_survival_proven"]


def test_parameter_box_ablations_and_symmetric_branching_are_exact() -> None:
    certificate = build_existence_certificate()
    parameter_box = certificate["structural_robustness"][
        "explicit_closed_parameter_box"
    ]
    ablations = certificate["single_term_ablation_lemmas"]
    branching = certificate["branching_lineage"]

    assert parameter_box["passed"]
    assert parameter_box["positive_width_dimensions"] == 7
    assert parameter_box["uniform_contraction_bound"]["exact"] == "11363/12500"
    assert ablations["no_boundary_production"][
        "generations_until_boundary_at_most_one_quarter"
    ] == 14
    assert ablations["no_boundary_production"][
        "post_decay_maximum_predivision_mass"
    ]["exact"] == "841/1152"
    assert ablations["no_inheritance"]["one_step_heredity"]["exact"] == "1/2"
    assert branching["passed"]
    assert branching["descendants_after_n_generations"] == "2^n"


def test_certificate_validation_rejects_claim_tampering() -> None:
    certificate = build_existence_certificate()
    assert validate_existence_certificate(certificate)

    tampered = deepcopy(certificate)
    tampered["empirical_autonomous_protocell_proven"] = True
    assert not validate_existence_certificate(tampered)


def test_independent_verifier_checks_v6_basin_and_parameter_box() -> None:
    certificate = build_existence_certificate()
    report = verify_existence_certificate(certificate)

    assert report.verified, report.errors
    assert len(report.checks) == 11

    tampered = deepcopy(certificate)
    tampered["certified_reproductive_basin"]["entry_rectangle_R0"][
        "predivision_mass_bounds"
    ][0]["exact"] = "4/5"
    assert not verify_existence_certificate(tampered).verified
