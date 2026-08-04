from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest

from reality_stone.clarus.origin_life_branching import (
    BranchingParameters,
    PARAMETERS,
    age_structured_mean_matrix,
    build_branching_certificate,
    complete_daughter_distribution,
    division_offspring_distribution,
    expected_complete_daughters,
    minimum_supercritical_copy_number,
    reset_census_two_tick_matrix,
    slow_sublineage_extinction_probability,
    total_extinction_probability,
    validate_branching_certificate,
)
from reality_stone.clarus.origin_life_coupled import expected_two_tick_transition


def test_correlated_partition_and_mutation_kernel_are_exact() -> None:
    probability_zero, probability_one, probability_two = (
        complete_daughter_distribution(7, 4)
    )
    assert (probability_zero, probability_one, probability_two) == (
        Fraction(16065105, 134217728),
        Fraction(65445871, 134217728),
        Fraction(52706752, 134217728),
    )
    assert probability_zero + probability_one + probability_two == 1
    assert expected_complete_daughters(7, 4) == Fraction(
        170859375, 134217728
    )

    specified = expected_complete_daughters(7, 4) / 2
    assert probability_two - specified**2 == Fraction(
        -896204010592801, 72057594037927936
    )

    fast = division_offspring_distribution("fast")
    slow = division_offspring_distribution("slow")
    assert sum(fast.values()) == 1
    assert sum(slow.values()) == 1
    assert sum(
        fast_count * probability
        for (fast_count, _), probability in slow.items()
    ) == expected_complete_daughters(7, 4) * Fraction(1, 16)
    assert sum(
        slow_count * probability
        for (_, slow_count), probability in slow.items()
    ) == expected_complete_daughters(7, 4) * Fraction(15, 16)


def test_age_structured_mean_operator_integrates_both_old_components() -> None:
    reproduction = Fraction(170859375, 134217728)
    nu = Fraction(1, 16)
    one_tick = age_structured_mean_matrix()
    assert one_tick == (
        (reproduction, Fraction(0), reproduction * nu),
        (Fraction(0), Fraction(0), reproduction * (1 - nu)),
        (Fraction(0), Fraction(1), Fraction(0)),
    )
    two_tick = reset_census_two_tick_matrix()
    assert two_tick == (
        (reproduction**2, reproduction * nu),
        (Fraction(0), reproduction * (1 - nu)),
    )
    assert reproduction**2 - reproduction * (1 - nu) == Fraction(
        7693841225390625, 18014398509481984
    )

    squared = tuple(
        tuple(
            sum(one_tick[row][inner] * one_tick[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )
    newborn_restriction = tuple(
        tuple(squared[row][column] for column in (0, 1))
        for row in (0, 1)
    )
    assert newborn_restriction == two_tick


def test_perfect_partition_limit_matches_prior_coupled_transition_api() -> None:
    certificate = build_branching_certificate()
    rows = certificate["proof_obligations"]["perfect_partition_limit"][
        "recovered_two_tick_matrix"
    ]
    matrix = tuple(
        tuple(Fraction(cell["exact"]) for cell in row)
        for row in rows
    )
    nu = Fraction(1, 16)

    assert expected_two_tick_transition(1, 0, nu) == (
        matrix[0][0],
        matrix[1][0],
    )
    assert expected_two_tick_transition(0, 1, nu) == (
        matrix[0][1],
        matrix[1][1],
    )


def test_copy_threshold_and_founder_extinction_probabilities_are_exact() -> None:
    assert minimum_supercritical_copy_number(7) == 4
    assert expected_complete_daughters(7, 3) < 1
    assert total_extinction_probability(7, 3) == 1

    extinction = Fraction(2295015, 7529536)
    survival = Fraction(5234521, 7529536)
    assert total_extinction_probability() == extinction
    assert 1 - total_extinction_probability() == survival

    certificate = build_branching_certificate()
    total = certificate["proof_obligations"]["total_sample_path_extinction"]
    assert total["fast_founder_extinction_probability"]["exact"] == str(extinction)
    assert total["slow_founder_extinction_probability"]["exact"] == str(extinction)
    assert total["positive_survival_probability"]["exact"] == str(survival)
    assert not total["certain_survival"]


def test_slow_lineage_persistence_refutes_almost_sure_strict_fixation() -> None:
    slow_extinction = Fraction(325781723, 741188700)
    slow_survival = Fraction(415406977, 741188700)
    assert slow_sublineage_extinction_probability() == slow_extinction
    assert 1 - slow_sublineage_extinction_probability() == slow_survival

    certificate = build_branching_certificate()
    slow = certificate["proof_obligations"]["persistent_slow_sublineage"]
    assert slow["initial_condition"] == "one_slow_newborn_founder"
    assert slow["slow_sublineage_survival_probability"]["exact"] == str(
        slow_survival
    )
    assert slow[
        "probability_slow_persists_from_slow_founder_given_total_survival"
    ]["exact"] == (
        "6646511632/8244370575"
    )
    assert slow[
        "probability_strict_fast_fixation_from_slow_founder_given_total_survival"
    ]["exact"] == "1597858943/8244370575"
    assert not slow["strict_fast_fixation_from_slow_founder_almost_sure"]
    assert not slow["relative_fast_frequency_limit_proven"]


def test_degenerate_immortal_chain_and_input_guards_are_explicit() -> None:
    assert complete_daughter_distribution(1, 1) == (
        Fraction(0),
        Fraction(1),
        Fraction(0),
    )
    assert total_extinction_probability(1, 1) == 0
    assert slow_sublineage_extinction_probability(1, 1, Fraction(0)) == 0

    with pytest.raises(ValueError):
        complete_daughter_distribution(0, 4)
    with pytest.raises(ValueError):
        complete_daughter_distribution(7, 0)
    with pytest.raises(ValueError):
        division_offspring_distribution("unknown")
    with pytest.raises(ValueError):
        division_offspring_distribution("slow", mutation_probability=Fraction(2))
    with pytest.raises(ValueError, match="fixed .* schema"):
        age_structured_mean_matrix(BranchingParameters(fast_cycle_ticks=2))
    with pytest.raises(ValueError, match="fixed .* schema"):
        reset_census_two_tick_matrix(BranchingParameters(slow_cycle_ticks=3))


def test_certificate_is_deterministic_and_fails_closed_on_overclaim() -> None:
    certificate = build_branching_certificate()
    assert certificate["all_exact_model_obligations_passed"]
    assert certificate["claim_scope"][
        "unified_age_structured_sample_path_process_defined"
    ]
    assert certificate["claim_scope"][
        "almost_sure_strict_fast_fixation_from_slow_founder_refuted"
    ]
    assert not certificate["claim_scope"]["relative_frequency_fixation_proven"]
    assert not certificate["claim_scope"]["empirical_autonomous_protocell_proven"]
    assert validate_branching_certificate(certificate)

    tampered = deepcopy(certificate)
    tampered["claim_scope"]["empirical_autonomous_protocell_proven"] = True
    assert not validate_branching_certificate(tampered)
    assert PARAMETERS.fast_cycle_ticks == 1
    assert PARAMETERS.slow_cycle_ticks == 2
