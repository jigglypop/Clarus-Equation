from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest

from reality_stone.clarus.origin_life_coupled import (
    INITIAL_PHASE,
    PARAMETERS,
    CellState,
    both_daughters_complete_probability,
    build_coupled_certificate,
    exact_cell_tick,
    exact_descendant_count,
    expected_complete_daughters,
    expected_counts_after_blocks,
    expected_two_tick_transition,
    growth_increment,
    minimum_supercritical_copy_number,
    specified_daughter_complete_probability,
    validate_coupled_certificate,
)


def test_transmitted_type_changes_the_cell_cycle_with_positive_margins() -> None:
    assert growth_increment(PARAMETERS.slow_type) == Fraction(5, 16)
    assert growth_increment(PARAMETERS.fast_type) == Fraction(9, 16)
    assert (
        growth_increment(PARAMETERS.fast_type)
        - growth_increment(PARAMETERS.slow_type)
        == Fraction(1, 4)
    )

    fast = exact_cell_tick(CellState(INITIAL_PHASE, PARAMETERS.fast_type))
    assert fast.divided
    assert fast.predivision_phase == Fraction(17, 16)
    assert len(fast.daughters) == 2
    assert all(daughter == CellState(INITIAL_PHASE, PARAMETERS.fast_type) for daughter in fast.daughters)

    slow_first = exact_cell_tick(CellState(INITIAL_PHASE, PARAMETERS.slow_type))
    assert not slow_first.divided
    assert slow_first.predivision_phase == Fraction(13, 16)
    assert len(slow_first.daughters) == 1
    slow_second = exact_cell_tick(slow_first.daughters[0])
    assert slow_second.divided
    assert slow_second.predivision_phase == Fraction(9, 8)
    assert len(slow_second.daughters) == 2


def test_exact_generation_time_selection_counts() -> None:
    expected_fast = [1, 2, 4, 8, 16, 32, 64]
    expected_slow = [1, 1, 2, 2, 4, 4, 8]
    assert [
        exact_descendant_count(PARAMETERS.fast_type, ticks) for ticks in range(7)
    ] == expected_fast
    assert [
        exact_descendant_count(PARAMETERS.slow_type, ticks) for ticks in range(7)
    ] == expected_slow
    assert [
        Fraction(expected_fast[ticks], expected_slow[ticks]) for ticks in range(7)
    ] == [Fraction(2 ** ((ticks + 1) // 2)) for ticks in range(7)]

    with pytest.raises(ValueError):
        exact_descendant_count(Fraction(1, 2), 2)


def test_two_tick_forward_mutation_selection_closed_form() -> None:
    nu = Fraction(1, 16)
    assert expected_two_tick_transition(0, 1, nu) == (
        Fraction(1, 8),
        Fraction(15, 8),
    )
    assert expected_counts_after_blocks(0, 1, 2, nu) == (
        Fraction(47, 64),
        Fraction(225, 64),
    )

    recurrent = (Fraction(2), Fraction(3))
    for blocks in range(8):
        assert recurrent == expected_counts_after_blocks(2, 3, blocks, nu)
        recurrent = expected_two_tick_transition(*recurrent, nu)


def test_partition_threshold_and_conditional_survival_values_are_exact() -> None:
    modules = 7
    assert specified_daughter_complete_probability(modules, 4) == Fraction(
        170859375, 268435456
    )
    assert both_daughters_complete_probability(modules, 4) == Fraction(
        823543, 2097152
    )
    assert expected_complete_daughters(modules, 3) == Fraction(
        823543, 1048576
    )
    assert expected_complete_daughters(modules, 4) == Fraction(
        170859375, 134217728
    )
    assert minimum_supercritical_copy_number(modules) == 4

    certificate = build_coupled_certificate()
    partition = certificate["proof_obligations"]["stochastic_partition_threshold"]
    galton_watson = partition["conditional_iid_Galton_Watson_at_k_4"]
    assert galton_watson["extinction_probability"]["exact"] == (
        "2295015/7529536"
    )
    assert galton_watson["positive_survival_probability"]["exact"] == (
        "5234521/7529536"
    )
    assert not galton_watson["certain_survival"]


def test_certificate_has_non_knifedge_parameter_box_and_scope_guards() -> None:
    certificate = build_coupled_certificate()
    robustness = certificate["proof_obligations"]["open_parameter_plateau"]

    assert robustness["passed"]
    assert [value["exact"] for value in robustness["slow_increment_bounds"]] == [
        "37/128",
        "43/128",
    ]
    assert [value["exact"] for value in robustness["fast_increment_bounds"]] == [
        "67/128",
        "77/128",
    ]
    assert robustness["fast_one_tick_lower_margin"]["exact"] == "3/128"
    assert certificate["all_exact_model_obligations_passed"]
    assert certificate["claim_scope"][
        "model_relative_genotype_phenotype_coupling_proven"
    ]
    assert not certificate["claim_scope"]["molecular_copying_mechanism_proven"]
    assert not certificate["claim_scope"]["sample_path_fixation_proven"]
    assert validate_coupled_certificate(certificate)

    tampered = deepcopy(certificate)
    tampered["claim_scope"]["empirical_autonomous_protocell_proven"] = True
    assert not validate_coupled_certificate(tampered)
