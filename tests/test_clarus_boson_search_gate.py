from examples.physics.clarus_boson_search_gate import (
    COMPTON_FM,
    MASS_WINDOW,
    M_PHI_MEV,
    ExperimentalResult,
    classify_result,
    in_mass_window,
)


def test_clarus_mass_and_range_are_registered():
    assert 29.5 < M_PHI_MEV < 29.8
    assert 6.5 < COMPTON_FM < 6.8
    lo, hi = MASS_WINDOW
    assert lo < M_PHI_MEV < hi
    assert abs((hi - lo) - 2.52) < 1e-12


def test_x17_is_not_a_clarus_mass_hit():
    assert not in_mass_window(17.0)
    decision = classify_result(
        ExperimentalResult("X17-like", 17.0, 6.0, pole_compatible=True)
    )
    assert decision.status == "open_test"


def test_sub_discovery_excess_in_window_is_candidate_only():
    decision = classify_result(
        ExperimentalResult("window excess", 29.7, 3.0, pole_compatible=True)
    )
    assert decision.status == "pole_candidate"


def test_five_sigma_pole_hit_in_window_promotes_bridge():
    decision = classify_result(
        ExperimentalResult("window discovery", 29.7, 5.1, pole_compatible=True)
    )
    assert decision.status == "pole_confirmed"


def test_full_mass_and_coupling_exclusion_rejects_bridge_not_core_field():
    decision = classify_result(
        ExperimentalResult(
            "full exclusion",
            None,
            None,
            pole_compatible=False,
            excludes_mass_window=True,
            excludes_bridge_coupling=True,
        )
    )
    assert decision.status == "bridge_rejected"
    assert "core Clarus field is not falsified" in decision.reason


def test_mass_only_exclusion_is_constraint_not_falsification():
    decision = classify_result(
        ExperimentalResult(
            "partial exclusion",
            None,
            None,
            pole_compatible=False,
            excludes_mass_window=True,
            excludes_bridge_coupling=False,
        )
    )
    assert decision.status == "bridge_constrained"
