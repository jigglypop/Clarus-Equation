import numpy as np
import pytest

from reality_stone.clarus.indra_causal_quotient import (
    equitable_orbit_quotient,
    evaluate_orbit_scaling,
    finite_causal_cone,
    normalized_orbit_expansion,
    quotient_closure_error,
)


def test_equitable_partition_closes_nonlinear_bootstrap_exactly() -> None:
    reduced = np.asarray(((1.2, 0.4), (0.3, 1.1)), dtype=np.float64)
    full, labels = normalized_orbit_expansion(reduced, (3, 5))
    quotient = equitable_orbit_quotient(full, labels)
    assert np.allclose(quotient.as_array(), reduced, atol=1e-14)
    assert quotient_closure_error(full, quotient, (0.25, 0.75)) <= 1e-14


def test_non_equitable_symmetry_break_is_rejected() -> None:
    reduced = np.asarray(((1.2, 0.4), (0.3, 1.1)), dtype=np.float64)
    full, labels = normalized_orbit_expansion(reduced, (3, 5))
    full[0, np.asarray(labels) == 1] *= 1.2
    with pytest.raises(ValueError, match="not equitable"):
        equitable_orbit_quotient(full, labels)


def test_finite_causal_cone_and_budget_bound_infinite_chain_approximation() -> None:
    adjacency = {node: (node - 1, node + 1) for node in range(-100, 101)}
    cone = finite_causal_cone(adjacency, (0,), generations=7)
    assert len(cone.active_nodes) == 15
    bounded = finite_causal_cone(adjacency, (0,), generations=7, active_budget=8)
    assert len(bounded.active_nodes) == 8
    assert bounded.budget_exhausted


def test_expanding_orbit_network_has_fixed_exact_quotient() -> None:
    result = evaluate_orbit_scaling()
    assert result["schema"] == "clarus.indra-orbit-causal-quotient.validation.v1"
    assert result["verdict"] == "GO"
    assert all(result["gates"].values())
    assert result["finite_open_chain_min_extinction"] == [1.0, 1.0, 1.0, 1.0]
    assert result["translation_quotient_extinction"] < 1.0
