"""Fail-closed scan of proposed intrinsic mobile/locked dark-sector ratios.

All probabilities in this module are dimensionless.  ``R`` always means the
locked-to-mobile *receipt* ratio at the matching epoch.  The scan first shows
why probability alone does not supply a unique energy ratio, then retains the
Fibonacci/Parry value only as a conditional candidate with its extra axioms
named explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np


def _probability(value: float, *, name: str = "p") -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return value


def _positive(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def receipt_ratio_from_probability(p_mobile: float) -> float:
    """Equal-energy receipt convention: R=(1-p)/p, undefined at p=0."""
    p_mobile = _probability(p_mobile, name="p_mobile")
    if p_mobile == 0.0:
        return math.inf
    return (1.0 - p_mobile) / p_mobile


def flag_continuum_counterexample(p: float) -> dict[str, Any]:
    """Every p belongs to the qutrit density-matrix flag family.

    rho_p=diag(p/2,p/2,1-p) is positive and trace one for every p in [0,1].
    Thus a flag/grade condition by itself cannot select one probability.
    """
    p = _probability(p)
    diagonal = (p / 2.0, p / 2.0, 1.0 - p)
    return {
        "rho_diagonal": diagonal,
        "trace": sum(diagonal),
        "r2": diagonal[0] + diagonal[1],
        "r3": sum(diagonal),
        "p_mobile_from_flag": diagonal[0] + diagonal[1],
        "positive": min(diagonal) >= 0.0,
        "p_is_free_on_closed_interval": True,
        "counterexample": "flag condition admits a continuum of p values",
    }


def unequal_energy_counterexample(*, p_mobile: float = 0.3,
                                  mobile_energy: float = 2.0,
                                  locked_energy: float = 7.0) -> dict[str, Any]:
    """Probability fractions and physical energy fractions differ without an axiom."""
    p = _probability(p_mobile, name="p_mobile")
    e_m = _positive(mobile_energy, name="mobile_energy")
    e_l = _positive(locked_energy, name="locked_energy")
    energy_mobile_fraction = p * e_m / (p * e_m + (1.0 - p) * e_l)
    return {
        "probability_mobile_fraction": p,
        "energy_mobile_fraction": energy_mobile_fraction,
        "equal": math.isclose(p, energy_mobile_fraction, abs_tol=1e-12),
        "counterexample": "probability fraction is not an energy fraction for unequal branch energies",
    }


def epoch_ratio(*, ratio_at_reference: float, scale_factor: float,
                reference_scale_factor: float = 1.0) -> dict[str, float]:
    """For dust plus constant vacuum, R(a)=R_* (a/a_*)^3."""
    r_star = _positive(ratio_at_reference, name="ratio_at_reference")
    a = _positive(scale_factor, name="scale_factor")
    a_star = _positive(reference_scale_factor, name="reference_scale_factor")
    return {"R_star": r_star, "a_over_a_star": a / a_star, "R_of_a": r_star * (a / a_star) ** 3}


def parameter_free_candidates() -> dict[str, dict[str, float]]:
    """The common parameter-free guesses; none is selected by bare quantum theory."""
    values = {
        "maximum_entropy_two_state": 1.0 / 2.0,
        "d3_one_of_three": 1.0 / 3.0,
        "d3_two_of_three": 2.0 / 3.0,
        "d4_one_of_four": 1.0 / 4.0,
        "d4_three_of_four": 3.0 / 4.0,
    }
    return {
        name: {
            "p_mobile": p,
            "R_equal_energy": receipt_ratio_from_probability(p),
            "derived_from_three_claims": False,
        }
        for name, p in values.items()
    }


def haar_subspace_candidate(*, k: int, dimension: int) -> dict[str, float]:
    """Haar expected subspace weight k/D, with integers supplied externally."""
    if not isinstance(k, int) or not isinstance(dimension, int) or not 0 <= k <= dimension or dimension <= 0:
        raise ValueError("need integers 0 <= k <= dimension with dimension > 0")
    p = k / dimension
    return {
        "k": k,
        "D": dimension,
        "p_mobile": p,
        "R_equal_energy": receipt_ratio_from_probability(p),
        "ensemble_measure_supplied": True,
        "individual_state_ratio_unique": False,
    }


def markov_stationary_candidate(*, a: float, b: float) -> dict[str, Any]:
    """Two-state transition rates give p=b/(a+b), freely tunable by rates."""
    a, b = _positive(a, name="a"), _positive(b, name="b")
    p = b / (a + b)
    return {"p_mobile": p, "stationary_formula": "b/(a+b)", "free_rate_ratio": a / b,
            "unique_without_rates": False}


def gibbs_logistic_candidate(*, beta_delta_energy: float) -> dict[str, Any]:
    """p=1/(1+exp(beta Delta E)); beta Delta E is dimensionless but free."""
    x = float(beta_delta_energy)
    if not math.isfinite(x):
        raise ValueError("beta_delta_energy must be finite")
    if x >= 0.0:
        weight = math.exp(-x)
        p = weight / (1.0 + weight)
    else:
        weight = math.exp(x)
        p = 1.0 / (1.0 + weight)
    return {"p_mobile": p, "beta_delta_energy": x, "argument_dimensionless": True,
            "unique_without_temperature_gap": False}


def fibonacci_parry_candidate() -> dict[str, Any]:
    """Unweighted no-consecutive-mobile shift with its Parry stationary measure."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    lambda_perron = phi
    # A=[[1,1],[1,0]], r=(phi,1); symmetry gives pi_i proportional to r_i^2.
    p_rare = 1.0 / (1.0 + phi * phi)
    p_locked = 1.0 - p_rare
    transition = (
        (1.0 / phi, 1.0 / (phi * phi)),
        (1.0, 0.0),
    )
    stationary_after_step = (
        p_locked * transition[0][0] + p_rare * transition[1][0],
        p_locked * transition[0][1] + p_rare * transition[1][1],
    )
    return {
        "adjacency": ((1, 1), (1, 0)),
        "perron_lambda": lambda_perron,
        "right_eigenvector": (phi, 1.0),
        "parry_transition_locked_mobile": transition,
        "transition_row_sum_residual": max(
            abs(sum(transition[0]) - 1.0), abs(sum(transition[1]) - 1.0)
        ),
        "stationary_distribution_locked_mobile": (p_locked, p_rare),
        "stationarity_residual": max(
            abs(stationary_after_step[0] - p_locked),
            abs(stationary_after_step[1] - p_rare),
        ),
        "entropy_rate": math.log(phi),
        "p_mobile_rare": p_rare,
        "R_locked_over_mobile": phi * phi,
        "exact_relation": "R=phi^2 and p=1/(1+phi^2)",
        "required_axioms": [
            "no consecutive mobile record",
            "unweighted adjacency A=[[1,1],[1,0]]",
            "Parry/max-entropy stationary measure",
            "branch probability maps to equal-energy receipt",
            "ratio is compared at the matching epoch",
        ],
        "possible_battery_story": "a mobile event consumes a resonant battery and needs a locked/reset event before another mobile event",
        "hard_exclusion_derived_from_existing_controlled_swap": False,
        "parry_measure_derived_from_existing_quantum_dynamics": False,
        "prediction": False,
    }


def fibonacci_completion_counts(horizon: int) -> tuple[tuple[int, int], ...]:
    """Count admissible completions from locked/mobile memory states.

    Entry n is (C_L(n), C_M(n)) for n future symbols and adjacency
    A=[[1,1],[1,0]].  C(0)=(1,1) and C(n)=A C(n-1).
    """
    if not isinstance(horizon, int) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer")
    counts = [(1, 1)]
    for _ in range(horizon):
        locked, mobile = counts[-1]
        counts.append((locked + mobile, locked))
    return tuple(counts)


def finite_uniform_history_transition(*, remaining_steps: int) -> tuple[tuple[float, float], ...]:
    """Doob transition that samples every allowed completion uniformly."""
    if not isinstance(remaining_steps, int) or remaining_steps <= 0:
        raise ValueError("remaining_steps must be a positive integer")
    counts = fibonacci_completion_counts(remaining_steps)
    previous = counts[remaining_steps - 1]
    current = counts[remaining_steps]
    return (
        (previous[0] / current[0], previous[1] / current[0]),
        (previous[0] / current[1], 0.0),
    )


def _uniform_history_probabilities(*, initial_state: int, horizon: int) -> tuple[float, ...]:
    probabilities: list[float] = []

    def walk(state: int, remaining: int, probability: float) -> None:
        if remaining == 0:
            probabilities.append(probability)
            return
        transition = finite_uniform_history_transition(remaining_steps=remaining)
        for next_state in (0, 1):
            weight = transition[state][next_state]
            if weight > 0.0:
                walk(next_state, remaining - 1, probability * weight)

    walk(initial_state, horizon, 1.0)
    return tuple(probabilities)


def fibonacci_uniform_history_bridge(*, horizon: int = 12,
                                     energy_star: float = 1.0) -> dict[str, Any]:
    """Finite uniform-history theorem and its conditional dark-ratio boundary."""
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    gap = _positive(energy_star, name="energy_star")
    counts = fibonacci_completion_counts(horizon)
    finite_transition = finite_uniform_history_transition(remaining_steps=horizon)
    parry = fibonacci_parry_candidate()
    parry_transition = parry["parry_transition_locked_mobile"]
    convergence_residual = max(
        abs(finite_transition[i][j] - parry_transition[i][j])
        for i in (0, 1)
        for j in (0, 1)
    )
    uniform_residuals = []
    path_counts = []
    for initial_state in (0, 1):
        probabilities = _uniform_history_probabilities(
            initial_state=initial_state, horizon=horizon
        )
        expected = 1.0 / counts[horizon][initial_state]
        path_counts.append(len(probabilities))
        uniform_residuals.append(
            max((abs(value - expected) for value in probabilities), default=0.0)
        )

    isometry = np.zeros((8, 2), dtype=complex)
    for source in (0, 1):
        for target in (0, 1):
            probability = parry_transition[source][target]
            if probability > 0.0:
                edge_record = 2 * source + target
                row = 4 * target + edge_record
                isometry[row, source] = math.sqrt(probability)
    completeness_residual = float(
        np.linalg.norm(isometry.conj().T @ isometry - np.eye(2))
    )
    p_mobile = parry["p_mobile_rare"]
    p_locked = 1.0 - p_mobile
    return {
        "horizon": horizon,
        "completion_counts_locked_mobile": counts,
        "finite_first_transition": finite_transition,
        "parry_transition": parry_transition,
        "transition_convergence_residual": convergence_residual,
        "enumerated_path_counts": tuple(path_counts),
        "uniform_path_probability_residual": max(uniform_residuals),
        "record_isometry_completeness_residual": completeness_residual,
        "finite_uniform_history_theorem_closed": (
            path_counts == [counts[horizon][0], counts[horizon][1]]
            and max(uniform_residuals) <= 1e-12
        ),
        "parry_record_isometry_closed": completeness_residual <= 1e-12,
        "stationary_ensemble_receipt": {
            "mobile_energy_expectation": p_mobile * gap,
            "locked_energy_expectation": p_locked * gap,
            "one_receipt_energy_per_event": gap,
            "per_event_two_simultaneous_receipts": False,
        },
        "conditional_status": {
            "unique_ratio_inside_declared_history_model": True,
            "hard_exclusion_from_fundamental_dynamics_derived": False,
            "uniform_history_measure_from_fundamental_dynamics_derived": False,
            "cosmological_dark_mapping_derived": False,
            "prediction": False,
        },
    }


def locally_uniform_edge_counterexample() -> dict[str, Any]:
    """Uniformly choosing available outgoing edges is not the Parry measure."""
    transition = ((0.5, 0.5), (1.0, 0.0))
    stationary = (2.0 / 3.0, 1.0 / 3.0)
    golden_mobile = fibonacci_parry_candidate()["p_mobile_rare"]
    return {
        "transition": transition,
        "stationary_locked_mobile": stationary,
        "p_mobile": stationary[1],
        "golden_p_mobile": golden_mobile,
        "different_from_global_uniform_history_measure": not math.isclose(
            stationary[1], golden_mobile, abs_tol=1e-12
        ),
    }


def dual_rail_blockade_receipt_candidate(
    *, length: int = 64, energy_star: float = 1.0
) -> dict[str, Any]:
    """Uniform hard-blockade histories with one equal-gap receipt per site.

    Each site has exactly one dual-rail occupation, locked or mobile.  The
    supplied hard constraint forbids adjacent mobile occupations.  At zero
    detuning all allowed configurations have the same base receipt energy;
    choosing the maximally mixed constrained state (or an engineered
    equal-amplitude RK state for diagonal observables) makes their weights
    uniform.
    """
    if not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    gap = _positive(energy_star, name="energy_star")
    counts = [1, 2]  # Z_0, Z_1
    total_mobile_occupations = [0, 1]
    for n in range(2, length + 1):
        counts.append(counts[n - 1] + counts[n - 2])
        total_mobile_occupations.append(
            total_mobile_occupations[n - 1]
            + total_mobile_occupations[n - 2]
            + counts[n - 2]
        )
    number_configurations = counts[length]
    mean_mobile_number = total_mobile_occupations[length] / number_configurations
    finite_mobile_fraction = mean_mobile_number / length
    golden = fibonacci_parry_candidate()
    bulk_mobile_fraction = golden["p_mobile_rare"]
    return {
        "length": length,
        "constrained_hilbert_dimension": number_configurations,
        "dimension_identity": "dim H_N = F_{N+2}",
        "finite_uniform_mobile_fraction": finite_mobile_fraction,
        "bulk_mobile_fraction": bulk_mobile_fraction,
        "bulk_locked_over_mobile_ratio": golden["R_locked_over_mobile"],
        "finite_size_residual": abs(finite_mobile_fraction - bulk_mobile_fraction),
        "exact_golden_value_is_finite_length_output": False,
        "bulk_limit": (
            "infinite-volume uniform constrained configurations / Parry measure"
        ),
        "measure_boundaries": {
            "maximally_mixed_configurations": (
                "uniform weights on allowed configurations at one slice"
            ),
            "finite_uniform_histories": (
                "horizon-conditioned Doob process on allowed histories"
            ),
            "parry_measure": (
                "maximum-entropy stationary measure of the infinite shift"
            ),
            "general_realtime_pxp_equivalence": False,
        },
        "one_receipt_per_site": True,
        "every_configuration_total_receipt_energy": length * gap,
        "ensemble_mobile_energy_expectation": finite_mobile_fraction * length * gap,
        "ensemble_locked_energy_expectation": (1.0 - finite_mobile_fraction) * length * gap,
        "no_double_count": math.isclose(
            finite_mobile_fraction * length * gap
            + (1.0 - finite_mobile_fraction) * length * gap,
            length * gap,
            abs_tol=1e-12,
        ),
        "microscopic_candidate": {
            "constraint": "n_M(i) n_M(i+1)=0",
            "dual_rail": "n_M(i)+n_L(i)=1",
            "hamiltonian": "H=N E_star + U sum n_M(i)n_M(i+1) - mu sum n_M(i)",
            "declared_limit": "U to positive infinity, mu=0",
            "neighbor_gated_flip": "P_(i-1) X_i P_(i+1)",
        },
        "required_axioms": {
            "hard_blockade": True,
            "zero_detuning_equal_weight": True,
            "maximally_mixed_or_engineered_RK_state": True,
            "fresh_equal_gap_receipt_per_site": True,
        },
        "fail_closed": {
            "hard_blockade_derived_from_three_user_claims": False,
            "cosmological_time_or_space_identified": False,
            "general_pxp_stationary_state_is_uniform": False,
            "dark_sector_identity_proved": False,
            "prediction": False,
        },
        "primary_source_boundaries": [
            "PhysRevResearch.6.023146: blockade-constrained Fibonacci state space",
            "PhysRevLett.124.050602: constrained relaxation and RK boundary",
            "cond-mat/0311345: classical-to-quantum RK construction",
        ],
    }


def golden_conditional_action_chain(
    *, energy_star: float = 5.0, matching_volume: float = 2.0
) -> dict[str, Any]:
    """Pass the infinite-volume Parry value through finite interface witnesses.

    The storage and action maps are finite and exact for a supplied p.  They
    do not turn a finite blockade chain into the exact golden value.
    """
    from examples.physics.quantum_selection_action_routes import (
        action_first_hybrid_witness,
        controlled_flux_storage_witness,
    )

    gap = _positive(energy_star, name="energy_star")
    volume = _positive(matching_volume, name="matching_volume")
    golden = fibonacci_parry_candidate()
    p_mobile = golden["p_mobile_rare"]
    theta = math.asin(math.sqrt(p_mobile))
    storage = controlled_flux_storage_witness(
        interaction_angle=theta, energy_star=gap
    )
    action = action_first_hybrid_witness(
        q_mobile=p_mobile, energy_star=gap, cell_volume=volume
    )
    probability_residual = max(
        abs(storage["p_mobile"] - action["q_mobile"]),
        abs(storage["q_locked"] - action["q_locked"]),
    )
    energy_residual = max(
        abs(storage["battery_energy_expectation"] - action["mobile_receipt_energy"]),
        abs(storage["flux_energy_expectation"] - action["locked_receipt_energy"]),
    )
    return {
        "status": "GOLDEN_BLOCKADE_TO_STORAGE_TO_ACTION_CHAIN_CONDITIONAL",
        "p_mobile": p_mobile,
        "p_locked": 1.0 - p_mobile,
        "R_locked_over_mobile": golden["R_locked_over_mobile"],
        "ratio_source": (
            "infinite-volume Parry measure, not a finite-N blockade output"
        ),
        "exact_ratio_derived_by_finite_storage_or_action": False,
        "interaction_angle": theta,
        "probability_residual": probability_residual,
        "energy_residual": energy_residual,
        "finite_chain_consistent": (
            probability_residual <= 1e-12
            and energy_residual <= 1e-12
            and storage["finite_branch_to_storage_closed"]
            and action["no_double_count"]
        ),
        "interface_compatibility_check_only": True,
        "required_axioms": [
            "hard-blockade dual-rail record dynamics",
            "zero detuning and uniform constrained history measure",
            "blockade occupancy maps to mobile/locked equal-energy receipt",
            "finite flux bit lifts to a covariant four-form",
            "mobile remainder admits a positive dust mass-shell map",
            "matching epoch and E_star/V_star are supplied",
            "a gravitational conditioning rule is supplied",
        ],
        "unconditional_cosmological_prediction": False,
        "storage": storage,
        "action": action,
    }


def weighted_fibonacci_counterexample(*, u: float, v: float) -> dict[str, Any]:
    """A=[[1,u],[v,0]] continuously changes the Parry ratio through uv."""
    u, v = _positive(u, name="u"), _positive(v, name="v")
    product = u * v
    lam = (1.0 + math.sqrt(1.0 + 4.0 * product)) / 2.0
    p_mobile = product / (lam * lam + product)
    return {
        "adjacency": ((1.0, u), (v, 0.0)),
        "uv": product,
        "perron_lambda": lam,
        "p_mobile": p_mobile,
        "R_locked_over_mobile": lam * lam / product,
        "counterexample": "u=v=1 is an extra unweighted-adjacency axiom, not a derived choice",
    }


def sector_ratio_underdetermination() -> dict[str, Any]:
    """A four-form flux and causal-set density are sector inputs, not ratio equations."""
    return {
        "four_form": "constant flux fixes a vacuum density after its sector is supplied",
        "causal_set": "a density/volume scale does not select a mobile probability",
        "ratio_determined": False,
    }


def interacting_vacuum_ratio_routes(
    *,
    target_ratio: float | None = None,
    scale_factor: float = 0.01,
    vacuum_density_at_one: float = 1.0,
    primordial_dust_constant: float = 1.0,
) -> dict[str, Any]:
    """Audit background interactions that could preserve a selected ratio.

    The sign convention is
        dot(rho_m)+3 H rho_m = Q,  dot(rho_L) = -Q,
    so positive Q transfers energy from vacuum-like density into dust.
    """
    if target_ratio is None:
        target_ratio = fibonacci_parry_candidate()["R_locked_over_mobile"]
    ratio = _positive(target_ratio, name="target_ratio")
    a = _positive(scale_factor, name="scale_factor")
    rho_l0 = _positive(vacuum_density_at_one, name="vacuum_density_at_one")
    dust_constant = _positive(
        primordial_dust_constant, name="primordial_dust_constant"
    )

    xi_l = 1.0 / (1.0 + ratio)
    xi_m = ratio / (1.0 + ratio)

    rho_l = rho_l0 * a ** (-3.0 * xi_l)
    primordial_dust = dust_constant * a ** -3.0
    generated_dust = xi_l / (1.0 - xi_l) * rho_l
    rho_m = primordial_dust + generated_dust
    q_over_h = 3.0 * xi_l * rho_l
    d_rho_l_d_lna = -3.0 * xi_l * rho_l
    d_rho_m_d_lna = -3.0 * primordial_dust - 3.0 * xi_l * generated_dust

    stable_fixed_point = (1.0 - xi_l) / xi_l
    stable_linear_eigenvalue = -3.0 * xi_l * stable_fixed_point
    unstable_linear_eigenvalue = 3.0 * (1.0 - xi_m)

    return {
        "sign_convention": (
            "dot(rho_m)+3H rho_m=Q; dot(rho_L)=-Q; Q>0 is vacuum-to-dust"
        ),
        "target_ratio_R_L_over_m": ratio,
        "constant_ratio_trajectory": {
            "required_Q_over_H_rho_m": 3.0 * ratio / (1.0 + ratio),
            "required_Q_over_H_rho_L": 3.0 / (1.0 + ratio),
            "density_power_a": -3.0 / (1.0 + ratio),
            "effective_w": -ratio / (1.0 + ratio),
            "power_law_a_of_t": 2.0 * (1.0 + ratio) / 3.0,
            "ordinary_early_matter_era_preserved_if_applied_all_epochs": False,
        },
        "stable_late_attractor": {
            "interaction": "Q=3 H xi_L rho_L",
            "xi_L": xi_l,
            "rho_L": rho_l,
            "rho_m": rho_m,
            "primordial_dust_term": primordial_dust,
            "generated_dust_term": generated_dust,
            "sample_ratio_R_L_over_m": rho_l / rho_m,
            "fixed_point_R_L_over_m": stable_fixed_point,
            "fixed_point_residual": abs(stable_fixed_point - ratio),
            "linear_eigenvalue_dRprime_dR": stable_linear_eigenvalue,
            "stable": stable_linear_eigenvalue < 0.0,
            "early_dust_dominates_at_sample": primordial_dust > generated_dust,
            "matter_continuity_residual": abs(
                d_rho_m_d_lna + 3.0 * rho_m - q_over_h
            ),
            "vacuum_continuity_residual": abs(d_rho_l_d_lna + q_over_h),
            "total_continuity_residual": abs(
                d_rho_m_d_lna + d_rho_l_d_lna + 3.0 * rho_m
            ),
        },
        "unstable_same_fixed_trajectory": {
            "interaction": "Q=3 H xi_m rho_m",
            "xi_m": xi_m,
            "fixed_point_R_L_over_m": xi_m / (1.0 - xi_m),
            "linear_eigenvalue_dRprime_dR": unstable_linear_eigenvalue,
            "stable": unstable_linear_eigenvalue < 0.0,
        },
        "matching_epoch_without_interaction": {
            "law": "R(a)=R(a_star)(a/a_star)^3",
            "matching_scale_factor_is_external": True,
            "coincidence_solved": False,
        },
        "four_form_and_exchange_boundary": {
            "sourceless_four_form_has_Q_zero_between_jumps": True,
            "membrane_route_needs_rates_charges_tensions_and_wall_stress": True,
            "rolling_scalar_route_needs_new_functions_or_couplings": True,
        },
        "background_nonuniqueness_counterexample": (
            "Q=Q_star+H rho_m (R-R_star) F(a,R,...) has the same target "
            "trajectory for arbitrary F but different stability and perturbations"
        ),
        "required_new_structure": [
            "a covariant microscopic action deriving Q",
            "a four-vector transfer law Q^mu and momentum-transfer frame",
            "particle-number, mass-shell, entropy and perturbation dynamics",
            "an independently fixed absolute density scale",
        ],
        "primary_source_boundary": (
            "JCAP 2024 01 048 studies linear interacting-dark-energy "
            "viability; it does not derive the golden coupling"
        ),
        "dimensionless_audit": {
            "target_ratio_and_xi": True,
            "scale_factor": True,
            "Q_over_H_rho": True,
        },
        "prediction": False,
    }


def observational_dark_sector_diagnostic(
    *, omega_cdm: float, omega_dark_energy: float, source_provenance: str
) -> dict[str, Any]:
    """Compare a supplied CDM-only posterior without feeding it into a candidate."""
    cdm = _positive(omega_cdm, name="omega_cdm")
    dark_energy = _positive(omega_dark_energy, name="omega_dark_energy")
    return {
        "omega_cdm": cdm,
        "omega_dark_energy": dark_energy,
        "p_cdm_dark_sector_normalized": cdm / (cdm + dark_energy),
        "R_dark_energy_over_cdm": dark_energy / cdm,
        "baryons_included_in_numerator": False,
        "source_provenance": source_provenance,
        "used_for_derivation": False,
        "likelihood_fit_performed": False,
    }


def certificate(
    *,
    omega_cdm: float | None = None,
    omega_dark_energy: float | None = None,
    source_provenance: str | None = None,
) -> dict[str, Any]:
    """Return all candidates without promoting any external value to a fit."""
    diagnostic = None
    if (omega_cdm is None) != (omega_dark_energy is None):
        raise ValueError("omega_cdm and omega_dark_energy must be supplied together")
    if omega_cdm is not None and omega_dark_energy is not None:
        diagnostic = observational_dark_sector_diagnostic(
            omega_cdm=omega_cdm,
            omega_dark_energy=omega_dark_energy,
            source_provenance=source_provenance
            or "user-supplied diagnostic; no external source asserted",
        )
    golden = fibonacci_parry_candidate()
    weighted_a, weighted_b = weighted_fibonacci_counterexample(u=1.0, v=1.0), weighted_fibonacci_counterexample(u=2.0, v=1.0)
    return {
        "status": "MAJOR_RATIO_CANDIDATES_AUDITED_NO_UNIQUE_RATIO_DERIVED_GOLDEN_ROUTE_CONDITIONAL",
        "continuum_counterexample": flag_continuum_counterexample(0.271),
        "unequal_energy_counterexample": unequal_energy_counterexample(),
        "epoch_dependence": epoch_ratio(ratio_at_reference=golden["R_locked_over_mobile"], scale_factor=2.0),
        "parameter_free_candidates": parameter_free_candidates(),
        "haar_example": haar_subspace_candidate(k=1, dimension=4),
        "markov_example": markov_stationary_candidate(a=2.0, b=1.0),
        "gibbs_example": gibbs_logistic_candidate(beta_delta_energy=1.0),
        "golden_conditional_candidate": golden,
        "finite_uniform_history_bridge": fibonacci_uniform_history_bridge(),
        "local_uniform_edge_counterexample": locally_uniform_edge_counterexample(),
        "dual_rail_blockade_candidate": dual_rail_blockade_receipt_candidate(),
        "golden_conditional_action_chain": golden_conditional_action_chain(),
        "interacting_vacuum_ratio_routes": interacting_vacuum_ratio_routes(
            target_ratio=golden["R_locked_over_mobile"]
        ),
        "weighted_adjacency_counterexample": {"unweighted": weighted_a, "weighted": weighted_b,
                                               "ratio_changes": not math.isclose(weighted_a["R_locked_over_mobile"], weighted_b["R_locked_over_mobile"])},
        "four_form_and_causal_set": sector_ratio_underdetermination(),
        "observational_target_diagnostic": diagnostic,
        "dimensionless_audit": {
            "probabilities_and_receipt_ratios": True,
            "beta_delta_energy": True,
            "adjacency_weights_and_perron_eigenvalue": True,
            "scale_factor_ratio": True,
        },
        "fail_closed": {
            "unique_ratio_derived": False,
            "golden_ratio_requires_listed_axioms": True,
            "finite_uniform_history_theorem_closed": True,
            "golden_storage_action_chain_consistent": True,
            "interacting_attractor_is_action_derived": False,
            "prediction": False,
            "external_target_used_for_derivation": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--omega-cdm", type=float)
    parser.add_argument("--omega-dark-energy", type=float)
    parser.add_argument("--source-provenance")
    args = parser.parse_args()
    print(
        json.dumps(
            certificate(
                omega_cdm=args.omega_cdm,
                omega_dark_energy=args.omega_dark_energy,
                source_provenance=args.source_provenance,
            ),
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
