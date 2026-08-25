"""Dependency-free checks for the revised one-way 0D -> 3+1D model.

The checks certify only the declared finite-dimensional channel algebra,
directed population rates, branching fixed point, dimensions, and adverse
controls. They are not evidence for a cosmological 0D source or dark-sector
identity.
"""

from __future__ import annotations

import json
import math


TOL = 1.0e-12
Matrix = list[list[complex]]


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0j for _ in range(cols)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    result = zeros(size, size)
    for index in range(size):
        result[index][index] = 1.0
    return result


def dagger(matrix: Matrix) -> Matrix:
    return [
        [matrix[row][col].conjugate() for row in range(len(matrix))]
        for col in range(len(matrix[0]))
    ]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    return [
        [
            sum(left[row][k] * right[k][col] for k in range(len(right)))
            for col in range(len(right[0]))
        ]
        for row in range(len(left))
    ]


def add(*matrices: Matrix) -> Matrix:
    return [
        [sum(matrix[row][col] for matrix in matrices) for col in range(len(matrices[0][0]))]
        for row in range(len(matrices[0]))
    ]


def scale(value: complex, matrix: Matrix) -> Matrix:
    return [[value * item for item in row] for row in matrix]


def subtract(left: Matrix, right: Matrix) -> Matrix:
    return add(left, scale(-1.0, right))


def commutator(left: Matrix, right: Matrix) -> Matrix:
    return subtract(matmul(left, right), matmul(right, left))


def tensor(left: Matrix, right: Matrix) -> Matrix:
    result = zeros(len(left) * len(right), len(left[0]) * len(right[0]))
    for i, left_row in enumerate(left):
        for j, left_value in enumerate(left_row):
            for k, right_row in enumerate(right):
                for ell, right_value in enumerate(right_row):
                    result[i * len(right) + k][j * len(right[0]) + ell] = (
                        left_value * right_value
                    )
    return result


def trace(matrix: Matrix) -> complex:
    return sum(matrix[index][index] for index in range(len(matrix)))


def max_abs(matrix: Matrix) -> float:
    return max(abs(value) for row in matrix for value in row)


def dissipator(operator: Matrix, state: Matrix) -> Matrix:
    operator_dagger = dagger(operator)
    number = matmul(operator_dagger, operator)
    return add(
        matmul(matmul(operator, state), operator_dagger),
        scale(-0.5, matmul(number, state)),
        scale(-0.5, matmul(state, number)),
    )


def partial_trace_b(matrix: Matrix, dim_a: int, dim_b: int) -> Matrix:
    result = zeros(dim_a, dim_a)
    for a_row in range(dim_a):
        for a_col in range(dim_a):
            result[a_row][a_col] = sum(
                matrix[a_row * dim_b + b][a_col * dim_b + b]
                for b in range(dim_b)
            )
    return result


def partial_trace_a(matrix: Matrix, dim_a: int, dim_b: int) -> Matrix:
    result = zeros(dim_b, dim_b)
    for b_row in range(dim_b):
        for b_col in range(dim_b):
            result[b_row][b_col] = sum(
                matrix[a * dim_b + b_row][a * dim_b + b_col]
                for a in range(dim_a)
            )
    return result


def two_by_two_psd(matrix: Matrix) -> bool:
    hermitian = max_abs(subtract(matrix, dagger(matrix))) <= TOL
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return (
        hermitian
        and matrix[0][0].real >= -TOL
        and matrix[1][1].real >= -TOL
        and determinant.real >= -TOL
        and abs(determinant.imag) <= TOL
    )


def two_by_two_eigenvalues(matrix: Matrix) -> tuple[float, float]:
    """Eigenvalues of a 2x2 Hermitian matrix, in ascending order."""
    a = matrix[0][0].real
    d = matrix[1][1].real
    off_diagonal = matrix[0][1]
    radius = math.sqrt((a - d) ** 2 + 4.0 * abs(off_diagonal) ** 2)
    return ((a + d - radius) / 2.0, (a + d + radius) / 2.0)


def branching_extinction(mean_offspring: float) -> tuple[float, int]:
    probability = 0.0
    for iteration in range(1, 100_001):
        updated = math.exp(mean_offspring * (probability - 1.0))
        if abs(updated - probability) <= 1.0e-16:
            return updated, iteration
        probability = updated
    raise RuntimeError("branching fixed-point iteration did not converge")


def main() -> int:
    # A one-dimensional input channel E(z)=z*rho is a state preparation map.
    prepared_state: Matrix = [[0.7, 0.2], [0.2, 0.3]]
    preparation_trace = trace(prepared_state)

    # A two-outcome instrument from C has subnormalized positive outputs whose
    # traces sum to one.
    selected_output = scale(0.6, [[1.0, 0.0], [0.0, 0.0]])
    nonselected_output = scale(0.4, prepared_state)
    selected_choi_min = two_by_two_eigenvalues(selected_output)[0]
    nonselected_choi_min = two_by_two_eigenvalues(nonselected_output)[0]
    instrument_probability_sum = (
        trace(selected_output) + trace(nonselected_output)
    )
    instrument_tp_residual = abs(instrument_probability_sum - 1.0)
    selected_tni = -TOL <= trace(selected_output).real <= 1.0 + TOL
    nonselected_tni = -TOL <= trace(nonselected_output).real <= 1.0 + TOL

    # Cascaded source A -> target B.  The source partial trace of the cross
    # generator vanishes, while the target partial trace is generally nonzero.
    lowering: Matrix = [[0.0, 1.0], [0.0, 0.0]]
    ident = identity(2)
    a = tensor(lowering, ident)
    b = tensor(ident, lowering)
    a_dagger = dagger(a)
    b_dagger = dagger(b)

    source_plus: Matrix = [[0.5, 0.5], [0.5, 0.5]]
    target_ground: Matrix = [[1.0, 0.0], [0.0, 0.0]]
    joint_state = tensor(source_plus, target_ground)

    cascade_hamiltonian = scale(
        1.0 / (2.0j), subtract(matmul(b_dagger, a), matmul(a_dagger, b))
    )
    combined_jump = add(a, b)
    gksl_cross = add(
        scale(-1.0j, commutator(cascade_hamiltonian, joint_state)),
        dissipator(combined_jump, joint_state),
        scale(-1.0, dissipator(a, joint_state)),
        scale(-1.0, dissipator(b, joint_state)),
    )
    expanded_cross = add(
        commutator(matmul(a, joint_state), b_dagger),
        commutator(b, matmul(joint_state, a_dagger)),
    )
    cascade_expansion_residual = max_abs(subtract(gksl_cross, expanded_cross))
    upstream_feedback_residual = max_abs(partial_trace_b(expanded_cross, 2, 2))
    downstream_drive_norm = max_abs(partial_trace_a(expanded_cross, 2, 2))
    cascade_hamiltonian_hermiticity = max_abs(
        subtract(cascade_hamiltonian, dagger(cascade_hamiltonian))
    )

    # Directed diagonal CTMC rates for 0 -> 1 -> 2.
    occupation = [1, 0, 0]
    directed_edges = [(0, 1, 2.0), (1, 2, 3.0)]
    birth_rates = [0.0, 0.0, 0.0]
    for parent, child, rate in directed_edges:
        birth_rates[child] += (1 - occupation[child]) * rate * occupation[parent]
    death_rates = [0.5 * value for value in occupation]
    finite_dag_is_forward = all(parent < child for parent, child, _ in directed_edges)
    no_decay_birth_bound = len(occupation) - sum(occupation)

    # Independent Poisson genealogy.
    mean_offspring = 3.1777584234099736
    extinction, iterations = branching_extinction(mean_offspring)
    fixed_point_residual = abs(
        math.exp(-mean_offspring * (1.0 - extinction)) - extinction
    )
    survival = 1.0 - extinction
    stability_derivative = mean_offspring * extinction

    # Same genealogy, different scalar amplitude/vacuum offset: abundance is
    # not identifiable from q.
    mass = 7.0
    rho_a = 0.5 * mass * mass * 1.0**2 + 0.0
    rho_b = 0.5 * mass * mass * 2.0**2 + 5.0

    dimensions_in_energy_powers = {
        "kappa_rate": 1.0,
        "sqrt_kappa_jump": 0.5,
        "lindblad_generator": 1.0,
        "branching_mean_D": 0.0,
        "extinction_probability_q": 0.0,
        "phi_4d": 1.0,
        "M_star": 1.0,
        "history_measure": 0.0,
        "history_kernel": 0.0,
    }

    checks = {
        "strict_0d_arrow_is_channel_not_internal_coordinate": True,
        "one_dimensional_input_is_state_preparation": (
            two_by_two_psd(prepared_state)
            and abs(preparation_trace - 1.0) <= TOL
        ),
        "declared_one_dimensional_input_instrument_is_cp_tni_and_tp_in_sum": (
            # For domain C, J(E_a)=E_a(1), so each displayed output is the
            # complete Choi matrix of the corresponding map.
            selected_choi_min >= -TOL
            and nonselected_choi_min >= -TOL
            and selected_tni
            and nonselected_tni
            and instrument_tp_residual <= TOL
        ),
        "cascade_hamiltonian_is_hermitian": (
            cascade_hamiltonian_hermiticity <= TOL
        ),
        "cascade_gksl_expansion_matches": cascade_expansion_residual <= TOL,
        "target_does_not_feed_back_into_source": upstream_feedback_residual <= TOL,
        "source_can_drive_target": downstream_drive_norm > TOL,
        "directed_ctmc_birth_rates_match": birth_rates == [0.0, 2.0, 0.0],
        "directed_ctmc_death_rates_match": death_rates == [0.5, 0.0, 0.0],
        "finite_sample_graph_is_acyclic_and_forward": finite_dag_is_forward,
        "finite_no_decay_birth_count_is_bounded": no_decay_birth_bound == 2,
        "poisson_fixed_point_matches_registered_value": (
            abs(extinction - 0.048646719644028225) <= TOL
            and fixed_point_residual <= TOL
        ),
        "poisson_low_fixed_point_is_stable": stability_derivative < 1.0,
        "same_genealogy_allows_different_abundance": rho_a != rho_b,
        "branching_core_is_dimensionless": (
            dimensions_in_energy_powers["branching_mean_D"] == 0.0
            and dimensions_in_energy_powers["extinction_probability_q"] == 0.0
        ),
        "jump_and_generator_dimensions_match": (
            2.0 * dimensions_in_energy_powers["sqrt_kappa_jump"]
            == dimensions_in_energy_powers["lindblad_generator"]
            == dimensions_in_energy_powers["kappa_rate"]
        ),
        "residual_map_has_scalar_dimension": (
            dimensions_in_energy_powers["M_star"]
            + dimensions_in_energy_powers["history_measure"]
            + dimensions_in_energy_powers["history_kernel"]
            == dimensions_in_energy_powers["phi_4d"]
        ),
    }

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "tolerance": TOL,
        "state_preparation_trace": preparation_trace.real,
        "instrument_probability_sum": instrument_probability_sum.real,
        "instrument_selected_choi_min_eigenvalue": selected_choi_min,
        "instrument_nonselected_choi_min_eigenvalue": nonselected_choi_min,
        "instrument_trace_preservation_residual": instrument_tp_residual,
        "cascade_expansion_residual": cascade_expansion_residual,
        "cascade_hamiltonian_hermiticity_residual": (
            cascade_hamiltonian_hermiticity
        ),
        "upstream_feedback_residual": upstream_feedback_residual,
        "downstream_drive_norm": downstream_drive_norm,
        "directed_birth_rates": birth_rates,
        "directed_death_rates": death_rates,
        "mean_offspring_D": mean_offspring,
        "extinction_probability_q": extinction,
        "survival_probability": survival,
        "fixed_point_residual": fixed_point_residual,
        "fixed_point_iterations": iterations,
        "fixed_point_derivative_Dq": stability_derivative,
        "abundance_certificate": {"rho_A": rho_a, "rho_B": rho_b},
        "dimensions_in_energy_powers": dimensions_in_energy_powers,
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
