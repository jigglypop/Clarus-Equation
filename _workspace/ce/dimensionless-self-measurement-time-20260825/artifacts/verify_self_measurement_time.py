"""Focused checks for the dimensionless self-measurement-depth model."""
from __future__ import annotations

import math
import numpy as np

TOL = 1e-12


def close(actual, expected, label, tol=TOL):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    assert error <= tol, f"{label}: {error}"
    print(f"PASS {label}: max error={error:.3e}")


I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
M = (X + Z) / math.sqrt(2.0)


def dephase(axis, rho):
    return 0.5 * (rho + axis @ rho @ axis)


def phi(axis, rho, eta):
    return (1.0 - eta) * rho + eta * dephase(axis, rho)


def choi(axis, eta):
    units = []
    for i in range(2):
        row = []
        for j in range(2):
            eij = np.zeros((2, 2), dtype=complex)
            eij[i, j] = 1.0
            row.append(phi(axis, eij, eta))
        units.append(row)
    return np.block(units)


def trace_norm(matrix):
    return float(np.sum(np.linalg.svd(matrix, compute_uv=False)))


def trace_distance(rho, sigma):
    return 0.5 * trace_norm(rho - sigma)


# CPTP spot certificate for every channel used below.
for axis_name, axis in [("Z", Z), ("M", M)]:
    close(axis.conj().T @ axis, I2, f"{axis_name} axis involution")
    for eta in (0.0, 0.2, 0.7, 1.0 - math.exp(-1.5)):
        J = choi(axis, eta)
        eig_min = float(np.min(np.linalg.eigvalsh(J)))
        assert eig_min >= -TOL, (axis_name, eta, eig_min)
        trace_blocks = np.array(
            [[np.trace(J[2*i:2*i+2, 2*j:2*j+2]) for j in range(2)]
             for i in range(2)]
        )
        close(trace_blocks, I2, f"{axis_name} eta={eta:.6f} trace preserving")
        print(f"PASS {axis_name} eta={eta:.6f} Choi min={eig_min:.3e}")


# Exact composition law for a fixed dephasing projector.
rho_plus = 0.5 * (I2 + X)
for eta1, eta2 in ((0.1, 0.2), (0.7, 0.2), (0.33, 0.61)):
    eta12 = eta1 + eta2 - eta1 * eta2
    sequential = phi(Z, phi(Z, rho_plus, eta1), eta2)
    direct = phi(Z, rho_plus, eta12)
    close(sequential, direct, f"fixed-partition composition {eta1},{eta2}")
    theta12 = -math.log1p(-eta12)
    theta_sum = -math.log1p(-eta1) - math.log1p(-eta2)
    close(theta12, theta_sum, f"theta additivity {eta1},{eta2}")


# One target depth equals every finite equal weak partition tested.
theta_star = 1.5
target_eta = 1.0 - math.exp(-theta_star)
target = phi(Z, rho_plus, target_eta)
partition_residuals = {}
for count in (1, 2, 5, 100):
    step_eta = 1.0 - math.exp(-theta_star / count)
    state = rho_plus.copy()
    for _ in range(count):
        state = phi(Z, state, step_eta)
    residual = float(np.linalg.norm(state - target, ord="fro"))
    assert residual <= TOL, (count, residual)
    partition_residuals[count] = residual
    print(f"PASS theta partition N={count}: residual={residual:.3e}")

# Analytic semigroup form D + exp(-theta)(I-D).
semigroup_target = dephase(Z, rho_plus) + math.exp(-theta_star) * (
    rho_plus - dephase(Z, rho_plus)
)
close(target, semigroup_target, "analytic dephasing semigroup")


# Self-nonidentity flow on the same fixed semigroup.
A = rho_plus - dephase(Z, rho_plus)
A_norm = trace_norm(A)
close(A_norm, 1.0, "initial off-diagonal trace norm")
theta0 = 0.7
h = 0.2
rho_theta0 = phi(Z, rho_plus, 1.0 - math.exp(-theta0))
rho_theta_h = phi(Z, rho_plus, 1.0 - math.exp(-(theta0 + h)))
increment = trace_distance(rho_theta_h, rho_theta0)
increment_exact = 0.5 * math.exp(-theta0) * (1.0 - math.exp(-h)) * A_norm
close(increment, increment_exact, "every-step self-nonidentity distance")
assert increment > 0.0

speed = 0.5 * math.exp(-theta_star) * A_norm
length = 0.5 * (1.0 - math.exp(-theta_star)) * A_norm
residual0 = trace_distance(rho_plus, dephase(Z, rho_plus))
residual_star = trace_distance(target, dephase(Z, rho_plus))
close(speed, 0.11156508007421491, "metric speed at theta=1.5")
close(length, 0.3884349199257851, "finite accumulated path length")
close(trace_distance(rho_plus, target), length,
      "straight-ray length equals endpoint distance")
close(-math.log(residual_star / residual0), theta_star,
      "logarithmic residual clock recovery")

# Infinite refinement does not change the finite path length.
for count in (1, 2, 5, 100, 1000):
    points = [
        phi(Z, rho_plus, 1.0 - math.exp(-theta_star * k / count))
        for k in range(count + 1)
    ]
    discrete_length = sum(
        trace_distance(points[k + 1], points[k]) for k in range(count)
    )
    close(discrete_length, length, f"finite length under refinement N={count}")


# Opportunity cost and its finite-alphabet bound.
p = np.array([0.8, 0.2], dtype=float)
cbar = float(np.sum(p * (1.0 - p) * (-np.log(p))))
cself = (1.0 - math.exp(-theta_star)) * cbar
bound = (1.0 - math.exp(-theta_star)) * math.log(len(p))
close(cbar, 0.29321303419972966, "constant predictive opportunity cost")
close(cself, 0.22778836292113697, "integrated self-measurement cost")
assert 0.0 <= cself <= bound <= math.log(len(p))
print(f"PASS finite-alphabet bound: C={cself:.15f}, bound={bound:.15f}")
close(cself, (2.0 * cbar / A_norm) * length,
      "opportunity cost per self-difference length")

# Cost and motion are not universally identical: a diagonal state is stationary.
rho_diag = np.diag(p).astype(complex)
stationary = phi(Z, rho_diag, target_eta)
close(stationary, rho_diag, "stationary dephased-state counterexample")
close(trace_distance(stationary, rho_diag), 0.0,
      "stationary path has zero length")
assert cbar > 0.0
print("PASS positive opportunity accounting can coexist with zero state motion")


# Noncommuting axes: equal listed strengths do not define an order-free scalar.
rho_m = 0.5 * (I2 + M)
z_then_m = phi(M, phi(Z, rho_m, 0.7), 0.2)
m_then_z = phi(Z, phi(M, rho_m, 0.2), 0.7)
order_difference = float(np.linalg.norm(z_then_m - m_then_z, ord="fro"))
close(order_difference, 0.04949747468305832,
      "noncommuting order counterexample")
assert order_difference > 1e-3


# Non-Markovian recoherence: a global monotone theta cannot survive revival.
lambda_values = [math.cos(x / 2.0) ** 2 for x in (0.0, math.pi, 2.0 * math.pi)]
close(lambda_values, [1.0, 0.0, 1.0], "recoherence endpoints")
theta_values = [(-math.log(value) if value > TOL else math.inf)
                for value in lambda_values]
assert theta_values[0] == 0.0
assert math.isinf(theta_values[1])
close(theta_values[2], 0.0, "recoherence returns theta to zero")
print("PASS non-Markovian monotonicity counterexample")


# Local self-difference alone has no arrow: a unitary orbit returns exactly.
ket_plus = np.array([[1.0], [1.0]], dtype=complex) / math.sqrt(2.0)
rho_unitary0 = ket_plus @ ket_plus.conj().T


def unitary_orbit(time):
    U = math.cos(time / 2.0) * I2 - 1j * math.sin(time / 2.0) * Z
    return U @ rho_unitary0 @ U.conj().T


small_unitary_step = trace_distance(unitary_orbit(0.1), rho_unitary0)
assert small_unitary_step > 0.0
close(unitary_orbit(2.0 * math.pi), rho_unitary0,
      "periodic unitary return counterexample")
unitary_count = 10000
unitary_points = [unitary_orbit(2.0 * math.pi * k / unitary_count)
                  for k in range(unitary_count + 1)]
unitary_length = sum(
    trace_distance(unitary_points[k + 1], unitary_points[k])
    for k in range(unitary_count)
)
close(unitary_length, math.pi, "periodic orbit path length", tol=1e-7)
print("PASS positive local motion does not imply an irreversible arrow")


print("partition_residuals=" + repr(partition_residuals))
print(f"noncommuting_order_difference={order_difference:.15f}")
print(f"cbar={cbar:.15f}")
print(f"cself={cself:.15f}")
print(f"self_difference_increment={increment:.15f}")
print(f"self_path_length={length:.15f}")
print(f"unitary_cycle_length={unitary_length:.15f}")
print("ALL CHECKS PASSED")
