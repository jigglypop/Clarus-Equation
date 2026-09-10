"""CE-AM1: exact finite event compression and retained hidden memory.

Run with Python and NumPy only. Fraction arithmetic and independent full-matrix
NumPy propagation check a supplied finite model, not a physical unification.
"""

from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


TOLERANCE = 1e-12
SEED = 7301


def matmul(a, b):
    return [[sum((x * y for x, y in zip(row, col)), F(0))
             for col in zip(*b)] for row in a]


def add(a, b, sign=1):
    return [[x + sign * y for x, y in zip(r, s)] for r, s in zip(a, b)]


def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def transpose(a):
    return [list(row) for row in zip(*a)]


def realified_pulse(h):
    """Real representation of I - i h, without its 1/sqrt(2) scale."""
    n = len(h)
    ident = eye(n)
    return [ident[i] + h[i] for i in range(n)] + [
        [-v for v in h[i]] + ident[i] for i in range(n)]


def exact_audit():
    h1 = [[F(int(i ^ 2 == j)) for j in range(4)] for i in range(4)]
    h2 = [[F(int(i ^ 1 == j)) for j in range(4)] for i in range(4)]
    zero4 = [[F(0)] * 4 for _ in range(4)]
    assert add(matmul(h1, h2), matmul(h2, h1), -1) == zero4
    a1, a2 = [[row[:3] for row in h[:3]] for h in (h1, h2)]
    b1, b2 = [[[row[3]] for row in h[:3]] for h in (h1, h2)]
    curvature = add(matmul(a1, a2), matmul(a2, a1), -1)
    hidden = add(matmul(b1, transpose(b2)), matmul(b2, transpose(b1)), -1)
    assert add(curvature, hidden) == [[F(0)] * 3 for _ in range(3)]
    assert sum(x*x for row in curvature for x in row) == 2

    # A3 = -i [A1,A2]. Check the spin-one su(2) closure in real arithmetic.
    assert add(matmul(a2, curvature), matmul(curvature, a2), -1) == [
        [-x for x in row] for row in a1]
    assert add(matmul(curvature, a1), matmul(a1, curvature), -1) == [
        [-x for x in row] for row in a2]
    casimir = add(add(matmul(a1, a1), matmul(a2, a2)), matmul(curvature, curvature), -1)
    assert casimir == [[2*x for x in row] for row in eye(3)]

    w1, w2 = [realified_pulse(h) for h in (h1, h2)]
    p = [[F(i == j and i not in (3, 7)) for j in range(8)] for i in range(8)]
    full = matmul(w2, w1)
    assert full == matmul(w1, w2)
    dropped = [matmul(p, matmul(b, matmul(p, matmul(a, p))))
               for a, b in ((w1, w2), (w2, w1))]

    def probabilities(matrix, vector, squared_scale):
        v = matmul(matrix, [[F(x)] for x in vector])
        return [(v[i][0]**2 + v[i+4][0]**2) * squared_scale for i in range(4)]

    initial = [0, 1, 0, 0, 0, 0, 0, 0]
    full_p = probabilities(full, initial, F(1, 4))
    drop_p = [probabilities(m, initial, F(1, 4)) for m in dropped]
    assert full_p == [F(1, 4)] * 4
    assert drop_p == [[F(1, 4), F(1, 4), F(0), F(0)],
                      [F(1, 4), F(1, 4), F(1, 4), F(0)]]
    phase_p = [probabilities(full, [1, 0, 0, sign, 0, 0, 0, 0], F(1, 8))
               for sign in (1, -1)]
    assert phase_p == [[F(0), F(1, 2), F(1, 2), F(0)],
                       [F(1, 2), F(0), F(0), F(1, 2)]]
    return {
        "full_probabilities": [str(x) for x in full_p],
        "deleted_Q_probabilities_1_then_2_and_2_then_1":
            [[str(x) for x in row] for row in drop_p],
        "opposite_hidden_phase_probabilities": [[str(x) for x in row] for row in phase_p],
        "compressed_generator_commutator_squared_frobenius_norm": 2,
        "hidden_curvature_cancellation_exact": True,
        "compressed_su2_closure_and_casimir_2I_exact": True,
        "arithmetic": "fractions.Fraction on realified Gaussian-rational matrices",
    }


def hermitian_exp(h, angle):
    values, vectors = np.linalg.eigh(h)
    return (vectors * np.exp(-1j * angle * values)) @ vectors.conj().T


def memory_propagate(events, initial, visible_dimension):
    """Eliminate Q with its exact initial term and full discrete memory kernel.

    Fixed P is the first visible_dimension coordinates. No projections are
    physically performed between events. All event matrices are supplied.
    """
    d = visible_dimension
    x0, y0 = initial[:d], initial[d:]
    blocks = [(u[:d, :d], u[:d, d:], u[d:, :d], u[d:, d:]) for u in events]
    visible_history = [x0.copy()]
    for k, (a, b, _, _) in enumerate(blocks):
        transport = np.eye(len(y0), dtype=complex)
        memory = np.zeros_like(y0)
        for m in range(k - 1, -1, -1):
            memory += transport @ blocks[m][2] @ visible_history[m]
            transport = transport @ blocks[m][3]
        hidden_at_k = transport @ y0 + memory
        visible_history.append(a @ visible_history[k] + b @ hidden_at_k)
    return visible_history


def numeric_audit():
    rng = np.random.default_rng(SEED)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    h1, h2 = np.kron(sigma_x, np.eye(2)), np.kron(np.eye(2), sigma_x)
    initial = np.eye(4, dtype=complex)[:, 1]
    u1, u2 = [hermitian_exp(h, np.pi / 4) for h in (h1, h2)]
    full = u2 @ u1 @ initial
    assert np.max(np.abs(np.abs(full)**2 - .25)) < TOLERANCE
    p = np.diag([1., 1., 1., 0.])
    q = np.eye(4) - p
    corrected = p @ u2 @ p @ u1 @ p + p @ u2 @ q @ u1 @ p
    two_event_residual = float(np.max(np.abs(corrected - p @ u2 @ u1 @ p)))
    assert two_event_residual < TOLERANCE

    compressed = [hermitian_exp(h[:3, :3], np.pi / 4) for h in (h1, h2)]
    compressed_p = [np.abs(b @ a @ initial[:3])**2
                    for a, b in (compressed, compressed[::-1])]
    assert np.max(np.abs(compressed_p[0] - [.5, .5, 0.])) < TOLERANCE
    assert np.max(np.abs(compressed_p[1] - [.25, .5, .25])) < TOLERANCE

    # Same supplied positive energy operator is conserved by either local event.
    energy = 2 * np.eye(4) + h1 + h2  # in units of an external E_* > 0
    energy_error = max(float(np.max(np.abs(u.conj().T @ energy @ u - energy)))
                       for u in (u1, u2))
    assert energy_error < TOLERANCE

    # Full-state and exact eliminated-Q routes are implemented independently.
    # Include nonzero initial Q amplitudes and noncommuting time-varying events.
    worst_memory = 0.
    worst_norm = 0.
    for dimension, visible in ((4, 3), (5, 2), (6, 4)):
        for _ in range(8):
            psi = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
            psi /= np.linalg.norm(psi)
            events = []
            for _ in range(7):
                m = rng.normal(size=(dimension, dimension)) + 1j*rng.normal(size=(dimension, dimension))
                events.append(hermitian_exp((m + m.conj().T) / 2, .31))
            memory = memory_propagate(events, psi, visible)
            direct = psi.copy()
            for k, u in enumerate(events):
                direct = u @ direct
                worst_memory = max(worst_memory, float(np.max(np.abs(direct[:visible] - memory[k+1]))))
                worst_norm = max(worst_norm, float(abs(np.vdot(direct, direct) - 1)))
    assert worst_memory < TOLERANCE and worst_norm < TOLERANCE

    # With no hidden space the memory route reduces to the full product.
    assert np.max(np.abs(memory_propagate([u1, u2], initial, 4)[-1] - full)) < TOLERANCE
    diagonal = np.diag(np.exp(-1j * np.array([.2, .3, .5, .7])))
    assert np.max(np.abs(p @ diagonal @ q)) == 0.
    assert np.max(np.abs(p @ diagonal @ p @ diagonal @ p - p @ diagonal @ diagonal @ p)) == 0.
    # Genuine noncommutativity of full overlapping events remains observable.
    z1 = np.kron(np.diag([1., -1.]), np.eye(2))
    v = hermitian_exp(z1, np.pi/4)
    noncommuting_error = float(np.max(np.abs(v @ u1 - u1 @ v)))
    assert noncommuting_error > .9

    # Final complete record instrument: R_r = |r><r|. Dilation keeps coherence.
    record = np.zeros((16, 4), dtype=complex)
    for r in range(4):
        record[4*r+r, r] = 1
    assert np.max(np.abs(record.conj().T @ record - np.eye(4))) == 0.
    rho = np.outer(full, full.conj())
    joint = record @ rho @ record.conj().T
    reduced = np.trace(joint.reshape(4, 4, 4, 4), axis1=1, axis2=3)
    assert np.max(np.abs(reduced - np.diag(np.diag(rho)))) < TOLERANCE
    assert abs(np.trace(joint @ joint) - 1) < TOLERANCE
    plus = np.ones(4, dtype=complex) / 2
    plus_rho = np.outer(plus, plus.conj())
    # A supplied record instrument does not supply its apparatus energy budget.
    record_energy_change = float(np.trace(energy @ (np.diag(np.diag(plus_rho)) - plus_rho)).real)
    assert abs(record_energy_change + 2.) < TOLERANCE
    return {
        "compressed_unitary_probabilities": [x.tolist() for x in compressed_p],
        "two_event_memory_reconstruction_max_error": two_event_residual,
        "general_memory_max_error": worst_memory,
        "general_norm_max_error": worst_norm,
        "supplied_energy_operator_max_error": energy_error,
        "noncommuting_full_event_max_difference": noncommuting_error,
        "random_cases": 24,
        "events_per_random_case": 7,
        "seed": SEED,
        "tolerance": TOLERANCE,
        "complete_record_dilation_checked": True,
        "record_system_energy_change_on_plus_plus_in_E_star_units": record_energy_change,
    }


def main():
    result = {
        "candidate": "CE-AM1",
        "status": "finite conditional construction; deletion and compressed-generator equivalence rejected",
        "exact": exact_audit(),
        "numeric": numeric_audit(),
        "not_derived": ["event law", "physical clock", "actual single outcome",
                        "local relativistic record", "metric and Einstein limit",
                        "three gauge sectors", "joint observational RMSE"],
        "provenance": {"python": platform.python_version(), "numpy": np.__version__,
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
