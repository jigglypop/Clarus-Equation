"""Finite-dimensional checks for one-way/record compatibility."""
from __future__ import annotations

import math
import numpy as np

TOL = 1e-12


def close(actual, expected, label):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    assert error <= TOL, f"{label}: {error}"
    print(f"PASS {label}: max error={error:.3e}")


I2 = np.eye(2, dtype=complex)
P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

# Informative projective record: the two input states give different records.
rho0 = P0.copy()
rho1 = P1.copy()
projective_effects = [P0, P1]
p_rho0 = np.array([np.trace(E @ rho0).real for E in projective_effects])
p_rho1 = np.array([np.trace(E @ rho1).real for E in projective_effects])
close(p_rho0, [1.0, 0.0], "projective record for |0>")
close(p_rho1, [0.0, 1.0], "projective record for |1>")
assert np.linalg.norm(p_rho0 - p_rho1, ord=1) > 1.0
print("PASS informative record signals M to R")

# No-signalling control: scalar effects give a constant distribution.
scalar_effects = [0.3 * I2, 0.7 * I2]
for index, rho in enumerate([rho0, rho1, 0.5 * I2]):
    probs = np.array([np.trace(E @ rho).real for E in scalar_effects])
    close(probs, [0.3, 0.7], f"constant record distribution {index}")
for E in scalar_effects:
    c = np.trace(E).real / 2.0
    close(E, c * I2, "scalar-effect theorem control")

# The informative projective effect is not proportional to identity.
c0 = np.trace(P0).real / 2.0
residual = float(np.linalg.norm(P0 - c0 * I2, ord="fro"))
close(residual, math.sqrt(0.5), "informative effect non-scalar residual")

# Explicit finite-duration system-apparatus interaction.
U = np.kron(P0, I2) - 1j * np.kron(P1, X)
I4 = np.eye(4, dtype=complex)
close(U.conj().T @ U, I4, "finite-duration interaction unitarity")
ket0 = np.array([[1.0], [0.0]], dtype=complex)
ket1 = np.array([[0.0], [1.0]], dtype=complex)
embed = np.kron(I2, ket0)
extract0 = np.kron(I2, ket0.conj().T)
extract1 = np.kron(I2, ket1.conj().T)
M0 = extract0 @ U @ embed
M1 = extract1 @ U @ embed
close(M0, P0, "pointer Kraus M0")
close(M1, -1j * P1, "pointer Kraus M1")
close(M0.conj().T @ M0 + M1.conj().T @ M1, I2,
      "pointer Kraus completeness")
close(M0.conj().T @ M0, P0, "pointer effect E0")
close(M1.conj().T @ M1, P1, "pointer effect E1")

print(f"informative_effect_residual={residual:.15f}")
print("ALL CHECKS PASSED")
