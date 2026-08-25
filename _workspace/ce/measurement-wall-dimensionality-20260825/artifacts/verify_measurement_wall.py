"""Independent numerical/algebraic checks for the finite-duration wall witness."""
from __future__ import annotations

import math
import numpy as np

TOL = 1e-12


def close(a, b, label):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert err <= TOL, f"{label}: {err}"
    print(f"PASS {label}: max error={err:.3e}")


P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
I = np.eye(2, dtype=complex)
rho0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)


def dephase(rho):
    return P0 @ rho @ P0 + P1 @ rho @ P1


eta = 1.0 - math.exp(-1.5)  # gamma=2, Delta t=.75
phi = (1.0 - eta) * rho0 + eta * dephase(rho0)
expected = np.array([[0.5, 0.5 * math.exp(-1.5)],
                     [0.5 * math.exp(-1.5), 0.5]], dtype=complex)

close(dephase(dephase(rho0)), dephase(rho0), "D_P idempotence")
close(P0.conj().T @ P0 + P1.conj().T @ P1, I, "D_P Kraus completeness")
close(phi, expected, "constant-rate generator solution")
close(np.diag(phi), np.diag(rho0), "diagonal preservation")
close(phi[0, 1], (1.0 - eta) * rho0[0, 1], "off-diagonal scaling")

# Kraus completeness for Phi_eta.
kraus = [math.sqrt(1.0 - eta) * I, math.sqrt(eta) * P0, math.sqrt(eta) * P1]
close(sum((K.conj().T @ K for K in kraus)), I, "Phi_eta Kraus completeness")
close(np.trace(phi), 1.0, "Phi_eta trace preservation on rho0")
assert float(np.min(np.linalg.eigvalsh(phi))) >= -TOL
print("PASS Phi_eta output positivity")

# A finite integrated rate cannot reach the hard-wall value eta=1.
assert 0.0 < eta < 1.0
print("PASS finite-strength partial-wall bound")

# Counterexample to treating arbitrary POVM effects as the projectors in the
# contract's sandwich formula: E0=E1=I/2 has E0+E1=I, but sum E_r^2=I/2.
E0 = 0.5 * I
E1 = 0.5 * I
close(E0 + E1, I, "POVM effect completeness")
close(E0 @ E0 + E1 @ E1, 0.5 * I, "sandwich trace loss for nonprojective effects")

# Predictable ensemble opportunity cost for a constant binary distribution.
p = np.array([0.8, 0.2], dtype=float)
cbar = float(np.sum(p * (1.0 - p) * (-np.log(p))))
expected_cbar = 0.8 * 0.2 * (-math.log(0.8 * 0.2))
close(cbar, expected_cbar, "binary predictable opportunity cost")
c_wall = eta * cbar  # integral dot(eta) cbar dt for constant p
close(c_wall, (1.0 - math.exp(-1.5)) * expected_cbar,
      "constant-probability wall-weighted cost")
print(f"eta={eta:.15f}; offdiag={phi[0,1].real:.15f}")
print(f"cbar={cbar:.15f}; c_wall={c_wall:.15f}")
print("ALL CHECKS PASSED")
