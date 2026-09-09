"""Local flavor response to q=Q/Mpl, not a fitted flavor theory.

M(q)=exp(alpha*q)*(M0+q*C) near q=0. M and C have equal units.
Input matrices and alpha are supplied. Eigenvalues must be positive and simple.
Majorana response uses M^dagger M; Dirac left response uses M M^dagger.
Diagonal eigenvector phases are gauge choices; modulus-squared mixing is used.
"""
import numpy as np


def mass_response(matrix, slope, *, alpha=0., majorana=False):
    m, c = np.asarray(matrix, complex), np.asarray(slope, complex)
    if m.shape != (3, 3) or c.shape != m.shape:
        raise ValueError("two 3x3 matrices required")
    if not (np.isfinite(m).all() and np.isfinite(c).all() and np.isfinite(alpha)):
        raise ValueError("finite inputs required")
    scale = max(float(np.linalg.norm(m)), float(np.linalg.norm(c)))
    if majorana and (not np.allclose(m, m.T, atol=1e-12*scale, rtol=1e-12)
                     or not np.allclose(c, c.T, atol=1e-12*scale, rtol=1e-12)):
        raise ValueError("Majorana matrix and slope must be complex symmetric")
    dm = alpha*m+c
    if majorana:
        h, dh = m.conj().T@m, dm.conj().T@m+m.conj().T@dm
    else:
        h, dh = m@m.conj().T, dm@m.conj().T+m@dm.conj().T
    eigenvalues, u = np.linalg.eigh(h)
    tolerance = 1e-12*max(float(eigenvalues[-1]), np.finfo(float).tiny)
    if eigenvalues[0] <= tolerance or np.any(np.diff(eigenvalues) <= tolerance):
        raise ValueError("positive resolved nondegenerate masses required")
    rotated = u.conj().T@dh@u
    k = np.zeros((3, 3), complex)
    for i in range(3):
        for j in range(3):
            if i != j:
                k[i, j] = rotated[i, j]/(eigenvalues[j]-eigenvalues[i])
    charges = rotated.diagonal().real/(2*eigenvalues)
    return {"masses": np.sqrt(eigenvalues), "basis": u,
            "basis_generator": k, "log_mass_derivatives": charges,
            "log_mass_ratio_derivatives": charges[:, None]-charges[None, :]}


def mixing_response(left, right):
    """CKM (up,down) or PMNS (charged Dirac,neutrino Majorana) response."""
    v = left["basis"].conj().T@right["basis"]
    dv = -left["basis_generator"]@v+v@right["basis_generator"]
    return {"modulus_squared": abs(v)**2,
            "modulus_squared_derivative": 2*np.real(v.conj()*dv)}


def differential_free_fall(alpha_source, alpha_a, alpha_b, force_shape):
    """Signed 2(a_A-a_B)/(a_A+a_B), same radius, weak unscreened point source.

    Composite-body charges must be supplied from atomic/nuclear matching.
    Scalar charge products need not be positive. Total inward force must be.
    """
    values = np.array([alpha_source, alpha_a, alpha_b, force_shape], float)
    if not np.isfinite(values).all() or not 0 <= force_shape <= 1:
        raise ValueError("finite charges and spectral force shape in [0,1] required")
    factors = 1+2*alpha_source*np.array([alpha_a, alpha_b])*force_shape
    if np.any(factors <= 0):
        raise ValueError("positive total inward forces required for this convention")
    return 2*alpha_source*(alpha_a-alpha_b)*force_shape/(1+alpha_source*(alpha_a+alpha_b)*force_shape)


def majorana_vacuum_probabilities(matrix, phase_scale):
    """Coherent relativistic vacuum probabilities; rows final, columns initial.

    phase_scale=L/(2E) in natural units consistent with M's mass units.
    Allows mass degeneracy; assumes a fixed charged-lepton flavor basis.
    No matter potential, source spectrum, detector response or averaging.
    """
    m = np.asarray(matrix, complex)
    if m.shape != (3,3) or not np.isfinite(m).all():
        raise ValueError("finite 3x3 mass matrix required")
    if not np.isfinite(phase_scale) or phase_scale < 0:
        raise ValueError("finite nonnegative phase scale required")
    if not np.allclose(m, m.T, rtol=1e-12, atol=1e-12*np.linalg.norm(m)):
        raise ValueError("Majorana mass matrix must be symmetric")
    lam, u = np.linalg.eigh(m.conj().T@m)
    # Remove an irrelevant common phase before exponentiation.
    evolution = (u*np.exp(-1j*phase_scale*(lam-lam[0])))@u.conj().T
    return abs(evolution)**2
