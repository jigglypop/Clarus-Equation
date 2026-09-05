"""Actual four-valent SU(2) coherent intertwiners and cross-face composition.

The supplied kinematics is Inv(H_j tensor H_j tensor H_j tensor H_j), with
four equal areas and dimension 2j+1. It is the Livine-Speziale construction
(arXiv:0705.0674), used as a concrete choice for the overlap in arXiv:2004.07013.
SU(2), four faces, the Hilbert space, and the coherent-state family are inputs;
none is derived from a CE 0D action or from the previous Regge length dynamics.

In the orthonormal basis coupling legs (12) and (34) to k and then a singlet,
I_k(n)=sum_M (-1)^(k-M)/sqrt(2k+1) A_kM(n1,n2) A_k,-M(n3,n4).
This is the UNNORMALIZED invariant projection of the four spin coherent states.
Its norm N is generally not one. G(a,b)=<Ihat(a)|Ihat(b)> keeps its complex phase.

A normalized one-spin quadrature obeys sum_q w_q |j,n_q><j,n_q|=identity.
The weights have total mass 2j+1. After tensoring four frames and projecting,
sum_q w_product |I_q><I_q|=identity_inv. For normalized I_q the weight is
w_product*N_q. The frame runs over ALL four independent normals, not just
the two-dimensional closed-normal shape slice used to probe localization.
Gauss-Legendre integrates diagonal polynomials of degree 2j; the azimuth sum
kills nonzero frequencies up to 2j. The chosen grid therefore integrates the
one-spin frame exactly, up to floating point.

Consequently sum_q weight_q G(a,q)G(q,b)=G(a,b).
Replacing amplitudes with squared moduli gives a different composition.
The rank-one Kraus choice sqrt(weight_q)|Ihat_q><Ihat_q| defines a trace-
preserving measurement/preparation channel, generally not the identity channel.
That instrument is another specified choice, not autonomous CE dynamics.
The quadrature is exact for the frame identity. Its squared-modulus and channel
values describe this finite frame, not exact quadratures of continuum instruments.

For the closed normals parameterized by theta,phi, the regular tetrahedron has
cos(theta)=1/sqrt(3), phi=pi/2. Local shape precision means
K=-Hessian_delta log|G(a,a+delta)|^2=2 Re <partial Ihat|(1-|Ihat><Ihat|)|partial Ihat>.
Its finite-spin growth is compared with j, not promoted to a proved asymptotic
limit or a physical length-resolution law. Pair-angle observables are dot products
J_a.J_b/[j(j+1)], and [J1.J2,J1.J3]=i J1.(J2 cross J3).

Source arXiv:2004.07013 uses L_P_source^2=8*pi*hbar*G/c^3, whereas preceding
repository length calculations used l_P^2=hbar*G/c^3. Physical areas would be
8*pi*gamma*l_P^2*sqrt(j(j+1)); gamma and this interpretation remain supplied.
No microscopic split/merge dynamics, energy budget, Regge phase, continuum
Einstein limit, or dynamic common-metric selection is derived in this module.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import permutations
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np


HERE = Path(__file__).resolve().parent
REGULAR = np.array([math.acos(1/math.sqrt(3)), math.pi/2])


def spin_dimension(two_j):
    if isinstance(two_j, bool) or not isinstance(two_j, int) or not 1 <= two_j <= 16:
        raise ValueError("twice the spin must be an integer from 1 through 16")
    return two_j+1


@lru_cache(maxsize=16)
def coupling(two_j):
    """Equal-spin Clebsch-Gordan coefficients from the factorial Racah sum."""
    d = spin_dimension(two_j)
    f = math.factorial
    out = []
    for k in range(d):
        block = np.zeros((2*k+1, d, d))
        for a in range(d):
            for b in range(d):
                magnetic = a+b-two_j
                if abs(magnetic) > k:
                    continue
                numerator = ((2*k+1)*f(k)**2*f(two_j-k)*f(k+magnetic)*f(k-magnetic)
                             *f(two_j-a)*f(a)*f(two_j-b)*f(b))
                prefactor = Fraction(numerator, f(two_j+k+1))
                lower = max(0, two_j-a-k, b-k)
                upper = min(two_j-k, two_j-a, b)
                series = sum((Fraction((-1)**z,
                    f(z)*f(two_j-k-z)*f(two_j-a-z)*f(b-z)*f(k-two_j+a+z)*f(k-b+z))
                    for z in range(lower, upper+1)), Fraction(0))
                block[magnetic+k, a, b] = math.sqrt(float(prefactor))*float(series)
        out.append(block)
    return tuple(out)


def spin_coherent(two_j, normal):
    d = spin_dimension(two_j)
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isclose(n @ n, 1., atol=1e-12, rtol=0):
        raise ValueError("The normal must be a finite unit vector in three dimensions")
    up = math.sqrt(max(0., (1+float(n[2]))/2))
    down = math.sqrt(max(0., (1-float(n[2]))/2))*np.exp(1j*math.atan2(n[1], n[0]))
    return np.array([math.sqrt(math.comb(two_j, a))*up**a*down**(two_j-a) for a in range(d)])


def intertwiner(two_j, normals, normalize=True):
    normals = np.asarray(normals, dtype=float)
    if normals.shape != (4, 3):
        raise ValueError("Four unit face normals are required")
    states = np.array([spin_coherent(two_j, n) for n in normals])
    values = []
    for k, cg in enumerate(coupling(two_j)):
        left = np.einsum("mab,a,b->m", cg, states[0], states[1])
        right = np.einsum("mab,a,b->m", cg, states[2], states[3])
        signs = (-1.)**np.arange(2*k, -1, -1)
        values.append(np.sum(signs*left*right[::-1])/math.sqrt(2*k+1))
    value = np.array(values)
    norm = float(np.vdot(value, value).real)
    if normalize:
        if norm < 1e-28:
            raise ValueError("The invariant projection has vanishing norm")
        value = value/math.sqrt(norm)
    return value, norm


def shape_normals(theta, phi):
    theta, phi = float(theta), float(phi)
    if (not math.isfinite(theta) or not 0 < theta < math.pi/2
            or not math.isfinite(phi) or abs(math.sin(phi)) < 1e-12):
        raise ValueError("A noncoplanar closed shape with theta between zero and pi/2 is required")
    s, c = math.sin(theta), math.cos(theta)
    x, y = s*math.cos(phi), s*math.sin(phi)
    return np.array([[s, 0, c], [-s, 0, c], [x, y, -c], [-x, -y, -c]])


def shape_state(two_j, coordinates):
    return intertwiner(two_j, shape_normals(*coordinates))[0]


def haar_projected_overlap(two_j, left_normals, right_normals):
    """Independent normalized Haar integral, without Clebsch-Gordan coefficients.

    Four equal spins have integer total angular momentum J<=4j=2*two_j.
    The two azimuth grids kill all nonzero magnetic frequencies; the remaining
    d^J_00 is a Legendre polynomial. This fixes an exact finite quadrature here.
    A 2pi interval suffices for both azimuths because the four-leg integrand is
    invariant under the central sign of SU(2). No such claim is made for one leg.
    """
    spin_dimension(two_j)
    left = np.asarray(left_normals, dtype=float)
    right = np.asarray(right_normals, dtype=float)
    if left.shape != (4, 3) or right.shape != (4, 3):
        raise ValueError("Each tetrahedron requires four unit normals")
    left_spinors = np.array([spin_coherent(1, n)[::-1] for n in left])
    right_spinors = np.array([spin_coherent(1, n)[::-1] for n in right])
    count = 2*two_j+1
    angles = 2*math.pi*np.arange(count)/count
    alpha, gamma = angles[:, None], angles[None, :]
    z, weights = np.polynomial.legendre.leggauss(two_j+1)
    integral = 0j
    for zz, weight in zip(z, weights):
        c, sine = math.sqrt((1+zz)/2), math.sqrt((1-zz)/2)
        group = np.empty((count, count, 2, 2), dtype=complex)
        group[:, :, 0, 0] = c*np.exp(-.5j*(alpha+gamma))
        group[:, :, 0, 1] = -sine*np.exp(-.5j*(alpha-gamma))
        group[:, :, 1, 0] = sine*np.exp(.5j*(alpha-gamma))
        group[:, :, 1, 1] = c*np.exp(.5j*(alpha+gamma))
        overlaps = np.einsum("ai,xyij,aj->xya", left_spinors.conj(), group, right_spinors)
        integral += weight*np.mean(np.prod(overlaps**two_j, axis=-1))/2
    return complex(integral)


def leading_shape_precision(two_j):
    """Leading coherent-distance saddle after minimizing a common rotation.

    This is the saddle comparison, not a proved error bound for finite spin.
    K_leading=j*D^T*(1-R*(R^T R)^-1*R^T)*D.
    """
    spin_dimension(two_j)
    theta, phi = REGULAR
    s, c = math.sin(theta), math.cos(theta)
    normals = shape_normals(theta, phi)
    def cross(n):
        x, y, z = n
        return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    rotation = np.vstack([-cross(n) for n in normals])
    dt = np.array([[c, 0., -s], [-c, 0., -s],
                   [c*math.cos(phi), c*math.sin(phi), s],
                   [-c*math.cos(phi), -c*math.sin(phi), s]])
    dp = np.array([[0., 0., 0.], [0., 0., 0.],
                   [-s*math.sin(phi), s*math.cos(phi), 0.],
                   [s*math.sin(phi), -s*math.cos(phi), 0.]])
    derivative = np.column_stack([dt.ravel(), dp.ravel()])
    horizontal = derivative-rotation @ np.linalg.solve(rotation.T @ rotation, rotation.T @ derivative)
    return (two_j/2)*horizontal.T @ horizontal


def shape_precision(two_j, step=1e-4):
    step = float(step)
    if not math.isfinite(step) or not 0 < step < .01:
        raise ValueError("A positive derivative step below 0.01 is required")
    center = shape_state(two_j, REGULAR)
    derivatives = []
    for axis in np.eye(2):
        estimates = [(shape_state(two_j, REGULAR+h*axis)-shape_state(two_j, REGULAR-h*axis))/(2*h)
                     for h in (step, step/2)]
        derivatives.append((4*estimates[1]-estimates[0])/3)
    tangent = np.array(derivatives).T
    tangent -= np.outer(center, center.conj() @ tangent)
    quantum_geometry = tangent.conj().T @ tangent
    precision = 2*quantum_geometry.real
    return precision, quantum_geometry


@lru_cache(maxsize=4)
def invariant_basis(two_j):
    """Small explicit tensor basis, used for independent projection and operators."""
    d = spin_dimension(two_j)
    if two_j > 4:
        raise ValueError("Explicit tensor checks are restricted to twice-spin at most four")
    basis = np.zeros((d, d, d, d, d))
    for k, cg in enumerate(coupling(two_j)):
        signs = (-1.)**np.arange(2*k, -1, -1)
        basis[k] = np.einsum("mab,mcd,m->abcd", cg, cg[::-1], signs)/math.sqrt(2*k+1)
    return basis


def single_spin_frame(two_j):
    """Positive quadrature whose coherent projector matrix integrates exactly."""
    d = spin_dimension(two_j)
    z, weights_z = np.polynomial.legendre.leggauss(d)
    normals, weights = [], []
    for zz, weight in zip(z, weights_z):
        for azimuth in 2*math.pi*np.arange(d)/d:
            radius = math.sqrt(1-zz*zz)
            normals.append([radius*math.cos(azimuth), radius*math.sin(azimuth), zz])
            weights.append(d*weight/(2*d))
    states = np.array([spin_coherent(two_j, n) for n in normals])
    weights = np.array(weights)
    frame = np.einsum("q,qa,qb->ab", weights, states, states.conj())
    return states, weights, frame


def frame_composition(two_j=2):
    """Explicit four-normal frame, including the normalized-state measure."""
    d = spin_dimension(two_j)
    if two_j > 2:
        raise ValueError("Explicit four-sphere quadrature is restricted to twice-spin at most two")
    states, weights, single_frame = single_spin_frame(two_j)
    basis = invariant_basis(two_j)
    amplitudes = np.einsum("kabcd,ia,jb,lc,md->kijlm",
                          basis.conj(), states, states, states, states, optimize=True).reshape(d, -1)
    product_weights = np.einsum("i,j,l,m->ijlm", weights, weights, weights, weights).ravel()
    norms = np.sum(abs(amplitudes)**2, axis=0)
    # Zero projections have zero measure. Discard only roundoff-size projection norms.
    live = norms > 1e-28
    amplitudes, product_weights, norms = amplitudes[:, live], product_weights[live], norms[live]
    normalized = amplitudes/np.sqrt(norms)
    measure = product_weights*norms
    frame = (normalized*measure) @ normalized.conj().T
    wrong = (normalized*product_weights) @ normalized.conj().T
    left = shape_state(two_j, REGULAR)
    right = shape_state(two_j, REGULAR+np.array([.13, .21]))
    overlap = np.vdot(left, right)
    via_frame = left.conj() @ frame @ right
    left_overlaps = left.conj() @ normalized
    right_overlaps = normalized.conj().T @ right
    probability_composed = float(np.sum(measure*abs(left_overlaps)**2*abs(right_overlaps)**2))
    # This explicit rank-one instrument is CP and TP by the tested frame identity.
    channel_output = (normalized*(measure*abs(left_overlaps)**2)) @ normalized.conj().T
    return {
        "two_j": two_j, "dimension": d, "nonzero_frame_states": int(live.sum()),
        "single_spin_frame_residual": float(np.linalg.norm(single_frame-np.eye(d))),
        "invariant_frame_residual": float(np.linalg.norm(frame-np.eye(d))),
        "normalized_measure_total": float(measure.sum()),
        "omitting_projection_norm_residual": float(np.linalg.norm(wrong-np.eye(d))),
        "omitting_norm_after_trace_rescaling_residual": float(np.linalg.norm(wrong*d/np.trace(wrong)-np.eye(d))),
        "amplitude": [float(overlap.real), float(overlap.imag)],
        "amplitude_composition_residual": float(abs(via_frame-overlap)),
        "direct_fidelity": float(abs(overlap)**2), "squared_moduli_composition": probability_composed,
        "measurement_channel_trace": float(np.trace(channel_output).real),
        "measurement_channel_purity": float(np.trace(channel_output @ channel_output).real),
        "instrument_uses_this_finite_frame": True,
    }


def spin_operators(two_j):
    d = spin_dimension(two_j)
    j = two_j/2
    magnetic = np.arange(d)-j
    raising = np.diag(np.sqrt(j*(j+1)-magnetic[:-1]*(magnetic[:-1]+1)), -1)
    return ((raising+raising.T)/2, (raising-raising.T)/(2j), np.diag(magnetic))


def _apply(operator, leg, states):
    return np.moveaxis(np.tensordot(operator, states, axes=(1, leg+1)), 0, leg+1)


def shape_operators(two_j=2):
    basis = invariant_basis(two_j)
    ops = spin_operators(two_j)
    flat = basis.reshape(len(basis), -1)
    def reduced(tensor):
        return flat.conj() @ tensor.reshape(len(basis), -1).T
    def pair(a, b):
        return reduced(sum(_apply(o, a, _apply(o, b, basis)) for o in ops))
    first, second = pair(0, 1), pair(0, 2)
    triple = np.zeros_like(basis, dtype=complex)
    for permutation in permutations(range(3)):
        sign = (-1)**sum(permutation[i] > permutation[j] for i in range(3) for j in range(i+1, 3))
        term = basis.astype(complex)
        for leg, component in enumerate(permutation):
            term = _apply(ops[component], leg, term)
        triple += sign*term
    volume = reduced(triple)
    total_residual = max(float(np.linalg.norm(sum(_apply(o, leg, basis) for leg in range(4)))) for o in ops)
    return first, second, volume, total_residual


def operator_case(two_j=2):
    first, second, volume, closure = shape_operators(two_j)
    j = two_j/2
    state = shape_state(two_j, REGULAR)
    operators = [first/(j*(j+1)), second/(j*(j+1))]
    centered = np.column_stack([(o-np.vdot(state, o @ state)*np.eye(len(state))) @ state for o in operators])
    moments = centered.conj().T @ centered
    covariance = moments.real
    uncertainty_floor = float(moments[0, 1].imag**2)
    return {
        "two_j": two_j, "singlet_generator_residual": closure,
        "commutator_triple_product_residual": float(np.linalg.norm(first @ second-second @ first-1j*volume)),
        "cosine_covariance": covariance.tolist(),
        "robertson_determinant": float(np.linalg.det(covariance)),
        "robertson_lower_bound": uncertainty_floor,
        "uncertainty_ratio": float(np.linalg.det(covariance)/uncertainty_floor),
    }


def run():
    localization = []
    for n in (2, 4, 8, 16):
        precision, _ = shape_precision(n)
        half, _ = shape_precision(n, 5e-5)
        state = shape_state(n, REGULAR)
        displaced = shape_state(n, REGULAR+np.array([.12, .16]))
        value, norm = intertwiner(n, shape_normals(*REGULAR))
        raw, _ = intertwiner(n, shape_normals(*REGULAR), normalize=False)
        raw_displaced, _ = intertwiner(n, shape_normals(*(REGULAR+np.array([.12, .16]))), normalize=False)
        haar_overlap = haar_projected_overlap(
            n, shape_normals(*REGULAR), shape_normals(*(REGULAR+np.array([.12, .16]))))
        leading = leading_shape_precision(n)
        localization.append({
            "two_j": n, "spin": n/2, "dimension": n+1, "projection_norm_squared": norm,
            "normalized_state_residual": float(abs(np.vdot(value, value)-1)),
            "shape_precision": precision.tolist(), "precision_eigenvalues": np.linalg.eigvalsh(precision).tolist(),
            "precision_divided_by_spin": (precision/(n/2)).tolist(),
            "difference_step_residual": float(np.linalg.norm(half-precision)),
            "displaced_shape_fidelity": float(abs(np.vdot(state, displaced))**2),
            "independent_haar_overlap_residual": float(abs(haar_overlap-np.vdot(raw, raw_displaced))),
            "leading_saddle_precision": leading.tolist(),
            "relative_saddle_difference": float(np.linalg.norm(precision-leading)/np.linalg.norm(leading)),
        })
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_sha256": {"coherent_tetrahedron_overlap.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "localization": localization, "frame_composition": [frame_composition(1), frame_composition(2)],
        "shape_operator_checks": [operator_case(1), operator_case(2), operator_case(4)],
        "scope": {
            "su2_four_face_kinematics_supplied": True, "equal_face_areas_supplied": True,
            "coherent_state_family_supplied": True, "full_complex_overlap_retained": True,
            "normalized_projection_measure_retained": True,
            "closed_shape_slice_used_as_full_frame_measure": False,
            "asymptotic_width_law_rigorously_proven_here": False,
            "physical_length_resolution_derived_from_ce": False,
            "regge_action_phase_or_refinement_dynamics_derived": False,
            "microscopic_split_merge_energy_budget_derived": False,
            "stationary_common_metric_sector_selected": False, "continuum_einstein_limit_derived": False,
        },
    }


if __name__ == "__main__":
    result = run()
    (HERE/"coherent_tetrahedron_overlap.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
