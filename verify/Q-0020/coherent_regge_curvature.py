"""대칭 LS 상태의 정확한 겹침과 실제 Regge 위상을 결합한다.

This is a specified effective spin-foam test, not a CE microscopic derivation.
The complex has cells (12345),(12356),(13456), one bulk triangle (135),
18 boundary triangles, 9 boundary tetrahedra, and no bulk edges.
We use the homogeneous-area branch b=c, x=y:
 x^2=4a/sqrt(3), z^2=4b^2/x^2+x^2/4.
Strict positive four-simplex Gram matrices require 0<a/b<3/2.
The weaker boundary-tetrahedron condition a/b<sqrt(3) is insufficient.

Two such complexes are joined across their nine boundary tetrahedra. With the
second bulk label fixed, its action contributes only a constant phase. We choose
normalized intertwiners, unit bulk spin measure by default, one orientation
sector, and SU(2)-admissible bulk labels. These are explicit model inputs.
The boundary tetrahedra have four equal face spins, so the previous equal-spin
LS implementation applies exactly along phi=pi/2, cos(theta)=a/(sqrt(3)*b).

For n=2j and c=cos(theta), the EXACT unnormalized LS components are
 I_k=i^n*(-i)^k*sqrt(2k+1)*(n!)^2/[(n-k)!(n+k+1)!]*(1-c^2)^n*R_k(t),
 t=2c/(1-c^2), R_0=1, R_1=t,
 (k+1)R_(k+1)=(2k+1)tR_k+kR_(k-1).
This follows from P_k(i t)=i^k R_k and the Wigner-d generating identity.
The logarithmic recurrence avoids factorial overflow. The k-dependent phases
are constant along this one-dimensional family and cancel between bra and ket.
The resulting overlap is positive here; no modulus or squared fidelity replaces
the amplitude. The gluing factor is the NINTH POWER of this exact overlap.

For the three-cell action, counting all 30 simplex-triangle incidences gives
 S=a*epsilon+9b*(2pi-theta_c-2*theta_b), dS/da=epsilon.
The supplementary reduced formula in arXiv:2004.07013 omits one theta_b term
in the publicly available text. We use its main definition (2), checked against
direct simplex assembly, rather than calibrating to its numerical tables.
This is also why the branch domain is checked from actual Gram positivity.

Geometry areas are dimensionless in units 8pi*gamma*l_P^2, l_P^2=hbar*G/c^3.
Default areas use sqrt(j*(j+1)); the source's j+1/2 spectrum is an optional
separate approximation. In that declared approximation spin zero has positive
area and is included when SU(2)-admissible; this does not validate the asymptotic
area approximation at low spin. The quantum phase is exp(i*gamma*S), including all
boundary contributions. gamma=0 is only a phase-off control.
Z=sum mu*G^9*exp(i*gamma*(S-S_reference)).
Complex amplitude expectations are not probabilities. The reported positive
envelope moments and labelwise squared-modulus averages are separate controls.
The finite sum includes ALL admissible nondegenerate labels in this branch.

The previously checked local saddle K_theta_theta~4j gives, for r=a/b,
 sigma_ja^2~j*(3-r^2)/18 for the amplitude envelope G^9. Thus the local
 phase width scales as gamma*abs(epsilon)*sqrt(j*(3-r^2)/18).
The joint-scaling controls hold gamma*sqrt(j) fixed. They test this local
asymptotic diagnostic; neither a uniform exact-sum limit nor a physical
running law for gamma is proved by these finite cases.

The two-bulk-label partial sum removes the fixed second bulk label, while
keeping all common face areas b fixed. Its kernel is K_ij=<I_i|I_j>^9 and
Z_+=u^T K u, u_i=mu_i exp(i gamma S_i). The bilinear product has NO conjugate:
the same global orientation adds S_i+S_j. The opposite-orientation quantity
u^dagger K u is a separate control, not the full orientation sum. Joining the
two sides adds their boundary pi terms to 2pi. The reported curvature is the
mean of the two bulk deficits. Only the phase-off envelope defines positive
joint probabilities. Soft matching and a prescribed common area do not derive
a CE measure, exact metric matching, or a complete closed-complex spin sum.

References: arXiv:0705.0674, arXiv:2004.07013 (main Eq.2 and supplement).
No general spin-foam refinement law, Lorentzian theory, physical gamma/measure,
autonomous split/merge energy budget, or continuum Einstein limit is derived.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

import fixed_boundary_composition as fixed
import coherent_tetrahedron_overlap as coherent


HERE = Path(__file__).resolve().parent
CELLS = ((1, 2, 3, 4, 5), (1, 2, 3, 5, 6), (1, 3, 4, 5, 6))
BULK = (1, 3, 5)


def _spin(two_j):
    if isinstance(two_j, bool) or not isinstance(two_j, int) or not 1 <= two_j <= 2000:
        raise ValueError("Twice the boundary spin must be an integer from 1 through 2000")


def _logsumexp(values, axis=0):
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum+np.log(np.sum(np.exp(values-maximum), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def symmetric_log_states(two_j, cosines):
    """Normalized component magnitudes and logarithmic raw projection norms."""
    _spin(two_j)
    c = np.atleast_1d(np.asarray(cosines, dtype=float))
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c <= 0) or np.any(c >= 1):
        raise ValueError("Cosines must be a nonempty finite vector strictly between zero and one")
    n = two_j
    log_t = np.log(2*c/(1-c*c))
    previous, current = np.zeros_like(c), log_t
    logs = np.empty((n+1, len(c)))
    log_ratio = -math.log(n+1)
    logs[0] = log_ratio
    for k in range(1, n+1):
        log_ratio += math.log(n-k+1)-math.log(n+k+1)
        logs[k] = .5*math.log(2*k+1)+log_ratio+current
        following = np.logaddexp(math.log(2*k+1)+log_t+current,
                                  math.log(k)+previous)-math.log(k+1)
        previous, current = current, following
    log_norm_without_common = _logsumexp(2*logs, axis=0)
    log_norm = log_norm_without_common+2*n*np.log(1-c*c)
    return logs-log_norm_without_common/2, log_norm


def symmetric_state(two_j, cosine, normalize=True):
    logs, log_norm = symmetric_log_states(two_j, [cosine])
    phase = (1j)**two_j*(-1j)**np.arange(two_j+1)
    state = np.exp(logs[:, 0])*phase
    if not normalize:
        state *= math.exp(float(log_norm[0])/2)
    return state, float(log_norm[0])


def geometric_functions(a, b):
    a, b = np.asarray(a, dtype=float), float(b)
    if not math.isfinite(b) or b <= 0 or not np.isfinite(a).all() or np.any(a <= 0) or np.any(a >= 1.5*b):
        raise ValueError("Nondegenerate branch requires b>0 and 0<a/b<3/2")
    ratio2 = (a/b)**2
    bulk_angle = np.arccos((9-7*ratio2)/(9-ratio2))
    theta_c = np.arccos((3-2*ratio2)/(6-2*ratio2))
    theta_b = np.arccos(ratio2/np.sqrt(27-12*ratio2+ratio2**2))
    deficit = 2*math.pi-3*bulk_angle
    action = a*deficit+9*b*(2*math.pi-theta_c-2*theta_b)
    phi = np.arccos(ratio2/3)
    return action, deficit, phi


def cell_lengths(cell, a, b):
    geometric_functions(a, b)
    x2 = 4*a/math.sqrt(3)
    z2 = 4*b*b/x2+x2/4
    return np.array([math.sqrt(x2 if (u % 2 == v % 2) else z2) for u, v in combinations(cell, 2)])


def direct_geometry(a, b):
    """Independent assembly from all actual simplex angle/area incidences."""
    expected, epsilon, _ = geometric_functions(a, b)
    owners = Counter(tuple(sorted(t)) for cell in CELLS for t in combinations(cell, 3))
    tetra_owners = Counter(tuple(sorted(t)) for cell in CELLS for t in combinations(cell, 4))
    action, gradients = 0., 0.
    bulk_angles = []
    min_gram = math.inf
    r = fixed.gluing.reference()
    for cell in CELLS:
        lengths = cell_lengths(cell, a, b)
        kappas = np.array([(2 if tuple(sorted(t)) == BULK else 1)*math.pi/owners[tuple(sorted(t))]
                           for t in combinations(cell, 3)])
        value, gradient, _, _ = fixed.simplex_derivatives(lengths, kappas)
        action += value
        squared = np.zeros((5, 5))
        for index, (u, v) in enumerate(combinations(range(5), 2)):
            squared[u, v] = squared[v, u] = lengths[index]**2
        gram = (squared[0, 1:, None]+squared[0, None, 1:]-squared[1:, 1:])/2
        min_gram = min(min_gram, float(np.linalg.eigvalsh(gram)[0]))
        local_bulk = tuple(cell.index(v) for v in BULK)
        bulk_angles.append(r.RG._dihedral_angle(tuple(range(5)), local_bulk, lengths, None))
        dx2 = 4/math.sqrt(3)
        x2 = 4*a/math.sqrt(3)
        dz2 = -4*b*b*dx2/x2**2+dx2/4
        direction = np.array([(dx2 if u % 2 == v % 2 else dz2)/(2*lengths[k])
                              for k, (u, v) in enumerate(combinations(cell, 2))])
        gradients += gradient @ direction
    tetra_lengths = cell_lengths((1, 2, 3, 4), a, b)
    squared = np.zeros((4, 4))
    for length, (u, v) in zip(tetra_lengths, combinations(range(4), 2)):
        squared[u, v] = squared[v, u] = length**2
    gram = (squared[0, 1:, None]+squared[0, None, 1:]-squared[1:, 1:])/2
    basis = np.vstack((-np.ones(3), np.eye(3)))
    normals = basis @ np.linalg.solve(gram, basis.T)
    boundary_phi = math.acos(-normals[2, 3]/math.sqrt(normals[2, 2]*normals[3, 3]))
    return {
        "a": a, "b": b, "triangles": len(owners),
        "boundary_phi_residual": float(abs(boundary_phi-geometric_functions(a, b)[2])),
        "boundary_tetrahedra": sum(n == 1 for n in tetra_owners.values()),
        "triangle_incidences": sum(owners.values()),
        "minimum_cell_gram_eigenvalue": min_gram,
        "direct_action": float(action), "closed_action": float(expected),
        "action_residual": float(abs(action-expected)),
        "deficit_residual": float(abs(2*math.pi-sum(bulk_angles)-epsilon)),
        "schlafli_chain_rule_residual": float(abs(gradients-epsilon)),
    }


def area_spectrum(two_spin, spectrum="casimir"):
    labels = np.asarray(two_spin, dtype=float)
    if not np.isfinite(labels).all() or np.any(labels < 0) or np.any(labels != np.floor(labels)):
        raise ValueError("Twice-spin labels must be finite nonnegative integers")
    j = labels/2
    if spectrum == "casimir":
        return np.sqrt(j*(j+1))
    if spectrum == "linear":
        return j+.5
    raise ValueError("Spectrum must be casimir or linear")


@lru_cache(maxsize=5)
def kernel_data(two_j, reference_ratio=1.28, spectrum="casimir"):
    _spin(two_j)
    reference_ratio = float(reference_ratio)
    if not math.isfinite(reference_ratio) or not 0 < reference_ratio < 1.5:
        raise ValueError("Reference area ratio must lie strictly between zero and 3/2")
    boundary_area = float(area_spectrum(two_j, spectrum))
    # Singlet admissibility for (j_a,j,j,j) requires j_a+3j integer and j_a<=3j.
    labels = np.arange(two_j % 2, 3*two_j+1, 2)
    areas = area_spectrum(labels, spectrum)
    valid = (areas > 0) & (areas < 1.5*boundary_area)
    labels, areas = labels[valid], areas[valid]
    if not len(labels):
        raise ValueError("No nondegenerate admissible bulk labels exist")
    reference_index = int(np.argmin(abs(areas-reference_ratio*boundary_area)))
    cosines = areas/(math.sqrt(3)*boundary_area)
    logs, _ = symmetric_log_states(two_j, cosines)
    overlap_logs = _logsumexp(logs+logs[:, reference_index, None], axis=0)
    if overlap_logs.max() > 1e-9:
        raise ArithmeticError("Normalized overlap exceeds one")
    overlap_logs = np.minimum(overlap_logs, 0.)
    action, deficit, phi = geometric_functions(areas, boundary_area)
    return {
        "labels": labels, "areas": areas, "boundary_area": boundary_area,
        "reference_index": reference_index, "log_gluing": 9*overlap_logs,
        "action": action, "deficit": deficit, "phi": phi,
    }


def amplitude_sum(two_j, gamma, reference_ratio=1.28, spectrum="casimir", measure="unit"):
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("Gamma must be finite and nonnegative; zero is a phase-off control")
    data = kernel_data(two_j, reference_ratio, spectrum)
    labels, areas, reference = data["labels"], data["areas"], data["reference_index"]
    log_measure = np.zeros(len(labels))
    if measure == "dimension":
        log_measure = np.log(labels+1)
    elif measure != "unit":
        raise ValueError("Measure must be unit or dimension")
    log_weights = data["log_gluing"]+log_measure
    shift = float(log_weights.max())
    weights = np.exp(log_weights-shift)
    phase = gamma*(data["action"]-data["action"][reference])
    if not np.isfinite(phase).all():
        raise ValueError("The requested phase exceeds finite floating-point range")
    amplitude = weights*np.exp(1j*phase)
    denominator = complex(np.sum(amplitude))
    numerator = complex(np.sum(amplitude*data["deficit"]))
    envelope_sum = float(weights.sum())
    cancellation = abs(denominator)/envelope_sum
    stable = cancellation > 1e-12
    expectation = numerator/denominator if stable else None
    probabilities = weights/envelope_sum
    spin = labels/2
    mean_spin = float(probabilities @ spin)
    sigma_spin = math.sqrt(float(probabilities @ (spin-mean_spin)**2))
    phase_scale = float(np.max(abs(phase)))
    scaled_phase = phase/phase_scale if phase_scale else np.zeros_like(phase)
    scaled_mean = float(probabilities @ scaled_phase)
    phase_sigma = phase_scale*math.sqrt(float(probabilities @ (scaled_phase-scaled_mean)**2))
    j_ref = spin[reference]
    derivative = (2*j_ref+1)/(2*areas[reference]) if spectrum == "casimir" else 1.
    slope_width = float(abs(data["deficit"][reference])*derivative*sigma_spin)
    linear_phase_sigma = gamma*slope_width
    if not math.isfinite(linear_phase_sigma) or not math.isfinite(phase_sigma):
        raise ValueError("The requested phase diagnostics exceed floating-point range")
    squared = weights**2/np.sum(weights**2)
    actual_ratio = float(areas[reference]/data["boundary_area"])
    leading_sigma = math.sqrt((two_j/2)*(3-actual_ratio**2)/18)
    return {
        "two_j": two_j, "boundary_spin": two_j/2, "gamma": gamma, "spectrum": spectrum, "measure": measure,
        "admissible_bulk_labels": len(labels), "reference_two_ja": int(labels[reference]),
        "requested_reference_area_ratio": reference_ratio,
        "actual_reference_area_ratio": float(areas[reference]/data["boundary_area"]),
        "reference_area_ratio_snap_error": float(areas[reference]/data["boundary_area"]-reference_ratio),
        "reference_curvature": float(data["deficit"][reference]),
        "complex_curvature_expectation": None if expectation is None else [expectation.real, expectation.imag],
        "scaled_partition": [denominator.real, denominator.imag],
        "scaled_curvature_numerator": [numerator.real, numerator.imag],
        "log_weight_shift": shift, "scaled_envelope_sum": envelope_sum,
        "cancellation_ratio": cancellation, "denominator_resolved": stable,
        "denominator_cancellation_threshold": 1e-12,
        "phase_off_envelope_curvature": float(probabilities @ data["deficit"]),
        "labelwise_squared_modulus_curvature": float(squared @ data["deficit"]),
        "envelope_spin_sigma": sigma_spin, "linear_phase_sigma": linear_phase_sigma,
        "leading_local_envelope_spin_sigma": leading_sigma,
        "envelope_width_relative_to_leading": sigma_spin/leading_sigma,
        "gamma_sqrt_boundary_spin": gamma*math.sqrt(two_j/2),
        "measured_phase_sigma": phase_sigma,
        "linear_gaussian_cancellation_estimate": math.exp(-.5*linear_phase_sigma*linear_phase_sigma),
        "all_admissible_labels_in_branch_summed": True,
    }


@lru_cache(maxsize=2)
def pair_kernel_data(two_j, spectrum="casimir"):
    """두 내부 라벨 전체의 정확한 겹침 행렬을 만든다."""
    data = kernel_data(two_j, spectrum=spectrum)
    cosines = data["areas"]/(math.sqrt(3)*data["boundary_area"])
    logs, _ = symmetric_log_states(two_j, cosines)
    states = np.exp(logs)
    overlap = states.T @ states
    if not np.isfinite(overlap).all() or overlap.max() > 1+1e-9:
        raise ArithmeticError("Pair overlaps must be finite and at most one")
    return {"labels": data["labels"], "areas": data["areas"],
            "boundary_area": data["boundary_area"], "action": data["action"],
            "deficit": data["deficit"], "kernel": np.minimum(overlap, 1.)**9}


def pair_sum(two_j, gamma, spectrum="casimir", measure="unit"):
    """공통 면적을 고정하고 두 내부 라벨을 모두 합산한다."""
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0:
        raise ValueError("Gamma must be finite and nonnegative")
    data = pair_kernel_data(two_j, spectrum)
    kernel, labels = data["kernel"], data["labels"]
    mu = np.ones(len(labels))
    if measure == "dimension":
        mu = labels.astype(float)+1
    elif measure != "unit":
        raise ValueError("Measure must be unit or dimension")
    measure_scale = float(mu.max())
    mu /= measure_scale
    # The reference only removes a common phase; it fixes neither bulk label.
    reference = int(np.argmin(abs(data["deficit"])))
    phase = gamma*(data["action"]-data["action"][reference])
    if not np.isfinite(phase).all():
        raise ValueError("The pair phase exceeds finite floating-point range")
    u = mu*np.exp(1j*phase)
    ku = kernel @ u
    marginal = u*ku
    partition = complex(marginal.sum())
    curvature_numerator = complex(data["deficit"] @ marginal)
    opposite = complex(np.vdot(u, ku))
    envelope_marginal = mu*(kernel @ mu)
    envelope_sum = float(envelope_marginal.sum())
    cancellation = abs(partition)/envelope_sum
    stable = cancellation > 1e-12
    ratio = data["areas"]/data["boundary_area"]
    # Direct nonnegative differences avoid subtracting close second moments.
    difference_squared = (ratio[:, None]-ratio[None, :])**2
    mismatch_kernel = kernel*difference_squared
    mismatch_numerator = complex(u @ mismatch_kernel @ u)
    envelope_mismatch = float(mu @ mismatch_kernel @ mu/envelope_sum)
    mean_ratio = float(ratio @ envelope_marginal/envelope_sum)
    centered = ratio-mean_ratio
    common_ratio_variance = float(
        ((centered**2) @ envelope_marginal
         +(centered*mu) @ kernel @ (centered*mu))/(2*envelope_sum))
    expectation = curvature_numerator/partition if stable else None
    mismatch_expectation = mismatch_numerator/partition if stable else None
    return {
        "two_j": two_j, "boundary_spin": two_j/2, "gamma": gamma,
        "spectrum": spectrum, "measure": measure,
        "bulk_label_count_per_side": len(labels), "bulk_label_pairs": len(labels)**2,
        "common_face_areas_fixed": True, "second_bulk_label_fixed": False,
        "orientation": "same", "curvature_observable": "mean_of_two_bulk_deficits",
        "phase_reference_two_ja": int(labels[reference]),
        "phase_reference_curvature": float(data["deficit"][reference]),
        "scaled_partition": [partition.real, partition.imag],
        "scaled_curvature_numerator": [curvature_numerator.real, curvature_numerator.imag],
        "scaled_relative_mismatch_numerator": [mismatch_numerator.real, mismatch_numerator.imag],
        "log_weight_shift": 2*math.log(measure_scale),
        "scaled_envelope_sum": envelope_sum, "cancellation_ratio": cancellation,
        "denominator_resolved": stable, "denominator_cancellation_threshold": 1e-12,
        "complex_mean_curvature": None if expectation is None else [expectation.real, expectation.imag],
        "complex_relative_area_mismatch": None if mismatch_expectation is None else
            [mismatch_expectation.real, mismatch_expectation.imag],
        "phase_off_mean_curvature": float(data["deficit"] @ envelope_marginal/envelope_sum),
        "phase_off_relative_area_mismatch": envelope_mismatch,
        "phase_off_common_area_ratio_mean": mean_ratio,
        "phase_off_common_area_ratio_variance": common_ratio_variance,
        "opposite_orientation_scaled_partition": [opposite.real, opposite.imag],
        "opposite_orientation_partition_ratio": opposite.real/envelope_sum,
        "opposite_orientation_is_separate_control": True,
        "all_admissible_bulk_pairs_in_branch_summed": True,
    }


def run():
    comparisons = []
    for n in (1, 2, 4, 8, 16):
        for c in (.4, 1/math.sqrt(3), .8):
            candidate, log_norm = symmetric_state(n, c, normalize=False)
            original, norm = coherent.intertwiner(n, coherent.shape_normals(math.acos(c), math.pi/2), normalize=False)
            comparisons.append({
                "two_j": n, "cosine": c, "state_residual": float(np.linalg.norm(candidate-original)),
                "projection_norm_residual": abs(math.exp(log_norm)-norm),
            })
    cases = [amplitude_sum(n, gamma) for n in (19, 199, 1999) for gamma in (0., .01, .1, .5)]
    cases += [amplitude_sum(199, .1, measure="dimension"), amplitude_sum(199, .1, spectrum="linear")]
    theta = math.acos(.25)
    files = ["coherent_regge_curvature.py", "coherent_tetrahedron_overlap.py",
             "fixed_boundary_composition.py", "length_gluing_bath.py", "local_refinement_bath.py",
             "continuum_bath.py", "F-01/predict_fold_budget.py",
             "F-01/regge_one_to_five_boundary_hessian.py", "F-01/regge_one_to_five_refinement.py"]
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "source_sha256": {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files},
        "exact_state_comparisons": comparisons,
        "geometry_checks": [direct_geometry(a, 1.) for a in (.6, 1., 1.28, 1.49)],
        "regular_action_check": {
            "incidence_action": 20*math.pi-30*theta,
            "supplement_single_theta_b_expression": 20*math.pi-21*theta,
            "difference": 9*theta,
        },
        "cases": cases,
        "joint_scaling_cases": [amplitude_sum(n, coupling/math.sqrt(n/2))
                               for coupling in (1., 3.) for n in (19, 199, 1999)],
        "two_bulk_partial_sums": [pair_sum(n, gamma)
                                  for n in (19, 99, 199, 399, 799)
                                  for gamma in (0., .1, .5)],
        "two_bulk_measure_control": pair_sum(199, .5, measure="dimension"),
        "scope": {
            "exact_symmetric_ls_overlap_used": True, "bulk_spin_singlet_parity_enforced": True,
            "regge_phase_includes_boundary_action": True,
            "normalized_intertwiners_and_measure_supplied": True,
            "barbero_immirzi_parameter_supplied": True, "one_orientation_sector_supplied": True,
            "general_area_to_length_branches_exhausted": False,
            "complex_expectation_is_probability_average": False,
            "low_spin_linear_spectrum_accuracy_proven": False,
            "floating_point_cancellation_error_rigorously_bounded": False,
            "source_gaussian_tables_used_as_validation_target": False,
            "uniform_exact_sum_joint_limit_proven": False,
            "gamma_running_law_physically_derived": False,
            "all_common_face_spins_summed": False,
            "full_orientation_sum_implemented": False,
            "two_bulk_sum_is_physical_refinement": False,
            "microscopic_ce_dynamics_derived": False,
            "autonomous_split_merge_energy_budget_derived": False,
            "common_metric_continuum_sector_proven": False, "continuum_einstein_limit_derived": False,
        },
    }


if __name__ == "__main__":
    result = run()
    (HERE/"coherent_regge_curvature.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
