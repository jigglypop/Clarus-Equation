"""Exact witnesses for CE-MATH-B1/R1/E1; general proofs are in the manuscript.

This verifies finite identities and counterexamples. It is not a proof assistant,
a Riemann-hypothesis proof, or a physical observation/selection model.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import sympy as s


B_PREREG = '3232e0eea0fa69bea180f508673298cb0f330fccc5e696190eebebe178d53921'
R_PREREG = 'b16bae05689ff530e994688baa98b9456389c6a3a1eb482954ed3b1dd969e8e6'
E_PREREG = '27b11733f2beec5c45f9afc524acd645c7ab8d54f18d6094b6e342060c670904'
TOL = 1e-10
I = s.eye(2)
X = s.Matrix([[0, 1], [1, 0]])
Y = s.Matrix([[0, -s.I], [s.I, 0]])
Z = s.diag(1, -1)
EXACT = {}
ERRORS = {}
COUNTEREXAMPLES = {}


def exact(name, value):
    values = list(value) if isinstance(value, s.MatrixBase) else [value]
    assert all(s.simplify(s.expand_complex(v)) == 0 for v in values), (name, value)
    EXACT[name] = True


def close(name, value):
    error = float(np.max(np.abs(value)))
    assert np.isfinite(error) and error < TOL, (name, error)
    ERRORS[name] = error


def to_numpy(matrix):
    return np.array(matrix.evalf(), dtype=complex)


def density(vector):
    return vector*vector.H


def curvature(metric, variables):
    """Independent Christoffel/Riemann contraction in two coordinates."""
    inv = metric.inv()
    gamma = [[[s.simplify(sum(inv[i, a]*(s.diff(metric[a, k], variables[j])
                + s.diff(metric[a, j], variables[k])
                - s.diff(metric[j, k], variables[a]))/2 for a in range(2)))
               for k in range(2)] for j in range(2)] for i in range(2)]
    riemann = s.diff(gamma[0][1][1], variables[0])-s.diff(gamma[0][0][1], variables[1])
    riemann += sum(gamma[0][0][m]*gamma[m][1][1]
                   - gamma[0][1][m]*gamma[m][0][1] for m in range(2))
    return s.trigsimp(s.simplify(metric[0, 0]*riemann/metric.det()))


def boundary_checks():
    rx, ry, rz, t = s.symbols('rx ry rz t', real=True)
    rho = (I+rx*X+ry*Y+rz*Z)/2
    exact('B2_qubit_toggle', X.H*Z*X+Z)
    exact('B3_Bloch_transform', X*rho*X-(I+rx*X-ry*Y-rz*Z)/2)
    exact('B3_fixed_diameter', X*((I+rx*X)/2)*X-(I+rx*X)/2)
    ystate = (I+Y)/2
    exact('B3_boundary_not_fixed_initial', s.trace(ystate*Z))
    exact('B3_boundary_not_fixed_final', s.trace(X*ystate*X*Z))
    assert X*ystate*X != ystate
    exact('B3_coherent_and_mixed_differ', s.trace(((I+X)/2-I/2)*X)-1)
    twirl = lambda v: (v+X*v*X)/2
    exact('B3_twirl_idempotence', twirl(twirl(rho))-twirl(rho))
    k0, k1 = s.Matrix([[0, 0], [1, 0]]), s.Matrix([[0, 1], [0, 0]])
    exact('B3_irreversible_toggle_TP', k0.H*k0+k1.H*k1-I)
    exact('B3_irreversible_dual_flip', k0.H*Z*k0+k1.H*Z*k1+Z)
    channel = k0*rho*k0.H+k1*rho*k1.H
    exact('B3_boundary_image_singleton', channel.subs(rz, 0)-I/2)
    COUNTEREXAMPLES['B3_not_every_boundary_state_is_fixed'] = True
    COUNTEREXAMPLES['B3_noninvertible_flip_not_onto_boundary'] = True

    a6 = s.diag(2, 1, 0, -1, -2, 0)
    u6 = s.zeros(6)
    for i, j in enumerate([4, 3, 2, 1, 0, 5]):
        u6[j, i] = 1
    exact('B2_spectral_pairing_unitarity', u6.H*u6-s.eye(6))
    exact('B2_spectral_pairing_toggle', u6.H*a6*u6+a6)
    exact('B2_spectral_pairing_involution', u6*u6-s.eye(6))
    assert s.trace(s.diag(1, 1, -1)) != s.trace(-s.diag(1, 1, -1))
    COUNTEREXAMPLES['B2_unbalanced_spectrum_has_no_unitary_toggle'] = True

    ux = s.cos(t/2)*I-s.I*s.sin(t/2)*X
    tangent_u = s.cos(t/2)*I-s.I*s.sin(t/2)*(X+Z)/s.sqrt(2)
    north = s.diag(1, 0)
    exact('B4_actual_crossing', s.trigsimp(s.trace(ux*north*ux.H*Z))-s.cos(t))
    tangent = s.trigsimp(s.trace(tangent_u*north*tangent_u.H*Z))
    exact('B4_tangent_touch', tangent-(1+s.cos(t))/2)
    exact('B4_tangent_derivative', s.diff(tangent, t).subs(t, s.pi))
    exact('B4_tangent_second_derivative', s.diff(tangent, t, 2).subs(t, s.pi)-s.Rational(1, 2))
    exact('B4_boundary_leaves_at_intermediate_time',
          s.trigsimp(s.trace(ux*ystate*ux.H*Z))-s.sin(t))
    COUNTEREXAMPLES['B4_zero_need_not_be_crossing_or_record'] = True

    chi, phi = s.symbols('chi phi', real=True)
    radius = s.symbols('r', positive=True)
    exact('B9_flat_boundary_curvature',
          curvature(s.diag(s.Rational(1, 2), radius**2/2), [radius, phi]))
    exact('B9_curved_boundary_curvature',
          curvature(s.diag(s.Rational(1, 4), s.sin(chi)**2/4), [chi, phi])-4)
    COUNTEREXAMPLES['B9_same_boundary_different_curvatures'] = [0, 4]


def observation_checks():
    p = s.diag(1, 1, 0, 0)
    ap = s.diag(1, -1, 0, 0)
    h = s.zeros(4)
    h[0, 2] = h[2, 0] = 1
    ks = [s.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]])]
    for j in [2, 3]:
        k = s.zeros(3, 4)
        k[2, j] = 1
        ks.append(k)
    exact('B6_observation_TP', sum((k.H*k for k in ks), s.zeros(4))-s.eye(4))
    dilation = sum((s.kronecker_product(k, s.eye(3)[:, j]) for j, k in enumerate(ks)),
                   s.zeros(9, 4))
    exact('B6_dilation_isometry', dilation.H*dilation-s.eye(4))
    choi = s.zeros(12)
    for k in ks:
        vec = s.Matrix([k[a, b] for b in range(4) for a in range(3)])
        choi += vec*vec.H
    assert all(v.is_nonnegative for v in choi.eigenvals())
    EXACT['B6_observation_Choi_positive'] = True
    states, outputs = [], []
    for sign in [1, -1]:
        psi = s.Matrix([1, 1, sign*s.I*s.sqrt(2), 0])/2
        rho = density(psi)
        exact(f'B7_norm_{sign}', s.trace(rho)-1)
        exact(f'B7_visible_trace_{sign}', s.trace(p*rho)-s.Rational(1, 2))
        exact(f'B7_initial_boundary_{sign}', s.trace(rho*ap))
        exact(f'B7_derivative_{sign}', s.I*s.trace(rho*(h*ap-ap*h))-sign/s.sqrt(2))
        exact(f'B7_same_initial_energy_{sign}', s.trace(rho*h))
        states.append(rho)
        outputs.append(sum((k*rho*k.H for k in ks), s.zeros(3)))
    exact('B7_identical_observed_outputs', outputs[0]-outputs[1])
    exact('B7_orthogonal_global_states', s.trace(states[0]*states[1]))
    total = dilation*states[0]*dilation.H
    traced = s.Matrix(3, 3, lambda a, b: sum(total[3*a+j, 3*b+j] for j in range(3)))
    exact('B6_dilation_partial_trace_output', traced-outputs[0])
    assert states[0] != states[1]
    assert all(v.is_nonnegative for v in (h+s.eye(4)).eigenvals())
    COUNTEREXAMPLES['B7_equal_visible_state_opposite_future_bias'] = ['+1/sqrt(2)', '-1/sqrt(2)']

    p2, e2 = s.diag(1, 0), (I+X)/2
    compressed = p2*e2*p2
    exact('B5_compression_defect', compressed-compressed**2-p2*e2*(I-p2)*e2*p2)
    exact('B5_nonprojective_compression', compressed[0, 0]-s.Rational(1, 2))
    COUNTEREXAMPLES['B6_compression_trace_not_one'] = '1/2'

    rng = np.random.default_rng(120911)
    errors, defect_errors = [], []
    for _ in range(16):
        z = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
        unitary, _ = np.linalg.qr(z)
        question = unitary[:, :2]@unitary[:, :2].conj().T
        proj = np.diag([1., 1., 0., 0.])
        f = proj@question@proj
        defect_errors.append(np.max(abs(f-f@f-proj@question@(np.eye(4)-proj)@question@proj)))
        left = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
        right = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
        r, y = left@left.conj().T, right@right.conj().T
        weight = np.trace(r+y).real
        r, y = r/weight, y/weight
        def sqrt_psd(m):
            val, vectors = np.linalg.eigh(m)
            return (vectors*np.sqrt(val))@vectors.conj().T
        sr, sy = sqrt_psd(r), sqrt_psd(y)
        k = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
        k *= .7/np.linalg.norm(k, 2)
        c = sr@k@sy
        full = np.block([[r, c], [c.conj().T, y]])
        assert np.linalg.eigvalsh(full)[0] >= -TOL
        recovered = np.linalg.solve(sr, c)@np.linalg.inv(sy)
        errors.append(np.max(abs(recovered-k)))
    close('B5_random_compression_identity', defect_errors)
    close('B6_contraction_block_reconstruction', errors)


def gibbs_checks():
    beta = s.symbols('beta', positive=True)
    probability_boundary = 5/(5+2*s.exp(-beta))
    atoms = [(I+Z)/2, (I-Z)/2, (I+X)/2, (I-X)/2, (I+Y)/2, (I-Y)/2, I/2]
    biases = [s.trace(rho*Z) for rho in atoms]
    raw = [s.exp(-beta*bias**2)/7 for bias in biases]
    probabilities = [weight/sum(raw) for weight in raw]
    exact('B8_seven_state_normalization', sum(probabilities)-1)
    exact('B8_seven_state_boundary_mass',
          sum(p for p, bias in zip(probabilities, biases) if bias == 0)-probability_boundary)
    exact('B8_symmetric_prior_odd_mean', sum(p*bias for p, bias in zip(probabilities, biases)))
    exact('B8_boundary_concentration_limit', s.limit(probability_boundary, beta, s.oo)-1)
    assert s.limit(probability_boundary/5, beta, s.oo) == s.Rational(1, 5)
    COUNTEREXAMPLES['B8_boundary_concentration_not_unique_record'] = 'five equal surviving boundary atoms'
    exact('B8_binary_operator_squared', Z**2-I)
    k = s.diag(1, s.Rational(1, 2))
    fail = s.diag(0, s.sqrt(3)/2)
    exact('B8_filter_complete_instrument', k.H*k+fail.H*fail-I)
    post = k*(I/2)*k.H
    post /= s.trace(post)
    exact('B8_conditional_filter_is_nonlinear', post-I/2-s.diag(s.Rational(3, 10), -s.Rational(3, 10)))
    COUNTEREXAMPLES['B8_state_cost_not_operator_square'] = True
    COUNTEREXAMPLES['B8_conditional_normalization_not_affine_channel'] = True


def riemann_checks():
    p, c, theta, phi = s.symbols('p c theta phi', real=True)
    rotation = lambda a: s.Matrix([[s.cos(a), -s.sin(a)], [s.sin(a), s.cos(a)]])
    exact('R1_real_rotation_isometry', rotation(theta).T*rotation(theta)-I)
    exact('R2_relative_rotation', rotation(theta).T*rotation(phi)-rotation(phi-theta))
    x, scale = s.symbols('x scale', positive=True)
    exact('R2_shifted_scale_log', s.expand_log(s.log(scale*x), force=True)-s.log(scale)-s.log(x))
    assert s.log(2) != s.log(s.Rational(3, 2))
    COUNTEREXAMPLES['R2_additive_translation'] = ['log(2)', 'log(3/2)']
    COUNTEREXAMPLES['R2_unshifted_dilation'] = ['log(2)', 'log(3)']
    a, b, shift = s.Rational(1, 10), s.Rational(3, 5), s.Rational(1, 2)
    assert abs(s.floor(a)-s.floor(b)) == 0
    assert abs(s.floor(a+shift)-s.floor(b+shift)) == 1
    COUNTEREXAMPLES['R3_sheet_common_shift_changes_distance'] = [0, 1]
    w = 1/(s.Rational(1, 2)+s.I)
    exact('R5_complex_weight_value', w-s.Rational(2, 5)+4*s.I/5)
    assert s.im(w) != 0
    COUNTEREXAMPLES['R5_tied_projection_complex_diagonal'] = str(s.simplify(w))
    wnum = complex(w)
    upper = .5*wnum*np.exp(-1j*np.log(4))
    lower = 2*wnum*np.exp(1j*np.log(4))
    assert abs(upper.real-lower.real) > .1
    COUNTEREXAMPLES['R5_real_score_not_symmetric'] = [upper.real, lower.real]
    soft = s.Matrix([[s.Rational(1, 2), s.Rational(1, 2)],
                     [s.Rational(1, 3), s.Rational(2, 3)]])
    assert soft != soft.T
    COUNTEREXAMPLES['R5_row_softmax_loses_symmetry'] = True
    bad = s.diag(2, s.Rational(1, 2))
    exact('R6_determinant_one_counterexample', bad.det()-1)
    assert max((bad.H*bad).eigenvals()) == 4
    markov = s.Matrix([[s.Rational(9, 10), s.Rational(1, 10)]]*2)
    assert max((markov.T*markov).eigenvals()) == s.Rational(41, 25)
    COUNTEREXAMPLES['R6_determinant_not_contraction'] = {'det': 1, 'norm': 2}
    COUNTEREXAMPLES['R6_stochastic_not_l2_contraction'] = 'sqrt(41)/5'
    COUNTEREXAMPLES['R6_contraction_residual_not_contraction'] = 'x + identity(x) = 2x'

    positions = np.array([0., 1., 2., 5., 13.])
    gamma, weights = np.array([1., 2., 3.]), np.array([.5, 1/3, 1/6])
    tau = np.log1p(positions)
    gram = np.exp(1j*tau[:, None]*gamma)*np.sqrt(weights)
    kernel = gram@gram.conj().T
    direct = np.sum(weights*np.exp(1j*(tau[:, None, None]-tau[None, :, None])*gamma), axis=2)
    close('R4_Gram_direct_kernel', kernel-direct)
    close('R4_Hermitian_kernel', kernel-kernel.conj().T)
    assert np.linalg.eigvalsh(kernel)[0] >= -TOL
    d = np.diag(np.exp(-tau/2))
    weighted = d@kernel@np.linalg.inv(d)
    metric = np.diag(np.exp(tau))
    close('R5_weighted_inner_product_identity', weighted.conj().T@metric-metric@weighted)
    rng = np.random.default_rng(120911)
    q = rng.normal(size=(5, 3))+1j*rng.normal(size=(5, 3))
    k = rng.normal(size=(5, 3))+1j*rng.normal(size=(5, 3))
    ws = 1/(.5+1j*gamma)
    raw = np.sqrt((1+positions[None, :])/(1+positions[:, None]))*np.sum(
        ws*np.exp(-1j*(tau[:, None, None]-tau[None, :, None])*gamma)
        *q[:, None, :]*k.conj()[None, :, :], axis=2)
    qtilde = np.exp(-tau[:, None]/2-1j*tau[:, None]*gamma)*q
    ktilde = np.exp(tau[:, None]/2-1j*tau[:, None]*gamma)*k
    close('R7_finite_MRA_factorization', raw-(qtilde*ws)@ktilde.conj().T)


def pre_equality_checks():
    beta, a, x = s.symbols('beta a x', positive=True)
    # Integrate a shrinking continuous well, independently of the bound in E2.
    well_mass = s.integrate(s.exp(-beta*x/a), (x, 0, a))
    outer_mass = s.exp(-beta)*(1-a)
    exact('E2_continuous_well_partition',
          well_mass+outer_mass-a*(1-s.exp(-beta))/beta-(1-a)*s.exp(-beta))
    ratios = []
    for n in [2, 4, 8, 12]:
        mass = float((well_mass/(well_mass+outer_mass)).subs({a: s.exp(-n*n), beta: n}))
        bound = float(s.exp(-n*n+n)/(n*(1-s.exp(-n*n))))
        assert 0 < mass <= bound
        ratios.append(mass)
    assert all(left > right for left, right in zip(ratios, ratios[1:]))
    assert ratios[-1] < 1e-50
    COUNTEREXAMPLES['E2_Gamma_unique_minimum_no_Gibbs_concentration'] = ratios
    retained = 1/(1+s.exp(-beta/2))
    exact('E2_exponential_recovery_still_concentrates', s.limit(retained, beta, s.oo)-1)
    lost = 1/(1+s.exp(beta**2-beta))
    exact('E2_positive_prior_can_lose_minimum', s.limit(lost, beta, s.oo))
    COUNTEREXAMPLES['E2_subexponential_condition_not_necessary'] = 'prior rate 1/2; gap 1'
    exact('E1_zero_partition_infinite_cost', s.exp(-s.oo))
    COUNTEREXAMPLES['E1_lsc_full_support_not_recovery'] = 'uniform law remains uniform'

    amplitude, t = s.symbols('amplitude t', real=True)
    path = 1+2*t+amplitude*s.sin(2*s.pi*t)
    kinetic = s.integrate(s.diff(path, t)**2/2, (t, 0, 1))
    exact('E3_kinetic_orthogonal_decomposition', kinetic-2-s.pi**2*amplitude**2)
    normal = s.exp(-t*t/2)/s.sqrt(2*s.pi)
    second = s.integrate(t*t*normal, (t, -s.oo, s.oo))
    fourth = s.integrate(t**4*normal, (t, -s.oo, s.oo))
    count = s.symbols('count', positive=True, integer=True)
    exact('E3_dyadic_quadratic_variation_mean', count*second/count-1)
    exact('E3_dyadic_quadratic_variation_variance', count*(fourth-second**2)/count**2-2/count)
    COUNTEREXAMPLES['E3_Brownian_kinetic_partition_zero'] = 'positive quadratic variation excludes W1p'
    exact('E4_global_lower_error_counterexample', (2*count)**2-(2*count)**2/2-2*count**2)
    COUNTEREXAMPLES['E4_local_uniform_not_global_lower_error'] = 'unbounded error x^2/2 outside mesh radius'

    def normalized_row(matrix):
        return s.diag(*[1/sum(matrix.row(i)) for i in range(matrix.rows)])*matrix
    k = s.Matrix([[1, 1]])
    ell = s.diag(1, 2)
    exact('E5_row_normalized_composition_counterexample',
          normalized_row(k*ell)-normalized_row(k)*normalized_row(ell)-s.Matrix([[-s.Rational(1, 6), s.Rational(1, 6)]]))
    mu = s.Matrix([[s.Rational(1, 3), s.Rational(2, 3)]])
    k = s.Matrix([[1, 2], [3, 1]])
    ell = s.Matrix([[2, 0], [1, 3]])
    step1 = mu*k
    step1 /= sum(step1)
    step2 = step1*ell
    step2 /= sum(step2)
    direct = mu*k*ell
    direct /= sum(direct)
    exact('E5_state_normalization_composes', step2-direct)
    COUNTEREXAMPLES['E5_row_normalization_not_functor'] = True
    COUNTEREXAMPLES['E5_tropicalization_can_be_negative'] = str(-s.log(2))
    assert -s.log(2) < 0

    q = s.symbols('q', positive=True)
    prior = s.Matrix([[s.Rational(1, 8), s.Rational(3, 8)],
                      [s.Rational(1, 4), s.Rational(1, 4)]])
    raw = prior.multiply_elementwise(s.Matrix([[1, q], [1, q*q]]))
    posterior = raw/sum(raw)
    limit = posterior.subs(q, 0)
    exact('E6_joint_multiple_minima_limit', limit-s.Matrix([[s.Rational(1, 3), 0], [s.Rational(2, 3), 0]]))
    exact('E6_raw_residual_vanishes', (posterior[0, 1]+posterior[1, 1]).subs(q, 0))
    COUNTEREXAMPLES['E6_nonunique_joint_minimum_not_Dirac'] = True
    COUNTEREXAMPLES['E6_weak_limit_discontinuous_readout'] = {'integrals': 0, 'limit_integral': 1}

    squared = [s.Rational(3, 4), s.Rational(1, 4)]
    alternative = squared[0]**2/sum(p*p for p in squared)
    fine = [squared[0]/3]*3+[squared[1]]
    coarse = sum(p*p for p in fine[:3])/sum(p*p for p in fine)
    exact('E7_refinement_Born_consistency', sum(fine[:3])-squared[0])
    exact('E7_refinement_power_four_violation', alternative-coarse-s.Rational(3, 20))
    COUNTEREXAMPLES['E7_B4_needed_beyond_other_axioms'] = ['9/10', '3/4']
    p0, p1 = (I+Z)/2, (I-Z)/2
    ms = [p0, p1/2, s.sqrt(3)*p1/2]
    exact('E8_instrument_complete', sum((m.H*m for m in ms), s.zeros(2))-I)
    rho = s.Matrix([[s.Rational(2, 3), s.Rational(1, 6)], [s.Rational(1, 6), s.Rational(1, 3)]])
    success = sum((m*rho*m.H for m in ms[:2]), s.zeros(2))
    exact('E8_Gibbs_success_probability', s.trace(success)-s.Rational(3, 4))
    exact('E8_Gibbs_conditional_law', success/s.trace(success)-s.diag(s.Rational(8, 9), s.Rational(1, 9)))
    exact('E8_commuting_energy_conservation', sum((m.H*Z*m for m in ms), s.zeros(2))-Z)
    coherent = ms[0]+ms[1]
    difference = coherent*rho*coherent.H-success
    exact('E8_same_success_law_different_coherence', difference-X/12)
    COUNTEREXAMPLES['E8_probability_law_does_not_fix_instrument'] = True


def main():
    folder = Path(__file__).resolve().parent
    paper = folder.parent/'paper'
    boundary = next(paper.glob('01_*/12_*.md'))
    riemann = next(paper.glob('*/8_*/math_claims_audit.md'))
    preeq = next(paper.glob('*/9_*/10_*.md'))
    for path, separator, expected in [(boundary, '## 12.2', B_PREREG), (riemann, '## 2.', R_PREREG)]:
        prefix = path.read_text(encoding='utf-8').split(separator)[0]
        assert hashlib.sha256(prefix.encode()).hexdigest() == expected, path
    prefix = preeq.read_text(encoding='utf-8').split('## 2.')[0].rstrip()+'\n'
    assert hashlib.sha256(prefix.encode()).hexdigest() == E_PREREG, preeq
    boundary_checks()
    observation_checks()
    gibbs_checks()
    riemann_checks()
    pre_equality_checks()
    result = {
        'schema_version': 2, 'candidate': ['CE-MATH-B1', 'CE-MATH-R1', 'CE-MATH-E1'],
        'scientific_success': False, 'full_joint_rmse': None,
        'new_observational_inputs_or_fits': False,
        'general_proof_location': [str(p.relative_to(folder.parent)) for p in [boundary, riemann, preeq]],
        'proof_scope': 'conditional quantum boundary, Gibbs/pathspace, kernel/Born and logarithmic theorems in cited manuscripts; finite checks are witnesses only',
        'proof_assistant_checked': False,
        'preregistration_sha256': {'B1': B_PREREG, 'R1': R_PREREG, 'E1': E_PREREG},
        'source_hashes': {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'manuscript_sha256': {str(p.relative_to(folder.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [boundary, riemann, preeq]},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'sympy': s.__version__},
        'symbolic_identities': EXACT, 'counterexamples': COUNTEREXAMPLES,
        'numerical_errors': ERRORS, 'maximum_numerical_error': max(ERRORS.values()),
        'witness_checks_passed': True,
        'unproved_extensions': ['proof-assistant formalization', 'Riemann hypothesis or Hilbert-Polya spectral identification',
                               'physical selection, actual CE path action/prior and full joint RMSE']}
    (folder/'ce_mathematical_boundary_audit.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'exact_identity_count': len(EXACT), 'counterexample_count': len(COUNTEREXAMPLES),
                      'numerical_error_max': result['maximum_numerical_error'],
                      'witness_checks_passed': True}, indent=2))


if __name__ == '__main__':
    main()
