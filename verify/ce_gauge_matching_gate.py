"""CE-LR3: no-refit, optimistic threshold envelope for two gauge metrics.

Same supplied Weyl/scalar content and fixed PDG summary for both candidates.
Exact nonnegative certificates plus independent LP checks. This is not a
joint likelihood or an all-orders exclusion of an unspecified UV theory.
"""
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.optimize import linprog
import sympy as sy

from ce_matter_representation_anomalies import matrices, matrix_anomalies, sums
from ce_record_weak_symmetry import old_model, extended_model


TOL = 1e-9
R = sy.Rational
BSM = sy.Matrix([-7, -R(19, 6), R(41, 6)])
DELTAS = sy.Matrix([[R(1, 6), R(1, 6), 1, 0, 0],
                   [0, 0, 0, R(1, 6), R(1, 6)],
                   [0, 0, 0, R(1, 6), R(1, 6)]])
CANDIDATES = [('CE-LR3-R18', sy.Matrix([4, 1, 1])),
              ('CE-LR3-R45', sy.Matrix([1, 1, R(5, 3)]))]


def error(x):
    return float(np.max(np.abs(x)))


def representation_checks():
    charges = (1, -4, 2, -3, 6)
    assert all(x == 0 for x in sums(charges).values())
    anomaly = matrix_anomalies(charges)
    assert max(anomaly.values()) < TOL
    one = matrices(charges)
    representatives = [one[4], one[1], one[0]/6]
    g45 = [np.kron(np.eye(3), g) for g in representatives]
    trace45 = np.array([np.trace(g@g).real for g in g45])
    exact45 = sy.Matrix([6, 6, 10])
    assert error(trace45-np.array(exact45, float).ravel()) < TOL
    initial, singlets, outside, mass, record, colors = old_model()
    _, gen18 = extended_model(initial, singlets, outside, mass, record, colors,
                              np.ones(2, complex)/np.sqrt(2))
    trace18 = np.array([np.trace(gen18[j]@gen18[j]).real for j in (0, 8, 11)])
    assert error(trace18-[4, 1, 1]) < TOL
    normalized = [trace18/trace18[1], trace45/trace45[1]]
    for actual, (_, k) in zip(normalized, CANDIDATES):
        assert error(actual-np.array(k, float).ravel()) < TOL
    # Complete matter has only the Z2 subgroup of the Weyl-only Z6 kernel.
    # Tuples: integer y=6Y, SU(2) doublet parity, color triality.
    weyl = [(1, 1, 1), (-4, 0, -1), (2, 0, -1), (-3, 1, 0), (6, 0, 0)]
    scalars = [(0, 0, 1), (0, 0, -1), (0, 0, 0), (3, 1, 0), (3, 1, 0), (3, 1, 0)]

    def central_kernel(reps):
        return [k for k in range(6) if all((Fraction(k*y, 6)+Fraction((k % 2)*d, 2)
                                          +Fraction((k % 3)*t, 3)).denominator == 1
                                         for y, d, t in reps)]

    assert central_kernel(weyl) == list(range(6))
    assert central_kernel(weyl+scalars) == [0, 3]
    assert BSM+sum((DELTAS[:, j] for j in range(5)), sy.zeros(3, 1)) == sy.Matrix([-R(17, 3), -R(17, 6), R(43, 6)])
    return dict(Weyl_internal_components=45, Weyl_trace_indices=trace45.tolist(),
                scalar_trace_indices=trace18.tolist(), normalized_metrics=[x.tolist() for x in normalized],
                local_chiral_anomalies=anomaly, SU2_doublet_count=12,
                Weyl_only_center_kernel_k=central_kernel(weyl),
                full_matter_center_kernel_k=central_kernel(weyl+scalars),
                simple_SU5_matter_completion=False,
                full_active_beta_exact=['-17/3', '-17/6', '43/6'])


def exact_certificate(k):
    # z=(a,l,c3,cbar,c8,dminus,dplus); x=Bz.
    B = sy.Matrix.hstack(k, BSM, DELTAS)
    y2, yy, lam = sy.symbols('y2 yy lam')
    remainder = B[0, :]-y2*B[1, :]-yy*B[2, :]
    target = sy.Matrix([[0, 2*lam, R(1, 6), R(1, 6), 1, -lam, -lam]])
    solution = sy.solve(list(remainder-target), [y2, yy, lam], dict=True)
    assert len(solution) == 1
    solution = solution[0]
    certificate = target.subs(solution)
    assert remainder.subs(solution) == certificate
    assert solution[lam] > 0
    # Two electroweak inputs fix the saturating relaxed witness (all colored t=0, d=l).
    x2, xy = sy.symbols('x2 xy', real=True)
    ew_effective = BSM+DELTAS[:, 3]+DELTAS[:, 4]
    witness_al = sy.Matrix([[k[1], ew_effective[1]], [k[2], ew_effective[2]]]).inv()*sy.Matrix([x2, xy])
    bound = sy.factor(solution[y2]*x2+solution[yy]*xy)
    witness_value = sy.factor(k[0]*witness_al[0]+ew_effective[0]*witness_al[1])
    assert sy.simplify(witness_value-bound) == 0
    return B, dict(y2=solution[y2], yY=solution[yy], lambda_cap=solution[lam],
                   bound=bound, witness_a=witness_al[0], witness_l=witness_al[1])


def envelope_lp(B_exact, certificate, A, s):
    B = np.array(B_exact, float)
    c, eq = B[0], B[1:]
    rhs = np.array([A*s, A*(1-s)])
    inequality = np.zeros((5, 7))
    for j in range(5):
        inequality[j, 1], inequality[j, j+2] = -1, 1
    lp = linprog(c, A_ub=inequality, b_ub=np.zeros(5), A_eq=eq, b_eq=rhs,
                 bounds=[(0, None)]*7, method='highs')
    assert lp.success, lp.message
    weights = np.array([float(certificate['y2']), float(certificate['yY'])])
    analytic = float(weights@rhs)
    assert analytic > 0
    errors = dict(primal_equalities=error(eq@lp.x-rhs),
                  primal_inequalities=max(0., float(np.max(inequality@lp.x))),
                  nonnegative_variables=max(0., -float(lp.x.min())),
                  analytic_vs_LP=abs(float(lp.fun)-analytic),
                  dual_stationarity=error(c-eq.T@lp.eqlin.marginals
                                          -inequality.T@lp.ineqlin.marginals-lp.lower.marginals),
                  dual_electroweak_weights=error(lp.eqlin.marginals-weights),
                  primal_dual_gap=abs(float(lp.fun-rhs@lp.eqlin.marginals)))
    assert max(errors.values()) < TOL
    # This extremizer proves a bound in a larger set; it is not a fitted model.
    saturating = np.zeros(7)
    saturating[0], saturating[1] = lp.x[0], lp.x[1]
    saturating[-2:] = lp.x[1]
    assert error(eq@saturating-rhs) < TOL and abs(c@saturating-analytic) < TOL
    assert lp.x[0] > 0 and lp.x[1] > 0
    return dict(alpha_em_inverse=A, sin_squared_theta_W=s,
                minimum_inverse_alpha_s=analytic, maximum_alpha_s=1/analytic,
                errors=errors, positive_saturating_relaxation_witness_exists=True,
                witness_is_physical_common_mass_spectrum=False)


def evaluate_candidate(name, k, inputs):
    B, certificate = exact_certificate(k)
    obs = inputs['observables']
    A, dA = obs['alpha_em_inverse']['value'], obs['alpha_em_inverse']['quoted_error']
    s, ds = obs['sin_squared_theta_W']['value'], obs['sin_squared_theta_W']['quoted_error']
    strong, dstrong = obs['alpha_s']['value'], obs['alpha_s']['quoted_error']
    factor = inputs['error_box_multiplier']
    central = envelope_lp(B, certificate, A, s)
    corners = [envelope_lp(B, certificate, A+i*factor*dA, s+j*factor*ds)
               for i, j in itertools.product((-1, 1), repeat=2)]
    max_alpha = max(row['maximum_alpha_s'] for row in corners)
    observed_low = strong-factor*dstrong
    separated = max_alpha < observed_low
    # Formula A*(yY+(y2-yY)*s) is increasing in A and s on this entire box.
    weights = [float(certificate[x]) for x in ('y2', 'yY')]
    assert weights[0]-weights[1] > 0
    assert min(row['minimum_inverse_alpha_s']/row['alpha_em_inverse'] for row in corners) > 0
    assert separated
    return dict(candidate=name, kinetic_metric_exact=list(map(str, k)),
                certificate_exact={key: str(value) for key, value in certificate.items()},
                central=central, error_box_corners=corners,
                whole_box_bound_justified_by_monotonicity=True,
                maximum_alpha_s_over_box=max_alpha, observed_alpha_s_box_lower=observed_low,
                disjoint_gap=observed_low-max_alpha,
                required_extra_delta_central_upper_bound=1/strong-central['minimum_inverse_alpha_s'],
                required_extra_delta_optimistic_box_upper_bound=1/observed_low-1/max_alpha,
                extra_delta_was_fitted_or_added=False,
                leading_log_zero_finite_status='REJECTED_BY_THRESHOLD_ENVELOPE',
                all_orders_status='NOT_EVALUATED',
                nonnegative_certificate='x3-y2*x2-yY*xY=c3/6+cbar/6+c8+lambda_cap*((l-dminus)+(l-dplus))',
                nuisance_variables_eliminated=7, optimized_parameters_adopted=False)


def run():
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).with_name('ce_gauge_matching_inputs.json')
    inputs = json.loads(source.read_text(encoding='utf-8'))
    assert inputs['covariance'] is None and inputs['blinded_holdout'] is False
    representation = representation_checks()
    rows = [evaluate_candidate(name, k, inputs) for name, k in CANDIDATES]
    chapter = next((root/'paper').glob('06_*/66_*.md'))
    prereg = chapter.read_text(encoding='utf-8').split('## 66.2')[0]
    sources = [Path(__file__), source]+[root/'verify'/name for name in (
        'ce_matter_representation_anomalies.py', 'ce_common_isotropic_frame.py',
        'ce_record_weak_symmetry.py', 'ce_color_covariant_record.py', 'ce_isometric_color_frame.py')]
    return dict(candidate_family='CE-LR3 trace and threshold gate', input_config_id=inputs['config_id'],
                representations=representation, candidates=rows, fitted_parameters=0,
                covariance_available=False, blinded_holdout=False,
                full_joint_rmse=None, scientific_success=False, narrow_checks_passed=True,
                limits=['one-loop leading-log matching; finite matching set to zero',
                        'SM Weyl content, scalar content and kinetic trace choices supplied',
                        'thresholds independently relaxed; a feasible point would not prove the common-mass model',
                        'quoted marginal error box is not a joint confidence region',
                        'no all-orders or ultraviolet-theory exclusion',
                        'no stable physical readout or full quantum gravity construction'],
                preregistration_section_sha256=hashlib.sha256(prereg.encode('utf-8')).hexdigest(),
                source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources},
                environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                                 sympy=sy.__version__, platform=platform.platform()))


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(representations=result['representations'],
                         candidates=[dict(candidate=r['candidate'], certificate=r['certificate_exact'],
                                          central_max_alpha_s=r['central']['maximum_alpha_s'],
                                          error_box_max_alpha_s=r['maximum_alpha_s_over_box'],
                                          observed_lower=r['observed_alpha_s_box_lower'],
                                          status=r['leading_log_zero_finite_status']) for r in result['candidates']],
                         scientific_success=False), indent=2))
