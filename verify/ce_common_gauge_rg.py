"""Chapter 65: supplied LR2 gauge/Higgs content, one-loop RG and portals.

No observational fit. R0 and P0 are falsified; R1/P1 are limited matching
constructions, not a complete beta system or physical unification result.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import sympy as sy

from ce_record_weak_symmetry import PAULI, old_model, extended_model


TOL = 1e-10
R = sy.Rational
I_TRACE = sy.Matrix([4, 1, 1])
CA = sy.Matrix([3, 2, 0])
MU0 = 100.
GSTAR = np.sqrt(2.)
LAMBDA_H = .25
TIMES = (-1., -.5, 0., .5, 1.)


def error(x):
    return float(np.max(np.abs(x)))


def representation_audit():
    initial, singlets, outside, mass, record, colors = old_model()
    eta = np.ones(2, complex)/np.sqrt(2)
    _, gen = extended_model(initial, singlets, outside, mass, record, colors, eta)
    direct = np.array([np.trace(gen[0]@gen[0]).real,
                       np.trace(gen[8]@gen[8]).real+.5,
                       np.trace(gen[11]@gen[11]).real+.5])
    # T(R) = C2(R) dim(R) / dim(G), independently of the 18x18 matrices.
    tf3, ta3, tf2 = R(4, 3)*3/8, R(3)*8/8, R(3, 4)*2/3
    scalar_sum = sy.Matrix([2*tf3+ta3, 3*tf2, 3*2*R(1, 2)**2])
    assert scalar_sum == sy.Matrix([4, R(3, 2), R(3, 2)])
    assert error(direct-np.array(scalar_sum, float).ravel()) < TOL
    b = -R(11, 3)*CA+scalar_sum/3
    assert b == sy.Matrix([-R(29, 3), -R(41, 6), R(1, 2)])

    # Separate normalization control: one standard chiral generation plus one Higgs.
    # It does not enter the LR2 candidate field list.
    weyl_fields = [
        ('Q', 3, 2, R(1, 6), R(1, 2), R(1, 2)),
        ('uc', 3, 1, -R(2, 3), R(1, 2), R(0)),
        ('dc', 3, 1, R(1, 3), R(1, 2), R(0)),
        ('ell', 1, 2, -R(1, 2), R(0), R(1, 2)),
        ('ec', 1, 1, R(1), R(0), R(0)),
    ]
    one_generation = sum((sy.Matrix([d2*t3, d3*t2, d3*d2*y*y])
                          for _, d3, d2, y, t3, t2 in weyl_fields), sy.zeros(3, 1))
    sm_b = -R(11, 3)*CA+R(2, 3)*3*one_generation+sy.Matrix([0, R(1, 2), R(1, 2)])/3
    assert sm_b == sy.Matrix([-7, -R(19, 6), R(41, 6)])
    assert R(3, 5)*sm_b[2] == R(41, 10)
    ratio = sy.Matrix([b[i]/I_TRACE[i] for i in range(3)])
    assert len(set(ratio)) == 3
    plane_normal = sy.Matrix.hstack(I_TRACE, b).T.nullspace()[0]
    plane_normal *= sy.ilcm(*(x.q for x in plane_normal))
    if plane_normal[0] < 0:
        plane_normal *= -1
    assert list(plane_normal) == [22, -35, -53]
    assert plane_normal.dot(I_TRACE) == plane_normal.dot(b) == 0
    return (dict(scalar_trace_indices_direct=direct.tolist(),
                 scalar_trace_indices_exact=list(map(str, scalar_sum)),
                 representation_trace_error=error(direct-np.array(scalar_sum, float).ravel()),
                 beta_coefficients_exact=list(map(str, b)),
                 beta_over_trace_indices_exact=list(map(str, ratio)),
                 single_trace_status='REJECTED_AS_ALL_SCALE_RELATION',
                 standard_model_control=dict(beta_coefficients_exact=list(map(str, sm_b)),
                                             GUT_normalized_b1=str(R(3, 5)*sm_b[2]),
                                             fields_included_in_candidate=False),
                 maximum_supplied_scalar_tree_mass=float(np.linalg.eigvalsh(mass).max()),
                 matching_plane_normal=list(map(int, plane_normal))),
            np.array(b, float).ravel(), gen)


def gauge_flow(b, plane_normal, maximum_scalar_mass):
    indices = np.array(I_TRACE, float).ravel()
    inv0 = 2*indices/GSTAR**2
    g0 = 1/np.sqrt(inv0)
    slope = -b/(8*np.pi**2)
    maximum_tree_mass = max(maximum_scalar_mass, np.sqrt(2*LAMBDA_H), .5, 1/np.sqrt(2))
    assert MU0*np.exp(min(TIMES)) > maximum_tree_mass
    middle = 1/3
    transported = inv0+slope*middle
    rows = []
    for time in TIMES:
        inverse = inv0+slope*time
        assert inverse.min() > 0
        direct = 1/np.sqrt(inverse)
        if time == 0:
            integrated = g0.copy()
        else:
            sol = solve_ivp(lambda _, g: b*g**3/(16*np.pi**2), (0, time), g0,
                            method='DOP853', rtol=1e-12, atol=1e-14)
            assert sol.success
            integrated = sol.y[:, -1]
        moved_origin = transported+slope*(time-middle)
        errors = dict(inverse_vs_g_ODE=error(direct-integrated),
                      transported_matching=error(moved_origin-inverse),
                      fixed_boundary_sum_rule=abs(float(np.dot(plane_normal, inverse))))
        assert max(errors.values()) < TOL
        reduced = inverse/indices
        rows.append(dict(log_mu_ratio=time, mu=MU0*np.exp(time), couplings=direct.tolist(),
                         inverse_couplings_squared=inverse.tolist(),
                         single_trace_spread=float(np.ptp(reduced)),
                         sin_squared_theta_W=float(direct[2]**2/(direct[1]**2+direct[2]**2)),
                         errors=errors))
    assert rows[-1]['single_trace_spread'] > .08
    return dict(status='CONDITIONAL_ONE_LOOP_BOUNDARY_FAMILY',
                reference_scale=MU0, common_boundary_gstar=GSTAR,
                initial_couplings=g0.tolist(), inverse_coupling_slopes=slope.tolist(),
                transported_reference_log_ratio=middle,
                transported_coefficients=transported.tolist(),
                transported_single_trace_spread=float(np.ptp(transported/indices)),
                boundary_parameters=['gstar(mu0)', 'mu0 in the fixed mass unit'],
                beta_order=1, no_thresholds_crossed_in_benchmark=True,
                maximum_supplied_tree_mass=maximum_tree_mass,
                all_supplied_tree_masses_below_benchmark_scales=True,
                physical_GeV_unit_assigned=False, rows=rows)


def symbolic_portal():
    h, x, y = sy.symbols('h x y', real=True)
    g2, gy = sy.symbols('g2 gy', positive=True)
    pauli = [sy.Matrix([[0, 1], [1, 0]]), sy.Matrix([[0, -sy.I], [sy.I, 0]]), sy.diag(1, -1)]
    t = [g2*p/2 for p in pauli]+[gy*sy.eye(2)/2]

    def matrix(fields):
        return sy.Matrix(4, 4, lambda a, b: sy.simplify(sum(
            2*sy.re(((t[a]*f).conjugate().T*t[b]*f)[0]) for f in fields)))

    higgs, record = sy.Matrix([h, 0]), sy.Matrix([x, y])
    mh, mr, total = matrix([higgs]), matrix([record]), matrix([higgs, record])
    mixed_trace = sy.expand(sy.trace(total*total-mh*mh-mr*mr))
    a = (3*g2**4+gy**4)/2-g2**2*gy**2
    c = 2*g2**2*gy**2
    assert sy.simplify(mixed_trace-a*h*h*(x*x+y*y)-c*h*h*x*x) == 0
    beta_portal = [sy.factor(R(3, 2)*a), sy.factor(R(3, 2)*c)]  # 16 pi^2 beta
    beta_higgs = sy.factor(R(3, 2)*sy.trace(mh*mh)/h**4)
    assert sy.simplify(beta_higgs-R(3, 8)*(3*g2**4+2*g2**2*gy**2+gy**4)) == 0
    at_boundary = [value.subs({g2: 1, gy: 1}) for value in beta_portal]
    assert at_boundary == [R(3, 2), R(3)]
    # An exact scale identity for the mixed explicit loop logarithm.
    log_scale = sy.symbols('t', real=True)
    quartic_running = (beta_portal[0]*h*h*(x*x+y*y)+beta_portal[1]*h*h*x*x)*log_scale/(16*sy.pi**2)
    explicit_log_change = -3*mixed_trace*log_scale/(32*sy.pi**2)
    assert sy.simplify(quartic_running+explicit_log_change) == 0
    return dict(status_zero_portal='REJECTED_AS_ALL_SCALE_CONSTRAINT',
                mixed_trace_polynomial=str(sy.factor(mixed_trace)),
                beta_times_16pi2=dict(norm_product=str(beta_portal[0]), overlap=str(beta_portal[1]),
                                    Higgs_self_gauge=str(beta_higgs)),
                boundary_beta_times_16pi2=list(map(str, at_boundary)),
                mixed_log_cancellation_exact=True,
                full_scalar_beta_system=False), np.array(at_boundary, float).ravel()/(16*np.pi**2)


def mass_matrix(fields, generators):
    # fields and generator lists may carry different representations of the same algebra.
    count = len(generators[0])
    out = np.zeros((count, count))
    for phi, gen in zip(fields, generators):
        vectors = np.array([g@phi for g in gen])
        out += 2*np.real(vectors.conj()@vectors.T)
    return out


def loop_potential(mass_squared, scale):
    values = np.linalg.eigvalsh(mass_squared)
    assert values.min() > -TOL
    values = values[values > 1e-12]
    return float(3*np.sum(values**2*(np.log(values/scale**2)-5/6))/(64*np.pi**2))


def numerical_portal(beta, gen18):
    rng = np.random.default_rng(6501)
    weak = [p/2 for p in PAULI]+[np.eye(2)/2]
    h_gen_full = [np.zeros((2, 2)) for _ in range(8)]+weak
    gen_full = [g*c for g, c in zip(gen18, [.5]*8+[1.]*4)]
    max_formula = max_covariance = max_scale = 0.
    min_vector_mass_squared = np.inf
    for _ in range(12):
        h, chi = (rng.normal(size=2)+1j*rng.normal(size=2) for _ in range(2))
        mh, mr = mass_matrix([h], [weak]), mass_matrix([chi], [weak])
        total = mh+mr
        uv = float(np.vdot(h, h).real*np.vdot(chi, chi).real)
        overlap = float(abs(np.vdot(h, chi))**2)
        mixed = float(np.trace(total@total-mh@mh-mr@mr))
        max_formula = max(max_formula, abs(mixed-uv-2*overlap))
        for time in TIMES:
            mu = MU0*np.exp(time)
            loop_now = loop_potential(total, mu)-loop_potential(mh, mu)-loop_potential(mr, mu)
            loop_ref = loop_potential(total, MU0)-loop_potential(mh, MU0)-loop_potential(mr, MU0)
            tree_running = time*(beta[0]*uv+beta[1]*overlap)
            max_scale = max(max_scale, abs(tree_running+loop_now-loop_ref))
        # Full 18+2 scalar source: a simultaneous gauge rotation preserves Tr M_V^4.
        phi = rng.normal(size=18)+1j*rng.normal(size=18)
        theta = rng.normal(size=12)
        u18 = expm(1j*sum(a*g for a, g in zip(theta, gen18)))
        u2 = expm(1j*sum(a*g for a, g in zip(theta, h_gen_full)))
        full = mass_matrix([phi, h], [gen_full, h_gen_full])
        rotated = mass_matrix([u18@phi, u2@h], [gen_full, h_gen_full])
        max_covariance = max(max_covariance, abs(float(np.trace(full@full-rotated@rotated))))
        min_vector_mass_squared = min(min_vector_mass_squared, float(np.linalg.eigvalsh(full).min()))
    assert max(max_formula, max_covariance, max_scale) < TOL
    assert min_vector_mass_squared > -TOL
    rows = [dict(log_mu_ratio=t, leading_log_norm_product=float(beta[0]*t),
                 leading_log_overlap=float(beta[1]*t)) for t in TIMES]
    return dict(status='CONDITIONAL_GAUGE_SOURCE_LEADING_LOG_COMPLETION', seed=6501,
                random_backgrounds=12, full_source_gauge_invariance_error=max_covariance,
                portal_formula_error=max_formula, leading_log_scale_cancellation_error=max_scale,
                minimum_background_vector_mass_squared=min_vector_mass_squared,
                fixed_Higgs_quartic_at_matching=LAMBDA_H, portal_at_matching=[0., 0.],
                rows=rows, finite_one_loop_matching_computed=False,
                global_effective_potential_stability_proved=False)


def run():
    representation, b, gen = representation_audit()
    flow = gauge_flow(b, representation['matching_plane_normal'],
                      representation['maximum_supplied_scalar_tree_mass'])
    symbolic, beta = symbolic_portal()
    portal = numerical_portal(beta, gen)
    root = Path(__file__).resolve().parents[1]
    doc = next((root/'paper').glob('06_*/65_*.md'))
    prereg = doc.read_text(encoding='utf-8').split('## 65.2')[0]
    sources = [Path(__file__), root/'verify/ce_record_weak_symmetry.py',
               root/'verify/ce_color_covariant_record.py', root/'verify/ce_isometric_color_frame.py']
    return dict(candidate='CE-LR2-R0/R1 and P0/P1', representation=representation,
                gauge_flow=flow, symbolic_portal=symbolic, numerical_portal=portal,
                narrow_checks_passed=True, fitted_parameters=0, full_joint_rmse=None,
                scientific_success=False,
                limitations=['standard 4D perturbative loop formulas and representations supplied',
                             'gauge running only at one loop; quartics only gauge g^4 source/leading log',
                             'no full mass, quartic, threshold or pole matching system',
                             'no data, measured couplings, covariance or holdout evaluation',
                             'matching scale and boundary parameters chosen, not derived',
                             'renormalization-scale flow is not physical time or a quantum channel',
                             'no interacting gauge-gravity quantum construction or stable actual record'],
                environment=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                                 sympy=sy.__version__, platform=platform.platform()),
                preregistration_section_sha256=hashlib.sha256(prereg.encode('utf-8')).hexdigest(),
                source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources})


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(beta=result['representation']['beta_coefficients_exact'],
                         ratios=result['representation']['beta_over_trace_indices_exact'],
                         inverse_coupling_sum_rule=result['representation']['matching_plane_normal'],
                         high_scale=result['gauge_flow']['rows'][-1],
                         portal=result['symbolic_portal']['boundary_beta_times_16pi2'],
                         portal_errors={k: v for k, v in result['numerical_portal'].items() if 'error' in k},
                         checks=result['narrow_checks_passed'], scientific_success=False), indent=2))
