"""Chapter 76: source stability and a conditional spatial scattering operator.

The Gaussian source, its preparation, and the static approximation are supplied.
No full instrument, full backreaction, or joint observational success.
"""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.special import roots_legendre, spherical_jn, spherical_yn, eval_legendre

from ce_singlet_record_observability import build_model
from ce_supersymmetric_record import f_terms


PREREG = '1ec530b8430650cb5d0a754dd08128e543ee8e56e305e2fea2616aa9cf7230b1'
M0, M, KAPPA = 1e-4, 1., .05
MASSES = M0*np.array([5+np.pi/2, 5-np.pi/2])
EPSILON = .1*MASSES[1]
XIS = [.03, .1]
A_COUPLING, B_COUPLING, C_COUPLING = 1/(2*np.sqrt(15)), 3/(2*np.sqrt(15)), 1/np.sqrt(15)
FINAL_TAU = 20.


def maximum(x):
    return float(np.max(np.abs(x)))


def field_values(z, s, mass):
    fz = -M*z+C_COUPLING*z*z/2+A_COUPLING*s*s
    fs = (mass+2*A_COUPLING*z)*s
    return fz, fs


def initial(mass, xi):
    amplitude = xi*np.sqrt(mass/KAPPA)
    scale = A_COUPLING*amplitude*amplitude/M
    z = A_COUPLING*amplitude*amplitude*(M/(M*M-4*mass*mass)-2*mass/(M*M))
    dz = -2j*mass*A_COUPLING*M*amplitude*amplitude/(M*M-4*mass*mass)
    return amplitude, scale, np.array([z/scale, 1., dz/(mass*scale), -1j], complex)


def action_checks():
    _, _, reps, matrix, cubic, vacuum = build_model(M0)
    _, w0 = f_terms(vacuum, matrix, cubic)
    assert abs(cubic[11, 11, 11]-C_COUPLING) < 1e-8
    assert abs(cubic[11, 35, 35]/2-A_COUPLING) < 1e-8
    assert abs(cubic[11, 75, 80]-B_COUPLING) < 1e-8
    errors, rows = [], []
    rng = np.random.default_rng(7601)
    for species, mass in enumerate(MASSES):
        basis = np.zeros((82, 2), complex)
        basis[11, 0] = 1
        basis[35, 1], basis[59, 1] = 1/np.sqrt(2), (1 if species == 0 else -1)/np.sqrt(2)
        errors.append(maximum(basis.T@w0@basis-np.diag([-M, mass])))
        for xi in XIS:
            amplitude, scale, y = initial(mass, xi)
            z, s = y[0]*scale, amplitude
            delta = basis@np.array([z, s])
            # Expansion about the known vacuum avoids cancellation of large
            # vacuum contributions against the tiny source curvature.
            def f_at(offset):
                return w0@offset+.5*np.einsum('ijk,j,k->i', cubic, offset, offset, optimize=True)
            f = f_at(delta)
            ff = np.array(field_values(z, s, mass))
            errors.append(maximum(f-basis@ff)/(mass*amplitude))
            w = w0+np.einsum('ijk,k->ij', cubic, delta, optimize=True)
            grad = w.conj().T@f
            reduced = np.array([(-M+C_COUPLING*z).conjugate()*ff[0]+2*A_COUPLING*s.conjugate()*ff[1],
                                2*A_COUPLING*s.conjugate()*ff[0]+(mass+2*A_COUPLING*z).conjugate()*ff[1]])
            errors.append(maximum(grad-basis@reduced)/(mass*mass*amplitude))
            current = np.einsum('i,aij,j->a', (vacuum+delta).conj(), reps, vacuum+delta, optimize=True)
            errors.append(maximum(current))
            higgs = [75, 80]
            aa = (w.conj().T@w)[np.ix_(higgs, higgs)]
            bb = np.einsum('k,kij->ij', f.conj(), cubic, optimize=True)[np.ix_(higgs, higgs)]
            hs = np.block([[(aa+bb).real, -aa.imag-bb.imag], [aa.imag-bb.imag, (aa-bb).real]])
            d, mixing = abs(B_COUPLING*z)**2, B_COUPLING*ff[0]
            exact = np.array([d-abs(mixing)]*2+[d+abs(mixing)]*2)
            errors.append(maximum(np.linalg.eigvalsh(hs)-exact)/(2*mass*KAPPA*amplitude**2))
            # Independent potential finite difference with quartic cancellation.
            for _ in range(2):
                dr = rng.normal(size=4)
                dr /= np.linalg.norm(dr)
                dh = np.zeros(82, complex)
                dh[higgs] = (dr[:2]+1j*dr[2:])/np.sqrt(2)
                def central(step):
                    v = lambda q: float(np.vdot(f_at(q), f_at(q)).real)
                    return (v(delta+step*dh)+v(delta-step*dh)-2*v(delta))/(step*step)
                h = amplitude/8
                fd = (4*central(h/2)-central(h))/3
                errors.append(abs(fd-dr@hs@dr)/(2*mass*KAPPA*amplitude**2))
            rows.append({'mass_over_V': float(mass), 'xi': xi,
                         'initial_Higgs_mass_squared_over_m_squared': (exact/mass**2).tolist(),
                         'leading_negative_mass_squared_over_m_squared': float(xi**4-2*xi**2)})
    assert max(errors) < 1e-8, max(errors)
    return {'max_scaled_independent_error': max(errors), 'rows': rows,
            'invariant_background_fields': 'Sigma singlet and one record mass species; Higgs background zero',
            'source_is_an_exact_static_solution': False}


def evolve(mass, xi, accuracy):
    amplitude, scale, bg = initial(mass, xi)
    waves = [xi/2, 2*xi]
    y0 = np.r_[bg, np.eye(4).ravel(), np.eye(4).ravel()].astype(complex)
    def rhs(tau, y):
        z, s = scale*y[0], amplitude*y[1]
        fz, fs = field_values(z, s, mass)
        gz = (-M+C_COUPLING*z).conjugate()*fz+2*A_COUPLING*s.conjugate()*fs
        gs = 2*A_COUPLING*s.conjugate()*fz+(mass+2*A_COUPLING*z).conjugate()*fs
        out = np.zeros_like(y)
        out[:4] = [y[2], y[3], -gz/(mass*mass*scale), -gs/(mass*mass*amplitude)]
        d, mix = abs(B_COUPLING*z)**2/mass**2, B_COUPLING*fz/mass**2
        q = np.array([[d, mix], [mix.conjugate(), d]])
        for i, wave in enumerate(waves):
            start = 4+16*i
            mat = y[start:start+16].reshape(4, 4)
            derivative = np.vstack([mat[2:], -(wave*wave*np.eye(2)+q)@mat[:2]])
            out[start:start+16] = derivative.ravel()
        return out
    solution = solve_ivp(rhs, [0, FINAL_TAU], y0, method='DOP853',
                         rtol=accuracy, atol=accuracy*.01, t_eval=np.linspace(0, FINAL_TAU, 201))
    assert solution.success, solution.message
    z, s = scale*solution.y[0], amplitude*solution.y[1]
    dz, ds = mass*scale*solution.y[2], mass*amplitude*solution.y[3]
    fz, fs = field_values(z, s, mass)
    energy = abs(dz)**2+abs(ds)**2+abs(fz)**2+abs(fs)**2
    energy_error = maximum(energy/energy[0]-1)
    j = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    rows, canonical_error = [], 0.
    for i, wave in enumerate(waves):
        mats = solution.y[4+16*i:20+16*i].T.reshape(-1, 4, 4)
        for mat in mats:
            canonical_error = max(canonical_error, maximum(mat.conj().T@j@mat-j))
        mat = mats[-1]
        rescale = np.diag([np.sqrt(wave)]*2+[1/np.sqrt(wave)]*2)
        canonical_map = rescale@mat@np.linalg.inv(rescale)
        gains = np.linalg.svd(canonical_map, compute_uv=False)
        beta = (mat[:2, :2]-mat[2:, 2:]+1j*wave*mat[:2, 2:]+1j*mat[2:, :2]/wave)/2
        rows.append({'k_over_m': wave, 'largest_free_Gaussian_variance_gain': float(gains[0]**2),
                     'reference_Bogoliubov_beta_norm_squared': float(np.sum(abs(beta)**2)),
                     'reference_scope': 'free initial Gaussian covariance; no asymptotic particle-number assertion'})
    assert max(energy_error, canonical_error) < 1e-6
    return {'mass_over_V': float(mass), 'xi': xi, 'final_mt': FINAL_TAU,
            'rhs_evaluations': solution.nfev, 'relative_background_energy_error': energy_error,
            'canonical_Wronskian_error': canonical_error, 'rows': rows,
            'background_return_not_assumed': True,
            'final_background_scaled_real_imag': [[float(v.real), float(v.imag)] for v in solution.y[:4, -1]]}


def dynamics():
    rows = []
    for mass in MASSES:
        for xi in XIS:
            start = time.perf_counter()
            low, high = evolve(mass, xi, 2e-8), evolve(mass, xi, 2e-10)
            comparisons = [abs(a['largest_free_Gaussian_variance_gain']/b['largest_free_Gaussian_variance_gain']-1)
                           for a, b in zip(low['rows'], high['rows'])]
            assert max(comparisons) < 1e-6, comparisons
            high['independent_accuracy_relative_gain_difference'] = max(comparisons)
            rows.append(high)
            print(json.dumps({'dynamic_case': [float(mass), xi], 'seconds': time.perf_counter()-start,
                              'gain': [r['largest_free_Gaussian_variance_gain'] for r in high['rows']]}), flush=True)
    return {'rows': rows, 'Higgs_backreaction_on_source': 'not included beyond linear order',
            'physical_total_source_plus_amplified_quantum_energy': 'not computed',
            'scope': 'exact homogeneous classical source trajectory and linear canonical Higgs propagation'}


def potential(x, xi, strength, sign):
    return strength*(sign*np.exp(-x*x)+xi*xi/2*np.exp(-2*x*x))


def critical_shoot(xi):
    def slope(strength):
        def rhs(x, y):
            return [y[1], potential(x, xi, strength, -1)*y[0]]
        sol = solve_ivp(rhs, [0, 10], [0, 1], method='DOP853', rtol=2e-12, atol=2e-13)
        assert sol.success
        return sol.y[1, -1]
    return brentq(slope, 2, 4.1, xtol=2e-11)


def critical_kernel(xi, order):
    nodes, weights = roots_legendre(order)
    x, w = 5*(nodes+1), 5*weights
    v = np.exp(-x*x)-xi*xi/2*np.exp(-2*x*x)
    root = np.sqrt(w*v)
    def apply(vector):
        a = root*vector
        left = np.cumsum(x*a)
        right = np.cumsum(a[::-1])[::-1]-a
        return root*(left+x*right)
    operator = LinearOperator((order, order), matvec=apply, dtype=float)
    eigen = eigsh(operator, k=1, which='LA', tol=1e-12, return_eigenvectors=False,
                  v0=np.ones(order))[0]
    return 1/eigen


def static_gate():
    rows = []
    for xi in XIS:
        shooting = critical_shoot(xi)
        low, high = critical_kernel(xi, 512), critical_kernel(xi, 1024)
        richardson = (4*high-low)/3
        assert abs(shooting-richardson) < 1e-4
        cases = []
        for strength in [1., 2., 4.]:
            trace_bound = strength*(.5-xi*xi/8)
            alpha = .5
            trial = 1.5*alpha-strength*(alpha/(alpha+1))**1.5+strength*xi*xi/2*(alpha/(alpha+2))**1.5
            if strength in [1., 2.]:
                assert trace_bound < 1 and strength < shooting
            else:
                assert trial < 0 and strength > shooting
            cases.append({'g': strength, 'zero_energy_Birman_Schwinger_trace': trace_bound,
                          'Gaussian_trial_energy_times_R_squared': trial,
                          'static_operator_negative_mode': bool(strength > shooting)})
        rows.append({'xi': xi, 'critical_g_shooting': shooting,
                     'critical_g_kernel_512_1024': [low, high],
                     'critical_g_kernel_extrapolated': richardson,
                     'independent_critical_g_error': abs(shooting-richardson), 'cases': cases})
    return {'rows': rows, 'operator': '-d_x^2 + g[(xi^2/2) exp(-2x^2) +/- exp(-x^2)]',
            'scope': 'registered static quadratic operator only; no full dynamic-source stability assertion'}


def partial_wave(xi, strength, momentum, angular, sign, accuracy=2e-10, end=10.):
    start = 1e-4
    v0 = potential(0, xi, strength, sign)
    initial_derivative = (angular+1)/start-(momentum*momentum-v0)*start/(2*angular+3)
    def rhs(x, y):
        v = angular*(angular+1)/(x*x)+potential(x, xi, strength, sign)-momentum*momentum
        return [y[1], v*y[0]]
    sol = solve_ivp(rhs, [start, end], [1., initial_derivative], method='DOP853',
                    rtol=accuracy, atol=accuracy*.01)
    assert sol.success
    u, du = sol.y[:, -1]/np.max(np.abs(sol.y[:, -1]))
    z = momentum*end
    jl, nl = spherical_jn(angular, z), spherical_yn(angular, z)
    j, n = z*jl, z*nl
    dj = momentum*(jl+z*spherical_jn(angular, z, derivative=True))
    dn = momentum*(nl+z*spherical_yn(angular, z, derivative=True))
    delta = np.arctan2(u*dj-du*j, u*dn-du*n)
    cin, cout = np.linalg.solve(np.array([[j-1j*n, j+1j*n], [dj-1j*dn, dj+1j*dn]]), [u, du])
    scattering = cout/cin
    assert abs(scattering-np.exp(2j*delta)) < 1e-8 and abs(abs(scattering)-1) < 1e-8
    return scattering


def cross_sections(matrix, momentum):
    ell = np.arange(len(matrix))
    weights = 2*ell+1
    kept, converted = matrix[:, 0], matrix[:, 1]
    elastic = np.pi/momentum**2*np.sum(weights*abs(kept-1)**2)
    conversion = np.pi/momentum**2*np.sum(weights*abs(converted)**2)
    optical = 2*np.pi/momentum**2*np.sum(weights*(1-kept.real))
    nodes, w = roots_legendre(max(48, 2*len(matrix)+2))
    p = np.array([eval_legendre(int(l), nodes) for l in ell])
    fc = np.sum((weights*converted)[:, None]*p, axis=0)/(2j*momentum)
    fe = np.sum((weights*(kept-1))[:, None]*p, axis=0)/(2j*momentum)
    angular = 2*np.pi*np.array([np.sum(w*abs(fe)**2), np.sum(w*abs(fc)**2)])
    assert maximum(angular/np.array([elastic, conversion])-1) < 1e-8
    assert abs(optical/(elastic+conversion)-1) < 1e-8
    return np.array([elastic, conversion]), abs(optical/(elastic+conversion)-1)


def spatial_scattering():
    rows = []
    for mass in MASSES:
        for xi in XIS:
            for strength in [1., 2.]:
                radius = np.sqrt(strength)/(np.sqrt(2)*xi*mass)
                momentum = EPSILON*radius
                lmax = int(np.ceil(3*momentum+10))
                matrices = []
                for accuracy, end in [(2e-9, 8.), (2e-11, 10.)]:
                    branches = np.array([[partial_wave(xi, strength, momentum, ell, sign, accuracy, end)
                                          for sign in [1, -1]] for ell in range(lmax+1)])
                    matrices.append(np.column_stack([(branches[:, 0]+branches[:, 1])/2,
                                                      (branches[:, 0]-branches[:, 1])/2]))
                low, _ = cross_sections(matrices[0], momentum)
                high, optical_error = cross_sections(matrices[1], momentum)
                short, _ = cross_sections(matrices[1][:-4], momentum)
                err = maximum(high/low-1)
                tail = maximum((high-short)/high)
                assert max(err, tail) < 1e-5, (mass, xi, strength, err, tail)
                amp = xi*np.sqrt(mass/KAPPA)
                energy = np.pi**1.5*amp*amp*radius**3*(2*mass*mass+1.5/radius**2)
                # Circular free coherent source: n=2m|S|^2. The source is not
                # derived from chapter 74. This is a dilute independent-collision
                # estimate, not a rigorous bound on all inelastic processes.
                optical_depth = KAPPA**2/(4*np.pi)*2*mass*amp*amp*radius*np.sqrt(np.pi)
                matrix = matrices[1]
                rows.append({'mass_over_V': float(mass), 'xi': xi, 'g': strength,
                             'R_times_V': float(radius), 'epsilon_R': float(momentum), 'lmax': lmax,
                             'sigma_elastic_conversion_over_R_squared': high.tolist(),
                             'independent_scattering_relative_difference': err, 'last_four_partial_wave_relative_contribution': tail,
                             'optical_theorem_relative_error': optical_error,
                             'partial_wave_no_conversion_real_imag': [[float(v.real), float(v.imag)] for v in matrix[:, 0]],
                             'partial_wave_conversion_probabilities': (abs(matrix[:, 1])**2).tolist(),
                             'max_partial_wave_flux_error': maximum(abs(matrix[:, 0])**2+abs(matrix[:, 1])**2-1),
                             'free_circular_Gaussian_source_energy_over_V': float(energy),
                             'free_particle_occupation_estimate': float(2*mass*amp*amp*np.pi**1.5*radius**3),
                             'crossing_over_free_spreading_time_scale': float(1/(mass*radius)),
                             'self_interaction_small_parameter_xi_squared': xi*xi,
                             'heavy_derivative_small_parameter_m_over_M': float(mass/M),
                             'central_dilute_independent_collision_optical_depth_estimate': float(optical_depth)})
                print(json.dumps({'static_scattering_case': [float(mass), xi, strength],
                                  'sigma_over_R2': high.tolist()}), flush=True)
    return {'rows': rows,
            'scope': 'unitary scalar scattering in the registered fixed quadratic background, not the full CE S matrix',
            'source_and_detector_prepared_by_action': False,
            'omitted': 'dynamic background, fermion and source fluctuations, competing radiation, actual detector and event rule'}


def main():
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/76_*.md'))
    digest = hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 76.2')[0].encode()).hexdigest()
    assert digest == PREREG
    result = {'schema_version': 1, 'candidate': 'CE-UR4-D1/D2', 'scientific_success': False,
              'full_joint_rmse': None, 'new_observational_inputs_or_fits': False,
              'preregistration_sha256': digest,
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), folder/'ce_singlet_record_observability.py',
                                 folder/'ce_supersymmetric_record.py', folder/'ce_simple_group_record.py']},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'action_and_initial_curvatures': action_checks()}
    print('action checks passed', flush=True)
    result['homogeneous_dynamics'] = dynamics()
    result['finite_source_static_gate'] = static_gate()
    print('finite-source spectral checks passed', flush=True)
    result['conditional_spatial_scattering'] = spatial_scattering()
    result['result'] = 'HOMOGENEOUS_LONG_WAVE_INSTABILITY; FINITE_STATIC_BRANCH_AND_COMPLEX_SCATTERING_PHASES'
    result['sources'] = ['https://arxiv.org/abs/hep-ph/9709356', 'https://arxiv.org/abs/math-ph/0411005',
                         'https://dlmf.nist.gov/10.47',
                         'https://pdg.lbl.gov/2025/reviews/rpp2025-rev-resonances.pdf']
    result['open_gates'] = ['physical finite-source preparation and symmetry resource',
                           'full dynamic-source scattering and inelastic losses', 'backreaction and full non-event instrument',
                           'actual asynchronous records and common gravity', 'frozen covariance-aware joint predictions']
    (folder/'ce_finite_record_medium.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'candidate': result['candidate'], 'result': result['result'],
                      'static_gate': result['finite_source_static_gate']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
