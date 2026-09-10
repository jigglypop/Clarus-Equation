"""Chapter 74: light-record pair preparation, partial cuts, and conditional transport.

The source spectrum, stationary probe layer, angular acceptance, and response
are supplied conditions. Cross-section products are not event probabilities.
No complete detector, no-event amplitude, or joint observational prediction.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import erfc, ndtr
import sympy as sy

from ce_singlet_record_observability import build_model
from ce_supersymmetric_record import error, f_terms


PREREG = 'e345b05cc71b6ce754637d8181b3e5a26d4b490743b265e97d9b3a232427e118'
TOL, DISPERSION_TOL, TRANSFER_TOL = 1e-8, 1e-7, 1e-6
M0, HEAVY, KAPPA = 1e-4, 1., .05
A, B = 1/(2*np.sqrt(15)), 3/(2*np.sqrt(15))
MASSES = M0*np.array([5+np.pi/2, 5-np.pi/2])
E0 = 2.05*MASSES[0]
WIDTH = .002*E0
EPSILON = .1*MASSES[1]
SOURCE_ANGLE, DETECTOR_ANGLE = .01, .02
SOURCE_FRACTION = 1-np.cos(SOURCE_ANGLE)
DETECTOR_SOLID_ANGLE = 2*np.pi*(1-np.cos(DETECTOR_ANGLE))
TRANSFER_SCALE = SOURCE_FRACTION*DETECTOR_SOLID_ANGLE*KAPPA**4/(128*np.pi**3)


def vertices():
    _, _, _, mass, cubic, vacuum = build_model(M0)
    _, w = f_terms(vacuum, mass, cubic)
    assert abs(-w[11, 11].real-HEAVY) < TOL
    assert abs(cubic[11, 35, 35]/2-A) < TOL
    assert abs(cubic[11, 75, 80]-B) < TOL
    assert abs(A*B/HEAVY-KAPPA) < TOL
    rotation = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    idx = [35, 59]
    assert error(rotation.T@w[np.ix_(idx, idx)]@rotation-np.diag(MASSES)) < TOL
    assert error(rotation.T@cubic[11][np.ix_(idx, idx)]@rotation-2*A*np.eye(2)) < TOL
    assert error(np.delete(cubic[:24, 35, 35], 11)) < TOL
    z, zb, x, xb, a, m, M = sy.symbols('z zb x xb a m M', real=True)
    potential = sy.expand((-M*z+a*x*x)*(-M*zb+a*xb*xb)
                          +(m+2*a*z)*(m+2*a*zb)*x*xb)
    zero = {z: 0, zb: 0, x: 0, xb: 0}
    ss = -sy.diff(potential, z, xb, xb).subs(zero)
    sa = -sy.diff(potential, z, x, xb).subs(zero)
    yukawa = sy.diff(m*x*x/2+a*z*x*x, z, x, x)
    assert ss == 2*M*a and sa == -2*a*m and yukawa == 2*a
    return {'a': A, 'b': B, 'M_over_V': HEAVY, 'kappa_times_V': KAPPA,
            'scalar_pair_vertex': str(ss), 'scalar_antiscalar_vertex': str(sa),
            'chiral_Majorana_yukawa': str(yukawa), 'initial_weak_components': 1,
            'initial_spin_average': False, 'final_identical_factors_SS_SA_FF': [.5, 1., .5]}


def gamma_matrices():
    zero, eye = np.zeros((2, 2)), np.eye(2)
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    g0 = np.block([[zero, eye], [eye, zero]])
    gamma = [g0]+[np.block([[zero, p], [-p, zero]]) for p in pauli]
    pr, pl = np.diag([0, 0, 1, 1]), np.diag([1, 1, 0, 0])
    return gamma, pr, pl


def spin_gram(s, m, direction):
    gamma, pr, pl = gamma_matrices()
    momentum = np.sqrt(s/4-m*m)*direction
    slash3 = np.sqrt(s)/2*gamma[0]-sum(p*g for p, g in zip(momentum, gamma[1:]))
    slash4 = np.sqrt(s)/2*gamma[0]+sum(p*g for p, g in zip(momentum, gamma[1:]))
    left, right = slash3+m*np.eye(4), slash4-m*np.eye(4)
    return np.array([[np.trace(left@pr@right@pl), np.trace(left@pr@right@pr)],
                     [np.trace(left@pl@right@pl), np.trace(left@pl@right@pr)]]).real


def source_sigma(s, m):
    beta = np.sqrt(1-4*m*m/s)
    ss = KAPPA**2*beta/(8*np.pi*(1-s/HEAVY**2)**2)
    return ss*np.array([1., 2*m*m/HEAVY**2, (s-2*m*m)/HEAVY**2])


def source_and_cut_checks():
    nodes, weights = np.polynomial.legendre.leggauss(12)
    rows, errors, positivity = [], [], []
    for energy in [E0, .01, .1]:
        s = energy*energy
        for m in MASSES:
            beta = np.sqrt(1-4*m*m/s)
            angular, cuts = np.zeros(3), np.zeros((2, 2))
            for c, weight in zip(nodes, weights):
                n = np.array([np.sqrt(1-c*c), 0, c])
                spin = spin_gram(s, m, n)
                expected_spin = np.array([[s-2*m*m, -2*m*m], [-2*m*m, s-2*m*m]])
                errors.append(error((spin-expected_spin)/s))
                positivity.append(float(np.min(np.linalg.eigvalsh(spin))/s))
                amp2 = B*B*s/(HEAVY**2-s)**2*np.array(
                    [(2*A*HEAVY)**2, (2*A*m)**2, (2*A)**2*spin[0, 0]])
                angular += weight*amp2*beta/(32*np.pi*s)*np.array([.5, 1, .5])
                # Im Pi = one half of the phase-space sum; SS and anti-SS
                # each feed just one source component. SA is common to both.
                css = beta/(32*np.pi)*(2*A*HEAVY)**2*np.eye(2)
                csa = beta/(16*np.pi)*(2*A*m)**2*np.ones((2, 2))
                cff = beta/(32*np.pi)*(2*A)**2*spin
                cuts += weight/2*(css+csa+cff)
                positivity += [float(np.min(np.linalg.eigvalsh(q))/(A*A*HEAVY**2))
                               for q in [css, csa, cff, css+csa+cff]]
            expected = source_sigma(s, m)
            errors.append(error(angular/expected-1))
            cut_diagonal = A*A*beta*(HEAVY**2+s)/(8*np.pi)
            errors.append(error(cuts/cut_diagonal-np.eye(2)))
            im_forward = B*B*s/(HEAVY**2-s)**2*cuts[0, 0]
            errors.append(abs(im_forward/(s*sum(expected))-1))
            annihilation_one_component = KAPPA**2/(4*np.pi*beta*(1-s/HEAVY**2)**2)
            errors.append(abs(expected[0]/annihilation_one_component-beta*beta/2))
            rows.append({'ecm_over_V': float(energy), 'mass_over_V': float(m),
                         'sigma_SS_SA_FF_times_V_squared': expected.tolist(),
                         'cut_matrix_over_V_squared': cuts.tolist(),
                         'SA_off_diagonal_cut': float(csa[0, 1]),
                         'FF_off_diagonal_cut': float(cff[0, 1]),
                         'source_partial_optical_theorem_ratio': float(im_forward/(s*sum(expected)))})
    assert max(errors) < TOL and min(positivity) > -TOL
    a, m, M, s, beta = sy.symbols('a m M s beta', positive=True)
    css = beta/(32*sy.pi)*(2*a*M)**2*sy.eye(2)
    csa = beta/(16*sy.pi)*(2*a*m)**2*sy.ones(2)
    cff = beta/(32*sy.pi)*(2*a)**2*sy.Matrix([[s-2*m*m, -2*m*m], [-2*m*m, s-2*m*m]])
    identity = sy.simplify(css+csa+cff-a*a*beta*(M*M+s)/(8*sy.pi)*sy.eye(2))
    assert identity == sy.zeros(2)
    return {'rows': rows, 'max_independent_relative_error': max(errors),
            'smallest_scaled_cut_or_spin_eigenvalue': min(positivity),
            'off_diagonal_cancellation_identity': str(identity),
            'scope': 'tree production and its leading light-record absorptive cut; other cuts and local terms excluded'}


def bubble_parameter(r):
    roots = [] if r < 4 else [(1-np.sqrt(1-4/r))/2, (1+np.sqrt(1-4/r))/2]
    intervals = [0.]+roots+[1.]
    real = sum(quad(lambda x: -np.log(abs(1-r*x*(1-x))), lo, hi,
                    epsabs=2e-11, epsrel=2e-11)[0] for lo, hi in zip(intervals[:-1], intervals[1:]))
    return complex(real, np.pi*np.sqrt(1-4/r) if r > 4 else 0.)


def bubble_analytic(r):
    if r < 0:
        beta = np.sqrt(1-4/r)
        return complex(2-beta*np.log((beta+1)/(beta-1)), 0.)
    if r < 4:
        b = np.sqrt(4/r-1)
        return complex(2-2*b*np.arctan(1/b), 0.)
    beta = np.sqrt(1-4/r)
    return 2-beta*np.log((1+beta)/(1-beta))+1j*np.pi*beta


def dispersion(r, m):
    numerator = lambda x: (HEAVY**2*x+4*m*m)*np.sqrt(1-x)
    if r > 4:
        integral, uncertainty = quad(numerator, 0, 1, weight='cauchy', wvar=4/r,
                                     epsabs=2e-11, epsrel=2e-11, limit=200)
        real = -r*integral/4
        uncertainty *= abs(r)/4
    else:
        integral, uncertainty = quad(lambda x: numerator(x)/(4-r*x), 0, 1,
                                     epsabs=2e-11, epsrel=2e-11)
        real, uncertainty = r*r*integral/4, r*r*uncertainty/4
    imaginary = (HEAVY**2+m*m*r)*np.pi*np.sqrt(1-4/r) if r > 4 else 0.
    return complex(real, imaginary), uncertainty


def dispersion_checks():
    coefficient = A*A/(8*np.pi**2)
    rows, errors = [], []
    for m in MASSES:
        for r in [-1., 1., 3., 4.5, 16., E0*E0/(m*m)]:
            s = m*m*r
            by_parameter = (HEAVY**2+s)*bubble_parameter(r)-HEAVY**2*r/6
            by_analytic = (HEAVY**2+s)*bubble_analytic(r)-HEAVY**2*r/6
            by_dispersion, estimate = dispersion(r, m)
            errors += [abs(by_parameter-by_analytic), abs(by_parameter-by_dispersion)]
            if r > 4:
                cut = A*A*np.sqrt(1-4/r)*(HEAVY**2+s)/(8*np.pi)
                errors.append(abs(coefficient*by_parameter.imag/cut-1))
            rows.append({'mass_over_V': float(m), 's_over_m_squared': float(r),
                         'twice_subtracted_Pi_over_coefficient': [by_parameter.real, by_parameter.imag],
                         'independent_dispersion_scaled_error': float(abs(by_parameter-by_dispersion)),
                         'quad_scaled_error_estimate': estimate})
    assert max(errors) < DISPERSION_TOL
    return {'coefficient_a_squared_over_8pi_squared': coefficient, 'rows': rows,
            'max_scaled_independent_error': max(errors), 'local_constant_and_slope': 'unfixed',
            'scope': 'light-record cut only, Pi(s)-Pi(0)-s Pi_prime(0); no full self-energy or fitted subtraction'}


def energy_rule(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    x = 8*nodes
    weights = 8*weights*np.exp(-x*x/2)/np.sqrt(2*np.pi)
    # Conditional central-domain integral. Tails are explicitly left unmodeled.
    return E0+WIDTH*x, weights/(1-erfc(8/np.sqrt(2)))


def cone_rule(angle, axis, polar_order, azimuth_order):
    nodes, weights = np.polynomial.legendre.leggauss(polar_order)
    width = 1-np.cos(angle)
    cosine = 1-width*(1-nodes)/2
    phi = 2*np.pi*np.arange(azimuth_order)/azimuth_order
    transverse = np.sqrt(1-cosine*cosine)
    x = (transverse[:, None]*np.cos(phi)).ravel()
    y = (transverse[:, None]*np.sin(phi)).ravel()
    z = np.broadcast_to(cosine[:, None], (polar_order, azimuth_order)).ravel()
    directions = np.array([x, y, z]).T
    if axis == 'y':
        directions = directions[:, [0, 2, 1]]
    return directions, np.repeat(weights/(2*azimuth_order), azimuth_order)


def preparation_checks():
    beta = np.sqrt(1-4*MASSES**2/E0**2)
    prior = beta/sum(beta)
    pair = np.array([np.sqrt(prior[0]), 0, 0, np.sqrt(prior[1])])
    joint = np.outer(pair, pair)
    local = np.trace(joint.reshape(2, 2, 2, 2), axis1=1, axis2=3)
    assert error(local-np.diag(prior)) < TOL
    rotation = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    copy = rotation@local@rotation.T
    exchange = np.array([[0, 1], [1, 0]])
    assert error(exchange@copy@exchange-copy) < TOL
    energy, weights = energy_rule(64)
    assert abs(sum(weights)-1) < TOL
    integrated = np.array([sum(w*source_sigma(e*e, m) for e, w in zip(energy, weights)) for m in MASSES])
    lo, hi = E0-8*WIDTH, E0+8*WIDTH
    assert lo > 2*MASSES[0] and hi < 4*MASSES[1]
    center = EPSILON*(E0/2)/(E0/2-E0*beta/2+EPSILON)
    control = EPSILON*(E0/2)/(E0/2+EPSILON)
    return {'ecm_over_V': E0, 'energy_width_over_V': WIDTH,
            'central_8sigma_bounds_over_V': [lo, hi],
            'pair_thresholds_over_V_plus_minus': (2*MASSES).tolist(),
            'four_light_record_threshold_over_V': float(4*MASSES[1]),
            'Gaussian_probability_outside_central_domain': float(erfc(8/np.sqrt(2))),
            'Gaussian_probability_below_heavy_pair_threshold': float(ndtr((2*MASSES[0]-E0)/WIDTH)),
            'Gaussian_probability_above_four_record_threshold': float(ndtr((E0-4*MASSES[1])/WIDTH)),
            'tail_scope': 'probability mass only, not a bound on omitted cross-section-weighted channels',
            'center_beta_plus_minus': beta.tolist(), 'center_scalar_pair_species_prior': prior.tolist(),
            'center_local_density_copy_basis': copy.tolist(),
            'center_conditional_species_mode_entanglement_bits': float(-sum(prior*np.log2(prior))),
            'companion_removed_from_full_state': False,
            'central_domain_source_cross_sections_SS_SA_FF_plus_minus': integrated.tolist(),
            'non_SS_fraction_of_source_cross_section': float(integrated[:, 1:].sum()/integrated.sum()),
            'readout_center_energy_over_epsilon_plus_minus': (center/EPSILON).tolist(),
            'control_center_energy_over_epsilon_both': float(control/EPSILON),
            'fixed_classification_threshold_over_V': float(np.mean(center)),
            'minimum_light_record_momentum_at_common_heavy_threshold_over_V': float(np.sqrt(MASSES[0]**2-MASSES[1]**2))}


def fold_transfer(m, species, direction, resolution, orders, threshold):
    ne, na, nphi = orders
    energies, ew = energy_rule(ne)
    source_n, sw = cone_rule(SOURCE_ANGLE, 'z', na, nphi)
    output_n, dw = cone_rule(DETECTOR_ANGLE, direction, na, nphi)
    values = np.zeros(4)
    max_onshell, max_collision = 0., 0.
    for energy, weight in zip(energies, ew):
        ep = energy/2
        beta = np.sqrt(1-m*m/(ep*ep))
        momentum = ep*beta*source_n
        px = momentum[:, 0, None]
        pout = momentum@output_n.T
        c = output_n[None, :, 0]
        denominator = ep-pout+EPSILON*(1-c)
        eout = EPSILON*(ep-px)/denominator
        s = m*m+2*EPSILON*(ep-px)
        t = -2*EPSILON*eout*(1-c)
        relative_speed = (ep-px)/ep
        residence = 1/(beta*source_n[:, 2, None])
        differential = KAPPA**2*s/(16*np.pi**2*denominator**2*(1-t/HEAVY**2)**2)
        measure = (weight*source_sigma(energy*energy, m)[0]*SOURCE_FRACTION
                   *relative_speed*residence*differential*DETECTOR_SOLID_ANGLE
                   *sw[:, None]*dw[None, :]/TRANSFER_SCALE)
        wrong = ndtr((eout-threshold)/(resolution*EPSILON)) if species == 0 else ndtr((threshold-eout)/(resolution*EPSILON))
        values += [np.sum(measure), np.sum(measure*wrong),
                   np.sum(measure*(s-m*m)/s), np.sum(measure*eout/EPSILON)]
        # p_record' = p_record + k - k'; test its shell, with all energies retained.
        pfinal = momentum[:, None, :]+np.array([EPSILON, 0, 0])-eout[:, :, None]*output_n
        efinal = ep+EPSILON-eout
        max_onshell = max(max_onshell, error((efinal*efinal-np.sum(pfinal*pfinal, axis=2)-m*m)/s))
        max_collision = max(max_collision, float(np.max(np.sqrt(s))))
        assert np.min(efinal) > 0 and np.min(measure) >= 0
    # Any extra light pair in a readout needs invariant energy at least m+2m_-.
    assert max_collision < m+2*MASSES[1] and max_onshell < TOL
    return values, max_onshell, max_collision


def transfers(threshold):
    rows, checks = [], []
    for direction in ['z', 'y']:
        for resolution in [.5, 1.]:
            batches = []
            for orders in [(32, 4, 8), (48, 6, 12)]:
                results = [fold_transfer(m, i, direction, resolution, orders, threshold)
                           for i, m in enumerate(MASSES)]
                batches.append(np.array([r[0] for r in results]))
                checks += [r[1] for r in results]
            delta = error(batches[1]-batches[0])
            assert delta < TRANSFER_TOL, (direction, resolution, delta)
            values = batches[-1]
            total = sum(values[:, 0])
            rows.append({'detector_direction': direction, 'detector_sigma_over_epsilon': resolution,
                         'quadrature_orders': [[32, 4, 8], [48, 6, 12]],
                         'max_scaled_quadrature_difference': delta,
                         'transfer_coefficient_times_V_fourth_plus_minus': (TRANSFER_SCALE*values[:, 0]).tolist(),
                         'accepted_plus_fraction_with_source_weights': float(values[0, 0]/total),
                         'error_given_species_plus_minus': (values[:, 1]/values[:, 0]).tolist(),
                         'error_conditioned_on_SS_sourced_accepted_event': float(sum(values[:, 1])/total),
                         'accepted_F_fraction_plus_minus': (values[:, 2]/values[:, 0]).tolist(),
                         'mean_energy_over_epsilon_plus_minus': (values[:, 3]/values[:, 0]).tolist(),
                         'max_collision_invariant_energy_over_V_plus_minus': [r[2] for r in results]})
    # Point-cone limit independently cancels production beta and layer residence.
    for m in MASSES:
        ep, s = E0/2, E0**2
        beta = np.sqrt(1-m*m/(ep*ep))
        for axis in ['z', 'y']:
            d = ep-(ep*beta if axis == 'z' else 0)+EPSILON
            eout = EPSILON*ep/d
            t = -2*EPSILON*eout
            collision_s = m*m+2*EPSILON*ep
            actual = source_sigma(s, m)[0]/beta*KAPPA**2*collision_s/(16*np.pi**2*d*d*(1-t/HEAVY**2)**2)
            exact = KAPPA**4*collision_s/(128*np.pi**3*d*d*(1-s/HEAVY**2)**2*(1-t/HEAVY**2)**2)
            checks.append(abs(actual/exact-1))
    assert max(checks) < TOL
    return {'epsilon_over_V': float(EPSILON), 'threshold_over_V': threshold,
            'source_cone_pair_fraction': float(SOURCE_FRACTION),
            'detector_solid_angle': float(DETECTOR_SOLID_ANGLE),
            'transfer_scale_times_V_fourth': float(TRANSFER_SCALE), 'rows': rows,
            'max_kinematic_or_point_cone_error': max(checks),
            'probability_formula': 'P_alpha = source_integrated_luminosity * probe_column_density * A_alpha',
            'dimensions': 'A: V^-4; source luminosity and probe column density: V^2 each',
            'conditions': 'central 8sigma luminosity spectrum; stationary thin z layer covering both arrival times; dilute separated collisions',
            'excluded': 'SA/FF source branches after preparation; source CM spread and collision position; finite propagation and actual detector dynamics'}


def main():
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/74_*.md'))
    digest = hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 74.2')[0].encode()).hexdigest()
    assert digest == PREREG
    preparation = preparation_checks()
    report = {'schema_version': 1, 'candidate': 'CE-UR4-P1', 'scientific_success': False,
              'full_joint_rmse': None, 'preregistration_sha256': digest,
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), folder/'ce_singlet_record_observability.py',
                                 folder/'ce_supersymmetric_record.py', folder/'ce_simple_group_record.py']},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sympy': sy.__version__},
              'vertices': vertices(), 'source_and_partial_cut': source_and_cut_checks(),
              'twice_subtracted_light_record_loop': dispersion_checks(),
              'pair_preparation': preparation,
              'conditional_source_to_readout': transfers(preparation['fixed_classification_threshold_over_V']),
              'sources': ['https://arxiv.org/abs/hep-ph/9709356', 'https://arxiv.org/abs/0709.1075',
                          'https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf',
                          'https://pdg.lbl.gov/2025/reviews/rpp2025-rev-resonances.pdf'],
              'new_observational_inputs_or_fits': False,
              'result': 'PARTIAL_SOURCE_CUT_AND_CONDITIONAL_TRANSPORT; COMPLETE_INSTRUMENT_OPEN',
              'open_gates': ['all source branches through the readout', 'physical source and detector preparation',
                             'complete visible scattering, non-event phase and local renormalization data',
                             'finite wave-packet propagation and definite asynchronous records',
                             'common gravity and frozen covariance-aware joint predictions']}
    (folder/'ce_record_pair_preparation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'candidate': report['candidate'], 'preparation': preparation,
                      'source_max_error': report['source_and_partial_cut']['max_independent_relative_error'],
                      'dispersion_max_error': report['twice_subtracted_light_record_loop']['max_scaled_independent_error'],
                      'readout': report['conditional_source_to_readout']}, indent=2))


if __name__ == '__main__':
    main()
