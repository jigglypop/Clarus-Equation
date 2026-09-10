"""Chapter 73: retained-record scattering, inclusive rates, and recoil information.

The packet source, geometric beam model, and detector response are supplied
conditions. No full S matrix, autonomous pointer, or joint empirical success.
"""
import hashlib
import itertools
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import ndtr, erfc
from scipy.integrate import quad
import sympy as sy

from ce_singlet_record_observability import build_model
from ce_supersymmetric_record import error, f_terms


TOL = 1e-8
INTEGRAL_TOL = 1e-7
PREREG = '66cb18c9e26338cb134105fad026143bb3caf7de179b959a34bef9c20d180c67'
M0, HEAVY, KAPPA = 1e-4, 1., .05
MASSES = M0*np.array([5+np.pi/2, 5-np.pi/2])
EPSILON = .1*MASSES[1]
PROBE_ENERGY_WIDTH = .002*EPSILON
BEAM_RATIO = 100.
SIGMA_REF = KAPPA**2/(4*np.pi)
CASES = [(0.001, .003), (.01, .003), (.01, .01), (.1, .03)]


def vertices():
    _, _, reps, matrix, cubic, vacuum = build_model(M0)
    _, w = f_terms(vacuum, matrix, cubic)
    M = -w[11, 11].real
    a, b = cubic[11, 35, 35].real/2, cubic[11, 75, 80].real
    assert abs(a*b/M-KAPPA) < TOL
    # Independent polynomial extraction, with conjugate variables independent.
    z, zb, x, xb, h, hb, hc, hbc = sy.symbols('z zb S Sb H Hb Hc Hbc')
    aa, bb, mm, MM, kk = sy.symbols('a b m M kappa', real=True)
    Fz = -MM*z+aa*x*x+bb*h*hb
    Fx = (mm+2*aa*z)*x
    Fh, Fhb = bb*z*hb, bb*z*h
    conjugates = {z: zb, zb: z, x: xb, xb: x, h: hc, hc: h, hb: hbc, hbc: hb}
    conj = lambda f: f.xreplace(conjugates)
    potential = sy.expand(sum(f*conj(f) for f in [Fz, Fx, Fh, Fhb]))
    zero = {v: 0 for v in [z, zb, x, xb, h, hb, hc, hbc]}
    source_record = sy.diff(potential, z, x, xb).subs(zero)
    source_probe = sy.diff(potential, zb, h, hb).subs(zero)
    assert source_record == 2*aa*mm and source_probe == -MM*bb
    W = mm*x*x/2+kk*x*x*h*hb
    F = sy.diff(W, x)
    eff_potential = sy.expand(F*conj(F))
    boson = sy.diff(eff_potential, x, xb, h, hb).subs(zero)
    fermion = sy.diff(W, x, hb, x, h)
    assert boson == 2*kk*mm and fermion == 2*kk
    # This incoming H weak component fixes the final component: no factor two.
    assert error(cubic[11, 75:77, 80:82]-b*np.eye(2)) < TOL
    permutation = np.r_[0:24, 48:72, 24:48, 72:82]
    symmetry_error = 0.
    for sign in [1, -1]:
        signs = np.ones(82)
        signs[24:72] = sign
        mass_transformed = matrix[np.ix_(permutation, permutation)]*signs[:, None]*signs[None, :]
        cubic_transformed = cubic[np.ix_(permutation, permutation, permutation)]
        cubic_transformed = cubic_transformed*signs[:, None, None]*signs[None, :, None]*signs[None, None, :]
        reps_transformed = reps[:, permutation, :][:, :, permutation]*signs[None, :, None]*signs[None, None, :]
        symmetry_error = max(symmetry_error, error(mass_transformed-matrix),
                             error(cubic_transformed-cubic), error(reps_transformed-reps))
    assert symmetry_error < TOL
    return {'UV_record_scalar_source': str(source_record),
            'UV_probe_scalar_source': str(source_probe),
            'boson_effective_vertex': str(boson), 'fermion_effective_vertex': str(fermion),
            'kappa_from_original_tensor': a*b/M,
            'full_action_species_parity_error': symmetry_error,
            'propagator_factor': '1/(1-t/M_Sigma^2)', 'final_weak_channels_for_fixed_probe': 1,
            'channels': ['S_alpha + H -> S_alpha + anti_Hbar',
                         'S_alpha + H -> Majorana_X_alpha + anti_Hbar_fermion']}


PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
SPIN_EPSILON = np.array([[0, 1], [-1, 0]])


def spin_sum(p, q):
    def roots(four):
        mat = four[0]*np.eye(2)+np.einsum('i,ijk->jk', four[1:], PAULI)
        e, v = np.linalg.eigh(mat)
        return v@np.diag(np.sqrt(np.maximum(e, 0)))
    return float(np.sum(np.abs(roots(p).T@SPIN_EPSILON@roots(q))**2))


def analytic_checks():
    nodes, weights = np.polynomial.legendre.leggauss(48)
    results, errors, conservation = [], [], []
    for m in MASSES:
        s = m*m+2*m*EPSILON
        pcm = (s-m*m)/(2*np.sqrt(s))
        e_record = (s+m*m)/(2*np.sqrt(s))
        sums = np.zeros(2)
        for cosine, weight in zip(nodes, weights):
            direction = np.array([np.sqrt(1-cosine*cosine), 0, cosine])
            p3, p4 = np.r_[e_record, pcm*direction], np.r_[pcm, -pcm*direction]
            spin = spin_sum(p3, p4)
            errors.append(abs(spin/(s-m*m)-1))
            t = -2*pcm*pcm*(1-cosine)
            prop = 1/(1-t/HEAVY**2)
            amps = 4*KAPPA**2*prop**2*np.array([m*m, spin])
            sums += weight*amps/(32*np.pi*s)
            conservation += [error(p3+p4-np.array([np.sqrt(s), 0, 0, 0])),
                             abs(p3[0]**2-np.dot(p3[1:], p3[1:])-m*m)/s]
        exact = SIGMA_REF/(1+4*pcm*pcm/HEAVY**2)*np.array([m*m/s, (s-m*m)/s])
        errors += [error(sums/exact-1)]
        # Independently integrate the lab differential rate, including recoil.
        lab = np.zeros(2)
        for cosine, weight in zip(nodes, weights):
            denominator = m+EPSILON*(1-cosine)
            outgoing = m*EPSILON/denominator
            t = -2*EPSILON*outgoing*(1-cosine)
            amps = 4*KAPPA**2*np.array([m*m, s-m*m])/(1-t/HEAVY**2)**2
            lab += weight*amps/(32*np.pi*denominator**2)
        errors.append(error(lab/exact-1))
        results.append({'mass_over_V': float(m), 'sigma_B_times_V_squared': float(exact[0]),
                        'sigma_F_times_V_squared': float(exact[1]), 'sigma_sum_times_V_squared': float(exact.sum()),
                        'conditional_F_fraction': float((s-m*m)/s),
                        'outgoing_energy_at_right_angle_over_V': float(m*EPSILON/(m+EPSILON))})
    assert max(errors+conservation) < TOL
    m, s, k, p, M = sy.symbols('m s k p M', positive=True)
    b = k*k*m*m/(4*sy.pi*s*(1+4*p*p/(M*M)))
    f = k*k*(s-m*m)/(4*sy.pi*s*(1+4*p*p/(M*M)))
    identity = sy.simplify(b+f-k*k/(4*sy.pi*(1+4*p*p/(M*M))))
    assert identity == 0
    return {'rows': results, 'independent_relative_error': max(errors),
            'on_shell_error': max(conservation), 'inclusive_symbolic_identity': str(identity),
            'contact_inclusive_sigma': 'kappa^2/(4 pi), independent of mass and incident energy',
            'retained_mass_label': True, 'scalar_subspace_preserved': False}


def gaussian_rule(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    assert np.max(np.abs(x)) < 8  # registered central domain; omitted Gaussian tail is bounded.
    return x, w/np.sqrt(2*np.pi)


def fold_case(zeta, resolution, mass, plus_label, orders):
    np_order, ne_order, na_order, nc_order = orders
    pnodes, pweights = gaussian_rule(np_order)
    enodes, eweights = gaussian_rule(ne_order)
    anodes, aweights = gaussian_rule(na_order)
    cnodes, cweights = np.polynomial.legendre.leggauss(nc_order)
    pgrid = np.array(list(itertools.product(range(np_order), repeat=3)))
    momenta = zeta*EPSILON*pnodes[pgrid]
    momentum_weights = np.prod(pweights[pgrid], axis=1)
    energies = EPSILON+PROBE_ENERGY_WIDTH*enodes
    weights_pe = momentum_weights[:, None]*eweights[None, :]
    ep = np.sqrt(mass*mass+np.sum(momenta*momenta, axis=1))[:, None]
    px, py, pz = [momenta[:, i, None] for i in range(3)]
    eps = energies[None, :]
    cutoff = np.mean(MASSES*EPSILON/(MASSES+EPSILON))
    result = np.zeros(4)
    angle_width = zeta/BEAM_RATIO
    sigma_x = 1/(2*zeta*EPSILON)
    sigma_z = 1/(2*PROBE_ENERGY_WIDTH)
    transverse_variance = (BEAM_RATIO**2+1)*sigma_x*sigma_x
    longitudinal_variance = sigma_x*sigma_x+sigma_z*sigma_z
    variance_ratio = longitudinal_variance/transverse_variance
    # Axial symmetry places the detector in the xz plane. Momentum remains in
    # this laboratory frame so its correlation with the finite beam is retained.
    for cnom, wc in zip(.05*cnodes, .05*cweights):
        for ix, iy in itertools.product(range(na_order), repeat=2):
            tx, ty = angle_width*anodes[ix], angle_width*anodes[iy]
            ni = np.array([tx, ty, 1])/np.sqrt(1+tx*tx+ty*ty)
            c = cnom*ni[2]+np.sqrt(1-cnom*cnom)*ni[0]
            p_in = px*ni[0]+py*ni[1]+pz*ni[2]
            p_out = px*np.sqrt(1-cnom*cnom)+pz*cnom
            denom = ep-p_out+eps*(1-c)
            eout = eps*(ep-p_in)/denom
            s = mass*mass+2*eps*(ep-p_in)
            t = -2*eps*eout*(1-c)
            velocity = (ep-p_in)/ep
            uz, ux, uy = pz/ep-ni[2], px/ep-ni[0], py/ep-ni[1]
            # Exact Gaussian x,t overlap for the supplied straight-trajectory
            # phase-space model. Overall 1/(2 pi transverse_variance) is outside.
            overlap = 1/np.sqrt(uz*uz+variance_ratio*(ux*ux+uy*uy))
            rate = KAPPA**2*s/(16*np.pi**2*denom**2*(1-t/HEAVY**2)**2)*velocity*overlap
            rate /= SIGMA_REF
            weighted = weights_pe*rate*(2*np.pi*wc*aweights[ix]*aweights[iy])
            wrong = ndtr((cutoff-eout)/(resolution*EPSILON)) if plus_label else ndtr((eout-cutoff)/(resolution*EPSILON))
            result += [np.sum(weighted), np.sum(weighted*wrong),
                       np.sum(weighted*(s-mass*mass)/s), np.sum(weighted*eout/EPSILON)]
    return result


def kinematic_domain_checks():
    rng = np.random.default_rng(7301)
    errors, margins = [], []
    for m in MASSES:
        # Conservative largest input in the registered 8-sigma domain.
        pmax = 8*.1*EPSILON
        emax = EPSILON+8*PROBE_ENERGY_WIDTH
        smax = m*m+2*emax*(np.sqrt(m*m+3*pmax*pmax)+np.sqrt(3)*pmax)
        margin = (m+2*MASSES[1])**2-smax
        assert margin > 0
        margins.append(float(margin))
        for _ in range(40):
            mom = rng.uniform(-pmax, pmax, size=3)
            ep = np.sqrt(m*m+np.dot(mom, mom))
            eps = rng.uniform(EPSILON-8*PROBE_ENERGY_WIDTH, emax)
            c = rng.uniform(-.1, .1)
            n = np.array([np.sqrt(1-c*c), 0, c])
            eout = eps*(ep-mom[2])/(ep-np.dot(mom, n)+eps*(1-c))
            incoming = np.r_[ep, mom]+np.array([eps, 0, 0, eps])
            outgoing_probe = np.r_[eout, eout*n]
            leftover = incoming-outgoing_probe
            errors.append(abs(leftover[0]**2-np.dot(leftover[1:], leftover[1:])-m*m)/(ep+eps)**2)
    # One-particle Wess-Zumino multiplet: two scalar charges, two fermion spins.
    label = np.diag([0]*4+[1]*4)
    for kind in range(4):
        channel = np.zeros((8, 2))
        channel[kind, 0], channel[4+kind, 1] = 1, 1
        assert error(label@channel-channel@np.diag([0, 1])) == 0
    assert max(errors) < TOL
    return {'moving_target_on_shell_relative_error': max(errors),
            'extra_record_pair_invariant_margin': margins,
            'one_record_space': 'direct sum over +/- of L2(d3p) tensor C4',
            'mass_projector_intertwining_error': 0.,
            'domain': 'one record multiplet, parity-preserving dynamics, below additional record-pair thresholds',
            'Gaussian_tail_union_bound': float(6*erfc(8/np.sqrt(2)))}


def source_overlap_checks():
    # Independent time integral after analytic convolution in position.
    # Coordinates are rescaled; the Gaussian covariance calculation is exact.
    relative_errors = []
    for variance_ratio, uz, ux, uy in [(1., .9, .1, .2), (.02, 1.1, -.2, .05), (5., .7, .3, -.1)]:
        A = uz*uz+variance_ratio*(ux*ux+uy*uy)
        numeric = quad(lambda t: np.exp(-A*t*t/2)/np.sqrt(2*np.pi), -np.inf, np.inf,
                       epsabs=1e-13, epsrel=1e-13)[0]
        relative_errors.append(abs(numeric*np.sqrt(A)-1))
    assert max(relative_errors) < TOL
    return {'Gaussian_overlap_time_integral_relative_error': max(relative_errors),
            'formula': 'L0 / sqrt(u_z^2 + (D_z/D_perp) (u_x^2+u_y^2))',
            'quantum_scope': 'Gaussian phase-space source with free straight trajectories; paraxial and low-momentum approximation, not exact packet QFT'}


def packets():
    output = []
    configs = [(7, 5, 3, 8), (11, 9, 5, 12), (17, 13, 7, 18)]
    for zeta, resolution in CASES:
        approximations, differences = [], []
        for orders in configs:
            values = np.array([fold_case(zeta, resolution, m, i == 0, orders)
                               for i, m in enumerate(MASSES)])
            approximations.append(values)
            if len(approximations) > 1:
                delta = error(values-approximations[-2])
                differences.append(delta)
                old_error = np.sum(approximations[-2][:, 1])/np.sum(approximations[-2][:, 0])
                new_error = np.sum(values[:, 1])/np.sum(values[:, 0])
                rare_error_converged = abs(old_error-new_error) < max(1e-12, .001*new_error)
                if delta < INTEGRAL_TOL and rare_error_converged:
                    break
        assert differences[-1] < INTEGRAL_TOL, (zeta, resolution, differences)
        assert rare_error_converged, (zeta, resolution, old_error, new_error)
        plus, minus = values
        rate_sum = plus[0]+minus[0]
        conditional_error = (plus[1]+minus[1])/rate_sum
        # Motion/diffraction are included above; this is a geometric single-pass
        # resource estimate, not an exact prepared-source scattering probability.
        sigma_p = zeta*EPSILON
        sigma_x = 1/(2*sigma_p)
        luminosity_scale = 1/(2*np.pi*(BEAM_RATIO**2+1)*sigma_x*sigma_x)
        mean_accepted_sigma = SIGMA_REF*rate_sum/2
        mean_event = mean_accepted_sigma*luminosity_scale
        tau = MASSES/(2*sigma_p*sigma_p)
        pulse_time = 1/(2*PROBE_ENERGY_WIDTH)
        spread_bounds = (pulse_time/tau)**2/(BEAM_RATIO**2+1)
        max_spread = float(max(spread_bounds))
        assert 0 < mean_event < 1
        assert 0 < conditional_error < .5
        output.append({'sigma_p_over_epsilon': zeta, 'detector_resolution_over_epsilon': resolution,
                       'orders_used': list(orders), 'successive_quadrature_scaled_differences': differences,
                       'last_conditional_error_change': float(abs(old_error-new_error)),
                       'accepted_sigma_over_reference_plus_minus': [float(plus[0]), float(minus[0])],
                       'accepted_posterior_plus_prior_half': float(plus[0]/rate_sum),
                       'error_given_plus_accepted': float(plus[1]/plus[0]),
                       'error_given_minus_accepted': float(minus[1]/minus[0]),
                       'error_conditioned_on_accepted_event': float(conditional_error),
                       'accepted_F_fractions_plus_minus': [float(plus[2]/plus[0]), float(minus[2]/minus[0])],
                       'mean_outgoing_energy_over_epsilon_plus_minus': [float(plus[3]/plus[0]), float(minus[3]/minus[0])],
                       'sigma_x_times_V': float(sigma_x), 'probe_beam_width_times_V': float(BEAM_RATIO*sigma_x),
                       'single_probe_accepted_probability_source_model': float(mean_event),
                       'target_marginal_spreading_indicator': max_spread,
                       'paraxial_angular_variance': float(2*(zeta/BEAM_RATIO)**2),
                       'fresh_preparation_energy_per_accepted_event_over_V_source_model': float(EPSILON/mean_event),
                       'measurement_complete': False})
    return {'epsilon_over_V': float(EPSILON), 'probe_energy_width_over_V': float(PROBE_ENERGY_WIDTH),
            'fixed_classification_threshold_over_V': float(np.mean(MASSES*EPSILON/(MASSES+EPSILON))),
            'reference_sigma_times_V_squared': float(SIGMA_REF), 'rows': output,
            'resource_scope': 'single collision, supplied Gaussian source and response; geometric paraxial estimate, no autonomous detector',
            'phase_of_full_scattering_state_or_non_event_instrument': 'not calculated'}


def main():
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/73_*.md'))
    digest = hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 73.2')[0].encode()).hexdigest()
    assert digest == PREREG
    report = {'schema_version': 1, 'candidate': 'CE-UR4-R2', 'scientific_success': False,
              'full_joint_rmse': None, 'preregistration_sha256': digest,
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), folder/'ce_singlet_record_observability.py',
                                 folder/'ce_supersymmetric_record.py', folder/'ce_simple_group_record.py']},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sympy': sy.__version__},
              'vertices': vertices(), 'two_channel_cross_sections': analytic_checks(),
              'conservation_and_domain': kinematic_domain_checks(),
              'source_overlap': source_overlap_checks(), 'packet_readout': packets(),
              'sources': ['https://arxiv.org/abs/hep-ph/9709356',
                          'https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf'],
              'new_observational_inputs_or_fits': False,
              'result': 'CONDITIONAL_RECOIL_INFORMATION_WITH_RETAINED_MASS_SPECIES; DETECTOR_AND_ACTUAL_EVENT_OPEN',
              'open_gates': ['physical packet source and detector dynamics',
                             'full coherent scattering and non-event instrument',
                             'repeatable protocol with momentum and supermultiplet backaction',
                             'asynchronous actual records and common gravity',
                             'frozen joint predictions and covariance']}
    target = folder/'ce_record_recoil_readout.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'candidate': report['candidate'], 'cross_sections': report['two_channel_cross_sections'],
                      'domain': report['conservation_and_domain'], 'packet_readout': report['packet_readout']}, indent=2))


if __name__ == '__main__':
    main()
