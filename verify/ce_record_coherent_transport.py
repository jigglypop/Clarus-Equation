"""Chapter 75: coherent pair preparation and all tree two-body record readouts.

Source/detector geometry and response remain supplied. No complete S matrix,
actual event law, finite-packet propagation, or joint empirical success.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.special import ndtr
import sympy as sy

from ce_record_pair_preparation import (
    MASSES, HEAVY, KAPPA, E0, EPSILON, SOURCE_ANGLE, DETECTOR_ANGLE,
    SOURCE_FRACTION, DETECTOR_SOLID_ANGLE, TRANSFER_SCALE,
    energy_rule, cone_rule, source_sigma, vertices,
)


PREREG = '566b10a9f56d149c7781b228b495b98df8668bc72c115b1251d4af6c0f05d18d'
TOL, QUAD_TOL = 1e-8, 1e-6
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
ZERO, EYE = np.zeros((2, 2)), np.eye(2)
GAMMA = np.array([np.block([[ZERO, EYE], [EYE, ZERO]])]
                 +[np.block([[ZERO, p], [-p, ZERO]]) for p in PAULI])
PL, PR = np.diag([1, 1, 0, 0]), np.diag([0, 0, 1, 1])


def maximum(a):
    return float(np.max(np.abs(a)))


def dot(p, q):
    return p[0]*q[0]-np.dot(p[1:], q[1:])


def slash(p):
    return p[0]*GAMMA[0]-np.einsum('i,ijk->jk', p[1:], GAMMA[1:])


def spinors(momentum, mass):
    energy = np.sqrt(mass*mass+np.dot(momentum, momentum))
    ps = np.einsum('i,ijk->jk', momentum, PAULI)
    u = np.vstack([(energy+mass)*EYE-ps, (energy+mass)*EYE+ps])/np.sqrt(2*(energy+mass))
    v = 1j*GAMMA[2]@u.conj()
    return u, v


def bar(u):
    return u.conj().T@GAMMA[0]


def scalar_vertices():
    original = vertices()
    z, zb, x, xb, h, hc, hb, hbc = sy.symbols('z zb x xb h hc hb hbc')
    a, b, m, M, t = sy.symbols('a b m M t', real=True)
    fields = [z, zb, x, xb, h, hc, hb, hbc]
    conj = lambda f: f.xreplace(dict(zip(fields, [zb, z, xb, x, hc, h, hbc, hb])))
    F = [-M*z+a*x*x+b*h*hb, (m+2*a*z)*x, b*z*hb, b*z*h]
    lagrangian = -sy.expand(sum(f*conj(f) for f in F))
    zero = dict.fromkeys(fields, 0)
    vertex = lambda *legs: sy.diff(lagrangian, *legs).subs(zero)
    contact = vertex(xb, xb, h, hb)
    rr, probe = vertex(z, xb, xb), vertex(zb, h, hb)
    same = vertex(z, x, xb)
    flip = sy.factor(contact+rr*probe/(M*M-t))
    retained = sy.factor(same*probe/(M*M-t))
    assert contact == -2*a*b and rr == 2*a*M and probe == b*M
    assert sy.simplify(flip-2*a*b*t/(M*M-t)) == 0
    assert sy.simplify(retained+2*a*b*M*m/(M*M-t)) == 0
    # Fermion block inverse gives a mass numerator and a momentum numerator.
    q0, q1 = sy.symbols('q0 q1', real=True)
    q = sy.Matrix([[q0, q1], [-q1, -q0]])
    assert sy.simplify((M*sy.eye(2)+q)*(M*sy.eye(2)-q)
                       -(M*M-q0*q0+q1*q1)*sy.eye(2)) == sy.zeros(2)
    return {'original_tensor_vertices': original, 'contact_Sbar_H_to_S_antiHbar': str(contact),
            'contact_plus_exchange': str(flip), 'retained_scalar_amplitude': str(retained),
            'fermion_propagator_numerators': 'M and slash(q); relative signs from eliminated heavy chiral action',
            'outgoing_visible_weak_components_for_fixed_H': 1}


def source_density(p, mass):
    u, _ = spinors(p, mass)
    _, vpartner = spinors(-p, mass)
    cs = np.array([[1, -mass/HEAVY], [-mass/HEAVY, 0]])
    cf = bar(u)@PL@vpartner/HEAVY
    return cs@cs.T, cf@cf.conj().T, cs, cf


def read_effects(p, k, pfinal, kfinal, mass):
    q = pfinal-p
    t = dot(q, q)
    g = 2*KAPPA/(HEAVY*(1-t/HEAVY**2))
    uin, vin = spinors(p[1:], mass)
    uout, _ = spinors(pfinal[1:], mass)
    _, vh = spinors(kfinal[1:], 0.)
    sb = g*np.array([[-HEAVY*mass, t], [0., -HEAVY*mass]])
    sf = np.stack([-g*HEAVY*bar(uout)@PL@vh,
                   g*bar(uout)@slash(q)@PL@vh], axis=-1).reshape(4, 2)
    fb = -g*HEAVY*bar(uout)@PL@uin
    fbar_scalar = (-g*HEAVY*bar(vin)@PL@vh).T
    fscalar = (g*bar(vin)@slash(q)@PL@vh).T
    scalar_b = sb.conj().T@sb
    scalar_f = sf.conj().T@sf
    fermion_b = fb.conj().T@fb
    fermion_f = fbar_scalar.conj().T@fbar_scalar+fscalar.conj().T@fscalar
    return scalar_b, scalar_f, fermion_b, fermion_f


def independent_checks():
    errors, eigenvalues, coherence_rows, source_rows = [], [], [], []
    directions = np.array([[0, 0, 1], [1, 2, 3], [-2, 1, -1]], float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for mass in MASSES:
        for energy in [E0, .01]:
            ep = energy/2
            pmag = np.sqrt(ep*ep-mass*mass)
            beta = pmag/ep
            polarization = 2*beta/(1+beta*beta)
            qf = (energy*energy-2*mass*mass)/HEAVY**2
            for n in directions:
                p = np.r_[ep, pmag*n]
                rs, rf, cs, cf = source_density(p[1:], mass)
                expected_rf = qf/2*(EYE+polarization*np.einsum('i,ijk->jk', n, PAULI))
                errors += [maximum((rf-expected_rf)/qf), abs(np.trace(rs).real+np.trace(rf).real-(1+energy*energy/HEAVY**2))]
                # Compare partial trace against the full pair vector.
                cp = np.zeros((4, 4), complex)
                cp[:2, :2], cp[2:, 2:] = cs, cf
                pair = cp.ravel()
                reduced = np.trace(np.outer(pair, pair.conj()).reshape(4, 4, 4, 4), axis1=1, axis2=3)
                errors.append(maximum(reduced-cp@cp.conj().T))
                for rho in [rs, rf, reduced]:
                    eigenvalues.append(float(np.min(np.linalg.eigvalsh(rho))/np.trace(rho).real))
                for ni in directions:
                    k = np.r_[EPSILON, EPSILON*ni]
                    for no in directions:
                        denominator = ep-np.dot(p[1:], no)+EPSILON*(1-np.dot(ni, no))
                        eout = EPSILON*(ep-np.dot(p[1:], ni))/denominator
                        ko = np.r_[eout, eout*no]
                        po = p+k-ko
                        s, t = dot(p+k, p+k), dot(k-ko, k-ko)
                        errors.append(abs((dot(po, po)-mass*mass)/s))
                        sb, sf, fb, ff = read_effects(p, k, po, ko, mass)
                        g2 = (2*KAPPA/(HEAVY*(1-t/HEAVY**2)))**2
                        expected_s = g2*np.diag([HEAVY**2*s, HEAVY**2*mass*mass-t*(s-mass*mass)])
                        errors.append(maximum((sb+sf-expected_s)/(g2*HEAVY**2*s)))
                        # Polarized invariant current is independent of the external spinor construction.
                        ms = np.r_[pmag, ep*n]
                        kl, kr = p-polarization*ms, p+polarization*ms
                        expected_b = qf*g2*HEAVY**2*dot(po, kl)
                        expected_f = qf*g2*(HEAVY**2*dot(ko, kl)-t*dot(k, kr))
                        actual_b, actual_f = np.trace(fb@rf).real, np.trace(ff@rf).real
                        scale = qf*g2*HEAVY**2*s
                        errors += [abs(actual_b-expected_b)/scale, abs(actual_f-expected_f)/scale]
                        unpolarized = np.trace(fb+ff).real/2
                        errors.append(abs(unpolarized-np.trace(sb+sf).real/2)/(g2*HEAVY**2*s))
                        interference_b = np.trace(sb@(rs-np.diag(np.diag(rs)))).real
                        interference_f = np.trace(sf@(rs-np.diag(np.diag(rs)))).real
                        errors += [abs(interference_b-2*g2*mass*mass*t)/(g2*HEAVY**2*s),
                                   abs(interference_b+interference_f)/(g2*HEAVY**2*s)]
                        for effect in [sb, sf, fb, ff, sb+sf, fb+ff]:
                            eigenvalues.append(float(np.min(np.linalg.eigvalsh(effect))/(g2*HEAVY**2*s)))
                        # Completeness and free Dirac equations, including charge conjugation.
                        u, v = spinors(p[1:], mass)
                        errors += [maximum((u@bar(u)-slash(p)-mass*np.eye(4))/ep),
                                   maximum((v@bar(v)-slash(p)+mass*np.eye(4))/ep)]
                if np.allclose(n, [0, 0, 1]):
                    source_rows.append({'mass_over_V': float(mass), 'ecm_over_V': energy,
                                        'scalar_density_relative_to_SS': rs.real.tolist(),
                                        'fermion_source_trace_relative_to_SS': qf,
                                        'fermion_helicity_polarization': polarization})
                    # One concrete detector-sensitive interference example.
                    k = np.array([EPSILON, EPSILON, 0, 0])
                    eo = EPSILON*ep/(ep-pmag+EPSILON)
                    ko = np.array([eo, 0, 0, eo])
                    sb, sf, _, _ = read_effects(p, k, p+k-ko, ko, mass)
                    cb = np.trace(sb@(rs-np.diag(np.diag(rs)))).real
                    cfraction = cb/np.trace(sb@rs).real
                    coherence_rows.append({'mass_over_V': float(mass), 'ecm_over_V': energy,
                                           'scalar_visible_interference_fraction': float(cfraction),
                                           'inclusive_scalar_interference': float(np.trace((sb+sf)@(rs-np.diag(np.diag(rs)))).real)})
    assert max(errors) < TOL, max(errors)
    assert min(eigenvalues) > -TOL, min(eigenvalues)
    # Non-numerical cancellation of the two scalar-sector off-diagonal effects.
    M, m, t, s, g = sy.symbols('M m t s g', real=True)
    rb = g*g*sy.Matrix([[M*M*m*m, -M*m*t], [-M*m*t, M*M*m*m+t*t]])
    rf = g*g*sy.Matrix([[M*M*(s-m*m), M*m*t], [M*m*t, -t*(s-m*m+t)]])
    exact = g*g*sy.diag(M*M*s, M*M*m*m-t*(s-m*m))
    assert sy.simplify(rb+rf-exact) == sy.zeros(2)
    return {'max_scaled_independent_error': max(errors),
            'smallest_scaled_effect_or_density_eigenvalue': min(eigenvalues),
            'source_rows': source_rows, 'channel_sensitive_interference': coherence_rows,
            'symbolic_inclusive_scalar_effect': str(exact),
            'unpolarized_fermion_equals_mean_scalar_responses': True,
            'scope': 'positive selected-event effects, not a normalized full instrument'}


def fold(mass, species, direction, resolution, orders, threshold):
    ne, na, nphi = orders
    energies, ew = energy_rule(ne)
    ns, sw = cone_rule(SOURCE_ANGLE, 'z', na, nphi)
    no, dw = cone_rule(DETECTOR_ANGLE, direction, na, nphi)
    result = np.zeros((6, 3))
    max_shell, max_s = 0., 0.
    for energy, weight in zip(energies, ew):
        ep = energy/2
        pmag = np.sqrt(ep*ep-mass*mass)
        beta, qf = pmag/ep, (energy*energy-2*mass*mass)/HEAVY**2
        polarization = 2*beta/(1+beta*beta)
        px = pmag*ns[:, 0, None]
        cosine = ns@no.T
        denominator = ep-pmag*cosine+EPSILON*(1-no[None, :, 0])
        eout = EPSILON*(ep-px)/denominator
        s = mass*mass+2*EPSILON*(ep-px)
        t = -2*EPSILON*eout*(1-no[None, :, 0])
        delta = s-mass*mass
        kin = EPSILON*(ep-px)
        spin = EPSILON*(pmag-ep*ns[:, 0, None])
        kl, kr = kin-polarization*spin, kin+polarization*spin
        kout_l = eout*(ep-polarization*pmag-(pmag-polarization*ep)*cosine)
        baseline = np.broadcast_to(HEAVY**2*s, t.shape)
        extra_scalar = mass*mass*(s+mass*mass)-mass*mass/HEAVY**2*t*delta
        fermion = qf*(HEAVY**2*(mass*mass+kl)-t*kr)
        fermion_unpolarized = qf*(HEAVY**2*(mass*mass+kin)-t*kin)
        scalar_visible = (mass*mass*(HEAVY**2+2*mass*mass+2*t+t*t/HEAVY**2)
                          +qf*HEAVY**2*(mass*mass+kl-kout_l))
        scalar_interference = 2*mass*mass*t
        kernels = np.array([baseline, extra_scalar, fermion, fermion_unpolarized,
                            scalar_visible, scalar_interference])
        g2 = (2*KAPPA/(HEAVY*(1-t/HEAVY**2)))**2
        measure = (weight*source_sigma(energy*energy, mass)[0]*SOURCE_FRACTION
                   *(1-beta*ns[:, 0, None])/(beta*ns[:, 2, None])
                   *g2/(64*np.pi**2*denominator**2)*DETECTOR_SOLID_ANGLE
                   *sw[:, None]*dw[None, :]/TRANSFER_SCALE)
        wrong = ndtr((eout-threshold)/(resolution*EPSILON)) if species == 0 else ndtr((threshold-eout)/(resolution*EPSILON))
        for j, factor in enumerate([np.ones_like(wrong), wrong, eout/EPSILON]):
            result[:, j] += np.sum(kernels*measure*factor, axis=(1, 2))
        assert np.min(kernels[:5]) >= 0
        po = pmag*ns[:, None, :]+np.array([EPSILON, 0, 0])-eout[:, :, None]*no
        efin = ep+EPSILON-eout
        max_shell = max(max_shell, maximum((efin*efin-np.sum(po*po, axis=2)-mass*mass)/s))
        max_s = max(max_s, float(np.max(np.sqrt(s))))
    assert max_shell < TOL and max_s < mass+2*MASSES[1]
    return result


def transport(previous):
    threshold = previous['pair_preparation']['fixed_classification_threshold_over_V']
    rows = []
    for direction in ['z', 'y']:
        for resolution in [.5, 1.]:
            trials = []
            for orders in [(32, 4, 8), (48, 6, 12)]:
                trials.append(np.array([fold(m, i, direction, resolution, orders, threshold)
                                        for i, m in enumerate(MASSES)]))
            delta = maximum(trials[1]-trials[0])
            correction_delta = maximum(trials[1][:, 1:]-trials[0][:, 1:])
            # Check the small correction itself, not only the much larger leading term.
            small_correction_relative_delta = maximum(
                (trials[1][:, 1:4, 0]-trials[0][:, 1:4, 0])/trials[1][:, 1:4, 0])
            assert delta < QUAD_TOL and small_correction_relative_delta < 1e-6
            values = trials[-1]
            full = values[:, :3].sum(axis=1)
            base = values[:, 0]
            unpol = values[:, 0]+values[:, 1]+values[:, 3]
            old = next(r for r in previous['conditional_source_to_readout']['rows']
                       if r['detector_direction'] == direction and r['detector_sigma_over_epsilon'] == resolution)
            old_rate = np.array(old['transfer_coefficient_times_V_fourth_plus_minus'])
            assert maximum(TRANSFER_SCALE*base[:, 0]/old_rate-1) < TOL
            err = full[:, 1].sum()/full[:, 0].sum()
            old_error = base[:, 1].sum()/base[:, 0].sum()
            err_unpol = unpol[:, 1].sum()/unpol[:, 0].sum()
            rows.append({'detector_direction': direction, 'detector_sigma_over_epsilon': resolution,
                         'full_two_body_transfer_times_V_fourth_plus_minus': (TRANSFER_SCALE*full[:, 0]).tolist(),
                         'relative_rate_increment_over_SS_plus_minus': ((values[:, 1, 0]+values[:, 2, 0])/base[:, 0]).tolist(),
                         'source_scalar_extra_relative_rate_plus_minus': (values[:, 1, 0]/base[:, 0]).tolist(),
                         'source_fermion_relative_rate_plus_minus': (values[:, 2, 0]/base[:, 0]).tolist(),
                         'full_error_conditioned_on_accepted_event': float(err),
                         'SS_only_error': float(old_error), 'error_shift_from_SS_only': float(err-old_error),
                         'error_if_source_fermions_unpolarized': float(err_unpol),
                         'relative_rate_bias_if_fermions_unpolarized_plus_minus': ((unpol[:, 0]-full[:, 0])/full[:, 0]).tolist(),
                         'accepted_plus_fraction': float(full[0, 0]/sum(full[:, 0])),
                         'outgoing_visible_fermion_fraction_plus_minus': (1-values[:, 4, 0]/full[:, 0]).tolist(),
                         'source_scalar_interference_contribution_to_visible_scalar_fraction_plus_minus': (values[:, 5, 0]/values[:, 4, 0]).tolist(),
                         'incoherent_scalar_source_inclusive_rate_difference': 0.,
                         'max_scaled_quadrature_difference': delta,
                         'max_scaled_nonleading_and_component_difference': correction_delta,
                         'small_correction_relative_quadrature_difference': small_correction_relative_delta,
                         'quadrature_orders': [[32, 4, 8], [48, 6, 12]]})
    return {'rows': rows, 'threshold_over_V': threshold, 'transfer_scale_times_V_fourth': TRANSFER_SCALE,
            'probability_formula': 'P_alpha ~= L_source N_H A_alpha, for dilute separated collisions',
            'geometry_and_detector': 'exactly chapter 74, central 8sigma conditional luminosity; equal scalar/fermion response',
            'units': 'A: V^-4; L_source and N_H: V^2 each',
            'scope': 'all light-record two-body source outputs and tree single readout; no extra radiation or full coherent propagation'}


def main():
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/75_*.md'))
    digest = hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 75.2')[0].encode()).hexdigest()
    assert digest == PREREG
    previous_path = folder/'ce_record_pair_preparation.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    for name, h in previous['source_hashes'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == h
    result = {'schema_version': 1, 'candidate': 'CE-UR4-P2', 'scientific_success': False,
              'full_joint_rmse': None, 'new_observational_inputs_or_fits': False,
              'preregistration_sha256': digest,
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), previous_path, folder/'ce_record_pair_preparation.py',
                                 folder/'ce_singlet_record_observability.py',
                                 folder/'ce_supersymmetric_record.py', folder/'ce_simple_group_record.py']},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sympy': sy.__version__},
              'vertices': scalar_vertices(), 'coherent_density_and_read_effects': independent_checks(),
              'transport': transport(previous),
              'sources': ['https://arxiv.org/abs/0812.1594', 'https://arxiv.org/abs/hep-ph/9709356',
                          'https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf'],
              'result': 'COHERENT_TWO_BODY_SOURCE_AND_TREE_READOUT; PHYSICAL_INSTRUMENT_OPEN',
              'open_gates': ['physical source and detector preparation', 'finite propagation and all competing radiation',
                             'complete non-event amplitude and instrument', 'repeatable asynchronous actual records',
                             'common gravity and frozen covariance-aware joint predictions']}
    (folder/'ce_record_coherent_transport.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'candidate': result['candidate'], 'independent': result['coherent_density_and_read_effects'],
                      'transport': result['transport']}, indent=2))


if __name__ == '__main__':
    main()
