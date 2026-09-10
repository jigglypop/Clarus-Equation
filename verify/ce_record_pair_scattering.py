"""Chapter 72: action-derived tree scattering and a conditional readout gate.

The Cayley matrix below is a stated unitary completion of the leading
partial wave, not the complete CE-UR4 S matrix or a macroscopic detector.
"""
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.integrate import quad_vec
from scipy.special import erf
import sympy as sy

from ce_singlet_record_observability import build_model
from ce_supersymmetric_record import error, f_terms


TOL = 1e-8
PREREG = 'c88960a499eaa9df2b861857ec65782196178ed3c80d24c61f488b3dfc6bf6a7'
M0 = 1e-4
HEAVY = 1.
KAPPA = .05
MASSES = M0*np.array([5+np.pi/2, 5-np.pi/2])


def vertex_and_symmetry():
    model = build_model(M0)
    _, _, reps, mass, cubic, vacuum = model
    _, w = f_terms(vacuum, mass, cubic)
    m = -w[11, 11].real
    source = cubic[11, 35, 35].real/2
    yukawa = cubic[11, 75, 80].real
    assert abs(m-HEAVY) < TOL
    assert max(abs(source-1/(2*np.sqrt(15))),
               abs(yukawa-3/(2*np.sqrt(15))), abs(source*yukawa/m-KAPPA)) < TOL
    # Independently differentiate the effective superpotential.
    a, b, M, k, s, x, y, h, hb = sy.symbols('a b M k s S1 S2 H Hbar')
    weff = k*(x*x+y*y)*h*hb
    derivatives = sy.Matrix([[sy.diff(weff, h, hb, r, q) for q in [x, y]] for r in [x, y]])
    assert derivatives == 2*k*sy.eye(2)
    uv = 2*M*a*b/(M*M-s)
    assert sy.simplify(uv.subs(s, 0)-2*a*b/M) == 0
    # Only the physical Sigma singlet couples two light singlet scalars.
    unwanted = np.delete(cubic[:24, 35, 35], 11)
    assert error(unwanted) < TOL
    assert error(cubic[11, 75:77, 80:82]-yukawa*np.eye(2)) < TOL
    weak_pair = np.array([1, 0, 0, 1])/np.sqrt(2)
    weak_generators = [np.array([[0, 1], [1, 0]])/2,
                       np.array([[0, -1j], [1j, 0]])/2, np.diag([1, -1])/2]
    weak_error = max(error((np.kron(t, np.eye(2))-np.kron(np.eye(2), t.T))@weak_pair)
                     for t in weak_generators)
    assert weak_error < TOL
    # Independent S_+ and S_- sign flips become +/- record exchange.
    permutation = np.r_[0:24, 48:72, 24:48, 72:82]
    sym_errors = []
    for sign in [1, -1]:
        signs = np.ones(82)
        signs[24:72] = sign
        mm = mass[np.ix_(permutation, permutation)]*signs[:, None]*signs[None, :]
        yy = cubic[np.ix_(permutation, permutation, permutation)]
        yy = yy*signs[:, None, None]*signs[None, :, None]*signs[None, None, :]
        tt = reps[:, permutation, :][:, :, permutation]*signs[None, :, None]*signs[None, None, :]
        sym_errors += [error(mm-mass), error(yy-cubic), error(tt-reps)]
    assert max(sym_errors) < TOL
    return {'heavy_singlet_mass': m, 'Sigma_Q_source': source,
            'Sigma_Higgsino_yukawa': yukawa, 'kappa_from_UV': source*yukawa/m,
            'effective_vertex': '2 kappa delta_rs',
            'UV_tree_factor_relative_to_contact': '1/(1-s/M_Sigma^2)',
            'independent_mass_sign_symmetry_error': max(sym_errors),
            'final_weak_singlet_error': weak_error, 'final_hypercharge_sum': 0,
            'scope': 'two scalar particles to two conjugate Higgsinos; no incoming antiparticle'}


def spinor_bracket(energy, cosine, azimuth):
    c, s = np.sqrt((1+cosine)/2), np.sqrt((1-cosine)/2)
    phase = np.exp(1j*azimuth)
    u = np.sqrt(2*energy)*np.array([c, phase*s])
    v = np.sqrt(2*energy)*np.array([s, -phase*c])
    return u[0]*v[1]-u[1]*v[0]


def cross_sections():
    nodes, weights = np.polynomial.legendre.leggauss(16)
    azimuths = 2*np.pi*np.arange(9)/9
    results, relative_errors, conservation_errors = [], [], []
    for ecm in [.01, .03, .1]:
        s = ecm**2
        propagator = 1/(1-s/HEAVY**2)
        for m in MASSES:
            beta = np.sqrt(1-4*m*m/s)
            pi, pf = ecm*beta/2, ecm/2
            angular = 0.
            for cosine, weight in zip(nodes, weights):
                for azimuth in azimuths:
                    bracket = spinor_bracket(ecm/2, cosine, azimuth)
                    relative_errors.append(abs(abs(bracket)**2/s-1))
                    amplitude_squared = 2*abs(2*KAPPA*propagator*bracket)**2
                    angular += weight*(2*np.pi/len(azimuths))*amplitude_squared/(64*np.pi**2*s*beta)
                    n = np.array([np.sqrt(1-cosine*cosine)*np.cos(azimuth),
                                  np.sqrt(1-cosine*cosine)*np.sin(azimuth), cosine])
                    incoming = np.array([[ecm/2, 0, 0, pi], [ecm/2, 0, 0, -pi]])
                    outgoing = np.array([np.r_[ecm/2, pf*n], np.r_[ecm/2, -pf*n]])
                    conservation_errors += [error(incoming.sum(axis=0)-outgoing.sum(axis=0)),
                                            error(incoming[:, 0]**2-np.sum(incoming[:, 1:]**2, axis=1)-m*m),
                                            error(outgoing[:, 0]**2-np.sum(outgoing[:, 1:]**2, axis=1))]
            expected = KAPPA**2*propagator**2/(2*np.pi*beta)
            amp2 = 8*KAPPA**2*s*propagator**2
            t_width = 4*pi*pf
            mandelstam = amp2*t_width/(64*np.pi*s*pi*pi)
            relative_errors += [abs(angular/expected-1), abs(mandelstam/expected-1)]
            # Normalized J=0 channel: 1/sqrt(2) for identical incoming bosons,
            # then sqrt(2) for the two orthogonal final weak components.
            partial = KAPPA*ecm*np.sqrt(beta)*propagator/(8*np.pi)
            from_partial = 8*np.pi/(s*beta*beta)*abs(2j*partial)**2
            relative_errors.append(abs(from_partial/expected-1))
            results.append({'ecm_over_V': ecm, 'mass_over_V': float(m),
                            'sigma_times_V_squared': float(expected),
                            'sigma_v_Moller_times_V_squared': float(2*beta*expected),
                            'contact_amplitude_relative_correction': float(propagator-1),
                            'contact_cross_section_relative_correction': float(propagator**2-1)})
    assert max(relative_errors+conservation_errors) < TOL
    return {'rows': results, 'independent_relative_error': max(relative_errors),
            'four_momentum_and_mass_shell_error': max(conservation_errors),
            'formula_sigma_v': 'kappa^2/pi/(1-s/M_Sigma^2)^2',
            'final_weak_channels': 2, 'final_identical_particle_factor': 1,
            'initial_pair_rate_factor': '1/2 in a gas rate n_alpha^2 sigma_v, not in sigma',
            'absolute_physical_rate_predicted': False}


def relational_kernel():
    I = np.eye(2)
    X, Z = np.array([[0, 1], [1, 0]]), np.diag([1, -1])
    hadamard = np.array([[1, 1], [1, -1]])/np.sqrt(2)
    change = np.kron(hadamard, hadamard)
    rel = (np.eye(4)-np.kron(Z, Z))/2
    phi_minus = np.array([1, 0, 0, -1])/np.sqrt(2)
    psi_minus = np.array([0, 1, -1, 0])/np.sqrt(2)
    annihilator = np.array([1, 0, 0, 1])
    hpair = np.kron(5*I+np.pi*X/2, I)+np.kron(I, 5*I+np.pi*X/2)
    # The two mode labels do not turn identical bosons into distinguishable
    # particles. Antisymmetric internal states carry odd relative orbitals.
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    symmetric, antisymmetric = (np.eye(4)+swap)/2, (np.eye(4)-swap)/2
    angular_embedding = np.kron(symmetric, np.array([[1], [0]]))
    angular_embedding += np.kron(antisymmetric, np.array([[0], [1]]))
    bose_error = max(error(angular_embedding.conj().T@angular_embedding-np.eye(4)),
                     error(np.kron(swap, Z)@angular_embedding-angular_embedding))
    assert bose_error < TOL
    assert np.linalg.matrix_rank(annihilator[None, :]) == 1
    assert error(annihilator@phi_minus) < TOL and error(annihilator@psi_minus) < TOL
    assert error(hpair@phi_minus-10*phi_minus) < TOL
    assert error(hpair@psi_minus-10*psi_minus) < TOL
    assert abs(np.vdot(phi_minus, rel@phi_minus)) < TOL
    assert abs(np.vdot(psi_minus, rel@psi_minus)-1) < TOL
    for state in [phi_minus, psi_minus]:
        mass_state = change@state
        assert error(mass_state[[0, 3]]) < TOL
        assert error(np.kron(Z, Z)@mass_state+mass_state) < TOL
    return {'local_annihilation_tensor_rank': 1,
            'Bose_symmetric_orbital_embedding_error': bose_error,
            'dark_states_in_copy_basis': ['(|00>-|11>)/sqrt(2)', '(|01>-|10>)/sqrt(2)'],
            'dark_relational_eigenvalues': [0, 1],
            'mass_basis_content': 'one S_plus and one S_minus; odd under both species parities',
            'optimal_dark_state_error_equal_priors': .5,
            'result': 'REJECTED_UNIVERSAL_RELATIONAL_READOUT_BY_RECORD_FREE_ANNIHILATION',
            'scope': 'no visible-only final state in the parity-preserving theory; radiation with record remnants is not excluded'}


def cayley_completion(energy):
    """A conditional completion matching the action's leading J=0 amplitudes."""
    beta = np.sqrt(1-4*MASSES**2/energy**2)
    q = KAPPA*energy*np.sqrt(beta)/(8*np.pi*(1-energy**2/HEAVY**2))
    kmatrix = np.zeros((5, 5))
    kmatrix[0, 4] = kmatrix[4, 0] = q[0]
    kmatrix[3, 4] = kmatrix[4, 3] = q[1]
    smatrix = np.linalg.solve((np.eye(5)-1j*kmatrix).T, (np.eye(5)+1j*kmatrix).T).T
    return smatrix, q


def packet_checks():
    p = 50*M0
    centers = 2*np.sqrt(p*p+MASSES**2)
    difference = float(centers[0]-centers[1])
    lo, hi = float(2*MASSES[0]), .05
    I, X, Z = np.eye(2), np.array([[0, 1], [1, 0]]), np.diag([1, -1])
    change = np.kron(np.array([[1, 1], [1, -1]])/np.sqrt(2),
                     np.array([[1, 1], [1, -1]])/np.sqrt(2))
    records, completion_errors = [], []
    for width_ratio in [.125, .5, 2., 8.]:
        width = width_ratio*difference
        means = np.array([centers[0], centers.mean(), centers.mean(), centers[1]])
        norms = .5*(erf((hi-means)/(np.sqrt(2)*width))-erf((lo-means)/(np.sqrt(2)*width)))
        def packets(energy):
            return np.exp(-(energy-means)**2/(4*width*width))/(np.sqrt(np.sqrt(2*np.pi)*width)*np.sqrt(norms))
        left, right = max(lo, means.min()-12*width), min(hi, means.max()+12*width)
        midpoint = means.mean()
        scale = float(abs(2*cayley_completion(midpoint)[1][0])**2)
        assert scale > 0
        def quantities(energy):
            smat, q = cayley_completion(energy)
            f = packets(energy)
            selected = smat[:, :4]*f[None, :]
            click = np.outer(selected[4].conj(), selected[4])
            no_click = selected[:4].conj().T@selected[:4]
            overlap = f[0]*f[3]
            born_row = np.zeros(4, complex)
            born_row[[0, 3]] = 2j*q*f[[0, 3]]
            born_click = np.outer(born_row.conj(), born_row)
            return np.r_[click.ravel()/scale, no_click.ravel(), f*f, overlap,
                         born_click.ravel()/scale]
        points = sorted(set(float(x) for mean in means for x in [mean-4*width, mean, mean+4*width]
                            if left < x < right))
        integral, _ = quad_vec(quantities, left, right, points=points, epsabs=1e-12, epsrel=1e-12)
        nodes, weights = np.polynomial.legendre.leggauss(160)
        nodes = (right+left)/2+(right-left)*nodes/2
        weights = weights*(right-left)/2
        independent = sum(weight*quantities(e) for e, weight in zip(nodes, weights))
        relative_error = error(integral-independent)
        assert relative_error < TOL
        click = integral[:16].reshape(4, 4)*scale
        no_click = integral[16:32].reshape(4, 4)
        normalized = integral[32:36].real
        born_click = integral[37:53].reshape(4, 4)*scale
        completion_errors += [error(click+no_click-np.eye(4)), error(normalized-1)]
        eigenvalues = np.linalg.eigvalsh(click/scale)
        assert eigenvalues.min() > -TOL and eigenvalues.max() > .1
        click_copy = change@click@change
        same, different = click_copy[0, 0].real, click_copy[1, 1].real
        contrast = float((same-different)/(same+different))
        born_copy = change@born_click@change
        born_same, born_different = born_copy[0, 0].real, born_copy[1, 1].real
        born_contrast = float((born_same-born_different)/(born_same+born_different))
        # Dephasing the two mass channels removes c but leaves their rates.
        dephased = change@np.diag(np.diag(click))@change
        assert abs(dephased[0, 0]-dephased[1, 1])/scale < TOL
        # A mixed copy reference is invariant and has no absolute-label contrast.
        abs0 = (click_copy[0, 0]+click_copy[1, 1])/2
        abs1 = (click_copy[2, 2]+click_copy[3, 3])/2
        assert abs(abs0-abs1)/scale < TOL
        exact_overlap = np.exp(-difference*difference/(8*width*width))
        exact_overlap *= (.5*(erf((hi-midpoint)/(np.sqrt(2)*width))-erf((lo-midpoint)/(np.sqrt(2)*width)))
                          /np.sqrt(norms[0]*norms[3]))
        assert abs(integral[36]-exact_overlap) < TOL
        # Positive quadrature Kraus operators, with the output energy traced.
        choi = np.zeros((20, 20), complex)
        max_unitarity, max_energy_error, max_born_change, shell_error = 0., 0., 0., 0.
        for energy, weight in zip(nodes, weights):
            smat, q = cayley_completion(energy)
            selected = smat[:, :4]*packets(energy)[None, :]
            kno, kyes = selected.copy(), selected.copy()
            kno[4] = 0
            kyes[:4] = 0
            for kk in [kno, kyes]:
                vec = np.sqrt(weight)*kk.ravel()
                choi += np.outer(vec, vec.conj())
            max_unitarity = max(max_unitarity, error(smat.conj().T@smat-np.eye(5)))
            max_energy_error = max(max_energy_error, error(smat.conj().T@(energy*np.eye(5))@smat-energy*np.eye(5)))
            max_born_change = max(max_born_change, abs(smat[4, 0]/(2j*q[0])-1))
            # Each coupled channel has different on-shell momenta, but the
            # same total four-momentum. No external time pulse supplies energy.
            for m1, m2 in [(MASSES[0], MASSES[0]), (MASSES[0], MASSES[1]),
                           (MASSES[1], MASSES[0]), (MASSES[1], MASSES[1]), (0., 0.)]:
                e1 = (energy**2+m1**2-m2**2)/(2*energy)
                e2 = energy-e1
                pcm = np.sqrt((energy**2-(m1+m2)**2)*(energy**2-(m1-m2)**2))/(2*energy)
                shell_error = max(shell_error, abs(e1**2-pcm**2-m1**2)/energy**2,
                                  abs(e2**2-pcm**2-m2**2)/energy**2)
        assert max(max_unitarity, max_energy_error, shell_error, -np.linalg.eigvalsh(choi).min()) < TOL
        records.append({'sigma_E_over_delta_E': width_ratio,
                        'sigma_E_over_V': width, 'normalized_packet_overlap': float(exact_overlap),
                        'Born_annihilation_contrast': born_contrast,
                        'Born_click_probability_same_product': float(born_same),
                        'Born_click_probability_different_product': float(born_different),
                        'Born_one_pass_binary_error_equal_priors': float((1-abs(born_same-born_different))/2),
                        'conditional_completion_contrast': contrast,
                        'click_probability_same_product': float(same),
                        'click_probability_different_product': float(different),
                        'mass_channel_click_Gram_eigenvalues_scaled': eigenvalues.tolist(),
                        'quad_vs_Gauss_scaled_error': relative_error,
                        'conditional_completion_unitarity_error': max_unitarity,
                        'conditional_completion_energy_error': max_energy_error,
                        'all_channels_mass_shell_relative_error': shell_error,
                        'conditional_completion_vs_Born_relative_amplitude': max_born_change,
                        'conditional_instrument_Choi_minimum': float(np.linalg.eigvalsh(choi).min())})
    assert max(completion_errors) < TOL
    assert abs(np.log(records[0]['normalized_packet_overlap'])+8) < TOL
    assert 0 < records[0]['Born_annihilation_contrast'] < .001
    assert records[-1]['Born_annihilation_contrast'] > .99
    return {'incoming_momentum_over_V': p, 'same_mass_pair_centers_over_V': centers.tolist(),
            'center_energy_difference_over_V': difference, 'energy_band_over_V': [lo, hi],
            'rows': records, 'conditional_CP_completeness_error': max(completion_errors),
            'packet_preparation_derived': False,
            'Cayley_completion_status': 'unitary consistency example matching the tree partial wave; omitted elastic channels and dispersive terms are not derived',
            'physical_macroscopic_record_or_full_S_matrix': False}


def main():
    folder = Path(__file__).resolve().parent
    chapter = next((folder.parent/'paper').glob('06_*/72_*.md'))
    digest = hashlib.sha256(chapter.read_text(encoding='utf-8').split('## 72.2')[0].encode()).hexdigest()
    assert digest == PREREG
    report = {'schema_version': 1, 'candidate': 'CE-UR4-R1',
              'scientific_success': False, 'full_joint_rmse': None,
              'preregistration_sha256': digest,
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), folder/'ce_singlet_record_observability.py',
                                 folder/'ce_supersymmetric_record.py', folder/'ce_simple_group_record.py']},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'sympy': sy.__version__},
              'vertex': vertex_and_symmetry(), 'cross_sections': cross_sections(),
              'relational_gate': relational_kernel(), 'packets': packet_checks(),
              'sources': ['https://arxiv.org/abs/hep-ph/9709356',
                          'https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf'],
              'new_observational_inputs_or_fits': False,
              'open_gates': ['packet and reference preparation from a physical source',
                             'full scattering channels, radiative corrections, and actual detector',
                             'non-annihilating relational readout or a new record observable',
                             'asynchronous actual events with preserved state and common gravity',
                             'frozen joint predictions and covariance']}
    target = folder/'ce_record_pair_scattering.json'
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'candidate': report['candidate'], 'vertex': report['vertex'],
                      'cross_section_error': report['cross_sections']['independent_relative_error'],
                      'relational_gate': report['relational_gate'], 'packets': report['packets']}, indent=2))


if __name__ == '__main__':
    main()
