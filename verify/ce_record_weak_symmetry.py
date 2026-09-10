"""Preregistered chapter 64: weak symmetry versus a neutral internal record.

Tests supplied scalar/Higgs/gauge representations, not physical SM matter.
The 18-component alternative adds degrees of freedom; no data are fitted.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.linalg import block_diag, expm

from ce_color_covariant_record import HF, GAMMA, REC, generators, setup


TOL = 1e-10
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)


def error(x):
    return float(np.max(np.abs(x)))


def comm(a, b):
    return a @ b - b @ a


def projector(v):
    return np.outer(v, v.conj())


def centralizer(matrices):
    n = len(matrices[0])
    gram = np.zeros((n*n, n*n), complex)
    for matrix in matrices:
        ad = np.kron(np.eye(n), matrix) - np.kron(matrix.T, np.eye(n))
        gram += ad.conj().T @ ad
    values, vectors = np.linalg.eigh(gram)
    selected = values < TOL
    assert values.min() > -TOL
    return (dict(dimension=int(selected.sum()), minimum_eigenvalue=float(values.min()),
                 first_positive_eigenvalue=float(values[~selected].min())), vectors[:, selected])


def old_model():
    initial, final, transition = setup()
    singlets = np.column_stack((initial, final))
    mass, record = HF + GAMMA*transition, np.kron(np.eye(4), REC)
    colors = []
    for g in generators():
        gs, ga = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
        gs[:3, :3], ga[1:, 1:] = g, -g.conj()
        colors.append(np.kron(gs, np.eye(4)) + np.kron(np.eye(4), ga))
    e = np.eye(16, dtype=complex)
    triplet = [e[:, 4*i] for i in range(3)]
    antitriplet = [e[:, i] for i in (13, 14, 15)]
    octet = [e[:, 4*i+j+1] for i in range(3) for j in range(3) if i != j]
    octet += [(e[:, 1]-e[:, 6])/np.sqrt(2),
              (e[:, 1]+e[:, 6]-2*e[:, 11])/np.sqrt(6)]
    outside = np.column_stack(triplet+antitriplet+octet)
    assert error(outside.conj().T@outside-np.eye(14)) < TOL
    assert error(outside@outside.conj().T+singlets@singlets.conj().T-np.eye(16)) < TOL
    return initial, singlets, outside, mass, record, colors


def frozen_symmetry(singlets, outside, mass, record, colors):
    mass_unit = mass/np.linalg.norm(mass, 2)
    color_info, _ = centralizer(colors)
    mass_info, _ = centralizer([mass_unit])
    joint_info, joint_null = centralizer(colors+[mass_unit])
    assert (color_info['dimension'], mass_info['dimension'], joint_info['dimension']) == (7, 84, 5)
    irreps = [outside[:, :3], outside[:, 3:6], outside[:, 6:]]
    blocks = [u@u.conj().T for u in irreps]
    blocks += [projector(singlets@np.array([1., sign])/np.sqrt(2)) for sign in (1, -1)]
    ranks = [3, 3, 8, 1, 1]
    basis = np.column_stack([(p/np.sqrt(rank)).reshape(-1, order='F')
                             for p, rank in zip(blocks, ranks)])
    analytic_errors = {
        'projector_resolution': error(sum(blocks)-np.eye(16)),
        'centralizer_basis_orthonormality': error(basis.conj().T@basis-np.eye(5)),
        'analytic_vs_gram_kernel': error(joint_null@joint_null.conj().T-basis@basis.conj().T),
        'all_centralizer_commutators': max(error(comm(a, b)) for a in blocks for b in blocks),
        'mass_commutators': max(error(comm(a, mass)) for a in blocks),
        'color_commutators': max(error(comm(a, c)) for a in blocks for c in colors),
    }
    assert max(analytic_errors.values()) < TOL
    weak = [singlets@(p/2)@singlets.conj().T for p in PAULI]
    su2_error = max(error(comm(weak[a], weak[b])-1j*weak[c])
                    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)))
    assert su2_error < TOL
    norms = [float(np.linalg.norm(comm(mass@mass, w), 'fro')) for w in weak]
    expected = [0., 10*GAMMA*np.sqrt(2), 10*GAMMA*np.sqrt(2)]
    assert error(np.array(norms)-expected) < TOL
    return dict(status='REJECTED_AS_UNBROKEN_WEAK_GAUGE_EXTENSION',
                color_representation=['3', 'bar3', '8', '1', '1'],
                color_centralizer=color_info, mass_centralizer=mass_info,
                joint_centralizer=joint_info, joint_compact_algebra='u(1)^5',
                analytic_errors=analytic_errors, singlet_su2_error=su2_error,
                squared_mass_weak_commutator_frobenius=norms,
                record_on_singlets=(singlets.conj().T@record@singlets).real.tolist())


def higgs_branches():
    # Independent adjoint (real-vector) and fundamental (complex-vector) kinetic terms.
    g2 = gy = v = 1.
    vev_adjoint = np.array([2*GAMMA, 0., 0.])
    derivatives_adjoint = [g2*np.cross(axis, vev_adjoint) for axis in np.eye(3)]
    derivatives_adjoint += [np.zeros(3)]
    adjoint_mass = np.array([[a@b for b in derivatives_adjoint] for a in derivatives_adjoint])
    adjoint_eigenvalues = np.linalg.eigvalsh(adjoint_mass)
    assert error(adjoint_mass-np.diag([0., 4*GAMMA**2, 4*GAMMA**2, 0.])) < TOL
    assert np.count_nonzero(adjoint_eigenvalues < TOL) == 2

    eta = np.ones(2, complex)/np.sqrt(2)
    h = v*eta/np.sqrt(2)
    t = [p/2 for p in PAULI]+[np.eye(2)/2]
    couplings = [g2, g2, g2, gy]
    derivatives = [g*generator@h for g, generator in zip(couplings, t)]
    gauge_mass = np.array([[2*np.vdot(a, b).real for b in derivatives] for a in derivatives])
    eigenvalues = np.linalg.eigvalsh(gauge_mass)
    mw2, mz2 = g2*g2*v*v/4, (g2*g2+gy*gy)*v*v/4
    photon = np.array([-gy, 0., 0., g2])/np.sqrt(g2*g2+gy*gy)
    q = (np.eye(2)-PAULI[0])/2
    old_mass = 5*np.eye(2)+GAMMA*PAULI[0]
    old_record = np.diag([0., 1.])
    kappa = 40*GAMMA/v**2
    promoted = (5-GAMMA)**2*np.eye(2)+kappa*projector(h)
    mass_errors = dict(eigenvalues=error(eigenvalues-np.array([0., mw2, mw2, mz2])),
                       photon_kernel=error(gauge_mass@photon), neutral_vev=error(q@h),
                       original_squared_mass=error(promoted-old_mass@old_mass),
                       mass_charge_commutator=error(comm(old_mass@old_mass, q)))
    assert max(mass_errors.values()) < TOL
    record_charge_norm = float(np.linalg.norm(comm(old_record, q), 2))
    assert abs(record_charge_norm-.5) < TOL
    rho = mw2/(mz2*g2*g2/(g2*g2+gy*gy))
    assert abs(rho-1) < TOL
    return dict(
        adjoint=dict(status='REJECTED_FOR_SINGLE_PHOTON_LIMIT',
                     mass_squared=adjoint_mass.tolist(), eigenvalues=adjoint_eigenvalues.tolist(),
                     massless_vectors=2),
        fundamental=dict(status='LIMITED_RECORD_REQUIRES_CHARGED_REFERENCE_OR_REDEFINITION',
                         couplings=dict(g2=g2, gY=gy, v=v),
                         mass_squared=gauge_mass.tolist(), eigenvalues=eigenvalues.tolist(),
                         massless_vectors=1, rho_tree=rho, kappa=kappa,
                         photon_vector=photon.tolist(), checks=mass_errors,
                         record_charge_commutator_operator_norm=record_charge_norm,
                         record_charge_commutator_frobenius=float(np.linalg.norm(comm(old_record, q)))),
    ), eta


def extended_model(initial, singlets, outside, mass, record, colors, eta):
    # Fixed 14+4 block layout; the extra weak partners are new physical field content.
    projected_mass = outside.conj().T@mass@outside
    projected_record = outside.conj().T@record@outside
    internal_mass = 5*np.eye(2)+GAMMA*PAULI[0]
    m18 = block_diag(projected_mass, np.kron(internal_mass, np.eye(2)))
    r18 = block_diag(projected_record, np.kron(np.diag([0., 1.]), np.eye(2)))
    c18 = [block_diag(outside.conj().T@c@outside, np.zeros((4, 4))) for c in colors]
    w18 = [block_diag(np.zeros((14, 14)), np.kron(np.eye(2), p/2)) for p in PAULI]
    y18 = block_diag(np.zeros((14, 14)), np.eye(4)/2)
    embedding = np.vstack((outside.conj().T, np.kron(np.eye(2), eta[:, None])@singlets.conj().T))
    all_generators = c18+w18+[y18]
    q18 = y18-w18[0]
    checks = dict(
        embedding_isometry=error(embedding.conj().T@embedding-np.eye(16)),
        mass_intertwining=error(m18@embedding-embedding@mass),
        record_intertwining=error(r18@embedding-embedding@record),
        color_intertwining=max(error(c@embedding-embedding@old) for c, old in zip(c18, colors)),
        all_squared_mass_gauge_commutators=max(error(comm(m18@m18, a)) for a in all_generators),
        all_record_gauge_commutators=max(error(comm(r18, a)) for a in all_generators),
        record_projector=error(r18@r18-r18),
        record_electromagnetic_commutator=error(comm(r18, q18)),
        initial_electromagnetic_charge=error(q18@embedding@initial),
        gauge_hermiticity=max(error(a-a.conj().T) for a in all_generators),
    )
    assert max(checks.values()) < TOL
    rows = []
    values, modes = np.linalg.eigh(m18)
    assert values.min() > 0
    for time in (0., .25, .5, .75, 1.):
        evolution = expm(-1j*time*m18)
        spectral = (modes*np.exp(-1j*time*values))@modes.conj().T
        chi = evolution@embedding@initial
        probability = float(np.vdot(chi, r18@chi).real)
        errors = dict(unitarity=error(evolution.conj().T@evolution-np.eye(18)),
                      expm_vs_spectrum=error(evolution-spectral),
                      all_state_evolution_intertwining=error(evolution@embedding-embedding@expm(-1j*time*mass)),
                      record_probability=abs(probability-np.sin(GAMMA*time)**2))
        assert max(errors.values()) < TOL
        rows.append(dict(time=time, record_probability=probability, errors=errors))
    old_image = embedding@embedding.conj().T
    leakage = [float(np.linalg.norm((np.eye(18)-old_image)@w@embedding, 2)) for w in w18]
    assert max(leakage) > .49
    return dict(status='CONDITIONAL_NEUTRAL_RECORD_AND_BOSONIC_GAUGE_CONSTRUCTION',
                internal_complex_components=18, independent_Higgs_complex_components=2,
                gauge_connection_components=12, old_complex_components=16,
                scalar_hypercharges='outside 14: 0; record x weak 4: 1/2',
                checks=checks, minimum_mass=float(values.min()),
                times=rows, old_image_weak_leakage_operator_norm=leakage), all_generators


def curvature_audit(all_generators):
    gen = np.array(all_generators)
    traces = np.array([[np.trace(a@b).real for b in gen] for a in gen])
    indices = np.array([4.]*8+[1.]*4)
    assert error(traces-np.diag(indices)) < TOL
    mixed_error = max(error(comm(a, b)) for left, right in ((gen[:8], gen[8:11]),
                       (gen[:8], gen[11:]), (gen[8:11], gen[11:])) for a in left for b in right)
    assert mixed_error < TOL
    # SU(3) structure constants from the separate fundamental representation.
    fundamental = generators()
    fc = np.array([[[2*np.trace((-1j*comm(a, b))@c).real for c in fundamental]
                    for b in fundamental] for a in fundamental])
    fw = np.zeros((3, 3, 3))
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        fw[a, b, c], fw[b, a, c] = 1, -1
    rng = np.random.default_rng(6401)
    coefficients = rng.normal(size=(4, 12))
    derivative = rng.normal(size=(4, 4, 12))  # [derivative direction, connection index, algebra]
    connection = np.einsum('ma,aij->mij', coefficients, gen)
    rows = []
    for mu in range(4):
        for nu in range(mu+1, 4):
            linear = derivative[mu, nu]-derivative[nu, mu]
            full = np.einsum('a,aij->ij', linear, gen)-1j*comm(connection[mu], connection[nu])
            projected = np.array([np.trace(full@t).real/index for t, index in zip(gen, indices)])
            separate = linear.copy()
            separate[:8] += np.einsum('abc,a,b->c', fc, coefficients[mu, :8], coefficients[nu, :8])
            separate[8:11] += np.einsum('abc,a,b->c', fw, coefficients[mu, 8:11], coefficients[nu, 8:11])
            errors = dict(hermiticity=error(full-full.conj().T),
                          projection_vs_separate_algebras=error(projected-separate),
                          reconstruction=error(full-np.einsum('a,aij->ij', projected, gen)),
                          positive_trace_norm=abs(np.trace(full@full).real-np.sum(indices*projected**2)))
            assert max(errors.values()) < TOL
            rows.append(dict(mu=mu, nu=nu, errors=errors))
    gstar = np.sqrt(2)
    canonical = gstar/np.sqrt(2*np.array([4., 1., 1.]))
    assert error(canonical-np.array([.5, 1., 1.])) < TOL
    return dict(seed=6401, representation_trace_indices=dict(color=4., weak=1., hypercharge=1.),
                mixed_commutator_error=mixed_error, trace_orthogonality_error=error(traces-np.diag(indices)),
                curvature_checks=rows, supplied_common_trace_gstar=gstar,
                canonical_couplings=dict(g3=float(canonical[0]), g2=float(canonical[1]), gY=float(canonical[2])),
                tree_sin_squared_theta_W=float(canonical[2]**2/(canonical[1]**2+canonical[2]**2)),
                general_invariant_positive_kinetic_coefficients=3,
                gauge_connections_derived_from_three_scalar_angles=False)


def run():
    initial, singlets, outside, mass, record, colors = old_model()
    frozen = frozen_symmetry(singlets, outside, mass, record, colors)
    higgs, eta = higgs_branches()
    extended, all_generators = extended_model(initial, singlets, outside, mass, record, colors, eta)
    curvature = curvature_audit(all_generators)
    root = Path(__file__).resolve().parents[1]
    document = next((root/'paper').glob('06_*/64_*.md'))
    sources = [Path(__file__), root/'verify/ce_color_covariant_record.py',
               root/'verify/ce_isometric_color_frame.py', root/'verify/ce_local_record_stress.py']
    prereg = document.read_text(encoding='utf-8').split('## 64.2')[0]
    return dict(candidate='CE-LR1-W0/WA/WH and CE-LR2', frozen=frozen, higgs=higgs,
                extended=extended, curvature=curvature, fitted_parameters=0,
                full_joint_rmse=None, scientific_success=False,
                narrow_algebra_checks_passed=True,
                limitations=['new scalar representations, Higgs and independent connections are supplied',
                             'no observed chiral fermion spectrum or physical hypercharges',
                             'no measured electroweak input, matching scale or RG comparison',
                             'one trace is an extra axiom; three kinetic coefficients are generally allowed',
                             'no full interacting quantum gauge-gravity construction',
                             'no autonomous stable readout or unique actual record proved here',
                             'the old state subspace is not closed under every weak generator',
                             'not a minimal-extension theorem'],
                environment=dict(python=sys.version.split()[0], numpy=np.__version__,
                                 scipy=scipy.__version__, platform=platform.platform()),
                preregistration_section_sha256=hashlib.sha256(prereg.encode('utf-8')).hexdigest(),
                source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources})


if __name__ == '__main__':
    result = run()
    output = Path(__file__).with_suffix('.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(candidate=result['candidate'], checks=result['narrow_algebra_checks_passed'],
                         frozen=result['frozen']['joint_centralizer'],
                         adjoint_massless=result['higgs']['adjoint']['massless_vectors'],
                         fundamental_record_charge_norm=result['higgs']['fundamental']['record_charge_commutator_operator_norm'],
                         extended_checks=result['extended']['checks'],
                         trace_indices=result['curvature']['representation_trace_indices'],
                         canonical_couplings=result['curvature']['canonical_couplings'],
                         scientific_success=result['scientific_success']), indent=2))
