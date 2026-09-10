"""CE-LM1: one supplied local fermion action, its vacuum and modular operator.

Independent Slater/Fock and correlation/partial-trace routes. This is a finite
matter regulator, not a common gauge/gravity theory or an observation fit.
"""
import hashlib
import itertools
import json
from pathlib import Path
import platform

import mpmath as mp
import numpy as np
from scipy import sparse


TOL = 1e-10
N = 8
BLOCK = [2, 3, 4, 5]


def annihilators(n):
    result = []
    for site in range(n):
        bit = 1 << (n-1-site)
        rows, cols, data = [], [], []
        for state in range(1 << n):
            if state & bit:
                rows.append(state ^ bit)
                cols.append(state)
                data.append((-1)**((state >> (n-site)).bit_count()))
        result.append(sparse.csr_matrix((data, (rows, cols)), shape=(1 << n, 1 << n)))
    return result


def second_quantize(h, c):
    out = sparse.csr_matrix(c[0].shape, dtype=complex)
    for i, j in zip(*np.nonzero(h)):
        out += h[i, j]*(c[i].T@c[j])
    return out


def comm(a, b):
    return a@b-b@a


def max_error(a):
    if sparse.issparse(a):
        return float(np.max(abs(a.data), initial=0.))
    return float(np.max(abs(a), initial=0.))


def reduced_density(rho, kept, total=N):
    outside = [i for i in range(total) if i not in kept]
    order = kept+outside+[i+total for i in kept+outside]
    shaped = rho.reshape([2]*(2*total)).transpose(order)
    dk, de = 1 << len(kept), 1 << len(outside)
    return np.trace(shaped.reshape(dk, de, dk, de), axis1=1, axis2=3)


def entropy(rho):
    values = np.linalg.eigvalsh(rho)
    assert values.min() > -TOL
    positive = values[values > 1e-14]
    return float(-np.sum(positive*np.log(positive)))


def local_current_audit(h, c, mass):
    full = second_quantize(h, c)
    number = [a.T@a for a in c]
    zero = sparse.csr_matrix(full.shape, dtype=complex)
    current = [-1j*(c[j].T@c[j+1]-c[j+1].T@c[j]) for j in range(N-1)]
    particle_error = max(max_error(1j*comm(full, number[j])
        - (current[j-1] if j else zero)+(current[j] if j < N-1 else zero))
        for j in range(N))
    energy_error = None
    if mass == 0:
        bonds = [-(c[j].T@c[j+1]+c[j+1].T@c[j]) for j in range(N-1)]
        flux = [zero]+[1j*(c[j-1].T@c[j+1]-c[j+1].T@c[j-1])
                       for j in range(1, N-1)]+[zero]
        energy_error = max(max_error(1j*comm(full, bonds[j])-flux[j]+flux[j+1])
                           for j in range(N-1))
        assert energy_error < TOL
    locality = max_error(comm(c[0].T@c[1]+c[1].T@c[0], c[6].T@c[7]+c[7].T@c[6]))
    assert particle_error < TOL and locality < TOL
    return dict(particle_current_operator_error=particle_error,
                massless_bond_energy_current_operator_error=energy_error,
                disjoint_even_bond_commutator=locality,
                scope='exact finite lattice continuity; not covariant stress or strict lightcone')


def local_quadratic_commutant(h):
    basis = []
    for i in BLOCK:
        a = np.zeros((N, N), complex)
        a[i, i] = 1
        basis.append(a)
    for i, j in itertools.combinations(BLOCK, 2):
        for value in (1., 1j):
            a = np.zeros((N, N), complex)
            a[i, j], a[j, i] = value, np.conjugate(value)
            basis.append(a)
    columns = []
    for a in basis:
        b = comm(h, a)
        columns.append(np.r_[b.real.reshape(-1), b.imag.reshape(-1)])
    singular = np.linalg.svd(np.array(columns).T, compute_uv=False)
    rank = int(np.count_nonzero(singular > TOL))
    assert rank == 16
    return dict(hermitian_basis_dimension=16, commutator_real_rank=rank,
                least_singular_value=float(singular[-1]),
                global_number_commutator=max_error(comm(h, np.eye(N))),
                local_nonconstant_conserved_quadratic_record=False)


def finite_vacuum_audit(mass, c):
    h = np.diag(mass*(-1.)**np.arange(N))-np.diag(np.ones(N-1), 1)-np.diag(np.ones(N-1), -1)
    vals, modes = np.linalg.eigh(h)
    occupied = modes[:, vals < 0]
    assert occupied.shape[1] == N//2
    slater = np.zeros(1 << N)
    for sites in itertools.combinations(range(N), N//2):
        state = sum(1 << (N-1-j) for j in sites)
        slater[state] = np.linalg.det(occupied[list(sites), :])
    full_h = second_quantize(h, c).toarray()
    energies, states = np.linalg.eigh(full_h)
    rho = np.outer(slater, slater.conj())
    block_rho = reduced_density(rho, BLOCK)
    correlator = occupied@occupied.T
    cb = correlator[np.ix_(BLOCK, BLOCK)]
    cv, cu = np.linalg.eigh(cb)
    assert cv.min() > 0 and cv.max() < 1
    modular_h = (cu*np.log((1-cv)/cv))@cu.T
    block_c = annihilators(len(BLOCK))
    many_modular = second_quantize(modular_h, block_c).toarray()
    kv, ku = np.linalg.eigh(many_modular)
    weights = np.exp(-(kv-kv.min()))
    gaussian_rho = (ku*(weights/weights.sum()))@ku.conj().T
    corr_direct = np.array([[np.trace(block_rho@(a.T@b).toarray()) for b in block_c] for a in block_c])
    errors = dict(slater_norm=abs(float(np.vdot(slater, slater).real)-1),
                  ground_projector=max_error(rho-np.outer(states[:, 0], states[:, 0].conj())),
                  ground_energy=abs(float(energies[0])-float(vals[vals < 0].sum())),
                  correlator=max_error(corr_direct-cb),
                  gaussian_density=max_error(block_rho-gaussian_rho),
                  block_trace=abs(float(np.trace(block_rho).real)-1))
    assert max(errors.values()) < TOL
    # Exact density logarithm supplies the additive constant independently.
    rv, ru = np.linalg.eigh(block_rho)
    assert rv.min() > 0
    direct_modular = -(ru*np.log(rv))@ru.conj().T
    difference = direct_modular-many_modular
    scalar = np.trace(difference)/len(difference)
    logarithm_error = max_error(difference-scalar*np.eye(len(difference)))
    assert logarithm_error < TOL
    # A local occupation instrument acts inside B and retains its global dilation.
    local_n = (c[3].T@c[3]).toarray()
    projectors = [np.eye(1 << N)-local_n, local_n]
    branches = [p@slater for p in projectors]
    output = sum(np.outer(v, v.conj()) for v in branches)
    environment = [i for i in range(N) if i not in BLOCK]
    reverse = sum(p@v for p, v in zip(projectors, branches))
    energy_cost = float(np.trace((output-rho)@full_h).real)
    bond_cost = float(2*(correlator[2, 3]+correlator[3, 4]))
    record_energy_operator = sum(p@full_h@p for p in projectors)-full_h
    expected_operator = (c[2].T@c[3]+c[3].T@c[2]+c[3].T@c[4]+c[4].T@c[3]).toarray()
    record_errors = dict(completeness=max_error(sum(p@p for p in projectors)-np.eye(1 << N)),
                         reverse_dilation=max_error(reverse-slater),
                         outside_marginal=max_error(reduced_density(output, environment)-reduced_density(rho, environment)),
                         energy_from_bond_correlations=abs(energy_cost-bond_cost),
                         energy_operator=max_error(record_energy_operator-expected_operator))
    assert max(record_errors.values()) < TOL and energy_cost > 0
    far = max(abs(modular_h[i, j]) for i in range(4) for j in range(4) if abs(i-j) > 1)
    return dict(mass=mass, ground_energy=float(energies[0]),
                ground_energy_gap=float(energies[1]-energies[0]),
                block_correlation_eigenvalues=cv.tolist(), block_entropy=entropy(block_rho),
                modular_one_particle_matrix=modular_h.tolist(),
                modular_non_nearest_max=float(far), modular_logarithm_error=logarithm_error,
                errors=errors, current=local_current_audit(h, c, mass),
                local_quadratic_commutant=local_quadratic_commutant(h),
                record=dict(errors=record_errors, ground_state_energy_cost=energy_cost,
                            probability=float(np.vdot(branches[1], branches[1]).real),
                            whole_state_preserved_in_dilation=True,
                            apparatus_energy_and_switching_work_not_derived=True))


def infinite_interval_audit(length):
    with mp.workdps(60):
        correlation = mp.matrix(length)
        for i in range(length):
            for j in range(length):
                d = i-j
                correlation[i, j] = mp.mpf('.5') if not d else mp.sin(mp.pi*d/2)/(mp.pi*d)
        vals, vectors = mp.eigsy(correlation)
        assert min(vals) > 0 and max(vals) < 1
        logarithms = mp.diag([mp.log((1-v)/v) for v in vals])
        modular = vectors*logarithms*vectors.T
        nonlocal_max = max(abs(modular[i, j]) for i in range(length) for j in range(length) if abs(i-j) > 1)
        # A tridiagonal commuting operator is not the modular Hamiltonian itself.
        tri = mp.matrix(length)
        for j in range(length-1):
            tri[j, j+1] = tri[j+1, j] = mp.mpf(j+1)*(length-j-1)/length**2
        comm_error = max(abs(v) for v in (tri*correlation-correlation*tri))
        assert comm_error < mp.mpf('1e-50') and nonlocal_max > mp.mpf('1e-10')
        nearest_only = mp.matrix(modular)
        for i in range(length):
            for j in range(length):
                if abs(i-j) > 1:
                    nearest_only[i, j] = 0
        nv, nu = mp.eigsy(nearest_only)
        reconstructed = nu*mp.diag([1/(1+mp.exp(v)) for v in nv])*nu.T
        reconstruction_error = max(abs(v) for v in reconstructed-correlation)
        return dict(interval_length=length, precision_digits=60,
                    correlation_min_eigenvalue=float(min(vals)),
                    modular_max_long_range_hopping=float(nonlocal_max),
                    modular_end_to_end=float(modular[0, length-1]),
                    tridiagonal_commutator_error=float(comm_error),
                    nearest_only_correlation_max_error=float(reconstruction_error),
                    disposition='exact_nearest_neighbor_modular_identity_rejected',
                    continuum_limit_claim=False)


def low_energy_modular_audit(length):
    with mp.workdps(80):
        correlation = mp.matrix(length)
        for i in range(length):
            for j in range(length):
                correlation[i, j] = mp.mpf('.5') if i == j else mp.sin(mp.pi*(i-j)/2)/(mp.pi*(i-j))
        vals, modes = mp.eigsy(correlation)
        assert min(vals) > 0 and max(vals) < 1
        h = modes*mp.diag([mp.log((1-v)/v) for v in vals])*modes.T
        nearest, conformal = mp.matrix(length), mp.matrix(length)
        for j in range(length-1):
            nearest[j, j+1] = nearest[j+1, j] = h[j, j+1]
            conformal[j, j+1] = conformal[j+1, j] = -mp.pi*(j+1)*(length-j-1)/length
        v = mp.matrix([mp.exp(mp.j*mp.pi*j/2)*mp.sin(mp.pi*(j+mp.mpf('.5'))/length)**2 for j in range(length)])
        v /= mp.norm(v)
        target = conformal*v
        norm = mp.norm(target)
        return dict(length=length, precision_digits=80,
                    full_action_relative_error=float(mp.norm(h*v-target)/norm),
                    nearest_only_action_relative_error=float(mp.norm(nearest*v-target)/norm),
                    full_vs_conformal_matrix_relative_error=float(mp.norm(h-conformal)/mp.norm(conformal)),
                    action_target_norm=float(norm),
                    role='fixed_low_energy_probe_not_uniform_operator_convergence')


def dispersion_audit():
    rows = []
    for q in (.1, .05, .025):
        exact = 2*np.sin(q)
        linear = 2*q
        error = abs(exact-linear)
        bound = q**3/3
        assert error <= bound
        mass = .3
        massive = np.sqrt(mass**2+exact**2)
        dirac = np.sqrt(mass**2+linear**2)
        massive_bound = bound*(abs(exact)+linear)/(massive+dirac)
        assert abs(massive-dirac) <= massive_bound+1e-15
        rows.append(dict(qa=q, massless_error=float(error), massless_bound=bound,
                         massive_error=float(abs(massive-dirac)), massive_bound=float(massive_bound)))
    return dict(fermi_velocity_in_J_a_over_hbar=2, rows=rows,
                continuum_spacetime_dimensions=2,
                scope='supplied free Dirac low-energy dispersion; no four-dimensional GR')


def run():
    c = annihilators(N)
    identity = sparse.eye(1 << N)
    car_error = max(max_error(c[i]@c[j].T+c[j].T@c[i]-(identity if i == j else 0*identity))
                    for i in range(N) for j in range(N))
    assert car_error < TOL
    return dict(candidate='CE-LM1', status='local_matter_vacuum_modular_gate',
                inputs=dict(sites=N, region=BLOCK, hopping=1, masses=[0., .3], fit_parameters=0),
                CAR_error=car_error, finite_cases=[finite_vacuum_audit(m, c) for m in (0., .3)],
                infinite_intervals=[infinite_interval_audit(n) for n in (4, 8, 12)],
                low_energy_modular=[low_energy_modular_audit(n) for n in (8, 12, 16, 24, 32)],
                dispersion=dispersion_audit(),
                full_joint_rmse=None, scientific_success=False,
                limits=['One spatial dimension; local fermion action and CAR are supplied.',
                        'No gauge interaction, covariant stress or emergent geometry is derived.',
                        'A continuum CFT limit is not certified by finite matrix locality.',
                        'Record apparatus energetics and autonomous definite outcome remain open.',
                        'Previous AM4 prepared states are not replaced or declared validated.',
                        'No new observational or independent-holdout evaluation.'])


if __name__ == '__main__':
    result = run()
    here = Path(__file__).resolve()
    result['source_sha256'] = hashlib.sha256(here.read_bytes()).hexdigest()
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__, mpmath=mp.__version__)
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(candidate=result['candidate'], finite_cases=[
        dict(mass=r['mass'], errors=r['errors'], current=r['current'], record=r['record'],
             modular_logarithm_error=r['modular_logarithm_error']) for r in result['finite_cases']],
        infinite_intervals=result['infinite_intervals']), indent=2))
