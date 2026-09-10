"""CE-GR7: linearized EH pullback, retarded operator, and record-source gate.

The EH action, Minkowski background and free-state quantization are inputs.
The radial PDE is a Green-operator test, not a conserved matter source model.
The two-point record energy profile is tested as a candidate complete T00.
"""
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp

from ce_embedding_einstein_variation import COMPONENTS, DYADS, ETA, exact_rank, free_embedding
from ce_color_covariant_record import HF, HS, HA, GAMMA, setup, unitary


TOL = 1e-10
BASIS = []
RAW = []
for i, j in COMPONENTS:
    element = np.zeros((4, 4))
    element[i, j] = element[j, i] = 1.
    RAW.append(element)
    BASIS.append(element if i == j else element/np.sqrt(2))
BASIS = np.array(BASIS)


def error(value):
    return float(np.max(abs(value)))


def coordinates(tensor):
    return np.einsum('aij,ij->a', BASIS, tensor)


def einstein_closed(h, k):
    raised_k = ETA@k
    k_squared = k@raised_k
    trace = np.trace(ETA@h)
    contraction = h@raised_k
    return (k_squared*h+np.outer(k, k)*trace-np.outer(k, contraction)
            -np.outer(contraction, k)-ETA*(k_squared*trace-raised_k@h@raised_k))/2


def einstein_connection(h, k):
    gamma = np.zeros((4, 4, 4), complex)
    for r in range(4):
        for m in range(4):
            for n in range(4):
                gamma[r, m, n] = .5j*sum(ETA[r, s]*(k[m]*h[s, n]+k[n]*h[s, m]-k[s]*h[m, n])
                                        for s in range(4))
    ricci = np.einsum('r,rmn->mn', 1j*k, gamma)-np.outer(np.einsum('rmr->m', gamma), 1j*k)
    result = ricci-.5*ETA*np.trace(ETA@ricci)
    assert error(result.imag) < TOL
    return result.real


def fourier_case(k):
    k = np.array(k, float)
    square = float(k@ETA@k)
    metric_hessian = np.column_stack([coordinates(-.5*ETA@einstein_closed(h, k)@ETA) for h in BASIS])
    gauge = np.column_stack([coordinates(np.outer(k, xi)+np.outer(xi, k)) for xi in np.eye(4)])
    raw_operator = np.array([[2*einstein_closed(h, k)[i, j] for h in RAW] for i, j in COMPONENTS])
    assert np.array_equal(raw_operator, np.rint(raw_operator))
    rational_rank = exact_rank(raw_operator.astype(int))
    expected_rank = 4 if square == 0. else 6
    assert rational_rank == np.linalg.matrix_rank(metric_hessian) == expected_rank
    algebra = dict(connection_vs_closed=max(error(einstein_closed(h, k)-einstein_connection(h, k)) for h in BASIS),
                   bianchi=max(error((ETA@k)@einstein_closed(h, k)) for h in BASIS),
                   hessian_symmetry=error(metric_hessian-metric_hessian.T),
                   gauge_kernel=error(metric_hessian@gauge))
    assert np.linalg.matrix_rank(gauge) == 4 and max(algebra.values()) < TOL
    plus = np.diag([0., 1., -1., 0.])/np.sqrt(2)
    cross = np.zeros((4, 4))
    cross[1, 2] = cross[2, 1] = 1/np.sqrt(2)
    tt = np.column_stack((coordinates(plus), coordinates(cross)))
    tt_hessian = tt.T@metric_hessian@tt
    assert error(tt_hessian+square*np.eye(2)/4) < TOL
    if square == 0.:
        assert np.linalg.matrix_rank(np.column_stack((gauge, tt))) == 6
        assert error(metric_hessian@tt) < TOL
    pulled = []
    for radius in (.05, .1, .2):
        geometry = free_embedding(radius, np.zeros(4))
        active = np.column_stack([coordinates(2*radius*dyad) for dyad in DYADS])
        full = np.column_stack((active, np.zeros((10, 10))))
        hessian = full.T@metric_hessian@full
        jacobian = float(abs(np.linalg.det(active)))
        jacobian_formula = 8*(2*radius)**10
        assert abs(jacobian/jacobian_formula-1) < TOL
        assert np.linalg.matrix_rank(full) == 10 and np.linalg.matrix_rank(hessian) == expected_rank
        normal_gauge = np.vstack((np.linalg.solve(active, gauge), np.zeros((10, 4))))
        fibers = np.vstack((np.zeros((10, 10)), np.eye(10)))
        redundant = np.column_stack((normal_gauge, fibers))
        assert np.linalg.matrix_rank(redundant) == 14 and error(hessian@redundant) < TOL
        source_h = sum((i+1)*element/13 for i, element in enumerate(BASIS))
        h_coordinates = coordinates(source_h)
        phi = np.linalg.solve(active, h_coordinates)
        action_error = abs(phi@(active.T@metric_hessian@active)@phi-h_coordinates@metric_hessian@h_coordinates)
        assert action_error < TOL
        pulled.append(dict(radius=radius, metric_rank=10, normal_dimension=20,
                           normal_hessian_rank=int(np.linalg.matrix_rank(hessian)),
                           metric_fiber_nullity=10, linear_metric_gauge_dimension=4,
                           on_shell_quotient_dimension=20-expected_rank-14,
                           coordinate_jacobian=jacobian, jacobian_formula=jacobian_formula,
                           reconstructed_action_error=float(action_error),
                           inherited_geometry_max_error=max(geometry['errors'].values())))
    return dict(covector=k.tolist(), minkowski_square=square, exact_rank=rational_rank,
                numeric_rank=int(np.linalg.matrix_rank(metric_hessian)), algebra_errors=algebra,
                TT_hessian=tt_hessian.tolist(), embedding_cases=pulled)


def helicity_case():
    plus = np.diag([0., 1., -1., 0.])/np.sqrt(2)
    cross = np.zeros((4, 4))
    cross[1, 2] = cross[2, 1] = 1/np.sqrt(2)
    tensors = [plus, cross]
    cases = []
    for angle in (np.pi/7, np.pi/4, np.pi/2):
        rotation = np.eye(4)
        rotation[1:3, 1:3] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        induced = np.array([[np.sum(a*(rotation@b@rotation.T)) for b in tensors] for a in tensors])
        target = np.array([[np.cos(2*angle), -np.sin(2*angle)], [np.sin(2*angle), np.cos(2*angle)]])
        residual = error(induced-target)
        assert residual < TOL
        cases.append(dict(angle=angle, spin_two_rotation_error=residual))
    return dict(cases=cases, kinetic_coefficient_per_Frobenius_TT_mode=.125,
                canonical_field='v_lambda = M_P q_lambda / 2',
                canonical_H='sum_lambda integral (pi_lambda^2 + |grad v_lambda|^2)/2',
                status='positive_free_TT_sector_only_with_supplied_EH_sign_and_linear_quotient')


def pulse(time):
    time = np.asarray(time)
    return np.where((time > 0.) & (time < 1.), np.sin(np.pi*time)**4, 0.)


def radial_wave():
    cases = []
    for spacing in (.05, .025, .0125):
        cells = round(6/spacing)
        radii = np.arange(1, cells)*spacing
        size = len(radii)
        def derivative(time, state):
            displacement, velocity = state[:size], state[size:]
            padded = np.concatenate(([float(pulse(time))], displacement, [0.]))
            acceleration = (padded[2:]-2*padded[1:-1]+padded[:-2])/spacing**2
            return np.concatenate((velocity, acceleration))
        solution = solve_ivp(derivative, (0., 4.), np.zeros(2*size), method='DOP853',
                             t_eval=[2., 4.], rtol=1e-10, atol=1e-12)
        assert solution.success, solution.message
        rows = []
        for index, time in enumerate(solution.t):
            exact = pulse(time-radii)
            computed = solution.y[:size, index]
            rows.append(dict(time=float(time), maximum_error=error(computed-exact),
                             numerical_tail_outside_continuum_front=error(computed[radii > time]),
                             exact_tail_outside_continuum_front=error(exact[radii > time])))
        cases.append(dict(spacing=spacing, interior_points=size, rows=rows,
                          maximum_error=max(row['maximum_error'] for row in rows)))
        print(f'radial spacing={spacing}: error={cases[-1]["maximum_error"]:.6g}', flush=True)
    assert cases[-1]['maximum_error'] < .01
    assert all(fine['maximum_error'] < coarse['maximum_error'] for coarse, fine in zip(cases[:-1], cases[1:]))
    return dict(status='retarded_Green_operator_test_not_a_conserved_stress', cases=cases,
                exact_solution='u(t,r)=f(t-r), u=r hbar',
                causal_support='Continuous exact solution vanishes for r>t; finite spatial lattices have small tails.')


def newton_case():
    # M_P^2=1 gives 8*pi*G=1 and k^2*hbar=2*T in the static Lorenz gauge.
    k = np.array([0., 1., 2., 3.])
    source = np.zeros((4, 4))
    source[0, 0] = 1.
    trace_reversed = 2*source/(k@ETA@k)
    h = trace_reversed-.5*ETA*np.trace(ETA@trace_reversed)
    equation_error = error(einstein_connection(h, k)-source)
    assert equation_error < TOL and error((ETA@k)@trace_reversed) < TOL
    mu = .01
    bending = []
    for impact in (1., 2.):
        integral, estimate = quad(lambda z: 2*mu*impact/(impact**2+z*z)**1.5,
                                  -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)
        formula = 4*mu/impact
        assert abs(integral-formula) < TOL
        bending.append(dict(impact_parameter=impact, integrated_deflection=integral,
                            linear_GR_deflection=formula, quadrature_error_estimate=estimate))
    return dict(static_fourier_Einstein_error=equation_error, G_times_source_mass=mu,
                point_metric='h00=h11=h22=h33=2 Gm/r; Phi=-Gm/r', bending=bending,
                source_condition='Static zeroth-order conserved source only; no moving-record stress inferred.')


def record_source_case():
    initial, _, transition = setup()
    hamiltonian = HF+GAMMA*transition
    dipole = -np.kron(HS, np.eye(4))+np.kron(np.eye(4), HA)
    first = 1j*(hamiltonian@dipole-dipole@hamiltonian)
    second = -(hamiltonian@(hamiltonian@dipole-dipole@hamiltonian)
               -(hamiltonian@dipole-dipole@hamiltonian)@hamiltonian)
    rows = []
    for time in (0., .25, .5, .75, 1.):
        state = unitary(hamiltonian, time)@initial
        values = [float(np.vdot(state, operator@state).real) for operator in (dipole, first, second)]
        p = np.sin(GAMMA*time)**2
        formulas = [-5+8*p, 8*GAMMA*np.sin(2*GAMMA*time), 16*GAMMA**2*np.cos(2*GAMMA*time)]
        formula_error = error(np.array(values)-formulas)
        source_energy = float(np.vdot(state, HF@state).real)
        total_energy = float(np.vdot(state, hamiltonian@state).real)
        interaction = float(np.vdot(state, (hamiltonian-HF)@state).real)
        assert formula_error < TOL and abs(source_energy-5) < TOL and abs(total_energy-5) < TOL
        rows.append(dict(time=time, dipole=values[0], dipole_velocity=values[1],
                         dipole_acceleration=values[2], independent_double_commutator_error=formula_error,
                         two_site_energy=source_energy, total_H_energy=total_energy,
                         interaction_energy_expectation=interaction))
    assert max(abs(row['dipole_acceleration']) for row in rows) > 1.
    return dict(status='rejected_as_complete_isolated_symmetric_stress_with_only_two_fixed_energy_sites',
                positions=[-1., 1.], gamma=GAMMA, rows=rows,
                necessary_identity='For symmetric conserved decaying T: d/dt integral x T00 = integral T0x, and its second derivative is zero.',
                scope='Additional recoil, support, field energy distribution or boundary flux can change T00 and the conclusion.',
                omitted_source_warning='Zero total expectation of record interaction energy does not imply zero local interaction-energy dipole.')


if __name__ == '__main__':
    result = dict(candidate='CE-GR7', status='conditional_linear_EH_recovery_and_record_stress_counterexample',
                  fourier_cases=[fourier_case(k) for k in ((2, 0, 0, 1), (1, 0, 0, 2), (1, 0, 0, 1))],
                  physical_TT=helicity_case(), retarded_operator=radial_wave(),
                  static_Newton=newton_case(), changing_record_source=record_source_case(),
                  fitted_parameters=0, full_joint_rmse=None, scientific_success=False,
                  assumptions=['Supplied EH action, Lorentzian free embedding and metric-only Y dependence.',
                               'Linear metric-null fibers are quotiented only as an additional conditional construction.',
                               'Retarded response requires a conserved zeroth-order stress.'],
                  limits=['No nonlinear embedding measure, constraint closure, UV quantization or rank preservation.',
                          'No microscopic derivation of EH, Newton constant, initial state or three gauge sectors.',
                          'A radial Green test is not a physical conserved pulse source.',
                          'A changing two-site energy profile fails a necessary isolated-stress condition.',
                          'No observations or full joint residuals are evaluated.'])
    here = Path(__file__).resolve()
    sources = {here}
    for name, module in list(sys.modules.items()):
        if name.startswith('ce_') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            if path.parent == here.parent:
                sources.add(path)
    result['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    result['environment'] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    here.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(fourier_ranks=[r['exact_rank'] for r in result['fourier_cases']],
                          physical_TT=result['physical_TT'], newton=result['static_Newton'],
                          record_source=result['changing_record_source']), indent=2))
