"""Chapter 13 common-action reduction; errors here are not observational RMSE.

Keep the original q,N,f and stiffness examples. Use the exact loop quadrature,
relax transverse initial coordinates, and compare full and reduced dynamics.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from numpy.polynomial.legendre import leggauss


def run(order=128):
    x, weights = leggauss(order)
    u = (x+1)/2
    t = u/(1-u)
    weights = weights/2*t/(1-u)**2/(16*np.pi**2)
    h = .6**2*.5*.4
    A = (t+1)*(t+1.5)-.6**2*.5
    B = 2*h**3/((A-2*h)*(A+h)**2)
    curvature = float(weights@B)

    def loop(phi):
        s, c = np.sin(phi), np.cos(phi)
        denominator = 1+B*(1-c)
        return np.array([weights@np.log1p(B*(1-c)),
                         weights@(B*s/denominator),
                         weights@(B*c/denominator-(B*s/denominator)**2)])/curvature

    f, q, N = .05, 3, 4
    D = np.eye(N, N+1)-q*np.eye(N, N+1, k=1)
    ell = q**np.arange(N, -1, -1, dtype=float)
    F2 = f*f*(ell@ell)
    eig, vectors = np.linalg.eigh(D.T@D)
    transverse = vectors[:, 1:]
    inverse = (transverse/eig[1:])@transverse.T
    coefficient = inverse[-1, -1]
    results = []
    for stiffness in [2., 8., 32.]:
        def potential(theta):
            link = D@theta
            W, W1, _ = loop(theta[-1])
            grad = stiffness*D.T@np.sin(link)
            grad[-1] += W1
            return stiffness*np.sum(1-np.cos(link))+W, grad

        phi0 = np.pi-.6
        relaxed = root(lambda eta: transverse.T@potential(ell*phi0+transverse@eta)[1],
                       np.zeros(N), tol=1e-10)
        assert np.linalg.norm(relaxed.fun) < 1e-9
        theta0 = ell*phi0+transverse@relaxed.x
        H0 = np.sqrt(potential(theta0)[0]/3)
        duration = 3/H0

        def full_rhs(time, y):
            theta, velocity = y[:N+1], y[N+1:2*(N+1)]
            V, grad = potential(theta)
            H = np.sqrt((.5*f*f*(velocity@velocity)+V)/3)
            return np.r_[velocity, -3*H*velocity-grad/f**2, H]

        grid = np.linspace(0, duration, 1001)
        full = solve_ivp(full_rhs, (0, duration), np.r_[theta0, np.zeros(N+2)],
                         method='DOP853', rtol=2e-10, atol=2e-12, t_eval=grid)
        assert full.success
        Hfull = np.array([full_rhs(time, y)[-1] for time, y in zip(grid, full.y.T)])
        errors = []
        for corrected in [False, True]:
            def reduced_rhs(time, y):
                phi, v, _ = y
                W, W1, W2 = loop(phi)
                V = W-coefficient*W1**2/(2*stiffness) if corrected else W
                force = W1-coefficient*W1*W2/stiffness if corrected else W1
                H = np.sqrt((.5*F2*v*v+V)/3)
                return [v, -3*H*v-force/F2, H]
            sol = solve_ivp(reduced_rhs, (0, duration), [phi0, 0, 0],
                            method='DOP853', rtol=2e-11, atol=2e-13, t_eval=grid)
            assert sol.success
            H = np.array([reduced_rhs(time, y)[-1] for time, y in zip(grid, sol.y.T)])
            errors.append(float(np.max(abs(H/Hfull-1))))
        assert errors[1] < errors[0]
        results.append(dict(stiffness=stiffness, leading_H_max_relative=errors[0],
                            corrected_H_max_relative=errors[1], reduction_factor=errors[0]/errors[1]))
    return dict(quadrature_order=order, W_second_zero=curvature, F_over_Mp=np.sqrt(F2),
                transverse_coefficient=coefficient, cases=results,
                observable_rmse=None, fitted_parameters=[],
                scope='Full versus reduced equations with relaxed initial state; no observed data or heavy-mode abundance.')


if __name__ == '__main__':
    result = run()
    check = run(192)
    error = max(abs(a['corrected_H_max_relative']-b['corrected_H_max_relative'])
                for a, b in zip(result['cases'], check['cases']))
    assert error < 1e-8
    result['quadrature_refinement_error'] = error
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
