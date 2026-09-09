"""Normalized collective probe of the unbounded continuous spectral candidate.

No physical UV cutoff: the exact resolvent uses an integral to infinity of
K(beta+a*t)/K(beta). beta is a supplied preparation scale, not a derived law.
This oscillator construction gives quantum response and relative vacuum energy,
not a local spacetime model, cosmological density, or observational RMSE gain.
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import gammaln, log_ndtr, roots_genlaguerre

from dimension_joint_candidate import dimension_kernel


def parameters(beta, eta, b, m, a, coupling):
    values = tuple(map(float, (beta, eta, b, m, a, coupling)))
    if not all(map(math.isfinite, values)):
        raise ValueError("finite parameters required")
    beta, eta, b, m, a, coupling = values
    if beta <= 0 or b <= 0 or m <= 0 or a <= 0 or coupling < 0 or not 0 <= eta <= 1:
        raise ValueError("beta,b,m,a > 0, coupling >= 0, 0 <= eta <= 1 required")
    return values


class BoundaryProbe:
    """Dimensionless D0=m**2+a*X; D=D0+coupling*|v><v|; ||v||=1.

    nu_beta(dx)=exp(-beta*x)*rho(x)*dx/K(beta), on [0,infinity).
    The spectral-density functional K is NOT a Hilbert trace on L2(nu_beta).
    hbar=1 and the fixed reference frequency is one.
    """

    def __init__(self, *, beta=1., eta=.1, b=1., m=1., a=1., coupling=.5):
        self.beta, self.eta, self.b, self.m, self.a, self.coupling = parameters(
            beta, eta, b, m, a, coupling)
        self.log_normalization = self.kernel(self.beta)["log_heat_trace"]

    def kernel(self, t):
        return dimension_kernel(t, eta=self.eta, b=self.b, d_ir=4.)

    def bare_resolvent(self, z=0., *, derivative_a=False):
        """g(z)=<v,(D0+z)^-1 v>, z>=0, via exact heat integral.

        Substitution r=(m²+z)*t removes a narrowing large-z peak. derivative_a
        holds the prepared measure fixed, rather than silently repreparing it.
        Returned error is quadrature error only, not a physical uncertainty.
        """
        z = float(z)
        if not math.isfinite(z) or z < 0:
            raise ValueError("finite z>=0 required")
        rate = self.m**2+z

        def integrand(r):
            t = r/rate
            tau = self.beta+self.a*t
            kernel = self.kernel(tau)
            weight = math.exp(-r+kernel["log_heat_trace"]-self.log_normalization)/rate
            if derivative_a:
                weight *= -t*kernel["dimension"]/(2*tau)
            return weight

        return quad(integrand, 0., math.inf, epsabs=2e-11, epsrel=2e-10)

    def response(self, z=0.):
        g, error = self.bare_resolvent(z)
        denominator = 1+self.coupling*g
        return {"bare": g, "coupled": g/denominator,
                "quadrature_error": error/denominator**2}

    def vacuum(self):
        """Finite relative energy, <Q²>, and relative derivative wrt a.

        E=(1/2pi)*integral_0^infty log(1+lambda*g(omega²)) domega.
        Every quantity uses the same probe, state measure, and coupling.
        The error is the outer quadrature estimate; inner tolerances above
        and the independent finite-oscillator checks must also be inspected.
        """
        def energy(omega):
            g = self.bare_resolvent(omega*omega)[0]
            return math.log1p(self.coupling*g)/(2*math.pi)

        def variance(omega):
            return self.response(omega*omega)["coupled"]/math.pi

        def derivative(omega):
            z = omega*omega
            g = self.bare_resolvent(z)[0]
            ga = self.bare_resolvent(z, derivative_a=True)[0]
            return self.coupling*ga/(1+self.coupling*g)/(2*math.pi)

        val, err = quad(energy, 0., math.inf, epsabs=2e-9, epsrel=1e-8)
        var, var_err = quad(variance, 0., math.inf, epsabs=2e-9, epsrel=1e-8)
        da, da_err = quad(derivative, 0., math.inf, epsabs=2e-9, epsrel=1e-8)
        return {"relative_energy": val, "variance_Q": var,
                "d_energy_d_coupling": var/2, "d_energy_d_a": da,
                "energy_upper_bound": self.coupling/(4*self.m),
                "outer_quadrature_errors": {"energy": err, "variance": var_err, "d_a": da_err}}

    def local_potential(self):
        """Four-dimensional constant-field one-loop potential, in a fixed scheme.

        Subtract the whole Taylor coefficients of orders lambda and lambda² at
        lambda=0. These are mass/quartic renormalization conditions, NOT predictions
        of their finite values or of the cosmological constant. A curved/time-varying
        theory needs additional terms; this is not the oscillator vacuum energy.
        """
        return local_potential_integral(lambda z: self.bare_resolvent(z)[0],
                                        self.coupling, self.m)

    def positive_quadrature(self, n_u=16, n_x=24):
        """Independent positive quadrature of nu, for finite-oscillator checks.

        The continuous u tail is bounded explicitly. Finite nodes approximate
        the integrals, and are NOT the physical dimension or energy cutoff.
        We do not renormalize the weights to conceal integration errors.
        """
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in (n_u, n_x)):
            raise ValueError("integer quadrature orders >=2 required")
        q = self.kernel(self.beta)["continuum_weight"]
        nodes, weights = [], []

        def append_component(u, weight):
            gamma_shape = 2+u
            x, w = roots_genlaguerre(n_x, gamma_shape-1)
            nodes.extend(x/self.beta)
            weights.extend(weight*w*math.exp(-gammaln(gamma_shape)))

        if q < 1:
            append_component(0., 1-q)
        tail_bound = 0.
        if q > 0:
            logbeta = math.log(self.beta)
            center = -logbeta/(2*self.b)
            sd = 1/math.sqrt(2*self.b)
            upper = max(center, 0.)+10*sd
            normal_log_mass = float(log_ndtr(center/sd))
            tail_bound = q*math.exp(float(log_ndtr((center-upper)/sd))-normal_log_mass)
            z, w = np.polynomial.legendre.leggauss(n_u)
            for u, weight in zip((z+1)*upper/2, w*upper/2):
                log_density = (-.5*((u-center)/sd)**2-math.log(sd*math.sqrt(2*math.pi))
                               -normal_log_mass)
                append_component(u, q*weight*math.exp(log_density))
        x, w = np.asarray(nodes), np.asarray(weights)
        if not (np.isfinite(x).all() and np.isfinite(w).all() and (w >= 0).all()):
            raise ValueError("quadrature exceeds numerical range")
        return x, w, {"weight_sum": float(w.sum()), "u_tail_probability_bound": tail_bound,
                      "nodes": len(x), "n_u": n_u, "n_x": n_x}

    def finite_oscillators(self, n_u=16, n_x=24):
        """Independent diagonalization: oscillator square roots, not logdet integral."""
        x, w, info = self.positive_quadrature(n_u, n_x)
        v = np.sqrt(w)
        d0 = self.m**2+self.a*x
        d = np.diag(d0)+self.coupling*np.outer(v, v)
        eigenvalues, eigenvectors = np.linalg.eigh(d)
        frequencies = np.sqrt(eigenvalues)
        probe = eigenvectors.T@v
        relative = .5*float(np.sum(frequencies)-np.sum(np.sqrt(d0)))
        variance = .5*float(np.sum(probe*probe/frequencies))
        inverse_sqrt_diagonal = (eigenvectors*eigenvectors)@(1/frequencies)
        derivative = .25*float(x@(inverse_sqrt_diagonal-1/np.sqrt(d0)))
        return {**info, "relative_energy": relative, "variance_Q": variance,
                "d_energy_d_a": derivative,
                "static_response": float(np.sum(probe*probe/eigenvalues)),
                "minimum_squared_frequency": float(eigenvalues.min())}


def local_potential_integral(resolvent, coupling, m):
    """Arithmetic for a supplied normalized positive-spectrum resolvent.

    The caller must establish 0<g(z)<=1/(z+m²); arbitrary functions are not
    certified as physical by this numerical integral.
    """
    if not math.isfinite(coupling) or not math.isfinite(m) or coupling < 0 or m <= 0:
        raise ValueError("finite coupling>=0 and m>0 required")

    def remainder(y):
        if y < .01:
            return math.fsum((-1)**(k+1)*y**k/k for k in range(3, 12))
        return math.log1p(y)-y+y*y/2

    def integrand(z, derivative=False):
        g = resolvent(z)
        y = coupling*g
        if derivative:
            return z*g*y*y/(1+y)/(32*math.pi**2)
        return z*remainder(y)/(32*math.pi**2)

    potential, error = quad(integrand, 0., math.inf, epsabs=2e-11, epsrel=2e-8)
    derivative, derivative_error = quad(lambda z: integrand(z, True), 0., math.inf,
                                        epsabs=2e-11, epsrel=2e-8)
    return {"renormalized_relative_potential": potential, "d_potential_d_coupling": derivative,
            "potential_upper_bound": coupling**3/(192*math.pi**2*m*m),
            "renormalization": "V(0)=V_prime(0)=V_second(0)=0_for_loop_remainder_only",
            "outer_quadrature_errors": {"potential": error, "derivative": derivative_error}}


def coupled_collective_evolution(*, duration=4., n_u=16, n_x=8, kappa=.5,
                                 omega_collective=.7, q_initial=.6,
                                 reciprocal=True, rtol=2e-11, atol=2e-13):
    """Conditional Gaussian/Ehrenfest dynamics with a classical collective q.

    H=p_q²/2+Omega²*q²/2+(P²+Q.D0.Q)/2+kappa*q²*Q_v²/2.
    D0 and nu are fixed; lambda(q)=kappa*q². The bath begins in the D0 vacuum.
    Both q and every bath covariance respond. This is not an FLRW model, and
    a classical mean-field q is not the exact fully quantized interacting theory.
    reciprocal=False deliberately removes bath force for an energy-failure test.
    """
    if (not all(math.isfinite(float(x)) for x in (duration, kappa, omega_collective, q_initial))
            or duration <= 0 or kappa < 0 or omega_collective <= 0):
        raise ValueError("finite duration,Omega>0, kappa>=0 required")
    probe = BoundaryProbe()
    x, weights, info = probe.positive_quadrature(n_u=n_u, n_x=n_x)
    v = np.sqrt(weights)
    d0 = probe.m**2+probe.a*x
    frequency = np.sqrt(d0)
    n = len(x)
    # Q and P are mode-function matrices, giving Sigma_QQ=Re(Q Q^dagger).
    Q = np.diag(1/np.sqrt(2*frequency)).astype(complex)
    P = -1j*np.diag(np.sqrt(frequency/2))
    initial = np.r_[complex(q_initial), 0j, Q.ravel(), P.ravel()]

    def unpack(state):
        return (float(state[0].real), float(state[1].real),
                state[2:2+n*n].reshape(n, n), state[2+n*n:].reshape(n, n))

    def rhs(_, state):
        q, velocity, modes, momenta = unpack(state)
        projected = v@modes
        variance = float(np.vdot(projected, projected).real)
        force = -omega_collective**2*q
        if reciprocal:
            force -= kappa*q*variance
        acceleration = -d0[:, None]*modes-kappa*q*q*v[:, None]*projected[None, :]
        return np.r_[complex(velocity), complex(force), momenta.ravel(), acceleration.ravel()]

    times = np.linspace(0., duration, 41)
    solution = solve_ivp(rhs, (0., duration), initial, method="DOP853", t_eval=times,
                         rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(solution.message)
    vacuum_reference = .5*float(frequency.sum())
    energies, positions, variances, commutator_errors = [], [], [], []
    for state in solution.y.T:
        q, velocity, modes, momenta = unpack(state)
        projected = v@modes
        variance = float(np.vdot(projected, projected).real)
        energy = (.5*velocity**2+.5*omega_collective**2*q*q
                  +.5*np.sum(np.abs(momenta)**2)+.5*np.sum(d0[:, None]*np.abs(modes)**2)
                  +.5*kappa*q*q*variance-vacuum_reference)
        # [q_i,p_j]=Q P^dagger - Q* P^T; subtracting P Q^dagger
        # instead is wrong for coupled modes with off-diagonal correlations.
        qp = modes@momenta.conj().T-modes.conj()@momenta.T
        qq = modes@modes.conj().T-modes.conj()@modes.T
        pp = momenta@momenta.conj().T-momenta.conj()@momenta.T
        energies.append(float(energy))
        positions.append(q)
        variances.append(variance)
        commutator_errors.append(float(max(np.max(np.abs(qp-1j*np.eye(n))),
                                            np.max(np.abs(qq)), np.max(np.abs(pp)))))
    return {"status": "conditional_gaussian_collective_dynamics_not_cosmology",
            "quadrature": info, "reciprocal_force": reciprocal,
            "kappa": kappa, "omega_collective": omega_collective,
            "q_initial": q_initial, "duration": duration, "rtol": rtol, "atol": atol,
            "times": times.tolist(), "q": positions, "variance_Q": variances,
            "relative_total_energy": energies,
            "max_absolute_energy_drift": float(max(abs(e-energies[0]) for e in energies)),
            "max_canonical_commutator_error": max(commutator_errors),
            "rhs_evaluations": solution.nfev}


def report():
    cases = []
    for beta in (.5, 1., 2.):
        probe = BoundaryProbe(beta=beta)
        exact = probe.vacuum()
        cases.append({"beta": beta, "continuum": exact, "static_response": probe.response(),
                      "local_four_dimensional_potential": probe.local_potential(),
                      "finite_oscillators": [probe.finite_oscillators(n_u=16, n_x=n) for n in (16, 32)]})
    return {"status": "conditional_collective_oscillator_not_observational_fit",
            "parameters": {"eta": .1, "b": 1., "m": 1., "a": 1., "coupling": .5},
            "prepared_beta_is_supplied": True, "spacetime_source_derived": False,
            "all_domain_rmse_reduced": False, "cases": cases,
            "coupled_dynamics": coupled_collective_evolution(),
            "missing_reaction_counterexample": coupled_collective_evolution(reciprocal=False)}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
