"""CE-RB9: probability vs trace, dimension-weighted UV and the actual rank-one probe.

The infinite-dimensional arguments are in chapter 27. Finite matrices only check identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import exp1, gammaln, log_ndtr, roots_genlaguerre

from verify_reverse import Evidence


def kernel(t, eta=.2, b=.7):
    logt = np.log(t)
    a = logt/(2*np.sqrt(b))
    logz = a*a+np.log(2)+log_ndtr(-np.sqrt(2)*a)
    logw = np.logaddexp(np.log1p(-eta), np.log(eta)+logz)
    tilted_weight = np.exp(np.log(eta)+logz-logw)
    mean_u = np.exp(-logz)/np.sqrt(np.pi*b)-logt/(2*b)
    return -2*logt+logw, 4+2*tilted_weight*mean_u


class Probe:
    def __init__(self, beta=1., m=1., a=.8):
        self.beta, self.m, self.a = beta, m, a
        self.lognorm = kernel(beta)[0]

    def g(self, z):
        rate = self.m*self.m+z
        return quad(lambda r: np.exp(-r+kernel(self.beta+self.a*r/rate)[0]-self.lognorm)/rate,
                    0, np.inf, epsabs=2e-12, epsrel=2e-11)[0]

    def energy(self, coupling):
        return quad(lambda omega: np.log1p(coupling*self.g(omega*omega))/(2*np.pi),
                    0, np.inf, epsabs=2e-10, epsrel=2e-9)[0]

    def positive_quadrature(self, nu=32, nx=96):
        eta, b = .2, .7
        total_nodes, total_weights = [], []
        atom = (1-eta)*self.beta**(-2)*np.exp(-self.lognorm)
        def append(u, probability):
            nodes, weights = roots_genlaguerre(nx, 1+u)
            total_nodes.extend(nodes/self.beta)
            total_weights.extend(probability*weights*np.exp(-gammaln(2+u)))
        append(0., atom)
        upper = max(0., -np.log(self.beta)/(2*b))+10/np.sqrt(2*b)
        nodes, weights = np.polynomial.legendre.leggauss(nu)
        for u, weight in zip((nodes+1)*upper/2, weights*upper/2):
            probability = weight*eta*2*np.sqrt(b/np.pi)*np.exp(-b*u*u-(2+u)*np.log(self.beta)-self.lognorm)
            append(u, probability)
        return np.array(total_nodes), np.array(total_weights)


def remainder(y):
    if y < .02:
        return y**3*sum((-y)**j/(j+3) for j in range(10))
    return np.log1p(y)-y+y*y/2


def symmetric_power(matrix, exponent):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors*values**exponent)@vectors.T


def run():
    e = Evidence()
    values = {}
    # R77: a probability with infinite first moment still has vanishing UV dimension.
    dimensions = []
    for t in (.01, .0001, .000001):
        probability = 1-t*np.exp(t)*exp1(t)
        numerator = t*((1+t)*np.exp(t)*exp1(t)-1)
        dimension = 2*numerator/probability
        cutoff = t**(-.5)
        tail = 1/(1+cutoff)
        bound = 2*(t*cutoff+tail/np.e)/((1-tail)*np.exp(-t*cutoff))
        dimensions.append(float(dimension))
        e.check('R77', f'infinite_mean_probability_tail_bound_{t}', 0 <= dimension <= bound,
                {'dimension': float(dimension), 'tail_bound': float(bound), 'cutoff': cutoff})
    e.check('R77', 'infinite_mean_probability_dimension_tends_down', dimensions[0] > dimensions[1] > dimensions[2] > 0,
            {'dimensions': dimensions})
    t = .1
    direct = quad(lambda x: np.exp(-t*x)/(1+x)**2, 0, np.inf, epsabs=1e-12)[0]
    e.close('R77', 'Pareto_Laplace_identity', direct, 1-t*np.exp(t)*exp1(t), 1e-11)
    # On a nonatomic bounded interval, infinitely many orthogonal indicator modes have a nonzero heat norm.
    heat_lower = np.exp(-.7*(1+1))
    for count in (8, 32, 128):
        lower_edges = 2.**(-np.arange(1, count+1))
        widths = lower_edges.copy()
        # Exact Rayleigh quotients for normalized indicators of disjoint dyadic intervals,
        # in L2 of Lebesgue probability on [0,1]. No finite matrix cutoff defines the trace.
        rayleigh = np.exp(-.7*(1+lower_edges))*(-np.expm1(-.7*widths))/(.7*widths)
        e.check('R77', f'orthogonal_heat_images_prevent_trace_class_{count}',
                np.all(rayleigh >= heat_lower) and np.sum(rayleigh) >= count*heat_lower,
                {'number_of_orthogonal_indicators': count, 'actual_partial_trace': float(np.sum(rayleigh)),
                 'partial_trace_lower_bound': count*heat_lower,
                 'uniform_positive_lower_bound': heat_lower})

    # R78: exact exponential tilting makes the dimension derivative a variance.
    for logt in (-6., -1., 0., 2., 5.):
        eta, b = .2, .7
        moment = lambda power: quad(lambda u: u**power*2*np.sqrt(b/np.pi)*np.exp(-b*u*u-logt*u),
                                   0, np.inf, epsabs=1e-11, epsrel=1e-11)[0]
        z0, z1, z2 = [moment(p) for p in (0, 1, 2)]
        partition = 1-eta+eta*z0
        mean = eta*z1/partition
        variance = eta*z2/partition-mean*mean
        logk, dimension = kernel(np.exp(logt))
        e.close('R78', f'independent_dimension_tilt_{logt}', dimension, 4+2*mean, 2e-10)
        step = 1e-5
        derivative = (kernel(np.exp(logt+step))[1]-kernel(np.exp(logt-step))[1])/(2*step)
        e.close('R78', f'dimension_derivative_is_minus_twice_variance_{logt}', derivative, -2*variance, 2e-8)
        e.check('R78', f'strict_dimension_monotonicity_{logt}', variance > 0,
                {'tilted_variance': float(variance)})
    for logt in (-20., -40.):
        dimension = kernel(np.exp(logt))[1]
        e.close('R78', f'UV_dimension_asymptote_{logt}', dimension, 4-logt/.7, 1e-9)

    # R79: finite moment cancellation cannot tame the actual half-Gaussian dimension tail.
    s, eps, t = sp.symbols('s eps t', positive=True)
    xs0, xspi = [s+2*eps, s-eps, s-eps], [s-2*eps, s+eps, s+eps]
    for power in (0, 1, 2):
        e.zero('R79', f'triplet_cancelled_moment_{power}', sum(x**power for x in xs0)-sum(x**power for x in xspi))
    e.zero('R79', 'first_surviving_relative_heat_coefficient',
           -(sum(x**3 for x in xs0)-sum(x**3 for x in xspi))/6+2*eps**3)
    for order in (3, 7, 20):
        b = .7
        ell = 4*b*(order-2)+10
        exponent = ell*ell/(4*b)-(order-2)*ell
        slope = ell/(2*b)-(order-2)
        e.check('R79', f'every_finite_subtraction_order_still_diverges_{order}', exponent > 0 and slope > 0,
                {'log_integrand_lower_asymptote': exponent, 'positive_log_slope': slope})
    for log_inverse_cutoff in (2., 8., 32.):
        length = log_inverse_cutoff
        critical = quad(lambda gap: -np.expm1(-length*gap)/gap if gap else length, 0, 1)[0]
        e.close('R79', f'critical_edge_loglog_divergence_{length}', critical, np.euler_gamma+np.log(length)+exp1(length), 1e-11)
        integrable = quad(lambda gap: 2*(-np.expm1(-length*gap)), 0, 1)[0]
        e.close('R79', f'vanishing_edge_weight_converges_{length}', integrable, 2*(1-(1-np.exp(-length))/length), 1e-11)

    # R80: same original prepared density, exact heat resolvent and independent positive Gamma quadrature.
    probe = Probe()
    nodes, weights = probe.positive_quadrature()
    e.close('R80', 'prepared_positive_quadrature_normalization', np.sum(weights), 1., 2e-10)
    mean_x = kernel(probe.beta)[1]/(2*probe.beta)
    e.close('R80', 'prepared_mean_from_heat_derivative', np.dot(weights, nodes), mean_x, 2e-9)
    for z in (0., .5, 10., 1000.):
        exact = probe.g(z)
        discrete = np.dot(weights, 1/(probe.m**2+probe.a*nodes+z))
        e.close('R80', f'heat_resolvent_vs_independent_quadrature_{z}', exact, discrete, 2e-9)
        step = 1e-4
        numerical = (Probe(beta=1+step).g(z)-Probe(beta=1-step).g(z))/(2*step)
        covariance_derivative = mean_x*exact-(1-(probe.m**2+z)*exact)/probe.a
        e.close('R80', f'preparation_derivative_covariance_identity_{z}', numerical, covariance_derivative, 3e-9)
        e.check('R80', f'response_depends_monotonically_on_preparation_{z}', covariance_derivative > 0,
                {'d_g_d_beta': float(covariance_derivative)})
    energies = []
    for coupling in (.2, .6, 1.):
        energy = probe.energy(coupling)
        mean_d = probe.m**2+probe.a*mean_x
        lower = (np.sqrt(mean_d+coupling)-np.sqrt(mean_d))/2
        upper = (np.sqrt(probe.m**2+coupling)-probe.m)/2
        e.check('R80', f'vacuum_energy_sharp_Jensen_bounds_{coupling}', lower < energy < upper,
                {'energy': energy, 'lower': lower, 'upper': upper})
        energies.append(energy)
    e.check('R80', 'relative_energy_strict_concavity', 2*energies[1] > energies[0]+energies[2],
            {'equally_spaced_couplings': [.2, .6, 1.], 'energies': energies})
    other_energies = [Probe(beta=beta).energy(.6) for beta in (.5, 2.)]
    e.check('R80', 'same_density_different_prepared_vacuum_response', other_energies[0] < energies[1] < other_energies[1],
            {'beta': [.5, 1., 2.], 'relative_energies': [other_energies[0], energies[1], other_energies[1]]})
    values['prepared_probe'] = {'eta': .2, 'b': .7, 'beta': 1., 'm': probe.m, 'a': probe.a,
                               'mean_internal_x': mean_x, 'g0': probe.g(0.), 'relative_energy_at_lambda_0p6': energies[1]}

    # R81: normalization fixes the leading spacetime UV tail, not its removal.
    for z in (100., 10000., 1000000.):
        bare = probe.g(z)
        dressed = bare/(1+.6*bare)
        e.check('R81', f'normalized_positive_resolvent_UV_{z}', 0 < z*bare < 1 and z*bare > 1-(probe.m**2+probe.a*mean_x)/z,
                {'z_g': z*bare, 'z_G': z*dressed})
    mu = probe.m**2+probe.a*mean_x
    z = 100000.
    coefficient = z*z*(probe.g(z)-1/z)
    e.close('R81', 'next_UV_moment_coefficient', coefficient, -mu, .002)
    for mass2 in (1., 3.):
        for coupling in (.1, .6, 2.):
            direct = quad(lambda z: z*remainder(coupling/(z+mass2))/(32*np.pi**2), 0, np.inf,
                          epsabs=1e-13, epsrel=1e-11)[0]
            closed = ((mass2+coupling)**2*np.log1p(coupling/mass2)-mass2*coupling-1.5*coupling**2)/(64*np.pi**2)
            e.close('R81', f'fixed_scheme_single_mass_integral_{mass2}_{coupling}', direct, closed, 2e-12)
            cutoff = 10.
            tail = quad(lambda z: z*remainder(coupling/(z+mass2))/(32*np.pi**2), cutoff, np.inf, epsabs=1e-13)[0]
            bound = coupling**3/(96*np.pi**2)*(cutoff+mass2/2)/(cutoff+mass2)**2
            e.check('R81', f'explicit_renormalized_tail_bound_{mass2}_{coupling}', 0 < tail <= bound,
                    {'tail': tail, 'proven_bound': bound})
    x, lam = sp.symbols('x lambda', positive=True)
    finite = ((x+lam)**2*sp.log(1+lam/x)-x*lam-sp.Rational(3, 2)*lam**2)/(64*sp.pi**2)
    for order in (0, 1, 2):
        e.zero('R81', f'fixed_scheme_boundary_condition_{order}', sp.diff(finite, lam, order).subs(lam, 0))
    e.zero('R81', 'fixed_scheme_third_derivative', sp.diff(finite, lam, 3)-1/(32*sp.pi**2*(x+lam)))

    # R82: bounded rank-one perturbation satisfies the infinite-Fock implementability criterion.
    rng = np.random.default_rng(20260924)
    for dimension in (3, 9, 30):
        d0 = np.diag(1+rng.uniform(0, 12, dimension))
        vector = rng.normal(size=dimension); vector /= np.linalg.norm(vector)
        coupling = .8
        d1 = d0+coupling*np.outer(vector, vector)
        a0, a1 = symmetric_power(d0, .5), symmetric_power(d1, .5)
        difference = a1-a0
        e.close('R82', f'Sylvester_identity_{dimension}', a1@difference+difference@a0, d1-d0, 2e-12)
        beta = (symmetric_power(d1, .25)@symmetric_power(d0, -.25)
                -symmetric_power(d1, -.25)@symmetric_power(d0, .25))/2
        factored = symmetric_power(d1, -.25)@difference@symmetric_power(d0, -.25)/2
        alpha = (symmetric_power(d1, .25)@symmetric_power(d0, -.25)
                 +symmetric_power(d1, -.25)@symmetric_power(d0, .25))/2
        e.close('R82', f'Bogoliubov_factorization_{dimension}', beta, factored, 2e-12)
        e.close('R82', f'Bogoliubov_canonical_identity_{dimension}', alpha@alpha.T-beta@beta.T, np.eye(dimension), 2e-12)
        e.close('R82', f'Bogoliubov_pair_symmetry_{dimension}', alpha@beta.T-beta@alpha.T, 0., 2e-12)
        e.check('R82', f'uniform_Hilbert_Schmidt_bound_{dimension}', np.linalg.norm(beta, 'fro') <= coupling/4,
                {'HS_norm_beta': float(np.linalg.norm(beta, 'fro')), 'proven_bound': coupling/4})
        e.check('R82', f'uniform_relative_trace_bound_{dimension}', 0 <= np.trace(difference)/2 <= coupling/4,
                {'relative_energy': float(np.trace(difference)/2), 'proven_bound': coupling/4})
    mass2, coupling = 1., .8
    single_number = (((mass2+coupling)/mass2)**.25-((mass2+coupling)/mass2)**(-.25))**2/4
    for dimension in (16, 64, 256):
        rank_one_eigenvalues = np.linalg.eigvalsh(mass2*np.eye(dimension)+coupling*np.ones((dimension, dimension))/dimension)
        rank_one_number = np.sum(((rank_one_eigenvalues/mass2)**.25-(rank_one_eigenvalues/mass2)**(-.25))**2)/4
        e.close('R82', f'independent_rank_one_particle_diagonalization_{dimension}', rank_one_number, single_number, 2e-12)
        e.check('R82', f'identity_perturbation_diverges_with_mode_count_{dimension}', dimension*single_number > single_number,
                {'number_of_modes': dimension, 'identity_perturbation_particle_number': dimension*single_number,
                 'normalized_rank_one_particle_number': single_number})

    ids = sorted({row['claim'] for row in e.checks})
    assert ids == [f'R{i:02d}' for i in range(77, 83)]
    return {'schema': 'CE-RB9-v1', 'scope': 'actual continuous-dimension weight and prepared rank-one branch',
            'observational_validation': False, 'full_CE_completion': False,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'helper_sha256': hashlib.sha256(Path(__file__).with_name('verify_reverse.py').read_bytes()).hexdigest(),
            'claim_ids': ids, 'number_of_checks': len(e.checks), 'all_passed': all(row['passed'] for row in e.checks),
            'checks': e.checks, 'values': values}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results_uv.json'))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f"PASS {result['number_of_checks']} checks; {len(result['claim_ids'])} claim groups")
