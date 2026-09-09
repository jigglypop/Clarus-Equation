"""Eliminate the common phase between scalar threshold and heavy EM response.

Conditional spectrum identity; no observational fitting or joint prediction.
"""
import json
import math
from pathlib import Path

import mpmath as mp
from scipy.optimize import brentq

from common_spectrum_muon import ALPHA, M_MU, Q2, exact_em


def matter_relation():
    """Equal conserved populations assumed; their abundance is not predicted."""
    rows = []
    for e in [.025, .15, .35, .49]:
        def direct(theta):
            return sum(math.sqrt(1+2*e*math.cos((theta+2*math.pi*j)/3)) for j in range(3))
        def reconstructed(theta):
            determinant = 1-3*e*e+2*e**3*math.cos(theta)
            return brentq(lambda total: (total*total-3)**2-12*(1-e*e)
                          -8*math.sqrt(determinant)*total,
                          math.sqrt(3), 3., xtol=1e-14)
        errors = [abs(reconstructed(math.pi*k/32)-direct(math.pi*k/32))
                  for k in range(33)]
        assert max(errors) < 2e-14
        rows.append(dict(epsilon_over_s=e, max_mass_sum_absolute_error=max(errors),
            mass_ratio_pi_to_zero=direct(math.pi)/direct(0),
            maximum_mass_decrease_percent=100*(1-direct(math.pi)/direct(0))))
    return dict(assumption='same triplet with equal conserved occupation numbers; nonrelativistic matter',
        checks=rows, abundance_predicted=False,
        caveat='A common phase alone does not make distinct sector mass scales or splittings equal.')


def state_metric_relation():
    """Conditional state rho=M^2/Tr(M^2); not a derived physical state."""
    rows = []
    with mp.workdps(60):
        for e, theta in [(mp.mpf('.15'), mp.mpf('.5')), (mp.mpf('.35'), mp.mpf('1.2'))]:
            def probabilities(t):
                return [(1+2*e*mp.cos((t+2*mp.pi*j)/3))/3 for j in range(3)]
            p = probabilities(theta)
            dp = [mp.diff(lambda t: probabilities(t)[j], theta) for j in range(3)]
            metric = sum(a*a/b for a,b in zip(dp,p))/4
            h = mp.mpf('1e-8')
            # Squared Bures distance = 2(1 - root fidelity).
            distance = 2*(1-sum(mp.sqrt(a*b) for a,b in zip(probabilities(theta-h/2), probabilities(theta+h/2))))
            error = abs(distance/h**2/metric-1)
            assert error < mp.mpf('1e-15')
            loop_over_s = sum((3*a)**2/(3*b) for a,b in zip(dp,p))/(96*mp.pi**2)
            assert abs(loop_over_s-metric/(8*mp.pi**2)) < mp.mpf('1e-55')
            rows.append(dict(r=float(e), theta=float(theta), Bures_metric=float(metric),
                fidelity_distance_relative_error=float(error), loop_Z_over_s=float(loop_over_s)))
    return dict(state_choice='rho = M_squared / (3s), not selected by the CE axioms',
        identity='Z_loop = s * g_Bures / (8*pi^2)', checks=rows,
        fixes_absolute_kinetic_scale=False,
        limitation='An information metric fixes a dimensionless shape after a state is chosen; its action coefficient and the physical state remain inputs.')


def run():
    rows = []
    with mp.workdps(70):
        for mass, frac, angle in [(1000, '.025', '1.2'),
                                  (1000, '.35', '.5'), (10, '.49', '.8')]:
            s = mp.mpf(mass)**2
            eps = mp.mpf(frac)*s
            theta = mp.mpf(angle)
            def roots(t):
                return [s+2*eps*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
            reference = roots(mp.pi)
            current = roots(theta)
            dpi = (s-2*eps)*(s+eps)**2
            ell = mp.log(mp.fprod(current)/mp.fprod(reference))
            direct = sum(1/x for x in current)-sum(1/x for x in reference)
            eliminated = 3*(s*s-eps*eps)/dpi*mp.expm1(-ell)
            error = abs((direct-eliminated)/direct)
            assert error < mp.mpf('1e-55')
            coefficient = (mp.mpf(str(ALPHA))/mp.pi)**2*mp.mpf(str(M_MU))**2*mp.mpf(str(Q2))/360
            heavy = coefficient*eliminated
            finite = exact_em(mass, float(frac), float(angle), relative=True)
            rows.append(dict(mass_GeV=mass, epsilon_over_s=float(frac), theta=float(theta),
                log_determinant_ratio=float(ell),
                scalar_delta_inverse_alpha=float(-mp.mpf(str(Q2))*ell/(12*mp.pi)),
                inverse_mass_identity_relative_error=float(error),
                heavy_relative_muon=float(heavy), finite_relative_muon=finite,
                heavy_approximation_relative_error=float(abs((finite-heavy)/heavy))))
    return dict(scope='conditional scalar spectrum; phase eliminated, mass and splitting still inputs',
        fitted_parameters=0, checks=rows, matter_relation=matter_relation(),
        state_metric_relation=state_metric_relation(), observational_rmse=None, joint_rmse=None,
        conclusion='No independently adjustable phase corrections for gauge and EM subsets; not a full theory prediction.')


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
