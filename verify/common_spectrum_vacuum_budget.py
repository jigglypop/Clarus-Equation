"""Audit the common-phase vacuum force before combining dark and muon fits.

Conditional one-loop calculation, not an observational likelihood. Both
sectors are complex scalar triplets with a common phase and positive masses.
Independent microscopic couplings/counterterms could change this model.
"""
from __future__ import annotations

import json
from pathlib import Path

import mpmath as mp

from common_spectrum_muon import ALPHA, M_MU, Q2


def shape(e, theta):
    """U / s**2, with theta=pi subtracted at the same s and epsilon."""
    def trace(angle):
        roots = [1 + 2*e*mp.cos((angle + 2*mp.pi*j)/3) for j in range(3)]
        return sum(x*x*(mp.log(x)-mp.mpf('1.5')) for x in roots)
    return (trace(theta)-trace(mp.pi))/(32*mp.pi**2)


def shape_integral(e, theta):
    """Independent Euclidean momentum determinant, mapped to [0,1]."""
    k = 2*e**3*(1+mp.cos(theta))
    def integrand(v):
        if v == 1:
            return k  # exact endpoint limit after mapping t=v/(1-v)
        t = v/(1-v)
        b = (1+t-2*e)*(1+t+e)**2
        return t*mp.log1p(k/b)/(1-v)**2
    return mp.quad(integrand, [0, mp.mpf('.5'), 1])/(16*mp.pi**2)


def force_integral(e, theta):
    k = 2*e**3*(1+mp.cos(theta))
    dk = -2*e**3*mp.sin(theta)
    def integrand(v):
        if v == 1:
            return dk
        t = v/(1-v)
        b = (1+t-2*e)*(1+t+e)**2
        return t*dk/(b+k)/(1-v)**2
    return mp.quad(integrand, [0, mp.mpf('.5'), 1])/(16*mp.pi**2)


def run():
    with mp.workdps(80):
        theta = mp.mpf('.5')
        charged_mass = mp.mpf('1000')  # GeV; previous fixed muon benchmark
        charged_e = mp.mpf('.025')
        multiplicity = 16  # 6+3+3+2+1+1 for the supplied representation set
        uc = multiplicity*charged_mass**4*shape(charged_e, theta)
        fc = multiplicity*charged_mass**4*mp.diff(lambda t: shape(charged_e, t), theta)
        rows = []
        # External manuscript 27's dark-sector benchmark scales, not predictions.
        for er, mass_ev in [('.15', '.027615'), ('.35', '.014414')]:
            e, mass = mp.mpf(er), mp.mpf(mass_ev)*mp.mpf('1e-9')
            ud = mass**4*shape(e, theta)
            fd = mass**4*mp.diff(lambda t: shape(e, t), theta)
            # Shape must change by this factor if a *new* common action is to
            # keep the charged contribution below this old neutral potential.
            target = ud/(multiplicity*charged_mass**4)
            leading_e = (target*16*mp.pi**2/(1+mp.cos(theta)))**(mp.mpf(1)/3)
            solved_e = mp.findroot(lambda q: mp.log(shape_integral(mp.exp(q), theta)/target),
                                  mp.log(leading_e))
            solved_e = mp.exp(solved_e)
            def visible_responses(frac):
                b = (1-2*frac)*(1+frac)**2
                k = 2*frac**3*(1+mp.cos(theta))
                gauge = -mp.mpf(str(Q2))/(12*mp.pi)*mp.log1p(k/b)
                inverse_mass_delta = -3*(1-frac**2)*k/(b*(b+k)*charged_mass**2)
                muon_delta = (mp.mpf(str(ALPHA))/mp.pi)**2 * mp.mpf(str(M_MU))**2 * mp.mpf(str(Q2))/360 * inverse_mass_delta
                return dict(delta_alpha_star_inverse=float(gauge),
                            delta_muon_em_heavy_leading=float(muon_delta))
            rows.append(dict(dark_epsilon_over_s=float(e), dark_mass_eV=float(mp.mpf(mass_ev)),
                dark_U_GeV4=float(ud), charged_U_GeV4=float(uc),
                charged_to_dark_potential=float(uc/ud),
                dark_dU_dtheta_GeV4=float(fd), charged_dU_dtheta_GeV4=float(fc),
                charged_to_dark_force=float(fc/fd),
                illustrative_charged_e_to_equal_dark_U=float(solved_e),
                visible_response_before=visible_responses(charged_e),
                visible_response_at_equal_U=visible_responses(solved_e),
                inversion_relative_residual=float(abs(shape_integral(solved_e,theta)/target-1)),
                inversion_status="one-point consistency bound, not an observational fit or full-history condition"))

        checks=[]
        for es, angle in [('.025', '.5'), ('.15', '.5'), ('.35', '.5'), ('.49', '2.4')]:
            e, th = mp.mpf(es), mp.mpf(angle)
            algebra, integral = shape(e, th), shape_integral(e, th)
            derivative = mp.diff(lambda t: shape(e, t), th)
            checks.append(dict(epsilon_over_s=float(e), theta=float(th),
                relative_error=float(abs(algebra-integral)/abs(algebra)),
                force_relative_error=float(abs(derivative-force_integral(e, th))/abs(derivative))))
        assert max(c['relative_error'] for c in checks) < 1e-60
        assert max(c['force_relative_error'] for c in checks) < 1e-60
        assert max(r['inversion_relative_residual'] for r in rows) < 1e-15
        return dict(role="common_action_consistency_audit", common_phase=float(theta),
            assumptions=["same dynamical theta in both sectors", "complex bosonic triplets",
                         "fixed masses and fractional splittings", "no cancelling theta-dependent sector",
                         "flat-space relative one-loop potential; no absolute vacuum prediction"],
            charged_mass_GeV=float(charged_mass), charged_epsilon_over_s=float(charged_e),
            charged_representation_dimension=multiplicity, cases=rows,
            determinant_checks=checks, decimal_precision=80,
            joint_rmse=None, scientific_success=False,
            conclusion="neutral-only background cannot be combined unchanged with the supplied charged sector",
            source_commit="05013ce3c653fc68c2fe5a4ab6294e95bd6c0a30")


if __name__ == '__main__':
    result=run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
