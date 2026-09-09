"""Test equilibrium selection in the supplied normalized late-time action.

Not a no-go theorem for CE or for non-equilibrium cosmology.
"""
import json
from pathlib import Path

import numpy as np
import mpmath as mp
from scipy.integrate import quad

from ce_obs32_profiled_matter import ce, SOURCE_SHA
from common_spectrum_vacuum_budget import shape, shape_integral


def absolute_scale_check():
    rows = []
    with mp.workdps(60):
        planck = mp.mpf('2.435e18')  # Fixed reduced Planck mass benchmark, GeV.
        kinetic_input = mp.mpf('1e-12')
        for r, mass_ev in [('.15', '.027615'), ('.35', '.014414')]:
            e = mp.mpf(r)
            dark_mass = mp.mpf(mass_ev)*mp.mpf('1e-9')
            physical_s_ratio = (dark_mass/planck)**2
            mass_from_kinetic = planck*mp.sqrt(kinetic_input)
            u_dark = dark_mass**4*shape(e, mp.mpf('.5'))
            u_kinetic = mass_from_kinetic**4*shape_integral(e, mp.mpf('.5'))
            ratio = u_kinetic/u_dark
            assert abs(ratio/(kinetic_input/physical_s_ratio)**2-1) < mp.mpf('1e-45')
            rows.append(dict(r=float(e), reference_dark_mass_eV=float(mass_ev),
                corresponding_s_over_Mp2=float(physical_s_ratio),
                source_kinetic_s_over_Mp2=float(kinetic_input),
                mass_implied_by_source_kinetic_GeV=float(mass_from_kinetic),
                reference_relative_U_GeV4=float(u_dark),
                relative_U_at_source_kinetic_mass_GeV4=float(u_kinetic),
                potential_ratio=float(ratio)))
    return dict(assumptions='same unsuppressed scalar triplet in potential and kinetic term; theta=.5; reference dark masses supplied, not predicted',
        checks=rows, observational_fit=False,
        conclusion='Normalized background shape alone does not establish one common absolute mass scale.')


def vacuum_ambiguity(zs, observed, covariance):
    """Two allowed constant terms: counterexample to unique absolute prediction."""
    p = ce.Parameters(.35, .5)
    sp = ce.Spectrum(p)
    minimum = sp.evaluate(np.pi)
    matter = p.Rm*((1-p.fc)+p.fc*minimum[3])
    rows = []
    for constant in [0., 1.]:
        def expansion(z):
            return np.sqrt((matter*(1+z)**3+constant)/(matter+constant))
        prediction = np.array([quad(lambda t: 1/expansion(t), 0, z)[0]*expansion(z) for z in zs])
        residual = np.linalg.solve(np.linalg.cholesky(covariance), prediction-observed)
        # A constant added at every theta cancels from the relative potential.
        difference_errors = [abs(((sp.evaluate(t)[0]+constant)-(minimum[0]+constant))-sp.evaluate(t)[0])
                             for t in [.1,.5,1.2]]
        assert max(difference_errors) < 3e-16
        rows.append(dict(constant_vacuum_in_Utop_units=constant,
            H0_in_sqrt_Utop_over_Mp_units=float(np.sqrt((matter+constant)/3)),
            AP_rmse=float(np.sqrt(residual@residual/len(observed))),
            maximum_relative_potential_change=max(difference_errors)))
    return dict(role='nonuniqueness witness, not selected or fitted CE candidate',
        fixed_state='theta=pi, zero velocity, same matter abundance and spectrum',
        relative_loop_responses_unchanged=True, cases=rows,
        missing_law='absolute renormalized vacuum condition',
        conclusion='Identical relative spectral information admits distinct absolute expansion and AP scores.')


def run():
    checks = []
    for r in [.15, .35]:
        sp = ce.Spectrum(ce.Parameters(r, .5))
        values = np.array([sp.evaluate(t) for t in np.linspace(.001, np.pi-.001, 101)])
        assert np.all(values[:, 1] < 0) and np.all(values[:, 4] < 0)
        top, bottom = sp.evaluate(0), sp.evaluate(np.pi)
        assert top[2] < 0 and top[5] < 0 and bottom[2] > 0 and bottom[5] > 0
        assert abs(bottom[0]) < 1e-14
        x, x1, _ = sp.eigs(0.)
        loop_z = sp.p.s_over_Mp2*float(np.sum(x1*x1/x))/(96*np.pi**2)
        # Independent analytic endpoint expression; do not subtract f^2
        # from Z, which would lose the tiny loop term to roundoff.
        analytic_z = sp.p.s_over_Mp2*r*r/(144*np.pi**2*(1-r))
        assert np.isclose(loop_z, analytic_z, rtol=1e-13, atol=0)
        curvature_ratio = 3*abs(top[2])/loop_z
        # Local growing mode at a vacuum-dominated hilltop, constant H.
        growth = (np.sqrt(9+4*curvature_ratio)-3)/2
        checks.append(dict(r=r, top_U_curvature=top[2], top_m_curvature=top[5],
            minimum_U_curvature=bottom[2], minimum_m_curvature=bottom[5], minimum_U=bottom[0],
            loop_only_Z_at_top=loop_z,
            loop_only_abs_mass_squared_over_H_squared=curvature_ratio,
            local_instability_efolds=1/growth,
            supplied_f=sp.p.f,
            supplied_kinetic_abs_mass_squared_over_H_squared=3*abs(top[2])/top[6],
            f_lower_bound_for_abs_mass_squared_below_H_squared=float(np.sqrt(max(0,3*abs(top[2])-loop_z)))))
    _, _, _, _, zs, observed, cov = ce.load_data()
    # Flat, pressureless, zero residual vacuum, zero phase velocity: E=(1+z)^1.5.
    predicted = 2*((1+zs)**1.5-(1+zs))
    numeric = np.array([quad(lambda x: (1+x)**(-1.5), 0, z)[0]*(1+z)**1.5 for z in zs])
    assert np.max(abs(predicted-numeric)) < 1e-13
    null = ce.Background(ce.Parameters(.35, 0.))
    baseline = np.array([null.distances(z)['F_AP'] for z in zs])
    def score(pred):
        residual = pred-observed
        white = np.linalg.solve(np.linalg.cholesky(cov), residual)
        chi = float(white@white)
        assert np.isclose(chi, residual@np.linalg.solve(cov, residual), rtol=1e-13)
        return float(np.sqrt(chi/len(zs)))
    return dict(source_sha256=SOURCE_SHA, stationary_checks=checks, absolute_scale_check=absolute_scale_check(),
        vacuum_ambiguity=vacuum_ambiguity(zs, observed, cov),
        assumption='Equilibrium at pi, zero phase velocity, flat late-time pressureless model; no added absolute vacuum constant',
        AP_baseline_rmse=score(baseline), AP_equilibrium_rmse=score(predicted),
        AP_equilibrium_prediction=predicted.tolist(), fitted_parameters=0,
        joint_rmse=None, independent_holdout=False,
        conclusion='Equilibrium removes relative dark energy and worsens these AP residuals; non-equilibrium initial state remains undetermined.')


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
