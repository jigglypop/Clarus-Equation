"""Research extension: reflection-symmetric conserved populations.

No optimization. Test the cold (0,1,1) boundary-state hypothesis, its linear
stability, its equilibrium alternative, and the existing seven-row score.
This does not derive the primordial abundance or select this state from RT31.
"""
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

from ce_obs32_occupation_background import ce, OccupationSpectrum, OccupationBackground
from common_spectrum_muon import exact_em


def run():
    *_, zs, observed, covariance = ce.load_data()
    chol = np.linalg.cholesky(covariance)
    gap, sigma = 385e-12, np.hypot(145, 620)*1e-12
    base_bg = ce.Background(ce.Parameters(.35, 0.))
    base_prediction = np.array([base_bg.distances(z)['F_AP'] for z in zs])
    base_residual = np.linalg.solve(chol, base_prediction-observed)
    chi_ap = float(base_residual @ base_residual)
    baseline = float(np.sqrt((chi_ap+(gap/sigma)**2)/7))
    rows = []
    for r, mass in [(.15, .027615), (.35, .014414)]:
        sD = (mass*1e-9)**2
        p = ce.Parameters(r, 0., s_over_Mp2=sD/(2.435e18)**2)
        sp = OccupationSpectrum(p, [0, 1, 1])
        U, U1, U2, m, m1, m2, Z, Z1 = sp.evaluate(0.)
        assert abs(U1) < 1e-12 and abs(m1) < 1e-12
        m2_analytic = r*(2-5*r)/(36*(1-r)**2)
        assert abs(m2-m2_analytic) < 1e-14
        acrit = (-p.fc*p.Rm*m2/U2)**(1/3)
        # Fixed-population effective potential: m(theta) includes all particles.
        h = 1e-3
        curvature_checks = []
        for a in [acrit/2, acrit, acrit*2]:
            def effective(t):
                v = sp.evaluate(t)
                return v[0]+p.fc*p.Rm/a**3*v[3]
            fd = (-effective(2*h)+16*effective(h)-30*effective(0.)
                  +16*effective(-h)-effective(-2*h))/(12*h*h)
            analytic = U2+p.fc*p.Rm/a**3*m2
            assert abs(fd-analytic) < 2e-7
            curvature_checks.append(dict(a=a, analytic=analytic, finite_difference=fd))
        bg = OccupationBackground(p, [0, 1, 1], initial_velocity=0.)
        prediction = np.array([bg.distances(z)['F_AP'] for z in zs])
        assert np.max(abs(prediction-base_prediction)) < 1e-10
        muon = exact_em(1000., r*sD/1e6, 0.)
        muon_check = exact_em(1000., r*sD/1e6, 0., representation='spectral')
        assert abs(muon/muon_check-1) < 1e-8
        # Score difference computed without subtracting nearby chi-squareds.
        delta_chi = (-2*gap*muon+muon*muon)/(sigma*sigma)
        score = np.sqrt(baseline*baseline+delta_chi/7)
        delta_score = delta_chi/(7*(score+baseline))
        assert delta_score < 0
        with mp.workdps(60):
            cb, gg, ss, dd = map(lambda x:mp.mpf(str(x)), [chi_ap,gap,sigma,muon])
            independent_delta = (mp.sqrt((cb+((gg-dd)/ss)**2)/7)
                                 -mp.sqrt((cb+(gg/ss)**2)/7))
            assert abs(float(independent_delta)-delta_score) < 1e-20
        # Linear perturbation about the exact symmetric background.
        def linear_rhs(N, y):
            a = np.exp(N)
            H2 = (p.Rm/a**3+1)/3
            hN = -p.Rm/a**3/(2*H2)
            mass_over_H2 = (U2+p.fc*p.Rm/a**3*m2)/(Z*H2)
            return [y[1], -(3+hN)*y[1]-mass_over_H2*y[0]]
        linear = solve_ivp(linear_rhs, (np.log(p.ai), 0.), [1., 0.],
                           method='DOP853', rtol=1e-11, atol=1e-13)
        assert linear.success
        # Finite differences of nonlinear trajectories, with fixed diagnostic seeds.
        nonlinear = []
        for seed in [1e-3, 1e-4]:
            pp = ce.Parameters(r, seed, s_over_Mp2=p.s_over_Mp2)
            perturbed = OccupationBackground(pp, [0, 1, 1], initial_velocity=0.)
            transfer = perturbed.quantities(0.)['theta']/seed
            pred = np.array([perturbed.distances(z)['F_AP'] for z in zs])
            residual = np.linalg.solve(chol, pred-observed)
            score_seed = np.sqrt((residual@residual+((gap-muon)/sigma)**2)/7)
            nonlinear.append(dict(initial_theta=seed, final_over_initial=transfer,
                                  partial_rmse=float(score_seed),
                                  delta_partial_rmse=float(score_seed-baseline)))
        assert abs(nonlinear[-1]['final_over_initial']-linear.y[0,-1]) < 1e-6
        # Three-level internal Gibbs diagnostic, not the full momentum-integrated
        # Bose gas: F''=<m''>-Var(m')/T. Test redistribution independently of the
        # frozen-population assumption using a high-precision derivative.
        thermal = []
        with mp.workdps(60):
            rr = mp.mpf(str(r))
            def masses(t):
                return [mp.sqrt(1+2*rr*mp.cos((t+2*mp.pi*j)/3)) for j in range(3)]
            ms = masses(mp.mpf(0))
            mpr = [mp.diff(lambda t: masses(t)[j], 0) for j in range(3)]
            mpp = [mp.diff(lambda t: masses(t)[j], 0, 2) for j in range(3)]
            for temperature in ['.01', '.1', '1']:
                T = mp.mpf(temperature)
                w = [mp.exp(-(x-min(ms))/T) for x in ms]
                w = [x/sum(w) for x in w]
                susceptibility = sum(q*c for q,c in zip(w,mpp))-(sum(q*c*c for q,c in zip(w,mpr))-sum(q*c for q,c in zip(w,mpr))**2)/T
                direct = mp.diff(lambda t: -T*mp.log(sum(mp.exp(-x/T) for x in masses(t))), 0, 2)
                assert abs(susceptibility-direct) < mp.mpf('1e-50')
                thermal.append(dict(T_over_sqrt_s=float(T), heavy_fraction=float(w[0]),
                                    free_energy_curvature=float(susceptibility)))
        # Maximum allowed heavy fraction in a reflection-symmetric frozen state.
        light_curvature = r*(2-5*r)/(36*(1-r)**1.5)
        heavy_curvature = -r/(9*np.sqrt(1+2*r))
        heavy_fraction_limit = light_curvature/(light_curvature-heavy_curvature)
        # At theta=0, x'_heavy=0. The local tree-level annihilation into two
        # canonical phase quanta proceeds through x''_heavy phi^2 |psi|^2/2f^2.
        # Nonrelativistic sigma*v includes the identical-final-particle factor.
        # This checks a specific dilute two-body channel, not every relaxation process.
        with mp.workdps(60):
            rr=mp.mpf(str(r));ss=(mp.mpf(str(mass))*mp.mpf('1e-9'))**2
            planck=mp.mpf('2.435e18');ff=planck
            def raw(t):
                xx=[ss*(1+2*rr*mp.cos((t+2*mp.pi*j)/3)) for j in range(3)]
                return sum(x*x*(mp.log(x)-mp.mpf('1.5')) for x in xx)/(32*mp.pi**2)
            top=raw(0)-raw(mp.pi)
            heavy_mass=mp.sqrt(ss*(1+2*rr))
            quartic=-2*rr*ss/(9*ff**2)
            sigma_v=quartic**2/(64*mp.pi*heavy_mass**2)
            rates=[]
            for aa in ['.01','1']:
                a=mp.mpf(aa)
                rho_c=mp.mpf('.84')*(mp.mpf(3)/7)*top/a**3
                hubble=mp.sqrt(top*(1+(mp.mpf(3)/7)/a**3)/(3*planck**2))
                upper_partner_density=rho_c/heavy_mass
                rates.append(dict(a=float(a), maximum_partner_rate_over_H=float(upper_partner_density*sigma_v/hubble)))
        rows.append(dict(r=r, supplied_mass_eV=mass, normalized_mass_curvature=m2,
                         maximum_heavy_fraction_for_positive_matter_curvature=heavy_fraction_limit,
                         critical_a=acrit, critical_z=1/acrit-1, curvature_checks=curvature_checks,
                         symmetric_AP_max_difference=float(np.max(abs(prediction-base_prediction))),
                         muon_EM=muon, partial_rmse=float(score), delta_partial_rmse=float(delta_score),
                         delta_partial_rmse_vs_same_charged_static_baseline=0.0,
                         delta_chi2=delta_chi, fractional_rmse_reduction=float(-delta_score/baseline),
                         linear_phase_transfer=float(linear.y[0,-1]), fixed_seed_checks=nonlinear,
                         heavy_to_two_phase_contact_sigma_v_GeV_minus2=float(sigma_v),
                         heavy_contact_channel_rate_checks=rates,
                         internal_Gibbs_redistribution_diagnostic=thermal))
    return dict(hypothesis='Frozen light-doublet populations (0,1,1), reflection-symmetric phase and zero velocity.',
                baseline_partial_rmse=baseline, cases=rows, fitted_parameters=[],
                score_scope='Six original AP data plus fixed muon EM subset; zero cross covariance.',
                state_generation_derived=False, absolute_abundance_derived=False,
                full_joint_rmse=None, independent_holdout=False,
                conclusion='Matter stabilization is possible for frozen populations; thermal redistribution can remove it. Tiny partial score improvement comes only from absolute EM term.')


if __name__ == '__main__':
    result = run()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
