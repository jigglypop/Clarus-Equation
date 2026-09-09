"""PD33 reconstruction and the literal-shared-epsilon photon response.

No photon amplitude is fitted. The sech pulse tests the solver, not CE's
cosmological state. Actual backgrounds are the six previously fixed cold
diagnostics; their photons use a finite-time instantaneous vacuum convention.
"""
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_ivp

from common_spectrum_muon import ALPHA, Q2
from ce_symmetric_small_f_stability import ce, StableBackground, StableLightSpectrum


def optical_checks():
    rng = np.random.default_rng(33)
    errors = []
    for _ in range(2000):
        markers = rng.normal(size=(2, 3)) + 1j*rng.normal(size=(2, 3))
        markers /= np.linalg.norm(markers, axis=1)[:, None]
        p = rng.random()
        overlap = np.vdot(markers[0], markers[1])
        visibility = 2*np.sqrt(p*(1-p))*abs(overlap)
        helstrom = p*np.outer(markers[0], markers[0].conj()) - (1-p)*np.outer(markers[1], markers[1].conj())
        distinguishability = np.sum(abs(np.linalg.eigvalsh(helstrom)))
        errors.append(abs(visibility**2+distinguishability**2-1))
    assert max(errors) < 5e-15
    bs = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
    interference_error = 0.
    for phase in np.linspace(0, 2*np.pi, 101):
        state = bs @ np.diag([1, np.exp(1j*phase)]) @ bs @ np.array([1, 0])
        interference_error = max(interference_error, float(np.max(abs(abs(state)**2-[np.sin(phase/2)**2, np.cos(phase/2)**2]))))
    assert interference_error < 2e-15
    # A detector map must be a contraction; linearity alone is insufficient.
    raw = rng.normal(size=(7, 256))+1j*rng.normal(size=(7, 256))
    K = .8*raw/np.linalg.svd(raw, compute_uv=False)[0]
    E0 = np.eye(256)-K.conj().T@K
    minimum = float(np.linalg.eigvalsh(E0)[0])
    psi = rng.normal(size=256)+1j*rng.normal(size=256)
    psi /= np.linalg.norm(psi)
    total = float(np.vdot(K@psi, K@psi).real + np.vdot(psi, E0@psi).real)
    assert minimum > 0 and abs(total-1) < 1e-14
    return dict(pure_complementarity_max_error=max(errors),
                mach_zehnder_probability_max_error=interference_error,
                no_click_effect_min_eigenvalue=minimum, total_probability=total,
                scope='Born detection and a specified instrument assumed; not derived from CE.')


def pulse_checks():
    rows = []
    for delta in [1e-3, 5e-4, 1e-4]:
        for k in [.25, .5, 1., 2., 3.]:
            def pump(eta):
                f = 1/np.cosh(eta)**2
                return -2*delta*f*np.tanh(eta)/(1+delta*f)
            def rhs(eta, y):
                W = pump(eta)
                return [W*np.exp(2j*k*eta)*y[1], W*np.exp(-2j*k*eta)*y[0]]
            sol = solve_ivp(rhs, (-16., 16.), np.array([1., 0.], complex),
                            method='DOP853', rtol=2e-11, atol=2e-14, max_step=.08)
            assert sol.success
            alpha, beta = sol.y[:, -1]
            born = 4j*np.pi*delta*k*k/np.sinh(np.pi*k)
            invariant_error = float(abs(abs(alpha)**2-abs(beta)**2-1))
            # Independent second-order canonical mode equation.
            def mode_rhs(eta, y):
                f = 1/np.cosh(eta)**2
                U = delta*(4*f-6*f*f)/(1+delta*f)
                return [y[1], -(k*k-U)*y[0]]
            v0 = np.exp(16j*k)/np.sqrt(2*k)
            modes = solve_ivp(mode_rhs, (-16., 16.), [v0, (-1j*k+pump(-16))*v0],
                              method='DOP853', rtol=2e-12, atol=2e-14, max_step=.06)
            assert modes.success
            v, vp = modes.y[:, -1]
            momentum = vp-pump(16)*v
            beta2 = np.exp(-16j*k)*(np.sqrt(k)*v-1j*momentum/np.sqrt(k))/np.sqrt(2)
            ode_error = float(abs(beta-beta2))
            L = 2*np.log1p(delta)
            assert invariant_error < 1e-12 and ode_error < 2e-11
            assert abs(beta) <= np.sinh(L)
            rows.append(dict(delta=delta, k_tau=k, number=float(abs(beta)**2),
                             born_beta_relative_error=float(abs(beta/born-1)),
                             bogoliubov_invariant_error=invariant_error,
                             independent_ode_beta_difference=ode_error))
    for index in range(5):
        assert rows[index+5]['born_beta_relative_error'] < .51*rows[index]['born_beta_relative_error']
    with mp.workdps(60):
        integral = 16*mp.quad(lambda x: x**7/mp.sinh(mp.pi*x)**2, [0, 1, 4, mp.inf])
        closed = 1260*mp.zeta(7)/mp.pi**8
        energy_error = abs(integral/closed-1)
        assert energy_error < mp.mpf('1e-50')
        energy = dict(coefficient=mp.nstr(closed, 30), relative_quadrature_error=mp.nstr(energy_error, 5))
    return dict(rows=rows, born_energy=energy,
                scope='tau=1 diagnostic pulse; delta is a solver input, never a fitted CE amplitude.')


def determinant_check():
    # Verify cancellation at the physical hierarchy without double precision
    # eigenvalue rounding silently turning a nonzero threshold into zero.
    with mp.workdps(180):
        r = mp.mpf('.35')*(mp.mpf('.014414')/mp.mpf('1e12'))**2
        theta = mp.mpf('.7')
        xs = lambda t: [1+2*r*mp.cos((t+2*mp.pi*j)/3) for j in range(3)]
        direct = sum(mp.log(x) for x in xs(theta))-sum(mp.log(x) for x in xs(mp.pi))
        stable = mp.log1p(2*r**3*(1+mp.cos(theta))/((1-2*r)*(1+r)**2))
        error = abs(direct/stable-1)
        assert error < mp.mpf('1e-85')
        return dict(r_H=mp.nstr(r, 25), log_determinant_ratio=mp.nstr(stable, 25),
                    relative_eigenvalue_crosscheck_error=mp.nstr(error, 5))


def background_checks():
    old = json.loads(Path(__file__).with_name('ce_symmetric_bao_ruler.json').read_text())
    # ln I = const + 1/2 ln(alpha_inverse). Its leading derivative is
    # c*r_H^3*sin(theta). Keep r_H^3 outside the integration to avoid loss.
    c = ALPHA*Q2/(12*np.pi)
    rows = []
    for previous in old['rows']:
        r, mass, seed = previous['r'], previous['sqrt_s_eV'], previous['theta_initial']
        p = ce.Parameters(r, seed, f=1/30, s_over_Mp2=(mass*1e-9/2.435e18)**2)
        sp = StableLightSpectrum(p)
        bg = StableBackground(p, sp)
        rH = r*(mass/1e12)**2  # same epsilon, 1 TeV charged mass; all masses in eV
        dmin = (1-2*rH)*(1+rH)**2
        b = Q2/(12*np.pi)
        # Rigorous prefactor bound for every theta, without subtracting ~1s.
        alpha_max = ALPHA/(1-ALPHA*b*np.log1p(4*rH**3/dmin))
        cmax = alpha_max*b/dmin
        sample = bg.sol.sol(np.linspace(bg.Ni, 0., 2001))[0]
        scale = max(float(np.max(abs(sample)))**2, 1e-30)
        kvals = np.array([.1, 1., 10.])
        def rhs(N, y):
            theta, velocity = bg.sol.sol(N)
            eta = y[0].real
            source = np.sin(theta)*velocity/scale
            phase = np.exp(-2j*kvals*eta)
            return np.r_[np.exp(-N)/bg.quantities(N)['H'],
                         source*phase, abs(source)]
        def integrate(step):
            sol = solve_ivp(rhs, (bg.Ni, 0.), np.zeros(5, complex), method='DOP853',
                            rtol=2e-9, atol=2e-12, max_step=step)
            assert sol.success
            return sol.y[:, -1]
        result = integrate(.02)
        check = integrate(.01)
        convergence = float(np.max(abs(result-check))/max(np.max(abs(check)), 1e-30))
        assert convergence < 2e-7
        # Integral with absolute value is independently evaluated per existing
        # background step. It also includes any early oscillations.
        total_variation = sum(quad(lambda N: abs(np.sin(bg.sol.sol(N)[0])*bg.sol.sol(N)[1])/scale,
                                   lo, hi, epsabs=1e-13, epsrel=1e-9)[0]
                              for lo, hi in zip(bg.sol.t[:-1], bg.sol.t[1:]))*scale
        assert abs(total_variation-result[4].real*scale) < max(1e-25, total_variation*2e-7)
        L = cmax*rH**3*total_variation
        occupation_bound = float(np.sinh(L)**2)
        leading_beta = c*rH**3*scale*result[1:4]
        occupations = abs(leading_beta)**2
        assert np.max(occupations) <= occupation_bound*(1+2e-7)
        rows.append(dict(r=r, sqrt_s_eV=mass, theta_initial=seed,
                         theta_final=float(bg.sol.y[0, -1]), r_H=rH,
                         integrated_abs_sin_theta_dtheta=total_variation,
                         integral_abs_dlnI_upper_bound=L,
                         vacuum_number_per_mode_upper_bound=occupation_bound,
                         k_in_background_conformal_units=kvals.tolist(),
                         leading_vacuum_number_per_helicity=occupations.tolist(),
                         integration_step_relative_difference=convergence,
                         previous_partial_rmse_14=previous['partial_rmse_14']))
    return dict(rows=rows, assumptions=['same six supplied late cold states, a=0.01 to 1',
                'one-loop local parity-even Maxwell action, k_gamma positive',
                'initial instantaneous photon vacuum, no plasma or photon backreaction',
                'm_H=1 TeV, same dimensionful epsilon as neutral sector',
                'k in sqrt(U_top)/M_P conformal units; finite-time final particle convention'],
                new_photon_likelihood=None, recomputed_joint_rmse=None,
                scope='Born occupations and an all-k vacuum occupation bound, not a CMB spectrum or energy-density bound.')


if __name__ == '__main__':
    result = dict(optics=optical_checks(), pulse=pulse_checks(),
                  determinant=determinant_check(), common_CE_photons=background_checks(),
                  fitted_parameters=[], full_joint_rmse=None)
    source = Path(__file__).resolve().parents[1]/'paper'/'참조'/'CE_PD33_공유원문_2026-09-10.txt'
    result['source'] = dict(path=source.relative_to(Path(__file__).resolve().parents[1]).as_posix(),
                          sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          status='User-pasted manuscript; its linked ZIP/code were not supplied. Independently reconstructed here.')
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
