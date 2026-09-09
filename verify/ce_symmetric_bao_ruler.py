"""Carry the six existing cold-state diagnostics to all 13 BAO entries.

One ruler calibration on the null's development data; no candidate refit.
Physical r_s is held fixed, not A=c/(H0*r_s). AP is a diagnostic derived
from these BAO data and is NOT added again to the 14-entry score.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from ce_symmetric_small_f_stability import ce, StableBackground, StableLightSpectrum
from common_spectrum_muon import exact_em


def run():
    z, observed, kind, cov, za, ya, ca = ce.load_data()
    chol = np.linalg.cholesky(cov)
    keys = {'DM_over_rs': 'DM_H0_over_c', 'DH_over_rs': 'DH_H0_over_c',
            'DV_over_rs': 'DV_H0_over_c'}

    def normalized_distances(bg):
        ds = {zz: bg.distances(zz) for zz in set(z)}
        return np.array([ds[zz][keys[kk]] for zz, kk in zip(z, kind)])

    # Same theta=0 geometry as the symmetric state, without a numerical seed.
    null = ce.Background(ce.Parameters(.35, 0.))
    b0 = normalized_distances(null)
    wb = np.linalg.solve(chol, b0)
    wy = np.linalg.solve(chol, observed)
    A0 = float(wb @ wy / (wb @ wb))
    mass0 = .014414  # sqrt(s_D) in eV; inherited benchmark, not optimized here
    top0 = null.sp.top * (mass0 * 1e-9)**4
    muon = exact_em(1000., .35 * (mass0 * 1e-9)**2 / 1e6, 0.)
    mu_residual = (muon - 385e-12) / (np.hypot(145, 620) * 1e-12)

    def statistics(pred):
        residual = np.linalg.solve(chol, pred - observed)
        chi = float(residual @ residual)
        chi_direct = float((pred-observed) @ np.linalg.solve(cov, pred-observed))
        assert abs(chi-chi_direct) < 1e-10
        return {'bao_chi2': chi, 'bao_rmse': float(np.sqrt(chi/13)),
                'partial_rmse_14': float(np.sqrt((chi+mu_residual**2)/14))}

    baseline = dict(A=A0, **statistics(A0*b0))
    null_residual = np.linalg.solve(chol, A0*b0-observed)
    unit_null = wb/np.linalg.norm(wb)
    assert abs(null_residual @ unit_null) < 1e-10
    rows = []
    for r, mass in [(.15, .027615), (.35, .014414)]:
        for seed in [1e-12, 1e-10, 1e-8]:
            p = ce.Parameters(r, seed, f=1/30,
                              s_over_Mp2=(mass*1e-9/2.435e18)**2)
            sp = StableLightSpectrum(p)
            bg = StableBackground(p, sp)
            # Internal H uses units sqrt(U_top)/M_P. Restore that unit
            # before comparing H0 and transporting a fixed physical ruler.
            top = sp.top * (mass*1e-9)**4
            Hratio = float(bg.H0/null.H0 * np.sqrt(top/top0))
            A = A0/Hratio
            pred = A*normalized_distances(bg)
            stat = statistics(pred)
            change = np.linalg.solve(chol, pred-A0*b0)
            parallel = float(change @ unit_null)
            perpendicular = change-parallel*unit_null
            shape_delta = float(2*null_residual @ perpendicular + perpendicular @ perpendicular)
            assert abs(stat['bao_chi2']-baseline['bao_chi2']-shape_delta-parallel**2) < 1e-10
            # Independent normalization: integrate 1/H_internal directly.
            factor = A0*null.H0*np.sqrt(top0/top)
            ds = {}
            for zz in set(z):
                dm = quad(lambda zz1: 1/bg.quantities(-np.log1p(zz1))['H'],
                          0., zz, epsabs=1e-10, epsrel=1e-10)[0]
                dh = 1/bg.quantities(-np.log1p(zz))['H']
                ds[zz] = {'DM_over_rs': factor*dm, 'DH_over_rs': factor*dh,
                          'DV_over_rs': factor*(zz*dm*dm*dh)**(1/3)}
            direct = np.array([ds[zz][kk] for zz, kk in zip(z, kind)])
            normalization_error = float(np.max(abs(direct-pred)))
            assert normalization_error < 1e-8
            ap = np.array([bg.distances(zz)['F_AP'] for zz in za])
            ar = np.linalg.solve(np.linalg.cholesky(ca), ap-ya)
            row = dict(r=r, sqrt_s_eV=mass, theta_initial=seed, A=A,
                       physical_H0_ratio_to_null=Hratio, **stat,
                       delta_rmse_14=stat['partial_rmse_14']-baseline['partial_rmse_14'],
                       AP_rmse_diagnostic_only=float(np.sqrt(ar@ar/6)),
                       prediction=pred.tolist(),
                       direct_distance_max_difference=normalization_error)
            row['chi2_change_decomposition'] = dict(
                along_null_distance_squared=parallel**2,
                perpendicular_change=shape_delta,
                note='Algebraic decomposition at the calibrated null; no candidate fit.')
            if seed == 1e-8:
                other = StableBackground(p, sp, method='Radau')
                Aother = A0/(other.H0/null.H0*np.sqrt(top/top0))
                other_pred = Aother*normalized_distances(other)
                err = float(np.max(abs(other_pred-pred)))
                score_error = statistics(other_pred)['partial_rmse_14']-stat['partial_rmse_14']
                assert err < 1e-7 and abs(score_error) < 1e-8
                row['independent_solver'] = dict(prediction_max_difference=err,
                                                rmse_14_difference=score_error)
            rows.append(row)
    return dict(baseline=baseline, rows=rows, observations=observed.tolist(),
                kinds=list(kind), z=z.tolist(), muon_EM_component=muon,
                muon_standardized_residual=float(mu_residual),
                calibration_parameters=['one null A=c/(H0*r_s) on the same BAO data'],
                candidate_fitted_parameters=[],
                assumptions=['fixed physical sound horizon, not derived early evolution',
                             'inherited masses, density and cold symmetric populations',
                             'same charged EM term in null and every candidate',
                             'BAO and muon blocks independent; AP not double-counted'],
                independent_holdout=False, full_joint_rmse=None)


if __name__ == '__main__':
    result = run()
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'baseline': result['baseline'], 'rows': [
        {k: v for k, v in row.items() if k != 'prediction'} for row in result['rows']]}, indent=2))
