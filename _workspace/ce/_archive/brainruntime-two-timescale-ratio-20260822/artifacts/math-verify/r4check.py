"""Is any R4-passing point a genuine steady state (stable N) and does any point
pass R4 together with R1? Also: exempt mass fraction f_top vs R3a = lam0*(1-f_top)."""
import os, csv, json
import numpy as np
import sim
OUTDIR = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(OUTDIR, 'scan.csv'))))
F = lambda k: np.array([float(r[k]) for r in rows])
R1, R3a, R4 = F('R1'), F('R3a'), np.abs(F('R4_drift100'))
idx = np.flatnonzero(R4 < 0.05)
res = []
for i in idx:
    p = {k: float(rows[i][k]) for k in ['eta', 'lam0', 'theta', 'rho_inf', 'kappa', 'T_m']}
    r = sim.run(**p, N_E=2000, T=700, seed=118001, trace_top=True)
    N0, N1 = r['N'][500], r['N'][699]
    ft = float(np.nanmean(r['Mtop'][500:700] / np.maximum(r['M'][500:700], 1e-9)))
    res.append(dict(i=int(i), R1=float(R1[i]), R3a=float(R3a[i]), R4=float(R4[i]),
                    N500=float(N0), N699=float(N1),
                    N_rel_drift=float((N1 - N0) / max(N0, 1e-9)),
                    f_top=ft, lam0=p['lam0'], check_R3a=float(p['lam0'] * (1 - ft))))
stable = [d for d in res if abs(d['N_rel_drift']) < 0.10]
out = dict(n_R4_pass=len(res),
           n_R4_pass_and_N_stable_10pct=len(stable),
           n_R4_pass_and_R1_band=sum(1 for d in res if 0.02 <= d['R1'] <= 0.08),
           median_abs_N_drift=float(np.median([abs(d['N_rel_drift']) for d in res])),
           stable_detail=stable,
           f_top_stats=dict(min=float(min(d['f_top'] for d in res)),
                            median=float(np.median([d['f_top'] for d in res])),
                            max=float(max(d['f_top'] for d in res))),
           R3a_identity_max_abs_err=float(max(abs(d['check_R3a'] - d['R3a']) for d in res)))
# f_top over the R1-band points
idx1 = np.flatnonzero((R1 >= 0.02) & (R1 <= 0.08))
ft1 = []
for i in idx1[:40]:
    p = {k: float(rows[i][k]) for k in ['eta', 'lam0', 'theta', 'rho_inf', 'kappa', 'T_m']}
    r = sim.run(**p, N_E=2000, T=700, seed=118001, trace_top=True)
    ft1.append(float(np.nanmean(r['Mtop'][500:700] / np.maximum(r['M'][500:700], 1e-9))))
out['f_top_R1band'] = dict(n=len(ft1), min=float(min(ft1)), median=float(np.median(ft1)), max=float(max(ft1)))
json.dump(out, open(os.path.join(OUTDIR, 'r4check.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != 'stable_detail'}, indent=1, default=float))
print('stable_detail:', json.dumps(out['stable_detail'], indent=1, default=float)[:1500])
