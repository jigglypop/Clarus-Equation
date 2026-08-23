"""Branch characterisation: which sub-population of the scan passes R4 / R3a,
and does q_theta stay above w_min there (Theorem 1 hypothesis)?"""
import os, csv, json
import numpy as np
import sim
OUTDIR = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(OUTDIR, 'scan.csv'))))
F = lambda k: np.array([float(r[k]) for r in rows])
R1, R3a, R4 = F('R1'), F('R3a'), np.abs(F('R4_drift100'))
out = {}
sel = {'R3a_in_band': (R3a >= .10) & (R3a <= .25),
       'R4_pass': R4 < .05,
       'R1_in_band': (R1 >= .02) & (R1 <= .08)}
for name, m in sel.items():
    idx = np.flatnonzero(m)[:6]
    det = []
    for i in idx:
        p = {k: float(rows[i][k]) for k in ['eta', 'lam0', 'theta', 'rho_inf', 'kappa', 'T_m']}
        r = sim.run(**p, N_E=2000, T=700, seed=118001, trace_top=True)
        q = r['q'][500:700]
        det.append(dict(params=p, R1=float(R1[i]), R3a=float(R3a[i]), R4=float(R4[i]),
                        frac_days_q_below_wmin=float(np.mean(q < 1.0)),
                        q_min=float(np.nanmin(q)), q_mean=float(np.nanmean(q)),
                        N_500=float(r['N'][500]), N_699=float(r['N'][699]),
                        M_500=float(r['M'][500]), M_699=float(r['M'][699]),
                        Mtop_frac_699=float(r['Mtop'][699] / max(r['M'][699], 1e-9)),
                        deaths_per_day_adult=float(r['rem'][500:700].mean())))
    out[name] = dict(n=int(m.sum()), detail=det)
json.dump(out, open(os.path.join(OUTDIR, 'branch.json'), 'w'), indent=1, default=float)
for k, v in out.items():
    print('==', k, 'n=', v['n'])
    for d in v['detail']:
        print('  R1=%.4g R3a=%.4g R4=%.3g fq<1=%.3f qmin=%.3g qmean=%.3g N=%.0f->%.0f M=%.4g->%.4g Mtopfrac=%.3f d/day=%.3g'
              % (d['R1'], d['R3a'], d['R4'], d['frac_days_q_below_wmin'], d['q_min'], d['q_mean'],
                 d['N_500'], d['N_699'], d['M_500'], d['M_699'], d['Mtop_frac_699'], d['deaths_per_day_adult']))
        print('     ', {kk: round(vv, 5) for kk, vv in d['params'].items()})
