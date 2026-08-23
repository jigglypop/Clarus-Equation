"""Apparatus check: does the N_E=2000 search reduction change any verdict vs the
contract's N_E=1e4? Also seed robustness at the best-effort point."""
import os, json
import numpy as np
import sim
OUTDIR = os.path.dirname(os.path.abspath(__file__))
pts = {
 'best_refined': dict(eta=3.0097507, lam0=0.0517045, theta=0.85, rho_inf=0.00056516, kappa=25.87485, T_m=147.14792),
 'R1band_opt': dict(eta=19.93332, lam0=0.05553, theta=0.85, rho_inf=0.00051, kappa=7.24822, T_m=284.33722),
 'R3aband_pt': dict(eta=0.35979, lam0=0.85239, theta=0.81095, rho_inf=0.07503, kappa=33.90731, T_m=5.92773),
}
keys = ['R1', 'R2a_new', 'R2b_new', 'R2a_pop', 'R2b_pop', 'R3a', 'R4_drift100', 'R5', 'R7', 'Nad']
out = {}
for name, p in pts.items():
    rec = {}
    for NE in (2000, 10000):
        for seed in (118001, 118101):
            g = sim.gates(sim.run(**p, N_E=NE, T=700, seed=seed))
            rec['NE%d_s%d' % (NE, seed)] = {k: float(g[k]) for k in keys}
    out[name] = dict(params=p, runs=rec)
json.dump(out, open(os.path.join(OUTDIR, 'nescale.json'), 'w'), indent=1, default=float)
for name, d in out.items():
    print('==', name)
    for tag, m in d['runs'].items():
        print('  %-16s ' % tag + ' '.join('%s=%.4g' % (k, m[k]) for k in keys))
