"""Conflict frontier: maximize R3a subject to R1 in [0.02,0.08] (and R3b), and
symmetrically minimize |R1| deviation subject to R3a in band. Multi-start local
search, fixed seed. Also reports R2b under both readings at each optimum."""
import os
OUTDIR = os.path.dirname(os.path.abspath(__file__))
import json
import numpy as np
import sim

rng = np.random.default_rng(4242)
STARTS = [dict(eta=3.0, lam0=0.052, theta=0.85, rho_inf=5.7e-4, kappa=26, T_m=147),
          dict(eta=20, lam0=0.2, theta=0.85, rho_inf=0.01, kappa=10, T_m=60),
          dict(eta=1.5, lam0=0.12, theta=0.8, rho_inf=0.002, kappa=15, T_m=100),
          dict(eta=50, lam0=0.3, theta=0.6, rho_inf=0.02, kappa=5, T_m=40),
          dict(eta=8, lam0=0.02, theta=0.85, rho_inf=1e-3, kappa=20, T_m=120)]


def ev(p):
    g = sim.gates(sim.run(N_E=2000, T=700, seed=118001, **p))
    return g


def obj_maxR3a(g):
    pen = 0.0
    if g['R1'] < 0.02:
        pen += (np.log(0.02 / max(g['R1'], 1e-6))) ** 2
    if g['R1'] > 0.08:
        pen += (np.log(g['R1'] / 0.08)) ** 2
    return -np.log(max(g['R3a'], 1e-9)) + 10 * pen


def obj_minR1(g):
    pen = 0.0
    if g['R3a'] < 0.10:
        pen += (np.log(0.10 / max(g['R3a'], 1e-9))) ** 2
    if g['R3a'] > 0.25:
        pen += (np.log(g['R3a'] / 0.25)) ** 2
    return np.log(max(g['R1'], 1e-6) / 0.08) ** 2 + 10 * pen


def descend(p0, obj, iters=150):
    best_p, gb = dict(p0), ev(p0)
    fb = obj(gb)
    sc = 0.3
    for it in range(iters):
        q = {k: v * float(np.exp(rng.normal(0, sc))) for k, v in best_p.items()}
        q['theta'] = min(max(q['theta'], 0.02), 0.85)
        q['rho_inf'] = min(q['rho_inf'], 0.6)
        q['lam0'] = min(q['lam0'], 0.95)
        q['kappa'] = min(q['kappa'], 300)
        g = ev(q)
        f = obj(g)
        if f < fb:
            fb, best_p, gb = f, q, g
        if it % 50 == 49:
            sc *= 0.5
    return fb, best_p, gb


res = {'maxR3a_st_R1band': [], 'minR1dev_st_R3aband': []}
for st in STARTS:
    f, p, g = descend(st, obj_maxR3a)
    res['maxR3a_st_R1band'].append(dict(params=p, R1=g['R1'], R3a=g['R3a'],
                                        R2b_new=g['R2b_new'], R2b_pop=g['R2b_pop'],
                                        R4=g['R4_drift100'], R5=g['R5'], R7=g['R7'], q_ad=g['q_ad'], Nad=g['Nad']))
    f, p, g = descend(st, obj_minR1)
    res['minR1dev_st_R3aband'].append(dict(params=p, R1=g['R1'], R3a=g['R3a'],
                                           R2b_new=g['R2b_new'], R2b_pop=g['R2b_pop'],
                                           R4=g['R4_drift100'], R5=g['R5'], R7=g['R7'], q_ad=g['q_ad'], Nad=g['Nad']))
ok = [r for r in res['maxR3a_st_R1band'] if 0.02 <= r['R1'] <= 0.08]
res['summary'] = dict(
    n_local_evals=len(STARTS) * 2 * 151,
    sup_R3a_given_R1_in_band=max([r['R3a'] for r in ok], default=None),
    inf_R1_given_R3a_in_band=min([r['R1'] for r in res['minR1dev_st_R3aband'] if 0.10 <= r['R3a'] <= 0.25], default=None))
json.dump(res, open(os.path.join(OUTDIR,'conflict.json'), 'w'), indent=1, default=float)
print(json.dumps(res['summary'], indent=1, default=float))
for k in res:
    if k == 'summary':
        continue
    print('==', k)
    for r in res[k]:
        print('  R1=%.4g R3a=%.4g R2b_new=%.3g R2b_pop=%.3g R4=%.3g R5=%.3g R7=%.3g q_ad=%.4g Nad=%.4g'
              % (r['R1'], r['R3a'], r['R2b_new'], r['R2b_pop'], r['R4'], r['R5'], r['R7'], r['q_ad'], r['Nad']))
        print('     ', {kk: round(vv, 5) for kk, vv in r['params'].items()})
