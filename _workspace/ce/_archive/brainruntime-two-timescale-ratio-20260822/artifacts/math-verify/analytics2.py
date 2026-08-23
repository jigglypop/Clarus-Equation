"""Extra checks: (i) rigorous top-k submartingale, (ii) p_hit / wake:sleep
design-constant sensitivity, (iii) noise-normalised identifiability,
(iv) R6 evaluation, (v) corrected newborn-survival ceiling table."""
import os, json
import numpy as np
import sim
OUTDIR = os.path.dirname(os.path.abspath(__file__))
P = 0.1
OUT = {}

# (v) corrected ceiling: valid only when the no-hit death day <= 8
def j0(lam0, w0=1.2, wmin=1.0):
    a, w, j = 1 - lam0, w0, 0
    while w >= wmin:
        w *= a; j += 1
        if j > 10000: return np.inf
    return j
T = []
for lam0 in [0.02, 0.05, 0.08, 0.10, 0.12, 0.167, 0.25, 0.5, 0.9]:
    j = j0(lam0); win = j + 1
    T.append(dict(lam0=lam0, death_day_no_hit=j + 1, hit_window=win,
                  bound_applies=bool(win <= 8),
                  S_new8_ceiling=float(1 - (1 - P) ** win) if win <= 8 else 1.0,
                  R3a_ceiling=lam0))
OUT['B2_corrected_ceiling'] = T

# (i) rigorous top-k submartingale S_k(t) with k fixed over the adult window
S = []
for p in [dict(eta=20, lam0=0.4, theta=0.7, rho_inf=0.02, kappa=10, T_m=60),
          dict(eta=5, lam0=0.2, theta=0.85, rho_inf=0.05, kappa=8, T_m=40),
          dict(eta=3, lam0=0.052, theta=0.85, rho_inf=5.7e-4, kappa=26, T_m=147)]:
    r = sim.run(**p, N_E=2000, T=700, seed=118001, trace_top=True, keep_topk=True)
    Sk = r['Sk'][500:700]
    d = np.diff(Sk)
    k = r['k_fixed']
    S.append(dict(params=p, k=int(k), Sk_500=float(Sk[0]), Sk_699=float(Sk[-1]),
                  n_decrease_days=int((d < -1e-9).sum()), mean_dSk=float(d.mean()),
                  eta_p_k=float(p['eta'] * P * k),
                  q_min_over_wmin=float(np.nanmin(r['q'][500:700]))))
OUT['C2_topk_submartingale'] = S

# (ii) design-constant sensitivity: p_hit
base = dict(eta=20, lam0=0.4, theta=0.7, rho_inf=0.02, kappa=10, T_m=60)
PH = []
for ph in (0.02, 0.05, 0.1, 0.3, 0.6):
    g = sim.gates(sim.run(**base, N_E=2000, T=700, seed=118001, p_hit=ph))
    PH.append(dict(p_hit=ph, R1=float(g['R1']), R2b_new=float(g['R2b_new']),
                   R3a=float(g['R3a']), R5=float(g['R5']), wbar=float(g['wbar'])))
OUT['D2_p_hit_sensitivity'] = PH

# (iii) noise-normalised identifiability
ref = dict(eta=20, lam0=0.25, theta=0.7, rho_inf=0.05, kappa=10, T_m=60)
names = ['eta', 'lam0', 'theta', 'rho_inf', 'kappa', 'T_m']
mets = ['R1', 'R2a_new', 'R2b_new', 'R3a', 'R5', 'R7']
def vec(p, seed=118001):
    g = sim.gates(sim.run(**p, N_E=4000, T=700, seed=seed))
    return np.array([max(g[m], 1e-6) for m in mets])
noise = np.std([np.log(vec(ref, seed=s)) for s in (118001, 118002, 118003, 118004, 118005, 118006)], axis=0)
h = 0.10
J = np.zeros((len(mets), len(names)))
for j, nm in enumerate(names):
    pp = dict(ref); pp[nm] = min(ref[nm] * (1 + h), 0.85) if nm == 'theta' else ref[nm] * (1 + h)
    pm = dict(ref); pm[nm] = ref[nm] * (1 - h)
    J[:, j] = (np.log(vec(pp)) - np.log(vec(pm))) / (np.log(pp[nm]) - np.log(pm[nm]))
Jn = J / np.maximum(noise, 1e-9)[:, None]
s_raw = np.linalg.svd(J, compute_uv=False)
u, s_n, vt = np.linalg.svd(Jn)
OUT['E2_identifiability'] = dict(
    metrics=mets, params=names, seed_noise_log_sd=dict(zip(mets, noise.tolist())),
    sv_raw=s_raw.tolist(), sv_noise_normalised=s_n.tolist(),
    cond_raw=float(s_raw[0] / s_raw[-1]), cond_norm=float(s_n[0] / s_n[-1]),
    eff_rank_at_1sigma=int((s_n > 1.0).sum()),
    weakest_direction=dict(zip(names, vt[-1].tolist())),
    second_weakest=dict(zip(names, vt[-2].tolist())))

# (iv) R6 at three points (learning day inside the adult window)
R6 = []
for p in [dict(eta=3.0, lam0=0.052, theta=0.85, rho_inf=5.7e-4, kappa=26, T_m=147),
          dict(eta=20, lam0=0.4, theta=0.7, rho_inf=0.02, kappa=10, T_m=60),
          dict(eta=1.5, lam0=0.12, theta=0.8, rho_inf=0.002, kappa=15, T_m=100)]:
    base_s, ev_s = [], []
    for seed in (118001, 118002, 118003, 118004):
        r0 = sim.run(**p, N_E=4000, T=620, seed=seed)
        r1 = sim.run(**p, N_E=4000, T=620, seed=seed, learn_day=600, learn_mult=5.0)
        b = np.nanmean(r0['nb_surv8'][560:600])
        e = r1['nb_surv8'][600]
        if np.isfinite(b) and np.isfinite(e):
            base_s.append(b); ev_s.append(e)
    R6.append(dict(params=p, baseline=float(np.mean(base_s)), event=float(np.mean(ev_s)),
                   ratio=float(np.mean(ev_s) / max(np.mean(base_s), 1e-9))))
OUT['R6_learning_event'] = R6

json.dump(OUT, open(os.path.join(OUTDIR, 'analytics2.json'), 'w'), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
