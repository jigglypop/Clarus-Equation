"""(a)(c)(d)(e) analytic checks for BA-TS1."""
import os
OUTDIR = os.path.dirname(os.path.abspath(__file__))
import json
import numpy as np
import sim

P = 0.1
OUT = {}

# ---------- A. model-free band arithmetic: R1 vs R2b (population reading) ----
# steady state: monthly removals = monthly formations = R1/2 per surviving edge.
# Worst case (all removals inside one 8-day window): S8 >= 1 - R1/2*... use
# both the concentrated bound and the exponential (homogeneous) estimate.
A = {}
for R1 in (0.02, 0.04, 0.08):
    d = R1 / 2.0
    A['R1=%.3f' % R1] = dict(S8_worstcase=1 - d, S8_uniform=(1 - d) ** (8 / 30))
OUT['A_R1_to_S8pop'] = A

# ---------- B. R3a -> lambda0 -> newborn 8-day survival ceiling -------------
def j0(lam0, w0=1.2, wmin=1.0):
    a = 1 - lam0
    j = 0
    w = w0
    while w >= wmin:
        w *= a
        j += 1
        if j > 10000:
            return np.inf
    return j  # first post-sleep day (from birth) with w < wmin, no hits

B = []
for lam0 in [0.02, 0.05, 0.08, 0.10, 0.12, 0.167, 0.2, 0.3, 0.5, 0.8]:
    j = j0(lam0)
    win = j + 1  # tau_e = 2 -> need j0 and j0+1 both low => hit window = j0+1 days
    B.append(dict(lam0=lam0, j0=j, hit_window=win,
                  S_new8_ceiling=1 - (1 - P) ** win,
                  R3a_max_from_lam0=lam0))
OUT['B_lam0_vs_newborn_survival'] = B

# ---------- C. exempt-set monotonicity + R3a ~ theta/((1-theta) t) ----------
C = []
pts = [dict(eta=20, lam0=0.4, theta=0.7, rho_inf=0.02, kappa=10, T_m=60),
       dict(eta=5, lam0=0.2, theta=0.85, rho_inf=0.05, kappa=8, T_m=40),
       dict(eta=50, lam0=0.15, theta=0.5, rho_inf=0.01, kappa=5, T_m=80),
       dict(eta=3, lam0=0.12, theta=0.3, rho_inf=0.1, kappa=3, T_m=30)]
for p in pts:
    r = sim.run(**p, N_E=2000, T=700, seed=118001, trace_top=True)
    Mt = r['Mtop']; N = r['N']
    d = np.diff(Mt[200:700])
    gain = P * (1 - p['theta']) * N[200:699]
    ratio = [float(np.nanmean(r['r3a'][t - 20:t + 20] * (1 - p['theta']) * t / p['theta']))
             for t in (100, 300, 600)]
    C.append(dict(params=p, Mtop_500=float(Mt[500]), Mtop_699=float(Mt[699]),
                  M_500=float(r['M'][500]), M_699=float(r['M'][699]),
                  frac_days_Mtop_decrease=float(np.mean(d < 0)),
                  mean_dMtop=float(np.mean(d)), mean_pred_gain=float(np.mean(gain)),
                  R3a_t100=float(np.nanmean(r['r3a'][80:120])),
                  R3a_t300=float(np.nanmean(r['r3a'][280:320])),
                  R3a_t600=float(np.nanmean(r['r3a'][580:620])),
                  bound_c_at_100_300_600=ratio,
                  q_over_wmin_adult=float(np.nanmean(r['q'][500:700])),
                  R1=float(sim.gates(r)['R1'])))
OUT['C_exempt_growth'] = C

# ---------- D. gauge check: homogeneity of degree 1 -------------------------
base = dict(eta=20, lam0=0.4, theta=0.7, rho_inf=0.02, kappa=10, T_m=60)
g1 = sim.gates(sim.run(**base, N_E=2000, T=700, seed=118001))
c = 3.7
sc = dict(base); sc['eta'] = base['eta'] * c
g2 = sim.gates(sim.run(**sc, N_E=2000, T=700, seed=118001, w_min=c, w0=1.2 * c))
keys = ['R1', 'R2a_new', 'R2b_new', 'R2b_pop', 'R3a', 'R5', 'R7']
OUT['D_gauge_scaling'] = dict(c=c, base={k: float(g1[k]) for k in keys},
                              scaled={k: float(g2[k]) for k in keys},
                              max_abs_rel_diff=float(max(abs(g2[k] - g1[k]) / max(abs(g1[k]), 1e-12) for k in keys)))
# w0/w_min is NOT gauge: vary w0 alone
W0S = []
for w0 in (1.05, 1.2, 2.0, 5.0):
    g = sim.gates(sim.run(**base, N_E=2000, T=700, seed=118001, w0=w0))
    W0S.append(dict(w0=w0, R1=float(g['R1']), R2b_new=float(g['R2b_new']), R3a=float(g['R3a'])))
OUT['D_w0_sensitivity'] = W0S
TAU = []
for te in (1, 2, 4, 8):
    g = sim.gates(sim.run(**base, N_E=2000, T=700, seed=118001, tau_e=te))
    TAU.append(dict(tau_e=te, R1=float(g['R1']), R2b_new=float(g['R2b_new']), R3a=float(g['R3a'])))
OUT['D_tau_sensitivity'] = TAU

# ---------- E. identifiability: Jacobian of log-ratios wrt log-params -------
ref = dict(eta=20, lam0=0.25, theta=0.7, rho_inf=0.05, kappa=10, T_m=60)
names = ['eta', 'lam0', 'theta', 'rho_inf', 'kappa', 'T_m']
mets = ['R1', 'R2a_new', 'R2b_new', 'R3a', 'R5', 'R7']


def vec(p, seed=118001):
    g = sim.gates(sim.run(**p, N_E=4000, T=700, seed=seed))
    return np.array([max(g[m], 1e-6) for m in mets])


h = 0.10
J = np.zeros((len(mets), len(names)))
f0 = np.log(vec(ref))
for j, nm in enumerate(names):
    pp = dict(ref); pp[nm] = ref[nm] * (1 + h)
    pm = dict(ref); pm[nm] = ref[nm] * (1 - h)
    if nm == 'theta':
        pp[nm] = min(pp[nm], 0.85)
    J[:, j] = (np.log(vec(pp)) - np.log(vec(pm))) / (np.log(pp[nm]) - np.log(pm[nm]))
# seed noise scale for comparison
noise = np.std([np.log(vec(ref, seed=s)) for s in (118001, 118002, 118003, 118004)], axis=0)
u, s, vt = np.linalg.svd(J)
OUT['E_identifiability'] = dict(metrics=mets, params=names, J=J.tolist(),
                                singular_values=s.tolist(),
                                cond=float(s[0] / s[-1]),
                                smallest_right_vector=dict(zip(names, vt[-1].tolist())),
                                seed_noise_log_sd=dict(zip(mets, noise.tolist())))

# ---------- F. R4 daily mass map ------------------------------------------
OUT['F_R4_map'] = dict(
    note='M_{t+1} = M_t - lam0*Mbelow_t + p*eta*N_t + w0*F_t - deaths_mass',
    contraction_factor='1 - lam0*phi_t with phi_t = Mbelow_t/M_t',
    fixed_point='M* = p*eta*N/(lam0*phi)  requires phi bounded away from 0',
    observed_phi_decay='phi_t = R3a_t/lam0 -> 0 like 1/t (see C)')

with open(os.path.join(OUTDIR,'analytics.json'), 'w') as f:
    json.dump(OUT, f, indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float)[:6000])
