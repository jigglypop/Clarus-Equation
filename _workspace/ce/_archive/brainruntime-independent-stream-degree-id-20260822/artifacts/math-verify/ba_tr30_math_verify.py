# BA-TR30 math-verifier independent recomputation (lane 11-math)
# Independent path: definitions only; does not import run implementation code.
import numpy as np, json
from math import comb

RHO = 0.5
D_SET = [1, 2, 3]
H_SET = [0.0, 1e-3, 1e-2]
N = 14
GATE = {0.0: 1e-10, 1e-3: 2e-2, 1e-2: 2e-1}
NDRAW = 200
MASTER_SEED = 900030

def monomial_exponents(d):
    return [(a, b) for s in range(d + 1) for a in range(s + 1) for b in [s - a]]

def phi(Z, d):
    ex = monomial_exponents(d)
    return np.stack([Z[:, 0] ** a * Z[:, 1] ** b for (a, b) in ex], axis=1)

def loo_hat(Phi, Y):
    G = Phi.T @ Phi
    cond = np.linalg.cond(Phi)
    C = np.linalg.lstsq(Phi, Y, rcond=None)[0]
    Ginv = np.linalg.inv(G)
    h = np.einsum('ij,jk,ik->i', Phi, Ginv, Phi)
    E = (Y - Phi @ C) / (1.0 - h)[:, None]
    s = np.mean(np.linalg.norm(E, axis=1))
    return s, h.max(), cond, C

def loo_refit(Phi, Y):
    n = Phi.shape[0]
    norms = []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        Ci = np.linalg.lstsq(Phi[m], Y[m], rcond=None)[0]
        norms.append(np.linalg.norm(Y[i] - Phi[i] @ Ci))
    return float(np.mean(norms))

def gen_instance(rng, dstar, eta, zscale=1.0):
    Z = rng.standard_normal((N, 2)) * zscale
    zq = rng.standard_normal((1, 2)) * zscale
    p = comb(dstar + 2, 2)
    Cstar = rng.uniform(-1, 1, size=(p, 6))
    Y = phi(Z, dstar) @ Cstar + eta * rng.standard_normal((N, 6))
    yq_clean = (phi(zq, dstar) @ Cstar)[0]
    return Z, zq, Y, yq_clean, Cstar

def select_degree(svals):
    smin = min(svals.values())
    for d in D_SET:
        if svals[d] <= (1 + RHO) * smin:
            return d
    return None

out = {}

# ---------- (a) feature counts + rank/conditioning ----------
counts = {d: len(monomial_exponents(d)) for d in [1, 2, 3, 4]}
out['feature_counts'] = counts
assert counts[1] == 3 and counts[2] == 6 and counts[3] == 10 and counts[4] == 15
out['binom_check'] = {d: comb(d + 2, 2) for d in [1, 2, 3, 4]}
out['loo_rows_ge_p'] = {'N_minus_1': N - 1, 'p_d3': 10, 'ok': (N - 1) >= 10}

rng = np.random.default_rng(MASTER_SEED)
conds = {d: [] for d in D_SET}
ranks_ok = {d: 0 for d in D_SET}
for t in range(500):
    Z = rng.standard_normal((N, 2))
    for d in D_SET:
        P = phi(Z, d)
        if np.linalg.matrix_rank(P) == counts[d]:
            ranks_ok[d] += 1
        conds[d].append(np.linalg.cond(P))
out['rank_full_of_500'] = ranks_ok
out['cond_quantiles'] = {d: {q: float(np.quantile(conds[d], qq))
                             for q, qq in [('p50', .5), ('p95', .95), ('p99', .99), ('max', 1.0)]}
                         for d in D_SET}

# degenerate cue placement probe (contract does not pin cue sampling): 14 points on a circle
th = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.123
Zc = np.stack([np.cos(th), np.sin(th)], axis=1) * 1.3
out['circle_ranks'] = {d: int(np.linalg.matrix_rank(phi(Zc, d))) for d in D_SET}

# ---------- (b) hat-matrix identity vs explicit refit ----------
diffs = []
rng = np.random.default_rng(MASTER_SEED + 1)
for t in range(50):
    dstar = int(rng.integers(1, 4)); eta = float(rng.choice(H_SET))
    Z, zq, Y, yq, _ = gen_instance(rng, dstar, eta)
    for d in D_SET:
        P = phi(Z, d)
        s_hat = loo_hat(P, Y)[0]
        s_ref = loo_refit(P, Y)
        diffs.append(abs(s_hat - s_ref) / s_ref if s_ref > 1e-13 else abs(s_hat - s_ref))
out['loo_identity_max_reldiff'] = float(np.max(diffs))

# ---------- (c) degree selection + abstain + prediction gates ----------
def cell_sim(dstar, eta, zscale=1.0, seed_shift=0):
    rng = np.random.default_rng(MASTER_SEED + 1000 * dstar + int(eta * 1e6) + seed_shift)
    tau = max(1e-8, 8 * eta)
    res = dict(abstain=0, sel_ok=0, sel_and_gate=0, e_list=[], smin_list=[],
               s_by_d={1: [], 2: [], 3: []}, hmax_list=[], forced_d1_gate_fail=0)
    for t in range(NDRAW):
        Z, zq, Y, yq, _ = gen_instance(rng, dstar, eta, zscale)
        svals, Cs = {}, {}
        hmax_all = 0.0
        for d in D_SET:
            P = phi(Z, d)
            s, hmax, cond, C = loo_hat(P, Y)
            svals[d] = s; Cs[d] = C; hmax_all = max(hmax_all, hmax)
            res['s_by_d'][d].append(s)
        res['hmax_list'].append(hmax_all)
        smin = min(svals.values()); res['smin_list'].append(smin)
        y1 = (phi(zq, 1) @ Cs[1])[0]
        e1 = np.linalg.norm(y1 - yq) / max(np.linalg.norm(yq), 1e-12)
        if dstar in (2, 3) and e1 > GATE[eta]:
            res['forced_d1_gate_fail'] += 1
        if smin > tau:
            res['abstain'] += 1
            continue
        dhat = select_degree(svals)
        yhat = (phi(zq, dhat) @ Cs[dhat])[0]
        e = np.linalg.norm(yhat - yq) / max(np.linalg.norm(yq), 1e-12)
        res['e_list'].append(float(e))
        if dhat == dstar:
            res['sel_ok'] += 1
            if e <= GATE[eta]:
                res['sel_and_gate'] += 1
    return res, tau

summary_c = {}
for dstar in D_SET:
    for eta in H_SET:
        res, tau = cell_sim(dstar, eta)
        e = np.array(res['e_list']) if res['e_list'] else np.array([np.nan])
        summary_c['d%d_eta%g' % (dstar, eta)] = {
            'tau_class': tau,
            'false_abstain_rate': res['abstain'] / NDRAW,
            'sel_ok_rate_of_nonabstain': res['sel_ok'] / max(NDRAW - res['abstain'], 1),
            'sel_and_gate_rate': res['sel_and_gate'] / NDRAW,
            'e_p50': float(np.nanquantile(e, .5)), 'e_p95': float(np.nanquantile(e, .95)),
            'e_max': float(np.nanmax(e)), 'gate': GATE[eta],
            's_med_by_d': {d: float(np.median(res['s_by_d'][d])) for d in D_SET},
            's_max_by_d': {d: float(np.max(res['s_by_d'][d])) for d in D_SET},
            'smin_max': float(np.max(res['smin_list'])),
            'hmax_p95': float(np.quantile(res['hmax_list'], .95)),
            'hmax_max': float(np.max(res['hmax_list'])),
            'forced_d1_gate_fail_rate': (res['forced_d1_gate_fail'] / NDRAW) if dstar in (2, 3) else None,
        }
out['cells'] = summary_c

# witness fold: degree-4 generator, eta=1e-3, must abstain (min_d s_d > tau)
rng = np.random.default_rng(MASTER_SEED + 77)
tau_w = max(1e-8, 8 * 1e-3)
w_abstain = 0; w_smins = []
for t in range(NDRAW):
    Z, zq, Y, yq, _ = gen_instance(rng, 4, 1e-3)
    svals = {d: loo_hat(phi(Z, d), Y)[0] for d in D_SET}
    smin = min(svals.values()); w_smins.append(smin)
    if smin > tau_w:
        w_abstain += 1
out['witness_d4'] = {'tau': tau_w, 'abstain_rate': w_abstain / NDRAW,
                     'smin_min': float(np.min(w_smins)),
                     'smin_p01': float(np.quantile(w_smins, .01)),
                     'smin_med': float(np.median(w_smins))}

# association shuffle control: does the class gate reject it?
rng = np.random.default_rng(MASTER_SEED + 88)
sh = {}
for eta in H_SET:
    tau = max(1e-8, 8 * eta); rej = 0
    for t in range(100):
        Z, zq, Y, yq, _ = gen_instance(rng, 2, eta)
        Ys = Y[rng.permutation(N)]
        smin = min(loo_hat(phi(Z, d), Ys)[0] for d in D_SET)
        if smin > tau:
            rej += 1
    sh['eta%g' % eta] = rej / 100
out['shuffle_reject_rate'] = sh

# (d)-adjacent numeric: bank selection robustness at worst declared noise
rng = np.random.default_rng(MASTER_SEED + 99)
sel_true = 0; ntr = 0
for t in range(200):
    Z, zq, Y, yq, _ = gen_instance(rng, 3, 1e-2)
    svals, Cs = {}, {}
    for d in D_SET:
        s, _, _, C = loo_hat(phi(Z, d), Y); svals[d] = s; Cs[d] = C
    if min(svals.values()) > 8e-2:
        continue
    dhat = select_degree(svals)
    yhat = (phi(zq, dhat) @ Cs[dhat])[0]
    bank = [yq]
    for k in range(7):
        v = rng.standard_normal(6)
        v = v / np.linalg.norm(v) * np.linalg.norm(Y[rng.integers(N)])
        bank.append(v)
    kstar = int(np.argmin([np.linalg.norm(yhat - c) for c in bank]))
    sel_true += (kstar == 0); ntr += 1
out['bank_selection'] = {'n': ntr, 'true_rate': sel_true / max(ntr, 1)}

# hidden-dof probe: cue scale A/B (zscale=3)
ab = {}
for dstar, eta in [(3, 0.0), (3, 1e-2), (1, 0.0)]:
    res, tau = cell_sim(dstar, eta, zscale=3.0, seed_shift=555)
    e = np.array(res['e_list']) if res['e_list'] else np.array([np.nan])
    ab['d%d_eta%g_zscale3' % (dstar, eta)] = {
        'false_abstain_rate': res['abstain'] / NDRAW,
        'sel_and_gate_rate': res['sel_and_gate'] / NDRAW,
        'e_max': float(np.nanmax(e)), 'gate': GATE[eta]}
out['zscale3_probe'] = ab

print(json.dumps(out, indent=2, default=float))
