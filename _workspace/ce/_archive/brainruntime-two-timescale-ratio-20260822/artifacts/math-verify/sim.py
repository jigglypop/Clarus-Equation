"""BA-TS1 frozen-family simulator (contract 00-contract.md section 3).

Mean-field edge population, no interaction except the sleep quantile q_theta.
Design constants (gauge/frozen): w_min=1, tau_e=2, p_hit=0.1, w0=1.2, N_E.
Free params: eta, lam0, theta, rho_inf, kappa, T_m.

Day order (apparatus convention A1): birth -> wake(+eta*Bern(p)) -> sleep
(x(1-lam0) if w<q_theta) -> death test (w<w_min for tau_e consecutive
post-sleep days).
"""
import numpy as np

W_MIN = 1.0
TAU_E = 2
P_HIT = 0.1
W0 = 1.2
KTOP = 600


def run(eta, lam0, theta, rho_inf, kappa, T_m, N_E=2000, T=700, seed=118001,
        learn_day=None, learn_mult=5.0, record_from=0, w_min=W_MIN, w0=W0,
        tau_e=TAU_E, p_hit=P_HIT, trace_top=False, keep_topk=False):
    rng = np.random.default_rng(seed)
    alive = np.zeros(N_E, bool)
    w = np.zeros(N_E)
    lowc = np.zeros(N_E, np.int16)
    bday = np.full(N_E, -1, np.int64)

    topcum = [None] * T
    rec = dict(N=np.zeros(T), M=np.zeros(T), rem=np.zeros(T), form=np.zeros(T),
               r3a=np.full(T, np.nan), q=np.full(T, np.nan),
               nb=np.zeros(T), nb_surv8=np.full(T, np.nan),
               hi_n=np.zeros(T), hi_loss30=np.full(T, np.nan))
    # cohort bookkeeping
    born_ids = {}
    hi_snap = {}
    pop_snap = {}
    pop_surv8 = np.full(T, np.nan)

    for t in range(T):
        # 1. birth on non-existing candidates
        rho = rho_inf * (1.0 + kappa * np.exp(-t / T_m))
        rho = min(rho, 1.0)
        free = np.flatnonzero(~alive)
        nb_mask = rng.random(free.size) < rho
        nb_idx = free[nb_mask]
        alive[nb_idx] = True
        w[nb_idx] = w0
        lowc[nb_idx] = 0
        bday[nb_idx] = t
        rec['form'][t] = nb_idx.size
        rec['nb'][t] = nb_idx.size
        born_ids[t] = nb_idx.copy()

        idx = np.flatnonzero(alive)
        n = idx.size
        if n == 0:
            continue
        # 2. wake
        e = eta * (learn_mult if (learn_day is not None and t == learn_day) else 1.0)
        hits = rng.random(n) < p_hit
        w[idx] += e * hits
        # 3. sleep
        q = np.quantile(w[idx], theta)
        pre = w[idx].sum()
        below = idx[w[idx] < q]
        w[below] *= (1.0 - lam0)
        post = w[idx].sum()
        rec['r3a'][t] = (pre - post) / pre if pre > 0 else np.nan
        rec['q'][t] = q
        # 4. death test
        low = w[idx] < w_min
        lowc[idx[low]] += 1
        lowc[idx[~low]] = 0
        dead = idx[lowc[idx] >= tau_e]
        alive[dead] = False
        w[dead] = 0.0
        rec['rem'][t] = dead.size
        rec['N'][t] = alive.sum()
        rec['M'][t] = w[alive].sum()
        if keep_topk:
            v = np.sort(w[alive])[::-1][:KTOP]
            topcum[t] = np.cumsum(v)
        if trace_top:
            exempt = idx[w[idx] >= q]
            rec.setdefault('Mtop', np.zeros(T))[t] = w[exempt].sum()
            rec.setdefault('Mbot', np.zeros(T))[t] = w[idx[w[idx] < q]].sum()
            rec.setdefault('hits_top', np.zeros(T))[t] = float(hits[np.isin(idx, exempt)].sum())

        # newborn 8-day survival (cohort born on day t-8)
        if t - 8 in born_ids:
            c = born_ids.pop(t - 8)
            if c.size > 0:
                rec['nb_surv8'][t - 8] = float(np.mean(alive[c] & (bday[c] == t - 8)))
        # population 8-day survival
        if t >= record_from:
            pop_snap[t] = (np.flatnonzero(alive), bday[alive].copy())
            hi = np.flatnonzero(alive & (w >= q))
            hi_snap[t] = (hi, bday[hi].copy())
            rec['hi_n'][t] = hi.size
        if t - 8 in pop_snap:
            ids, b = pop_snap.pop(t - 8)
            if ids.size:
                pop_surv8[t - 8] = float(np.mean(alive[ids] & (bday[ids] == b)))
        if t - 30 in hi_snap:
            ids, b = hi_snap.pop(t - 30)
            if ids.size:
                rec['hi_loss30'][t - 30] = float(np.mean(~(alive[ids] & (bday[ids] == b))))
        for k in list(pop_snap):
            if k < t - 8:
                pop_snap.pop(k)
        for k in list(hi_snap):
            if k < t - 30:
                hi_snap.pop(k)
        for k in list(born_ids):
            if k < t - 8:
                born_ids.pop(k)
    rec['pop_surv8'] = pop_surv8
    if keep_topk:
        k = max(int(np.floor((1 - theta) * rec['N'][500:700].min())), 1)
        k = min(k, KTOP)
        rec['k_fixed'] = k
        rec['Sk'] = np.array([cs[k - 1] if cs is not None and cs.size >= k else np.nan
                              for cs in topcum])
    return rec


def gates(rec, dev_lo=0, dev_hi=300, ad_lo=500, ad_hi=700):
    N = rec['N']; M = rec['M']
    ad = slice(ad_lo, ad_hi)
    Nad = N[ad].mean()
    out = {}
    # R1 monthly turnover (removal+formation)/surviving
    out['R1'] = (rec['rem'][ad].sum() + rec['form'][ad].sum()) / max(Nad, 1e-9) * 30.0 / (ad_hi - ad_lo)
    # R2 (two readings)
    out['R2a_pop'] = np.nanmean(rec['pop_surv8'][dev_lo:dev_hi]) if np.any(~np.isnan(rec['pop_surv8'][dev_lo:dev_hi])) else np.nan
    out['R2b_pop'] = np.nanmean(rec['pop_surv8'][ad])
    dpk = int(np.argmax(N[dev_lo:dev_hi])) + dev_lo
    out['peak_day'] = dpk
    out['R2a_new'] = np.nanmean(rec['nb_surv8'][max(dpk - 10, 0):dpk + 10])
    out['R2b_new'] = np.nanmean(rec['nb_surv8'][ad])
    # R3a
    out['R3a'] = np.nanmean(rec['r3a'][ad])
    # R4 relative drift per 100 d
    x = np.arange(ad_hi - ad_lo, dtype=float)
    y = M[ad]
    sl = np.polyfit(x, y, 1)[0]
    out['R4_drift100'] = sl * 100.0 / max(y.mean(), 1e-12)
    out['M_end_over_start'] = y[-1] / max(y[0], 1e-12)
    # R5 density overshoot
    out['R5'] = N[dev_lo:dev_hi].max() / max(Nad, 1e-9)
    tail = N[dpk:ad_hi]
    out['R5_rerise'] = float(np.max(tail[1:] - np.minimum.accumulate(tail[:-1])) / max(Nad, 1e-9)) if tail.size > 2 else 0.0
    # R7 monthly loss of high-strength edges
    out['R7'] = np.nanmean(rec['hi_loss30'][ad_lo:ad_hi - 30]) if ad_hi - 30 > ad_lo else np.nan
    out['Nad'] = Nad
    out['wbar'] = M[ad].mean() / max(Nad, 1e-9)
    out['q_ad'] = np.nanmean(rec['q'][ad])
    return out
