"""BA-V3-1 mean-field surrogate of the frozen model (contract section 3).

Declared surrogate axioms S1-S6 (divergence from the real spiking model is a
stated limitation, see 11-math.md section 7):
  S1 spikes/delays/STDP kernel integrated out: wake operation on a plastic
     E->E edge is an additive, w-INDEPENDENT random gain
     Delta = eta * g(t) * e,  e ~ Gamma(shape=2, mean=1).  Contract 3.1 has no
     weight dependence in Delta w; with A_-/A_+=0.5 and tau_-/tau_+=2 the
     uncorrelated pair contribution A_+tau_+ - A_-tau_- vanishes, so eta
     absorbs the (unknown) mean net eligibility per day.
  S2 homeostasis (3.4) = per-postsynaptic-neuron multiplicative scaling of the
     input weight vector toward target input mass Sstar (proxy for r*), once
     at end of wake, gain beta (beta=1 = full correction).
  S3 lateral inhibition / Dale / delays enter only through S1-S2: no explicit
     competition term, so emergent WTA is absent.  The surrogate therefore
     cannot decide E2 and can only bound E1.
  S4 sleep = contract 3.2 verbatim on every alive edge.
  S5 existence = contract 3.3 verbatim (rho(t), w0=1.2, w<1 for tau_e=2 days).
  S6 wake:sleep 16:8 enters only as a scale factor on eta (degenerate).
"""
import numpy as np

WMIN = 1.0


def run(p, seed, NE=64, T=716, beta=1.0, w0=1.2, tau_e=2, gain_shape=2.0,
        sigma_h=0.0,
        events=(505, 545, 585, 625, 665), adult=(500, 700), dev=(20, 70),
        peak_win=(0, 300)):
    rng = np.random.default_rng(seed)
    eta, lam0, kappa = p["eta"], p["lam0"], p["kappa"]
    rho_inf, kappa_m, T_m = p["rho_inf"], p["kappa_m"], p["T_m"]
    Sstar, g1g0 = p["Sstar"], p["g1g0"]
    P = NE * NE
    w = np.zeros(P)
    alive = np.zeros(P, dtype=bool)
    diag = np.zeros(P, dtype=bool)
    diag[np.arange(NE) * NE + np.arange(NE)] = True
    birth = np.full(P, -1, dtype=np.int32)
    low = np.zeros(P, dtype=np.int16)
    het = np.ones(P)   # quenched per-edge wake-gain heterogeneity
    life = []
    N_t = np.zeros(T); Mpost_t = np.zeros(T); Mpre_t = np.zeros(T)
    r3a_t = np.full(T, np.nan); r3b_t = np.full(T, np.nan)
    ftop_t = np.full(T, np.nan); thG_t = np.full(T, np.nan)
    gam_t = np.full(T, np.nan); c_t = np.full(T, np.nan)
    ev = set(events)
    for t in range(T):
        rho = rho_inf * (1.0 + kappa_m * np.exp(-t / T_m))
        rho = min(max(rho, 0.0), 1.0)
        cand = (~alive) & (~diag)
        nc = int(cand.sum())
        if nc and rho > 0:
            idx = np.flatnonzero(cand)[rng.random(nc) < rho]
            if idx.size:
                w[idx] = w0; alive[idx] = True; birth[idx] = t; low[idx] = 0
                if sigma_h > 0:
                    het[idx] = np.exp(rng.normal(-0.5 * sigma_h ** 2,
                                                 sigma_h, size=idx.size))
        if not alive.any():
            N_t[t] = 0; continue
        g = g1g0 if t in ev else 1.0
        e = rng.gamma(gain_shape, 1.0 / gain_shape, size=P)
        dw = eta * g * e * het * alive
        Mb = w.sum()
        w += dw
        gam_t[t] = dw.sum() / Mb if Mb > 0 else np.nan
        rs = w.reshape(NE, NE).sum(axis=1)
        c = np.ones(NE); ok = rs > 0
        c[ok] = (Sstar / rs[ok]) ** beta
        w = (w.reshape(NE, NE) * c[:, None]).ravel()
        c_t[t] = float(np.average(c, weights=np.maximum(rs, 1e-12)))
        wa = w[alive]; Mpre = wa.sum(); Mpre_t[t] = Mpre
        k = max(1, int(round(0.2 * wa.size)))
        thr = np.partition(wa, wa.size - k)[wa.size - k]
        top = alive & (w >= thr)
        Mtop = w[top].sum()
        ftop_t[t] = Mtop / Mpre if Mpre > 0 else np.nan
        dwa = dw * np.repeat(c, NE)
        s_all = dwa[alive].sum()
        thG_t[t] = dwa[top].sum() / s_all if s_all > 0 else np.nan
        lam = lam0 / (1.0 + (w / kappa) ** 2)
        loss_ = w * lam * alive
        w -= loss_
        r3a_t[t] = loss_.sum() / Mpre if Mpre > 0 else np.nan
        r3b_t[t] = loss_[top].sum() / Mtop if Mtop > 0 else np.nan
        lowhit = alive & (w < WMIN)
        low[lowhit] += 1
        low[alive & (~lowhit)] = 0
        dead = alive & (low >= tau_e)
        if dead.any():
            di = np.flatnonzero(dead)
            life.append(np.stack([birth[di], np.full(di.size, t)], axis=1))
            alive[dead] = False; w[dead] = 0.0; low[dead] = 0
        N_t[t] = alive.sum(); Mpost_t[t] = w.sum()
    return _metrics(locals())


def _skew(x):
    x = np.asarray(x, float)
    if x.size < 3: return np.nan
    m, s = x.mean(), x.std()
    return float(((x - m) ** 3).mean() / s ** 3) if s > 0 else np.nan


def _metrics(L):
    T, w, alive, birth = L["T"], L["w"], L["alive"], L["birth"]
    adult, dev, peak_win, events, NE = (L["adult"], L["dev"], L["peak_win"],
                                        L["events"], L["NE"])
    life = L["life"]
    A = np.concatenate(life, axis=0) if life else np.zeros((0, 2), np.int32)
    ai = np.flatnonzero(alive)
    b = np.concatenate([A[:, 0], birth[ai]])
    d = np.concatenate([A[:, 1], np.full(ai.size, T)])
    cens = np.concatenate([np.zeros(A.shape[0], bool), np.ones(ai.size, bool)])
    ll = d - b
    usable = (~cens) | (ll >= 8)
    o = {}

    def coh(a, z):
        m = (b >= a) & (b <= z) & usable
        return (float((ll[m] >= 8).mean()), int(m.sum())) if m.sum() >= 20 else (np.nan, int(m.sum()))

    def prev(a, z):
        m = (b <= z) & (d > a) & usable
        return (float((ll[m] >= 8).mean()), int(m.sum())) if m.sum() >= 20 else (np.nan, int(m.sum()))

    o["R2dev_Na"], o["n_R2dev"] = coh(*dev)
    o["R2ad_Na"], o["n_R2ad"] = coh(adult[0], adult[1] - 16)
    o["R2dev_Nb"], _ = prev(*dev)
    o["R2ad_Nb"], _ = prev(*adult)
    rA, rB, rApm, rBpm = [], [], [], []
    for t0 in range(adult[0], adult[1] - 30, 10):
        pool = (b <= t0 - 8) & (d > t0)
        n = int(pool.sum())
        if n < 20: continue
        dall = int(((d > t0) & (d <= t0 + 30) & (~cens)).sum())
        dper = int((pool & (d <= t0 + 30) & (~cens)).sum())
        fnew = int(((b > t0) & (b <= t0 + 30)).sum())
        rA.append(dper / n); rB.append(dall / n)
        rApm.append((dper + fnew) / n); rBpm.append((dall + fnew) / n)
    for nm, arr in (("R1_A", rA), ("R1_B", rB), ("R1_Apm", rApm), ("R1_Bpm", rBpm)):
        o[nm] = float(np.mean(arr)) if arr else np.nan
    sl = slice(adult[0], adult[1])
    for nm, arr in (("R3a", L["r3a_t"]), ("R3b", L["r3b_t"]), ("f_top", L["ftop_t"]),
                    ("theta_G", L["thG_t"]), ("gamma", L["gam_t"]), ("c_hom", L["c_t"])):
        v = arr[sl]
        o[nm] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    days = np.arange(adult[0], adult[1])
    Mday = 0.5 * (L["Mpre_t"][sl] + L["Mpost_t"][sl])
    good = np.isfinite(Mday) & (Mday > 0)
    o["R4"] = (float(abs(np.polyfit(days[good], Mday[good], 1)[0] * 100.0 / Mday[good].mean()))
               if good.sum() > 20 else np.nan)
    ftop_series = L["ftop_t"][sl]
    gf = np.isfinite(ftop_series)
    o["drift_ftop"] = (float(np.polyfit(days[gf], ftop_series[gf], 1)[0] * 100.0
                             / max(np.nanmean(ftop_series), 1e-12))
                       if gf.sum() > 20 else np.nan)
    Nser = L["N_t"][sl]
    gn = Nser > 0
    o["drift_N"] = (float(np.polyfit(days[gn], Nser[gn], 1)[0] * 100.0 / Nser[gn].mean())
                    if gn.sum() > 20 else np.nan)
    N_t = L["N_t"]
    Nad = float(N_t[sl].mean())
    o["N_adult"] = Nad
    Npk = float(N_t[peak_win[0]:peak_win[1]].max())
    o["R5"] = Npk / Nad if Nad > 0 else np.nan
    tpk = int(N_t[peak_win[0]:peak_win[1]].argmax())
    seg = N_t[tpk:adult[0]]
    o["R5_mono_viol"] = float(np.mean(np.diff(seg) > 0)) if seg.size > 2 else np.nan

    def s8(days_list):
        tot = hit = 0
        for dd in days_list:
            m = b == dd
            if not m.any(): continue
            tot += int(m.sum()); hit += int((ll[m] >= 8).sum())
        return (hit / tot if tot >= 20 else np.nan), tot
    es, ne = s8(list(events))
    bd = [x for x in range(adult[0], adult[1] - 16) if all(abs(x - e) > 9 for e in events)]
    bs, nb = s8(bd)
    o["R6"] = es / bs if (np.isfinite(es) and np.isfinite(bs) and bs > 0) else np.nan
    o["n_R6_event"], o["n_R6_base"] = ne, nb
    wa = w[alive]
    if wa.size > 50:
        o["E1_skew_w"] = _skew(wa)
        o["E1_skew_logw"] = _skew(np.log(np.maximum(wa, 1e-12)))
        o["sigma_logw"] = float(np.std(np.log(np.maximum(wa, 1e-12))))
        rs = w.reshape(NE, NE).sum(axis=1)
        pos = rs[rs > 0]
        o["E2_skew_rate_proxy"] = _skew(pos) if pos.size > 10 else np.nan
        o["E2_cv_rate_proxy"] = float(pos.std() / pos.mean()) if pos.size > 10 else np.nan
        o["wbar_adult"] = float(np.nanmean(L["Mpre_t"][sl]) / Nad) if Nad > 0 else np.nan
    else:
        for k in ("E1_skew_w", "E1_skew_logw", "sigma_logw", "E2_skew_rate_proxy",
                  "E2_cv_rate_proxy", "wbar_adult"):
            o[k] = np.nan
    return o


BANDS = {"R1_A": (0.02, 0.08), "R2dev_Na": (0.25, 0.45), "R2ad_Na": (0.60, 0.85),
         "R3a": (0.10, 0.25), "R3b": (1e-12, 0.05), "R4": (0.0, 0.05),
         "R5": (1.3, 1.8), "R6": (1.3, np.inf)}
TARGETS = {"R1_A": 0.04, "R2dev_Na": 0.35, "R2ad_Na": 0.73, "R3a": 0.18, "R5": 1.5}
INEQ = {"R3b": ("le", 0.05), "R4": ("le", 0.05), "R6": ("ge", 1.3)}


def gates_pass(m):
    ok = {}
    for k, (lo, hi) in BANDS.items():
        v = m.get(k, np.nan)
        ok[k] = bool(np.isfinite(v) and lo <= v <= hi)
    ok["R2_monotone"] = bool(np.isfinite(m.get("R2ad_Na", np.nan)) and
                             np.isfinite(m.get("R2dev_Na", np.nan)) and
                             m["R2ad_Na"] > m["R2dev_Na"])
    return ok


def loss(m, eps=1e-6):
    tot = 0.0
    for k, tv in TARGETS.items():
        v = m.get(k, np.nan)
        if not np.isfinite(v): return 1e6
        tot += np.log((v + eps) / (tv + eps)) ** 2
    for k, (dr, bd) in INEQ.items():
        v = m.get(k, np.nan)
        if not np.isfinite(v): return 1e6
        tot += (max(0.0, v - bd) if dr == "le" else max(0.0, bd - v)) ** 2
    return float(tot)
