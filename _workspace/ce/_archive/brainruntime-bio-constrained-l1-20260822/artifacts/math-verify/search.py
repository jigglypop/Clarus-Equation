"""LHS + local refinement over the 8 free parameters of contract section 3.5.
Calibration seeds only: 119001 (search), 119002..119006 (noise / robustness).
Development block 119101+ is NOT touched."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import surrogate as S

HERE = os.path.dirname(os.path.abspath(__file__))
SEED_SEARCH = 119001
SEEDS_NOISE = [119002, 119003, 119004, 119005, 119006]
SPEC = [("eta", 0.05, 30.0, True), ("lam0", 0.15, 0.99, False),
        ("kappa", 0.5, 300.0, True), ("rho_inf", 1e-5, 3e-2, True),
        ("kappa_m", 0.05, 300.0, True), ("T_m", 5.0, 250.0, False),
        ("Sstar", 5.0, 5000.0, True), ("g1g0", 1.0, 15.0, False)]
KEYS = [s[0] for s in SPEC]


def lhs(n, rng):
    X = np.zeros((n, len(SPEC)))
    for j, (_, lo, hi, lg) in enumerate(SPEC):
        u = (rng.permutation(n) + rng.random(n)) / n
        X[:, j] = np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))) if lg else lo + u * (hi - lo)
    return X


def to_p(x): return {k: float(v) for k, v in zip(KEYS, x)}


def clip(x):
    y = x.copy()
    for j, (_, lo, hi, _lg) in enumerate(SPEC):
        y[j] = min(max(y[j], lo), hi)
    return y


def main(n=1000):
    rng = np.random.default_rng(20260822)
    X = lhs(n, rng)
    rows, t0 = [], time.time()
    for i in range(n):
        m = S.run(to_p(X[i]), SEED_SEARCH)
        m["loss"] = S.loss(m)
        m["npass"] = int(sum(S.gates_pass(m).values()))
        m.update(to_p(X[i]))
        rows.append(m)
        if (i + 1) % 100 == 0:
            print("%d/%d  %.0fs" % (i + 1, n, time.time() - t0), flush=True)
    keys = sorted(rows[0])
    with open(os.path.join(HERE, "lhs.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("%.6g" % float(r.get(k, float("nan"))) for k in keys) + "\n")
    # gate statistics
    gp = [S.gates_pass(r) for r in rows]
    gk = list(gp[0])
    stat = {"n": n, "per_gate": {k: int(sum(g[k] for g in gp)) for k in gk}}
    import itertools
    stat["pairs"] = {}
    for a, b in itertools.combinations(gk, 2):
        stat["pairs"]["%s^%s" % (a, b)] = int(sum(g[a] and g[b] for g in gp))
    stat["all_L1"] = int(sum(all(g.values()) for g in gp))
    stat["npass_hist"] = {str(k): int(sum(1 for r in rows if r["npass"] == k))
                          for k in range(0, 10)}
    # constrained sup/inf
    def sub(cond):
        return [r for r, g in zip(rows, gp) if cond(r, g)]
    inband = sub(lambda r, g: g["R1_A"])
    stat["sup_R3a_given_R1"] = max((r["R3a"] for r in inband), default=None)
    r3 = sub(lambda r, g: g["R3a"])
    stat["R1_range_given_R3a"] = [min((r["R1_A"] for r in r3), default=None),
                                  max((r["R1_A"] for r in r3), default=None)]
    stat["sup_R2ad_given_R3a"] = max((r["R2ad_Na"] for r in r3), default=None)
    stat["R4_stats"] = {"max": float(np.nanmax([r["R4"] for r in rows])),
                        "median": float(np.nanmedian([r["R4"] for r in rows])),
                        "frac_lt_0.5": float(np.nanmean([r["R4"] < 0.5 for r in rows]))}
    stat["E2_cv_stats"] = {"median": float(np.nanmedian([r["E2_cv_rate_proxy"] for r in rows])),
                           "max": float(np.nanmax([r["E2_cv_rate_proxy"] for r in rows])),
                           "frac_skew_gt_0.5": float(np.nanmean([r["E2_skew_rate_proxy"] > 0.5 for r in rows]))}
    stat["E1_stats"] = {"frac_pass": float(np.nanmean([(abs(r["E1_skew_logw"]) < 0.5) and (r["E1_skew_w"] > 1.0) for r in rows]))}
    # closed-form budget identity check (independent validation of B1)
    chk = []
    for r in rows:
        s, st, ft, tg, gm = r["R3a"], r["R3b"], r["f_top"], r["theta_G"], r["gamma"]
        if not all(np.isfinite([s, st, ft, tg, gm])) or ft <= 0: continue
        # identity (I) presupposes cyclic stationarity: require small drift of
        # both total mass and the top-20% mass share over the adult window
        if not (np.isfinite(r.get("drift_ftop", np.nan)) and
                abs(r["drift_ftop"]) < 1.0 and abs(r["R4"]) < 1.0): continue
        rho = (1 - s) / (1 - st); psi = tg / ft
        if rho <= psi: continue
        pred = (1 - rho) / (rho - psi)
        if gm > 1e-4:
            chk.append(abs(pred - gm) / gm)
    stat["budget_identity_rel_err"] = {
        "n": len(chk), "median": float(np.median(chk)) if chk else None,
        "p90": float(np.quantile(chk, 0.9)) if chk else None,
        "max": float(np.max(chk)) if chk else None}
    best = sorted(rows, key=lambda r: (-r["npass"], r["loss"]))[:12]
    stat["best"] = [{k: float(r[k]) for k in KEYS + ["loss", "npass", "R1_A", "R2dev_Na",
                    "R2ad_Na", "R3a", "R3b", "R4", "R5", "R6", "f_top", "theta_G",
                    "gamma", "c_hom", "sigma_logw", "E1_skew_w", "E1_skew_logw",
                    "E2_skew_rate_proxy", "N_adult", "wbar_adult"]} for r in best]
    with open(os.path.join(HERE, "lhs_summary.json"), "w") as f:
        json.dump(stat, f, indent=1, default=float)
    print(json.dumps({k: stat[k] for k in ("per_gate", "all_L1", "npass_hist",
                                           "sup_R3a_given_R1", "R1_range_given_R3a",
                                           "budget_identity_rel_err")},
                     indent=1, default=float))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
