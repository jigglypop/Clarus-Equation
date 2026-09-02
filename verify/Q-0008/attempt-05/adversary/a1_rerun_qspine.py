"""Adversary a1 (Q-0008 attempt-05, ladder step 7 execution audit)."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
F02 = ROOT / "verify" / "Q-0008" / "F-02"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))
import check_modes as cm
from driver_numbers import qspine_block

DEPTHS = tuple(cm.QSPINE_DEPTHS)
TRIALS = cm.QSPINE_TRIALS
IID_N = cm.QSPINE_IID_N
DELTA = cm.DELTA
E_N = [b * (b + 1) // 2 for b in DEPTHS]
OFFICIAL = json.loads((F02 / "result.json").read_text(encoding="utf-8"))["qspine"]
ATT5 = json.loads((ROOT / "verify" / "Q-0008" / "attempt-05" / "result.json").read_text(encoding="utf-8"))


def run(seed):
    rng = np.random.default_rng(seed)
    rng_i = np.random.default_rng(seed + 1)
    V, NS, rej = [], [], 0
    for b in DEPTHS:
        vals, ns = [], []
        while len(vals) < TRIALS:
            parent = qspine_block(b, rng)
            n = len(parent)
            v = cm.block_residual(cm.heritable_labels(parent, rng.normal(size=(n, 4, 4))), DELTA)
            if math.isfinite(v):
                vals.append(v)
                ns.append(n)
            else:
                rej += 1
        V.append(np.array(vals))
        NS.append(np.array(ns))
    I = np.array([cm.sample_iid(IID_N, rng_i, DELTA) for _ in range(TRIALS)])
    rms_b = [float(np.sqrt(np.mean(v * v))) for v in V]
    rms_i = float(np.sqrt(np.mean(I * I)))
    return dict(V=V, NS=NS, I=I, rms=rms_b, rms_iid=rms_i, rej=rej,
                slope_En=cm.fit_slope(E_N, rms_b),
                slope_meann=cm.fit_slope([float(np.mean(n)) for n in NS], rms_b),
                slope_b=cm.fit_slope(DEPTHS, rms_b),
                mean_n=[float(np.mean(n)) for n in NS],
                sd_n=[float(np.std(n, ddof=1)) for n in NS],
                ratio=rms_b[-1] / rms_i)


def var_log_rms(v):
    m2 = float(np.mean(v * v))
    return float(np.var(v * v, ddof=1) / (4 * len(v) * m2 * m2))


def ses(r, bseed=20260902, B=2000):
    x = np.log(np.array(E_N, float))
    c = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    se_slope_d = float(np.sqrt(np.sum(c * c * np.array([var_log_rms(v) for v in r["V"]]))))
    se_ratio_d = float(r["ratio"] * math.sqrt(var_log_rms(r["V"][-1]) + var_log_rms(r["I"])))
    brng = np.random.default_rng(bseed)
    bs, br = [], []
    for _ in range(B):
        rr = [float(np.sqrt(np.mean(v[brng.integers(0, len(v), len(v))] ** 2))) for v in r["V"]]
        ri = float(np.sqrt(np.mean(r["I"][brng.integers(0, len(r["I"]), len(r["I"]))] ** 2)))
        bs.append(cm.fit_slope(E_N, rr))
        br.append(rr[-1] / ri)
    return dict(se_slope_delta=se_slope_d, se_ratio_delta=se_ratio_d,
                se_slope_boot=float(np.std(bs, ddof=1)), se_ratio_boot=float(np.std(br, ddof=1)),
                ci_slope=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                ci_ratio=[float(np.percentile(br, 2.5)), float(np.percentile(br, 97.5))])


def driver(parent):
    n = len(parent)
    ch = [[] for _ in range(n)]
    root = -1
    for v, p in enumerate(parent):
        if p >= 0:
            ch[p].append(v)
        else:
            root = v
    order = [root]
    i = 0
    while i < len(order):
        order.extend(ch[order[i]])
        i += 1
    dep = np.zeros(n, np.int64)
    sub = np.ones(n, np.int64)
    pre = np.zeros(n, np.int64)
    for v in order[1:]:
        dep[v] = dep[parent[v]] + 1
    for v in reversed(order):
        if parent[v] >= 0:
            sub[parent[v]] += sub[v]
    for v in order:
        pre[v] = sub[v] + (pre[parent[v]] if parent[v] >= 0 else 0)
    s = sub.astype(float)
    D = float(np.sum((2 * dep + 1) * s * s) - 2 * np.sum(pre.astype(float) ** 2) / n + np.sum(s * s) ** 2 / n ** 2)
    return D, n


out = {}
t0 = time.perf_counter()
print("== (1) same-seed 20260902 re-run ==", flush=True)
r0 = run(cm.SEED)
same = {"rms_bitwise_equal": [a == b for a, b in zip(r0["rms"], OFFICIAL["rms"])],
        "max_abs_diff_rms": float(np.max(np.abs(np.array(r0["rms"]) - np.array(OFFICIAL["rms"])))),
        "rms_iid_equal": r0["rms_iid"] == OFFICIAL["rms_iid_36"],
        "slope_equal": r0["slope_En"] == OFFICIAL["slope_vs_En"],
        "ratio_equal": r0["ratio"] == OFFICIAL["ratio_b8_over_iid36"],
        "mean_n_equal": [a == b for a, b in zip(r0["mean_n"], OFFICIAL["mean_n"])],
        "slope_En": r0["slope_En"], "ratio": r0["ratio"], "rejections": r0["rej"]}
print(json.dumps(same, indent=1), flush=True)
out["same_seed"] = same

print("== (3) statistics method ==", flush=True)
S = ses(r0)
meth = dict(S)
meth["slope_on_exact_En_grid_PREREGISTERED"] = r0["slope_En"]
meth["slope_on_observed_mean_n_grid_SENSITIVITY"] = r0["slope_meann"]
meth["slope_vs_b"] = r0["slope_b"]
meth["mean_n"] = r0["mean_n"]
meth["sd_n"] = r0["sd_n"]
meth["mean_n_z_vs_exact"] = [(m - e) / (s / math.sqrt(TRIALS)) for m, e, s in zip(r0["mean_n"], E_N, r0["sd_n"])]
meth["reported_se_slope_boot"] = ATT5["stats"]["qspine_slope_vs_En"]["se"]
meth["reported_se_ratio_boot"] = ATT5["stats"]["qspine_ratio_b8_over_iid36"]["se"]
print(json.dumps(meth, indent=1), flush=True)
out["method"] = meth

print("== (2) other seeds ==", flush=True)
oth = []
for sd in (20260903, 424242):
    r = run(sd)
    s = ses(r, bseed=sd, B=800)
    row = {"seed": sd, "slope": r["slope_En"], "ratio": r["ratio"],
           "se_slope": s["se_slope_boot"], "se_ratio": s["se_ratio_boot"],
           "slope_in_window": bool(0.42 <= r["slope_En"] <= 0.59),
           "ratio_in_window": bool(6.01 <= r["ratio"] <= 7.65),
           "in_window": bool(0.42 <= r["slope_En"] <= 0.59 and 6.01 <= r["ratio"] <= 7.65),
           "slope_on_mean_n_grid": r["slope_meann"], "mean_n": r["mean_n"],
           "rms": r["rms"], "rms_iid": r["rms_iid"], "rejections": r["rej"]}
    print(json.dumps(row, indent=1), flush=True)
    oth.append(row)
out["other_seeds"] = oth

print("== (5) independent tree-only MC of E[D/n^2] ==", flush=True)
CARD = [0.1017, 0.2126, 0.3558, 0.5327, 0.7411, 0.9842, 1.2607]
rng = np.random.default_rng(777000777)
tree = []
for k, b in enumerate(DEPTHS):
    vals, ns = [], []
    for _ in range(60000):
        D, n = driver(qspine_block(b, rng))
        vals.append(D / n ** 2)
        ns.append(n)
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    tree.append({"b": b, "E_D_over_n2_adv": m, "se": se, "card": CARD[k], "z": (m - CARD[k]) / se,
                 "E_n_obs": float(np.mean(ns)), "E_n_exact": E_N[k], "sd_n_tree": float(np.std(ns, ddof=1)),
                 "max_D_over_n2": float(np.max(vals)), "bound_b2": b * b})
    print(json.dumps(tree[-1]), flush=True)
adv_pred_slope = cm.fit_slope(E_N, [math.sqrt(t["E_D_over_n2_adv"]) for t in tree])
adv_pred_ratio = math.sqrt(tree[-1]["E_D_over_n2_adv"]) * 36 / math.sqrt(35)
out["tree_mc"] = {"table": tree, "adv_pred_slope": adv_pred_slope, "adv_pred_ratio": adv_pred_ratio,
                  "card_pred_slope": 0.5047, "card_pred_ratio": 6.832}
print(json.dumps({"adv_pred_slope": adv_pred_slope, "adv_pred_ratio": adv_pred_ratio}, indent=1), flush=True)

out["runtime_s"] = time.perf_counter() - t0
(HERE / "a1_rerun_qspine.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("DONE", out["runtime_s"], flush=True)
