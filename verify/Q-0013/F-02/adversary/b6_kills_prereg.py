"""adversary b6: kill executability (smoke sizes, off-grid, different seed),
preregistration hygiene (PRED/WINDOW vs structure_constants.json), MIN_DET behaviour,
and the already_observed transfer e01 -> e03.
"""
from __future__ import annotations
import importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
spec = importlib.util.spec_from_file_location("cf", ROOT / "verify/Q-0013/F-02/check_floor.py")
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)
SC = json.loads((ROOT / "verify/Q-0013/F-02/structure_constants.json").read_text(encoding="utf-8"))


def main():
    out = {}
    # (1) preregistration bookkeeping
    ep, wm = SC["exact_pred"], SC["window_from_model"]
    se, mism = SC["model_se"], {}
    for k, v in cf.PRED.items():
        mism[k] = {"card_script_PRED": v, "exact_pred": ep.get(k),
                   "abs_diff": abs(v - ep[k]) if k in ep else None}
    wmis = {}
    for k, (lo, hi) in cf.WINDOW.items():
        if k not in wm:
            wmis[k] = "no model window (zero mode)"
            continue
        wmis[k] = {"script": [lo, hi], "model": wm[k],
                   "lo_diff": lo - wm[k][0], "hi_diff": hi - wm[k][1],
                   "script_wider": bool(lo <= wm[k][0] + 1e-9 and hi >= wm[k][1] - 1e-9)}
    rule = {}
    for k in se:
        v = ep[k]
        half = 4.0 * se[k] + (0.0 if k in ("ker_slope", "diag_slope", "univ_floor_hat_over_delta2",
                                           "cross_eps2_sq_over_delta4") else 0.01 * abs(v))
        if k == "cross_eps2_sq_over_delta4":
            half += 0.02 * ep["iso_eps2_over_delta2"] ** 2
        rule[k] = {"half_width_rule": half, "script_half": (cf.WINDOW[k][1] - cf.WINDOW[k][0]) / 2.0,
                   "sigma_equivalent": (cf.WINDOW[k][1] - cf.WINDOW[k][0]) / 2.0 / se[k]}
    out["prereg"] = {"PRED_vs_exact": mism, "WINDOW_vs_model": wmis, "window_rule_check": rule,
                     "script_WINDOW_recorded_in_constants_json": SC["script_WINDOW"]}

    # (2) kill executability: off-grid sizes, different seed, tiny trials
    cf.SEED = 20260903
    exe = {}
    t0 = time.time()
    for name, fn, args in (("kernel", cf.mode_kernel, ((5, 9), 12)),
                           ("diag", cf.mode_diag, ((5, 9), 12)),
                           ("univ", cf.mode_univ, ((5, 9), 12)),
                           ("zero", cf.mode_zero, ((5, 9), 12)),
                           ("axis", cf.mode_axis, ((5, 9), 12))):
        try:
            if name == "diag":
                st = fn(args[0], args[1], n2_trials=40)
            else:
                st = fn(*args)
            exe[name] = {"ok": True, "stats": {k: (v if not isinstance(v, dict) else "curve")
                                               for k, v in st.items()}}
        except Exception as exc:
            exe[name] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        print(name, json.dumps(exe[name])[:260], flush=True)
    out["kill_executability_offgrid"] = exe

    # (3) MIN_DET at the F-01 failure point delta=0.3 and at the preregistered 0.1
    dz = {}
    for deltas in ((0.1,), (0.3,), (1.0,)):
        cf.ZERO_DELTAS = deltas
        try:
            st = cf.mode_zero((5, 9), 24)
            dz[str(deltas[0])] = {"ok": True, "max_residual": st["zero_max_residual"],
                                  "resampled": st["zero_resampled"]}
        except Exception as exc:
            dz[str(deltas[0])] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    cf.ZERO_DELTAS = (0.005, 0.1)
    out["zero_mode_min_det"] = dz

    # (4) resampling probability at delta=0.1 for the two preregistered zero modes
    rng = np.random.default_rng(20260903)
    probs = {}
    for name, spec_ in (("zero_11", cf.SPECS["zero_11"]), ("zero_3diag", cf.SPECS["zero_3diag"]),
                        ("iso16", cf.SPECS["iso16"]), ("kernel", cf.SPECS["kernel"])):
        A = cf.factor(spec_)
        g = rng.normal(size=(200000, A.shape[1]))
        lab = (g @ A.T).reshape(-1, 4, 4)
        for d in (0.005, 0.1, 0.3):
            det = np.linalg.det(np.eye(4)[None] + d * lab)
            probs["%s_d%s" % (name, d)] = float(np.mean(det <= cf.MIN_DET))
    out["p_det_below_min_det_per_cell"] = probs

    # (5) already_observed transfer: e01 vs e03 rank-1 with common random numbers
    def sweep_common(specA, specB, n, trials, delta, seed):
        rngl = np.random.default_rng(seed)
        A1, A2 = cf.factor(specA), cf.factor(specB)
        a, b = [], []
        for _ in range(trials):
            g = rngl.normal(size=(n, A1.shape[1]))
            l1 = (g @ A1.T).reshape(n, 4, 4)
            l2 = (g @ A2.T).reshape(n, 4, 4)
            a.append(cf.block_residual(l1, delta))
            b.append(cf.block_residual(l2, delta))
        a, b = np.asarray(a), np.asarray(b)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))
    e = cf.e
    out["per_realization_identity"] = {
        "e01_vs_e03_max_rel_diff": sweep_common([(e(0, 1), 1.0)], [(e(0, 3), 1.0)], 9, 40, 0.005, 20260903),
        "e01_vs_e02_max_rel_diff": sweep_common([(e(0, 1), 1.0)], [(e(0, 2), 1.0)], 9, 40, 0.005, 20260904),
        "e00e11e22_vs_e00e11e33_max_rel_diff": sweep_common(
            [((e(0, 0) + e(1, 1) + e(2, 2)) / math.sqrt(3), 1.0)],
            [((e(0, 0) + e(1, 1) + e(3, 3)) / math.sqrt(3), 1.0)], 9, 40, 0.005, 20260905),
    }
    out["_meta"] = {"seed": 20260903, "seconds": time.time() - t0,
                    "note": "off-grid sizes and different seed; the preregistered-size kills were NOT run"}
    (OUT / "b6_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"exe": {k: v.get("ok") for k, v in exe.items()},
                      "zero_min_det": dz, "probs": probs,
                      "ident": out["per_realization_identity"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
