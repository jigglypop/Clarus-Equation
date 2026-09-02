"""a3: does the O(delta) truncation error swamp the fourth-cumulant signal?

K5 (card ladder step 4) checks the quadratic-form identity ONLY for gauss / rademacher /
heritable_gauss labels.  The flagship predictions (P2, P5) are carried by spike64, whose label
magnitude is 8, i.e. the expansion parameter delta*|zeta| is 8x larger than for a Gaussian draw.

Two measurements, both at small size and with a NON-preregistered seed:
  (A) form-mode extended to all five pre-registered laws (deterministic, per configuration).
  (B) paired estimator of the bias that the truncation puts into the predicted ratio rho:
      for the same labels compute the physical residual^2 and the quadratic-form residual^2, then
          rho_phys / rho_form = [E eps2_phys(d) / E eps2_form(d)] / [same for gauss] =: B_d / B_g.
      rho_form is (up to MC error) the card's exact 1 + kappa4 c4 S_gen/D, so B_d/B_g - 1 is
      exactly the systematic error of the card's prediction at delta = 0.005.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
from check_cumulant import (linear_map, quadratic_tensor, gram_form, REFERENCE, caterpillar,
                            ancestor_matrix, heritable, block_residual, quadratic_residual,
                            uniform_to_label, normal_cdf, KAPPA4, DISTS)

OUT = Path(__file__).parent
ADV_SEED = 20260902 + 31337          # deliberately NOT the pre-registered seed
DISTS5 = ("gauss", "rademacher", "uniform", "laplace", "spike64")


def analytic_rho(HA, dist):
    B = HA.T @ HA
    D = float(np.trace(B @ B))
    S = float(np.sum(np.diag(B) ** 2))
    return 1.0 + KAPPA4[dist] * S / (60.0 * D)


def mode_form(M, norm_g0, sizes=(3, 5, 8, 12), deltas=(0.005, 0.001)):
    rng = np.random.default_rng(ADV_SEED)
    rows = []
    for n in sizes:
        z = rng.standard_normal((n, 4, 4))
        u = normal_cdf(z)
        parent = [-1] + [(i - 1) // 2 for i in range(1, n)]
        for dist in DISTS5:
            zeta = uniform_to_label(u, z, dist)
            for tag, labels in (("iid", zeta), ("her", heritable(parent, zeta))):
                row = {"n": n, "dist": dist, "mode": tag}
                for d in deltas:
                    act = block_residual(labels, d)
                    pred = quadratic_residual(labels, d, M, norm_g0)
                    row["d%g" % d] = {"actual": act, "form": pred,
                                      "rel_err": (act - pred) / pred if pred > 0 else float("nan")}
                rows.append(row)
    return rows


def _ratio_se(a, b):
    """delta-method se of mean(a)/mean(b) for paired samples."""
    n = len(a)
    ma, mb = float(a.mean()), float(b.mean())
    c = np.cov(a, b)
    var = (c[0, 0] / mb ** 2 - 2 * ma * c[0, 1] / mb ** 3 + ma ** 2 * c[1, 1] / mb ** 4) / n
    return ma / mb, math.sqrt(max(var, 0.0))


def mode_paired(M, norm_g0, n, parent, trials, delta, seed):
    HA = np.eye(n) - np.ones((n, n)) / n
    if parent is not None:
        HA = HA @ ancestor_matrix(parent)
    rng = np.random.default_rng(seed)
    phys = {d: np.empty(trials) for d in DISTS5}
    form = {d: np.empty(trials) for d in DISTS5}
    for t in range(trials):
        z = rng.standard_normal((n, 4, 4))
        u = normal_cdf(z)
        for dist in DISTS5:
            zeta = uniform_to_label(u, z, dist)
            labels = zeta if parent is None else heritable(parent, zeta)
            p = block_residual(labels, delta)
            q = quadratic_residual(labels, delta, M, norm_g0)
            phys[dist][t] = p * p
            form[dist][t] = q * q
    out = {"n": n, "trials": trials, "delta": delta, "seed": seed, "rows": {}}
    for dist in DISTS5:
        rho_p, se_p = _ratio_se(phys[dist], phys["gauss"])
        rho_f, se_f = _ratio_se(form[dist], form["gauss"])
        bd = float(phys[dist].mean() / form[dist].mean())
        bg = float(phys["gauss"].mean() / form["gauss"].mean())
        ratio_of_B = bd / bg
        rb = np.random.default_rng(seed + 99)
        boot = np.empty(600)
        for k in range(600):
            idx = rb.integers(0, trials, trials)
            boot[k] = ((phys[dist][idx].mean() / form[dist][idx].mean())
                       / (phys["gauss"][idx].mean() / form["gauss"][idx].mean()))
        se_rb = float(boot.std(ddof=1))
        out["rows"][dist] = {
            "kappa4": KAPPA4[dist], "rho_phys": rho_p, "se_phys": se_p,
            "rho_form": rho_f, "se_form": se_f, "rho_analytic": analytic_rho(HA, dist),
            "B_dist": bd, "B_gauss": bg, "bias_rho_phys_over_form": ratio_of_B, "se_bias": se_rb,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="form", choices=("form", "paired12", "paired36", "paired36iid"))
    ap.add_argument("--trials", type=int, default=384)
    args = ap.parse_args()
    M = quadratic_tensor(linear_map())
    norm_g0 = float(np.linalg.norm(gram_form(REFERENCE, REFERENCE)))
    t0 = time.time()

    if args.part == "form":
        rows = mode_form(M, norm_g0)
        agg = {}
        for dist in DISTS5:
            for mode in ("iid", "her"):
                sel = [r for r in rows if r["dist"] == dist and r["mode"] == mode]
                c = np.array([r["d0.005"]["rel_err"] for r in sel])
                f = np.array([r["d0.001"]["rel_err"] for r in sel])
                agg["%s_%s" % (dist, mode)] = {
                    "max_abs_rel_err_d0005": float(np.max(np.abs(c))),
                    "rms_rel_err_d0005": float(np.sqrt(np.mean(c ** 2))),
                    "rms_rel_err_d0001": float(np.sqrt(np.mean(f ** 2))),
                    "delta_scaling_ratio": float(np.sqrt(np.mean(c ** 2)) / np.sqrt(np.mean(f ** 2))),
                }
                print("%-11s %-4s  max|rel|(d=5e-3) %9.2e   rms %9.2e   rms(d=1e-3) %9.2e   ratio %5.2f"
                      % (dist, mode, agg["%s_%s" % (dist, mode)]["max_abs_rel_err_d0005"],
                         agg["%s_%s" % (dist, mode)]["rms_rel_err_d0005"],
                         agg["%s_%s" % (dist, mode)]["rms_rel_err_d0001"],
                         agg["%s_%s" % (dist, mode)]["delta_scaling_ratio"]))
        out = {"kind": "form_extended", "seed": ADV_SEED, "aggregate": agg, "rows": rows}
        name = "a3_form_extended.json"
    else:
        n, parent = {"paired12": (12, None), "paired36iid": (36, None), "paired36": (36, caterpillar(6))}[args.part]
        blocks = {}
        for delta in (0.005, 0.001):
            b = mode_paired(M, norm_g0, n, parent, args.trials, delta, ADV_SEED + int(delta * 1e6))
            blocks["delta%g" % delta] = b
            print("--- n=%d %s delta=%g trials=%d ---"
                  % (n, "iid" if parent is None else "cat6", delta, args.trials))
            for dist in DISTS5:
                r = b["rows"][dist]
                print("  %-11s rho_phys %8.5f+-%7.5f  rho_form %8.5f+-%7.5f  analytic %8.5f   "
                      "bias(phys/form) %+9.2e +- %7.2e"
                      % (dist, r["rho_phys"], r["se_phys"], r["rho_form"], r["se_form"],
                         r["rho_analytic"], r["bias_rho_phys_over_form"] - 1.0, r["se_bias"]))
        out = {"kind": "paired", "seed": ADV_SEED, "blocks": blocks}
        name = "a3_%s.json" % args.part
    out["wall_seconds"] = time.time() - t0
    (OUT / name).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wall %.1f s -> %s" % (out["wall_seconds"], name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
