"""b4 (re-audit): is the surrogate standard error the right yardstick for the physical run?

The pre-registered K1-K4 half-widths are 4 x the SURROGATE delta-method se at N = 8192.  The only
physical (tetrad) measurement of the same statistic is a3's 384-trial paired run, whose delta-method
se for rho_cat6_spike64 was 0.2717 (form) / 0.2772 (physics) -- rescaled to N = 8192 that is 0.0588,
i.e. 1.53x the surrogate value 0.0385 that fixes the window.  If the physical sd really were 1.5x
the design sd, the '4 sigma' windows would be 2.6 sigma.

Cheap discriminator: the same SURROGATE at 384 trials, replicated 400 times.  If a delta-method se
of 0.27 at n = 384 is an ordinary draw of the heavy-tailed spike64 statistic, a3's number is
sampling noise of the se estimator and the design is intact; if it is far in the tail, the physical
run is genuinely more variable than the design object and the windows are narrower than claimed.
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0012" / "F-01" / "adversary"))
from check_cumulant import linear_map, quadratic_tensor, caterpillar, ancestor_matrix, DISTS  # noqa
from a5_window_coverage import labels_from_z, ratio_and_se, N_CELLS  # noqa

OUT = Path(__file__).parent
A3_SE_FORM = {"cat6_spike64": 0.27174741111305883, "iid36_spike64": 0.12108460728389069,
              "cat6_laplace": 0.02854282731519862, "cat6_rademacher": 0.0529078934358427}


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    M = quadratic_tensor(linear_map())
    H = np.eye(N_CELLS) - np.ones((N_CELLS, N_CELLS)) / N_CELLS
    gens = {"iid36": H, "cat6": H @ ancestor_matrix(caterpillar(6))}
    rng = np.random.default_rng(777001)
    ses = {f"{m}_{d}": [] for m in gens for d in DISTS if d != "gauss"}
    rhos = {k: [] for k in ses}
    for _ in range(reps):
        z = rng.standard_normal((trials, N_CELLS, 16))
        vals = {}
        for dist in DISTS:
            zeta = labels_from_z(z, dist)
            for mode, HA in gens.items():
                D = np.einsum("vu,tua->tva", HA, zeta, optimize=True)
                G = np.einsum("tva,tvb->tab", D, D, optimize=True)
                phi = np.einsum("tab,abij->tij", G, M, optimize=True)
                vals[f"{mode}_{dist}"] = np.sum(phi * phi, axis=(1, 2))
        for mode in gens:
            for dist in DISTS:
                if dist == "gauss":
                    continue
                r, se = ratio_and_se(vals[f"{mode}_{dist}"], vals[f"{mode}_gauss"])
                ses[f"{mode}_{dist}"].append(se)
                rhos[f"{mode}_{dist}"].append(r)
    res = {"replicates": reps, "trials": trials, "note": "surrogate only; compares a3's 384-trial se"}
    for k in ses:
        s = np.array(ses[k]); r = np.array(rhos[k])
        row = {"se_median": float(np.median(s)), "se_p05": float(np.quantile(s, .05)),
               "se_p95": float(np.quantile(s, .95)), "se_max": float(s.max()),
               "rho_sd_over_replicates": float(r.std(ddof=1)),
               "sd_over_median_se": float(r.std(ddof=1) / np.median(s))}
        if k in A3_SE_FORM:
            row["a3_se_384"] = A3_SE_FORM[k]
            row["a3_quantile_in_surrogate"] = float(np.mean(s <= A3_SE_FORM[k]))
            row["a3_over_median"] = A3_SE_FORM[k] / float(np.median(s))
        res[k] = row
        print("%-18s se med %.4f [p05 %.4f p95 %.4f max %.4f]  sd(rho) %.4f  sd/medse %.2f  %s"
              % (k, row["se_median"], row["se_p05"], row["se_p95"], row["se_max"],
                 row["rho_sd_over_replicates"], row["sd_over_median_se"],
                 ("a3 se %.4f -> quantile %.3f" % (row["a3_se_384"], row["a3_quantile_in_surrogate"]))
                 if k in A3_SE_FORM else ""))
    (OUT / "b4_se_scale_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
