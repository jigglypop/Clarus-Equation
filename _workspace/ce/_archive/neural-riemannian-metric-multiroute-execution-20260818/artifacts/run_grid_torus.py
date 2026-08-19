"""Frozen v3 Grid-Torus topology/mobility analysis; do not run before audit."""
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path
import numpy as np

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN / "artifacts" / "input" / "grid_torus"
DATA = ROOT / "extracted" / "Toroidal_topology_grid_cell_data"
UTILS = ROOT / "GridCellTorus-code" / "utils.py"
OUT = RUN / "artifacts" / "grid-torus-v3-results.json"
FILES = (("Q", "rat_q_grid_modules_1_2.npz", ""), ("R", "rat_r_day2_grid_modules_1_2_3.npz", "day2"), ("S", "rat_s_grid_modules_1.npz", ""))
BIN, WAKE_SIG, SLEEP_SIG, PC, RIDGE, SPLIT_SEED, EPS = .01, .05, .025, 6, 1e-3, 1701, 1e-12

def load_analysis_dependencies():
    global gaussian_filter1d, PCA, StandardScaler, ripser
    from scipy.ndimage import gaussian_filter1d
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from ripser import ripser

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def facts():
    out = {}
    for n in ast.parse(UTILS.read_text()).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript) and getattr(n.targets[0].value, "id", None) == "times_all":
            out[ast.literal_eval(n.targets[0].slice)] = ast.literal_eval(n.value)
    return out

def bin_bouts(spikes, intervals):
    out = []
    for bout, (lo, hi) in enumerate(intervals):
        n = int(np.floor((hi - lo) / BIN)); edges = lo + np.arange(n + 1) * BIN
        if n < 1000: continue
        counts = np.vstack([np.histogram(np.asarray(spikes[k])[(np.asarray(spikes[k]) >= lo) & (np.asarray(spikes[k]) < edges[-1])], bins=edges)[0] for k in sorted(spikes)]).T.astype(float)
        out.append({"bout_id": bout, "lo": lo, "hi": hi, "counts": counts})
    return out

def blocks(bouts, state):
    out, size = [], int(10 / BIN); sigma = WAKE_SIG / BIN if state == "wake" else SLEEP_SIG / BIN
    for bout in bouts:
        for i in range(len(bout["counts"]) // size):
            # Filtering never crosses a 10 s block or bout boundary.
            x = bout["counts"][i * size:(i + 1) * size]
            out.append({"bout_id": bout["bout_id"], "block_id": i, "data": gaussian_filter1d(x, sigma, axis=0)})
    return out

def ids(xs): return [{"bout_id": x["bout_id"], "block_id": x["block_id"]} for x in xs]

def split_roles(xs, seed, calibration=False):
    order = np.arange(len(xs)); np.random.default_rng(seed).shuffle(order)
    nc = max(1, int(np.floor(.2 * len(xs)))) if calibration else 0
    c, remaining = [xs[i] for i in order[:nc]], order[nc:]; cut = len(remaining) // 2
    return c, [xs[i] for i in remaining[:cut]], [xs[i] for i in remaining[cut:]]

def check_roles(c, a, b):
    ci, ai, bi = ({(x["bout_id"], x["block_id"]) for x in q} for q in (c, a, b))
    if not a or not b or ci & ai or ci & bi or ai & bi: raise ValueError("FAILED_EXECUTION non-disjoint or empty C/A/B blocks")

def fit_chart(c):
    x = np.concatenate([q["data"] for q in c]); s = StandardScaler().fit(x); p = PCA(n_components=PC, random_state=SPLIT_SEED).fit(s.transform(x))
    h = hashlib.sha256()
    for q in (s.mean_, s.scale_, p.components_, p.explained_variance_): h.update(np.ascontiguousarray(q).tobytes())
    return s, p, {"fit": "wake_C_only", "n_blocks": len(c), "n_samples": len(x), "dimension": PC, "parameter_sha256": h.hexdigest()}

def project(xs, s, p): return [p.transform(s.transform(q["data"])) for q in xs]

def mobility(zs):
    dz, z = np.concatenate([np.diff(q, axis=0) for q in zs]), np.concatenate(zs)
    base = np.cov(dz.T) + RIDGE * np.cov(z.T); ev = np.linalg.eigvalsh(base)
    if not np.isfinite(base).all() or ev.min() <= 0: raise ValueError("FAILED_EXECUTION non-SPD mobility precision")
    return np.linalg.inv(base), {"n_blocks": len(zs), "n_samples": len(z), "n_within_block_diffs": len(dz), "condition": float(np.linalg.cond(base))}

def topology(zs):
    z = np.concatenate(zs); take = np.linspace(0, len(z) - 1, min(500, len(z)), dtype=int)
    dgm = ripser(z[take], metric="cosine", maxdim=1, coeff=47)["dgms"][1]
    life = sorted((float(q[1] - q[0]) for q in dgm if np.isfinite(q[1])), reverse=True)[:3]
    if len(life) < 2: raise ValueError("FAILED_EXECUTION insufficient finite H1 lifetimes")
    return {"n_blocks": len(zs), "n_samples": len(z), "n_points": len(take), "h1_top3_lifetime": life}

def padded(q): return np.pad(np.asarray(q["h1_top3_lifetime"], float), (0, 3 - len(q["h1_top3_lifetime"])))
def topdist(a, b):
    """Symmetric normalized L2 distance for cross-state and A/B comparisons."""
    na, nb = float(np.linalg.norm(padded(a))), float(np.linalg.norm(padded(b)))
    return float(np.linalg.norm(padded(a) - padded(b)) / max((na + nb) / 2, EPS))
def airm(a, b):
    e, v = np.linalg.eigh(a); r = v @ np.diag(1 / np.sqrt(e)) @ v.T
    return float(np.linalg.norm(np.log(np.linalg.eigvalsh(r @ b @ r))))
def ratio(distance, local_noise, wake_noise): return float(distance / max((local_noise + wake_noise) / 2, EPS))

def run_module(rat, module, states):
    try:
        seed = SPLIT_SEED + 1000 * ord(rat) + 100 * module
        roles = {"wake": split_roles(states["wake"], seed, True)}
        roles.update({state: split_roles(states[state], seed + i, False) for i, state in enumerate(("REM", "SWS"), 1)})
        for c, a, b in roles.values(): check_roles(c, a, b)
        s, p, chart = fit_chart(roles["wake"][0]); records = {}
        for state, (c, a, b) in roles.items():
            za, zb = project(a, s, p), project(b, s, p); ta, tb = topology(za), topology(zb); ma, mai = mobility(za); mb, mbi = mobility(zb)
            records[state] = {"chart_parameter_sha256": chart["parameter_sha256"], "calibration_C": ids(c), "analysis_A": ids(a), "analysis_B": ids(b), "topology_A": ta, "topology_B": tb, "mobility_precision_A": ma.tolist(), "mobility_precision_B": mb.tolist(), "mobility_A": mai, "mobility_B": mbi, "within_split_noise": {"topology_A_B_normalized_l2": topdist(ta, tb), "metric_A_B_airm": airm(ma, mb)}}
        wake, contrasts = records["wake"], {}
        for state, r in records.items():
            pt, pm = topdist(r["topology_A"], wake["topology_A"]), airm(np.asarray(r["mobility_precision_B"]), np.asarray(wake["mobility_precision_B"]))
            st, sm = topdist(r["topology_B"], wake["topology_B"]), airm(np.asarray(r["mobility_precision_A"]), np.asarray(wake["mobility_precision_A"]))
            ptt, pmm = ratio(pt, r["within_split_noise"]["topology_A_B_normalized_l2"], wake["within_split_noise"]["topology_A_B_normalized_l2"]), ratio(pm, r["within_split_noise"]["metric_A_B_airm"], wake["within_split_noise"]["metric_A_B_airm"])
            stt, smm = ratio(st, r["within_split_noise"]["topology_A_B_normalized_l2"], wake["within_split_noise"]["topology_A_B_normalized_l2"]), ratio(sm, r["within_split_noise"]["metric_A_B_airm"], wake["within_split_noise"]["metric_A_B_airm"])
            ok = ptt <= 1 and pmm > 1 and stt <= 1 and smm > 1
            contrasts[state] = {"state_reference": "wake", "primary": {"topology_A_to_wake_A": pt, "metric_B_to_wake_B_airm": pm, "topology_ratio": ptt, "metric_ratio": pmm}, "swap": {"topology_B_to_wake_B": st, "metric_A_to_wake_A_airm": sm, "topology_ratio": stt, "metric_ratio": smm}, "dissociation_compatible": ok, "dissociation_compatible_label": "one_split_heuristic_no_p_or_population_inference"}
        return {"rat": rat, "module": module, "status": "COMPLETE", "chart": chart, "states": records, "contrasts": contrasts}
    except Exception as e: return {"rat": rat, "module": module, "status": "FAILED_EXECUTION", "error": str(e)}

def fixtures():
    spikes = {0: np.asarray([0., .009, .01, .019, .02, .029, .03])}
    assert bin_bouts(spikes, [(0., 10.035)])[0]["counts"][:3].ravel().tolist() == [2., 2., 2.]
    xs = [{"bout_id": 0, "block_id": i, "data": np.ones((3, 1))} for i in range(10)]; c, a, b = split_roles(xs, 1, True); check_roles(c, a, b); assert len(c) == 2
    rng = np.random.default_rng(4); z = [rng.normal(size=(20, 6)), rng.normal(size=(20, 6))]; assert mobility(z)[1]["n_within_block_diffs"] == 38
    eye, signature = np.eye(6), {"h1_top3_lifetime": [3., 2., 1.]}; changed = {"h1_top3_lifetime": [2., 1., .5]}
    assert airm(eye, eye) == 0. and topdist(signature, signature) == 0. and topdist(signature, changed) == topdist(changed, signature)
    print('{"status":"PASS","fixtures":"strict_bins_block_diffs_calibration_disjointness_zero_distances"}')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--fixtures", action="store_true"); args = ap.parse_args()
    if args.fixtures: fixtures(); return
    load_analysis_dependencies()
    fs, modules, hashes = facts(), [], {"utils.py": sha(UTILS)}
    for rat, filename, day in FILES:
        path = DATA / filename; hashes[filename] = sha(path); loaded = np.load(path, allow_pickle=True)
        for module in range(1, 4):
            if f"spikes_mod{module}" not in loaded: continue
            spikes = loaded[f"spikes_mod{module}"].item(); states = {name: blocks(bin_bouts(spikes, fs[f"rat_{rat.lower()}_{tag}{day}"]), name) for name, tag in (("wake", "OF"), ("REM", "REM"), ("SWS", "SWS"))}
            modules.append(run_module(rat, module, states) if all(states.values()) else {"rat": rat, "module": module, "status": "FAILED_EXECUTION", "error": "empty_state_blocks"})
    payload = {"schema": "nrm-grid-torus-v3", "claim_boundary": "topology_metric_dissociation_only_no_structural_W", "inputs_sha256": hashes, "frozen": {"bin_seconds": BIN, "interval_rule": "[lo,hi) with incomplete_tail_discarded", "split_seed": SPLIT_SEED, "split": "wake_C_20_percent_then_A_B; REM_SWS_A_B; disclosed_pre_outcome_audit_amendment", "pca_dim": PC, "metric": "mobility_precision=(C_delta+lambda_R_C)^-1", "ridge": RIDGE, "topology": "ripser cosine maxdim=1 coeff=47", "physical_h": "unmeasured", "wake_reference": "state_reference_only_not_longitudinal_g0_to_gt", "pure_cell_filter": "not_available_in_payload"}, "modules": modules, "population_inference": "not_performed_N_equals_3"}
    if OUT.exists() and not args.overwrite: raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2) + "\n"); print(OUT)
if __name__ == "__main__": main()
