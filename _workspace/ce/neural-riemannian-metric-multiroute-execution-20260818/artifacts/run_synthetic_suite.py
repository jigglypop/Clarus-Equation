"""Frozen blind synthetic v3 metric tournament. Do not tune after execution."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, tempfile
from pathlib import Path
import numpy as np

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "artifacts" / "synthetic-v3-results.json"
D, NODES, TRAIN, TEST, TRAJ, STEPS, DT = 3, 8, 20, 20, 24, 320, 0.02
SIGMA, K_TRUE, K_FLOOR = 0.2, np.array([0.55, 0.8, 1.05]), 1e-6
KINDS = ("G1_metric", "G2_direct_vq", "G3_flat_pullback", "G4_gain_noise", "G5_shortcut", "G6_null")
BASELINES = ("direct_vQ", "gain_noise", "noise_only", "euclidean", "flat_pullback")

def h_value(phi, x): return (phi - 0.28) * (1.0 + 0.2 * np.tanh(x[..., 0]))
def f_map(z, c=0.22):
    y = z.copy(); y[..., 1] += c * z[..., 0] * z[..., 2]; return y
def f_inverse(y, c):
    z = y.copy(); z[..., 1] -= c * y[..., 0] * y[..., 2]; return z
def force_series(direction):
    force = np.zeros((TRAJ, STEPS, D)); force[:, STEPS // 2 :] = 0.22 * direction; return force

def inverse_pushforward_force(y, force_y, c):
    """For y=f(z), map the same physical observed-y force into z coordinates."""
    result = force_y.copy()
    result[..., 1] -= c * y[..., 2] * force_y[..., 0] + c * y[..., 0] * force_y[..., 2]
    return result

def forward_jacobian(z, c):
    return np.array(((1., 0., 0.), (c * z[2], 1., c * z[0]), (0., 0., 1.)))

def make_w(rng):
    w = (rng.random((NODES, NODES)) < rng.uniform(0.08, 0.48)).astype(float)
    np.fill_diagonal(w, 0.0)
    return w, float(w.sum() / (NODES * (NODES - 1)))

def simulate(kind, seed, direction):
    rng = np.random.default_rng(seed); w, phi = make_w(rng)
    if kind == "G5_shortcut":
        w[0, 5] = w[5, 0] = 1.0; phi = float(w.sum() / (NODES * (NODES - 1)))
    x, force_y = np.zeros((TRAJ, STEPS + 1, D)), force_series(direction)
    for t in range(STEPS):
        state, h = x[:, t], h_value(phi, x[:, t])
        if kind == "G1_metric":
            m = np.c_[np.ones(TRAJ), np.ones(TRAJ), np.exp(-1.2 * h)]; q, drift = SIGMA**2 * m, -m * K_TRUE * state
        elif kind == "G2_direct_vq":
            m = np.c_[np.ones(TRAJ), np.ones(TRAJ), np.exp(-1.2 * h)]; q, drift = np.full((TRAJ, D), SIGMA**2), -m * K_TRUE * state
        elif kind == "G4_gain_noise":
            m = np.exp(-h)[:, None]; q, drift = SIGMA**2 * np.repeat(m, D, axis=1), -m * K_TRUE * state
        else:
            q, drift = np.full((TRAJ, D), SIGMA**2), -K_TRUE * state
        force = inverse_pushforward_force(f_map(state), force_y[:, t], .22) if kind == "G3_flat_pullback" else force_y[:, t]
        x[:, t + 1] = state + (drift + force) * DT + rng.normal(size=(TRAJ, D)) * np.sqrt(q * DT)
    observed = f_map(x) if kind == "G3_flat_pullback" else x
    true = {"Kdiag": K_TRUE.tolist(), "sigma": SIGMA, "mobility": "euclidean", "beta": 0., "c": 0.}
    if kind == "G1_metric": true.update({"mobility": "anisotropic_coupled", "beta": 1.2})
    elif kind == "G2_direct_vq": true.update({"mobility": "anisotropic_direct_q_euclidean", "beta": 1.2})
    elif kind == "G3_flat_pullback": true.update({"mobility": "flat_pullback", "c": .22})
    elif kind == "G4_gain_noise": true.update({"mobility": "conformal_coupled", "beta": 1.})
    elif kind == "G5_shortcut": true.update({"mobility": "euclidean_shortcut"})
    return {"W": w, "phi": phi, "paths": observed, "direction": np.asarray(direction),
            "intervention_start": STEPS // 2, "truth": true}

def samples(observations):
    xs, ds, fs, phis = [], [], [], []
    for o in observations:
        path = o["paths"]
        xs.append(path[:, :-1].reshape(-1, D)); ds.append((path[:, 1:] - path[:, :-1]).reshape(-1, D))
        fs.append(force_series(o["direction"]).reshape(-1, D)); phis.append(np.full(TRAJ * STEPS, o["phi"]))
    return tuple(np.concatenate(v) for v in (xs, ds, fs, phis))

def model_fields(name, param, phi, x):
    h = h_value(phi, x)
    if name == "metric":
        m = np.c_[np.ones(len(x)), np.ones(len(x)), np.exp(-param * h)]; return m, m
    if name == "direct_vQ":
        bv, bq = param; return np.c_[np.ones(len(x)), np.ones(len(x)), np.exp(-bv * h)], np.c_[np.ones(len(x)), np.ones(len(x)), np.exp(-bq * h)]
    if name == "gain_noise":
        m = np.exp(-param * h)[:, None]; return np.repeat(m, D, axis=1), np.repeat(m, D, axis=1)
    if name == "noise_only":
        q = np.c_[np.ones(len(x)), np.ones(len(x)), np.exp(-param * h)]; return np.ones_like(q), q
    return np.ones((len(x), D)), np.ones((len(x), D))

def fit_diagonal(x, delta, force, phi, name, param):
    mv, mq, target = *model_fields(name, param, phi, x), delta / DT - force
    k = np.empty(D)
    for j in range(D):
        a, weights = mv[:, j] * x[:, j], 1.0 / mq[:, j]
        k[j] = max(K_FLOOR, -np.sum(weights * a * target[:, j]) / np.sum(weights * a * a))
    residual = delta - (-mv * k * x + force) * DT
    sigma2 = float(np.mean(np.sum(residual * residual / mq, axis=1)) / (D * DT))
    return k, max(sigma2, 1e-12)

def nll(x, delta, force, phi, name, param, k, sigma2):
    mv, mq = model_fields(name, param, phi, x); residual = delta - (-mv * k * x + force) * DT; qdt = sigma2 * mq * DT
    return 0.5 * np.sum(residual * residual / qdt + np.log(qdt) + math.log(2.0 * math.pi), axis=1)

def fit_grid(x, delta, force, phi, name, grid):
    best = None
    for param in grid:
        k, sigma2 = fit_diagonal(x, delta, force, phi, name, param)
        value = float(np.mean(nll(x, delta, force, phi, name, param, k, sigma2)))
        if best is None or value < best[0]: best = value, param, k, sigma2
    return {"name": name, "param": best[1], "Kdiag": best[2], "sigma2": best[3], "train_nll": best[0]}

def fit_flat(x, delta, force, phi):
    best = None
    for c in np.arange(-0.35, 0.3501, 0.05):
        z, dz = f_inverse(x, c), f_inverse(x + delta, c) - f_inverse(x, c)
        force_z = inverse_pushforward_force(x, force, c)
        k, sigma2 = fit_diagonal(z, dz, force_z, phi, "euclidean", 0.0)
        value = float(np.mean(nll(z, dz, force_z, phi, "euclidean", 0.0, k, sigma2)))
        if best is None or value < best[0]: best = value, float(c), k, sigma2
    return {"name": "flat_pullback", "param": best[1], "Kdiag": best[2], "sigma2": best[3], "train_nll": best[0]}

def fit_candidates(observations):
    """Blind API: observations only; no generator label or truth enters."""
    x, delta, force, phi = samples(observations); coarse = np.arange(-2.0, 2.0001, 0.1)
    direct = list(itertools.product(np.arange(-2.0, 2.0001, 0.2), repeat=2))
    return {"metric": fit_grid(x, delta, force, phi, "metric", coarse), "direct_vQ": fit_grid(x, delta, force, phi, "direct_vQ", direct), "gain_noise": fit_grid(x, delta, force, phi, "gain_noise", coarse), "noise_only": fit_grid(x, delta, force, phi, "noise_only", coarse), "euclidean": fit_grid(x, delta, force, phi, "euclidean", (0.0,)), "flat_pullback": fit_flat(x, delta, force, phi)}

def score_fit(fit, observation):
    x, delta, force, phi = samples((observation,))
    if fit["name"] == "flat_pullback":
        c, z = fit["param"], f_inverse(x, fit["param"])
        return nll(z, f_inverse(x + delta, c) - z, inverse_pushforward_force(x, force, c), phi, "euclidean", 0.0, fit["Kdiag"], fit["sigma2"])
    return nll(x, delta, force, phi, fit["name"], fit["param"], fit["Kdiag"], fit["sigma2"])

def sign_p(differences, seed):
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(4096, len(differences))); observed = float(np.mean(differences))
    return float((1 + np.count_nonzero(np.mean(signs * differences, axis=1) >= observed)) / 4097)
def holm(pvalues):
    reject = {key: False for key in pvalues}
    for rank, key in enumerate(sorted(pvalues, key=pvalues.get)):
        if pvalues[key] <= 0.05 / (len(pvalues) - rank): reject[key] = True
        else: break
    return reject

def scalar_curvature(metric, point, eps=1e-4):
    """Finite-difference R = g^ij R_ij for a C2 SPD metric field."""
    point = np.asarray(point, float)
    def gamma(p):
        g, gi, dg = metric(p), np.linalg.inv(metric(p)), np.empty((D, D, D))
        for a in range(D):
            step = np.zeros(D); step[a] = eps; dg[a] = (metric(p + step) - metric(p - step)) / (2 * eps)
        out = np.zeros((D, D, D))
        for upper in range(D):
            for i in range(D):
                for j in range(D): out[upper, i, j] = .5 * sum(gi[upper, m] * (dg[i, m, j] + dg[j, m, i] - dg[m, i, j]) for m in range(D))
        return out
    gi, gam, dgamm = np.linalg.inv(metric(point)), gamma(point), np.empty((D, D, D, D))
    for a in range(D):
        step = np.zeros(D); step[a] = eps; dgamm[a] = (gamma(point + step) - gamma(point - step)) / (2 * eps)
    ric = np.zeros((D, D))
    for i in range(D):
        for j in range(D): ric[i, j] = sum(dgamm[k, k, i, j] - dgamm[j, k, i, k] + sum(gam[k, k, ell] * gam[ell, i, j] - gam[k, j, ell] * gam[ell, i, k] for ell in range(D)) for k in range(D))
    return float(np.sum(gi * ric))

def pullback_metric(c):
    def metric(y):
        jinv = np.array(((1., 0., 0.), (-c * y[2], 1., -c * y[0]), (0., 0., 1.))); return jinv.T @ jinv
    return metric
def chart_metric(metric, p):
    inv = np.linalg.inv(p); return lambda yprime: inv.T @ metric(inv @ yprime) @ inv
def curvature_fixture():
    points = (np.zeros(D), np.array((.2, -.1, .3)), np.array((-.15, .25, -.2)))
    flat = [abs(scalar_curvature(lambda _: np.eye(D), p)) for p in points]
    pullback = [abs(scalar_curvature(pullback_metric(.22), p)) for p in points]
    curved = [abs(scalar_curvature(lambda x: np.exp(.3 * np.dot(x, x)) * np.eye(D), p)) for p in points]
    return {"euclidean_max_abs_R": max(flat), "pullback_max_abs_R": max(pullback), "conformal_max_abs_R": max(curved)}

def candidate_sign_fixture():
    x, phi = np.zeros((1, D)), np.array([0.33])
    _, q = model_fields("metric", 1.2, phi, x)
    expected = math.exp(-1.2 * h_value(phi, x)[0])
    assert np.isclose(q[0, 2], expected)
    return {"h": float(h_value(phi, x)[0]), "metric_e3_factor": float(q[0, 2]), "expected": expected}

def force_pushforward_fixture():
    z, force_y = np.array((.17, -.09, .23)), np.array((.0, .0, .22))
    force_z = inverse_pushforward_force(f_map(z), force_y, .22)
    assert np.allclose(forward_jacobian(z, .22) @ force_z, force_y)
    return {"force_y": force_y.tolist(), "force_z": force_z.tolist()}

def truth_mesh(observation, truth):
    mesh = np.array(((0., 0., 0.), (.2, -.1, .3), (-.15, .25, -.2)))
    h = h_value(observation["phi"], mesh)
    if truth["mobility"] == "anisotropic_coupled":
        matrices = [np.diag((1., 1., math.exp(1.2 * value))).tolist() for value in h]
    elif truth["mobility"] == "flat_pullback":
        matrices = [pullback_metric(.22)(point).tolist() for point in mesh]
    elif truth["mobility"] == "conformal_coupled":
        matrices = [(math.exp(value) * np.eye(D)).tolist() for value in h]
    else:
        matrices = [np.eye(D).tolist() for _ in mesh]
    return {"points": mesh.tolist(), "g_matrices": matrices}

def first_passage(paths):
    hit = np.linalg.norm(paths, axis=-1) >= .8
    return [int(np.flatnonzero(row)[0]) if np.any(row) else -1 for row in hit]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def publish_exclusive(temp_path, final_path):
    try:
        with temp_path.open("rb") as source, open(final_path, "xb") as target:
            while block := source.read(1024 * 1024): target.write(block)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {final_path}")

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--overwrite", action="store_true"); parser.add_argument("--curvature-fixture", action="store_true"); parser.add_argument("--candidate-sign-fixture", action="store_true"); parser.add_argument("--force-fixture", action="store_true"); args = parser.parse_args()
    if args.curvature_fixture: print(json.dumps(curvature_fixture(), sort_keys=True)); return
    if args.candidate_sign_fixture: print(json.dumps(candidate_sign_fixture(), sort_keys=True)); return
    if args.force_fixture: print(json.dumps(force_pushforward_fixture(), sort_keys=True)); return
    trace_out = RUN / "artifacts" / "synthetic-v3-traces.npz"
    if (OUT.exists() or trace_out.exists()) and not args.overwrite: raise FileExistsError("refusing to overwrite frozen synthetic-v3 result or trace")
    records, train_paths, test_paths, train_w, test_w, train_phi, test_phi = [], [], [], [], [], [], []
    train_seeds, test_seeds = [], []
    for gi, kind in enumerate(KINDS):
        group_train_paths, group_test_paths, group_train_w, group_test_w, group_train_phi, group_test_phi = [], [], [], [], [], []
        group_train_seeds, group_test_seeds = [], []
        for i in range(TEST):
            train_seed, test_seed = 11000 + 1000 * gi + i, 21000 + 1000 * gi + i
            train, test = simulate(kind, train_seed, np.eye(D)[i % 2]), simulate(kind, test_seed, np.eye(D)[2])
            fits = fit_candidates((train,)); scores = {name: score_fit(fit, test).reshape(TRAJ, STEPS).mean(axis=1).tolist() for name, fit in fits.items()}
            counts = {"metric": 5, "direct_vQ": 6, "gain_noise": 5, "noise_only": 5, "euclidean": 4, "flat_pullback": 5}
            records.append({"generator": kind, "test_circuit": i, "fit_id": f"{kind}-{i}", "train_seed": train_seed, "test_seed": test_seed, "phi": test["phi"], "held_direction": test["direction"].tolist(), "scores_by_trajectory": scores, "first_passage_threshold": .8, "test_first_passage": first_passage(test["paths"]), "evaluator_truth": {**test["truth"], "mesh": truth_mesh(test, test["truth"])}, "fits": {name: {"param": np.asarray(fit["param"]).tolist(), "Kdiag": fit["Kdiag"].tolist(), "sigma2": fit["sigma2"], "parameter_count": counts[name]} for name, fit in fits.items()}})
            group_train_paths.append(train["paths"]); group_test_paths.append(test["paths"]); group_train_w.append(train["W"]); group_test_w.append(test["W"]); group_train_phi.append(train["phi"]); group_test_phi.append(test["phi"]); group_train_seeds.append(train_seed); group_test_seeds.append(test_seed)
        train_paths.append(group_train_paths); test_paths.append(group_test_paths); train_w.append(group_train_w); test_w.append(group_test_w); train_phi.append(group_train_phi); test_phi.append(group_test_phi); train_seeds.append(group_train_seeds); test_seeds.append(group_test_seeds)
    g1 = [r for r in records if r["generator"] == "G1_metric"]
    pvalues = {base: sign_p(np.array([np.mean(r["scores_by_trajectory"][base]) - np.mean(r["scores_by_trajectory"]["metric"]) for r in g1]), 501 + j) for j, base in enumerate(BASELINES)}
    recovery = [abs(float(r["fits"]["metric"]["param"]) - 1.2) / 1.2 <= .25 and float(r["fits"]["metric"]["param"]) > 0 for r in g1]
    false_rows = []
    for r in records:
        if r["generator"] == "G1_metric": continue
        ps = {base: sign_p(np.array(r["scores_by_trajectory"][base]) - np.array(r["scores_by_trajectory"]["metric"]), 7000 + 100 * KINDS.index(r["generator"]) + 5 * r["test_circuit"] + j) for j, base in enumerate(BASELINES)}
        reject = holm(ps); false_rows.append({"generator": r["generator"], "test_circuit": r["test_circuit"], "pvalues": ps, "holm": reject, "any_reject": any(reject.values()), "full_promotion": all(reject.values())})
    rng, curvature = np.random.default_rng(90210), []
    g3_records = [r for r in records if r["generator"] == "G3_flat_pullback"]
    for circuit, record in enumerate(g3_records):
        fitted_c, mesh_point, matrices, conditions = float(record["fits"]["flat_pullback"]["param"]), np.array((.11, -.17, .19)), [], []
        metrics = [pullback_metric(fitted_c)]
        while len(conditions) < 3:
            p = np.eye(D) + rng.normal(0, .12, size=(D, D)); condition = float(np.linalg.cond(p))
            if np.isfinite(condition) and condition <= 5.0:
                metrics.append(chart_metric(pullback_metric(fitted_c), p)); matrices.append(p.tolist()); conditions.append(condition)
        values = [abs(scalar_curvature(metric, mesh_point)) for metric in metrics]
        curvature.append({"test_circuit": circuit, "fitted_c": fitted_c, "mesh_point": mesh_point.tolist(), "affine_matrices": matrices, "affine_chart_condition_numbers": conditions, "values": values, "max_abs_R": max(values), "false_positive": max(values) > 1e-3})
    c_recovery = [abs(float(r["fits"]["flat_pullback"]["param"]) - .22) <= .05 for r in g3_records]
    with tempfile.TemporaryDirectory(dir=RUN / "artifacts") as tmpdir:
        temp_trace, temp_json = Path(tmpdir) / "traces.npz", Path(tmpdir) / "result.json"
        np.savez_compressed(temp_trace, train_paths=np.asarray(train_paths), test_paths=np.asarray(test_paths), train_W=np.asarray(train_w), test_W=np.asarray(test_w), train_phi=np.asarray(train_phi), test_phi=np.asarray(test_phi), train_seeds=np.asarray(train_seeds), test_seeds=np.asarray(test_seeds), train_directions=np.asarray([[np.eye(D)[i % 2] for i in range(TEST)] for _ in KINDS]), test_directions=np.asarray([[np.eye(D)[2] for _ in range(TEST)] for _ in KINDS]))
        trace_hash = sha256_file(temp_trace)
        payload = {"schema": "nrm-synthetic-v3", "status": "EXECUTED_UNVERIFIED", "trace_sha256": trace_hash, "frozen": {"D": D, "nodes": NODES, "train": TRAIN, "test": TEST, "trajectories": TRAJ, "steps": STEPS, "dt": DT, "train_directions": ["e1", "e2"], "test_direction": "e3", "intervention": "second_half", "g1_beta": 1.2, "g3_c": .22, "K_floor": K_FLOOR, "parameter_counts": {"metric": 5, "direct_vQ": 6, "gain_noise": 5, "noise_only": 5, "euclidean": 4, "flat_pullback": 5}}, "records": records, "evaluator": {"g1_recovery": recovery, "g1_pvalues": pvalues, "g1_holm": holm(pvalues), "false_positive_rows": false_rows, "g3_c_recovery": c_recovery, "curvature": curvature, "curvature_fixture": curvature_fixture(), "candidate_sign_fixture": candidate_sign_fixture(), "force_pushforward_fixture": force_pushforward_fixture()}}
        temp_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if args.overwrite:
            temp_trace.replace(trace_out); temp_json.replace(OUT)
        else:
            publish_exclusive(temp_trace, trace_out); publish_exclusive(temp_json, OUT)
    print(OUT)
if __name__ == "__main__": main()
