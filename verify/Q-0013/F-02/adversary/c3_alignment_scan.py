"""adversary c3: how much alignment freedom does the new step-4 sentence tolerate?
tl gram(Y) = -n delta^2 sum_v tl M(xi~_v,xi~_v) + O(delta^3) claimed for "an arbitrary per-cell
alignment map". Scan (a) delta at polar alignment, (b) an extra per-cell rotation of size s*delta.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(r"C:/dev/ce/Clarus-Equation")
OUT = ROOT / "verify" / "Q-0013" / "F-02" / "adversary"
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram
from examples.physics.urbantke_shape_matching_rg import optimal_internal_alignment

spec = importlib.util.spec_from_file_location("cf", ROOT / "verify/Q-0013/F-02/check_floor.py")
cf = importlib.util.module_from_spec(spec); sys.modules["cf"] = cf; spec.loader.exec_module(cf)
Mt = np.load(OUT / "b1_Mt.npy")
REF = geometric_self_dual_triple(np.eye(4))


def tl(m):
    return m - np.trace(m) / 3.0 * np.eye(3)


def rot(A):
    w, V = np.linalg.eigh(1j * A)
    return (V @ np.diag(np.exp(-1j * w)) @ V.conj().T).real


rng = np.random.default_rng(20260903)
n = 5
A16 = cf.factor(cf.SPECS["univ_o"])
g = rng.normal(size=(n, A16.shape[1]))
labels = (g @ A16.T).reshape(n, 4, 4)
xi = labels.reshape(n, 16)
xit = xi - xi.mean(axis=0, keepdims=True)
core = -np.einsum("va,vb,abij->ij", xit, xit, Mt)
# fixed random antisymmetric generators, unit norm, reused across deltas/scales
gens = []
for _ in range(n):
    a = rng.normal(size=(3, 3)); a = a - a.T
    gens.append(a / np.linalg.norm(a))

rep = {"n": n, "sigma": "univ_o", "note": "rel_err = ||tl gram(Y) - (-n d^2 sum tl M)|| / ||rhs||"}
polar = {}
for delta in (0.05, 0.02, 0.005, 0.001):
    tris = [optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + delta * l)).aligned_candidate
            for l in labels]
    Y = sum(tris)
    lhs, rhs = tl(plebanski_gram(Y)), n * delta**2 * core
    polar[str(delta)] = {"rel_err": float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)),
                         "rel_err_over_delta": float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs) / delta)}
rep["polar_alignment_vs_delta"] = polar

scan = {}
delta = 0.005
tris = [optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + delta * l)).aligned_candidate
        for l in labels]
rhs = n * delta**2 * core
for s in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0):
    Y = sum(rot(s * delta * gens[v]) @ tris[v] for v in range(n))
    lhs = tl(plebanski_gram(Y))
    scan[f"extra_rot_{s}xdelta"] = {
        "rel_err": float(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)),
        "abs_dev_over_delta2": float(np.linalg.norm(lhs - rhs) / delta**2)}
rep["extra_rotation_scan_at_delta0.005"] = scan
# same scan at another delta -> is the contamination O(delta^2) (same order as signal)?
scan2 = {}
delta2 = 0.02
tris2 = [optimal_internal_alignment(REF, geometric_self_dual_triple(np.eye(4) + delta2 * l)).aligned_candidate
         for l in labels]
rhs2 = n * delta2**2 * core
for s in (0.0, 1.0, 3.0):
    Y = sum(rot(s * delta2 * gens[v]) @ tris2[v] for v in range(n))
    lhs = tl(plebanski_gram(Y))
    scan2[f"extra_rot_{s}xdelta"] = {"rel_err": float(np.linalg.norm(lhs - rhs2) / np.linalg.norm(rhs2))}
rep["extra_rotation_scan_at_delta0.02"] = scan2
# unaligned (R=I) for reference
Yn = sum(geometric_self_dual_triple(np.eye(4) + delta * l) for l in labels)
rep["no_alignment_rel_err_delta0.005"] = float(np.linalg.norm(tl(plebanski_gram(Yn)) - rhs) / np.linalg.norm(rhs))
(OUT / "c3_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rep, ensure_ascii=False, indent=2))
