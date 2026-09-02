"""A1: independent recomputation of c_theta, the normalization convention, and its content.

Audit questions:
  (i)  Is theta = c_theta*eps/sqrt(1-eps^2) an identity for ARBITRARY Gram, or a physical law?
  (ii) Does c_theta = sqrt(3)/2 come from G_0 = 2 det(e) 1_3, or only from "3x3 + trace-normalized"?
  (iii) Does the card's Delta agree with the 14.2 implementation Delta = 0.25*sum(l^2-1)^2 ?
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.physics.causal_face_simplicity import (
    geometric_self_dual_triple, plebanski_gram, simplicity_residual,
)

out = {}
rng = np.random.default_rng(20260902)

# ---------- (i) identity theta = c_theta eps / sqrt(1-eps^2) for ARBITRARY symmetric G (dim 3)
def theta_eps(G):
    d = G.shape[0]
    tl = G - np.trace(G) / d * np.eye(d)
    q = np.linalg.norm(tl)
    return (d / 2.0) * q / np.trace(G), q / np.linalg.norm(G)

worst = 0.0
samples = []
for _ in range(2000):
    M = rng.standard_normal((3, 3))
    G = M @ M.T + 3.0 * np.eye(3)          # arbitrary SPD gram, tr>0, NOT near isotropic
    th, ep = theta_eps(G)
    pred = (math.sqrt(3) / 2) * ep / math.sqrt(1 - ep**2)
    worst = max(worst, abs(th - pred) / max(pred, 1e-300))
    samples.append(ep)
out["identity_arbitrary_gram_max_rel_err"] = worst
out["identity_arbitrary_gram_eps_range"] = [float(min(samples)), float(max(samples))]

# ---------- (ii) c_theta in dimension d: is sqrt(3)/2 special to Plebanski, or = sqrt(d)/2 ?
cth_by_dim = {}
for d in (2, 3, 4, 5, 6):
    ws = 0.0
    for _ in range(500):
        M = rng.standard_normal((d, d))
        G = M @ M.T + d * np.eye(d)
        tl = G - np.trace(G) / d * np.eye(d)
        q = np.linalg.norm(tl); th = (d / 2.0) * q / np.trace(G); ep = q / np.linalg.norm(G)
        pred = (math.sqrt(d) / 2) * ep / math.sqrt(1 - ep**2)
        ws = max(ws, abs(th - pred) / pred)
    cth_by_dim[d] = {"c_theta_sqrt_d_over_2": math.sqrt(d) / 2, "max_rel_err": ws}
out["c_theta_is_sqrt_d_over_2"] = cth_by_dim

# ---------- (ii-b) does G_0 = 2 det(e) 1_3 matter?  c_theta = 0.5 tr(c*1_3)/||c*1_3|| for ANY c>0
out["c_theta_from_any_isotropic_G0"] = {
    str(c): 0.5 * np.trace(c * np.eye(3)) / np.linalg.norm(c * np.eye(3))
    for c in (1e-6, 0.5, 2.0, 2 * 1.0, 2 * 7.3, 1e6)
}
# and the actual G_0 for a random nondegenerate tetrad (is it really 2 det(e) 1_3 ?)
g0s = {}
for name, e in (("identity", np.eye(4)),
                ("scaled_2.5", 2.5 * np.eye(4)),
                ("random", np.eye(4) + 0.3 * rng.standard_normal((4, 4)))):
    G0 = plebanski_gram(geometric_self_dual_triple(e))
    g0s[name] = {"G0": G0.tolist(), "2detE": 2 * float(np.linalg.det(e)),
                 "isotropic_dev": float(np.linalg.norm(G0 - np.trace(G0) / 3 * np.eye(3))),
                 "c_theta": 0.5 * float(np.trace(G0)) / float(np.linalg.norm(G0))}
out["G0_check"] = g0s

# ---------- (iii) card Delta (trace-normalized) vs 14.2 implementation Delta (absolute reference)
def delta_impl(l2):            # zerod_plebanski_closure._typed_history_member
    return 0.25 * sum((x - 1.0) ** 2 for x in l2)
def delta_card(l2):            # card: Ghat = 3 G / tr G, Delta = (1/4)||Ghat - 1||^2
    g = np.diag(np.asarray(l2, float))
    ghat = 3 * g / np.trace(g)
    return 0.25 * float(np.linalg.norm(ghat - np.eye(3))) ** 2

cases = {
    "pure_conformal_l=(1.2,1.2,1.2)": (1.44, 1.44, 1.44),
    "pure_conformal_l=(1.05,1.05,1.05)": (1.1025,) * 3,
    "pure_shear_trace3": (0.8, 1.0, 1.2),
    "generic": (0.9, 1.3, 1.1),
}
cmp = {}
for name, l2 in cases.items():
    g = np.diag(np.asarray(l2, float))
    tl = g - np.trace(g) / 3 * np.eye(3)
    eps = float(np.linalg.norm(tl) / np.linalg.norm(g))
    di, dc = delta_impl(l2), delta_card(l2)
    cmp[name] = {"eps_12_4": eps, "Delta_impl_14_2": di, "theta_impl": math.sqrt(di),
                 "Delta_card": dc, "theta_card": math.sqrt(dc),
                 "c_theta*eps/sqrt(1-eps^2)": (math.sqrt(3) / 2) * eps / math.sqrt(1 - eps**2)}
out["delta_convention_clash"] = cmp

# ---------- (iii-b) does the shipped 14.2 member reproduce theta = c_theta eps ?
try:
    from examples.physics.zerod_plebanski_closure import _typed_history_member
    rows = {}
    for tag, x in (("x=(0,0,0)", (0.0, 0.0, 0.0)),
                   ("x=(0.2,0.2,0.2) conformal", (0.2, 0.2, 0.2)),
                   ("x=(0.2,0,-0.2) shear-ish", (0.2, 0.0, -0.2)),
                   ("x=(0.1,-0.05,0.02)", (0.1, -0.05, 0.02))):
        member, _ = _typed_history_member("aud", ("t",), 0, x)
        l2 = member.squared_length_readout_over_planck_area
        g = np.diag(np.asarray(l2, float))
        tl = g - np.trace(g) / 3 * np.eye(3)
        eps = float(np.linalg.norm(tl) / np.linalg.norm(g))
        card_theta = (math.sqrt(3) / 2) * eps / math.sqrt(1 - eps**2)
        rows[tag] = {"impl_connection_angle": float(member.connection_angle),
                     "impl_defect": float(member.common_metric_defect),
                     "eps_12_4": eps, "card_theta": card_theta,
                     "impl/card": (float(member.connection_angle) / card_theta) if card_theta > 0 else None}
    out["shipped_14_2_member"] = rows
except Exception as exc:                                  # pragma: no cover
    out["shipped_14_2_member_error"] = repr(exc)

print(json.dumps(out, indent=2, ensure_ascii=False))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
