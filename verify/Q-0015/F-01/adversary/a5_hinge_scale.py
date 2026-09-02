"""A5: K5 (scale) and K6 (hinge) -- can they fail for any reason other than a coding bug?
Plus: does the card's theta have ANY overlap with the one exactly-known curvature angle
in the corpus (15.4 phi_kappa)?"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "verify" / "Q-0015" / "F-01"))
sys.path.insert(0, str(ROOT / "verify" / "Q-0008" / "F-02"))
import check_theta as C  # noqa: E402
from examples.physics.causal_face_simplicity import geometric_self_dual_triple, plebanski_gram  # noqa: E402
from examples.physics.curved_plebanski_hinge import (  # noqa: E402
    exact_primal_triangle_holonomy, de_sitter_plebanski_point_audit, reference_vertex_coordinates,
)

out = {}

# ---- K5: is c_theta(alpha) invariance a measurement or an algebraic tautology?
rows = {}
base = np.eye(4) + 0.37 * np.random.default_rng(7).standard_normal((4, 4))
for a in (1e-3, 0.4, 1.0, 2.5, 1e3):
    G = plebanski_gram(geometric_self_dual_triple(a * np.eye(4)))
    Gr = plebanski_gram(geometric_self_dual_triple(a * base))
    rows[str(a)] = {"c_theta_identity_tetrad": 0.5 * float(np.trace(G)) / float(np.linalg.norm(G)),
                    "c_theta_random_tetrad": 0.5 * float(np.trace(Gr)) / float(np.linalg.norm(Gr)),
                    "G_scales_as_alpha^4": float(np.linalg.norm(Gr) / (a ** 4 * np.linalg.norm(
                        plebanski_gram(geometric_self_dual_triple(base)))))}
out["K5_scale"] = rows
out["K5_note"] = ("theta = (3/2)||tlG||/trG and eps = ||tlG||/||G|| are EACH invariant under "
                  "G -> lambda G, so their ratio cannot depend on alpha; and G(alpha e) = alpha^4 G(e) "
                  "exactly.  K5 has no failure mode other than a coding bug.")
# K5 on a random ANISOTROPIC gram (the check the card does NOT do)
rng = np.random.default_rng(11)
M = rng.standard_normal((3, 3)); Gani = M @ M.T + 2 * np.eye(3)
out["K5_anisotropic_gram_ratio_is_still_alpha_free"] = {
    str(a): 0.5 * np.trace(a * Gani) / np.linalg.norm(a * Gani) for a in (0.4, 1.0, 2.5)}

# ---- K6: hinge
worst = 0.0
per_vertex = {}
for name, coord in reference_vertex_coordinates().items():
    aud = de_sitter_plebanski_point_audit(tuple(float(v) for v in coord),
                                          curvature_times_reference_length_squared=1.0)
    per_vertex[str(name)] = float(aud.simplicity_tracefree_residual)
    worst = max(worst, float(aud.simplicity_tracefree_residual))
prim = exact_primal_triangle_holonomy(curvature_times_reference_length_squared=1.0)
out["K6_hinge"] = {"per_vertex_simplicity_tracefree_residual": per_vertex,
                   "worst": worst, "c_theta_times_worst": math.sqrt(3) / 2 * worst,
                   "K6_window_upper": 1.0e-12,
                   "margin_orders_of_magnitude": math.log10(1e-12 / max(worst * math.sqrt(3) / 2, 1e-300)),
                   "phi_kappa_exact_primal_holonomy": float(prim.rotation_angle)}
out["K6_note"] = ("15.5 already reports these residuals at ~1e-16 (exact solution, machine zero). "
                  "K6 multiplies an ALREADY PUBLISHED machine-zero by 0.866 and asks whether it is "
                  "below 1e-12.  It has no failure mode short of a coding bug.")

# ---- two-channel claim: does theta_tot^2 = theta_iso^2 + theta_mis^2 have a computed theta_iso?
# curvature at the hinge is entirely isotropic (Psi=0, F = kappa Sigma), phi_kappa = 0.429.
# the card's theta on the same configuration:
out["two_channel"] = {
    "phi_kappa (exact transport angle, 15.4)": float(prim.rotation_angle),
    "card_theta_mis (from gram residual)": math.sqrt(3) / 2 * worst,
    "ratio_card_theta_over_phi_kappa": (math.sqrt(3) / 2 * worst) / float(prim.rotation_angle),
    "theta_iso_computed_anywhere": False,
    "note": ("theta_tot^2 = theta_iso^2 + theta_mis^2 is asserted from Frobenius orthogonality of "
             "Psi and (lambda/3)delta in F = (Psi + lambda/3 delta) Sigma.  That orthogonality is a "
             "statement about the CURVATURE 2-FORM components; theta_mis here is defined from a GRAM "
             "mismatch of a block SUM of Sigma's, with no computed map to either F component.  "
             "No theta_iso is ever constructed, so the identity is never evaluated with two nonzero "
             "terms.  In the single corpus configuration where an exact curvature angle exists "
             "(phi_kappa = 0.429), the card's theta is 1e-16 -- i.e. zero overlap.")}

# ---- can theta ever be large enough for sqrt(Delta) to be an angle at all?  (parked item)
out["small_angle_regime"] = {
    "eps_rms_n128_measured": 2.2e-4, "eps_max_n128_measured": 7.4e-4,
    "nonlinear_correction_1/sqrt(1-eps^2)-1_at_n128": 1 / math.sqrt(1 - 7.4e-4 ** 2) - 1,
    "note": ("the 'exact' 1/sqrt(1-eps^2) factor that distinguishes the card from the naive "
             "theta = c_theta*eps is 2.7e-7 at the largest eps on the frozen grid, i.e. ~6 orders "
             "below the tightest window (K1 half-width 0.01).  The exactness is untested.")}

print(json.dumps(out, indent=2, ensure_ascii=False))
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
