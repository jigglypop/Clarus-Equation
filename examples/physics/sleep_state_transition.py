"""Stage-1 sleep forcing pilot from q_sleep to p_r.

This script uses the q_sleep burden estimated in sleep_q_pilot.py and applies
the minimal sleep forcing

    p' = Pi_Delta((1-rho)p* + rho p + z h_sleep)

at p = p*. At the fixed point the first two terms reduce to p*, so the pilot
measures how large a local sensitivity alpha_r must be before q_sleep produces
an observable x_a / x_b shift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


P_STAR = np.array([0.0487, 0.2623, 0.6891], dtype=np.float64)
P_STAR = P_STAR / P_STAR.sum()
Q_RESULT_PATH = Path(__file__).with_name("sleep_q_pilot_results.json")


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    if v.ndim != 1:
        raise ValueError("Expected a vector.")
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, v.size + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        raise ValueError("Simplex projection failed.")
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0)


def transition(z_sleep: float, alpha_r: float) -> dict[str, float | list[float]]:
    h_sleep = np.array([alpha_r, 0.0, -alpha_r], dtype=np.float64)
    p_next = project_simplex(P_STAR + z_sleep * h_sleep)
    delta = p_next - P_STAR
    return {
        "alpha_r": alpha_r,
        "p_next": [float(x) for x in p_next],
        "delta_x_a": float(delta[0]),
        "delta_x_s": float(delta[1]),
        "delta_x_b": float(delta[2]),
        "l2_shift": float(np.linalg.norm(delta)),
    }


def main() -> None:
    q_payload = json.loads(Q_RESULT_PATH.read_text(encoding="utf-8"))
    z_mean = float(q_payload["q_sleep"]["z_mean"])
    z_geom = float(q_payload["q_sleep"]["z_geometric_mean"])
    alpha_grid = [0.05, 0.10, 0.15, 0.20, 0.25]

    payload = {
        "p_star": [float(x) for x in P_STAR],
        "z_sleep_inputs": {
            "z_mean": z_mean,
            "z_geometric_mean": z_geom,
        },
        "model": "p_next = project_simplex(p_star + z_sleep * (alpha_r, 0, -alpha_r))",
        "sweeps": {
            "z_mean": [transition(z_mean, alpha_r) for alpha_r in alpha_grid],
            "z_geometric_mean": [transition(z_geom, alpha_r) for alpha_r in alpha_grid],
        },
    }

    out_path = Path(__file__).with_name("sleep_state_transition_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Stage-1 sleep forcing pilot")
    print(f"  z_mean = {z_mean:.6f}")
    print(f"  z_geometric_mean = {z_geom:.6f}")
    print("  alpha_r  delta_x_a(mean)  delta_x_b(mean)  delta_x_a(geom)  delta_x_b(geom)")
    for alpha_r, mean_row, geom_row in zip(
        alpha_grid,
        payload["sweeps"]["z_mean"],
        payload["sweeps"]["z_geometric_mean"],
    ):
        print(
            f"  {alpha_r:7.2f}"
            f"  {mean_row['delta_x_a']:15.6f}"
            f"  {mean_row['delta_x_b']:15.6f}"
            f"  {geom_row['delta_x_a']:15.6f}"
            f"  {geom_row['delta_x_b']:15.6f}"
        )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
