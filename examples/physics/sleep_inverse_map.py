"""Inverse map from observed sleep-deprivation state shifts to alpha_r and gamma.

The forward graph model is

    e = ((1 - rho_B) I + gamma L_G)^-1 z_sleep alpha.

Given an observed regional active-state shift e, the inverse at a fixed gamma is

    alpha_hat(gamma) = ((1 - rho_B) I + gamma L_G) e / z_sleep.

Without repeated observations gamma is not uniquely determined. This pilot shows
how to constrain gamma by stability, positivity, plausible alpha bounds, and a
weak anatomical prior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


GRAPH_RESULT_PATH = Path(__file__).with_name("sleep_graph_spread_results.json")
Q_RESULT_PATH = Path(__file__).with_name("sleep_q_pilot_results.json")
RHO_B = 0.155
ALPHA_MIN = 0.0
ALPHA_MAX = 0.25


def adjacency(nodes: list[str], edges: dict[str, float]) -> np.ndarray:
    index = {node: i for i, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for key, weight in edges.items():
        left, right = key.split("-")
        i, j = index[left], index[right]
        matrix[i, j] = float(weight)
        matrix[j, i] = float(weight)
    return matrix


def laplacian(matrix: np.ndarray) -> np.ndarray:
    return np.diag(matrix.sum(axis=1)) - matrix


def infer_alpha(observed_e: np.ndarray, z_sleep: float, gamma: float, lap: np.ndarray) -> np.ndarray:
    system = (1.0 - RHO_B) * np.eye(observed_e.size) + gamma * lap
    return system @ observed_e / z_sleep


def smoothness(alpha: np.ndarray, lap: np.ndarray) -> float:
    return float(alpha @ lap @ alpha)


def main() -> None:
    graph_payload = json.loads(GRAPH_RESULT_PATH.read_text(encoding="utf-8"))
    q_payload = json.loads(Q_RESULT_PATH.read_text(encoding="utf-8"))
    nodes = list(graph_payload["nodes"].keys())
    z_sleep = float(q_payload["q_sleep"]["z_mean"])
    edges = {str(key): float(value) for key, value in graph_payload["edges"].items()}
    alpha_prior = np.asarray([graph_payload["alpha_prior"][node] for node in nodes], dtype=np.float64)
    lap = laplacian(adjacency(nodes, edges))
    eigvals = np.linalg.eigvalsh(lap)
    gamma_upper = float((1.0 + RHO_B) / eigvals.max())

    # In lieu of raw fMRI observations, use the gamma=0.20 forward pilot response
    # as a synthetic observed regional map. Raw data will replace this vector.
    observed_map = graph_payload["responses"]["0.20"]["delta_x_a_by_node"]
    observed_e = np.asarray([observed_map[node] for node in nodes], dtype=np.float64)

    gamma_grid = np.linspace(0.0, min(0.38, gamma_upper * 0.98), 77)
    candidates = []
    for gamma in gamma_grid:
        alpha_hat = infer_alpha(observed_e, z_sleep, float(gamma), lap)
        stable = bool(max(abs(RHO_B - gamma * eigvals)) < 1.0)
        plausible = bool(np.all(alpha_hat >= ALPHA_MIN) and np.all(alpha_hat <= ALPHA_MAX))
        prior_rmse = float(np.sqrt(np.mean((alpha_hat - alpha_prior) ** 2)))
        candidates.append(
            {
                "gamma": float(gamma),
                "stable": stable,
                "plausible_alpha": plausible,
                "alpha_hat": {node: float(alpha_hat[i]) for i, node in enumerate(nodes)},
                "alpha_min": float(alpha_hat.min()),
                "alpha_max": float(alpha_hat.max()),
                "alpha_smoothness": smoothness(alpha_hat, lap),
                "prior_rmse": prior_rmse,
            }
        )

    feasible = [row for row in candidates if row["stable"] and row["plausible_alpha"]]
    best_prior = min(feasible, key=lambda row: row["prior_rmse"])
    best_smooth = min(feasible, key=lambda row: row["alpha_smoothness"])

    out = {
        "model": "alpha_hat(gamma) = ((1-rho_B)I + gamma L_G) observed_e / z_sleep",
        "nodes": nodes,
        "z_sleep": z_sleep,
        "rho_b": RHO_B,
        "gamma_stable_upper": gamma_upper,
        "observed_delta_x_a": {node: float(observed_e[i]) for i, node in enumerate(nodes)},
        "observed_source": "synthetic gamma=0.20 response from sleep_graph_spread_results.json",
        "alpha_prior": {node: float(alpha_prior[i]) for i, node in enumerate(nodes)},
        "feasible_gamma_min": float(min(row["gamma"] for row in feasible)),
        "feasible_gamma_max": float(max(row["gamma"] for row in feasible)),
        "best_by_prior": best_prior,
        "best_by_smoothness": best_smooth,
        "candidates": candidates,
        "next_real_data_step": "Replace observed_delta_x_a with measured regional SD-NS active-state shifts.",
    }

    out_path = Path(__file__).with_name("sleep_inverse_map_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Sleep inverse map pilot")
    print(f"  z_sleep = {z_sleep:.6f}")
    print(f"  gamma stable upper = {gamma_upper:.6f}")
    print(f"  feasible gamma range = {out['feasible_gamma_min']:.4f} .. {out['feasible_gamma_max']:.4f}")
    print(f"  best gamma by prior = {best_prior['gamma']:.4f}, rmse={best_prior['prior_rmse']:.6f}")
    print(
        "  best gamma by smoothness = "
        f"{best_smooth['gamma']:.4f}, smoothness={best_smooth['alpha_smoothness']:.6f}"
    )
    print("  alpha_hat at best-prior gamma:")
    for node, value in best_prior["alpha_hat"].items():
        print(f"    {node}: {value:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
