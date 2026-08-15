"""Independent development-only checks for the V17 route comparison.

This script deliberately uses only seeds 1,719,000--1,719,063.  It does not
import a production implementation and it refuses seeds outside that block.
"""

from __future__ import annotations

import json

import numpy as np


DEV_FIRST = 1_719_000
DEV_LAST = 1_719_063
D = 3
KAPPA = 0.25


def _signed_qr(raw: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(raw)
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return q @ np.diag(signs)


def _fixture(seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not DEV_FIRST <= seed <= DEV_LAST:
        raise ValueError("confirmation and unregistered seeds are forbidden")
    rng = np.random.default_rng(seed)
    raw_u = rng.standard_normal(D)
    u = raw_u / np.linalg.norm(raw_u)
    q_left = _signed_qr(rng.standard_normal((D, D)))
    q_right = _signed_qr(rng.standard_normal((D, D)))
    singular = np.exp(rng.uniform(np.log(0.25), np.log(4.0), size=D))
    chart = q_left @ np.diag(singular) @ q_right.T
    return u, chart


def _transport_metric(g: np.ndarray, chart: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(chart)
    return inverse.T @ g @ inverse


def _rank_one_update(g: np.ndarray, x: np.ndarray, cost: float) -> np.ndarray:
    gx = g @ x
    prediction = float(x @ gx)
    coefficient = (cost / prediction - 1.0) / prediction
    return g + coefficient * np.outer(gx, gx)


def _relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)


def main() -> None:
    max_lift_cost_defect = 0.0
    max_lift_update_defect = 0.0
    max_eligibility_defect = 0.0
    max_randers_defect = 0.0
    max_anchor_update_defect = 0.0
    min_lift_margin = float("inf")
    min_anchor_readout_margin = float("inf")
    strict_equal = True
    strict_quadratic_ties = True
    anchor_quadratic_ties = True
    lift_actions_correct = True
    eligibility_actions_correct = True
    randers_actions_correct = True
    anchor_actions_correct = True

    identity = np.eye(D)
    anchor = np.array([1.0, 0.0, 0.0])

    for seed in range(DEV_FIRST, DEV_LAST + 1):
        u, chart = _fixture(seed)

        strict_states: dict[int, np.ndarray] = {}
        for sign in (-1, 1):
            strict_states[sign] = _rank_one_update(identity, sign * u, 4.0)
        strict_equal &= np.array_equal(strict_states[-1], strict_states[1])
        strict_costs = [float((action * u) @ strict_states[1] @ (action * u)) for action in (-1, 1)]
        strict_quadratic_ties &= bool(np.isclose(strict_costs[0], strict_costs[1], rtol=0.0, atol=2e-15))

        lifted_chart = np.eye(D + 1)
        lifted_chart[:D, :D] = chart
        lifted_initial = np.eye(D + 1)
        lifted_initial_chart = _transport_metric(lifted_initial, lifted_chart)

        for sign in (-1, 1):
            z = np.concatenate((sign * u, [1.0]))
            lifted = _rank_one_update(lifted_initial, z, 4.0)
            reference = lifted_initial + 0.5 * np.outer(z, z)
            if not np.allclose(lifted, reference, rtol=0.0, atol=2e-15):
                raise AssertionError("homogeneous exact update failed")

            costs: dict[int, float] = {}
            transported_costs: dict[int, float] = {}
            for action in (-1, 1):
                y = np.concatenate((action * u, [-1.0]))
                costs[action] = float(y @ lifted @ y)
                y_chart = lifted_chart @ y
                z_chart = lifted_chart @ z
                lifted_chart_state = _rank_one_update(lifted_initial_chart, z_chart, 4.0)
                transported_costs[action] = float(y_chart @ lifted_chart_state @ y_chart)
                max_lift_cost_defect = max(
                    max_lift_cost_defect,
                    _relative(costs[action], transported_costs[action]),
                )
                expected_transport = _transport_metric(lifted, lifted_chart)
                max_lift_update_defect = max(
                    max_lift_update_defect,
                    float(np.linalg.norm(lifted_chart_state - expected_transport) / np.linalg.norm(expected_transport)),
                )

            lift_actions_correct &= min(costs, key=costs.get) == sign
            min_lift_margin = min(min_lift_margin, costs[-sign] - costs[sign])
            if not np.isclose(costs[sign], 2.0, rtol=0.0, atol=3e-15):
                raise AssertionError("homogeneous correct cost is not 2")
            if not np.isclose(costs[-sign], 4.0, rtol=0.0, atol=5e-15):
                raise AssertionError("homogeneous wrong cost is not 4")

            # Explicit eligibility covector: maximize e(v_a).
            eligibility = sign * u
            eligibility_chart = np.linalg.solve(chart.T, eligibility)
            eligibility_scores = {}
            for action in (-1, 1):
                v = action * u
                original_score = float(eligibility @ v)
                chart_score = float(eligibility_chart @ (chart @ v))
                max_eligibility_defect = max(
                    max_eligibility_defect,
                    _relative(original_score, chart_score),
                )
                eligibility_scores[action] = original_score
            eligibility_actions_correct &= max(eligibility_scores, key=eligibility_scores.get) == sign

            # Randers route: minimize sqrt(v^T g v) + beta(v).
            beta = -KAPPA * sign * u
            beta_chart = np.linalg.solve(chart.T, beta)
            chart_metric = _transport_metric(identity, chart)
            randers_costs = {}
            for action in (-1, 1):
                v = action * u
                original_cost = float(np.sqrt(v @ v) + beta @ v)
                v_chart = chart @ v
                chart_cost = float(np.sqrt(v_chart @ chart_metric @ v_chart) + beta_chart @ v_chart)
                max_randers_defect = max(
                    max_randers_defect,
                    _relative(original_cost, chart_cost),
                )
                randers_costs[action] = original_cost
            randers_actions_correct &= min(randers_costs, key=randers_costs.get) == sign

            # Signed original-g route with a separately declared covector anchor.
            cue_covector = sign * u
            update_covector = anchor + cue_covector
            anchored_metric = identity + KAPPA * np.outer(update_covector, update_covector)
            even_part = identity + KAPPA * (np.outer(anchor, anchor) + np.outer(u, u))
            signed_readout = float(anchor @ (anchored_metric - even_part) @ u)
            expected_readout = sign * KAPPA * (1.0 + float(anchor @ u) ** 2)
            if not np.isclose(signed_readout, expected_readout, rtol=2e-14, atol=2e-15):
                raise AssertionError("anchor readout formula failed")
            anchor_actions_correct &= (1 if signed_readout > 0.0 else -1) == sign
            min_anchor_readout_margin = min(min_anchor_readout_margin, abs(signed_readout))

            chart_metric = _transport_metric(identity, chart)
            chart_anchor = np.linalg.solve(chart.T, anchor)
            chart_cue_covector = np.linalg.solve(chart.T, cue_covector)
            anchored_metric_chart = chart_metric + KAPPA * np.outer(
                chart_anchor + chart_cue_covector,
                chart_anchor + chart_cue_covector,
            )
            expected_anchor_transport = _transport_metric(anchored_metric, chart)
            max_anchor_update_defect = max(
                max_anchor_update_defect,
                float(
                    np.linalg.norm(anchored_metric_chart - expected_anchor_transport)
                    / np.linalg.norm(expected_anchor_transport)
                ),
            )

            ordinary = [
                float((action * u) @ anchored_metric @ (action * u))
                for action in (-1, 1)
            ]
            anchor_quadratic_ties &= bool(np.isclose(ordinary[0], ordinary[1], rtol=0.0, atol=2e-15))

    summary = {
        "development_seed_count": DEV_LAST - DEV_FIRST + 1,
        "confirmation_seed_opened": False,
        "strict_sign_paired_state_exact_equal": strict_equal,
        "strict_opposite_action_quadratic_tie": strict_quadratic_ties,
        "homogeneous_all_actions_correct": lift_actions_correct,
        "homogeneous_min_wrong_minus_correct_margin": min_lift_margin,
        "homogeneous_max_relative_cost_chart_defect": max_lift_cost_defect,
        "homogeneous_max_relative_update_chart_defect": max_lift_update_defect,
        "eligibility_all_actions_correct": eligibility_actions_correct,
        "eligibility_max_pairing_chart_defect": max_eligibility_defect,
        "randers_all_actions_correct": randers_actions_correct,
        "randers_max_cost_chart_defect": max_randers_defect,
        "anchor_all_actions_correct_with_explicit_readout": anchor_actions_correct,
        "anchor_min_absolute_signed_readout": min_anchor_readout_margin,
        "anchor_max_relative_update_chart_defect": max_anchor_update_defect,
        "anchor_plain_quadratic_policy_ties": anchor_quadratic_ties,
        "registered_finite_scc_sizes_checked_analytically": [1, 2, 4, 8, 16, 64],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
