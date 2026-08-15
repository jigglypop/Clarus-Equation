"""Development-only scratch comparison for the preregistered V16 update."""

from __future__ import annotations

import math

import numpy as np


RATES = (0.05, 0.1, 0.2, 0.4, 1.0)
DEVELOPMENT_SEEDS = range(917_000, 917_064)


def hidden_metric(rng: np.random.Generator) -> np.ndarray:
    basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    eigenvalues = np.exp(rng.uniform(math.log(0.25), math.log(4.0), 3))
    return basis @ np.diag(eigenvalues) @ basis.T


def unit_vectors(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.normal(size=shape)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def project_spd(metric: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (metric + metric.T))
    clipped = np.clip(eigenvalues, 1.0e-6, 1.0e6)
    return (eigenvectors * clipped) @ eigenvectors.T


def tied_argmin(values: np.ndarray) -> int:
    minimum = float(np.min(values))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(values))),
    )
    return int(np.flatnonzero(values <= minimum + tolerance)[0])


def score_seed(seed: int, learner: str, rate: float) -> tuple[float, ...]:
    rng = np.random.default_rng(seed)
    target = hidden_metric(rng)
    candidates = unit_vectors(rng, (128, 4, 3))
    noise = rng.normal(size=128)
    routes = unit_vectors(rng, (64, 4, 2, 3))
    metric = np.eye(3)
    conformal_scale = 1.0
    online_regret: list[float] = []

    for step, options in enumerate(candidates):
        if learner == "conformal":
            predicted = conformal_scale * np.einsum("ki,ki->k", options, options)
        else:
            predicted = np.einsum("ki,ij,kj->k", options, metric, options)
        action = (step // 4) % 4 if step % 4 == 0 else tied_argmin(predicted)
        vector = options[action]
        true_costs = np.einsum("ki,ij,kj->k", options, target, options)
        online_regret.append(
            float((true_costs[action] - np.min(true_costs)) / np.min(true_costs))
        )
        observed_cost = float(true_costs[action] * math.exp(0.05 * noise[step]))

        if learner == "v16":
            prediction = float(vector @ metric @ vector)
            residual = math.log(prediction / observed_cost)
            coefficient = math.expm1(-rate * residual)
            covector = metric @ vector
            metric += (coefficient / prediction) * np.outer(covector, covector)
            metric = 0.5 * (metric + metric.T)
        elif learner == "additive":
            prediction = float(vector @ metric @ vector)
            residual = math.log(prediction / observed_cost)
            metric = project_spd(
                metric - rate * (residual / prediction) * np.outer(vector, vector)
            )
        elif learner == "conformal":
            prediction = conformal_scale * float(vector @ vector)
            residual = math.log(prediction / observed_cost)
            conformal_scale *= math.exp(-rate * residual)
        elif learner == "identity":
            pass
        else:  # pragma: no cover - development script invariant
            raise ValueError(learner)

    if learner == "conformal":
        metric = conformal_scale * np.eye(3)
    true_route_cost = np.einsum("...i,ij,...j->...", routes, target, routes).sum(axis=2)
    predicted_route_cost = np.einsum(
        "...i,ij,...j->...", routes, metric, routes
    ).sum(axis=2)
    choice = np.asarray([tied_argmin(row) for row in predicted_route_cost])
    optimum = np.asarray([tied_argmin(row) for row in true_route_cost])
    selected_cost = true_route_cost[np.arange(64), choice]
    best_cost = np.min(true_route_cost, axis=1)
    accuracy = float(np.mean(choice == optimum))
    regret = float(np.mean((selected_cost - best_cost) / best_cost))
    target_eigenvalues, target_eigenvectors = np.linalg.eigh(target)
    inverse_sqrt = (
        target_eigenvectors * target_eigenvalues ** -0.5
    ) @ target_eigenvectors.T
    generalized = np.linalg.eigvalsh(inverse_sqrt @ metric @ inverse_sqrt)
    metric_error = float(np.sqrt(np.mean(np.log(generalized) ** 2)))
    return accuracy, regret, metric_error, float(np.mean(online_regret[32:]))


def main() -> None:
    for learner in ("v16", "additive", "conformal"):
        for rate in RATES:
            scores = np.asarray(
                [score_seed(seed, learner, rate) for seed in DEVELOPMENT_SEEDS]
            )
            print(
                learner,
                rate,
                "accuracy",
                float(np.mean(scores[:, 0])),
                "regret",
                float(np.mean(scores[:, 1])),
                "median_metric_error",
                float(np.median(scores[:, 2])),
                "online_regret_after_32",
                float(np.mean(scores[:, 3])),
            )
    identity = np.asarray(
        [score_seed(seed, "identity", 0.0) for seed in DEVELOPMENT_SEEDS]
    )
    print(
        "identity",
        "accuracy",
        float(np.mean(identity[:, 0])),
        "regret",
        float(np.mean(identity[:, 1])),
        "median_metric_error",
        float(np.median(identity[:, 2])),
        "online_regret_after_32",
        float(np.mean(identity[:, 3])),
    )


if __name__ == "__main__":
    main()
