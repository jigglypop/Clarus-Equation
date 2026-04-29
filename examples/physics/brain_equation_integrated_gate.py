"""Integrated prediction gate for the global brain-state equation.

This pilot mirrors the docs/6_뇌 global equation:

    P_{n+1} = Pi((1-rho_B) P* + rho_B P_n
                 + gamma Delta_G P_n
                 + U_task + H(Q-Q*)
                 + F_syn + F_slow)

It compares the full equation against ablations. The point is not that these
synthetic numbers are final evidence, but that every retained term has a
concrete holdout gate and a path to real transition data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np


REGIONS = [
    "cortex",
    "thalamus",
    "hippocampus",
    "salience",
    "hypothalamus",
    "brainstem",
    "autonomic",
]
COEFFICIENT_NAMES = [
    "rho_b",
    "gamma",
    "task_scale",
    "homeostasis_scale",
    "syn_scale",
    "slow_scale",
]
COEFFICIENT_AXES = {
    "rho_b": "recovery_depth",
    "gamma": "relay_graph_axis",
    "task_scale": "task_recruitment",
    "homeostasis_scale": "homeostatic_depth",
    "syn_scale": "plasticity_axis",
    "slow_scale": "sleep_recovery_axis",
}
FAMILY_COEFFICIENT_COUNT = len(REGIONS) * len(COEFFICIENT_NAMES)
GENERATOR_COEFFICIENT_COUNT = len(COEFFICIENT_NAMES) * 2
MAX_REAL_CONDITION_NUMBER = 1.0e8

EDGES = [
    ("cortex", "thalamus", 0.90),
    ("cortex", "hippocampus", 0.55),
    ("cortex", "salience", 0.45),
    ("thalamus", "salience", 0.40),
    ("hippocampus", "salience", 0.35),
    ("hypothalamus", "brainstem", 0.85),
    ("brainstem", "autonomic", 0.75),
    ("hypothalamus", "salience", 0.45),
    ("hypothalamus", "thalamus", 0.30),
    ("brainstem", "thalamus", 0.30),
]

P_STAR = {
    "cortex": np.asarray([0.30, 0.24, 0.46], dtype=np.float64),
    "thalamus": np.asarray([0.28, 0.25, 0.47], dtype=np.float64),
    "hippocampus": np.asarray([0.29, 0.28, 0.43], dtype=np.float64),
    "salience": np.asarray([0.31, 0.25, 0.44], dtype=np.float64),
    "hypothalamus": np.asarray([0.26, 0.24, 0.50], dtype=np.float64),
    "brainstem": np.asarray([0.25, 0.23, 0.52], dtype=np.float64),
    "autonomic": np.asarray([0.23, 0.23, 0.54], dtype=np.float64),
}
RHO_B = 0.155
GAMMA = 0.12
TASK_LOAD = 1.0
D_SLEEP = 24.0

Q_DELTA = {
    "sleep": 0.3445,
    "arousal": 0.1800,
    "metabolic": 0.1200,
}

D_SENSITIVITY = {
    "cortex": {"sleep": 0.1040, "arousal": 0.1497, "metabolic": 0.0766},
    "thalamus": {"sleep": 0.1510, "arousal": 0.1280, "metabolic": 0.0810},
    "hippocampus": {"sleep": 0.1273, "arousal": 0.1193, "metabolic": 0.0875},
    "salience": {"sleep": 0.1180, "arousal": 0.1510, "metabolic": 0.0900},
    "hypothalamus": {"sleep": 0.1865, "arousal": 0.1390, "metabolic": 0.0965},
    "brainstem": {"sleep": 0.1369, "arousal": 0.1596, "metabolic": 0.1051},
    "autonomic": {"sleep": 0.1100, "arousal": 0.1410, "metabolic": 0.1240},
}

TASK_BETA = {
    "cortex": 0.42,
    "thalamus": 0.22,
    "hippocampus": 0.18,
    "salience": 0.26,
    "hypothalamus": 0.04,
    "brainstem": 0.03,
    "autonomic": 0.02,
}

SYN_LOAD = {
    "cortex": 0.050946,
    "thalamus": 0.031423,
    "hippocampus": 0.065743,
    "salience": 0.040000,
    "hypothalamus": 0.018000,
    "brainstem": 0.015000,
    "autonomic": 0.012000,
}

SYN_ZETA = {
    "cortex": 0.10,
    "thalamus": 0.06,
    "hippocampus": 0.14,
    "salience": 0.09,
    "hypothalamus": 0.04,
    "brainstem": 0.03,
    "autonomic": 0.03,
}

SYN_NU = {
    "cortex": 0.10,
    "thalamus": 0.04,
    "hippocampus": 0.16,
    "salience": 0.08,
    "hypothalamus": 0.02,
    "brainstem": 0.02,
    "autonomic": 0.01,
}

SLOW_OMEGA = {
    "cortex": (0.05, 0.06),
    "thalamus": (0.04, 0.05),
    "hippocampus": (0.04, 0.08),
    "salience": (0.05, 0.05),
    "hypothalamus": (0.06, 0.05),
    "brainstem": (0.06, 0.04),
    "autonomic": (0.04, 0.04),
}

ANATOMICAL_AXES = {
    "cortex": {
        "recovery_depth": 1.00,
        "relay_graph_axis": 0.73,
        "task_recruitment": 1.00,
        "homeostatic_depth": 0.00,
        "plasticity_axis": 0.40,
        "sleep_recovery_axis": 0.00,
    },
    "thalamus": {
        "recovery_depth": 0.60,
        "relay_graph_axis": 1.00,
        "task_recruitment": 0.79,
        "homeostatic_depth": 0.214,
        "plasticity_axis": 0.267,
        "sleep_recovery_axis": 0.222,
    },
    "hippocampus": {
        "recovery_depth": 0.80,
        "relay_graph_axis": 0.40,
        "task_recruitment": 0.63,
        "homeostatic_depth": 0.143,
        "plasticity_axis": 1.00,
        "sleep_recovery_axis": 0.556,
    },
    "salience": {
        "recovery_depth": 0.65,
        "relay_graph_axis": 0.60,
        "task_recruitment": 0.895,
        "homeostatic_depth": 0.429,
        "plasticity_axis": 0.60,
        "sleep_recovery_axis": 0.222,
    },
    "hypothalamus": {
        "recovery_depth": 0.35,
        "relay_graph_axis": 0.20,
        "task_recruitment": 0.16,
        "homeostatic_depth": 0.929,
        "plasticity_axis": 0.133,
        "sleep_recovery_axis": 0.889,
    },
    "brainstem": {
        "recovery_depth": 0.20,
        "relay_graph_axis": 0.13,
        "task_recruitment": 0.105,
        "homeostatic_depth": 0.786,
        "plasticity_axis": 0.067,
        "sleep_recovery_axis": 0.778,
    },
    "autonomic": {
        "recovery_depth": 0.00,
        "relay_graph_axis": 0.00,
        "task_recruitment": 0.00,
        "homeostatic_depth": 1.00,
        "plasticity_axis": 0.00,
        "sleep_recovery_axis": 1.00,
    },
}

TRUE_GENERATOR_PARAMETERS = {
    "rho_b": {"base": 0.110, "slope": 0.060},
    "gamma": {"base": 0.075, "slope": 0.075},
    "task_scale": {"base": 0.300, "slope": 0.950},
    "homeostasis_scale": {"base": 0.750, "slope": 0.700},
    "syn_scale": {"base": 0.600, "slope": 0.750},
    "slow_scale": {"base": 0.850, "slope": 0.450},
}

CURRENT_P = np.asarray(
    [
        [0.332, 0.250, 0.418],
        [0.326, 0.252, 0.422],
        [0.318, 0.266, 0.416],
        [0.334, 0.258, 0.408],
        [0.342, 0.255, 0.403],
        [0.336, 0.258, 0.406],
        [0.302, 0.262, 0.436],
    ],
    dtype=np.float64,
)

PERTURBATION = np.asarray(
    [
        [0.0015, -0.0008, -0.0007],
        [-0.0006, 0.0004, 0.0002],
        [-0.0009, 0.0007, 0.0002],
        [0.0006, -0.0003, -0.0003],
        [0.0012, -0.0006, -0.0006],
        [-0.0007, 0.0006, 0.0001],
        [0.0008, -0.0004, -0.0004],
    ],
    dtype=np.float64,
)

SPLIT_CASES = [
    {
        "case_id": "s1_base_a",
        "subject": "s1",
        "session": "baseline",
        "task": "task_a",
        "state_shift": 0.00,
        "task_input": 0.70,
        "homeostasis_input": 0.55,
        "syn_input": 0.60,
        "slow_input": 0.15,
        "noise": 0.25,
    },
    {
        "case_id": "s1_deprived_b",
        "subject": "s1",
        "session": "deprivation",
        "task": "task_b",
        "state_shift": 0.35,
        "task_input": 1.05,
        "homeostasis_input": 1.20,
        "syn_input": 0.90,
        "slow_input": 0.45,
        "noise": -0.20,
    },
    {
        "case_id": "s1_recovery_c",
        "subject": "s1",
        "session": "recovery",
        "task": "task_c",
        "state_shift": -0.20,
        "task_input": 0.85,
        "homeostasis_input": 0.70,
        "syn_input": 1.10,
        "slow_input": 0.35,
        "noise": 0.10,
    },
    {
        "case_id": "s2_base_b",
        "subject": "s2",
        "session": "baseline",
        "task": "task_b",
        "state_shift": -0.30,
        "task_input": 0.90,
        "homeostasis_input": 0.60,
        "syn_input": 0.75,
        "slow_input": 0.20,
        "noise": -0.15,
    },
    {
        "case_id": "s2_deprived_c",
        "subject": "s2",
        "session": "deprivation",
        "task": "task_c",
        "state_shift": 0.45,
        "task_input": 0.80,
        "homeostasis_input": 1.35,
        "syn_input": 1.20,
        "slow_input": 0.55,
        "noise": 0.30,
    },
    {
        "case_id": "s2_recovery_a",
        "subject": "s2",
        "session": "recovery",
        "task": "task_a",
        "state_shift": 0.15,
        "task_input": 1.20,
        "homeostasis_input": 0.85,
        "syn_input": 0.95,
        "slow_input": 0.40,
        "noise": -0.10,
    },
    {
        "case_id": "s3_base_c",
        "subject": "s3",
        "session": "baseline",
        "task": "task_c",
        "state_shift": 0.10,
        "task_input": 0.65,
        "homeostasis_input": 0.50,
        "syn_input": 1.00,
        "slow_input": 0.25,
        "noise": 0.20,
    },
    {
        "case_id": "s3_deprived_a",
        "subject": "s3",
        "session": "deprivation",
        "task": "task_a",
        "state_shift": 0.55,
        "task_input": 1.30,
        "homeostasis_input": 1.45,
        "syn_input": 1.05,
        "slow_input": 0.65,
        "noise": -0.25,
    },
    {
        "case_id": "s3_recovery_b",
        "subject": "s3",
        "session": "recovery",
        "task": "task_b",
        "state_shift": -0.10,
        "task_input": 1.00,
        "homeostasis_input": 0.90,
        "syn_input": 0.85,
        "slow_input": 0.50,
        "noise": 0.15,
    },
]

REAL_TRANSITION_REQUIRED_FIELDS = {
    "case_id",
    "subject",
    "session",
    "task",
    "current_p",
    "observed_next_p",
}


def project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 1e-12)
    return values / values.sum()


def laplacian(edges: list[tuple[str, str, float]] | None = None) -> np.ndarray:
    index = {name: i for i, name in enumerate(REGIONS)}
    adjacency = np.zeros((len(REGIONS), len(REGIONS)), dtype=np.float64)
    for left, right, weight in edges or EDGES:
        i = index[left]
        j = index[right]
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    degree = np.diag(adjacency.sum(axis=1))
    return degree - adjacency


def graph_delta(lap: np.ndarray, state: np.ndarray) -> np.ndarray:
    return -lap @ state


def p_star_matrix() -> np.ndarray:
    return np.asarray([P_STAR[region] for region in REGIONS], dtype=np.float64)


def homeostatic_forcing() -> np.ndarray:
    return homeostatic_forcing_from_q_delta(Q_DELTA)


def homeostatic_forcing_from_q_delta(q_delta: dict[str, float]) -> np.ndarray:
    forcing = []
    for region in REGIONS:
        sensitivity = D_SENSITIVITY[region]
        burden = sum(sensitivity[axis] * q_delta.get(axis, 0.0) for axis in Q_DELTA)
        # Burden shifts reserve mass into active and structural channels.
        forcing.append([0.70 * burden, 0.20 * burden, -0.90 * burden])
    return np.asarray(forcing, dtype=np.float64)


def task_forcing() -> np.ndarray:
    return task_forcing_for_load(TASK_LOAD)


def task_forcing_for_load(task_load: float) -> np.ndarray:
    forcing = []
    for region in REGIONS:
        active_drive = 0.02 * TASK_BETA[region] * task_load
        forcing.append([active_drive, 0.0, -active_drive])
    return np.asarray(forcing, dtype=np.float64)


def synaptic_forcing() -> np.ndarray:
    forcing = []
    for region in REGIONS:
        scale = SYN_ZETA[region] * SYN_LOAD[region]
        nu = SYN_NU[region]
        forcing.append(scale * np.asarray([nu, 1.0, -(1.0 + nu)], dtype=np.float64))
    return np.asarray(forcing, dtype=np.float64)


def slow_floor_strength(d_sleep: float) -> float:
    return float(np.clip(-0.4162 + 0.02579 * d_sleep, 0.0, 1.0))


def slow_forcing() -> np.ndarray:
    return slow_forcing_for_d_sleep(D_SLEEP)


def slow_forcing_for_d_sleep(d_sleep: float) -> np.ndarray:
    phi = slow_floor_strength(d_sleep)
    forcing = []
    for region in REGIONS:
        omega_a, omega_s = SLOW_OMEGA[region]
        forcing.append(phi * np.asarray([omega_a, omega_s, -(omega_a + omega_s)]))
    return np.asarray(forcing, dtype=np.float64)


def predict(
    current_p: np.ndarray,
    *,
    include_task: bool,
    include_graph: bool,
    include_homeostasis: bool,
    include_syn: bool,
    include_slow: bool,
) -> np.ndarray:
    lap = laplacian()
    delta_g = graph_delta(lap, current_p) if include_graph else np.zeros_like(current_p)
    task = task_forcing() if include_task else np.zeros_like(current_p)
    forcing = homeostatic_forcing() if include_homeostasis else np.zeros_like(current_p)
    syn = synaptic_forcing() if include_syn else np.zeros_like(current_p)
    slow = slow_forcing() if include_slow else np.zeros_like(current_p)

    predicted = []
    for region_index, p in enumerate(current_p):
        region = REGIONS[region_index]
        raw = (
            (1.0 - RHO_B) * P_STAR[region]
            + RHO_B * p
            + GAMMA * delta_g[region_index]
            + task[region_index]
            + forcing[region_index]
            + syn[region_index]
            + slow[region_index]
        )
        predicted.append(project_simplex(raw))
    return np.asarray(predicted, dtype=np.float64)


def squared_loss(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sum((observed - predicted) ** 2))


def component_matrix(
    current_p: np.ndarray,
    *,
    task_input: float = 1.0,
    homeostasis_input: float = 1.0,
    syn_input: float = 1.0,
    slow_input: float = 1.0,
    q_delta: dict[str, float] | None = None,
    d_sleep: float | None = None,
    graph_edges: list[tuple[str, str, float]] | None = None,
) -> dict[str, np.ndarray]:
    lap = laplacian(graph_edges)
    homeostasis = (
        homeostatic_forcing_from_q_delta(q_delta)
        if q_delta is not None
        else homeostatic_forcing()
    )
    slow = slow_forcing_for_d_sleep(d_sleep) if d_sleep is not None else slow_forcing()
    return {
        "rho_b": current_p - p_star_matrix(),
        "gamma": graph_delta(lap, current_p),
        "task_scale": task_forcing_for_load(task_input),
        "homeostasis_scale": homeostasis_input * homeostasis,
        "syn_scale": syn_input * synaptic_forcing(),
        "slow_scale": slow_input * slow,
    }


def fit_global_coefficients(observed: np.ndarray) -> dict[str, object]:
    components = component_matrix(CURRENT_P)
    names = list(components)
    design = np.column_stack([components[name].reshape(-1) for name in names])
    target = (observed - p_star_matrix()).reshape(-1)

    coeffs, residuals, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    fitted = p_star_matrix() + sum(
        coeff * components[name] for coeff, name in zip(coeffs, names)
    )

    true_coeffs = {
        "rho_b": RHO_B,
        "gamma": GAMMA,
        "task_scale": 1.0,
        "homeostasis_scale": 1.0,
        "syn_scale": 1.0,
        "slow_scale": 1.0,
    }
    estimated = {name: float(coeff) for name, coeff in zip(names, coeffs)}
    abs_error = {
        name: abs(estimated[name] - true_coeffs[name])
        for name in names
    }

    return {
        "estimated": estimated,
        "true": true_coeffs,
        "abs_error": abs_error,
        "max_abs_error": max(abs_error.values()),
        "fit_loss": squared_loss(observed, fitted),
        "residual_sum": float(residuals[0]) if residuals.size else 0.0,
        "rank": int(rank),
        "singular_values": singular_values.tolist(),
    }


def true_coefficients() -> dict[str, float]:
    return {
        "rho_b": RHO_B,
        "gamma": GAMMA,
        "task_scale": 1.0,
        "homeostasis_scale": 1.0,
        "syn_scale": 1.0,
        "slow_scale": 1.0,
    }


def generator_coefficients(
    parameters: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        region: {
            name: (
                parameters[name]["base"]
                + parameters[name]["slope"]
                * ANATOMICAL_AXES[region][COEFFICIENT_AXES[name]]
            )
            for name in COEFFICIENT_NAMES
        }
        for region in REGIONS
    }


def true_family_coefficients() -> dict[str, dict[str, float]]:
    return generator_coefficients(TRUE_GENERATOR_PARAMETERS)


def add_components(
    components: dict[str, np.ndarray],
    coeffs: dict[str, float],
) -> np.ndarray:
    raw = p_star_matrix().copy()
    for name, coeff in coeffs.items():
        raw += coeff * components[name]
    return raw


def add_family_components(
    components: dict[str, np.ndarray],
    coeffs: dict[str, dict[str, float]],
) -> np.ndarray:
    raw = p_star_matrix().copy()
    for region_index, region in enumerate(REGIONS):
        for name in COEFFICIENT_NAMES:
            raw[region_index] += coeffs[region][name] * components[name][region_index]
    return raw


def case_state(case: dict[str, object]) -> np.ndarray:
    shifted = CURRENT_P + float(case["state_shift"]) * PERTURBATION
    return np.asarray([project_simplex(row) for row in shifted], dtype=np.float64)


def case_components(case: dict[str, object]) -> dict[str, np.ndarray]:
    return component_matrix(
        case_state(case),
        task_input=float(case["task_input"]),
        homeostasis_input=float(case["homeostasis_input"]),
        syn_input=float(case["syn_input"]),
        slow_input=float(case["slow_input"]),
    )


def case_observed(case: dict[str, object]) -> np.ndarray:
    raw = add_components(case_components(case), true_coefficients())
    noisy = raw + float(case["noise"]) * PERTURBATION
    return np.asarray([project_simplex(row) for row in noisy], dtype=np.float64)


def family_case_observed(case: dict[str, object]) -> np.ndarray:
    raw = add_family_components(case_components(case), true_family_coefficients())
    noisy = raw + float(case["noise"]) * PERTURBATION
    return np.asarray([project_simplex(row) for row in noisy], dtype=np.float64)


def fit_scalar_cases(
    cases: list[dict[str, object]],
    observed_fn: Callable[[dict[str, object]], np.ndarray],
) -> tuple[dict[str, float], int]:
    names = list(true_coefficients())
    design_rows = []
    target_rows = []
    for case in cases:
        components = case_components(case)
        design_rows.append(
            np.column_stack([components[name].reshape(-1) for name in names])
        )
        target_rows.append((observed_fn(case) - p_star_matrix()).reshape(-1))

    design = np.vstack(design_rows)
    target = np.concatenate(target_rows)
    coeffs, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    return {name: float(coeff) for name, coeff in zip(names, coeffs)}, int(rank)


def fit_cases(cases: list[dict[str, object]]) -> tuple[dict[str, float], int]:
    return fit_scalar_cases(cases, case_observed)


def predict_case(case: dict[str, object], coeffs: dict[str, float]) -> np.ndarray:
    raw = add_components(case_components(case), coeffs)
    return np.asarray([project_simplex(row) for row in raw], dtype=np.float64)


def split_loss(
    cases: list[dict[str, object]],
    coeffs: dict[str, float],
    observed_fn: Callable[[dict[str, object]], np.ndarray] = case_observed,
) -> float:
    return sum(
        squared_loss(observed_fn(case), predict_case(case, coeffs))
        for case in cases
    )


def family_parameter_names() -> list[tuple[str, str]]:
    return [
        (region, name)
        for region in REGIONS
        for name in COEFFICIENT_NAMES
    ]


def family_design_column(
    components: dict[str, np.ndarray],
    region: str,
    name: str,
) -> np.ndarray:
    column = np.zeros_like(components[name])
    column[REGIONS.index(region)] = components[name][REGIONS.index(region)]
    return column.reshape(-1)


def fit_family_cases(
    cases: list[dict[str, object]],
) -> tuple[dict[str, dict[str, float]], int]:
    parameter_names = family_parameter_names()
    design_rows = []
    target_rows = []
    for case in cases:
        components = case_components(case)
        design_rows.append(
            np.column_stack(
                [
                    family_design_column(components, region, name)
                    for region, name in parameter_names
                ]
            )
        )
        target_rows.append((family_case_observed(case) - p_star_matrix()).reshape(-1))

    design = np.vstack(design_rows)
    target = np.concatenate(target_rows)
    coeffs, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    estimated = {
        region: {
            name: float(coeffs[index])
            for index, (param_region, name) in enumerate(parameter_names)
            if param_region == region
        }
        for region in REGIONS
    }
    return estimated, int(rank)


def predict_family_case(
    case: dict[str, object],
    coeffs: dict[str, dict[str, float]],
) -> np.ndarray:
    raw = add_family_components(case_components(case), coeffs)
    return np.asarray([project_simplex(row) for row in raw], dtype=np.float64)


def family_split_loss(
    cases: list[dict[str, object]],
    coeffs: dict[str, dict[str, float]],
) -> float:
    return sum(
        squared_loss(family_case_observed(case), predict_family_case(case, coeffs))
        for case in cases
    )


def max_family_abs_error(coeffs: dict[str, dict[str, float]]) -> float:
    true = true_family_coefficients()
    return max(
        abs(coeffs[region][name] - true[region][name])
        for region in REGIONS
        for name in COEFFICIENT_NAMES
    )


def generator_parameter_names() -> list[tuple[str, str]]:
    return [
        (name, kind)
        for name in COEFFICIENT_NAMES
        for kind in ("base", "slope")
    ]


def generator_design_column(
    components: dict[str, np.ndarray],
    name: str,
    kind: str,
) -> np.ndarray:
    if kind == "base":
        return components[name].reshape(-1)
    axis = COEFFICIENT_AXES[name]
    weights = np.asarray(
        [ANATOMICAL_AXES[region][axis] for region in REGIONS],
        dtype=np.float64,
    )
    return (weights[:, None] * components[name]).reshape(-1)


def fit_generator_cases(
    cases: list[dict[str, object]],
) -> tuple[dict[str, dict[str, float]], int]:
    parameter_names = generator_parameter_names()
    design_rows = []
    target_rows = []
    for case in cases:
        components = case_components(case)
        design_rows.append(
            np.column_stack(
                [
                    generator_design_column(components, name, kind)
                    for name, kind in parameter_names
                ]
            )
        )
        target_rows.append((family_case_observed(case) - p_star_matrix()).reshape(-1))

    design = np.vstack(design_rows)
    target = np.concatenate(target_rows)
    coeffs, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    parameters = {
        name: {
            kind: float(coeffs[index])
            for index, (param_name, kind) in enumerate(parameter_names)
            if param_name == name
        }
        for name in COEFFICIENT_NAMES
    }
    return generator_coefficients(parameters), int(rank)


def max_generator_abs_error(coeffs: dict[str, dict[str, float]]) -> float:
    return max_family_abs_error(coeffs)


def evaluate_family_split(split_key: str, holdout_value: str) -> dict[str, object]:
    train = [case for case in SPLIT_CASES if case[split_key] != holdout_value]
    holdout = [case for case in SPLIT_CASES if case[split_key] == holdout_value]
    family_coeffs, family_rank = fit_family_cases(train)
    generator_coeffs, generator_rank = fit_generator_cases(train)
    scalar_coeffs, scalar_rank = fit_scalar_cases(train, family_case_observed)
    family_loss = family_split_loss(holdout, family_coeffs)
    generator_loss = family_split_loss(holdout, generator_coeffs)
    scalar_loss = split_loss(holdout, scalar_coeffs, family_case_observed)

    return {
        "split_key": split_key,
        "holdout_value": holdout_value,
        "family_rank": family_rank,
        "required_family_rank": FAMILY_COEFFICIENT_COUNT,
        "generator_rank": generator_rank,
        "required_generator_rank": GENERATOR_COEFFICIENT_COUNT,
        "scalar_rank": scalar_rank,
        "family_holdout_loss": family_loss,
        "generator_holdout_loss": generator_loss,
        "scalar_holdout_loss": scalar_loss,
        "family_over_scalar": family_loss / scalar_loss,
        "generator_over_scalar": generator_loss / scalar_loss,
        "generator_over_family": generator_loss / family_loss,
        "max_family_abs_error": max_family_abs_error(family_coeffs),
        "max_generator_abs_error": max_generator_abs_error(generator_coeffs),
        "passed": generator_loss < scalar_loss
        and generator_rank == GENERATOR_COEFFICIENT_COUNT,
    }


def evaluate_family_inverse() -> dict[str, object]:
    splits = [
        evaluate_family_split("subject", "s3"),
        evaluate_family_split("session", "recovery"),
        evaluate_family_split("task", "task_c"),
    ]
    return {
        "criterion": "12-parameter anatomical generator must beat global scalar coefficients on heterogeneous synthetic transitions",
        "coefficient_count": FAMILY_COEFFICIENT_COUNT,
        "generator_coefficient_count": GENERATOR_COEFFICIENT_COUNT,
        "anatomical_axes": ANATOMICAL_AXES,
        "true_generator_parameters": TRUE_GENERATOR_PARAMETERS,
        "true_coefficients": true_family_coefficients(),
        "splits": splits,
        "passed_all_splits": all(split["passed"] for split in splits),
        "max_family_abs_error": max(split["max_family_abs_error"] for split in splits),
        "max_generator_abs_error": max(
            split["max_generator_abs_error"] for split in splits
        ),
    }


def evaluate_split(split_key: str, holdout_value: str) -> dict[str, object]:
    train = [case for case in SPLIT_CASES if case[split_key] != holdout_value]
    holdout = [case for case in SPLIT_CASES if case[split_key] == holdout_value]
    estimated, rank = fit_cases(train)

    full_loss = split_loss(holdout, estimated)
    ablation_losses = {}
    for name in estimated:
        ablated = dict(estimated)
        ablated[name] = 0.0
        ablation_losses[f"no_{name}"] = split_loss(holdout, ablated)

    inertial = {name: 0.0 for name in estimated}
    inertial["rho_b"] = estimated["rho_b"]
    ablation_losses["inertial"] = split_loss(holdout, inertial)
    best_ablation = min(ablation_losses.values())

    true = true_coefficients()
    abs_error = {name: abs(estimated[name] - true[name]) for name in estimated}
    return {
        "split_key": split_key,
        "holdout_value": holdout_value,
        "train_cases": [str(case["case_id"]) for case in train],
        "holdout_cases": [str(case["case_id"]) for case in holdout],
        "estimated": estimated,
        "abs_error": abs_error,
        "max_abs_error": max(abs_error.values()),
        "rank": rank,
        "holdout_loss": full_loss,
        "ablation_losses": ablation_losses,
        "full_over_best_ablation": full_loss / best_ablation,
        "passed": full_loss < best_ablation,
    }


def evaluate_split_inverse() -> dict[str, object]:
    splits = [
        evaluate_split("subject", "s3"),
        evaluate_split("session", "recovery"),
        evaluate_split("task", "task_c"),
    ]
    return {
        "criterion": "train on non-held-out cases; predict held-out subject/session/task",
        "splits": splits,
        "passed_all_splits": all(split["passed"] for split in splits),
        "max_split_abs_error": max(split["max_abs_error"] for split in splits),
    }


def parse_state_matrix(value: object, *, field: str, case_id: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    expected_shape = (len(REGIONS), 3)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{case_id}.{field} must have shape {expected_shape}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{case_id}.{field} contains non-finite values")
    if np.any(matrix < 0.0):
        raise ValueError(f"{case_id}.{field} contains negative simplex entries")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"{case_id}.{field} rows must sum to 1")
    return matrix


def parse_float_field(
    raw: dict[str, object],
    *,
    case_id: str,
    names: tuple[str, ...],
    default: float | None = None,
) -> float:
    for name in names:
        if name in raw:
            value = float(raw[name])
            if not np.isfinite(value):
                raise ValueError(f"{case_id}.{name} must be finite")
            return value
    if default is not None:
        return default
    raise ValueError(f"{case_id} must include one of {names}")


def parse_q_delta(raw: dict[str, object], *, case_id: str) -> dict[str, float] | None:
    if "q_delta" not in raw:
        return None
    value = raw["q_delta"]
    if not isinstance(value, dict):
        raise ValueError(f"{case_id}.q_delta must be an object")
    parsed = {}
    for axis in Q_DELTA:
        parsed[axis] = float(value.get(axis, 0.0))
        if not np.isfinite(parsed[axis]):
            raise ValueError(f"{case_id}.q_delta.{axis} must be finite")
    return parsed


def parse_graph_edges(
    raw: dict[str, object],
    *,
    case_id: str,
) -> list[tuple[str, str, float]] | None:
    if "graph_edges" not in raw and "edges" not in raw:
        return None
    value = raw.get("graph_edges", raw.get("edges"))
    if not isinstance(value, list):
        raise ValueError(f"{case_id}.graph_edges must be a list")

    parsed = []
    valid_regions = set(REGIONS)
    for edge_index, edge in enumerate(value):
        if isinstance(edge, dict):
            left = str(edge.get("left", edge.get("source", "")))
            right = str(edge.get("right", edge.get("target", "")))
            weight = float(edge.get("weight", 0.0))
        elif isinstance(edge, (list, tuple)) and len(edge) == 3:
            left = str(edge[0])
            right = str(edge[1])
            weight = float(edge[2])
        else:
            raise ValueError(
                f"{case_id}.graph_edges[{edge_index}] must be an object or [left, right, weight]"
            )
        if left not in valid_regions or right not in valid_regions:
            raise ValueError(
                f"{case_id}.graph_edges[{edge_index}] uses unknown region {left!r}, {right!r}"
            )
        if left == right:
            raise ValueError(f"{case_id}.graph_edges[{edge_index}] is a self-edge")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"{case_id}.graph_edges[{edge_index}] weight must be finite and non-negative"
            )
        parsed.append((left, right, weight))
    return parsed


def parse_real_transition(raw: object, index: int) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"transition #{index} must be an object")
    missing = sorted(REAL_TRANSITION_REQUIRED_FIELDS.difference(raw))
    case_id = str(raw.get("case_id", f"transition_{index}"))
    if missing:
        raise ValueError(f"{case_id} is missing required fields: {missing}")
    has_homeostasis = "homeostasis_input" in raw or "q_delta" in raw
    has_slow = "slow_input" in raw or "d_sleep" in raw
    if not has_homeostasis:
        raise ValueError(f"{case_id} must include homeostasis_input or q_delta")
    if not has_slow:
        raise ValueError(f"{case_id} must include slow_input or d_sleep")

    return {
        "case_id": case_id,
        "subject": str(raw["subject"]),
        "session": str(raw["session"]),
        "task": str(raw["task"]),
        "current_p": parse_state_matrix(raw["current_p"], field="current_p", case_id=case_id),
        "observed_next_p": parse_state_matrix(
            raw["observed_next_p"], field="observed_next_p", case_id=case_id
        ),
        "task_input": parse_float_field(raw, case_id=case_id, names=("task_input", "task_load")),
        "homeostasis_input": parse_float_field(
            raw, case_id=case_id, names=("homeostasis_input",), default=1.0
        ),
        "q_delta": parse_q_delta(raw, case_id=case_id),
        "syn_input": parse_float_field(
            raw, case_id=case_id, names=("syn_input", "plasticity_input")
        ),
        "slow_input": parse_float_field(
            raw, case_id=case_id, names=("slow_input",), default=1.0
        ),
        "d_sleep": (
            parse_float_field(raw, case_id=case_id, names=("d_sleep",), default=0.0)
            if "d_sleep" in raw
            else None
        ),
        "graph_edges": parse_graph_edges(raw, case_id=case_id),
    }


def validate_declared_region_order(data: dict[str, object]) -> None:
    if "region_order" not in data:
        return
    declared = data["region_order"]
    if not isinstance(declared, list) or [str(region) for region in declared] != REGIONS:
        raise ValueError(
            "top-level region_order must match canonical order: "
            f"{REGIONS}"
        )


def load_real_transitions(path: Path) -> list[dict[str, object]]:
    raw_text = sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8")
    data = json.loads(raw_text)
    if isinstance(data, dict):
        validate_declared_region_order(data)
        data = data.get("transitions")
    if not isinstance(data, list):
        raise ValueError("real transition JSON must be a list or an object with transitions")
    if not data:
        raise ValueError("real transition JSON contains no transitions")
    return [parse_real_transition(item, index) for index, item in enumerate(data)]


def real_transition_input_summary(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "case_count": len(cases),
        "region_order": REGIONS,
        "uses_case_graphs": any(case["graph_edges"] is not None for case in cases),
        "case_graph_count": sum(case["graph_edges"] is not None for case in cases),
        "q_delta_case_count": sum(case["q_delta"] is not None for case in cases),
        "d_sleep_case_count": sum(case["d_sleep"] is not None for case in cases),
        "counts": {
            "subject": count_by_key(cases, "subject"),
            "session": count_by_key(cases, "session"),
            "task": count_by_key(cases, "task"),
        },
        "split_values": real_split_values(cases),
    }


def check_real_transitions(path: Path) -> dict[str, object]:
    cases = load_real_transitions(path)
    gate = evaluate_real_data_gate(cases)
    return {
        "input": real_transition_input_summary(cases),
        "readiness": gate["readiness"],
        "graph_diagnostics": gate["graph_diagnostics"],
        "passed_all_evaluated_splits": gate["passed_all_evaluated_splits"],
        "passed_real_gate": gate["passed_real_gate"],
    }


def real_transition_schema() -> dict[str, object]:
    return {
        "accepted_top_level": [
            "list[transition]",
            {
                "region_order": REGIONS,
                "transitions": "list[transition]",
            },
        ],
        "required_fields": sorted(REAL_TRANSITION_REQUIRED_FIELDS),
        "region_order": REGIONS,
        "state_matrix": {
            "fields": ["current_p", "observed_next_p"],
            "shape": [len(REGIONS), 3],
            "components": ["x_a", "x_s", "x_b"],
            "row_constraints": {
                "non_negative": True,
                "sum_to_one_atol": 1.0e-5,
            },
        },
        "control_inputs": {
            "task": ["task_input", "task_load"],
            "homeostasis": ["homeostasis_input", "q_delta"],
            "q_delta_axes": list(Q_DELTA),
            "synaptic": ["syn_input", "plasticity_input"],
            "slow": ["slow_input", "d_sleep"],
        },
        "graph_edges": {
            "optional": True,
            "aliases": ["graph_edges", "edges"],
            "formats": [
                ["left_region", "right_region", "weight"],
                {"source": "left_region", "target": "right_region", "weight": 0.0},
                {"left": "left_region", "right": "right_region", "weight": 0.0},
            ],
            "constraints": {
                "known_regions_only": True,
                "no_self_edges": True,
                "finite_non_negative_weight": True,
            },
            "default_if_omitted": "canonical graph",
        },
        "split_fields": ["subject", "session", "task"],
        "claim_readiness": {
            "requires_evaluated_splits": True,
            "requires_no_skipped_splits": True,
            "requires_identifiability": {
                "rank": len(COEFFICIENT_NAMES),
                "max_condition_number": MAX_REAL_CONDITION_NUMBER,
            },
            "requires_full_better_than_ablations": True,
            "requires_stable_graphs": True,
        },
    }


def state_template_rows() -> list[list[float]]:
    return [
        [float(round(value, 6)) for value in P_STAR[region]]
        for region in REGIONS
    ]


def real_transition_template() -> dict[str, object]:
    state_rows = state_template_rows()
    return {
        "region_order": REGIONS,
        "transitions": [
            {
                "case_id": "subject01_session01_task01",
                "subject": "subject01",
                "session": "session01",
                "task": "task01",
                "current_p": state_rows,
                "observed_next_p": state_rows,
                "task_load": 1.0,
                "q_delta": {
                    "sleep": 0.0,
                    "arousal": 0.0,
                    "metabolic": 0.0,
                },
                "plasticity_input": 1.0,
                "d_sleep": 0.0,
                "graph_edges": [
                    [left, right, float(weight)]
                    for left, right, weight in EDGES
                ],
            }
        ],
    }


def graph_edges_for_case(case: dict[str, object]) -> list[tuple[str, str, float]]:
    return case["graph_edges"] if case["graph_edges"] is not None else EDGES


def shuffled_graph_edges(case: dict[str, object]) -> list[tuple[str, str, float]]:
    region_map = {
        region: REGIONS[(index + 1) % len(REGIONS)]
        for index, region in enumerate(REGIONS)
    }
    return [
        (region_map[left], region_map[right], weight)
        for left, right, weight in graph_edges_for_case(case)
    ]


def edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def degree_preserving_graph_edges(case: dict[str, object]) -> list[tuple[str, str, float]]:
    edges = graph_edges_for_case(case)
    existing = {edge_key(left, right) for left, right, _ in edges}
    rewired = list(edges)

    for first in range(0, len(edges) - 1, 2):
        left_a, right_a, weight_a = rewired[first]
        left_b, right_b, weight_b = rewired[first + 1]
        candidate_a = edge_key(left_a, right_b)
        candidate_b = edge_key(left_b, right_a)
        if (
            left_a == right_b
            or left_b == right_a
            or candidate_a in existing
            or candidate_b in existing
        ):
            continue

        existing.discard(edge_key(left_a, right_a))
        existing.discard(edge_key(left_b, right_b))
        existing.add(candidate_a)
        existing.add(candidate_b)
        rewired[first] = (left_a, right_b, weight_a)
        rewired[first + 1] = (left_b, right_a, weight_b)

    return rewired


def transition_components(
    case: dict[str, object],
    *,
    graph_edges: list[tuple[str, str, float]] | None = None,
) -> dict[str, np.ndarray]:
    return component_matrix(
        case["current_p"],
        task_input=float(case["task_input"]),
        homeostasis_input=float(case["homeostasis_input"]),
        syn_input=float(case["syn_input"]),
        slow_input=float(case["slow_input"]),
        q_delta=case["q_delta"],
        d_sleep=case["d_sleep"],
        graph_edges=graph_edges if graph_edges is not None else case["graph_edges"],
    )


def transition_observed(case: dict[str, object]) -> np.ndarray:
    return case["observed_next_p"]


def fit_transition_cases(cases: list[dict[str, object]]) -> tuple[dict[str, float], dict[str, object]]:
    names = COEFFICIENT_NAMES
    design_rows = []
    target_rows = []
    for case in cases:
        components = transition_components(case)
        design_rows.append(
            np.column_stack([components[name].reshape(-1) for name in names])
        )
        target_rows.append((transition_observed(case) - p_star_matrix()).reshape(-1))

    design = np.vstack(design_rows)
    target = np.concatenate(target_rows)
    coeffs, _, rank, singular_values = np.linalg.lstsq(design, target, rcond=None)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    diagnostics = {
        "rank": int(rank),
        "required_rank": len(names),
        "condition_number": condition_number,
        "singular_values": singular_values.tolist(),
        "passed_identifiability": int(rank) == len(names)
        and condition_number <= MAX_REAL_CONDITION_NUMBER,
    }
    return {name: float(coeff) for name, coeff in zip(names, coeffs)}, diagnostics


def predict_transition_case(
    case: dict[str, object],
    coeffs: dict[str, float],
    *,
    graph_mode: str = "actual",
) -> np.ndarray:
    if graph_mode == "shuffled":
        graph_edges = shuffled_graph_edges(case)
    elif graph_mode == "degree_preserving":
        graph_edges = degree_preserving_graph_edges(case)
    else:
        graph_edges = None
    raw = add_components(transition_components(case, graph_edges=graph_edges), coeffs)
    return np.asarray([project_simplex(row) for row in raw], dtype=np.float64)


def transition_loss(
    cases: list[dict[str, object]],
    coeffs: dict[str, float],
    *,
    graph_mode: str = "actual",
) -> float:
    return sum(
        squared_loss(
            transition_observed(case),
            predict_transition_case(case, coeffs, graph_mode=graph_mode),
        )
        for case in cases
    )


def graph_diagnostics_for_case(case: dict[str, object]) -> dict[str, object]:
    edges = graph_edges_for_case(case)
    lap = laplacian(edges)
    eigvals = np.linalg.eigvalsh(lap)
    modal_radius = float(np.max(np.abs(RHO_B - GAMMA * eigvals)))
    delta_g = graph_delta(lap, case["current_p"])
    region_energy = [
        {
            "region": region,
            "coupling_energy": float(np.sum(delta_g[index] ** 2)),
        }
        for index, region in enumerate(REGIONS)
    ]
    edge_energy = [
        {
            "left": left,
            "right": right,
            "weight": float(weight),
            "energy": float(
                weight
                * np.sum(
                    (
                        case["current_p"][REGIONS.index(left)]
                        - case["current_p"][REGIONS.index(right)]
                    )
                    ** 2
                )
            ),
        }
        for left, right, weight in edges
    ]
    return {
        "case_id": str(case["case_id"]),
        "uses_case_graph": case["graph_edges"] is not None,
        "modal_radius": modal_radius,
        "stability_margin": 1.0 - modal_radius,
        "passed_stability": modal_radius < 1.0,
        "coupling_energy": float(np.sum(delta_g**2)),
        "lambda_max": float(eigvals.max()),
        "top_regions": sorted(
            region_energy,
            key=lambda row: row["coupling_energy"],
            reverse=True,
        )[:3],
        "top_edges": sorted(
            edge_energy,
            key=lambda row: row["energy"],
            reverse=True,
        )[:5],
    }


def graph_diagnostics(cases: list[dict[str, object]]) -> dict[str, object]:
    case_rows = [graph_diagnostics_for_case(case) for case in cases]
    return {
        "criterion": "all real transition graphs must satisfy max_k |rho_B - gamma lambda_k| < 1",
        "cases": case_rows,
        "max_modal_radius": max(row["modal_radius"] for row in case_rows),
        "min_stability_margin": min(row["stability_margin"] for row in case_rows),
        "max_coupling_energy": max(row["coupling_energy"] for row in case_rows),
        "passed_all_graphs": all(row["passed_stability"] for row in case_rows),
    }


def evaluate_real_split(
    cases: list[dict[str, object]],
    split_key: str,
    holdout_value: str,
) -> dict[str, object]:
    train = [case for case in cases if case[split_key] != holdout_value]
    holdout = [case for case in cases if case[split_key] == holdout_value]
    estimated, identifiability = fit_transition_cases(train)

    full_loss = transition_loss(holdout, estimated)
    ablation_losses = {}
    for name in estimated:
        ablated = dict(estimated)
        ablated[name] = 0.0
        ablation_losses[f"no_{name}"] = transition_loss(holdout, ablated)
    ablation_losses["shuffled_graph"] = transition_loss(
        holdout, estimated, graph_mode="shuffled"
    )
    ablation_losses["degree_preserving_graph"] = transition_loss(
        holdout, estimated, graph_mode="degree_preserving"
    )

    inertial = {name: 0.0 for name in estimated}
    inertial["rho_b"] = estimated["rho_b"]
    ablation_losses["inertial"] = transition_loss(holdout, inertial)
    best_ablation = min(ablation_losses.values())

    return {
        "split_key": split_key,
        "holdout_value": holdout_value,
        "train_cases": [str(case["case_id"]) for case in train],
        "holdout_cases": [str(case["case_id"]) for case in holdout],
        "estimated": estimated,
        "rank": identifiability["rank"],
        "identifiability": identifiability,
        "holdout_loss": full_loss,
        "ablation_losses": ablation_losses,
        "full_over_best_ablation": full_loss / best_ablation,
        "passed": full_loss < best_ablation
        and bool(identifiability["passed_identifiability"]),
    }


def real_split_values(cases: list[dict[str, object]]) -> dict[str, list[str]]:
    return {
        key: sorted({str(case[key]) for case in cases})
        for key in ("subject", "session", "task")
    }


def count_by_key(cases: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        value = str(case[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def readiness_next_actions(
    skipped: list[dict[str, str]],
    identifiability_failures: list[dict[str, object]],
    prediction_failures: list[dict[str, object]],
    graph_failures: list[dict[str, object]],
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for skipped_split in skipped:
        split_key = skipped_split["split_key"]
        actions.append(
            {
                "level": "data",
                "reason": f"{split_key} split has fewer than two values",
                "action": f"add transitions covering at least two {split_key} values",
            }
        )
    if identifiability_failures:
        actions.append(
            {
                "level": "design",
                "reason": "train design matrix cannot separate all six coefficients",
                "action": (
                    "add transitions with independent variation in task, "
                    "homeostasis, synaptic, slow, graph, and recovery terms"
                ),
                "affected_splits": [
                    {
                        "split_key": item["split_key"],
                        "holdout_value": item["holdout_value"],
                        "rank": item["rank"],
                        "required_rank": item["required_rank"],
                        "condition_number": item["condition_number"],
                    }
                    for item in identifiability_failures
                ],
            }
        )
    if prediction_failures:
        actions.append(
            {
                "level": "model",
                "reason": "full equation does not beat the best ablation on holdout",
                "action": "inspect the failed holdout cases and compare term-specific ablation losses",
                "affected_splits": prediction_failures,
            }
        )
    if graph_failures:
        actions.append(
            {
                "level": "graph",
                "reason": "one or more transition graphs fail the stability gate",
                "action": "inspect top_regions and top_edges before using graph-based claims",
                "affected_cases": graph_failures,
            }
        )
    if not actions:
        actions.append(
            {
                "level": "claim",
                "reason": "readiness gate has no blocking failures",
                "action": "run full real gate and report only the evaluated split results",
            }
        )
    return actions


def real_data_readiness(
    cases: list[dict[str, object]],
    split_values: dict[str, list[str]],
    splits: list[dict[str, object]],
    skipped: list[dict[str, str]],
    graph_gate: dict[str, object],
) -> dict[str, object]:
    identifiability_failures = [
        {
            "split_key": split["split_key"],
            "holdout_value": split["holdout_value"],
            "rank": split["identifiability"]["rank"],
            "required_rank": split["identifiability"]["required_rank"],
            "condition_number": split["identifiability"]["condition_number"],
        }
        for split in splits
        if not split["identifiability"]["passed_identifiability"]
    ]
    prediction_failures = [
        {
            "split_key": split["split_key"],
            "holdout_value": split["holdout_value"],
            "full_over_best_ablation": split["full_over_best_ablation"],
        }
        for split in splits
        if split["full_over_best_ablation"] >= 1.0
    ]
    graph_failures = [
        {
            "case_id": row["case_id"],
            "modal_radius": row["modal_radius"],
            "stability_margin": row["stability_margin"],
        }
        for row in graph_gate["cases"]
        if not row["passed_stability"]
    ]

    return {
        "case_count": len(cases),
        "counts": {
            "subject": count_by_key(cases, "subject"),
            "session": count_by_key(cases, "session"),
            "task": count_by_key(cases, "task"),
        },
        "split_value_count": {key: len(values) for key, values in split_values.items()},
        "evaluated_split_count": len(splits),
        "skipped": skipped,
        "identifiability_failures": identifiability_failures,
        "prediction_failures": prediction_failures,
        "graph_failures": graph_failures,
        "next_actions": readiness_next_actions(
            skipped,
            identifiability_failures,
            prediction_failures,
            graph_failures,
        ),
        "ready_for_claim": bool(splits)
        and not skipped
        and not identifiability_failures
        and not prediction_failures
        and not graph_failures,
    }


def evaluate_real_data_gate(cases: list[dict[str, object]]) -> dict[str, object]:
    split_values = real_split_values(cases)
    graph_gate = graph_diagnostics(cases)
    splits = []
    skipped = []
    for split_key, values in split_values.items():
        if len(values) < 2:
            skipped.append({"split_key": split_key, "reason": "need at least two values"})
            continue
        for holdout_value in values:
            splits.append(evaluate_real_split(cases, split_key, holdout_value))
    readiness = real_data_readiness(cases, split_values, splits, skipped, graph_gate)

    return {
        "criterion": "real transition JSON must pass subject/session/task holdout when those splits exist",
        "case_count": len(cases),
        "regions": REGIONS,
        "uses_case_graphs": any(case["graph_edges"] is not None for case in cases),
        "readiness": readiness,
        "graph_diagnostics": graph_gate,
        "split_values": split_values,
        "splits": splits,
        "skipped": skipped,
        "passed_all_evaluated_splits": bool(splits) and all(split["passed"] for split in splits),
        "passed_real_gate": graph_gate["passed_all_graphs"]
        and bool(splits)
        and all(split["passed"] for split in splits),
    }


def prediction_suite() -> dict[str, np.ndarray]:
    return {
        "full": predict(
            CURRENT_P,
            include_task=True,
            include_graph=True,
            include_homeostasis=True,
            include_syn=True,
            include_slow=True,
        ),
        "no_task": predict(
            CURRENT_P,
            include_task=False,
            include_graph=True,
            include_homeostasis=True,
            include_syn=True,
            include_slow=True,
        ),
        "no_graph": predict(
            CURRENT_P,
            include_task=True,
            include_graph=False,
            include_homeostasis=True,
            include_syn=True,
            include_slow=True,
        ),
        "no_homeostasis": predict(
            CURRENT_P,
            include_task=True,
            include_graph=True,
            include_homeostasis=False,
            include_syn=True,
            include_slow=True,
        ),
        "no_syn": predict(
            CURRENT_P,
            include_task=True,
            include_graph=True,
            include_homeostasis=True,
            include_syn=False,
            include_slow=True,
        ),
        "no_slow": predict(
            CURRENT_P,
            include_task=True,
            include_graph=True,
            include_homeostasis=True,
            include_syn=True,
            include_slow=False,
        ),
        "inertial": predict(
            CURRENT_P,
            include_task=False,
            include_graph=False,
            include_homeostasis=False,
            include_syn=False,
            include_slow=False,
        ),
    }


def observed_from_full(full: np.ndarray) -> np.ndarray:
    return np.asarray([project_simplex(row) for row in full + PERTURBATION])


def loss_table(observed: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        name: squared_loss(observed, prediction)
        for name, prediction in predictions.items()
    }


def ce_bridge_metrics(losses: dict[str, float], best_ablation: float) -> dict[str, object]:
    lap = laplacian()
    eigvals = np.linalg.eigvalsh(lap)
    modal_radius = float(np.max(np.abs(RHO_B - GAMMA * eigvals)))
    delta_g = graph_delta(lap, CURRENT_P)
    forcing = (
        task_forcing()
        + homeostatic_forcing()
        + synaptic_forcing()
        + slow_forcing()
    )

    return {
        "criterion": "CE-style stability and effective-action diagnostics",
        "modal_radius": modal_radius,
        "stability_margin": 1.0 - modal_radius,
        "passed_stability": modal_radius < 1.0,
        "coupling_energy": float(np.sum(delta_g**2)),
        "forcing_energy": float(np.sum(forcing**2)),
        "ablation_gain": 1.0 - losses["full"] / best_ablation,
        "laplacian_eigenvalues": eigvals.tolist(),
        "interpretation": {
            "modal_radius": "max_k |rho_B - gamma lambda_k(L_G)|",
            "coupling_energy": "||Delta_G P||^2",
            "forcing_energy": "||U_task + H + F_syn + F_slow||^2",
            "ablation_gain": "1 - L_full / L_best_ablation",
        },
    }


def region_rows(observed: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    homeostasis = homeostatic_forcing()
    task = task_forcing()
    syn = synaptic_forcing()
    slow = slow_forcing()
    for index, region in enumerate(REGIONS):
        rows.append(
            {
                "region": region,
                "current_p": CURRENT_P[index].tolist(),
                "observed_next_p": observed[index].tolist(),
                "full_prediction": predictions["full"][index].tolist(),
                "no_task_prediction": predictions["no_task"][index].tolist(),
                "no_graph_prediction": predictions["no_graph"][index].tolist(),
                "no_homeostasis_prediction": predictions["no_homeostasis"][index].tolist(),
                "no_syn_prediction": predictions["no_syn"][index].tolist(),
                "no_slow_prediction": predictions["no_slow"][index].tolist(),
                "task_forcing": task[index].tolist(),
                "homeostatic_forcing": homeostasis[index].tolist(),
                "synaptic_forcing": syn[index].tolist(),
                "slow_forcing": slow[index].tolist(),
            }
        )
    return rows


def build_output(real_transition_path: Path | None = None) -> dict[str, object]:
    predictions = prediction_suite()
    observed = observed_from_full(predictions["full"])

    losses = loss_table(observed, predictions)
    best_ablation = min(value for key, value in losses.items() if key != "full")
    inverse_fit = fit_global_coefficients(observed)
    split_inverse = evaluate_split_inverse()
    family_inverse = evaluate_family_inverse()
    bridge_metrics = ce_bridge_metrics(losses, best_ablation)

    out = {
        "criterion": "L_full < min(L_no_task, L_no_graph, L_no_homeostasis, L_no_syn, L_no_slow, L_inertial)",
        "rho_b": RHO_B,
        "gamma": GAMMA,
        "q_delta": Q_DELTA,
        "task_load": TASK_LOAD,
        "d_sleep": D_SLEEP,
        "slow_floor_strength": slow_floor_strength(D_SLEEP),
        "losses": losses,
        "full_over_best_ablation": losses["full"] / best_ablation,
        "passed_integrated_gate": losses["full"] < best_ablation,
        "inverse_fit": inverse_fit,
        "split_inverse": split_inverse,
        "family_inverse": family_inverse,
        "ce_bridge_metrics": bridge_metrics,
        "regions": region_rows(observed, predictions),
        "interpretation": {
            "full": "inertia, task, graph, homeostasis, synaptic, and slow recovery terms",
            "no_task": "removes U_task",
            "no_graph": "removes Delta_G",
            "no_homeostasis": "removes H_r(q_n-q*)",
            "no_syn": "removes F_syn",
            "no_slow": "removes F_slow",
            "inertial": "keeps only relaxation toward p* and previous state",
        },
    }
    if real_transition_path is not None:
        cases = load_real_transitions(real_transition_path)
        out["real_data_gate"] = evaluate_real_data_gate(cases)
    return out


def print_summary(out: dict[str, object]) -> None:
    losses = out["losses"]
    inverse_fit = out["inverse_fit"]
    split_inverse = out["split_inverse"]
    family_inverse = out["family_inverse"]
    bridge_metrics = out["ce_bridge_metrics"]
    print("Integrated brain-equation gate")
    for name, value in losses.items():
        print(f"  L_{name:14s} = {value:.8f}")
    print(f"  full/best_ablation = {out['full_over_best_ablation']:.6f}")
    print("  inverse fit:")
    for name, value in inverse_fit["estimated"].items():
        print(f"    {name:17s} = {value:.6f}")
    print(f"    max_abs_error     = {inverse_fit['max_abs_error']:.6f}")
    print(f"    fit_loss          = {inverse_fit['fit_loss']:.8f}")
    print("  CE bridge metrics:")
    print(f"    modal_radius      = {bridge_metrics['modal_radius']:.6f}")
    print(f"    stability_margin  = {bridge_metrics['stability_margin']:.6f}")
    print(f"    coupling_energy   = {bridge_metrics['coupling_energy']:.8f}")
    print(f"    forcing_energy    = {bridge_metrics['forcing_energy']:.8f}")
    print(f"    ablation_gain     = {bridge_metrics['ablation_gain']:.6f}")
    print("  split inverse:")
    for split in split_inverse["splits"]:
        print(
            "    "
            f"{split['split_key']}={split['holdout_value']}: "
            f"loss={split['holdout_loss']:.8f}, "
            f"ratio={split['full_over_best_ablation']:.6f}, "
            f"max_error={split['max_abs_error']:.6f}, "
            f"passed={split['passed']}"
        )
    print(f"    passed_all_splits = {split_inverse['passed_all_splits']}")
    print("  family inverse:")
    for split in family_inverse["splits"]:
        print(
            "    "
            f"{split['split_key']}={split['holdout_value']}: "
            f"generator_loss={split['generator_holdout_loss']:.8f}, "
            f"family_loss={split['family_holdout_loss']:.8f}, "
            f"scalar_loss={split['scalar_holdout_loss']:.8f}, "
            f"ratio={split['generator_over_scalar']:.6f}, "
            f"rank={split['generator_rank']}/{split['required_generator_rank']}, "
            f"passed={split['passed']}"
        )
    print(f"    passed_all_splits = {family_inverse['passed_all_splits']}")
    if "real_data_gate" in out:
        real_gate = out["real_data_gate"]
        print("  real data gate:")
        print(f"    cases              = {real_gate['case_count']}")
        print(f"    uses_case_graphs   = {real_gate['uses_case_graphs']}")
        readiness = real_gate["readiness"]
        print(f"    evaluated_splits   = {readiness['evaluated_split_count']}")
        print(f"    ready_for_claim    = {readiness['ready_for_claim']}")
        print(f"    next_actions       = {len(readiness['next_actions'])}")
        graph_gate = real_gate["graph_diagnostics"]
        print(f"    max_modal_radius   = {graph_gate['max_modal_radius']:.6f}")
        print(f"    min_stability_margin = {graph_gate['min_stability_margin']:.6f}")
        print(f"    max_coupling_energy = {graph_gate['max_coupling_energy']:.8f}")
        print(f"    passed_all_graphs  = {graph_gate['passed_all_graphs']}")
        print(
            "    passed_all_evaluated_splits = "
            f"{real_gate['passed_all_evaluated_splits']}"
        )
        print(f"    passed_real_gate   = {real_gate['passed_real_gate']}")
        for split in real_gate["splits"]:
            print(
                "    "
                f"{split['split_key']}={split['holdout_value']}: "
                f"loss={split['holdout_loss']:.8f}, "
                f"ratio={split['full_over_best_ablation']:.6f}, "
                f"rank={split['identifiability']['rank']}/"
                f"{split['identifiability']['required_rank']}, "
                f"passed={split['passed']}"
            )
    print(f"  passed = {out['passed_integrated_gate']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-real-schema",
        action="store_true",
        help="Print the accepted real transition JSON schema and exit.",
    )
    parser.add_argument(
        "--print-real-template",
        action="store_true",
        help="Print a parseable real transition JSON template and exit.",
    )
    parser.add_argument(
        "--real-transitions",
        type=Path,
        default=None,
        help="Optional JSON list of real P_n -> P_n+1 transition rows.",
    )
    parser.add_argument(
        "--check-real-transitions",
        type=Path,
        default=None,
        help="Load real transitions, print readiness JSON, and exit. Use '-' for stdin.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.print_real_schema:
        print(json.dumps(real_transition_schema(), indent=2, ensure_ascii=False))
        return
    if args.print_real_template:
        print(json.dumps(real_transition_template(), indent=2, ensure_ascii=False))
        return
    if args.check_real_transitions is not None:
        checked = check_real_transitions(args.check_real_transitions)
        print(json.dumps(checked, indent=2, ensure_ascii=False))
        return
    out = build_output(args.real_transitions)
    out_path = Path(__file__).with_name("brain_equation_integrated_gate_results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print_summary(out)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
