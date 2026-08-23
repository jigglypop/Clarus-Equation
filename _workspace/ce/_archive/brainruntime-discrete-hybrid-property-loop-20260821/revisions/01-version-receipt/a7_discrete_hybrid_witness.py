"""Frozen synthetic witness for the A7-H BrainRuntime discrete hybrid map.

No empirical response data are opened.  The float64 mirror validates the
declared branch equations and generalized derivatives; isolated float32
runtime arms bind those equations to the frozen Torch/Rust implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from reality_stone.clarus import runtime as runtime_module
from reality_stone.clarus.runtime import (
    BrainRuntime,
    BrainRuntimeConfig,
    ModuleLifecycle,
    RuntimeMode,
    _HAS_RUST_KERNEL,
    _LIFECYCLE_TO_CODE,
)


Q = 3
L = 2
FD_STEP = 2.0**-17
ONE_SIDED_STEPS = (2.0**-12, 2.0**-15, 2.0**-18)
GUARD_EPS = 2.0**-12
MIRROR_TOL = 2.0e-6
FACE_TOL = 3.0e-6
PERMUTATION_TOL = 1.0e-10
TORCH_TOL = 2.0e-6
BACKEND_TOL = 1.0e-5
STRICT_MARGIN_MIN = 1.0e-4
DELAY_ZERO_TOL = 1.0e-7
DELAY_EFFECT_MIN = 1.0e-4
FACE_EQUALITY_TOL = 2.0e-10

ALPHA = 0.0015
BETA = 0.008
STP_U = 0.5
ACTIVATION_DECAY = 0.18
ACTIVATION_GAIN = 0.82
REFRACTORY_DECAY = 0.12
REFRACTORY_GAIN = 0.24
MEMORY_DECAY = 0.01
ADAPTATION_DECAY = 0.005
ADAPTATION_COUPLING = 0.12
ADAPTATION_CLAMP = 2.0
EXTERNAL_GAIN = 0.45
GOAL_GAIN = 0.20
REPLAY_MIX = 0.08
REFRACTORY_SCALE = 0.35
BIT_LOWER = 0.10
BIT_UPPER = 0.30
ACTIVE_THRESHOLD = 0.22
IDLE_THRESHOLD = 0.08

EXPECTED_HASHES = {
    "runtime.py": "4d73dce1ad79dd51e4bcf757b7a97f0302c43add0478071fbff9197a831f901a",
    "kernel.rs": "0b26f3e99b5208181402898805d966b7564b25c4fbe567033e2da75c5a0d68c2",
    "tests/test_runtime_contracts.py": "b365f061fb74353a724988d6fc266c286e38296107007e1a7746dca46bfffefd",
}


@dataclass(frozen=True)
class ContinuousState:
    a: np.ndarray
    r: np.ndarray
    m: np.ndarray
    w: np.ndarray
    u: np.ndarray
    x: np.ndarray
    d: np.ndarray

    def copy(self) -> "ContinuousState":
        return ContinuousState(
            self.a.copy(),
            self.r.copy(),
            self.m.copy(),
            self.w.copy(),
            self.u.copy(),
            self.x.copy(),
            self.d.copy(),
        )

    def flat(self) -> np.ndarray:
        return np.concatenate((self.a, self.r, self.m, self.w, self.u, self.x, self.d.reshape(-1)))

    @staticmethod
    def from_flat(value: np.ndarray) -> "ContinuousState":
        z = np.asarray(value, dtype=np.float64).reshape((6 + L) * Q)
        blocks = [z[index * Q : (index + 1) * Q].copy() for index in range(6)]
        delay = z[6 * Q :].reshape(L, Q).copy()
        return ContinuousState(*blocks, delay)


@dataclass(frozen=True)
class Fixture:
    weight: np.ndarray
    state: ContinuousState
    q_prev: np.ndarray
    bit: np.ndarray
    counter: int
    external: np.ndarray
    goal: np.ndarray
    replay: np.ndarray


@dataclass(frozen=True)
class MirrorStep:
    state: ContinuousState
    recurrent: np.ndarray
    salience: np.ndarray
    bit: np.ndarray
    mask: np.ndarray
    counter: int
    raw: dict[str, np.ndarray]


def fixture() -> Fixture:
    return Fixture(
        weight=np.array(
            [[0.32, -0.18, 0.07], [0.11, 0.27, -0.21], [-0.15, 0.09, 0.24]],
            dtype=np.float64,
        ),
        state=ContinuousState(
            a=np.array([0.22, -0.31, 0.17]),
            r=np.array([0.06, 0.11, 0.04]),
            m=np.array([-0.08, 0.05, 0.12]),
            w=np.array([0.09, 0.13, 0.07]),
            u=np.array([0.41, 0.58, 0.36]),
            x=np.array([0.83, 0.71, 0.92]),
            d=np.array([[0.55, -0.24, 0.33], [-0.12, 0.44, 0.26]]),
        ),
        q_prev=np.array([1.0, 0.0, 1.0]),
        bit=np.array([0, 1, 0], dtype=np.uint8),
        counter=0,
        external=np.array([0.14, -0.09, 0.21]),
        goal=np.array([0.08, 0.03, -0.05]),
        replay=np.array([-0.04, 0.12, 0.06]),
    )


def normalized_error(left: np.ndarray, right: np.ndarray) -> float:
    lval = np.asarray(left, dtype=np.float64)
    rval = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(lval - rval) / max(1.0, float(np.linalg.norm(rval))))


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0.0 else "-Infinity"
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def select_mask(salience: np.ndarray, budget: int = 2) -> np.ndarray:
    values = np.asarray(salience, dtype=np.float64)
    eligible = np.flatnonzero(values >= ACTIVE_THRESHOLD)
    mask = np.zeros(Q, dtype=bool)
    if eligible.size == 0 or budget <= 0:
        return mask
    count = min(int(budget), int(eligible.size))
    order = np.argsort(-values[eligible], kind="mergesort")
    mask[eligible[order[:count]]] = True
    return mask


def mirror_step(
    fx: Fixture,
    state: ContinuousState,
    *,
    external: np.ndarray | None = None,
    q_prev: np.ndarray | None = None,
    bit: np.ndarray | None = None,
    counter: int | None = None,
    delay: bool = True,
) -> MirrorStep:
    e = fx.external if external is None else np.asarray(external, dtype=np.float64)
    qmask = fx.q_prev if q_prev is None else np.asarray(q_prev, dtype=np.float64)
    old_bit = fx.bit if bit is None else np.asarray(bit, dtype=np.uint8)
    kappa = fx.counter if counter is None else int(counter)

    raw_u = state.u - ALPHA * state.u + STP_U * (1.0 - state.u) * qmask
    next_u = np.clip(raw_u, 0.0, 1.0)
    raw_x = state.x + BETA * (1.0 - state.x) - state.u * state.x * qmask
    next_x = np.clip(raw_x, 0.0, 1.0)
    slot = kappa % L
    source = state.d[slot] if delay else state.a
    recurrent = fx.weight @ (qmask * next_u * next_x * source)
    drive = (
        recurrent
        + EXTERNAL_GAIN * e
        + GOAL_GAIN * fx.goal
        + REPLAY_MIX * fx.replay
        - REFRACTORY_SCALE * state.r
        - ADAPTATION_COUPLING * state.w
    )
    raw_a = (1.0 - ACTIVATION_DECAY) * state.a + ACTIVATION_GAIN * np.tanh(drive)
    next_a = np.clip(raw_a, -1.0, 1.0)
    next_r = (1.0 - REFRACTORY_DECAY) * state.r + REFRACTORY_GAIN * next_a**2
    next_m = (1.0 - MEMORY_DECAY) * state.m + MEMORY_DECAY * next_a
    raw_w = (1.0 - ADAPTATION_DECAY) * state.w + ADAPTATION_DECAY * next_a**2
    next_w = np.clip(raw_w, 0.0, ADAPTATION_CLAMP)
    next_d = state.d.copy()
    next_counter = kappa
    if delay:
        next_d[slot] = state.a
        next_counter += 1
    next_bit = old_bit.copy()
    next_bit[next_a >= BIT_UPPER] = 1
    next_bit[next_a <= BIT_LOWER] = 0
    salience = (
        np.abs(next_a)
        + 0.35 * np.abs(e)
        + 0.25 * np.abs(fx.replay)
        + 0.20 * np.abs(fx.goal)
        - 0.15 * next_r
    )
    state_out = ContinuousState(next_a, next_r, next_m, next_w, next_u, next_x, next_d)
    return MirrorStep(
        state=state_out,
        recurrent=recurrent,
        salience=salience,
        bit=next_bit,
        mask=select_mask(salience),
        counter=next_counter,
        raw={"u": raw_u, "x": raw_x, "a": raw_a, "w": raw_w, "drive": drive},
    )


def clip_direction(raw: np.ndarray, direction: np.ndarray, lower: float, upper: float) -> np.ndarray:
    raw_value = np.asarray(raw, dtype=np.float64)
    dvalue = np.asarray(direction, dtype=np.float64)
    result = np.zeros_like(dvalue)
    interior = (raw_value > lower + FACE_EQUALITY_TOL) & (raw_value < upper - FACE_EQUALITY_TOL)
    at_lower = np.abs(raw_value - lower) <= FACE_EQUALITY_TOL
    at_upper = np.abs(raw_value - upper) <= FACE_EQUALITY_TOL
    result[interior] = dvalue[interior]
    result[at_lower] = np.maximum(dvalue[at_lower], 0.0)
    result[at_upper] = np.minimum(dvalue[at_upper], 0.0)
    return result


def directional_step(
    fx: Fixture,
    state: ContinuousState,
    direction: ContinuousState,
    *,
    external: np.ndarray | None = None,
    q_prev: np.ndarray | None = None,
    counter: int | None = None,
    delay: bool = True,
) -> np.ndarray:
    base = mirror_step(
        fx,
        state,
        external=external,
        q_prev=q_prev,
        counter=counter,
        delay=delay,
    )
    qmask = fx.q_prev if q_prev is None else np.asarray(q_prev, dtype=np.float64)
    kappa = fx.counter if counter is None else int(counter)
    slot = kappa % L

    draw_u = (1.0 - ALPHA - STP_U * qmask) * direction.u
    du = clip_direction(base.raw["u"], draw_u, 0.0, 1.0)
    draw_x = (1.0 - BETA - state.u * qmask) * direction.x - state.x * qmask * direction.u
    dx = clip_direction(base.raw["x"], draw_x, 0.0, 1.0)
    source = state.d[slot] if delay else state.a
    dsource = direction.d[slot] if delay else direction.a
    drho = fx.weight @ (
        qmask
        * (
            du * base.state.x * source
            + base.state.u * dx * source
            + base.state.u * base.state.x * dsource
        )
    )
    dh = drho - REFRACTORY_SCALE * direction.r - ADAPTATION_COUPLING * direction.w
    sech_sq = 1.0 - np.tanh(base.raw["drive"]) ** 2
    draw_a = (1.0 - ACTIVATION_DECAY) * direction.a + ACTIVATION_GAIN * sech_sq * dh
    da = clip_direction(base.raw["a"], draw_a, -1.0, 1.0)
    dr = (1.0 - REFRACTORY_DECAY) * direction.r + 2.0 * REFRACTORY_GAIN * base.state.a * da
    dm = (1.0 - MEMORY_DECAY) * direction.m + MEMORY_DECAY * da
    draw_w = (1.0 - ADAPTATION_DECAY) * direction.w + 2.0 * ADAPTATION_DECAY * base.state.a * da
    dw = clip_direction(base.raw["w"], draw_w, 0.0, ADAPTATION_CLAMP)
    dd = direction.d.copy()
    if delay:
        dd[slot] = direction.a
    return ContinuousState(da, dr, dm, dw, du, dx, dd).flat()


def finite_difference_jacobian(fx: Fixture, state: ContinuousState) -> np.ndarray:
    base = state.flat()
    dimension = base.size
    result = np.zeros((dimension, dimension), dtype=np.float64)
    for column in range(dimension):
        plus = base.copy()
        minus = base.copy()
        plus[column] += FD_STEP
        minus[column] -= FD_STEP
        result[:, column] = (
            mirror_step(fx, ContinuousState.from_flat(plus)).state.flat()
            - mirror_step(fx, ContinuousState.from_flat(minus)).state.flat()
        ) / (2.0 * FD_STEP)
    return result


def analytic_jacobian(fx: Fixture, state: ContinuousState) -> np.ndarray:
    dimension = state.flat().size
    result = np.zeros((dimension, dimension), dtype=np.float64)
    for column in range(dimension):
        direction = np.zeros(dimension, dtype=np.float64)
        direction[column] = 1.0
        result[:, column] = directional_step(fx, state, ContinuousState.from_flat(direction))
    return result


def runtime_config(*, delay: bool) -> BrainRuntimeConfig:
    return BrainRuntimeConfig(
        dim=Q,
        active_ratio=2.0 / 3.0,
        active_threshold=ACTIVE_THRESHOLD,
        force_all_active_selection=False,
        noise_sigma=0.0,
        dale_law=False,
        axon_delay=delay,
        max_axon_delay=L,
        memory_capacity=1,
        memory_topk=1,
        hippocampal_encoding_enabled=False,
        f1_self_measure=False,
        stdp_enabled=False,
    )


def seed_runtime(
    fx: Fixture,
    state: ContinuousState,
    *,
    backend: str,
    delay: bool,
    weight: np.ndarray | None = None,
    q_prev: np.ndarray | None = None,
    bit: np.ndarray | None = None,
    counter: int | None = None,
) -> BrainRuntime:
    runtime = BrainRuntime(
        torch.tensor(fx.weight if weight is None else weight, dtype=torch.float32),
        config=runtime_config(delay=delay),
        backend=backend,
        device="cpu",
    )
    for name, value in (
        ("activation", state.a),
        ("refractory", state.r),
        ("memory_trace", state.m),
        ("adaptation", state.w),
        ("stp_u", state.u),
        ("stp_x", state.x),
    ):
        getattr(runtime, name).copy_(torch.tensor(value, dtype=torch.float32))
    runtime.bitfield.copy_(torch.tensor(fx.bit if bit is None else bit, dtype=torch.uint8))
    mask = fx.q_prev if q_prev is None else np.asarray(q_prev, dtype=np.float64)
    codes = [
        _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE]
        if bool(value)
        else _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT]
        for value in mask
    ]
    runtime.lifecycle.copy_(torch.tensor(codes, dtype=torch.int64))
    runtime.inactive_steps.zero_()
    runtime.set_goal(torch.tensor(fx.goal, dtype=torch.float32))
    runtime.step_index = 0
    if delay:
        assert runtime._delay_buffer is not None
        runtime._delay_buffer.copy_(torch.tensor(state.d, dtype=torch.float32))
        runtime._delay_idx = fx.counter if counter is None else int(counter)
    return runtime


def extracted_state(runtime: BrainRuntime, fallback_delay: np.ndarray) -> ContinuousState:
    delay = (
        runtime._delay_buffer.detach().cpu().numpy().astype(np.float64).copy()
        if runtime._delay_buffer is not None
        else np.asarray(fallback_delay, dtype=np.float64).copy()
    )
    return ContinuousState(
        runtime.activation.detach().cpu().numpy().astype(np.float64).copy(),
        runtime.refractory.detach().cpu().numpy().astype(np.float64).copy(),
        runtime.memory_trace.detach().cpu().numpy().astype(np.float64).copy(),
        runtime.adaptation.detach().cpu().numpy().astype(np.float64).copy(),
        runtime.stp_u.detach().cpu().numpy().astype(np.float64).copy(),
        runtime.stp_x.detach().cpu().numpy().astype(np.float64).copy(),
        delay,
    )


def torch_cell(
    fx: Fixture,
    state: ContinuousState,
    *,
    delay: bool,
    weight: np.ndarray | None = None,
    q_prev: np.ndarray | None = None,
    bit: np.ndarray | None = None,
    counter: int | None = None,
    external: np.ndarray | None = None,
) -> dict[str, Any]:
    runtime = seed_runtime(
        fx,
        state,
        backend="torch",
        delay=delay,
        weight=weight,
        q_prev=q_prev,
        bit=bit,
        counter=counter,
    )
    e = fx.external if external is None else np.asarray(external, dtype=np.float64)
    salience, recurrent, energy = runtime._step_torch(
        torch.tensor(e, dtype=torch.float32),
        torch.tensor(fx.replay, dtype=torch.float32),
        RuntimeMode.WAKE,
    )
    mask = runtime._select_active(salience, runtime._f1_effective_budget(RuntimeMode.WAKE))
    return {
        "runtime": runtime,
        "state": extracted_state(runtime, state.d),
        "bit": runtime.bitfield.detach().cpu().numpy().copy(),
        "salience": salience.detach().cpu().numpy().astype(np.float64),
        "recurrent": recurrent.detach().cpu().numpy().astype(np.float64),
        "mask": mask.detach().cpu().numpy().copy(),
        "counter": int(runtime._delay_idx),
        "energy": float(energy),
    }


def source_receipt(repo: Path) -> dict[str, Any]:
    paths = {
        "runtime.py": repo / "reality_stone/python/reality_stone/clarus/runtime.py",
        "kernel.rs": repo / "reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs",
        "tests/test_runtime_contracts.py": repo / "tests/test_runtime_contracts.py",
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    return {
        "expected": EXPECTED_HASHES,
        "actual": actual,
        "all_match": actual == EXPECTED_HASHES,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
    }


def test_h_a(fx: Fixture) -> dict[str, Any]:
    mirror = mirror_step(fx, fx.state)
    actual = torch_cell(fx, fx.state, delay=True)
    errors = {
        "continuous_max_abs": max_abs(actual["state"].flat(), mirror.state.flat()),
        "recurrent_max_abs": max_abs(actual["recurrent"], mirror.recurrent),
        "salience_max_abs": max_abs(actual["salience"], mirror.salience),
    }
    bit_exact = bool(np.array_equal(actual["bit"], mirror.bit))
    mask_exact = bool(np.array_equal(actual["mask"], mirror.mask))
    counter_exact = actual["counter"] == mirror.counter == 1
    passed = (
        max(errors.values()) <= TORCH_TOL
        and bit_exact
        and mask_exact
        and counter_exact
    )
    return {
        "pass": passed,
        "errors": errors,
        "bit_exact": bit_exact,
        "mask_exact": mask_exact,
        "counter_exact": counter_exact,
        "mirror_mask": mirror.mask,
        "runtime_mask": actual["mask"],
        "post_counter": actual["counter"],
    }


def clip_margins(step: MirrorStep) -> dict[str, float]:
    margins = {
        "u": float(np.min(np.minimum(step.raw["u"], 1.0 - step.raw["u"]))),
        "x": float(np.min(np.minimum(step.raw["x"], 1.0 - step.raw["x"]))),
        "a": float(np.min(np.minimum(step.raw["a"] + 1.0, 1.0 - step.raw["a"]))),
        "w": float(np.min(np.minimum(step.raw["w"], ADAPTATION_CLAMP - step.raw["w"]))),
    }
    return margins


def test_h_b(fx: Fixture) -> tuple[dict[str, Any], np.ndarray]:
    base_step = mirror_step(fx, fx.state)
    margins = clip_margins(base_step)
    analytic = analytic_jacobian(fx, fx.state)
    finite = finite_difference_jacobian(fx, fx.state)
    error = normalized_error(analytic, finite)
    dimension = analytic.shape[0]
    passed = (
        dimension == (6 + L) * Q
        and min(margins.values()) >= STRICT_MARGIN_MIN
        and error <= MIRROR_TOL
    )
    return {
        "pass": passed,
        "dimension": dimension,
        "clip_margins": margins,
        "minimum_clip_margin": min(margins.values()),
        "normalized_frobenius_error": error,
        "covered_blocks": ["a", "r", "m", "w", "u", "x", "d0", "d1"],
    }, analytic


def external_for_activation(fx: Fixture, state: ContinuousState, coordinate: int, target: float) -> np.ndarray:
    preliminary = mirror_step(fx, state, external=np.zeros(Q))
    tanh_target = (target - (1.0 - ACTIVATION_DECAY) * state.a[coordinate]) / ACTIVATION_GAIN
    if not (-1.0 < tanh_target < 1.0):
        raise ValueError("requested activation face is not reachable")
    desired_drive = math.atanh(float(tanh_target))
    other = (
        preliminary.recurrent[coordinate]
        + GOAL_GAIN * fx.goal[coordinate]
        + REPLAY_MIX * fx.replay[coordinate]
        - REFRACTORY_SCALE * state.r[coordinate]
        - ADAPTATION_COUPLING * state.w[coordinate]
    )
    external = fx.external.copy()
    external[coordinate] = (desired_drive - other) / EXTERNAL_GAIN
    return external


def face_case(
    fx: Fixture,
    *,
    name: str,
    state: ContinuousState,
    external: np.ndarray,
    raw_name: str,
    coordinate: int,
    boundary: float,
    output_coordinate: int,
    directions: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    base_step = mirror_step(fx, state, external=external)
    boundary_error = abs(float(base_step.raw[raw_name][coordinate]) - boundary)
    direction_rows: list[dict[str, Any]] = []
    for label, flat_direction in directions:
        direction_state = ContinuousState.from_flat(flat_direction)
        analytic = directional_step(fx, state, direction_state, external=external)
        errors: list[float] = []
        for epsilon in ONE_SIDED_STEPS:
            perturbed = ContinuousState.from_flat(state.flat() + epsilon * flat_direction)
            finite = (
                mirror_step(fx, perturbed, external=external).state.flat()
                - base_step.state.flat()
            ) / epsilon
            errors.append(normalized_error(analytic, finite))
        component = float(analytic[output_coordinate])
        qualitative = abs(component) <= 1.0e-9 if label == "outward" else abs(component) >= 1.0e-6
        direction_rows.append(
            {
                "label": label,
                "errors": errors,
                "final_error": errors[-1],
                "analytic_face_component": component,
                "qualitative_pass": qualitative,
                "pass": errors[-1] <= FACE_TOL and qualitative,
            }
        )
    return {
        "name": name,
        "raw_name": raw_name,
        "coordinate": coordinate,
        "boundary": boundary,
        "boundary_error": boundary_error,
        "directions": direction_rows,
        "pass": boundary_error <= FACE_EQUALITY_TOL and all(row["pass"] for row in direction_rows),
    }


def basis_direction(block: int, coordinate: int, sign: float) -> np.ndarray:
    result = np.zeros((6 + L) * Q, dtype=np.float64)
    result[block * Q + coordinate] = sign
    return result


def scalar_face_diagnostic(boundary: float, lower: float, upper: float) -> dict[str, Any]:
    rows = []
    for label, direction in (("inward", -1.0), ("outward", 1.0)):
        analytic = min(direction, 0.0)
        errors = []
        for epsilon in ONE_SIDED_STEPS:
            finite = (np.clip(boundary + epsilon * direction, lower, upper) - boundary) / epsilon
            errors.append(abs(float(finite - analytic)))
        rows.append({"label": label, "final_error": errors[-1], "errors": errors})
    return {"boundary": boundary, "rows": rows, "pass": max(row["final_error"] for row in rows) <= FACE_TOL}


def test_h_c(fx: Fixture) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    state = fx.state.copy()
    state.u[1] = 0.0
    cases.append(
        face_case(
            fx,
            name="stp_u_lower",
            state=state,
            external=fx.external,
            raw_name="u",
            coordinate=1,
            boundary=0.0,
            output_coordinate=4 * Q + 1,
            directions=[("inward", basis_direction(4, 1, 1.0)), ("outward", basis_direction(4, 1, -1.0))],
        )
    )

    state = fx.state.copy()
    state.x[1] = 1.0
    cases.append(
        face_case(
            fx,
            name="stp_x_upper",
            state=state,
            external=fx.external,
            raw_name="x",
            coordinate=1,
            boundary=1.0,
            output_coordinate=5 * Q + 1,
            directions=[("inward", basis_direction(5, 1, -1.0)), ("outward", basis_direction(5, 1, 1.0))],
        )
    )

    state = fx.state.copy()
    state.u[0] = 1.0
    state.x[0] = 1.0
    cases.append(
        face_case(
            fx,
            name="stp_x_lower",
            state=state,
            external=fx.external,
            raw_name="x",
            coordinate=0,
            boundary=0.0,
            output_coordinate=5 * Q,
            directions=[("inward", basis_direction(5, 0, -1.0)), ("outward", basis_direction(5, 0, 1.0))],
        )
    )

    for name, coordinate, target, inward_sign, outward_sign in (
        ("activation_upper", 0, 1.0, 1.0, -1.0),
        ("activation_lower", 1, -1.0, -1.0, 1.0),
    ):
        state = fx.state.copy()
        external = external_for_activation(fx, state, coordinate, target)
        cases.append(
            face_case(
                fx,
                name=name,
                state=state,
                external=external,
                raw_name="a",
                coordinate=coordinate,
                boundary=target,
                output_coordinate=coordinate,
                directions=[
                    ("inward", basis_direction(1, coordinate, inward_sign)),
                    ("outward", basis_direction(1, coordinate, outward_sign)),
                ],
            )
        )

    state = fx.state.copy()
    state.w[2] = 0.0
    external = external_for_activation(fx, state, 2, 0.0)
    cases.append(
        face_case(
            fx,
            name="adaptation_lower",
            state=state,
            external=external,
            raw_name="w",
            coordinate=2,
            boundary=0.0,
            output_coordinate=3 * Q + 2,
            directions=[("inward", basis_direction(3, 2, 1.0)), ("outward", basis_direction(3, 2, -1.0))],
        )
    )

    primitive = {
        "stp_u_unreachable_upper": scalar_face_diagnostic(1.0, 0.0, 1.0),
        "adaptation_unreachable_upper": scalar_face_diagnostic(2.0, 0.0, 2.0),
    }
    passed = all(case["pass"] for case in cases) and all(row["pass"] for row in primitive.values())
    return {"pass": passed, "runtime_reachable_faces": cases, "primitive_only": primitive}


def actual_bit_arm(fx: Fixture, coordinate: int, target: float, initial_bit: int, sign: float) -> dict[str, Any]:
    state = fx.state.copy()
    external = external_for_activation(fx, state, coordinate, target)
    external[coordinate] += sign * GUARD_EPS
    bit = fx.bit.copy()
    bit[coordinate] = initial_bit
    row = torch_cell(fx, state, delay=True, bit=bit, external=external)
    return {"activation": float(row["state"].a[coordinate]), "bit": int(row["bit"][coordinate])}


def selection_arm(salience: np.ndarray) -> np.ndarray:
    fx = fixture()
    runtime = seed_runtime(fx, fx.state, backend="torch", delay=False)
    tensor = torch.tensor(salience, dtype=torch.float32)
    return runtime._select_active(tensor, 2).detach().cpu().numpy().copy()


def lifecycle_arm(salience: np.ndarray) -> np.ndarray:
    fx = fixture()
    runtime = seed_runtime(fx, fx.state, backend="torch", delay=False, q_prev=np.zeros(Q))
    runtime.inactive_steps.zero_()
    runtime._update_lifecycle(
        torch.tensor(salience, dtype=torch.float32),
        torch.zeros(Q, dtype=torch.bool),
    )
    return runtime.lifecycle.detach().cpu().numpy().copy()


def test_h_d(fx: Fixture) -> dict[str, Any]:
    upper_minus = actual_bit_arm(fx, 0, BIT_UPPER, 0, -1.0)
    upper_plus = actual_bit_arm(fx, 0, BIT_UPPER, 0, 1.0)
    lower_minus = actual_bit_arm(fx, 1, BIT_LOWER, 1, -1.0)
    lower_plus = actual_bit_arm(fx, 1, BIT_LOWER, 1, 1.0)

    delta = 2.0**-8
    eligibility_minus = selection_arm(np.array([ACTIVE_THRESHOLD - delta, ACTIVE_THRESHOLD - GUARD_EPS, ACTIVE_THRESHOLD + 2.0 * delta]))
    eligibility_plus = selection_arm(np.array([ACTIVE_THRESHOLD - delta, ACTIVE_THRESHOLD + GUARD_EPS, ACTIVE_THRESHOLD + 2.0 * delta]))
    kth_plus = selection_arm(np.array([0.70, 0.50 + GUARD_EPS, 0.50 - GUARD_EPS]))
    kth_minus = selection_arm(np.array([0.70, 0.50 - GUARD_EPS, 0.50 + GUARD_EPS]))
    tie = selection_arm(np.array([0.70, 0.50, 0.50]))

    lifecycle_minus = lifecycle_arm(np.array([IDLE_THRESHOLD - GUARD_EPS, 0.30, 0.30]))
    lifecycle_plus = lifecycle_arm(np.array([IDLE_THRESHOLD + GUARD_EPS, 0.30, 0.30]))
    expected_dormant = _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT]
    expected_idle = _LIFECYCLE_TO_CODE[ModuleLifecycle.IDLE]

    bit_pass = (
        upper_minus["bit"] == 0
        and upper_plus["bit"] == 1
        and lower_minus["bit"] == 0
        and lower_plus["bit"] == 1
    )
    selection_pass = (
        not np.array_equal(eligibility_minus, eligibility_plus)
        and not np.array_equal(kth_plus, kth_minus)
    )
    lifecycle_pass = (
        int(lifecycle_minus[0]) == expected_dormant
        and int(lifecycle_plus[0]) == expected_idle
    )
    return {
        "pass": bit_pass and selection_pass and lifecycle_pass,
        "derivative_status": "UNDEFINED_DISCRETE_EVENT",
        "guard_epsilon": GUARD_EPS,
        "bit_upper": {"minus": upper_minus, "plus": upper_plus, "crossed": bit_pass},
        "bit_lower": {"minus": lower_minus, "plus": lower_plus, "crossed": bit_pass},
        "eligibility": {"minus_mask": eligibility_minus, "plus_mask": eligibility_plus},
        "topk_boundary": {"minus_mask": kth_minus, "plus_mask": kth_plus},
        "exact_tie": {"mask_receipt": tie, "status": "TIE_POLICY_UNSPECIFIED"},
        "lifecycle_idle": {"minus_codes": lifecycle_minus, "plus_codes": lifecycle_plus},
    }


def test_h_e(fx: Fixture) -> dict[str, Any]:
    zero_state = fx.state.copy()
    zero_state.d[:] = 0.0
    runtime = seed_runtime(fx, zero_state, backend="torch", delay=True, q_prev=np.ones(Q))
    rows = []
    for call in range(3):
        before_buffer = runtime._delay_buffer.detach().cpu().numpy().astype(np.float64).copy()
        before_activation = runtime.activation.detach().cpu().numpy().astype(np.float64).copy()
        slot = int(runtime._delay_idx % L)
        _, recurrent, _ = runtime._step_torch(
            torch.tensor(fx.external, dtype=torch.float32),
            torch.tensor(fx.replay, dtype=torch.float32),
            RuntimeMode.WAKE,
        )
        after_buffer = runtime._delay_buffer.detach().cpu().numpy().astype(np.float64).copy()
        rows.append(
            {
                "call": call,
                "read_slot": slot,
                "read_value": before_buffer[slot],
                "written_old_activation": before_activation,
                "post_slot_value": after_buffer[slot],
                "post_counter": int(runtime._delay_idx),
                "recurrent_norm": float(torch.linalg.vector_norm(recurrent).item()),
            }
        )
    ring_pass = (
        rows[0]["post_counter"] == 1
        and rows[1]["post_counter"] == 2
        and rows[2]["post_counter"] == 3
        and rows[0]["read_slot"] == 0
        and rows[1]["read_slot"] == 1
        and rows[2]["read_slot"] == 0
        and rows[0]["recurrent_norm"] <= DELAY_ZERO_TOL
        and rows[1]["recurrent_norm"] <= DELAY_ZERO_TOL
        and rows[2]["recurrent_norm"] >= DELAY_EFFECT_MIN
        and max_abs(rows[0]["post_slot_value"], rows[0]["written_old_activation"]) <= TORCH_TOL
    )

    post = torch_cell(fx, fx.state, delay=False)
    post_state = post["state"]
    salience_left = np.array([0.70, 0.50 + GUARD_EPS, 0.50 - GUARD_EPS])
    salience_right = np.array([0.70, 0.50 - GUARD_EPS, 0.50 + GUARD_EPS])
    mask_left = selection_arm(salience_left)
    mask_right = selection_arm(salience_right)
    left = seed_runtime(fx, post_state, backend="torch", delay=False, q_prev=np.zeros(Q), bit=post["bit"])
    right = seed_runtime(fx, post_state, backend="torch", delay=False, q_prev=np.zeros(Q), bit=post["bit"])
    left._update_lifecycle(torch.tensor(salience_left, dtype=torch.float32), torch.tensor(mask_left))
    right._update_lifecycle(torch.tensor(salience_right, dtype=torch.float32), torch.tensor(mask_right))
    same_tick_error = max_abs(left.activation.detach().numpy(), right.activation.detach().numpy())
    _, recurrent_left, _ = left._step_torch(
        torch.tensor(fx.external, dtype=torch.float32), torch.tensor(fx.replay, dtype=torch.float32), RuntimeMode.WAKE
    )
    _, recurrent_right, _ = right._step_torch(
        torch.tensor(fx.external, dtype=torch.float32), torch.tensor(fx.replay, dtype=torch.float32), RuntimeMode.WAKE
    )
    next_recurrent_difference = float(torch.linalg.vector_norm(recurrent_left - recurrent_right).item())
    lag_pass = (
        same_tick_error <= DELAY_ZERO_TOL
        and not np.array_equal(mask_left, mask_right)
        and next_recurrent_difference >= DELAY_EFFECT_MIN
    )
    return {
        "pass": ring_pass and lag_pass,
        "ring_pass": ring_pass,
        "ring_calls": rows,
        "lifecycle_lag": {
            "pass": lag_pass,
            "left_mask": mask_left,
            "right_mask": mask_right,
            "same_tick_activation_error": same_tick_error,
            "next_tick_recurrent_difference_norm": next_recurrent_difference,
        },
    }


def permute_state(state: ContinuousState, permutation: np.ndarray) -> ContinuousState:
    return ContinuousState(
        state.a[permutation],
        state.r[permutation],
        state.m[permutation],
        state.w[permutation],
        state.u[permutation],
        state.x[permutation],
        state.d[:, permutation],
    )


def permuted_fixture(fx: Fixture, permutation: np.ndarray) -> Fixture:
    return Fixture(
        weight=fx.weight[np.ix_(permutation, permutation)],
        state=permute_state(fx.state, permutation),
        q_prev=fx.q_prev[permutation],
        bit=fx.bit[permutation],
        counter=fx.counter,
        external=fx.external[permutation],
        goal=fx.goal[permutation],
        replay=fx.replay[permutation],
    )


def test_h_f(fx: Fixture, jacobian: np.ndarray) -> dict[str, Any]:
    permutation = np.array([2, 0, 1], dtype=int)
    pfx = permuted_fixture(fx, permutation)
    base = mirror_step(fx, fx.state)
    transformed = mirror_step(pfx, pfx.state)
    expected_state = permute_state(base.state, permutation)
    mirror_state_error = normalized_error(transformed.state.flat(), expected_state.flat())
    mirror_recurrent_error = normalized_error(transformed.recurrent, base.recurrent[permutation])
    mirror_salience_error = normalized_error(transformed.salience, base.salience[permutation])
    mirror_bit_exact = bool(np.array_equal(transformed.bit, base.bit[permutation]))
    mirror_mask_exact = bool(np.array_equal(transformed.mask, base.mask[permutation]))

    index = np.concatenate([block * Q + permutation for block in range(6 + L)])
    lift = np.eye((6 + L) * Q, dtype=np.float64)[index]
    transformed_jacobian = analytic_jacobian(pfx, pfx.state)
    expected_jacobian = lift @ jacobian @ lift.T
    jacobian_error = normalized_error(transformed_jacobian, expected_jacobian)

    pairwise = [abs(float(base.salience[i] - base.salience[j])) for i in range(Q) for j in range(i + 1, Q)]
    salience_gap = min(pairwise)
    threshold_margin = float(np.min(np.abs(base.salience - ACTIVE_THRESHOLD)))

    actual = torch_cell(fx, fx.state, delay=True)
    actual_p = torch_cell(pfx, pfx.state, delay=True)
    actual_expected = permute_state(actual["state"], permutation)
    torch_state_error = max_abs(actual_p["state"].flat(), actual_expected.flat())
    torch_recurrent_error = max_abs(actual_p["recurrent"], actual["recurrent"][permutation])
    torch_bit_exact = bool(np.array_equal(actual_p["bit"], actual["bit"][permutation]))
    torch_mask_exact = bool(np.array_equal(actual_p["mask"], actual["mask"][permutation]))

    passed = (
        max(mirror_state_error, mirror_recurrent_error, mirror_salience_error, jacobian_error) <= PERMUTATION_TOL
        and mirror_bit_exact
        and mirror_mask_exact
        and salience_gap >= STRICT_MARGIN_MIN
        and threshold_margin >= STRICT_MARGIN_MIN
        and max(torch_state_error, torch_recurrent_error) <= TORCH_TOL
        and torch_bit_exact
        and torch_mask_exact
    )
    return {
        "pass": passed,
        "permutation": permutation,
        "mirror": {
            "state_error": mirror_state_error,
            "recurrent_error": mirror_recurrent_error,
            "salience_error": mirror_salience_error,
            "jacobian_error": jacobian_error,
            "bit_exact": mirror_bit_exact,
            "mask_exact": mirror_mask_exact,
        },
        "no_tie_receipt": {"minimum_pairwise_gap": salience_gap, "eligibility_margin": threshold_margin},
        "torch": {
            "state_max_abs": torch_state_error,
            "recurrent_max_abs": torch_recurrent_error,
            "bit_exact": torch_bit_exact,
            "mask_exact": torch_mask_exact,
        },
    }


def rust_cell(fx: Fixture, state: ContinuousState, *, delay: bool) -> dict[str, Any]:
    runtime = seed_runtime(fx, state, backend="rust", delay=delay)
    initial_buffer = (
        runtime._delay_buffer.detach().cpu().numpy().astype(np.float64).copy()
        if runtime._delay_buffer is not None
        else None
    )
    active_count, energy = runtime._step_rust(
        torch.tensor(fx.external, dtype=torch.float32),
        torch.tensor(fx.replay, dtype=torch.float32),
        RuntimeMode.WAKE,
    )
    return {
        "runtime": runtime,
        "state": extracted_state(runtime, state.d),
        "bit": runtime.bitfield.detach().cpu().numpy().copy(),
        "counter": int(runtime._delay_idx),
        "energy": float(energy),
        "private_active_count_receipt_only": int(active_count),
        "initial_buffer": initial_buffer,
    }


def test_h_g(fx: Fixture, repo: Path) -> dict[str, Any]:
    module_name = getattr(getattr(runtime_module, "_rust_brain_step", None), "__module__", None)
    extension_path = None
    if module_name:
        try:
            extension_path = getattr(importlib.import_module(module_name), "__file__", None)
        except ImportError:
            extension_path = None
    availability = {
        "has_rust_kernel": bool(_HAS_RUST_KERNEL),
        "callable_module": module_name,
        "extension_path": extension_path,
    }
    if not _HAS_RUST_KERNEL:
        return {
            "pass": False,
            "status": "BLOCKED_KERNEL_UNAVAILABLE",
            "availability": availability,
            "no_delay": None,
            "delay": None,
        }

    torch_no_delay = torch_cell(fx, fx.state, delay=False)
    rust_no_delay = rust_cell(fx, fx.state, delay=False)
    no_delay_state_error = max_abs(torch_no_delay["state"].flat()[: 6 * Q], rust_no_delay["state"].flat()[: 6 * Q])
    no_delay_bit_exact = bool(np.array_equal(torch_no_delay["bit"], rust_no_delay["bit"]))
    no_delay_energy_error = abs(torch_no_delay["energy"] - rust_no_delay["energy"])
    no_delay_pass = (
        no_delay_state_error <= BACKEND_TOL
        and no_delay_bit_exact
        and no_delay_energy_error <= BACKEND_TOL
    )

    torch_delay = torch_cell(fx, fx.state, delay=True)
    rust_delay = rust_cell(fx, fx.state, delay=True)
    activation_error = max_abs(torch_delay["state"].a, rust_delay["state"].a)
    torch_slot_written = max_abs(torch_delay["state"].d[0], fx.state.a) <= TORCH_TOL
    rust_buffer_unchanged = max_abs(rust_delay["state"].d, fx.state.d) <= TORCH_TOL
    expected_delay_mismatch = (
        torch_delay["counter"] == 1
        and rust_delay["counter"] == 0
        and torch_slot_written
        and rust_buffer_unchanged
        and activation_error >= DELAY_EFFECT_MIN
    )
    kernel_text = (
        repo / "reality_stone/python/reality_stone/clarus/core/src/engine/kernel.rs"
    ).read_text(encoding="utf-8")
    signature_receipt = {
        "contains_delay_buffer_identifier": "delay_buffer" in kernel_text,
        "contains_delay_index_identifier": "delay_idx" in kernel_text,
    }
    return {
        "pass": no_delay_pass and expected_delay_mismatch,
        "status": "NO_DELAY_PASS / DELAY_PARITY_FAIL_EXPECTED",
        "availability": availability,
        "no_delay": {
            "pass": no_delay_pass,
            "state_max_abs": no_delay_state_error,
            "bit_exact": no_delay_bit_exact,
            "energy_abs_error": no_delay_energy_error,
            "rust_private_active_count_receipt_only": rust_no_delay["private_active_count_receipt_only"],
        },
        "delay": {
            "pass_expected_mismatch": expected_delay_mismatch,
            "activation_max_abs": activation_error,
            "torch_post_counter": torch_delay["counter"],
            "rust_post_counter": rust_delay["counter"],
            "torch_slot_written": torch_slot_written,
            "rust_buffer_unchanged": rust_buffer_unchanged,
            "source_signature": signature_receipt,
        },
    }


def constants_receipt() -> dict[str, Any]:
    cfg = runtime_config(delay=True)
    actual = {
        "alpha": float(runtime_module.STP_TAU_FAC_INV),
        "beta": float(runtime_module.STP_TAU_REC),
        "stp_u": float(runtime_module.STP_U_BASE),
        "activation_decay": cfg.activation_decay(RuntimeMode.WAKE),
        "activation_gain": cfg.activation_gain(RuntimeMode.WAKE),
        "refractory_decay": cfg.refractory_decay(RuntimeMode.WAKE),
        "refractory_gain": cfg.refractory_gain(RuntimeMode.WAKE),
        "memory_decay": float(runtime_module.MEMORY_TRACE_DECAY),
        "adaptation_decay": float(runtime_module.ADAPTATION_DECAY),
        "adaptation_coupling": float(runtime_module.ADAPTATION_COUPLING),
        "adaptation_clamp": float(runtime_module.ADAPTATION_CLAMP),
        "external_gain": cfg.external_gain,
        "goal_gain": cfg.goal_gain,
        "replay_mix": cfg.replay_mix(RuntimeMode.WAKE),
        "refractory_scale": cfg.refractory_scale,
        "bit_lower": cfg.bit_lower_threshold,
        "bit_upper": cfg.bit_upper_threshold,
        "active_threshold": cfg.active_threshold,
        "max_axon_delay": cfg.max_axon_delay,
        "active_budget": cfg.energy_budget(RuntimeMode.WAKE),
        "dale_law": cfg.dale_law,
        "force_all_active_selection": cfg.force_all_active_selection,
    }
    expected = {
        "alpha": ALPHA,
        "beta": BETA,
        "stp_u": STP_U,
        "activation_decay": ACTIVATION_DECAY,
        "activation_gain": ACTIVATION_GAIN,
        "refractory_decay": REFRACTORY_DECAY,
        "refractory_gain": REFRACTORY_GAIN,
        "memory_decay": MEMORY_DECAY,
        "adaptation_decay": ADAPTATION_DECAY,
        "adaptation_coupling": ADAPTATION_COUPLING,
        "adaptation_clamp": ADAPTATION_CLAMP,
        "external_gain": EXTERNAL_GAIN,
        "goal_gain": GOAL_GAIN,
        "replay_mix": REPLAY_MIX,
        "refractory_scale": REFRACTORY_SCALE,
        "bit_lower": BIT_LOWER,
        "bit_upper": BIT_UPPER,
        "active_threshold": ACTIVE_THRESHOLD,
        "max_axon_delay": L,
        "active_budget": 2,
        "dale_law": False,
        "force_all_active_selection": False,
    }
    return {"expected": expected, "actual": actual, "all_match": actual == expected}


def thresholds_receipt() -> dict[str, Any]:
    return {
        "central_difference_step": FD_STEP,
        "one_sided_steps": ONE_SIDED_STEPS,
        "guard_epsilon": GUARD_EPS,
        "mirror_error_max": MIRROR_TOL,
        "face_error_max": FACE_TOL,
        "float64_permutation_max": PERMUTATION_TOL,
        "torch_max_abs": TORCH_TOL,
        "backend_max_abs": BACKEND_TOL,
        "strict_margin_min": STRICT_MARGIN_MIN,
        "delay_zero_max": DELAY_ZERO_TOL,
        "delay_effect_min": DELAY_EFFECT_MIN,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("a7_discrete_hybrid_result.json"),
    )
    args = parser.parse_args()
    source_path = Path(__file__).resolve()
    run_dir = source_path.parent.parent
    repo = source_path.parents[4]
    contract_path = run_dir / "00-contract.md"
    fx = fixture()

    sources = source_receipt(repo)
    constants = constants_receipt()
    h_a = test_h_a(fx)
    h_b, jacobian = test_h_b(fx)
    h_c = test_h_c(fx)
    h_d = test_h_d(fx)
    h_e = test_h_e(fx)
    h_f = test_h_f(fx, jacobian)
    h_g = test_h_g(fx, repo)
    passed = all(
        (
            sources["all_match"],
            constants["all_match"],
            h_a["pass"],
            h_b["pass"],
            h_c["pass"],
            h_d["pass"],
            h_e["pass"],
            h_f["pass"],
            h_g["pass"],
        )
    )
    try:
        package_version = importlib.metadata.version("reality-stone")
    except importlib.metadata.PackageNotFoundError:
        package_version = "UNAVAILABLE"
    result = {
        "status": "PROPERTY_PASS" if passed else "PROPERTY_FAIL",
        "claim_ceiling": (
            "DISCRETE_HYBRID_SPEC_PASS / RUNTIME_DELAY_PARITY_BLOCKED / "
            "HETEROGENEOUS_THRESHOLD_RUNTIME_UNIMPLEMENTED / EMPIRICAL_UNTESTED"
        ),
        "empirical_assets_opened": False,
        "learning_validated": False,
        "agi_validated": False,
        "cortical_folding_validated": False,
        "thresholds": thresholds_receipt(),
        "fixture": {
            "q": Q,
            "delay_length": L,
            "lifecycle": ["ACTIVE", "DORMANT", "ACTIVE"],
            "q_prev": fx.q_prev,
            "counter": fx.counter,
            "slot": fx.counter % L,
        },
        "provenance": {
            "source_path": str(source_path),
            "source_sha256": sha256(source_path),
            "contract_path": str(contract_path.resolve()),
            "contract_sha256": sha256(contract_path),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "reality_stone_version": package_version,
            "platform": platform.platform(),
            "sources": sources,
            "constants": constants,
        },
        "H_A_torch_mirror": h_a,
        "H_B_interior_jacobian": h_b,
        "H_C_clip_faces": h_c,
        "H_D_discrete_guards": h_d,
        "H_E_delay_and_lifecycle": h_e,
        "H_F_permutation": h_f,
        "H_G_backend_boundary": h_g,
    }
    safe = json_safe(result)
    args.output.write_text(
        json.dumps(safe, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(safe, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
