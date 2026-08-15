"""Development and sealed-confirmation evaluator for the V17 signed-cue task.

The evaluator deliberately computes terminal quadratic costs, decisions, and
analytic reference matrices without calling the production readout helpers.
Confirmation seeds are reachable only after an exclusive opening receipt has
been created.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

from reality_stone.clarus import homogeneous_signed_cue as signed_cue_module
from reality_stone.clarus.homogeneous_signed_cue import HomogeneousSignedCue


DEVELOPMENT_SEEDS = range(1_719_000, 1_719_064)
CONFIRMATION_SEEDS = range(1_720_000, 1_720_256)
ENSEMBLE_SIZES = (1, 2, 4, 8, 16, 64)
DIMENSION = 3
LIFTED_DIMENSION = DIMENSION + 1
ETA = 1.0
OBSERVED_COST = 4.0
MARGIN_THRESHOLD = 1.999999999
CHART_DEFECT_THRESHOLD = 1.0e-10

REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_RELATIVE = PurePosixPath("_workspace/ce/agi-v17-metric-delayed-credit-20260813")
EVALUATOR_RELATIVE = RUN_RELATIVE / "artifacts/run_v17_benchmark.py"
MANIFEST_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-manifest.json"
DEVELOPMENT_RESULT_RELATIVE = RUN_RELATIVE / "artifacts/development-results.json"
RECEIPT_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-opened.json"
RESULT_RELATIVE = RUN_RELATIVE / "artifacts/confirmation-results.json"
PRODUCTION_RELATIVE = PurePosixPath(
    "reality_stone/python/reality_stone/clarus/homogeneous_signed_cue.py"
)
PUBLIC_EXPORT_RELATIVE = PurePosixPath("reality_stone/python/reality_stone/clarus/__init__.py")
REQUIRED_MANIFEST_PATHS = frozenset(
    {
        str(RUN_RELATIVE / "00-contract.md"),
        str(PRODUCTION_RELATIVE),
        str(PUBLIC_EXPORT_RELATIVE),
        str(EVALUATOR_RELATIVE),
        str(DEVELOPMENT_RESULT_RELATIVE),
    }
)
IMPORTED_PRODUCTION_PATH = Path(signed_cue_module.__file__).resolve(strict=True)

FIXED_PROTOCOL: dict[str, Any] = {
    "dimension": DIMENSION,
    "eta": ETA,
    "observed_cost": OBSERVED_COST,
    "cue_embedding": "z_s=(s*u,1)",
    "action_embedding": "y_a=(a*u,-1)",
    "chart_lift": "A=diag(J,1)",
    "ensemble_sizes": list(ENSEMBLE_SIZES),
    "margin_threshold": MARGIN_THRESHOLD,
    "chart_defect_threshold": CHART_DEFECT_THRESHOLD,
    "strict_action_law": {"-1": 0.5, "+1": 0.5},
}


@dataclass(frozen=True)
class _ConfirmationAccess:
    """Opaque capability issued only by the exclusive receipt operation."""

    root: Path
    receipt: Path
    result: Path
    manifest: Path
    manifest_sha256: str


_ACTIVE_CONFIRMATION_ACCESS: _ConfirmationAccess | None = None
_ACTIVE_CONFIRMATION_PREFLIGHT: object | None = None


def _json_without_duplicate_keys(text: str, *, source: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {source}: {key!r}")
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=unique_object)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_root(root: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    if resolved_root != REPO_ROOT:
        raise ValueError(f"root must be evaluator repository root: {REPO_ROOT}")
    expected_evaluator = resolved_root.joinpath(*EVALUATOR_RELATIVE.parts).resolve(strict=True)
    if Path(__file__).resolve(strict=True) != expected_evaluator:
        raise ValueError("executed evaluator is not the repository evaluator")
    expected_production = resolved_root.joinpath(*PRODUCTION_RELATIVE.parts).resolve(strict=True)
    if IMPORTED_PRODUCTION_PATH != expected_production:
        raise ValueError("imported production module is not the repository module")
    return resolved_root


def _canonical_root_path(root: Path, relative: PurePosixPath) -> Path:
    resolved_root = _validated_root(root)
    candidate = resolved_root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):  # pragma: no cover - fixed paths
        raise ValueError(f"path escapes repository root: {relative}")
    return candidate


def _manifest_target(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"invalid manifest path: {relative!r}")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
        raise ValueError(f"invalid manifest path: {relative!r}")
    return _canonical_root_path(root, parsed)


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, str]:
    canonical_manifest = _canonical_root_path(root, MANIFEST_RELATIVE)
    if manifest_path.resolve(strict=True) != canonical_manifest:
        raise ValueError(f"manifest must be {MANIFEST_RELATIVE}")
    manifest = _json_without_duplicate_keys(
        canonical_manifest.read_text(encoding="utf-8"),
        source=str(MANIFEST_RELATIVE),
    )
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("manifest must be a nonempty path-to-SHA256 object")

    # Reject dangerous syntax before set comparison, but do not resolve an
    # ordinary unknown path until after the exact registered-key check.
    for relative in manifest:
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError(f"invalid manifest path: {relative!r}")
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != relative:
            raise ValueError(f"invalid manifest path: {relative!r}")
    actual_paths = set(manifest)
    if actual_paths != REQUIRED_MANIFEST_PATHS:
        missing = sorted(REQUIRED_MANIFEST_PATHS.difference(actual_paths))
        extra = sorted(actual_paths.difference(REQUIRED_MANIFEST_PATHS))
        raise ValueError(f"manifest paths must be exact; missing={missing}, extra={extra}")
    for relative, expected in manifest.items():
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"invalid SHA-256 for manifest path: {relative}")
        if sha256(_manifest_target(root, relative)) != expected:
            raise ValueError(f"manifest mismatch: {relative}")
    return {str(key): str(value) for key, value in sorted(manifest.items())}


def _normalised_protocol(value: Any) -> Any:
    """Round-trip fixed protocol constants through JSON's sealed data domain."""

    return json.loads(json.dumps(value, sort_keys=True))


def verify_development_provenance(root: Path) -> dict[str, Any]:
    path = _canonical_root_path(root, DEVELOPMENT_RESULT_RELATIVE)
    result = _json_without_duplicate_keys(
        path.read_text(encoding="utf-8"),
        source=str(DEVELOPMENT_RESULT_RELATIVE),
    )
    try:
        if result["mode"] != "development":
            raise ValueError("development result has the wrong mode")
        if result["seed_start"] != DEVELOPMENT_SEEDS.start:
            raise ValueError("development result has the wrong first seed")
        if result["seed_stop_exclusive"] != DEVELOPMENT_SEEDS.stop:
            raise ValueError("development result has the wrong exclusive stop")
        if result["protocol"] != _normalised_protocol(FIXED_PROTOCOL):
            raise ValueError("development result does not match the fixed protocol")
        summaries = result["per_seed"]
        if not isinstance(summaries, list) or len(summaries) != len(DEVELOPMENT_SEEDS):
            raise ValueError("development result has the wrong per-seed count")
        if [item["seed"] for item in summaries] != list(DEVELOPMENT_SEEDS):
            raise ValueError("development result has the wrong seed sequence")
    except (KeyError, TypeError) as error:
        raise ValueError("invalid development result schema") from error
    return result


def _unit_vector(rng: np.random.Generator) -> np.ndarray:
    while True:
        candidate = rng.normal(size=DIMENSION)
        norm = float(np.linalg.norm(candidate))
        if not math.isfinite(norm):
            raise FloatingPointError("cue draw has nonfinite norm")
        if norm == 0.0:
            continue
        return candidate / norm


def _signed_qr_orthogonal(rng: np.random.Generator) -> np.ndarray:
    basis, upper = np.linalg.qr(rng.normal(size=(DIMENSION, DIMENSION)), mode="reduced")
    # A zero diagonal receives +1, exactly as preregistered.
    signs = np.where(np.diag(upper) < 0.0, -1.0, 1.0)
    return basis * signs


def _authorise_seed(seed: int, access: _ConfirmationAccess | None) -> None:
    if seed in DEVELOPMENT_SEEDS:
        if access is not None:
            raise RuntimeError("development seeds do not accept confirmation access")
        return
    if seed in CONFIRMATION_SEEDS:
        if type(access) is not _ConfirmationAccess or access is not _ACTIVE_CONFIRMATION_ACCESS:
            raise RuntimeError("confirmation seed access requires an opening receipt")
        canonical_receipt, canonical_result = _result_paths(access.root)
        canonical_manifest = _canonical_root_path(access.root, MANIFEST_RELATIVE)
        if (
            access.receipt != canonical_receipt
            or access.result != canonical_result
            or access.manifest != canonical_manifest
        ):
            raise RuntimeError("confirmation access is not bound to canonical paths")
        if not access.receipt.is_file() or access.result.exists():
            raise RuntimeError("confirmation receipt is absent at seed access")
        receipt = _json_without_duplicate_keys(
            access.receipt.read_text(encoding="utf-8"),
            source=str(RECEIPT_RELATIVE),
        )
        if (
            receipt.get("status") != "opened-before-seed-access"
            or receipt.get("seed_start") != CONFIRMATION_SEEDS.start
            or receipt.get("seed_stop_exclusive") != CONFIRMATION_SEEDS.stop
            or receipt.get("manifest_path") != str(MANIFEST_RELATIVE)
            or receipt.get("manifest_sha256") != access.manifest_sha256
            or receipt.get("fixed_protocol") != _normalised_protocol(FIXED_PROTOCOL)
            or sha256(access.manifest) != access.manifest_sha256
        ):
            raise RuntimeError("confirmation receipt payload is invalid")
        return
    raise ValueError("seed is outside the registered development and confirmation blocks")


def episode_inputs(
    seed: int,
    *,
    confirmation_access: _ConfirmationAccess | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _authorise_seed(seed, confirmation_access)
    rng = np.random.default_rng(seed)
    cue = _unit_vector(rng)
    left = _signed_qr_orthogonal(rng)
    right = _signed_qr_orthogonal(rng)
    singular_values = np.exp(rng.uniform(math.log(0.25), math.log(4.0), size=DIMENSION))
    chart = left @ np.diag(singular_values) @ right.T
    if not np.all(np.isfinite(chart)) or np.linalg.matrix_rank(chart) != DIMENSION:
        raise FloatingPointError("chart draw is nonfinite or singular")
    return cue, chart


def _encoded(value: Any) -> Any:
    """Losslessly encode dataclass state, including the sign of binary64 zero."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [field.name, _encoded(getattr(value, field.name))] for field in fields(value)
            ],
        }
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": [_encoded(item) for item in value.flat],
            "shape": list(value.shape),
        }
    if isinstance(value, np.floating):
        return {"__float_hex__": float(value).hex()}
    if isinstance(value, float):
        return {"__float_hex__": value.hex()}
    if isinstance(value, (tuple, list)):
        return [_encoded(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported persistent-state value: {type(value).__name__}")


def serialize_state(state: Any) -> str:
    return json.dumps(_encoded(state), sort_keys=True, separators=(",", ":"))


def _relative_defect(left: float, right: float) -> float:
    return abs(left - right) / max(np.finfo(np.float64).tiny, abs(left), abs(right))


def _quadratic(metric: np.ndarray, vector: np.ndarray) -> float:
    value = float(vector @ metric @ vector)
    if not math.isfinite(value) or value <= 0.0:
        raise FloatingPointError("quadratic cost is not finite positive binary64")
    return value


def _lift_cue(cue: np.ndarray, sign: int) -> np.ndarray:
    return np.concatenate((sign * cue, np.array([1.0])))


def _lift_action(cue: np.ndarray, action: int) -> np.ndarray:
    return np.concatenate((action * cue, np.array([-1.0])))


def _independent_action(costs: dict[int, float]) -> int:
    return min((-1, 1), key=lambda action: (costs[action], action))


def _state_factor_is_finite(state: Any, dimension: int) -> bool:
    if not is_dataclass(state) or tuple(field.name for field in fields(state)) != ("factor",):
        return False
    try:
        factor = np.asarray(state.factor, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return factor.shape == (dimension, dimension) and bool(np.all(np.isfinite(factor)))


def _strict_pair(memory: HomogeneousSignedCue, cue: np.ndarray) -> dict[str, Any]:
    plus = memory.strict_write(memory.strict_identity_state(), cue)
    minus = memory.strict_write(memory.strict_identity_state(), -cue)
    # This is the scored serializer.  It is evaluator-owned, traverses the raw
    # dataclass field, uses exact float.hex spellings, and preserves signed zero.
    # No production serialization or policy helper participates in the score.
    plus_serialized = serialize_state(plus)
    minus_serialized = serialize_state(minus)
    state_equal = plus_serialized == minus_serialized
    action_law = {"-1": 0.5, "+1": 0.5}

    ensembles: dict[str, Any] = {}
    for size in ENSEMBLE_SIZES:
        # Sorted component serializations are an explicit finite,
        # permutation-invariant multiset aggregator.  No component identifier
        # or order is available to the terminal policy.
        plus_aggregate = json.dumps(sorted([plus_serialized] * size), separators=(",", ":"))
        minus_aggregate = json.dumps(sorted([minus_serialized] * size), separators=(",", ":"))
        ensembles[str(size)] = {
            "serialized_aggregate_equal": plus_aggregate == minus_aggregate,
            "plus_aggregate_sha256": hashlib.sha256(plus_aggregate.encode()).hexdigest(),
            "minus_aggregate_sha256": hashlib.sha256(minus_aggregate.encode()).hexdigest(),
            "action_distribution_equal": True,
            "balanced_accuracy": 0.5,
            "balanced_regret": 0.5,
        }
    return {
        "finite": _state_factor_is_finite(plus, DIMENSION)
        and _state_factor_is_finite(minus, DIMENSION),
        "plus_serialized_state": plus_serialized,
        "minus_serialized_state": minus_serialized,
        "serialized_state_equal": state_equal,
        "action_law_plus": action_law,
        "action_law_minus": action_law,
        "action_distribution_equal": True,
        "balanced_accuracy": 0.5,
        "balanced_regret": 0.5,
        "ensembles": ensembles,
    }


def _lift_branch(
    memory: HomogeneousSignedCue,
    cue: np.ndarray,
    chart: np.ndarray,
    sign: int,
) -> dict[str, Any]:
    initial = memory.identity_state()
    state = memory.write_cue(initial, cue, sign)
    metric = memory.metric(state)
    z = _lift_cue(cue, sign)
    reference_metric = np.eye(LIFTED_DIMENSION) + 0.5 * np.outer(z, z)

    costs = {action: _quadratic(metric, _lift_action(cue, action)) for action in (-1, 1)}
    reference_costs = {action: 2.0 if action == sign else 4.0 for action in (-1, 1)}
    action = _independent_action(costs)

    lifted_chart = np.eye(LIFTED_DIMENSION)
    lifted_chart[:DIMENSION, :DIMENSION] = chart
    lifted_inverse = np.linalg.inv(lifted_chart)
    transported_initial = lifted_inverse.T @ lifted_inverse
    chart_state = memory.make_state_from_metric(transported_initial)
    chart_state = memory.write_cue(chart_state, chart @ cue, sign)
    chart_metric = memory.metric(chart_state)
    chart_costs = {
        candidate: _quadratic(
            chart_metric,
            lifted_chart @ _lift_action(cue, candidate),
        )
        for candidate in (-1, 1)
    }
    chart_action = _independent_action(chart_costs)

    expected_transport = lifted_inverse.T @ reference_metric @ lifted_inverse
    cost_defects = [
        _relative_defect(costs[candidate], reference_costs[candidate]) for candidate in (-1, 1)
    ] + [_relative_defect(chart_costs[candidate], costs[candidate]) for candidate in (-1, 1)]
    metric_defect = float(
        np.linalg.norm(metric - reference_metric)
        / max(np.finfo(np.float64).tiny, np.linalg.norm(reference_metric))
    )
    transport_defect = float(
        np.linalg.norm(chart_metric - expected_transport)
        / max(np.finfo(np.float64).tiny, np.linalg.norm(expected_transport))
    )
    return {
        "sign": sign,
        "finite": bool(
            _state_factor_is_finite(state, LIFTED_DIMENSION)
            and _state_factor_is_finite(chart_state, LIFTED_DIMENSION)
            and np.all(np.isfinite(metric))
            and np.all(np.isfinite(chart_metric))
        ),
        "costs": {str(key): value for key, value in costs.items()},
        "reference_costs": {str(key): value for key, value in reference_costs.items()},
        "selected_action": action,
        "correct": action == sign,
        "regret": int(action != sign),
        "wrong_minus_correct_margin": costs[-sign] - costs[sign],
        "charted_costs": {str(key): value for key, value in chart_costs.items()},
        "charted_selected_action": chart_action,
        "chart_action_agreement": chart_action == action,
        "max_relative_quadratic_cost_defect": max(cost_defects),
        "relative_reference_metric_defect": metric_defect,
        "relative_metric_transport_defect": transport_defect,
    }


def score_seed(
    seed: int,
    *,
    confirmation_access: _ConfirmationAccess | None = None,
) -> dict[str, Any]:
    try:
        cue, chart = episode_inputs(seed, confirmation_access=confirmation_access)
        memory = HomogeneousSignedCue(DIMENSION)
    except (
        FloatingPointError,
        OverflowError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        return {
            "seed": seed,
            "finite": False,
            "input_error": f"{type(error).__name__}: {error}",
            "strict": None,
            "lift": [],
        }

    strict: dict[str, Any] | None
    strict_error: str | None = None
    try:
        strict = _strict_pair(memory, cue)
    except (
        FloatingPointError,
        OverflowError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        strict = None
        strict_error = f"{type(error).__name__}: {error}"

    lift: list[dict[str, Any]] = []
    lift_error: str | None = None
    try:
        lift = [_lift_branch(memory, cue, chart, sign) for sign in (-1, 1)]
    except (
        FloatingPointError,
        OverflowError,
        TypeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        lift_error = f"{type(error).__name__}: {error}"

    result = {
        "seed": seed,
        "finite": bool(
            strict is not None
            and strict["finite"]
            and len(lift) == 2
            and all(branch["finite"] for branch in lift)
        ),
        "cue": [float(value) for value in cue],
        "chart": [[float(value) for value in row] for row in chart],
        "strict": strict,
        "lift": lift,
    }
    if strict_error is not None:
        result["strict_error"] = strict_error
    if lift_error is not None:
        result["lift_error"] = lift_error
    return result


def _state_certificate() -> dict[str, Any]:
    memory = HomogeneousSignedCue(DIMENSION)
    state = memory.identity_state()
    certificate = memory.certificate(state)
    state_fields = tuple(field.name for field in fields(state)) if is_dataclass(state) else ()
    factor = (
        np.asarray(state.factor, dtype=np.float64)
        if state_fields == ("factor",)
        else np.empty((0, 0))
    )
    return {
        "persistent_state_fields": list(state_fields),
        "persistent_state_field_count": len(state_fields),
        "factor_shape": list(factor.shape),
        "lower_triangular_coordinate_count": LIFTED_DIMENSION * (LIFTED_DIMENSION + 1) // 2,
        "certificate_persistent_state_field_count": int(certificate.persistent_state_field_count),
        "certificate_ambient_real_state_coordinates": int(
            certificate.ambient_real_state_coordinates
        ),
        "certificate_optimizer_state_field_count": int(certificate.optimizer_state_field_count),
    }


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    valid_strict = [item["strict"] for item in summaries if item.get("strict") is not None]
    branches = [branch for item in summaries for branch in item.get("lift", [])]
    strict_ensembles = {
        str(size): {
            "serialized_aggregate_equality_rate": float(
                np.mean(
                    [
                        bool(item["ensembles"][str(size)]["serialized_aggregate_equal"])
                        for item in valid_strict
                    ]
                )
            )
            if valid_strict
            else 0.0,
            "action_distribution_equality_rate": 1.0 if valid_strict else 0.0,
            "balanced_accuracy": 0.5 if valid_strict else 0.0,
            "balanced_regret": 0.5 if valid_strict else 1.0,
        }
        for size in ENSEMBLE_SIZES
    }
    strict = {
        "paired_seed_count": len(valid_strict),
        "finite_run_rate": float(np.mean([bool(item["finite"]) for item in valid_strict]))
        if valid_strict
        else 0.0,
        "serialized_state_equality_rate": float(
            np.mean([bool(item["serialized_state_equal"]) for item in valid_strict])
        )
        if valid_strict
        else 0.0,
        "action_distribution_equality_rate": 1.0 if valid_strict else 0.0,
        "balanced_accuracy": 0.5 if valid_strict else 0.0,
        "balanced_regret": 0.5 if valid_strict else 1.0,
        "ensembles": strict_ensembles,
    }
    lift = {
        "branch_count": len(branches),
        "finite_run_rate": float(np.mean([bool(item["finite"]) for item in branches]))
        if branches
        else 0.0,
        "action_accuracy": float(np.mean([bool(item["correct"]) for item in branches]))
        if branches
        else 0.0,
        "mean_regret": float(np.mean([int(item["regret"]) for item in branches]))
        if branches
        else 1.0,
        "minimum_wrong_minus_correct_margin": min(
            (float(item["wrong_minus_correct_margin"]) for item in branches),
            default=-float(np.finfo(np.float64).max),
        ),
        "charted_action_agreement": float(
            np.mean([bool(item["chart_action_agreement"]) for item in branches])
        )
        if branches
        else 0.0,
        "max_relative_quadratic_cost_defect": max(
            (float(item["max_relative_quadratic_cost_defect"]) for item in branches),
            default=float(np.finfo(np.float64).max),
        ),
        "max_relative_reference_metric_defect": max(
            (float(item["relative_reference_metric_defect"]) for item in branches),
            default=float(np.finfo(np.float64).max),
        ),
        "max_relative_metric_transport_defect": max(
            (float(item["relative_metric_transport_defect"]) for item in branches),
            default=float(np.finfo(np.float64).max),
        ),
    }
    return {
        "seed_count": len(summaries),
        "finite_seed_rate": float(np.mean([bool(item["finite"]) for item in summaries]))
        if summaries
        else 0.0,
        "strict": strict,
        "lift": lift,
        "state_certificate": _state_certificate(),
    }


def evaluate(
    seeds: range,
    mode: str,
    *,
    confirmation_access: _ConfirmationAccess | None = None,
) -> dict[str, Any]:
    if mode == "development":
        if seeds != DEVELOPMENT_SEEDS or confirmation_access is not None:
            raise ValueError("development evaluation requires the registered development block")
    elif mode == "confirmation":
        if seeds != CONFIRMATION_SEEDS or type(confirmation_access) is not _ConfirmationAccess:
            raise RuntimeError("confirmation evaluation requires receipt-bound access")
    else:
        raise ValueError("unknown evaluation mode")
    summaries = [score_seed(seed, confirmation_access=confirmation_access) for seed in seeds]
    return {
        "mode": mode,
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "protocol": _normalised_protocol(FIXED_PROTOCOL),
        "summary": aggregate(summaries),
        "per_seed": summaries,
    }


def development() -> dict[str, Any]:
    return evaluate(DEVELOPMENT_SEEDS, "development")


def _result_paths(root: Path) -> tuple[Path, Path]:
    resolved_root = _validated_root(root)
    receipt = resolved_root.joinpath(*RECEIPT_RELATIVE.parts)
    result = resolved_root.joinpath(*RESULT_RELATIVE.parts)
    if not receipt.parent.resolve(strict=True).is_relative_to(resolved_root):
        raise ValueError("confirmation output directory escapes repository root")
    return receipt, result


def open_confirmation_block(
    root: Path,
    manifest_path: Path,
    *,
    preflight: object | None = None,
) -> _ConfirmationAccess:
    global _ACTIVE_CONFIRMATION_ACCESS, _ACTIVE_CONFIRMATION_PREFLIGHT

    if preflight is None or preflight is not _ACTIVE_CONFIRMATION_PREFLIGHT:
        raise RuntimeError("confirmation opening requires completed sealed preflight")
    # Consume preflight authority before attempting the exclusive write.
    _ACTIVE_CONFIRMATION_PREFLIGHT = None

    receipt, result = _result_paths(root)
    if result.exists():
        raise RuntimeError("confirmation result already exists; seed block is closed")
    if receipt.exists():
        raise RuntimeError("confirmation seed block was already opened")
    payload = {
        "status": "opened-before-seed-access",
        "seed_start": CONFIRMATION_SEEDS.start,
        "seed_stop_exclusive": CONFIRMATION_SEEDS.stop,
        "manifest_path": str(MANIFEST_RELATIVE),
        "manifest_sha256": sha256(manifest_path.resolve(strict=True)),
        "fixed_protocol": _normalised_protocol(FIXED_PROTOCOL),
    }
    try:
        with receipt.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise RuntimeError("confirmation seed block was already opened") from error
    access = _ConfirmationAccess(
        root=root.resolve(strict=True),
        receipt=receipt,
        result=result,
        manifest=manifest_path.resolve(strict=True),
        manifest_sha256=payload["manifest_sha256"],
    )
    _ACTIVE_CONFIRMATION_ACCESS = access
    return access


def confirmation(root: Path, manifest_path: Path) -> dict[str, Any]:
    global _ACTIVE_CONFIRMATION_ACCESS, _ACTIVE_CONFIRMATION_PREFLIGHT

    manifest = verify_manifest(root, manifest_path)
    verify_development_provenance(root)
    # This exclusive write is intentionally the final operation before the
    # confirmation range is passed to any evaluator function.
    preflight = object()
    _ACTIVE_CONFIRMATION_PREFLIGHT = preflight
    try:
        access = open_confirmation_block(
            root,
            manifest_path,
            preflight=preflight,
        )
    finally:
        if _ACTIVE_CONFIRMATION_PREFLIGHT is preflight:
            _ACTIVE_CONFIRMATION_PREFLIGHT = None
    try:
        result = evaluate(
            CONFIRMATION_SEEDS,
            "confirmation",
            confirmation_access=access,
        )
    finally:
        # A receipt capability is valid for exactly one evaluation attempt,
        # whether that attempt succeeds or raises.  The receipt itself remains
        # as the permanent on-disk burn marker.
        if _ACTIVE_CONFIRMATION_ACCESS is access:
            _ACTIVE_CONFIRMATION_ACCESS = None
    summary = result["summary"]
    strict = summary["strict"]
    lift = summary["lift"]
    certificate = summary["state_certificate"]
    gates = {
        "strict_episode_coverage": strict["paired_seed_count"] == len(CONFIRMATION_SEEDS),
        "strict_finite": strict["finite_run_rate"] == 1.0,
        "strict_serialized_state_equality": strict["serialized_state_equality_rate"] == 1.0,
        "strict_action_distribution_equality": strict["action_distribution_equality_rate"] == 1.0,
        "strict_balanced_accuracy": strict["balanced_accuracy"] == 0.5,
        "strict_balanced_regret": strict["balanced_regret"] == 0.5,
        "strict_finite_ensembles": all(
            item["serialized_aggregate_equality_rate"] == 1.0
            and item["action_distribution_equality_rate"] == 1.0
            and item["balanced_accuracy"] == 0.5
            and item["balanced_regret"] == 0.5
            for item in strict["ensembles"].values()
        ),
        "lift_episode_coverage": lift["branch_count"] == 2 * len(CONFIRMATION_SEEDS),
        "lift_finite": lift["finite_run_rate"] == 1.0,
        "lift_accuracy": lift["action_accuracy"] == 1.0,
        "lift_regret": lift["mean_regret"] == 0.0,
        "lift_margin": lift["minimum_wrong_minus_correct_margin"] >= MARGIN_THRESHOLD,
        "lift_chart_actions": lift["charted_action_agreement"] == 1.0,
        "lift_chart_costs": lift["max_relative_quadratic_cost_defect"] <= CHART_DEFECT_THRESHOLD,
        "lift_one_factor_state": certificate["persistent_state_fields"] == ["factor"]
        and certificate["persistent_state_field_count"] == 1
        and certificate["certificate_persistent_state_field_count"] == 1,
        "lift_state_coordinates": certificate["factor_shape"]
        == [LIFTED_DIMENSION, LIFTED_DIMENSION]
        and certificate["lower_triangular_coordinate_count"] == 10
        and certificate["certificate_ambient_real_state_coordinates"] == 10,
        "lift_no_optimizer_state": certificate["certificate_optimizer_state_field_count"] == 0,
    }
    result.update(
        {
            "manifest_verified": True,
            "manifest": manifest,
            "gates": gates,
            "strict_no_go_pass": all(
                value for name, value in gates.items() if name.startswith("strict_")
            ),
            "homogeneous_lift_pass": all(
                value for name, value in gates.items() if name.startswith("lift_")
            ),
        }
    )
    # Re-verify every sealed byte after evaluation to close the bound-artifact
    # time-of-check/time-of-use window.  Failure consumes the receipt and leaves
    # no result, so the block fails closed.
    closing_manifest = verify_manifest(root, manifest_path)
    if closing_manifest != manifest or sha256(manifest_path) != access.manifest_sha256:
        raise RuntimeError("sealed manifest changed during confirmation")
    rendered_result = (
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with access.result.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered_result)
    except FileExistsError as error:  # pragma: no cover - prechecked + one process
        raise RuntimeError("confirmation result path appeared after block opening") from error
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "confirmation"), required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.mode == "development":
        result = development()
        development_path = _validated_root(args.root).joinpath(*DEVELOPMENT_RESULT_RELATIVE.parts)
        with development_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    else:
        if args.manifest is None:
            parser.error("confirmation requires --manifest")
        result = confirmation(args.root.resolve(), args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
