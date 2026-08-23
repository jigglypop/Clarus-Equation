from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from _run_paths import run_dir


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments"
    / "preregistration"
    / "agi_world_memory_integration_v3.json"
)
BASE_CONFIG = (
    ROOT
    / "experiments"
    / "preregistration"
    / "agi_world_memory_integration_v2.json"
)
AMENDMENT = (
    run_dir("agi-world-memory-integration-v1-20260810")
    / "revisions"
    / "31-v3-boundary-amendment.md"
)
BASE_CONTRACT = (
    run_dir("agi-world-memory-integration-v1-20260810")
    / "revisions"
    / "00-contract-v2-draft.md"
)

CONFIG_RAW_SHA256 = (
    "bb98be25d60484f0f477b052e97e66be0aa416ebf33712ca269c09f7bfa3758b"
)
BASE_CONFIG_RAW_SHA256 = (
    "b336fed11bf964512d1a2d50dd6c103a9593b426a986d4fe3b26e0bafa1338c2"
)
AMENDMENT_RAW_SHA256 = (
    "9b2e7cc13675798ca2db303aa4bebe984fad9705b12984560a7ad1ef955a7340"
)
BASE_CONTRACT_RAW_SHA256 = (
    "842512a55764e20a1b1f11c50c708b89bd8a8fe33b5c82f88a143f0cb36f7e70"
)
MERGED_REGISTRATION_SHA256 = (
    "37e7bfb6ee100c47164bec49f2e151234a647964839189ba47bf504552e1644b"
)
ALLOCATION_LEDGER_SHA256 = (
    "7f5c52b1b4aa01f8141ce821ed1bf4164e3fdf131ae828f08b20a8280f3079b4"
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _transport_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    assert raw, path
    assert not raw.startswith(b"\xef\xbb\xbf"), path
    assert b"\r" not in raw, path
    assert raw.endswith(b"\n"), path
    assert not raw.endswith(b"\n\n"), path
    raw.decode("utf-8")
    return raw


def _reject_constant(value: str) -> None:
    raise AssertionError(f"non-finite JSON literal is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON object key: {key}"
        result[key] = value
    return result


def _json_bytes(raw: bytes) -> dict[str, Any]:
    value = json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    assert isinstance(value, dict)
    return value


def _json_file(path: Path) -> dict[str, Any]:
    return _json_bytes(_transport_bytes(path))


def _canonical_artifact_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _artifact(path: Path) -> dict[str, Any]:
    raw = _transport_bytes(path)
    value = _json_bytes(raw)
    assert raw == _canonical_artifact_bytes(value), path
    return value


def _canonical_payload_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(raw)


def _delete_exact(target: dict[str, Any], path: Iterable[str]) -> None:
    segments = tuple(path)
    assert segments
    cursor = target
    for segment in segments[:-1]:
        assert segment in cursor and isinstance(cursor[segment], dict), segments
        cursor = cursor[segment]
    assert segments[-1] in cursor, segments
    del cursor[segments[-1]]


def _merge_exact(
    target: dict[str, Any],
    override: Mapping[str, Any],
    allowed_new: set[tuple[str, ...]],
    consumed: list[tuple[str, ...]],
    prefix: tuple[str, ...] = (),
) -> None:
    for key, value in override.items():
        path = (*prefix, key)
        if key not in target:
            assert path in allowed_new, path
            consumed.append(path)
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_exact(target[key], value, allowed_new, consumed, path)
        else:
            target[key] = copy.deepcopy(value)


def _merged_registration() -> dict[str, Any]:
    raw_v3 = _transport_bytes(CONFIG)
    raw_base = _transport_bytes(BASE_CONFIG)
    assert _sha256(raw_v3) == CONFIG_RAW_SHA256
    assert _sha256(raw_base) == BASE_CONFIG_RAW_SHA256
    assert _sha256(_transport_bytes(AMENDMENT)) == AMENDMENT_RAW_SHA256
    assert _sha256(_transport_bytes(BASE_CONTRACT)) == BASE_CONTRACT_RAW_SHA256

    v3 = _json_bytes(raw_v3)
    base = _json_bytes(raw_base)
    integrity = v3["amendment_integrity"]
    assert integrity == {
        "path": (
            "_workspace/ce/agi-world-memory-integration-v1-20260810/"
            "revisions/31-v3-boundary-amendment.md"
        ),
        "raw_sha256": AMENDMENT_RAW_SHA256,
        "status": "LOCKED_PRE_IMPLEMENTATION",
        "base_contract_path": (
            "_workspace/ce/agi-world-memory-integration-v1-20260810/"
            "revisions/00-contract-v2-draft.md"
        ),
        "base_contract_raw_sha256": BASE_CONTRACT_RAW_SHA256,
        "base_registration_path": (
            "experiments/preregistration/agi_world_memory_integration_v2.json"
        ),
        "base_registration_raw_sha256": BASE_CONFIG_RAW_SHA256,
    }

    merged = copy.deepcopy(base)
    delete_paths = [tuple(path) for path in v3["delete_paths"]]
    assert len(delete_paths) == len(set(delete_paths)) == 5
    for path in delete_paths:
        _delete_exact(merged, path)

    allowed_order = [
        tuple(path)
        for path in v3["merge_semantics"]["allowed_new_override_paths"]
    ]
    assert len(allowed_order) == len(set(allowed_order)) == 18
    consumed: list[tuple[str, ...]] = []
    _merge_exact(merged, v3["overrides"], set(allowed_order), consumed)
    assert consumed == allowed_order
    for key, value in v3.items():
        if key != "overrides":
            merged[key] = copy.deepcopy(value)

    assert _canonical_payload_sha256(merged) == MERGED_REGISTRATION_SHA256
    assert merged["experiment"] == "agi_world_memory_integration_v3"
    assert merged["data_roles"]["train"]["seeds"] == list(range(92100, 92140))
    assert merged["data_roles"]["validation"]["seeds"] == list(
        range(93100, 93140)
    )
    assert merged["data_roles"]["test"]["seeds"] == list(range(94100, 94160))
    return merged


def _git(*arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
    )


def _assert_head_identical(relative: str) -> None:
    tracked = _git("ls-files", "--error-unmatch", "--", relative)
    assert tracked.returncode == 0, tracked.stderr.decode(errors="replace")
    status = _git("status", "--porcelain=v1", "--", relative)
    assert status.returncode == 0, status.stderr.decode(errors="replace")
    assert status.stdout == b"", relative
    head = _git("show", f"HEAD:{relative}")
    assert head.returncode == 0, head.stderr.decode(errors="replace")
    assert head.stdout == (ROOT / relative).read_bytes(), relative


def _source_records(registration: Mapping[str, Any]) -> list[dict[str, str]]:
    result = []
    for relative in registration["implementation_dependency_manifest"][
        "ordered_source_paths"
    ]:
        path = ROOT / relative
        assert path.is_file(), relative
        result.append({"path": relative, "raw_sha256": _sha256(path.read_bytes())})
    return result


def _resolve_callable(dotted: str) -> Callable[..., object]:
    module_name, attribute = dotted.rsplit(".", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    assert callable(value), dotted
    return value


def _callable_records(registration: Mapping[str, Any]) -> list[dict[str, str]]:
    result = []
    boundaries = registration["implementation_dependency_manifest"][
        "callable_boundaries"
    ]
    assert len(boundaries) == 6
    for symbol in boundaries:
        text = inspect.getsource(_resolve_callable(symbol))
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        assert text.endswith("\n"), symbol
        assert not text.endswith("\n\n"), symbol
        result.append(
            {
                "symbol": symbol,
                "source_sha256": _sha256(text.encode("utf-8")),
            }
        )
    return result


def _allocation_sha256(registration: Mapping[str, Any]) -> str:
    ledger = registration["resources"]["allocation_ledger"]
    raw = json.dumps(
        ledger,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert len(ledger) == 36
    assert sum(int(entry["bytes"]) for entry in ledger) == 393216
    return _sha256(raw)


def _paths(registration: Mapping[str, Any]) -> dict[str, Path]:
    lock = registration["test_lock"]
    return {
        "implementation_lock": ROOT / lock["implementation_lock_path"],
        "calibration": ROOT / lock["calibration_path"],
        "validation": ROOT / lock["validation_path"],
        "test": ROOT / lock["test_path"],
        "integrity": ROOT / lock["integrity_path"],
    }


def _assert_exact_keys(value: Mapping[str, Any], expected: Iterable[str]) -> None:
    expected_keys = tuple(expected)
    assert len(expected_keys) == len(set(expected_keys))
    assert set(value) == set(expected_keys)


def _all_finite(values: object, expected_length: int) -> bool:
    if not isinstance(values, list) or len(values) != expected_length:
        return False
    return all(
        isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        for item in values
    )


def _assert_implementation_lock(
    registration: Mapping[str, Any],
    path: Path,
    source_records: list[dict[str, str]],
    callable_records: list[dict[str, str]],
) -> dict[str, Any]:
    report = _artifact(path)
    required = registration["artifact_state_machine"]["stages"][
        "implementation_lock"
    ]["required_output_fields"]
    _assert_exact_keys(report, required)
    assert report["experiment"] == "agi_world_memory_integration_v3"
    assert report["stage"] == "implementation_lock"
    assert report["registration_raw_sha256"] == CONFIG_RAW_SHA256
    assert report["contract_raw_sha256"] == AMENDMENT_RAW_SHA256
    assert report["ordered_path_raw_sha256"] == source_records
    assert report["callable_source_sha256_by_symbol"] == callable_records
    assert report["numpy_version"] == np.__version__
    assert report["ordered_allocation_ledger_sha256"] == ALLOCATION_LEDGER_SHA256
    assert report["registered_budget_vector"] == registration["resources"][
        "registered_budget_vector"
    ]
    assert len(report["registered_budget_vector"]) == 29
    handcrafted = report["handcrafted_test_results"]
    assert isinstance(handcrafted, dict) and handcrafted
    assert all(value is True for value in handcrafted.values())
    assert report["registered_seed_execution_count"] == 0
    return report


def _assert_calibration(
    registration: Mapping[str, Any],
    path: Path,
    implementation_raw_sha256: str,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    report = _artifact(path)
    required = registration["artifact_state_machine"]["stages"]["calibration"][
        "required_output_fields"
    ]
    _assert_exact_keys(report, required)
    assert report["experiment"] == "agi_world_memory_integration_v3"
    assert report["stage"] == "calibration"
    assert report["registration_raw_sha256"] == CONFIG_RAW_SHA256
    assert report["implementation_lock_raw_sha256"] == implementation_raw_sha256
    for key in (
        "ordered_path_raw_sha256",
        "callable_source_sha256_by_symbol",
        "numpy_version",
        "ordered_allocation_ledger_sha256",
    ):
        assert report[key] == implementation[key]
    assert report["registered_seed_execution_count"] == 40
    assert report["population_counts"] == registration["calibration"][
        "population_counts"
    ]
    assert isinstance(report["calibration_passed"], bool)
    assert report["status"] == ("PASS" if report["calibration_passed"] else "FAIL")
    assert isinstance(report["failure_reasons"], list)
    if report["calibration_passed"]:
        assert report["failure_reasons"] == []
        assert _all_finite(report["core_coefficients_20"], 20)
        assert _all_finite(report["mu_x"], 4)
        assert _all_finite(report["sigma_x"], 4)
        assert min(report["sigma_x"]) >= 0.05
        assert _all_finite(report["mu_codec"], 96)
        assert _all_finite(report["sigma_codec"], 96)
        assert min(report["sigma_codec"]) >= 1e-8
        assert math.isfinite(float(report["tau_recall"]))
        assert math.isfinite(float(report["tau_join"]))
        assert _all_finite(report["recall_positive_confidence_pool"], 960)
        assert _all_finite(report["recall_lure_confidence_pool"], 960)
        assert _all_finite(report["join_endpoint_value_pool"], 3840)
    else:
        assert report["failure_reasons"]
    payload = dict(report)
    declared = payload.pop("canonical_payload_sha256_excluding_this_field")
    assert declared == _canonical_payload_sha256(payload)
    return report


def _vector(
    primitive: Mapping[str, Any],
    path: tuple[str, ...],
    length: int,
    *,
    kind: str = "float",
) -> np.ndarray:
    value: Any = primitive
    for key in path:
        assert isinstance(value, dict) and key in value, path
        value = value[key]
    assert isinstance(value, list) and len(value) == length, path
    if kind == "bool":
        assert all(type(item) is bool for item in value), path
        return np.asarray(value, dtype=np.bool_)
    if kind == "int":
        assert all(type(item) is int and item >= 0 for item in value), path
        return np.asarray(value, dtype=np.int64)
    assert all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ), path
    result = np.asarray(value, dtype=np.float64)
    assert np.all(np.isfinite(result)), path
    return result


def _interval(vector: np.ndarray, critical: float) -> tuple[float, float, float]:
    assert vector.ndim == 1 and vector.size >= 2 and np.all(np.isfinite(vector))
    mean = float(np.mean(vector))
    sample_sd = float(np.std(vector, ddof=1))
    half = critical * sample_sd / math.sqrt(vector.size)
    return mean, mean - half, mean + half


def _positive_ratio(numerator: float, denominator: float) -> float:
    assert math.isfinite(numerator)
    assert math.isfinite(denominator) and denominator > 0.0
    return numerator / denominator


def _recompute_checks(
    registration: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, bool]:
    split = report["split"]
    assert split in {"validation", "test"}
    length = 40 if split == "validation" else 60
    critical = float(registration["paired_inference"][f"critical_{split}"])
    primitive = report["primitive_seed_vectors"]
    assert isinstance(primitive, dict)

    def f(*path: str) -> np.ndarray:
        return _vector(primitive, tuple(path), length)

    def i(*path: str) -> np.ndarray:
        return _vector(primitive, tuple(path), length, kind="int")

    def b(*path: str) -> np.ndarray:
        return _vector(primitive, tuple(path), length, kind="bool")

    eall = {
        cell: f("E_all_H20", cell) for cell in ("M00", "M10", "M01", "M11")
    }
    eall_h5 = f("E_all_H5", "M11")
    euv = {
        cell: f("E_uv_H20", cell) for cell in ("M00", "M10", "M01", "M11")
    }
    regret = {cell: f("regret", cell) for cell in ("M00", "M11")}
    success = {cell: f("success", cell) for cell in ("M00", "M11")}

    ltm = 0.5 * ((eall["M00"] - eall["M10"]) + (eall["M01"] - eall["M11"]))
    dream_00 = euv["M00"] - euv["M01"]
    dream_10 = euv["M10"] - euv["M11"]
    joint = eall["M00"] - eall["M11"]
    persistence = f("E_all_H20", "persistence")
    persistence_benefit = persistence - eall["M11"]
    regret_benefit = regret["M00"] - regret["M11"]
    success_gain = success["M11"] - success["M00"]

    means = {
        name: float(np.mean(values))
        for name, values in {
            **{f"Eall_{key}": value for key, value in eall.items()},
            **{f"Euv_{key}": value for key, value in euv.items()},
            "Eall_M11_H5": eall_h5,
            "persistence": persistence,
            "regret_M00": regret["M00"],
            "regret_M11": regret["M11"],
            "success_M11": success["M11"],
        }.items()
    }
    _, ltm_lower, _ = _interval(ltm, critical)
    _, dream_00_lower, _ = _interval(dream_00, critical)
    _, dream_10_lower, _ = _interval(dream_10, critical)
    _, joint_lower, _ = _interval(joint, critical)
    _, persistence_lower, _ = _interval(persistence_benefit, critical)
    _, regret_lower, _ = _interval(regret_benefit, critical)
    success_gain_mean, success_gain_lower, _ = _interval(success_gain, critical)

    checks: dict[str, bool] = {
        "prediction.marginal_ltm_relative_reduction": 1.0
        - _positive_ratio(
            means["Eall_M10"] + means["Eall_M11"],
            means["Eall_M00"] + means["Eall_M01"],
        )
        >= 0.10,
        "prediction.marginal_ltm_ci_lower": ltm_lower > 0.0,
        "prediction.marginal_ltm_strict_win_fraction": float(np.mean(ltm > 0.0))
        >= 0.65,
        "prediction.dream_M00_to_M01_relative_reduction": 1.0
        - _positive_ratio(means["Euv_M01"], means["Euv_M00"])
        >= 0.10,
        "prediction.dream_M00_to_M01_ci_lower": dream_00_lower > 0.0,
        "prediction.dream_M00_to_M01_strict_win_fraction": float(
            np.mean(dream_00 > 0.0)
        )
        >= 0.65,
        "prediction.dream_M10_to_M11_relative_reduction": 1.0
        - _positive_ratio(means["Euv_M11"], means["Euv_M10"])
        >= 0.10,
        "prediction.dream_M10_to_M11_ci_lower": dream_10_lower > 0.0,
        "prediction.dream_M10_to_M11_strict_win_fraction": float(
            np.mean(dream_10 > 0.0)
        )
        >= 0.65,
        "prediction.joint_relative_reduction": 1.0
        - _positive_ratio(means["Eall_M11"], means["Eall_M00"])
        >= 0.10,
        "prediction.joint_ci_lower": joint_lower > 0.0,
        "prediction.joint_strict_win_fraction": float(np.mean(joint > 0.0)) >= 0.65,
        "prediction.M11_E_all_H20": means["Eall_M11"] <= 1.00,
        "prediction.M01_E_uv_H20": means["Euv_M01"] <= 1.00,
        "prediction.M11_E_uv_H20": means["Euv_M11"] <= 1.00,
        "prediction.M11_H20_over_H5": _positive_ratio(
            means["Eall_M11"], means["Eall_M11_H5"]
        )
        <= 2.00,
        "prediction.M11_vs_persistence_relative_reduction": 1.0
        - _positive_ratio(means["Eall_M11"], means["persistence"])
        >= 0.10,
        "prediction.M11_vs_persistence_ci_lower": persistence_lower > 0.0,
        "prediction.M11_vs_persistence_strict_win_fraction": float(
            np.mean(persistence_benefit > 0.0)
        )
        >= 0.65,
        "planning.M11_regret_relative_reduction_vs_M00": 1.0
        - _positive_ratio(means["regret_M11"], means["regret_M00"])
        >= 0.20,
        "planning.regret_ci_lower": regret_lower > 0.0,
        "planning.success_gain": success_gain_mean >= 0.10,
        "planning.success_gain_ci_lower": success_gain_lower > 0.0,
        "planning.M11_success_mean": means["success_M11"] >= 0.75,
        "planning.M11_invalid_selected_count": int(
            np.sum(i("invalid_selected_count", "M11"))
        )
        == 0,
    }

    for cell in ("M10", "M11"):
        coverage = f("recall", "coverage", cell)
        identity = f("recall", "identity_accuracy", cell)
        wrong_all = f("recall", "wrong_all", cell)
        wrong_given = f("recall", "wrong_given_accept", cell)
        false_lure = f("recall", "false_lure", cell)
        _, _, false_lure_upper = _interval(false_lure, critical)
        checks.update(
            {
                f"recall.{cell}_coverage": float(np.mean(coverage)) >= 0.80,
                f"recall.{cell}_identity_accuracy": float(np.mean(identity)) >= 0.80,
                f"recall.{cell}_wrong_all": float(np.mean(wrong_all)) <= 0.05,
                f"recall.{cell}_wrong_given_accept": float(np.mean(wrong_given))
                <= 0.05,
                f"recall.{cell}_false_lure_mean": float(np.mean(false_lure)) <= 0.05,
                f"recall.{cell}_false_lure_ci_upper": false_lure_upper <= 0.05,
                f"recall.{cell}_cross_port_accept_count": int(
                    np.sum(i("recall", "cross_port_accept_count", cell))
                )
                == 0,
            }
        )

    for cell in ("M01", "M11"):
        checks.update(
            {
                f"dream.{cell}_missing_binding_coverage": float(
                    np.mean(f("dream", "missing_binding_coverage", cell))
                )
                >= 0.80,
                f"dream.{cell}_accepted_invalid_splice_count": int(
                    np.sum(i("dream", "accepted_invalid_splice_count", cell))
                )
                == 0,
                f"dream.{cell}_observed_overwrite_count": int(
                    np.sum(i("dream", "observed_overwrite_count", cell))
                )
                == 0,
            }
        )

    recall_antagonism = f("E_recall", "M11") - 1.02 * f("E_recall", "M10")
    dream_antagonism = euv["M11"] - 1.02 * euv["M01"]
    shuffled = f("E_all_H20", "shuffled_binding") - eall["M10"]
    zero_q = f("E_all_H20", "zero_q") - eall["M10"]
    lesion = f("dream", "invalid_splice_rate", "unconstrained_lesion") - f(
        "dream", "invalid_splice_rate", "M01"
    )
    zero_synthetic = f("E_uv_H20", "zero_synthetic") - euv["M01"]
    checks.update(
        {
            "no_antagonism.recall_paired_upper": _interval(
                recall_antagonism, critical
            )[2]
            <= 0.0,
            "no_antagonism.dream_paired_upper": _interval(
                dream_antagonism, critical
            )[2]
            <= 0.0,
            "attribution.shuffled_vs_M10_ci_lower": _interval(shuffled, critical)[1]
            > 0.0,
            "attribution.zero_q_vs_M10_ci_lower": _interval(zero_q, critical)[1]
            > 0.0,
            "attribution.lesion_invalid_splice_vs_M01_ci_lower": _interval(
                lesion, critical
            )[1]
            > 0.0,
            "attribution.zero_synthetic_vs_M01_E_uv_ci_lower": _interval(
                zero_synthetic, critical
            )[1]
            > 0.0,
            "stability.all_finite": bool(np.all(b("stability", "all_finite"))),
            "stability.max_abs_prediction": float(
                np.max(f("stability", "max_abs_prediction"))
            )
            <= 2.0,
            "stability.max_seed_invalid_predicted_transition_rate": float(
                np.max(f("stability", "invalid_predicted_transition_rate"))
            )
            <= 0.01,
            "stability.h5_h20_bit_exact": bool(
                np.all(b("stability", "h5_h20_bit_exact"))
            ),
        }
    )
    expected = registration["all_of_gate"]["split_pass_mapping"][
        "checks_exact_keyset"
    ]
    assert set(checks) == set(expected)
    return {key: checks[key] for key in expected}


def _assert_pass_mapping(
    registration: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    mapping = registration["all_of_gate"]["split_pass_mapping"]
    checks = report["checks"]
    hard_checks = report["hard_zero_checks"]
    resource_checks = report["resource_checks"]
    hard_counts = report["hard_zero_counts"]
    assert isinstance(checks, dict) and checks
    assert isinstance(hard_checks, dict) and hard_checks
    assert isinstance(resource_checks, dict) and resource_checks
    assert isinstance(hard_counts, dict) and hard_counts
    assert set(checks) == set(mapping["checks_exact_keyset"])
    assert set(hard_checks) == set(mapping["hard_zero_checks_exact_keyset"])
    assert set(resource_checks) == set(mapping["resource_checks_exact_keyset"])
    assert set(hard_counts) == set(mapping["hard_zero_checks_exact_keyset"])
    assert all(isinstance(value, bool) for value in checks.values())
    assert all(isinstance(value, bool) for value in hard_checks.values())
    assert all(isinstance(value, bool) for value in resource_checks.values())
    assert all(type(value) is int and value >= 0 for value in hard_counts.values())
    assert hard_checks == {key: value == 0 for key, value in hard_counts.items()}
    assert checks == _recompute_checks(registration, report)
    performance = all(checks.values())
    integrity = all(hard_checks.values())
    resource = all(resource_checks.values())
    assert report["performance_passed"] is performance
    assert report["integrity_passed"] is integrity
    assert report["resource_passed"] is resource
    assert report["passed"] is (performance and integrity and resource)


def _assert_split(
    registration: Mapping[str, Any],
    path: Path,
    split: str,
    seed_count: int,
    implementation_raw_sha256: str,
    calibration_raw_sha256: str,
    implementation: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    report = _artifact(path)
    required = registration["artifact_state_machine"]["stages"][split][
        "required_output_fields"
    ]
    _assert_exact_keys(report, required)
    assert report["experiment"] == "agi_world_memory_integration_v3"
    assert report["stage"] == split
    assert report["split"] == split
    assert report["registration_raw_sha256"] == CONFIG_RAW_SHA256
    assert report["implementation_lock_raw_sha256"] == implementation_raw_sha256
    assert report["calibration_raw_sha256"] == calibration_raw_sha256
    assert report["core_payload_sha256"] == calibration["core_payload_sha256"]
    for key in (
        "ordered_path_raw_sha256",
        "callable_source_sha256_by_symbol",
        "numpy_version",
        "ordered_allocation_ledger_sha256",
    ):
        assert report[key] == implementation[key]
    assert report["registered_seed_execution_count"] == seed_count
    assert isinstance(report["primitive_seed_vectors"], dict)
    assert report["primitive_seed_vectors"]
    assert isinstance(report["cell_and_control_summaries"], dict)
    assert report["cell_and_control_summaries"]
    assert isinstance(report["effect_and_interaction_reports"], dict)
    assert report["effect_and_interaction_reports"]
    _assert_pass_mapping(registration, report)
    return report


def _validation_is_committed_clean_head_identical(relative: str, raw: bytes) -> bool:
    tracked = _git("ls-files", "--error-unmatch", "--", relative)
    if tracked.returncode != 0:
        return False
    status = _git("status", "--porcelain=v1", "--", relative)
    if status.returncode != 0 or status.stdout:
        return False
    head = _git("show", f"HEAD:{relative}")
    return head.returncode == 0 and head.stdout == raw


def _assert_integrity_artifact(
    registration: Mapping[str, Any],
    path: Path,
    source_records: list[dict[str, str]],
    expected_counts: Mapping[str, int],
) -> None:
    report = _artifact(path)
    required = registration["artifact_state_machine"]["stages"]["integrity"][
        "required_output_fields"
    ]
    _assert_exact_keys(report, required)
    assert report["experiment"] == "agi_world_memory_integration_v3"
    assert report["stage"] == "integrity"
    assert report["ordered_path_raw_sha256"] == source_records
    assert report["registered_execution_counts"] == dict(expected_counts)
    assert report["all_hashes_match"] is True
    assert report["scientific_world_generation_count"] == 0
    assert report["scientific_artifact_mutation_count"] == 0
    artifact_paths = _paths(registration)
    history = [
        {
            "stage": "implementation_lock",
            "run_count": 0,
            "artifact_raw_sha256": _sha256(
                _transport_bytes(artifact_paths["implementation_lock"])
            ),
        },
        {
            "stage": "calibration",
            "run_count": 1,
            "artifact_raw_sha256": _sha256(
                _transport_bytes(artifact_paths["calibration"])
            ),
        },
    ]
    if expected_counts["validation"]:
        history.append(
            {
                "stage": "validation",
                "run_count": 1,
                "artifact_raw_sha256": _sha256(
                    _transport_bytes(artifact_paths["validation"])
                ),
            }
        )
    if expected_counts["test"]:
        history.append(
            {
                "stage": "test",
                "run_count": 1,
                "artifact_raw_sha256": _sha256(
                    _transport_bytes(artifact_paths["test"])
                ),
            }
        )
    history.append(
        {
            "stage": "integrity",
            "run_count": 0,
            # A raw hash inside the file being hashed would be self-referential.
            "artifact_raw_sha256": None,
        }
    )
    assert report["stage_history"] == history


def test_v3_registration_merge_and_source_boundary_are_byte_exact() -> None:
    registration = _merged_registration()
    assert _allocation_sha256(registration) == ALLOCATION_LEDGER_SHA256
    assert len(registration["resources"]["registered_budget_vector"]) == 29
    assert len({item["name"] for item in registration["resources"]["registered_budget_vector"]}) == 29
    _assert_head_identical(
        "experiments/preregistration/agi_world_memory_integration_v3.json"
    )
    _assert_head_identical(AMENDMENT.relative_to(ROOT).as_posix())
    source_records = _source_records(registration)
    callable_records = _callable_records(registration)
    assert len(source_records) == 8
    assert len(callable_records) == 6


def test_v3_artifact_state_machine_without_preunlock_test_read() -> None:
    registration = _merged_registration()
    artifact_paths = _paths(registration)
    source_records = _source_records(registration)
    callable_records = _callable_records(registration)

    implementation_path = artifact_paths["implementation_lock"]
    calibration_path = artifact_paths["calibration"]
    validation_path = artifact_paths["validation"]
    test_path = artifact_paths["test"]
    integrity_path = artifact_paths["integrity"]

    if not implementation_path.exists():
        assert not calibration_path.exists()
        assert not validation_path.exists()
        # Existence is the only operation permitted on the locked test path here.
        assert not test_path.exists()
        assert not integrity_path.exists()
        return

    implementation = _assert_implementation_lock(
        registration,
        implementation_path,
        source_records,
        callable_records,
    )
    implementation_raw_sha256 = _sha256(_transport_bytes(implementation_path))

    if not calibration_path.exists():
        assert not validation_path.exists()
        assert not test_path.exists()
        assert not integrity_path.exists()
        return

    calibration = _assert_calibration(
        registration,
        calibration_path,
        implementation_raw_sha256,
        implementation,
    )
    calibration_raw_sha256 = _sha256(_transport_bytes(calibration_path))

    if not calibration["calibration_passed"]:
        assert not validation_path.exists()
        assert not test_path.exists()
        if integrity_path.exists():
            _assert_integrity_artifact(
                registration,
                integrity_path,
                source_records,
                {"train": 40, "validation": 0, "test": 0},
            )
        return

    if not validation_path.exists():
        assert not test_path.exists()
        assert not integrity_path.exists()
        return

    validation = _assert_split(
        registration,
        validation_path,
        "validation",
        40,
        implementation_raw_sha256,
        calibration_raw_sha256,
        implementation,
        calibration,
    )
    validation_raw = _transport_bytes(validation_path)
    validation_raw_sha256 = _sha256(validation_raw)

    validation_relative = registration["test_lock"]["validation_path"]
    committed_unlock = validation["passed"] and (
        _validation_is_committed_clean_head_identical(
            validation_relative,
            validation_raw,
        )
    )
    if not committed_unlock:
        # Do not call read_bytes, JSON parsing, hashing, or Git-show on test_path.
        assert not test_path.exists()
        if integrity_path.exists():
            assert validation["passed"] is False
            _assert_integrity_artifact(
                registration,
                integrity_path,
                source_records,
                {"train": 40, "validation": 40, "test": 0},
            )
        return

    if not test_path.exists():
        assert not integrity_path.exists()
        return

    # This is the first locked-test read, and it is dominated by the complete
    # validation PASS plus committed/clean/HEAD-identical unlock predicate above.
    test = _assert_split(
        registration,
        test_path,
        "test",
        60,
        implementation_raw_sha256,
        calibration_raw_sha256,
        implementation,
        calibration,
    )
    assert test["validation_raw_sha256"] == validation_raw_sha256
    unlock = test["unlock_record"]
    expected_unlock_keys = registration["artifact_state_machine"]["stages"][
        "test"
    ]["unlock_record_exact_fields"]
    _assert_exact_keys(unlock, expected_unlock_keys)
    assert unlock == {
        "validation_raw_sha256": validation_raw_sha256,
        "registration_raw_sha256": CONFIG_RAW_SHA256,
        "implementation_lock_raw_sha256": implementation_raw_sha256,
        "calibration_raw_sha256": calibration_raw_sha256,
        "ordered_path_raw_sha256": source_records,
        "test_unlocked": True,
    }

    if integrity_path.exists():
        _assert_integrity_artifact(
            registration,
            integrity_path,
            source_records,
            {"train": 40, "validation": 40, "test": 60},
        )
