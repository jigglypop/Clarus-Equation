"""One-shot development-only runner for causal recurrent geometry Phase A V1."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import sys
from types import ModuleType
from typing import Any, Callable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_RELATIVE = PurePosixPath(
    "experiments/preregistration/causal_recurrent_geometry_phase_a_v1.json"
)
RUN_RELATIVE = PurePosixPath(
    "_workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816"
)
CONTRACT_RELATIVE = RUN_RELATIVE / "00-contract.md"
PRODUCTION_RELATIVE = PurePosixPath(
    "reality_stone/python/reality_stone/clarus/"
    "causal_recurrent_geometry_benchmark.py"
)
TEST_RELATIVE = PurePosixPath("tests/test_causal_recurrent_geometry_benchmark.py")
RUNNER_RELATIVE = PurePosixPath(
    "examples/agi/causal_recurrent_geometry_development_run.py"
)
RESULT_RELATIVE = RUN_RELATIVE / "artifacts/development-results.json"
REQUIRED_ARTIFACT_PATHS = frozenset(
    {
        str(CONTRACT_RELATIVE),
        str(PRODUCTION_RELATIVE),
        str(TEST_RELATIVE),
        str(RUNNER_RELATIVE),
    }
)
MANIFEST_SCHEMA = "ce.causal_recurrent_geometry.phase_a.manifest.v1"
HASH_POLICY = "sha256-canonical-json-excluding-manifest_sha256"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_without_duplicate_keys(text: str, *, source: str) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {source}: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant in {source}: {value}")

    return json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _canonical_relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid repository-relative path: {value!r}")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or str(parsed) != value:
        raise ValueError(f"invalid repository-relative path: {value!r}")
    return parsed


def _repository_path(root: Path, value: object, *, must_exist: bool) -> Path:
    relative = _canonical_relative_path(value)
    candidate = root.joinpath(*relative.parts).resolve(strict=must_exist)
    if not candidate.is_relative_to(root):
        raise ValueError(f"path escapes repository root: {value!r}")
    return candidate


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    return _sha256_bytes(_canonical_bytes(unsigned))


def _exact_int_list(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty integer list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise TypeError(f"{name} must contain only integers")
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be nonnegative and unique")
    return result


def _validate_confirmation_seal(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("confirmation seal must be an object")
    required = {
        "commitment_domain",
        "commitment_scheme",
        "custody_status",
        "disjoint_from_pilot_and_development",
        "execution_authorized",
        "holdout_status",
        "opaque_commitment",
        "raw_seed_material_present",
        "reservation_kind",
        "seed_count",
        "status",
    }
    if set(value) != required:
        raise ValueError("confirmation seal has an unexpected field")
    if value["status"] != "reserved_unopened":
        raise ValueError("confirmation must remain reserved_unopened")
    if value["commitment_domain"] != (
        "CE-PHASE-A-V1-CONFIRMATION-SPLIT-ROTATION-2"
    ):
        raise ValueError("confirmation commitment domain is not registered")
    if value["commitment_scheme"] != (
        "sha256(canonical-json-secret-seed-block-v1)"
    ):
        raise ValueError("confirmation commitment scheme is not registered")
    if value["reservation_kind"] != "reservation_only":
        raise ValueError("confirmation is only a reservation marker")
    if value["custody_status"] != "custody_unverified":
        raise ValueError("confirmation custody must remain unverified")
    if value["holdout_status"] != "not_executable_holdout":
        raise ValueError("confirmation is not an executable holdout")
    if value["execution_authorized"] is not False:
        raise PermissionError("confirmation execution is not authorized")
    if value["raw_seed_material_present"] is not False:
        raise PermissionError("raw confirmation seed material is forbidden")
    if value["disjoint_from_pilot_and_development"] is not True:
        raise ValueError("confirmation disjointness commitment is required")
    if (
        isinstance(value["seed_count"], bool)
        or not isinstance(value["seed_count"], int)
        or value["seed_count"] <= 0
    ):
        raise ValueError("confirmation seed_count must be positive")
    commitment = value["opaque_commitment"]
    if not isinstance(commitment, str) or re.fullmatch(
        r"[0-9a-f]{64}", commitment
    ) is None:
        raise ValueError("confirmation opaque_commitment must be lowercase SHA-256")
    return dict(value)


def _verify_required_artifacts(
    root: Path, value: object
) -> dict[str, str]:
    if not isinstance(value, list) or not value:
        raise ValueError("required_artifacts must be a nonempty list")
    paths: list[str] = []
    hashes: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("required artifact must contain exactly path and sha256")
        relative = str(_canonical_relative_path(item["path"]))
        expected = item["sha256"]
        if not isinstance(expected, str) or re.fullmatch(
            r"[0-9a-f]{64}", expected
        ) is None:
            raise ValueError(f"invalid SHA-256 for {relative}")
        paths.append(relative)
        hashes.append(expected)
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate required artifact path")
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate required artifact hash")
    if set(paths) != REQUIRED_ARTIFACT_PATHS:
        missing = sorted(REQUIRED_ARTIFACT_PATHS.difference(paths))
        extra = sorted(set(paths).difference(REQUIRED_ARTIFACT_PATHS))
        raise ValueError(
            f"required artifact paths must be exact; missing={missing}, extra={extra}"
        )
    verified: dict[str, str] = {}
    for relative, expected in zip(paths, hashes, strict=True):
        actual = _sha256_path(_repository_path(root, relative, must_exist=True))
        if actual != expected:
            raise ValueError(f"required artifact hash mismatch: {relative}")
        verified[relative] = expected
    return dict(sorted(verified.items()))


def _validate_dimensionless_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "dimension_tags",
        "normalized_coordinates",
        "reference_scales",
    }:
        raise ValueError("dimensionless fields must be exact")
    if value["normalized_coordinates"] is not True:
        raise ValueError("normalized_coordinates must be true")
    tags = value["dimension_tags"]
    required_tags = {"gaussian_residual", "input", "noise", "state"}
    if not isinstance(tags, dict) or set(tags) != required_tags:
        raise ValueError("dimensionless tags must be exact")
    if any(tags[name] != "DIMENSIONLESS" for name in required_tags):
        raise ValueError("every registered dimension tag must be DIMENSIONLESS")
    scales = value["reference_scales"]
    if not isinstance(scales, dict) or set(scales) != {"input", "noise", "state"}:
        raise ValueError("dimensionless reference scale fields must be exact")
    for name, values in scales.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"dimensionless {name} scale must be nonempty")
        for item in values:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TypeError(f"dimensionless {name} scale must be numeric")
            if not np.isfinite(item) or item <= 0.0:
                raise ValueError(f"dimensionless {name} scale must be finite positive")
    return dict(value)


def validate_manifest_payload(
    payload: object,
    *,
    root: Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate canonical fields, split sealing, dimensions, and artifact hashes."""

    resolved_root = root.resolve(strict=True)
    if resolved_root != REPO_ROOT.resolve(strict=True):
        raise ValueError("root must be the canonical repository root")
    if not isinstance(payload, dict):
        raise ValueError("manifest must be an object")
    required_top = {
        "allowed_claim_boundary",
        "bootstrap",
        "confirmation",
        "development_output_path",
        "dimensionless",
        "dof_accounting",
        "generator",
        "manifest_hash_policy",
        "manifest_sha256",
        "primary_endpoint",
        "required_artifacts",
        "schema",
        "seed_roles",
        "status",
        "stop_rules",
    }
    if set(payload) != required_top:
        raise ValueError("manifest top-level fields must be exact")
    if payload["schema"] != MANIFEST_SCHEMA:
        raise ValueError("unknown manifest schema")
    if payload["status"] != "preregistered_development":
        raise ValueError("manifest status must be preregistered_development")
    if payload["manifest_hash_policy"] != HASH_POLICY:
        raise ValueError("unknown manifest hash policy")
    expected_manifest_digest = payload["manifest_sha256"]
    if not isinstance(expected_manifest_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", expected_manifest_digest
    ) is None:
        raise ValueError("manifest_sha256 must be lowercase SHA-256")
    if _manifest_digest(payload) != expected_manifest_digest:
        raise ValueError("manifest self-hash mismatch")
    output = str(_canonical_relative_path(payload["development_output_path"]))
    if output != str(RESULT_RELATIVE):
        raise ValueError("development output path is not canonical")
    roles = payload["seed_roles"]
    if not isinstance(roles, dict) or set(roles) != {
        "development_graph_seeds",
        "pilot_graph_seeds",
    }:
        raise ValueError("seed_roles must contain exactly pilot and development")
    pilot = _exact_int_list("pilot_graph_seeds", roles["pilot_graph_seeds"])
    development = _exact_int_list(
        "development_graph_seeds", roles["development_graph_seeds"]
    )
    if set(pilot).intersection(development):
        raise ValueError("pilot and development graph seeds overlap")
    confirmation = _validate_confirmation_seal(payload["confirmation"])
    bootstrap = payload["bootstrap"]
    if not isinstance(bootstrap, dict) or set(bootstrap) != {"samples", "seed"}:
        raise ValueError("bootstrap must contain samples and seed")
    if any(
        isinstance(bootstrap[key], bool)
        or not isinstance(bootstrap[key], int)
        or bootstrap[key] <= 0
        for key in ("samples", "seed")
    ):
        raise ValueError("bootstrap samples and seed must be positive integers")
    generator = payload["generator"]
    if not isinstance(generator, dict):
        raise ValueError("generator must be an object")
    if set(generator) != {
        "context_count",
        "context_heterogeneity",
        "experiment_version",
        "heldout_intervention_scale",
        "heldout_steps",
        "input_dimension",
        "master_seed",
        "noise_sigma",
        "ridge",
        "state_dimension",
        "train_intervention_scale",
        "train_steps",
    }:
        raise ValueError("generator fields must be exact")
    dimensionless = _validate_dimensionless_payload(payload["dimensionless"])
    reference_scales = dimensionless.get("reference_scales")
    if not isinstance(reference_scales, dict):
        raise ValueError("dimensionless reference_scales must be an object")
    expected_scale_lengths = {
        "input": generator["input_dimension"],
        "noise": 1,
        "state": generator["state_dimension"],
    }
    for name, expected_length in expected_scale_lengths.items():
        values = reference_scales.get(name)
        if not isinstance(values, list) or len(values) != expected_length:
            raise ValueError(
                f"dimensionless {name} reference scale length must be "
                f"{expected_length}"
            )
    if payload["primary_endpoint"] != (
        "graph-seed paired delta_nll = pooled_nll - factorized_nll"
    ):
        raise ValueError("primary endpoint differs from registration")
    if payload["stop_rules"] != {
        "PA-H1": "STOP unless mean, median, and paired-bootstrap lower bound are positive",
        "PA-H2": "STOP unless mean, median, and paired-bootstrap lower bound are positive",
    }:
        raise ValueError("STOP rules differ from registration")
    boundary = payload["allowed_claim_boundary"]
    if boundary != {
        "exact_edge_only_when_registered_conjunction": True,
        "evidence_excluded": ["SCC", "memory", "biology", "consciousness", "AGI"],
        "phase_a_synthetic_identification_only": True,
    }:
        raise ValueError("allowed claim boundary differs from registration")
    dof = payload["dof_accounting"]
    if not isinstance(dof, dict) or set(dof) != {
        "factorized_formula",
        "factorized_minus_pooled_formula",
        "pooled_formula",
        "report_effective_dof",
    }:
        raise ValueError("dof accounting fields must be exact")
    if dof != {
        "factorized_formula": "n*(K*n+m)",
        "factorized_minus_pooled_formula": "(K-1)*n^2",
        "pooled_formula": "n*(n+m)",
        "report_effective_dof": True,
    }:
        raise ValueError("dof accounting differs from registration")
    artifacts = payload["required_artifacts"]
    if verify_files:
        verified_artifacts = _verify_required_artifacts(resolved_root, artifacts)
    else:
        if not isinstance(artifacts, list):
            raise ValueError("required_artifacts must be a list")
        verified_artifacts = {}
    return {
        "bootstrap": dict(bootstrap),
        "confirmation": confirmation,
        "development_graph_seeds": development,
        "dimensionless": dimensionless,
        "generator": dict(generator),
        "manifest_sha256": expected_manifest_digest,
        "pilot_graph_seeds": pilot,
        "required_artifacts": verified_artifacts,
    }


def load_and_validate_manifest(path: Path, *, root: Path = REPO_ROOT) -> dict[str, Any]:
    resolved_root = root.resolve(strict=True)
    canonical = _repository_path(
        resolved_root, str(MANIFEST_RELATIVE), must_exist=True
    )
    if path.resolve(strict=True) != canonical:
        raise ValueError(f"manifest must be {MANIFEST_RELATIVE}")
    text = canonical.read_text(encoding="utf-8")
    payload = _json_without_duplicate_keys(text, source=str(MANIFEST_RELATIVE))
    validated = validate_manifest_payload(payload, root=resolved_root)
    validated["manifest_payload"] = payload
    validated["manifest_file_sha256"] = _sha256_bytes(canonical.read_bytes())
    return validated


def _isolated_load(path: Path, expected_sha256: str) -> tuple[ModuleType, str]:
    """Compile and execute the exact immutable byte buffer that was hashed."""

    resolved = path.resolve(strict=True)
    source = resolved.read_bytes()
    digest = _sha256_bytes(source)
    if digest != expected_sha256:
        raise ValueError("production bytes changed after manifest validation")
    name = f"_ce_phase_a_development_{digest}"
    if name in sys.modules:
        raise RuntimeError("isolated production module was already loaded")
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None:
        raise ImportError("cannot construct isolated production module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(source, str(resolved), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    module.__ce_loaded_sha256__ = digest
    return module, digest


def _reserve_output_once(output: Path) -> None:
    if not output.parent.is_dir():
        raise FileNotFoundError("development result parent must already exist")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("stale temporary result exists; manual audit required")
    with output.open("xb") as reservation:
        reservation.flush()
        os.fsync(reservation.fileno())


def _finalize_reserved_output(output: Path, payload: object) -> None:
    if not output.is_file() or output.stat().st_size != 0:
        raise RuntimeError("one-shot output reservation is missing or nonempty")
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("stale temporary result exists; manual audit required")
    serialized = _canonical_bytes(payload) + b"\n"
    try:
        with temporary.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        # Preserve the exclusive output reservation and any temporary evidence.
        raise


def _execute_one_shot_reserved(
    output: Path, evaluator: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """Reserve before evaluator entry and retain zero-byte evidence on failure."""

    _reserve_output_once(output)
    payload = evaluator()
    _finalize_reserved_output(output, payload)
    return payload


def _build_result_payload(
    registration: Mapping[str, Any],
    development_result: Mapping[str, Any],
    *,
    loaded_sha256: str,
) -> dict[str, Any]:
    confirmation_status = "reserved_unopened"
    return {
        "confirmation": {
            "custody_status": "custody_unverified",
            "execution_authorized": False,
            "holdout_status": "not_executable_holdout",
            "raw_seed_material_present": False,
            "reservation_kind": "reservation_only",
            "status": confirmation_status,
        },
        "confirmation_status": confirmation_status,
        "development": dict(development_result),
        "environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "implementation_sha256": loaded_sha256,
        "manifest_file_sha256": registration["manifest_file_sha256"],
        "manifest_sha256": registration["manifest_sha256"],
        "mode": "development",
        "required_artifact_sha256": registration["required_artifacts"],
        "schema": "ce.causal_recurrent_geometry.phase_a.one-shot-result.v1",
    }


def run_registered_development(
    manifest_path: Path, *, root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Execute the one registered development block exactly once."""

    resolved_root = root.resolve(strict=True)
    registration = load_and_validate_manifest(manifest_path, root=resolved_root)
    output = _repository_path(resolved_root, str(RESULT_RELATIVE), must_exist=False)

    def evaluate_after_reservation() -> dict[str, Any]:
        production_sha256 = registration["required_artifacts"][
            str(PRODUCTION_RELATIVE)
        ]
        production_path = _repository_path(
            resolved_root, str(PRODUCTION_RELATIVE), must_exist=True
        )
        production, loaded_sha256 = _isolated_load(
            production_path, production_sha256
        )
        dimensions = production.dimensionless_certificate(
            registration["dimensionless"]
        )
        if not dimensions.passed:
            raise ValueError("dimensionless certificate failed")
        config = production.PhaseAConfig(**registration["generator"])
        result = production.run_development_benchmark(
            config,
            graph_seeds=registration["development_graph_seeds"],
            registered_development_graph_seeds=registration[
                "development_graph_seeds"
            ],
            bootstrap_seed=registration["bootstrap"]["seed"],
            bootstrap_samples=registration["bootstrap"]["samples"],
        )
        # Close the read/execute interval before finalizing the reserved result.
        closing = load_and_validate_manifest(manifest_path, root=resolved_root)
        if closing["manifest_sha256"] != registration["manifest_sha256"]:
            raise RuntimeError("manifest changed during development execution")
        if closing["manifest_file_sha256"] != registration["manifest_file_sha256"]:
            raise RuntimeError("manifest bytes changed during development execution")
        if _sha256_path(production_path) != loaded_sha256:
            raise RuntimeError("production source changed during development execution")
        return _build_result_payload(
            registration, result, loaded_sha256=loaded_sha256
        )

    return _execute_one_shot_reserved(output, evaluate_after_reservation)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit(
            "usage: causal_recurrent_geometry_development_run.py <manifest.json>"
        )
    payload = run_registered_development(Path(arguments[0]))
    summary = {
        "PA-H1": payload["development"]["claim_status"]["PA-H1"],
        "PA-H2": payload["development"]["claim_status"]["PA-H2"],
        "mode": payload["mode"],
        "result_path": str(RESULT_RELATIVE),
    }
    print(_canonical_bytes(summary).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
