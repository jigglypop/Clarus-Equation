"""One-shot confirmation runner for the frozen V10 local/cloud development."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

from reality_stone.clarus.local_cloud_benchmark import (
    LocalCloudBenchmarkConfig,
    evaluate_registered_development,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _canonical(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_registration(registration: dict[str, object], repository: Path) -> tuple[int, ...]:
    if registration.get("schema") != "clarus.local_cloud.confirmation.v1":
        raise ValueError("unknown confirmation registration schema")
    development_path = repository / str(registration["development_result_path"])
    if _sha256(development_path) != registration["development_result_sha256"]:
        raise ValueError("development result hash mismatch")
    development = json.loads(development_path.read_text(encoding="utf-8"))
    development_registration = development["registration"]
    for relative, expected in dict(registration["sha256"]).items():
        actual = _sha256(repository / relative)
        if actual != expected:
            raise ValueError(f"hash mismatch for {relative}: {actual} != {expected}")
    for relative, expected in development_registration["sha256"].items():
        if relative.endswith("local_cloud_development_run.py"):
            continue
        if registration["sha256"].get(relative) != expected:
            raise ValueError(f"confirmation changed a development-locked file: {relative}")
    seeds = tuple(registration["confirmation_seeds"])
    reserved = tuple(development_registration["reserved_confirmation_seeds"])
    if seeds != reserved:
        raise ValueError("confirmation seeds must exactly equal the pre-development reservation")
    if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("confirmation seeds must be unique exact nonnegative integers")
    if set(seeds).intersection(development_registration["development_seeds"]):
        raise ValueError("confirmation overlaps development seeds")
    if set(seeds).intersection(development_registration["burned_diagnostic_seeds"]):
        raise ValueError("confirmation overlaps burned diagnostic seeds")
    if registration["config"] != development_registration["config"]:
        raise ValueError("confirmation config differs from development")
    if registration["bootstrap_samples"] != development_registration["bootstrap_samples"]:
        raise ValueError("confirmation bootstrap sample count differs from development")
    if registration["bootstrap_seed"] != development_registration["bootstrap_seed"]:
        raise ValueError("confirmation bootstrap seed differs from development")
    return seeds


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: local_cloud_confirmation_run.py <registration.json>")
    registration_path = Path(sys.argv[1]).resolve()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    repository = Path(registration["repository_root"]).resolve()
    if repository != Path.cwd().resolve():
        raise ValueError("runner must execute from the registered repository root")
    seeds = _validate_registration(registration, repository)
    output = (repository / registration["result_path"]).resolve()
    if output.exists():
        raise FileExistsError("registered confirmation result already exists; rerun forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = LocalCloudBenchmarkConfig(**registration["config"])
    result = evaluate_registered_development(
        seeds,
        config,
        bootstrap_samples=registration["bootstrap_samples"],
        bootstrap_seed=registration["bootstrap_seed"],
    )
    payload = {
        "schema": "clarus.local_cloud.confirmation.result.v1",
        "registration_sha256": _sha256(registration_path),
        "registration": registration,
        "result": asdict(result),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(_canonical(payload) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(_canonical({"overall": result.overall, "result_path": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
