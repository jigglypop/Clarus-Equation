"""One-shot registered V11 learned-comparator/OOD development runner."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

from reality_stone.clarus.local_cloud_ood_benchmark import (
    LearnedOODConfig,
    evaluate_ood_development,
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


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: local_cloud_ood_run.py <registration.json>")
    registration_path = Path(sys.argv[1]).resolve()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema") != "clarus.local_cloud.ood_development.v1":
        raise ValueError("unknown V11 registration schema")
    repository = Path(registration["repository_root"]).resolve()
    if repository != Path.cwd().resolve():
        raise ValueError("runner must execute from the registered repository root")
    for relative, expected in registration["sha256"].items():
        actual = _sha256(repository / relative)
        if actual != expected:
            raise ValueError(f"hash mismatch for {relative}: {actual} != {expected}")
    seeds = tuple(registration["development_seeds"])
    forbidden = set(registration["all_prior_seeds"])
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("development seeds must be exact nonnegative integers")
    if len(set(seeds)) != len(seeds) or forbidden.intersection(seeds):
        raise ValueError("V11 seeds are duplicate or overlap prior roles")
    output = (repository / registration["result_path"]).resolve()
    if output.exists():
        raise FileExistsError("registered V11 result already exists; rerun forbidden")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate_ood_development(
        seeds,
        LearnedOODConfig(**registration["config"]),
        bootstrap_samples=registration["bootstrap_samples"],
        bootstrap_seed=registration["bootstrap_seed"],
    )
    payload = {
        "schema": "clarus.local_cloud.ood_development.result.v1",
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
