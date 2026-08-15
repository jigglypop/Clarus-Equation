from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (
    ROOT
    / "_workspace"
    / "ce"
    / "agi-v16-covariant-metric-flow-20260813"
    / "artifacts"
    / "run_v16_benchmark.py"
)


def _load() -> dict[str, Any]:
    return runpy.run_path(str(EVALUATOR))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed_fixture(
    tmp_path: Path,
    namespace: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "repo"
    required = namespace["REQUIRED_MANIFEST_PATHS"]
    for relative in required:
        path = root.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    rates = root.joinpath(*namespace["RATES_RELATIVE"].parts)
    rates.write_text(
        json.dumps({"additive": 0.2, "conformal": 0.05, "v16": 0.4}),
        encoding="utf-8",
    )
    development = root.joinpath(*namespace["DEVELOPMENT_RESULT_RELATIVE"].parts)
    development.write_text(
        json.dumps(
            {
                "mode": "development",
                "learners": {
                    name: {
                        "selected_rate": rate,
                        "rates": {
                            str(candidate): {
                                "mean_normalized_regret": abs(candidate - rate)
                            }
                            for candidate in namespace["RATES"]
                        },
                    }
                    for name, rate in {
                        "additive": 0.2,
                        "conformal": 0.05,
                        "v16": 0.4,
                    }.items()
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = root.joinpath(*namespace["MANIFEST_RELATIVE"].parts)
    manifest.write_text(
        json.dumps(
            {
                relative: _sha256(root.joinpath(*Path(relative).parts))
                for relative in sorted(required)
            }
        ),
        encoding="utf-8",
    )

    globals_ = namespace["verify_manifest"].__globals__
    production = root.joinpath(*namespace["PRODUCTION_RELATIVE"].parts)
    evaluator = root.joinpath(
        *(namespace["RUN_RELATIVE"] / "artifacts/run_v16_benchmark.py").parts
    )
    monkeypatch.setitem(globals_, "REPO_ROOT", root.resolve())
    monkeypatch.setitem(globals_, "IMPORTED_PRODUCTION_PATH", production.resolve())
    monkeypatch.setitem(globals_, "__file__", str(evaluator.resolve()))
    return root, manifest, rates


def test_manifest_requires_every_bound_artifact_and_rejects_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path, _ = _sealed_fixture(tmp_path, namespace, monkeypatch)
    verify = namespace["verify_manifest"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(next(iter(namespace["REQUIRED_MANIFEST_PATHS"])))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required paths"):
        verify(root, manifest_path)

    root, manifest_path, _ = _sealed_fixture(tmp_path / "second", namespace, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["../outside"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid manifest path"):
        verify(root, manifest_path)


def test_confirmation_rejects_unsealed_rate_path_before_opening_seed_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path, _ = _sealed_fixture(tmp_path, namespace, monkeypatch)
    alternate = root / "alternate-rates.json"
    alternate.write_text(
        json.dumps({"additive": 0.2, "conformal": 0.05, "v16": 0.4}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rates must be"):
        namespace["confirmation"](root, manifest_path, alternate)

    receipt = root.joinpath(*namespace["RECEIPT_RELATIVE"].parts)
    assert not receipt.exists()


def test_confirmation_receipt_is_atomic_and_blocks_a_second_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path, rates_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    globals_ = namespace["confirmation"].__globals__
    monkeypatch.setitem(globals_, "CONFIRMATION_SEEDS", range(2))

    def fake_aggregate(seeds: range, learner: str, rate: float) -> dict[str, float | int]:
        del seeds, rate
        regret = {"v16": 0.001, "additive": 0.01, "conformal": 0.3, "identity": 0.3}[
            learner
        ]
        online = {"v16": 0.1, "additive": 0.2, "conformal": 0.3, "identity": 0.3}[
            learner
        ]
        return {
            "seeds": 2,
            "finite_episode_rate": 1.0,
            "route_accuracy": 0.99,
            "mean_normalized_regret": regret,
            "median_invariant_metric_error": 0.1,
            "mean_online_regret_after_32": online,
        }

    def fake_chart(seed: int, rate: float) -> dict[str, float | bool]:
        del seed, rate
        return {
            "finite": True,
            "action_matches": 128,
            "action_count": 128,
            "max_relative_prediction_error": 0.0,
            "relative_metric_transport_error": 0.0,
            "target_transport_identity_error": 0.0,
        }

    monkeypatch.setitem(globals_, "aggregate", fake_aggregate)
    monkeypatch.setitem(globals_, "chart_episode", fake_chart)
    first = namespace["confirmation"](root, manifest_path, rates_path)

    assert first["learning_chart_closed_loop_pass"]
    assert root.joinpath(*namespace["RECEIPT_RELATIVE"].parts).is_file()
    assert root.joinpath(*namespace["RESULT_RELATIVE"].parts).is_file()
    with pytest.raises(RuntimeError, match="already exists|already opened"):
        namespace["confirmation"](root, manifest_path, rates_path)


def test_development_chart_fixture_exercises_nested_routes() -> None:
    result = _load()["chart_episode"](917_000, 0.4)

    assert result["finite"]
    assert result["action_matches"] == result["action_count"] == 128
    assert float(result["max_relative_prediction_error"]) <= 1.0e-10
