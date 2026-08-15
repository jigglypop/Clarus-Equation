from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (
    ROOT
    / "_workspace"
    / "ce"
    / "agi-v17-metric-delayed-credit-20260813"
    / "artifacts"
    / "run_v17_benchmark.py"
)


def _load() -> dict[str, Any]:
    return runpy.run_path(str(EVALUATOR))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _development_fixture(namespace: dict[str, Any]) -> dict[str, Any]:
    seeds = namespace["DEVELOPMENT_SEEDS"]
    return {
        "mode": "development",
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "protocol": namespace["_normalised_protocol"](namespace["FIXED_PROTOCOL"]),
        "summary": {},
        "per_seed": [{"seed": seed} for seed in seeds],
    }


def _sealed_fixture(
    tmp_path: Path,
    namespace: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    required = namespace["REQUIRED_MANIFEST_PATHS"]
    for relative in required:
        path = root.joinpath(*Path(relative).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture for {relative}\n", encoding="utf-8")

    development = root.joinpath(*namespace["DEVELOPMENT_RESULT_RELATIVE"].parts)
    development.write_text(
        json.dumps(_development_fixture(namespace)),
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
    evaluator = root.joinpath(*namespace["EVALUATOR_RELATIVE"].parts)
    monkeypatch.setitem(globals_, "REPO_ROOT", root.resolve())
    monkeypatch.setitem(globals_, "IMPORTED_PRODUCTION_PATH", production.resolve())
    monkeypatch.setitem(globals_, "__file__", str(evaluator.resolve()))
    return root, manifest


def test_lossless_state_serialization_distinguishes_signed_zero() -> None:
    namespace = _load()

    @dataclass(frozen=True)
    class State:
        factor: tuple[tuple[float, ...], ...]

    positive = State(((0.0,),))
    negative = State(((-0.0,),))

    assert namespace["serialize_state"](positive) != namespace["serialize_state"](negative)


def test_development_draw_is_deterministic_and_uses_registered_chart_range() -> None:
    namespace = _load()
    first_cue, first_chart = namespace["episode_inputs"](1_719_000)
    second_cue, second_chart = namespace["episode_inputs"](1_719_000)

    assert np.array_equal(first_cue, second_cue)
    assert np.array_equal(first_chart, second_chart)
    assert np.linalg.norm(first_cue) == pytest.approx(1.0, abs=2e-15)
    singular_values = np.linalg.svd(first_chart, compute_uv=False)
    assert np.min(singular_values) >= 0.25 - 1e-14
    assert np.max(singular_values) <= 4.0 + 1e-14


def test_independent_reference_decision_closes_strict_and_lift_dev_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    memory_type = namespace["HomogeneousSignedCue"]

    # The evaluator must not delegate the scored decision to the production
    # readout helper.
    monkeypatch.setattr(
        memory_type,
        "readout",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("readout used")),
    )
    result = namespace["score_seed"](1_719_000)

    assert result["finite"]
    assert result["strict"]["serialized_state_equal"]
    assert result["strict"]["balanced_accuracy"] == 0.5
    assert result["strict"]["balanced_regret"] == 0.5
    assert all(
        item["serialized_aggregate_equal"] for item in result["strict"]["ensembles"].values()
    )
    for branch in result["lift"]:
        sign = branch["sign"]
        assert branch["selected_action"] == sign
        assert branch["charted_selected_action"] == sign
        assert branch["reference_costs"][str(sign)] == 2.0
        assert branch["reference_costs"][str(-sign)] == 4.0
        assert branch["wrong_minus_correct_margin"] >= 1.999999999
        assert branch["max_relative_quadratic_cost_defect"] <= 1e-10


def test_public_export_is_bound_to_production_type() -> None:
    namespace = _load()
    import reality_stone.clarus as clarus

    assert clarus.HomogeneousSignedCue is namespace["HomogeneousSignedCue"]


def test_manifest_requires_exact_bound_artifacts_and_rejects_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    verify = namespace["verify_manifest"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop(next(iter(namespace["REQUIRED_MANIFEST_PATHS"])))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest paths must be exact"):
        verify(root, manifest_path)

    root, manifest_path = _sealed_fixture(tmp_path / "second", namespace, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["../outside"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid manifest path"):
        verify(root, manifest_path)

    root, manifest_path = _sealed_fixture(tmp_path / "third", namespace, monkeypatch)
    extra = root / "extra.txt"
    extra.write_text("not registered\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["extra.txt"] = _sha256(extra)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest paths must be exact"):
        verify(root, manifest_path)


def test_confirmation_rejects_tampered_fixed_config_before_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    development_path = root.joinpath(*namespace["DEVELOPMENT_RESULT_RELATIVE"].parts)
    development = json.loads(development_path.read_text(encoding="utf-8"))
    development["protocol"]["eta"] = 0.5
    development_path.write_text(json.dumps(development), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[str(namespace["DEVELOPMENT_RESULT_RELATIVE"])] = _sha256(development_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed protocol"):
        namespace["confirmation"](root, manifest_path)
    assert not root.joinpath(*namespace["RECEIPT_RELATIVE"].parts).exists()


def _fake_confirmation_result(seeds: range) -> dict[str, Any]:
    ensembles = {
        str(size): {
            "serialized_aggregate_equality_rate": 1.0,
            "action_distribution_equality_rate": 1.0,
            "balanced_accuracy": 0.5,
            "balanced_regret": 0.5,
        }
        for size in (1, 2, 4, 8, 16, 64)
    }
    return {
        "mode": "confirmation",
        "seed_start": seeds.start,
        "seed_stop_exclusive": seeds.stop,
        "protocol": {},
        "summary": {
            "seed_count": len(seeds),
            "finite_seed_rate": 1.0,
            "strict": {
                "paired_seed_count": len(seeds),
                "finite_run_rate": 1.0,
                "serialized_state_equality_rate": 1.0,
                "action_distribution_equality_rate": 1.0,
                "balanced_accuracy": 0.5,
                "balanced_regret": 0.5,
                "ensembles": ensembles,
            },
            "lift": {
                "branch_count": 2 * len(seeds),
                "finite_run_rate": 1.0,
                "action_accuracy": 1.0,
                "mean_regret": 0.0,
                "minimum_wrong_minus_correct_margin": 2.0,
                "charted_action_agreement": 1.0,
                "max_relative_quadratic_cost_defect": 1e-15,
            },
            "state_certificate": {
                "persistent_state_fields": ["factor"],
                "persistent_state_field_count": 1,
                "factor_shape": [4, 4],
                "lower_triangular_coordinate_count": 10,
                "certificate_persistent_state_field_count": 1,
                "certificate_ambient_real_state_coordinates": 10,
                "certificate_optimizer_state_field_count": 0,
            },
        },
        "per_seed": [{"seed": seed} for seed in seeds],
    }


def test_confirmation_receipt_precedes_seed_access_and_blocks_second_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    globals_ = namespace["confirmation"].__globals__
    harmless_test_seeds = range(11, 13)
    monkeypatch.setitem(globals_, "CONFIRMATION_SEEDS", harmless_test_seeds)

    def fake_evaluate(
        seeds: range,
        mode: str,
        *,
        confirmation_access: Any = None,
    ) -> dict[str, Any]:
        assert seeds is harmless_test_seeds
        assert mode == "confirmation"
        assert confirmation_access is not None
        receipt = root.joinpath(*namespace["RECEIPT_RELATIVE"].parts)
        assert receipt.is_file(), "seed evaluator ran before exclusive receipt"
        return _fake_confirmation_result(seeds)

    monkeypatch.setitem(globals_, "evaluate", fake_evaluate)
    first = namespace["confirmation"](root, manifest_path)

    assert first["strict_no_go_pass"]
    assert first["homogeneous_lift_pass"]
    result_path = root.joinpath(*namespace["RESULT_RELATIVE"].parts)
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["per_seed"] == [{"seed": 11}, {"seed": 12}]
    with pytest.raises(RuntimeError, match="already exists|already opened"):
        namespace["confirmation"](root, manifest_path)


def test_import_and_evaluator_are_bound_to_canonical_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    globals_ = namespace["verify_manifest"].__globals__

    monkeypatch.setitem(globals_, "IMPORTED_PRODUCTION_PATH", (root / "elsewhere.py"))
    with pytest.raises(ValueError, match="imported production module"):
        namespace["verify_manifest"](root, manifest_path)


def test_confirmation_seed_helpers_fail_without_receipt_capability() -> None:
    namespace = _load()

    with pytest.raises(RuntimeError, match="opening receipt"):
        namespace["episode_inputs"](1_720_000)
    with pytest.raises(RuntimeError, match="opening receipt"):
        namespace["score_seed"](1_720_000)
    with pytest.raises(RuntimeError, match="receipt-bound"):
        namespace["evaluate"](namespace["CONFIRMATION_SEEDS"], "confirmation")


def test_direct_confirmation_open_requires_verified_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)

    with pytest.raises(RuntimeError, match="sealed preflight"):
        namespace["open_confirmation_block"](root, manifest_path)
    assert not root.joinpath(*namespace["RECEIPT_RELATIVE"].parts).exists()


def test_invalid_manifest_cannot_obtain_preflight_or_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    manifest_path.write_text('{"not/a/bound/path":"' + "0" * 64 + '"}', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest paths must be exact"):
        namespace["confirmation"](root, manifest_path)
    assert not root.joinpath(*namespace["RECEIPT_RELATIVE"].parts).exists()

    with pytest.raises(RuntimeError, match="sealed preflight"):
        namespace["open_confirmation_block"](root, manifest_path)
    assert not root.joinpath(*namespace["RECEIPT_RELATIVE"].parts).exists()


def test_forged_receipt_capability_cannot_open_confirmation_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    globals_ = namespace["episode_inputs"].__globals__
    harmless_test_seeds = range(31, 32)
    monkeypatch.setitem(globals_, "CONFIRMATION_SEEDS", harmless_test_seeds)
    fake_receipt = tmp_path / "receipt.json"
    fake_manifest = tmp_path / "manifest.json"
    fake_result = tmp_path / "result.json"
    fake_receipt.write_text("{}", encoding="utf-8")
    fake_manifest.write_text("{}", encoding="utf-8")
    forged = namespace["_ConfirmationAccess"](
        root=tmp_path,
        receipt=fake_receipt,
        result=fake_result,
        manifest=fake_manifest,
        manifest_sha256=_sha256(fake_manifest),
    )

    with pytest.raises(RuntimeError, match="opening receipt"):
        namespace["episode_inputs"](31, confirmation_access=forged)


def test_closing_manifest_verification_fails_closed_after_midrun_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    globals_ = namespace["confirmation"].__globals__
    harmless_test_seeds = range(21, 23)
    monkeypatch.setitem(globals_, "CONFIRMATION_SEEDS", harmless_test_seeds)
    bound_path = root.joinpath(*namespace["PRODUCTION_RELATIVE"].parts)
    original_bound_bytes = bound_path.read_bytes()

    def mutate_during_evaluation(
        seeds: range,
        mode: str,
        *,
        confirmation_access: Any = None,
    ) -> dict[str, Any]:
        assert confirmation_access is not None
        bound_path.write_text("mutated during run\n", encoding="utf-8")
        return _fake_confirmation_result(seeds)

    monkeypatch.setitem(globals_, "evaluate", mutate_during_evaluation)
    with pytest.raises(ValueError, match="manifest mismatch"):
        namespace["confirmation"](root, manifest_path)

    assert root.joinpath(*namespace["RECEIPT_RELATIVE"].parts).is_file()
    assert not root.joinpath(*namespace["RESULT_RELATIVE"].parts).exists()
    bound_path.write_bytes(original_bound_bytes)
    with pytest.raises(RuntimeError, match="already opened"):
        namespace["confirmation"](root, manifest_path)


def test_failed_evaluation_consumes_in_memory_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load()
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    globals_ = namespace["confirmation"].__globals__
    harmless_test_seeds = range(41, 42)
    monkeypatch.setitem(globals_, "CONFIRMATION_SEEDS", harmless_test_seeds)
    captured: list[Any] = []

    def failing_evaluate(
        seeds: range,
        mode: str,
        *,
        confirmation_access: Any = None,
    ) -> dict[str, Any]:
        del seeds, mode
        captured.append(confirmation_access)
        raise FloatingPointError("deliberate evaluator failure")

    monkeypatch.setitem(globals_, "evaluate", failing_evaluate)
    with pytest.raises(FloatingPointError, match="deliberate"):
        namespace["confirmation"](root, manifest_path)
    assert captured and captured[0] is not None

    with pytest.raises(RuntimeError, match="opening receipt"):
        namespace["episode_inputs"](41, confirmation_access=captured[0])
