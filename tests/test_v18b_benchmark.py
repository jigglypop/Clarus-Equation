from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = (
    ROOT
    / "_workspace"
    / "ce"
    / "agi-v18b-learned-delayed-credit-20260814"
    / "artifacts"
    / "run_v18b_benchmark.py"
)


@pytest.fixture(scope="module")
def namespace() -> dict[str, Any]:
    return runpy.run_path(str(EVALUATOR))


@pytest.fixture(scope="module")
def scored_development_seed(namespace: dict[str, Any]) -> dict[str, Any]:
    # Exactly one registered development seed is intentionally exercised.  No
    # confirmation-range value is passed to a seed generator in this suite.
    result = namespace["score_seed"](namespace["DEVELOPMENT_SEEDS"].start)
    assert result.get("finite"), result.get("error")
    return result


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
    development.write_text(json.dumps(_development_fixture(namespace)), encoding="utf-8")
    manifest = root.joinpath(*namespace["MANIFEST_RELATIVE"].parts)
    manifest.parent.mkdir(parents=True, exist_ok=True)
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


@pytest.mark.skip(reason="seal pins clarus/__init__.py bytes; the package initializer was intentionally restructured (MULTIREPO_PLAN.md P2-3) and the v18b run is ABANDONED — see _workspace/ce/agi-v18b-learned-delayed-credit-20260814/40-final-report.md")
def test_production_is_isolated_from_package_initializer(namespace: dict[str, Any]) -> None:
    module = namespace["PRODUCTION_MODULE"]
    expected = ROOT.joinpath(*namespace["PRODUCTION_RELATIVE"].parts).resolve()

    assert Path(module.__file__).resolve() == expected
    assert module.__name__ == "_ce_v18b_sealed_delayed_linear_credit"
    assert module.EligibilityLearner is namespace["EligibilityLearner"]
    assert "reality_stone.clarus" not in namespace["EligibilityLearner"].__module__


def test_lossless_state_serialization_distinguishes_signed_zero(
    namespace: dict[str, Any],
) -> None:
    @dataclass(frozen=True)
    class State:
        classifier: tuple[float, ...]

    assert namespace["serialize_state"](State((0.0,))) != namespace["serialize_state"](
        State((-0.0,))
    )


def test_seed_namespaces_are_deterministic_and_separate(namespace: dict[str, Any]) -> None:
    seed = namespace["DEVELOPMENT_SEEDS"].start
    first = namespace["_seed_namespaces"](seed, None)
    second = namespace["_seed_namespaces"](seed, None)
    names = (
        "teacher",
        "epoch-order",
        "cue-sign",
        "training-delay",
        "training-distractor",
        "evaluation-query",
        "evaluation-distractor",
        "training-lesion",
        "evaluation-update",
        "evaluation-ensemble",
        "evaluation-policy",
    )
    first_digests = [first.digest(name) for name in names]

    assert first_digests == [second.digest(name) for name in names]
    assert len(set(first_digests)) == len(names)


def test_confirmation_authority_fails_closed_without_using_real_block(
    namespace: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    globals_ = namespace["_authorise_seed"].__globals__
    harmless_range = range(91, 92)
    monkeypatch.setitem(globals_, "CONFIRMATION_SEEDS", harmless_range)

    with pytest.raises(RuntimeError, match="opening receipt"):
        namespace["_authorise_seed"](91, None)
    with pytest.raises(ValueError, match="outside"):
        namespace["_authorise_seed"](90, None)


def test_query_filter_uses_integer_margin_and_finite_redraw_cap(
    namespace: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    theta = np.ones(namespace["DIMENSION"], dtype=np.float64)
    seed = namespace["DEVELOPMENT_SEEDS"].start
    namespaces = namespace["_seed_namespaces"](seed, None)
    queries = namespace["_accepted_queries"](namespaces, theta)

    assert len(queries) == namespace["QUERY_PAIR_COUNT"]
    assert all(query.integer_margin != 0 for query in queries)
    assert all(
        query.integer_margin
        == int(np.asarray(query.rademacher, dtype=np.int64) @ theta.astype(np.int64))
        for query in queries
    )

    globals_ = namespace["_accepted_queries"].__globals__
    monkeypatch.setitem(globals_, "QUERY_PAIR_COUNT", 1)
    monkeypatch.setitem(globals_, "QUERY_REDRAW_CAP", 2)
    balanced = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)
    monkeypatch.setitem(globals_, "_rademacher", lambda rng: balanced.copy())
    with pytest.raises(RuntimeError, match="redraw cap exhausted"):
        namespace["_accepted_queries"](namespaces, theta)


def test_one_development_seed_closes_every_numeric_gate(
    namespace: dict[str, Any], scored_development_seed: dict[str, Any]
) -> None:
    gates = namespace["_gate_decisions"]([scored_development_seed])

    assert all(gates.values()), gates
    assert len(scored_development_seed["queries"]) == 128
    assert scored_development_seed["state_certificate"][
        "homogeneous_independent_factor_coordinates"
    ] == 45
    assert scored_development_seed["state_certificate"][
        "homogeneous_dense_serialized_entries"
    ] == 81


def test_all_32_training_timing_checks_and_exact_reference_updates_are_retained(
    scored_development_seed: dict[str, Any],
) -> None:
    for route in ("eligibility", "homogeneous"):
        records = scored_development_seed["training"][route]["timing"]
        assert len(records) == 32
        assert sum(record["checked_distractor_count"] for record in records) == sum(
            record["delay"] for record in records
        )
        assert all(
            record["w_after_cue_equal"]
            and record["w_after_every_distractor_equal"]
            and record["w_pre_reward_equal"]
            and record["update_match"]
            and record["atomic_reset"]
            for record in records
        )


def test_complete_paired_trajectory_is_shared_and_strict_is_pointwise_half(
    namespace: dict[str, Any], scored_development_seed: dict[str, Any]
) -> None:
    strict = scored_development_seed["evaluation"]["strict"]
    for delay in namespace["EVALUATION_DELAYS"]:
        result = strict[str(delay)]
        assert result["accuracy"] == 0.5
        assert result["all_state_serializations_equal"]
        assert result["all_checkpoint_serializations_equal"]
        assert all(pair["equal"] for pair in result["state_pairs"])
        for size in namespace["ENSEMBLE_SIZES"]:
            ensemble = result["ensembles"][str(size)]
            assert ensemble["accuracy"] == 0.5
            assert ensemble["all_aggregate_serializations_equal"]

    # The recorded hash is generated from the full nuisance tuple, and a
    # deterministic regeneration gives the same byte-level trajectory.
    seed = scored_development_seed["seed"]
    namespaces = namespace["_seed_namespaces"](seed, None)
    first = namespace["_paired_nuisance"](namespaces, 0, 128)
    second = namespace["_paired_nuisance"](namespaces, 0, 128)
    assert first == second
    assert first["sha256"] == second["sha256"]
    assert len(first["distractors"]) == 128
    assert len(first["messages"]) == 129


def test_gate_fails_closed_when_one_timing_episode_is_missing(
    namespace: dict[str, Any], scored_development_seed: dict[str, Any]
) -> None:
    altered = json.loads(json.dumps(scored_development_seed))
    altered["training"]["eligibility"]["timing"].pop()

    gates = namespace["_gate_decisions"]([altered])
    assert not gates["all_32_classifier_timing_checks"]


@pytest.mark.skip(reason="seal pins clarus/__init__.py bytes; the package initializer was intentionally restructured (MULTIREPO_PLAN.md P2-3) and the v18b run is ABANDONED — see _workspace/ce/agi-v18b-learned-delayed-credit-20260814/40-final-report.md")
def test_manifest_requires_exact_five_paths_and_rejects_duplicate_or_traversal(
    tmp_path: Path,
    namespace: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    verified = namespace["verify_manifest"](root, manifest_path)
    assert set(verified) == namespace["REQUIRED_MANIFEST_PATHS"]
    assert len(verified) == 5

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["../outside"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid manifest path"):
        namespace["verify_manifest"](root, manifest_path)

    root, manifest_path = _sealed_fixture(tmp_path / "duplicate", namespace, monkeypatch)
    relative, digest = next(iter(json.loads(manifest_path.read_text()).items()))
    manifest_path.write_text(
        '{"' + relative + '":"' + digest + '","' + relative + '":"' + digest + '"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        namespace["verify_manifest"](root, manifest_path)


def test_repository_module_closure_rejects_unsealed_loaded_file(
    tmp_path: Path, namespace: dict[str, Any]
) -> None:
    root = tmp_path / "repo"
    allowed = root.joinpath(*namespace["PRODUCTION_RELATIVE"].parts)
    forbidden = root / "reality_stone/python/reality_stone/clarus/hidden_helper.py"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("allowed\n", encoding="utf-8")
    forbidden.write_text("hidden\n", encoding="utf-8")
    modules = [
        ("allowed", SimpleNamespace(__file__=str(allowed))),
        ("hidden", SimpleNamespace(__file__=str(forbidden))),
        ("stdlib", SimpleNamespace(__file__=str(Path(np.__file__).resolve()))),
    ]

    violations = namespace["_loaded_repository_module_violations"](
        root, modules=modules
    )
    assert violations == [
        "hidden:reality_stone/python/reality_stone/clarus/hidden_helper.py"
    ]


@pytest.mark.skip(reason="seal pins clarus/__init__.py bytes; the package initializer was intentionally restructured (MULTIREPO_PLAN.md P2-3) and the v18b run is ABANDONED — see _workspace/ce/agi-v18b-learned-delayed-credit-20260814/40-final-report.md")
def test_confirmation_receipt_is_exclusive_and_precedes_fake_seed_evaluation(
    tmp_path: Path,
    namespace: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    globals_ = namespace["confirmation"].__globals__
    harmless_range = range(71, 73)
    monkeypatch.setitem(globals_, "CONFIRMATION_SEEDS", harmless_range)

    def fake_evaluate(
        seeds: range,
        mode: str,
        *,
        confirmation_access: Any = None,
    ) -> dict[str, Any]:
        assert seeds is harmless_range
        assert mode == "confirmation"
        assert confirmation_access is not None
        assert root.joinpath(*namespace["RECEIPT_RELATIVE"].parts).is_file()
        return {
            "mode": mode,
            "seed_start": seeds.start,
            "seed_stop_exclusive": seeds.stop,
            "protocol": namespace["_normalised_protocol"](namespace["FIXED_PROTOCOL"]),
            "summary": {},
            "gates": {},
            "per_seed": [{"seed": seed} for seed in seeds],
        }

    monkeypatch.setitem(globals_, "evaluate", fake_evaluate)
    result = namespace["confirmation"](root, manifest_path)
    assert result["manifest_verified"]
    result_path = root.joinpath(*namespace["RESULT_RELATIVE"].parts)
    assert result_path.is_file()
    with pytest.raises(RuntimeError, match="already exists|already opened|closed"):
        namespace["confirmation"](root, manifest_path)


@pytest.mark.skip(reason="seal pins clarus/__init__.py bytes; the package initializer was intentionally restructured (MULTIREPO_PLAN.md P2-3) and the v18b run is ABANDONED — see _workspace/ce/agi-v18b-learned-delayed-credit-20260814/40-final-report.md")
def test_closing_rehash_fails_closed_after_midrun_mutation(
    tmp_path: Path,
    namespace: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _sealed_fixture(tmp_path, namespace, monkeypatch)
    globals_ = namespace["confirmation"].__globals__
    harmless_range = range(81, 82)
    monkeypatch.setitem(globals_, "CONFIRMATION_SEEDS", harmless_range)
    production = root.joinpath(*namespace["PRODUCTION_RELATIVE"].parts)

    def mutate_evaluate(
        seeds: range,
        mode: str,
        *,
        confirmation_access: Any = None,
    ) -> dict[str, Any]:
        del seeds, mode
        assert confirmation_access is not None
        production.write_text("changed during fake evaluation\n", encoding="utf-8")
        return {"per_seed": []}

    monkeypatch.setitem(globals_, "evaluate", mutate_evaluate)
    with pytest.raises(ValueError, match="manifest mismatch"):
        namespace["confirmation"](root, manifest_path)
    assert root.joinpath(*namespace["RECEIPT_RELATIVE"].parts).is_file()
    assert not root.joinpath(*namespace["RESULT_RELATIVE"].parts).exists()

