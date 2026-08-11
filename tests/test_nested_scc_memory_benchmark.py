import json

import pytest

from reality_stone.clarus.nested_scc_memory_benchmark import (
    ARMS,
    MemoryBenchmarkConfig,
    canonical_json,
    generate_seed_episodes,
    predict_arm,
    preregistration_payload,
    run_locked_phase,
    verify_preregistration,
)


def test_episode_generation_is_balanced_deterministic_and_current_input_blind() -> None:
    config = MemoryBenchmarkConfig(episode_count=8)
    first = generate_seed_episodes(17, config)
    second = generate_seed_episodes(17, config)
    assert first == second
    assert sorted(episode.target_action for episode in first) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert all(len(episode.observations) == config.decision_tick + 1 for episode in first)


def test_every_registered_arm_returns_a_finite_registered_action() -> None:
    config = MemoryBenchmarkConfig(episode_count=4, maximum_depth=3)
    episode = generate_seed_episodes(3, config)[0]
    predictions = {}
    for arm in ARMS:
        prediction, mediation_violations, nonfinite = predict_arm(arm, episode.observations, config)
        predictions[arm] = prediction
        assert prediction in range(4)
        assert nonfinite == 0
        if arm == "v9":
            assert mediation_violations == 0
    assert set(predictions) == set(ARMS)


def test_preregistration_is_hash_bound_and_rejects_mutation(tmp_path) -> None:
    root = tmp_path / "repo"
    sources = (
        "reality_stone/python/reality_stone/clarus/nested_scc_memory_benchmark.py",
        "reality_stone/python/reality_stone/clarus/nested_scc_tower.py",
        "reality_stone/python/reality_stone/clarus/adaptive_scc_tower_controller.py",
    )
    repository_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for source in sources:
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository_root / source).read_bytes())
    payload = preregistration_payload(repository_root=root, config=MemoryBenchmarkConfig())
    assert verify_preregistration(payload, repository_root=root) == MemoryBenchmarkConfig()
    mutated = json.loads(canonical_json(payload))
    mutated["primary_gates"]["mean_improvement"] = ">= -1"
    with pytest.raises(ValueError, match="content"):
        verify_preregistration(mutated, repository_root=root)


def test_locked_runner_refuses_missing_audit_existing_result_and_early_confirmation(
    tmp_path,
) -> None:
    repository_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    prereg = preregistration_payload(
        repository_root=repository_root, config=MemoryBenchmarkConfig()
    )
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(canonical_json(prereg), encoding="utf-8")
    audit = tmp_path / "audit.md"
    result = tmp_path / "result.json"
    with pytest.raises(PermissionError, match="audit"):
        run_locked_phase(
            repository_root=repository_root,
            preregistration_path=prereg_path,
            result_path=result,
            phase="development",
            authorization_path=audit,
        )
    audit.write_text("Gate: PASS\n", encoding="utf-8")
    result.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="reruns"):
        run_locked_phase(
            repository_root=repository_root,
            preregistration_path=prereg_path,
            result_path=result,
            phase="development",
            authorization_path=audit,
        )
    result.unlink()
    with pytest.raises(PermissionError, match="development result"):
        run_locked_phase(
            repository_root=repository_root,
            preregistration_path=prereg_path,
            result_path=result,
            phase="confirmation",
            authorization_path=audit,
        )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"episode_count": 5},
        {"shell_width": 3},
        {"slow_gap": True},
        {"noise_sigma": "0.05"},
        {"bootstrap_samples": 10},
    ),
)
def test_config_fails_closed(kwargs) -> None:
    with pytest.raises(ValueError):
        MemoryBenchmarkConfig(**kwargs)
