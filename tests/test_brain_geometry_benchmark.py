import math

from reality_stone.clarus.brain_geometry_benchmark import (
    BrainGeometryBenchConfig,
    ResidualReplayBenchConfig,
    StnBoundaryBenchConfig,
    evaluate_brain_geometry,
    evaluate_residual_replay,
    evaluate_stn_boundary,
    pure_diffusion_mode,
)


def test_pure_diffusion_mode_matches_exact_decay() -> None:
    config = BrainGeometryBenchConfig()
    observed = pure_diffusion_mode(25, config)
    expected = math.exp(-2.0 * config.heat_diffusion * config.step * 25)
    assert abs(observed - expected) <= 1e-12
    assert observed < 1.0


def test_small_brain_geometry_benchmark_has_integrity_guards() -> None:
    result = evaluate_brain_geometry(BrainGeometryBenchConfig(trials=48, seeds=3))
    assert result["schema"] == "clarus.brain-geometry.validation.v1"
    assert result["future_reads"] == 0
    assert result["environment_clone_calls"] == 0
    assert result["pure_diffusion_mode"]["absolute_error"] <= 1e-10
    for domain in ("id", "ood"):
        for arm in (
            "pure_diffusion",
            "fixed_attractor",
            "md_attractor",
            "md_context_shuffle",
            "oracle_context_md",
        ):
            assert result[domain][arm]["bounded"]
            assert result[domain][arm]["nonfinite_count"] == 0


def test_small_residual_replay_benchmark_has_causal_integrity() -> None:
    base = BrainGeometryBenchConfig(trials=48, seeds=3, blocks=(12, 16))
    result = evaluate_residual_replay(ResidualReplayBenchConfig(base=base))
    assert result["schema"] == "clarus.residual-replay.validation.v1"
    assert result["future_reads"] == 0
    assert result["environment_clone_calls"] == 0
    for domain in ("id", "ood", "stationary"):
        for arm in (
            "md_checkpoint",
            "residual_replay",
            "residual_sign_flip",
            "oracle_context_md",
        ):
            assert result[domain][arm]["bounded"]
            assert result[domain][arm]["nonfinite_count"] == 0


def test_small_stn_benchmark_preserves_memory_trace() -> None:
    base = BrainGeometryBenchConfig(trials=48, seeds=3, blocks=(12, 16))
    residual = ResidualReplayBenchConfig(base=base)
    result = evaluate_stn_boundary(StnBoundaryBenchConfig(residual=residual))
    assert result["schema"] == "clarus.stn-boundary.validation.v1"
    assert result["future_reads"] == 0
    assert result["environment_clone_calls"] == 0
    assert result["id"]["memory_trace_identical"]
    assert result["ood"]["memory_trace_identical"]
