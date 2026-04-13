from __future__ import annotations

import pytest
import torch

import clarus
from clarus.engine import CEEngine
from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode
from tests.test_sleep import make_runtime_artifact


def make_weight(dim: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    w = torch.randn(dim, dim)
    w = 0.5 * (w + w.t())
    w.fill_diagonal_(0.0)
    return w.float()


def test_clarus_import_surface_exposes_runtime_without_missing_rust_symbols():
    assert hasattr(clarus, "BrainRuntime")
    assert hasattr(clarus, "BrainRuntimeConfig")
    assert clarus.topk_sparse is None or callable(clarus.topk_sparse)


def test_ce_engine_build_brain_runtime_keeps_python_as_control_plane(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    eng = CEEngine(str(path), device="cpu", backend="torch")
    runtime = eng.build_brain_runtime()
    assert runtime.config.dim == eng.d
    assert runtime.device.type == "cpu"
    assert runtime.backend in {"torch", "rust", "auto"}


def test_ce_engine_rejects_clone_bearing_runtime_artifact(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    artifact["clone_state"] = {"bad": torch.ones(1)}
    bad_path = tmp_path / "runtime_clone.pt"
    torch.save(artifact, bad_path)
    with pytest.raises(RuntimeError, match="teacher-bearing artifact"):
        CEEngine(str(bad_path), device="cpu", backend="torch")


def test_ce_engine_rejects_pretrained_fallback_in_runtime_artifact(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    artifact["allow_pretrained_fallback"] = True
    bad_path = tmp_path / "runtime_pretrained.pt"
    torch.save(artifact, bad_path)
    with pytest.raises(RuntimeError, match="pretrained fallback"):
        CEEngine(str(bad_path), device="cpu", backend="torch")


def test_ce_engine_requires_tokenizer_json_in_runtime_artifact(tmp_path):
    path = make_runtime_artifact(tmp_path, decoder_query_blend=1.0)
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    artifact.pop("tokenizer_json", None)
    bad_path = tmp_path / "runtime_no_tokenizer.pt"
    torch.save(artifact, bad_path)
    with pytest.raises(RuntimeError, match="tokenizer_json"):
        CEEngine(str(bad_path), device="cpu", backend="torch")


def test_brain_runtime_respects_sparse_energy_budget_and_uses_hippocampus():
    w = make_weight()
    runtime = BrainRuntime(
        w,
        config=BrainRuntimeConfig(dim=64, active_ratio=0.125, memory_capacity=8),
        backend="torch",
        device="cpu",
    )
    runtime.set_goal(torch.ones(64))
    step = runtime.step(external_input=torch.linspace(0.0, 1.0, 64))
    assert step.active_modules <= runtime.config.energy_budget(RuntimeMode.WAKE)
    assert len(runtime.hippocampus) >= 1


def test_brain_runtime_mode_transitions_cover_wake_nrem_rem():
    w = make_weight(seed=1)
    runtime = BrainRuntime(
        w,
        config=BrainRuntimeConfig(dim=64, active_ratio=0.125),
        backend="torch",
        device="cpu",
    )
    runtime.sleep_pressure = 1.2
    step_nrem = runtime.step(external_input=torch.zeros(64))
    assert step_nrem.mode is RuntimeMode.NREM

    runtime.sleep_pressure = 0.3
    step_rem = runtime.step(external_input=torch.zeros(64))
    assert step_rem.mode is RuntimeMode.REM

    step_wake = runtime.step(external_input=torch.ones(64))
    assert step_wake.mode is RuntimeMode.WAKE


def test_brain_runtime_snapshot_restore_is_continuous():
    w = make_weight(seed=2)
    runtime = BrainRuntime(
        w,
        config=BrainRuntimeConfig(dim=64, active_ratio=0.125, memory_capacity=8),
        backend="torch",
        device="cpu",
    )
    runtime.set_goal(torch.randn(64))
    runtime.step(external_input=torch.randn(64))
    runtime.step(external_input=torch.randn(64))
    snapshot = runtime.snapshot()

    rt_a = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    rt_b = BrainRuntime.from_snapshot(snapshot, backend="torch", device="cpu")
    shared_input = torch.randn(64)

    out_a = rt_a.step(external_input=shared_input)
    out_b = rt_b.step(external_input=shared_input)

    assert out_a.mode is out_b.mode
    assert out_a.active_modules == out_b.active_modules
    assert out_a.energy == pytest.approx(out_b.energy, rel=1e-6, abs=1e-6)
    assert torch.allclose(rt_a.activation, rt_b.activation, atol=1e-6, rtol=1e-6)
    assert torch.equal(rt_a.lifecycle, rt_b.lifecycle)
