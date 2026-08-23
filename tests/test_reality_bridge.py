from __future__ import annotations

import numpy as np
import torch

import reality_stone.clarus as clarus
from reality_stone.clarus import reality


def test_reality_bridge_is_exposed_on_module_import_surface():
    assert hasattr(clarus, "BrainRuntime")
    assert reality.RealityStoneStatus is not None
    assert callable(reality.has_reality_stone)
    assert callable(reality.status)


def test_reality_bridge_status_finds_local_reality_stone_source():
    status = reality.status()

    assert status.available
    assert status.version == "0.2.10"
    assert status.error is None


def test_metric_attention_bridge_runs_torch_fallback_path():
    attn = reality.metric_attention(hidden_size=4, rank=2)
    q = torch.randn(1, 1, 3, 4)
    k = torch.randn(1, 1, 3, 4)
    v = torch.randn(1, 1, 3, 4)

    out = attn(q, k, v, causal=True)

    assert out.shape == v.shape
    assert torch.isfinite(out).all()


def test_unified_riemannian_layer_bridge_runs_numpy_fallback_path():
    layer = reality.unified_riemannian_layer(
        metric_type="euclidean",
        curvature=0.0,
        input_dim=2,
        enable_bellman=True,
    )
    x = np.array([[0.0, 1.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0]], dtype=np.float32)

    out, energy = layer.forward(x, target)

    assert out.shape == x.shape
    assert np.all(np.isfinite(out))
    assert energy is not None
    assert "bellman_residual" in energy
