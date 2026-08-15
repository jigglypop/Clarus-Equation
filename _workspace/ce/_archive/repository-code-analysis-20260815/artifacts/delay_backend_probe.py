"""Reproduce the BrainRuntime Torch/Rust axon-delay dispatch mismatch.

This is an observational probe for the repository-code-analysis run, not a
product regression test.  It requires the optional ``clarus._rust`` extension.
"""

from __future__ import annotations

import warnings

import torch

from reality_stone.clarus.runtime import (
    BrainRuntime,
    BrainRuntimeConfig,
    RuntimeMode,
    _HAS_RUST_KERNEL,
)


def main() -> int:
    print(f"has_rust={_HAS_RUST_KERNEL}")
    if not _HAS_RUST_KERNEL:
        print("status=SKIP (reality_stone.clarus._rust is unavailable)")
        return 0

    # Keep the probe output focused on the backend-state comparison.
    warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
    warnings.filterwarnings("ignore", message="Sparse CSR tensor support.*")

    dim = 8
    torch.manual_seed(7)
    weight = torch.randn(dim, dim) * 0.03
    config = BrainRuntimeConfig(
        dim=dim,
        active_ratio=0.5,
        noise_sigma=0.0,
        dale_law=False,
        axon_delay=True,
        max_axon_delay=2,
        memory_capacity=2,
    )

    torch_runtime = BrainRuntime(
        weight, config=config, backend="torch", device="cpu"
    )
    rust_runtime = BrainRuntime(
        weight, config=config, backend="rust", device="cpu"
    )

    activation = torch.linspace(-0.4, 0.5, dim)
    external = torch.linspace(0.1, 0.6, dim)
    for runtime in (torch_runtime, rust_runtime):
        runtime.activation = activation.clone()
        runtime.lifecycle.fill_(0)

    torch_runtime.step(external_input=external, force_mode=RuntimeMode.WAKE)
    rust_runtime.step(external_input=external, force_mode=RuntimeMode.WAKE)

    max_diff = float(
        (torch_runtime.activation - rust_runtime.activation).abs().max().item()
    )
    torch_delay_sum = float(torch_runtime._delay_buffer.abs().sum().item())
    rust_delay_sum = float(rust_runtime._delay_buffer.abs().sum().item())
    reproduced = (
        max_diff > 0.0
        and torch_runtime._delay_idx == 1
        and rust_runtime._delay_idx == 0
        and torch_delay_sum > 0.0
        and rust_delay_sum == 0.0
    )

    print(f"max_activation_diff={max_diff:.15f}")
    print(f"torch_delay_idx={torch_runtime._delay_idx}")
    print(f"rust_delay_idx={rust_runtime._delay_idx}")
    print(f"torch_delay_sum={torch_delay_sum:.15f}")
    print(f"rust_delay_sum={rust_delay_sum:.15f}")
    print(f"mismatch_reproduced={reproduced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
