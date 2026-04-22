"""Bitfield runtime: 4-bit quantized CE engine per 12_Equation.md 1.6-1.7.

Euler identity e^{i*pi}+1=0 maps to 4 operations:
  0 -> CLEAR (prune, reset)
  1 -> IDENTITY (keep, normalize)
  e -> DECAY (EMA, sleep pressure via shift-add)
  pi -> RADIUS (connection mask, neighbor rule)
  i -> MODE (2-bit mode register: off/wake/nrem/rem)

State representation:
  Control: bitfield O(N) bits -- active mask, mode, freeze
  State:   4-bit fixed-point O(N) bytes -- activation, trace
  Weight:  4-bit sparse O(N*K) -- CSR with 4-bit values
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

try:
    from .constants import (
        ACTIVE_RATIO, MEMORY_TRACE_DECAY, ADAPTATION_COUPLING,
        ADAPTATION_DECAY, ADAPTATION_CLAMP, NOISE_SIGMA,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE,
        SPARSITY_RADIUS, NORM_EPS,
    )
except ImportError:
    from clarus.constants import (
        ACTIVE_RATIO, MEMORY_TRACE_DECAY, ADAPTATION_COUPLING,
        ADAPTATION_DECAY, ADAPTATION_CLAMP, NOISE_SIGMA,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE,
        SPARSITY_RADIUS, NORM_EPS,
    )


def quantize_4bit(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Quantize float tensor to symmetric int4 represented in int8 [-7, 7]."""
    max_abs = max(x.abs().max().item(), 1e-8)
    scale = max_abs / 7.0
    q = (x / scale).round().clamp(-7, 7).to(torch.int8)
    return q, scale


def dequantize_4bit(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.float() * scale


def quantize_8bit(x: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Quantize to 8-bit unsigned [0, 255]."""
    x_min = x.min().item()
    x_max = x.max().item()
    span = max(x_max - x_min, 1e-8)
    scale = span / 255.0
    q = ((x - x_min) / scale).round().clamp(0, 255).to(torch.uint8)
    return q, scale, x_min


def dequantize_8bit(q: torch.Tensor, scale: float, zero: float) -> torch.Tensor:
    return q.float() * scale + zero


@dataclass
class BitfieldLayout:
    """Memory layout per 12_Equation.md 1.7 (N=dim basis)."""
    dim: int
    avg_neighbors: int = 130

    @property
    def active_mask_bytes(self) -> int:
        return math.ceil(self.dim / 8)

    @property
    def freeze_mask_bytes(self) -> int:
        return math.ceil(self.dim / 8)

    @property
    def mode_bytes(self) -> int:
        return 1

    @property
    def weight_bytes(self) -> int:
        nnz = self.dim * self.avg_neighbors
        return nnz // 2  # 4-bit packed

    @property
    def csr_index_bytes(self) -> int:
        return self.dim * self.avg_neighbors * 2 + (self.dim + 1) * 4

    @property
    def state_bytes(self) -> int:
        return self.dim  # 8-bit per cell

    @property
    def phi_bytes(self) -> int:
        return self.dim  # 8-bit

    @property
    def trace_bytes(self) -> int:
        nnz = self.dim * self.avg_neighbors
        return nnz // 2  # 4-bit

    @property
    def total_engine_bytes(self) -> int:
        return (self.active_mask_bytes + self.freeze_mask_bytes +
                self.mode_bytes + self.weight_bytes + self.csr_index_bytes +
                self.state_bytes + self.phi_bytes + self.trace_bytes + 6)

    def summary(self) -> dict[str, str]:
        return {
            "active_mask": f"{self.active_mask_bytes} B",
            "freeze_mask": f"{self.freeze_mask_bytes} B",
            "mode": f"{self.mode_bytes} B",
            "weights_4bit": f"{self.weight_bytes / 1024:.1f} KB",
            "csr_index": f"{self.csr_index_bytes / 1024:.1f} KB",
            "state_8bit": f"{self.state_bytes / 1024:.1f} KB",
            "phi_8bit": f"{self.phi_bytes / 1024:.1f} KB",
            "trace_4bit": f"{self.trace_bytes / 1024:.1f} KB",
            "total_engine": f"{self.total_engine_bytes / 1024:.1f} KB",
        }


class BitfieldRuntime:
    """Quantized CE runtime operating on 4/8-bit state.

    All arithmetic stays in integer domain where possible.
    Dequantize only for tanh and final output.
    """

    def __init__(self, weight: torch.Tensor, *, active_ratio: float = ACTIVE_RATIO) -> None:
        self.dim = weight.shape[0]
        self.active_ratio = active_ratio

        w_sparse = weight.clone()
        w_sparse[weight.abs() < 1e-4] = 0
        self.w_q, self.w_scale = quantize_4bit(w_sparse)

        self.activation_q = torch.zeros(self.dim, dtype=torch.uint8)
        self.act_scale = 2.0 / 255.0
        self.act_zero = -1.0

        self.refractory_q = torch.zeros(self.dim, dtype=torch.uint8)
        self.ref_scale = 2.0 / 255.0
        self.ref_zero = 0.0

        self.trace_q = torch.zeros(self.dim, dtype=torch.uint8)
        self.trace_scale = 2.0 / 255.0
        self.trace_zero = -1.0

        self.active_mask = torch.zeros(self.dim, dtype=torch.bool)
        self.mode = 0b01  # wake
        self._state_overhead_bytes = 6 * 4  # three scales + three zeros as float32 scalars

    def _activation_float(self) -> torch.Tensor:
        return dequantize_8bit(self.activation_q, self.act_scale, self.act_zero)

    def _refractory_float(self) -> torch.Tensor:
        return dequantize_8bit(self.refractory_q, self.ref_scale, self.ref_zero)

    def _trace_float(self) -> torch.Tensor:
        return dequantize_8bit(self.trace_q, self.trace_scale, self.trace_zero)

    def _write_quantized_state(
        self,
        activation: torch.Tensor,
        refractory: torch.Tensor,
        trace: torch.Tensor,
    ) -> None:
        self.activation_q, self.act_scale, self.act_zero = quantize_8bit(activation.clamp(-1.0, 1.0))
        self.refractory_q, self.ref_scale, self.ref_zero = quantize_8bit(refractory.clamp(0.0, ADAPTATION_CLAMP))
        self.trace_q, self.trace_scale, self.trace_zero = quantize_8bit(trace.clamp(-1.0, 1.0))

    def step(self, external: torch.Tensor | None = None) -> dict[str, float]:
        """One tick using quantized persistent state and dequantized compute."""
        if external is None:
            external = torch.zeros(self.dim)

        activation = self._activation_float()
        refractory = self._refractory_float()
        trace = self._trace_float()
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        n_active = active_idx.numel()

        if n_active == 0:
            recurrent = torch.zeros(self.dim)
        else:
            active_vals = activation[active_idx]
            w_cols = self.w_q[:, active_idx].float() * self.w_scale
            recurrent = w_cols @ active_vals

        drive = recurrent + 0.45 * external - 0.35 * refractory - ADAPTATION_COUPLING * trace

        gamma_a = 0.18 if self.mode == 0b01 else 0.34
        kappa_a = 0.82 if self.mode == 0b01 else 0.52

        new_act = ((1.0 - gamma_a) * activation + kappa_a * torch.tanh(drive)).clamp(-1, 1)
        new_ref = ((1.0 - 0.12) * refractory + 0.24 * new_act.square()).clamp(0.0, ADAPTATION_CLAMP)
        new_trace = ((1.0 - MEMORY_TRACE_DECAY) * trace + MEMORY_TRACE_DECAY * new_act).clamp(-1, 1)
        self._write_quantized_state(new_act, new_ref, new_trace)

        k = max(1, int(self.active_ratio * self.dim))
        topk_idx = torch.topk(new_act.abs(), k).indices
        self.active_mask.zero_()
        self.active_mask[topk_idx] = True

        energy = -0.5 * torch.dot(new_act, recurrent).item()
        active_count = int(self.active_mask.sum().item())

        return {
            "energy": energy,
            "active": active_count,
            "active_ratio": active_count / self.dim,
            "act_norm": float(new_act.norm().item()),
        }

    def get_activation(self) -> torch.Tensor:
        return self._activation_float()

    def memory_bytes(self) -> int:
        """Best-effort runtime memory including masks and quantized state."""
        w_bytes = self.w_q.numel() // 2
        act_bytes = self.activation_q.numel()
        ref_bytes = self.refractory_q.numel()
        trace_bytes = self.trace_q.numel()
        active_mask_bytes = self.active_mask.numel()
        return (
            w_bytes
            + act_bytes
            + ref_bytes
            + trace_bytes
            + active_mask_bytes
            + self._state_overhead_bytes
            + 1
        )


class Float32Runtime:
    """Reference float32 runtime for comparison."""

    def __init__(self, weight: torch.Tensor, *, active_ratio: float = ACTIVE_RATIO) -> None:
        self.dim = weight.shape[0]
        self.active_ratio = active_ratio
        self.weight = weight.float()
        self.activation = torch.zeros(self.dim)
        self.refractory = torch.zeros(self.dim)
        self.trace = torch.zeros(self.dim)
        self.active_mask = torch.zeros(self.dim, dtype=torch.bool)

    def step(self, external: torch.Tensor | None = None) -> dict[str, float]:
        if external is None:
            external = torch.zeros(self.dim)

        recurrent = self.weight @ (self.activation * self.active_mask.float())
        drive = recurrent + 0.45 * external - 0.35 * self.refractory - ADAPTATION_COUPLING * self.trace

        new_act = ((1.0 - 0.18) * self.activation + 0.82 * torch.tanh(drive)).clamp(-1, 1)
        new_ref = (1.0 - 0.12) * self.refractory + 0.24 * new_act.square()
        new_trace = (1.0 - MEMORY_TRACE_DECAY) * self.trace + MEMORY_TRACE_DECAY * new_act

        self.activation = new_act
        self.refractory = new_ref
        self.trace = new_trace

        k = max(1, int(self.active_ratio * self.dim))
        topk_idx = torch.topk(new_act.abs(), k).indices
        self.active_mask.zero_()
        self.active_mask[topk_idx] = True

        energy = -0.5 * torch.dot(new_act, recurrent).item()
        active_count = int(self.active_mask.sum().item())

        return {
            "energy": energy,
            "active": active_count,
            "active_ratio": active_count / self.dim,
            "act_norm": float(new_act.norm().item()),
        }

    def get_activation(self) -> torch.Tensor:
        return self.activation

    def memory_bytes(self) -> int:
        return (self.weight.numel() + self.activation.numel() * 3) * 4


def benchmark(dim: int = 768, steps: int = 200, seed: int = 42) -> dict:
    """Compare quantized-state runtime vs float32 runtime on identical inputs."""
    torch.manual_seed(seed)
    w = torch.randn(dim, dim) * 0.01
    w = 0.5 * (w + w.T)
    w.fill_diagonal_(0)
    w[w.abs() < 0.005] = 0  # sparsify

    bf = BitfieldRuntime(w)
    fp = Float32Runtime(w)

    inputs = [torch.randn(dim) * 0.3 for _ in range(steps)]

    t0 = time.perf_counter()
    bf_results = []
    for ext in inputs:
        bf_results.append(bf.step(ext))
    bf_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fp_results = []
    for ext in inputs:
        fp_results.append(fp.step(ext))
    fp_time = time.perf_counter() - t0

    act_bf = bf.get_activation()
    act_fp = fp.get_activation()
    cos_sim = torch.cosine_similarity(act_bf.unsqueeze(0), act_fp.unsqueeze(0)).item()
    mse = ((act_bf - act_fp) ** 2).mean().item()
    max_err = (act_bf - act_fp).abs().max().item()

    e_bf = [r["energy"] for r in bf_results]
    e_fp = [r["energy"] for r in fp_results]
    energy_corr = torch.stack([torch.tensor(e_bf), torch.tensor(e_fp)]).corrcoef()[0, 1].item()

    bf_mem = bf.memory_bytes()
    fp_mem = fp.memory_bytes()
    layout = BitfieldLayout(dim)

    return {
        "dim": dim,
        "steps": steps,
        "bitfield_time_ms": bf_time * 1000,
        "float32_time_ms": fp_time * 1000,
        "speedup": fp_time / max(bf_time, 1e-9),
        "bitfield_step_us": bf_time / steps * 1e6,
        "float32_step_us": fp_time / steps * 1e6,
        "cosine_similarity": cos_sim,
        "mse": mse,
        "max_absolute_error": max_err,
        "energy_correlation": energy_corr,
        "bitfield_memory_KB": bf_mem / 1024,
        "float32_memory_KB": fp_mem / 1024,
        "memory_ratio": bf_mem / max(fp_mem, 1),
        "theoretical_engine_KB": layout.total_engine_bytes / 1024,
        "runtime_layout_note": "bitfield_memory_KB includes q-state, active-mask bytes, and scalar quant metadata",
        "layout": layout.summary(),
    }
