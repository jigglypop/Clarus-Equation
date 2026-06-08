# LLM Context Chunk

---
## File: `reality_stone/python/reality_stone/clarus/runtime.py`

```python
from __future__ import annotations
"""Reference brain runtime for the Python control plane.

This module intentionally keeps policy in Python while delegating reusable
numeric kernels to `reality_stone.clarus.ce_ops` / `reality_stone.clarus._rust`.

Concept layout from the refactor plan:
- `BrainRuntimeConfig`: global mode, lifecycle, and energy-budget policy
- `HippocampusMemory`: minimal fast-memory / replay subsystem
- `BrainRuntime`: sparse lifecycle + mode switching + snapshot continuity
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict

import numpy as np
import torch

try:
    from .ce_ops import pack_sparse
    from .constants import (
        MEMORY_TRACE_DECAY, ADAPTATION_DECAY, ADAPTATION_COUPLING,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE, ADAPTATION_CLAMP,
        TAU_W_STEPS, TAU_S_STEPS, SLEEP_PRESSURE_MAX, REM_TAU_FACTOR,
        NORM_EPS, NOISE_SIGMA, DALE_EI_RATIO, DALE_INH_GAIN,
        AXON_DELAY_MAX, CIRCADIAN_PERIOD, CIRCADIAN_AMP, CIRCADIAN_BASE,
        NREM_LENGTH_DECAY, FORGET_TAU, RECALL_SIMILARITY_THRESHOLD,
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        BOOTSTRAP_CONTRACTION, BAND_DELTA, BAND_THETA, BAND_ALPHA,
        BAND_BETA, BAND_GAMMA,
    )
    from .stdp import (
        STDPConfig, EligibilityTracker, compute_learning_gate, apply_stdp_update,
    )
except ImportError:
    from reality_stone.clarus.ce_ops import pack_sparse
    from reality_stone.clarus.constants import (
        MEMORY_TRACE_DECAY, ADAPTATION_DECAY, ADAPTATION_COUPLING,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE, ADAPTATION_CLAMP,
        TAU_W_STEPS, TAU_S_STEPS, SLEEP_PRESSURE_MAX, REM_TAU_FACTOR,
        NORM_EPS, NOISE_SIGMA, DALE_EI_RATIO, DALE_INH_GAIN,
        AXON_DELAY_MAX, CIRCADIAN_PERIOD, CIRCADIAN_AMP, CIRCADIAN_BASE,
        NREM_LENGTH_DECAY, FORGET_TAU, RECALL_SIMILARITY_THRESHOLD,
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        BOOTSTRAP_CONTRACTION, BAND_DELTA, BAND_THETA, BAND_ALPHA,
        BAND_BETA, BAND_GAMMA,
    )
    from reality_stone.clarus.stdp import (
        STDPConfig, EligibilityTracker, compute_learning_gate, apply_stdp_update,
    )

try:
    from ._rust import nn_brain_step as _rust_brain_step
    _HAS_RUST_KERNEL = True
except ImportError:
    _HAS_RUST_KERNEL = False

_MODE_TO_INT = {
    "WAKE": 0,
    "NREM": 1,
    "REM": 2,
}


class RuntimeMode(str, Enum):
    WAKE = "WAKE"
    NREM = "NREM"
    REM = "REM"


class ModuleLifecycle(str, Enum):
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    DORMANT = "DORMANT"
    SLEEPING = "SLEEPING"


_LIFECYCLE_TO_CODE = {
    ModuleLifecycle.ACTIVE: 0,
    ModuleLifecycle.IDLE: 1,
    ModuleLifecycle.DORMANT: 2,
    ModuleLifecycle.SLEEPING: 3,
}
_CODE_TO_LIFECYCLE = {value: key for key, value in _LIFECYCLE_TO_CODE.items()}


try:
    from .utils import normalize_vector as _normalize
except ImportError:
    from reality_stone.clarus.utils import normalize_vector as _normalize


@dataclass
class BrainRuntimeConfig:
    """Global runtime knobs for sparse activation, modes, and replay."""
    dim: int
    active_ratio: float = 0.125
    idle_threshold: float = 0.08
    active_threshold: float = 0.22
    bit_lower_threshold: float = 0.10
    bit_upper_threshold: float = 0.30
    refractory_scale: float = 0.35
    replay_gain: float = 0.28
    goal_gain: float = 0.20
    external_gain: float = 0.45
    zero_tol: float = 0.0
    dormant_after: int = 3
    sleeping_after: int = 6
    wake_threshold: float = 0.18
    memory_capacity: int = 32
    memory_topk: int = 4
    noise_sigma: float = NOISE_SIGMA
    dale_law: bool = True
    axon_delay: bool = True
    max_axon_delay: int = AXON_DELAY_MAX
    forget_tau: float = FORGET_TAU
    # F1 self-organization (docs/7_AGI/12_Equation.md A.2 condition #2).
    # When enabled, the runtime feeds the empirical active ratio
    #   p_emp = |A_t| / dim
    # back into the next budget so it contracts toward ACTIVE_RATIO (epsilon^2).
    f1_self_measure: bool = False
    f1_pull_strength: float = 0.5
    f1_ema_alpha: float = 0.1
    f1_min_ratio: float = 0.005
    f1_max_ratio: float = 0.5
    # F14 local learning. Kept off by default so inference/runtime tests remain
    # deterministic unless the caller explicitly opts into plastic weights.
    stdp_enabled: bool = False
    stdp_interval: int = 1
    stdp_apply_interval: int = 10
    stdp_lr: float = 0.001
    stdp_density: float = ACTIVE_RATIO
    stdp_gate_threshold: float = 1e-3
    stdp_spike_threshold: float = 0.3

    def __post_init__(self) -> None:
        self.dim = int(self.dim)
        if self.dim <= 0:
            raise ValueError("runtime dimension must be positive")
        self.active_ratio = min(max(float(self.active_ratio), 0.0), 1.0)
        self.memory_topk = max(1, int(self.memory_topk))
        self.memory_capacity = max(1, int(self.memory_capacity))
        self.f1_pull_strength = min(max(float(self.f1_pull_strength), 0.0), 1.0)
        self.f1_ema_alpha = min(max(float(self.f1_ema_alpha), 0.0), 1.0)
        self.f1_min_ratio = min(max(float(self.f1_min_ratio), 0.0), 1.0)
        self.f1_max_ratio = min(max(float(self.f1_max_ratio), self.f1_min_ratio), 1.0)
        self.stdp_interval = max(1, int(self.stdp_interval))
        self.stdp_apply_interval = max(self.stdp_interval, int(self.stdp_apply_interval))
        self.stdp_lr = max(float(self.stdp_lr), 0.0)
        self.stdp_density = min(max(float(self.stdp_density), 0.0), 1.0)
        self.stdp_gate_threshold = max(float(self.stdp_gate_threshold), 0.0)
        self.stdp_spike_threshold = max(float(self.stdp_spike_threshold), 0.0)

    def energy_budget(self, mode: RuntimeMode) -> int:
        base = max(1, int(round(self.dim * self.active_ratio)))
        if mode is RuntimeMode.NREM:
            return max(1, int(round(base * 0.5)))
        if mode is RuntimeMode.REM:
            return max(1, int(round(base * 0.75)))
        return base

    def activation_decay(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.18,
            RuntimeMode.NREM: 0.34,
            RuntimeMode.REM: 0.22,
        }[mode]

    def activation_gain(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.82,
            RuntimeMode.NREM: 0.52,
            RuntimeMode.REM: 0.68,
        }[mode]

    def refractory_decay(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.12,
            RuntimeMode.NREM: 0.26,
            RuntimeMode.REM: 0.18,
        }[mode]

    def refractory_gain(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.24,
            RuntimeMode.NREM: 0.12,
            RuntimeMode.REM: 0.18,
        }[mode]

    def replay_mix(self, mode: RuntimeMode) -> float:
        return {
            RuntimeMode.WAKE: 0.08,
            RuntimeMode.NREM: self.replay_gain,
            RuntimeMode.REM: self.replay_gain * 1.25,
        }[mode]


@dataclass
class RuntimeStep:
    """High-level runtime summary returned to the Python control plane."""
    step: int
    mode: RuntimeMode
    energy: float
    active_modules: int
    replay_norm: float
    sleep_pressure: float
    arousal: float
    lifecycle_counts: Dict[str, int]
    stdp_gate: float = 0.0
    stdp_updates: int = 0


@dataclass
class BrainRuntimeSnapshot:
    """Serializable runtime state used for warm snapshots / restore."""
    config: BrainRuntimeConfig
    weight: torch.Tensor
    activation: torch.Tensor
    refractory: torch.Tensor
    memory_trace: torch.Tensor
    adaptation: torch.Tensor
    stp_u: torch.Tensor
    stp_x: torch.Tensor
    bitfield: torch.Tensor
    goal: torch.Tensor
    lifecycle: torch.Tensor
    inactive_steps: torch.Tensor
    mode: RuntimeMode
    sleep_pressure: float
    arousal: float
    step: int
    hippocampus: dict[str, object]
    mode_occupancy: Dict[str, int] = field(default_factory=dict)
    active_ratio_ema: float = -1.0
    stdp_tracker: dict[str, torch.Tensor] | None = None
    stdp_prev_critic_score: float = 0.0
    stdp_updates: int = 0


@dataclass
class HippocampusMemory:
    """Minimal fast-memory subsystem: encode, recall, replay priority."""
    dim: int
    capacity: int = 32
    device: str | torch.device = "cpu"
    _keys: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _values: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _priority: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dim = int(self.dim)
        self.capacity = max(1, int(self.capacity))
        self.device = torch.device(self.device)

    def __len__(self) -> int:
        return len(self._priority)

    def encode(
        self,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        priority: float = 1.0,
    ) -> None:
        key = _normalize(key).to(self.device)
        value = key if value is None else value.detach().float().to(self.device)
        priority = float(max(priority, 1e-6))
        if len(self._priority) >= self.capacity:
            drop_idx = min(range(len(self._priority)), key=self._priority.__getitem__)
            self._keys.pop(drop_idx)
            self._values.pop(drop_idx)
            self._priority.pop(drop_idx)
        self._keys.append(key)
        self._values.append(value)
        self._priority.append(priority)

    def decay_priorities(self, steps: int = 1) -> None:
        """Exponential priority decay: P *= exp(-dt/tau_forget). (15_Equations D)"""
        if not self._priority:
            return
        factor = math.exp(-steps / FORGET_TAU)
        self._priority = [p * factor for p in self._priority]

    def recall(self, cue: torch.Tensor, *, topk: int = 4) -> torch.Tensor:
        if not self._keys:
            return torch.zeros(self.dim, device=self.device)
        cue = _normalize(cue).to(self.device)
        keys = torch.stack(self._keys, dim=0)
        values = torch.stack(self._values, dim=0)
        priority = torch.tensor(self._priority, dtype=torch.float32, device=self.device)
        similarity = keys @ cue
        above_threshold = similarity >= RECALL_SIMILARITY_THRESHOLD
        if not above_threshold.any():
            return torch.zeros(self.dim, device=self.device)
        score = similarity + priority.log()
        score = score.masked_fill(~above_threshold, float("-inf"))
        k = min(max(int(topk), 1), int(above_threshold.sum().item()))
        top_score, top_idx = torch.topk(score, k=k)
        weights = torch.softmax(top_score, dim=0)
        return torch.sum(values[top_idx] * weights.unsqueeze(1), dim=0)

    def replay(self, mode: RuntimeMode) -> torch.Tensor:
        if not self._keys:
            return torch.zeros(self.dim, device=self.device)
        k = 1 if mode is RuntimeMode.NREM else min(3, len(self._keys))
        priority = torch.tensor(self._priority, dtype=torch.float32, device=self.device)
        top_idx = torch.topk(priority, k=k).indices
        values = torch.stack(self._values, dim=0)[top_idx]
        weights = torch.softmax(priority[top_idx], dim=0)
        return torch.sum(values * weights.unsqueeze(1), dim=0)

    def state_dict(self) -> dict[str, object]:
        keys = torch.stack(self._keys, dim=0).cpu() if self._keys else torch.empty((0, self.dim))
        values = torch.stack(self._values, dim=0).cpu() if self._values else torch.empty((0, self.dim))
        return {
            "dim": self.dim,
            "capacity": self.capacity,
            "keys": keys,
            "values": values,
            "priority": list(self._priority),
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        *,
        device: str | torch.device = "cpu",
    ) -> "HippocampusMemory":
        mem = cls(int(state["dim"]), capacity=int(state["capacity"]), device=device)
        keys = state.get("keys", torch.empty((0, mem.dim)))
        values = state.get("values", torch.empty((0, mem.dim)))
        priority = state.get("priority", [])
        if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor):
            for idx, score in enumerate(priority):
                mem._keys.append(keys[idx].to(mem.device).float())
                mem._values.append(values[idx].to(mem.device).float())
                mem._priority.append(float(score))
        return mem


class BrainRuntime:
    """Reference runtime stack.

    Layering:
    - kernel/coupling: sparse recurrent update over `weight`
    - mode update: `RuntimeMode`
    - hippocampus/replay: `HippocampusMemory`
    - global summary: `RuntimeStep` and `BrainRuntimeSnapshot`
    """
    def __init__(
        self,
        weight: torch.Tensor,
        *,
        config: BrainRuntimeConfig,
        backend: str = "auto",
        device: str | torch.device | None = None,
    ) -> None:
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError("weight must be a square matrix")
        if weight.shape[0] != config.dim:
            raise ValueError("weight dimension must match BrainRuntimeConfig.dim")

        self.config = config
        self.device = torch.device(device) if device is not None else weight.device
        self.backend = backend
        self.weight = weight.detach().float().to(self.device)
        pack_backend = "torch" if self.backend == "cuda" else self.backend
        values, col_idx, row_ptr = pack_sparse(
            self.weight.detach().cpu(),
            zero_tol=self.config.zero_tol,
            backend=pack_backend,
        )
        self.values = values.to(self.device)
        self.col_idx = col_idx.to(self.device)
        self.row_ptr = row_ptr.to(self.device)
        self.sparse_weight = torch.sparse_csr_tensor(
            self.row_ptr.to(torch.int64),
            self.col_idx.to(torch.int64),
            self.values,
            size=self.weight.shape,
            device=self.device,
            dtype=self.weight.dtype,
            check_invariants=False,
        )

        self.activation = torch.zeros(self.config.dim, device=self.device)
        self.refractory = torch.zeros(self.config.dim, device=self.device)
        self.memory_trace = torch.zeros(self.config.dim, device=self.device)
        self.adaptation = torch.zeros(self.config.dim, device=self.device)
        self.stp_u = torch.full((self.config.dim,), 0.5, device=self.device)
        self.stp_x = torch.ones(self.config.dim, device=self.device)
        self.bitfield = torch.zeros(self.config.dim, dtype=torch.uint8, device=self.device)
        self.goal = torch.zeros(self.config.dim, device=self.device)
        self.lifecycle = torch.full(
            (self.config.dim,),
            _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT],
            dtype=torch.int64,
            device=self.device,
        )
        self.inactive_steps = torch.zeros(self.config.dim, dtype=torch.int64, device=self.device)
        self.mode = RuntimeMode.WAKE
        self.sleep_pressure = 0.0
        self.arousal = 0.0
        self.step_index = 0
        self.circadian_phase = 0.0
        self.nrem_cycle_count = 0
        self.mode_occupancy: Dict[str, int] = {
            RuntimeMode.WAKE.value: 0,
            RuntimeMode.NREM.value: 0,
            RuntimeMode.REM.value: 0,
        }
        self.active_ratio_ema: float = float(self.config.active_ratio)

        # Dale's Law: E:I = 80:20 sign mask
        n_exc = int(self.config.dim * DALE_EI_RATIO)
        self.dale_sign = torch.ones(self.config.dim, device=self.device)
        self.dale_sign[n_exc:] = -DALE_INH_GAIN
        if self.config.dale_law:
            self.weight = self.weight.abs() * self.dale_sign.unsqueeze(1)
            self._rebuild_sparse()

        # Axon delay buffer: ring buffer of recent activations
        if self.config.axon_delay:
            self._delay_buffer = torch.zeros(
                self.config.max_axon_delay, self.config.dim, device=self.device
            )
            self._delay_idx = 0
        else:
            self._delay_buffer = None
            self._delay_idx = 0

        # Brainwave history for FFT
        self._brainwave_history: list[float] = []
        self._brainwave_max_len = 1024

        self.hippocampus = HippocampusMemory(
            self.config.dim,
            capacity=self.config.memory_capacity,
            device=self.device,
        )
        self.stdp_tracker = None
        if self.config.stdp_enabled:
            self.stdp_tracker = EligibilityTracker(
                STDPConfig(
                    dim=self.config.dim,
                    spike_threshold=self.config.stdp_spike_threshold,
                    lr=self.config.stdp_lr,
                ),
                device=self.device,
            )
        self._stdp_prev_critic_score = 0.0
        self._stdp_updates = 0
        self._last_stdp_gate = 0.0

    def _rebuild_sparse(self) -> None:
        """Rebuild CSR sparse weight from dense weight."""
        pack_backend = "torch" if self.backend == "cuda" else self.backend
        values, col_idx, row_ptr = pack_sparse(
            self.weight.detach().cpu(),
            zero_tol=self.config.zero_tol,
            backend=pack_backend,
        )
        self.values = values.to(self.device)
        self.col_idx = col_idx.to(self.device)
        self.row_ptr = row_ptr.to(self.device)
        self.sparse_weight = torch.sparse_csr_tensor(
            self.row_ptr.to(torch.int64),
            self.col_idx.to(torch.int64),
            self.values,
            size=self.weight.shape,
            device=self.device,
            dtype=self.weight.dtype,
            check_invariants=False,
        )

    def _apply_dale_sign(self) -> None:
        if not self.config.dale_law:
            return
        self.weight = self.weight.abs() * self.dale_sign.unsqueeze(1)

    def _apply_runtime_stdp(self, active_count: int, energy: float) -> float:
        """Optional F14 closed-loop plasticity over the runtime weight matrix."""
        if self.stdp_tracker is None:
            self._last_stdp_gate = 0.0
            return 0.0

        tick = self.step_index + 1
        if tick % self.config.stdp_interval != 0:
            self._last_stdp_gate = 0.0
            return 0.0

        self.stdp_tracker.update(self.activation)
        if tick % self.config.stdp_apply_interval != 0:
            self._last_stdp_gate = 0.0
            return 0.0

        active_ratio = float(active_count) / float(max(self.config.dim, 1))
        gate = compute_learning_gate(
            critic_score=float(energy),
            prev_critic_score=self._stdp_prev_critic_score,
            active_ratio=active_ratio,
            alpha_g=self.stdp_tracker.config.alpha_g,
        )
        self._stdp_prev_critic_score = float(energy)
        self._last_stdp_gate = float(gate)

        if abs(gate) <= self.config.stdp_gate_threshold:
            return float(gate)

        self.weight = apply_stdp_update(
            self.weight,
            self.stdp_tracker,
            gate,
            lr=self.config.stdp_lr,
            density=self.config.stdp_density,
        ).to(self.device)
        self.weight.fill_diagonal_(0.0)
        self._apply_dale_sign()
        self._rebuild_sparse()
        self.stdp_tracker.reset()
        self._stdp_updates += 1
        return float(gate)

    def brainwave_observable(self) -> dict[str, float]:
        """Compute global brainwave and band powers via FFT (Layer B / F.21)."""
        psi = float(self.activation.abs().mean().item())
        self._brainwave_history.append(psi)
        if len(self._brainwave_history) > self._brainwave_max_len:
            self._brainwave_history = self._brainwave_history[-self._brainwave_max_len:]
        result: dict[str, float] = {"psi_global": psi}
        if len(self._brainwave_history) < 8:
            return result
        sig = torch.tensor(self._brainwave_history, dtype=torch.float32)
        fft_vals = torch.fft.rfft(sig - sig.mean())
        power = (fft_vals.abs() ** 2) / len(sig)
        fs = 1000.0  # 1 step = 1ms
        freqs = torch.fft.rfftfreq(len(sig), d=1.0 / fs)
        for name, (lo, hi) in [
            ("delta", BAND_DELTA), ("theta", BAND_THETA),
            ("alpha", BAND_ALPHA), ("beta", BAND_BETA), ("gamma", BAND_GAMMA),
        ]:
            mask = (freqs >= lo) & (freqs < hi)
            result[name] = float(power[mask].sum().item()) if mask.any() else 0.0
        return result

    def energy_full(self) -> float:
        """Full energy E({a_i}) per 15_Equations.md B.3."""
        coupling = -0.5 * torch.dot(self.activation, self._matvec(self.activation))
        local = -(self.refractory * self.activation).sum()
        adapt = -ADAPTATION_COUPLING * (self.adaptation * self.activation).sum()
        return float((coupling + local + adapt).item())

    def compute_self_state(self) -> dict[str, float]:
        """Layer E: Self_t = S(G_t) -- global self-state summary."""
        active_frac = float(self.active_mask().float().mean().item())
        target = torch.tensor([ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO])
        lc = self.lifecycle_counts()
        total = max(sum(lc.values()), 1)
        current = torch.tensor([
            lc.get("ACTIVE", 0) / total,
            (lc.get("IDLE", 0) + lc.get("SLEEPING", 0)) / total,
            lc.get("DORMANT", 0) / total,
        ])
        bootstrap_deviation = float((current - target).norm().item())
        return {
            "active_fraction": active_frac,
            "bootstrap_deviation": bootstrap_deviation,
            "sleep_pressure": self.sleep_pressure,
            "arousal": self.arousal,
            "mode": self.mode.value,
            "energy": self.energy_full(),
            "consciousness_depth": 0.0,  # filled by agent layer
        }

    def set_goal(self, goal: torch.Tensor | None) -> None:
        if goal is None:
            self.goal.zero_()
            return
        goal = goal.detach().float().to(self.device)
        if goal.numel() != self.config.dim:
            raise ValueError("goal size must match runtime dimension")
        self.goal = goal.view(self.config.dim)

    def active_mask(self) -> torch.Tensor:
        return self.lifecycle == _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE]

    def lifecycle_counts(self) -> Dict[str, int]:
        counts = {}
        for code, lifecycle in _CODE_TO_LIFECYCLE.items():
            counts[lifecycle.value] = int((self.lifecycle == code).sum().item())
        return counts

    def mode_occupancy_kl(self, eps: float = 1e-9) -> Dict[str, float]:
        """F3 ergodic gate (docs/7_AGI/12_Equation.md A.3).

        Reports the empirical mode occupancy measure pi_brain on the 3-simplex
        and its KL divergence to the CE bootstrap fixed point
        p* = (Omega_Lambda, Omega_DM, Omega_b) = (BACKGROUND_RATIO, STRUCT_RATIO, ACTIVE_RATIO).

        Mapping: WAKE -> Omega_Lambda, NREM -> Omega_DM, REM -> Omega_b.
        """
        total = sum(self.mode_occupancy.values())
        if total <= 0:
            return {
                "samples": 0,
                "pi_wake": 0.0,
                "pi_nrem": 0.0,
                "pi_rem": 0.0,
                "kl_to_p_star": float("nan"),
            }
        pi_wake = self.mode_occupancy.get(RuntimeMode.WAKE.value, 0) / total
        pi_nrem = self.mode_occupancy.get(RuntimeMode.NREM.value, 0) / total
        pi_rem = self.mode_occupancy.get(RuntimeMode.REM.value, 0) / total
        pi = (pi_wake, pi_nrem, pi_rem)
        p_star = (BACKGROUND_RATIO, STRUCT_RATIO, ACTIVE_RATIO)
        kl = 0.0
        for p_i, q_i in zip(pi, p_star):
            if p_i > eps:
                kl += p_i * (np.log(p_i + eps) - np.log(q_i + eps))
        return {
            "samples": total,
            "pi_wake": pi_wake,
            "pi_nrem": pi_nrem,
            "pi_rem": pi_rem,
            "kl_to_p_star": float(kl),
        }

    def reset_mode_occupancy(self) -> None:
        """Zero the F3 mode occupancy counter (e.g. between sleep cycles)."""
        for key in self.mode_occupancy:
            self.mode_occupancy[key] = 0

    def _f1_effective_budget(self, mode: RuntimeMode) -> int:
        """Self-measured energy budget (gate F1, docs/7_AGI/12_Equation.md A.2 #2).

        Static fallback: config.energy_budget(mode). When f1_self_measure is
        on, the empirical EMA p_emp is convexly pulled toward ACTIVE_RATIO:
            r_eff = clip(beta * ACTIVE_RATIO + (1 - beta) * ema, lo, hi).
        Mode multipliers are preserved (1.0/0.5/0.75 for WAKE/NREM/REM).
        """
        if not self.config.f1_self_measure:
            return self.config.energy_budget(mode)
        beta = self.config.f1_pull_strength
        r_eff = beta * ACTIVE_RATIO + (1.0 - beta) * self.active_ratio_ema
        r_eff = min(max(r_eff, self.config.f1_min_ratio), self.config.f1_max_ratio)
        base = max(1, int(round(self.config.dim * r_eff)))
        if mode is RuntimeMode.NREM:
            return max(1, int(round(base * 0.5)))
        if mode is RuntimeMode.REM:
            return max(1, int(round(base * 0.75)))
        return base

    def _f1_update_ema(self, active_count: int) -> None:
        if not self.config.f1_self_measure:
            return
        p_emp = float(active_count) / float(self.config.dim)
        alpha = self.config.f1_ema_alpha
        self.active_ratio_ema = (1.0 - alpha) * self.active_ratio_ema + alpha * p_emp

    def bridge_gate_report(self) -> Dict[str, Dict[str, float]]:
        """AGI bridge gate aggregator (docs/7_AGI/12_Equation.md appendix A).

        Returns whatever measurements are currently available. Gate keys
        always exist; values are scalar reports or empty dicts when the
        underlying signal is not yet measurable.

        Current coverage:
        - F1 (self-organization, A.2 #2): empirical p_emp EMA vs ACTIVE_RATIO.
          Always reported; deviation is meaningful even when feedback is off.
        - F2 (ISS ball, A.1): exposed only when relax has been driven by a
          higher-level engine (see reality_stone.clarus.ce_ops.relax hist['iss']).
          BrainRuntime itself does not run the gradient relax, so F2 here is
          left empty by design.
        - F3 (ergodic KL, A.3): wraps mode_occupancy_kl().
        - F4 (PCI regression, A.4) is an experiment-level gate.
        """
        return {
            "F1_self_organization": {
                "active_ratio_ema": float(self.active_ratio_ema),
                "active_ratio_target": float(ACTIVE_RATIO),
                "deviation": float(self.active_ratio_ema - ACTIVE_RATIO),
                "self_measure_on": float(self.config.f1_self_measure),
            },
            "F2_iss_ball": {},
            "F3_ergodic_kl": self.mode_occupancy_kl(),
            "F4_pci_regression": {},
        }

    def _matvec(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.sparse_weight, x.unsqueeze(1)).squeeze(1)

    def _select_active(self, salience: torch.Tensor, budget: int) -> torch.Tensor:
        budget = max(0, min(int(budget), salience.numel()))
        mask = torch.zeros_like(salience, dtype=torch.bool)
        if budget == 0:
            return mask
        eligible = salience >= self.config.active_threshold
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            return mask
        budget = min(budget, eligible_count)
        scored = salience.masked_fill(~eligible, float("-inf"))
        _, idx = torch.topk(scored, k=budget)
        mask[idx] = True
        return mask

    def _auto_mode(self, external_norm: float) -> RuntimeMode:
        if self.mode is RuntimeMode.WAKE:
            if self.sleep_pressure > 1.0 and external_norm < self.config.wake_threshold:
                return RuntimeMode.NREM
            return RuntimeMode.WAKE
        if self.mode is RuntimeMode.NREM:
            if external_norm > self.config.wake_threshold * 1.5:
                return RuntimeMode.WAKE
            if self.sleep_pressure < 0.45:
                return RuntimeMode.REM
            return RuntimeMode.NREM
        if external_norm > self.config.wake_threshold or self.sleep_pressure < 0.15:
            return RuntimeMode.WAKE
        return RuntimeMode.REM

    def _update_sleep_state(self, mode: RuntimeMode, active_count: int, external_norm: float) -> None:
        """Borbely 2-Process model with circadian (15_Equations.md C.2)."""
        self.arousal = float(external_norm)
        tau_w_inv = 1.0 / TAU_W_STEPS
        tau_s_inv = 1.0 / TAU_S_STEPS

        # Process C: circadian modulation
        self.circadian_phase += 1.0
        circadian = CIRCADIAN_BASE + CIRCADIAN_AMP * math.cos(
            2.0 * math.pi * self.circadian_phase / CIRCADIAN_PERIOD
        )

        # Process S: homeostatic pressure
        if mode is RuntimeMode.WAKE:
            self.sleep_pressure += (SLEEP_PRESSURE_MAX - self.sleep_pressure) * tau_w_inv
        elif mode is RuntimeMode.NREM:
            self.sleep_pressure -= self.sleep_pressure * tau_s_inv
            self.nrem_cycle_count += 1
        else:
            self.sleep_pressure -= self.sleep_pressure * tau_s_inv * REM_TAU_FACTOR
        self.sleep_pressure = float(max(0.0, min(self.sleep_pressure, SLEEP_PRESSURE_MAX)))
        self._circadian_value = circadian

    def nrem_target_length(self) -> float:
        """T_NREM(n) = T0 * alpha^n -- decreasing NREM length within a night."""
        base = TAU_S_STEPS * 2.0
        return base * (NREM_LENGTH_DECAY ** self.nrem_cycle_count)

    def _update_lifecycle(self, salience: torch.Tensor, active_mask: torch.Tensor) -> None:
        self.inactive_steps = torch.where(
            active_mask,
            torch.zeros_like(self.inactive_steps),
            self.inactive_steps + 1,
        )
        lifecycle = torch.full_like(
            self.lifecycle,
            _LIFECYCLE_TO_CODE[ModuleLifecycle.IDLE],
        )
        lifecycle[salience < self.config.idle_threshold] = _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT]
        lifecycle[
            (self.inactive_steps >= self.config.dormant_after)
            & (salience < self.config.idle_threshold)
        ] = _LIFECYCLE_TO_CODE[ModuleLifecycle.DORMANT]
        lifecycle[self.inactive_steps >= self.config.sleeping_after] = _LIFECYCLE_TO_CODE[
            ModuleLifecycle.SLEEPING
        ]
        lifecycle[active_mask] = _LIFECYCLE_TO_CODE[ModuleLifecycle.ACTIVE]
        self.lifecycle = lifecycle

    def _energy(self, recurrent: torch.Tensor, replay: torch.Tensor) -> float:
        coupling = 0.5 * torch.dot(self.activation, recurrent).abs()
        local = (self.refractory.mean()
                 + 0.25 * self.memory_trace.abs().mean()
                 + 0.10 * self.adaptation.abs().mean())
        replay_term = 0.1 * replay.abs().mean()
        total = coupling + local + replay_term
        return float(total.item())

    def _use_rust(self) -> bool:
        if not _HAS_RUST_KERNEL:
            return False
        if self.backend == "rust":
            return True
        if self.backend == "auto" and self.device.type == "cpu":
            return True
        return False

    def _step_rust(
        self,
        external: torch.Tensor,
        replay: torch.Tensor,
        mode: RuntimeMode,
    ) -> tuple[int, float]:
        """Delegate the cell-step hot path to the Rust kernel."""
        budget = self.config.energy_budget(mode)
        mode_int = _MODE_TO_INT.get(mode.value, 0)
        act_np = self.activation.detach().cpu().numpy().astype(np.float32)
        ref_np = self.refractory.detach().cpu().numpy().astype(np.float32)
        mem_np = self.memory_trace.detach().cpu().numpy().astype(np.float32)
        adapt_np = self.adaptation.detach().cpu().numpy().astype(np.float32)
        su_np = self.stp_u.detach().cpu().numpy().astype(np.float32)
        sx_np = self.stp_x.detach().cpu().numpy().astype(np.float32)
        bit_np = self.bitfield.detach().cpu().numpy().astype(np.uint8)
        active_np = self.active_mask().detach().cpu().numpy().astype(np.uint8)
        ext_np = external.detach().cpu().numpy().astype(np.float32)
        goal_np = self.goal.detach().cpu().numpy().astype(np.float32)
        replay_np = replay.detach().cpu().numpy().astype(np.float32)
        noise_scale = {
            RuntimeMode.WAKE: 1.0,
            RuntimeMode.NREM: 0.3,
            RuntimeMode.REM: 0.7,
        }[mode]
        gen = torch.Generator(device=self.activation.device)
        gen.manual_seed(self.step_index * 31337 + 7)
        noise = self.config.noise_sigma * noise_scale * torch.randn(
            self.activation.shape,
            generator=gen,
            device=self.activation.device,
            dtype=self.activation.dtype,
        )
        noise_np = noise.detach().cpu().numpy().astype(np.float32)
        val_np = self.values.detach().cpu().numpy().astype(np.float32)
        col_np = self.col_idx.detach().cpu().numpy().astype(np.int32)
        row_np = self.row_ptr.detach().cpu().numpy().astype(np.int32)

        (new_act, new_ref, new_mem, new_adapt,
         new_su, new_sx, new_bit, active_count, energy) = _rust_brain_step(
            val_np, col_np, row_np,
            act_np, ref_np, mem_np, adapt_np, su_np, sx_np, bit_np,
            active_np, ext_np, goal_np, replay_np, noise_np,
            mode_int, budget,
            self.config.activation_decay(mode),
            self.config.activation_gain(mode),
            self.config.refractory_decay(mode),
            self.config.refractory_gain(mode),
            self.config.replay_mix(mode),
            self.config.refractory_scale,
            self.config.goal_gain,
            self.config.external_gain,
            self.config.bit_lower_threshold,
            self.config.bit_upper_threshold,
            STP_TAU_FAC_INV,
            STP_TAU_REC,
            STP_U_BASE,
            ADAPTATION_COUPLING,
            ADAPTATION_DECAY,
            MEMORY_TRACE_DECAY,
            ADAPTATION_CLAMP,
        )
        self.activation = torch.from_numpy(np.array(new_act, dtype=np.float32)).to(self.device)
        self.refractory = torch.from_numpy(np.array(new_ref, dtype=np.float32)).to(self.device)
        self.memory_trace = torch.from_numpy(np.array(new_mem, dtype=np.float32)).to(self.device)
        self.adaptation = torch.from_numpy(np.array(new_adapt, dtype=np.float32)).to(self.device)
        self.stp_u = torch.from_numpy(np.array(new_su, dtype=np.float32)).to(self.device)
        self.stp_x = torch.from_numpy(np.array(new_sx, dtype=np.float32)).to(self.device)
        self.bitfield = torch.from_numpy(np.array(new_bit, dtype=np.uint8)).to(self.device)
        return int(active_count), float(energy)

    def _compute_salience(
        self,
        activation: torch.Tensor,
        external: torch.Tensor,
        replay: torch.Tensor,
        refractory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute module salience for active selection (shared by step logic)."""
        return (
            activation.abs()
            + 0.35 * external.abs()
            + 0.25 * replay.abs()
            + 0.20 * self.goal.abs()
            - 0.15 * refractory
        )

    def _step_torch(
        self,
        external: torch.Tensor,
        replay: torch.Tensor,
        mode: RuntimeMode,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Pure-torch cell step (fallback path). Eq A.1--A.7, J.19--J.20.

        Returns (salience, recurrent, energy) to avoid recomputation in step().
        """
        prev_active = self.active_mask().float()

        spike = prev_active
        stp_u = self.stp_u + (-STP_TAU_FAC_INV * self.stp_u + STP_U_BASE * (1.0 - self.stp_u) * spike)
        stp_x = self.stp_x + (STP_TAU_REC * (1.0 - self.stp_x) - self.stp_u * self.stp_x * spike)
        stp_u = stp_u.clamp(0.0, 1.0)
        stp_x = stp_x.clamp(0.0, 1.0)

        pre = stp_u * stp_x * self.activation * prev_active

        # Axon delay: use delayed activation for recurrent input
        if self._delay_buffer is not None:
            delayed = self._delay_buffer[self._delay_idx % self.config.max_axon_delay]
            pre_delayed = stp_u * stp_x * delayed * prev_active
            recurrent = self._matvec(pre_delayed)
            self._delay_buffer[self._delay_idx % self.config.max_axon_delay] = self.activation.detach()
            self._delay_idx += 1
        else:
            recurrent = self._matvec(pre)

        adapt_force = ADAPTATION_COUPLING * self.adaptation

        # Noise injection (15_Equations A.2): mode-scaled, seeded for reproducibility
        noise_scale = {
            RuntimeMode.WAKE: 1.0,
            RuntimeMode.NREM: 0.3,
            RuntimeMode.REM: 0.7,
        }[mode]
        gen = torch.Generator(device=self.activation.device)
        gen.manual_seed(self.step_index * 31337 + 7)
        noise = self.config.noise_sigma * noise_scale * torch.randn(
            self.activation.shape, generator=gen, device=self.activation.device, dtype=self.activation.dtype
        )

        drive = (
            recurrent
            + self.config.external_gain * external
            + self.config.goal_gain * self.goal
            + self.config.replay_mix(mode) * replay
            - self.config.refractory_scale * self.refractory
            - adapt_force
            + noise
        )
        activation = (
            (1.0 - self.config.activation_decay(mode)) * self.activation
            + self.config.activation_gain(mode) * torch.tanh(drive)
        ).clamp(-1.0, 1.0)
        refractory = (
            (1.0 - self.config.refractory_decay(mode)) * self.refractory
            + self.config.refractory_gain(mode) * activation.square()
        )
        memory_trace = (1.0 - MEMORY_TRACE_DECAY) * self.memory_trace + MEMORY_TRACE_DECAY * activation
        adaptation = (
            (1.0 - ADAPTATION_DECAY) * self.adaptation + ADAPTATION_DECAY * activation.square()
        ).clamp(0.0, ADAPTATION_CLAMP)

        bitfield = self.bitfield.clone()
        bitfield[activation >= self.config.bit_upper_threshold] = 1
        bitfield[activation <= self.config.bit_lower_threshold] = 0

        self.activation = activation
        self.refractory = refractory
        self.memory_trace = memory_trace
        self.adaptation = adaptation
        self.stp_u = stp_u
        self.stp_x = stp_x
        self.bitfield = bitfield

        salience = self._compute_salience(activation, external, replay, refractory)
        energy = self._energy(recurrent, replay)
        return salience, recurrent, energy

    def step(
        self,
        *,
        external_input: torch.Tensor | None = None,
        cue: torch.Tensor | None = None,
        force_mode: RuntimeMode | None = None,
    ) -> RuntimeStep:
        external = (
            torch.zeros(self.config.dim, device=self.device)
            if external_input is None
            else external_input.detach().float().to(self.device).view(self.config.dim)
        )
        cue = self.activation if cue is None else cue.detach().float().to(self.device).view(self.config.dim)
        external_norm = float(external.norm().item())
        mode = force_mode or self._auto_mode(external_norm)
        replay = self.hippocampus.recall(cue, topk=self.config.memory_topk)
        if mode is not RuntimeMode.WAKE and len(self.hippocampus) > 0:
            replay = 0.5 * replay + 0.5 * self.hippocampus.replay(mode)

        if self._use_rust():
            active_count, energy = self._step_rust(external, replay, mode)
            salience = self._compute_salience(self.activation, external, replay, self.refractory)
        else:
            salience, _recurrent, energy = self._step_torch(external, replay, mode)

        active_mask = self._select_active(salience, self._f1_effective_budget(mode))
        active_count = int(active_mask.sum().item())
        self._f1_update_ema(active_count)
        stdp_gate = self._apply_runtime_stdp(active_count, energy)
        self.mode = mode
        self.mode_occupancy[mode.value] = self.mode_occupancy.get(mode.value, 0) + 1
        self._update_lifecycle(salience, active_mask)

        priority = float((salience[active_mask].mean().item() if active_count else salience.mean().item()) + external_norm)
        if mode is RuntimeMode.WAKE and (external_norm > NORM_EPS or self.goal.norm().item() > NORM_EPS):
            self.hippocampus.encode(self.activation, value=self.memory_trace, priority=priority)
        elif mode is not RuntimeMode.WAKE and len(self.hippocampus) > 0:
            consolidated = 0.85 * self.activation + 0.15 * replay
            self.hippocampus.encode(consolidated, value=self.memory_trace, priority=priority * 0.5)

        self.hippocampus.decay_priorities()
        self._update_sleep_state(mode, active_count, external_norm)
        self.brainwave_observable()
        self.step_index += 1
        return RuntimeStep(
            step=self.step_index,
            mode=self.mode,
            energy=energy,
            active_modules=active_count,
            replay_norm=float(replay.norm().item()),
            sleep_pressure=self.sleep_pressure,
            arousal=self.arousal,
            lifecycle_counts=self.lifecycle_counts(),
            stdp_gate=stdp_gate,
            stdp_updates=self._stdp_updates,
        )

    def snapshot(self) -> BrainRuntimeSnapshot:
        return BrainRuntimeSnapshot(
            config=self.config,
            weight=self.weight.detach().cpu(),
            activation=self.activation.detach().cpu(),
            refractory=self.refractory.detach().cpu(),
            memory_trace=self.memory_trace.detach().cpu(),
            adaptation=self.adaptation.detach().cpu(),
            stp_u=self.stp_u.detach().cpu(),
            stp_x=self.stp_x.detach().cpu(),
            bitfield=self.bitfield.detach().cpu(),
            goal=self.goal.detach().cpu(),
            lifecycle=self.lifecycle.detach().cpu(),
            inactive_steps=self.inactive_steps.detach().cpu(),
            mode=self.mode,
            sleep_pressure=float(self.sleep_pressure),
            arousal=float(self.arousal),
            step=self.step_index,
            hippocampus=self.hippocampus.state_dict(),
            mode_occupancy=dict(self.mode_occupancy),
            active_ratio_ema=float(self.active_ratio_ema),
            stdp_tracker=(
                self.stdp_tracker.state_dict() if self.stdp_tracker is not None else None
            ),
            stdp_prev_critic_score=float(self._stdp_prev_critic_score),
            stdp_updates=int(self._stdp_updates),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: BrainRuntimeSnapshot,
        *,
        backend: str = "auto",
        device: str | torch.device | None = None,
    ) -> "BrainRuntime":
        runtime = cls(
            snapshot.weight,
            config=snapshot.config,
            backend=backend,
            device=device,
        )
        runtime.activation = snapshot.activation.to(runtime.device).float()
        runtime.refractory = snapshot.refractory.to(runtime.device).float()
        runtime.memory_trace = snapshot.memory_trace.to(runtime.device).float()
        runtime.adaptation = snapshot.adaptation.to(runtime.device).float()
        runtime.stp_u = snapshot.stp_u.to(runtime.device).float()
        runtime.stp_x = snapshot.stp_x.to(runtime.device).float()
        runtime.bitfield = snapshot.bitfield.to(runtime.device).to(torch.uint8)
        runtime.goal = snapshot.goal.to(runtime.device).float()
        runtime.lifecycle = snapshot.lifecycle.to(runtime.device).to(torch.int64)
        runtime.inactive_steps = snapshot.inactive_steps.to(runtime.device).to(torch.int64)
        runtime.mode = snapshot.mode
        runtime.sleep_pressure = float(snapshot.sleep_pressure)
        runtime.arousal = float(snapshot.arousal)
        runtime.step_index = int(snapshot.step)
        runtime.hippocampus = HippocampusMemory.from_state_dict(
            snapshot.hippocampus,
            device=runtime.device,
        )
        if snapshot.mode_occupancy:
            for key in runtime.mode_occupancy:
                runtime.mode_occupancy[key] = int(snapshot.mode_occupancy.get(key, 0))
        if snapshot.active_ratio_ema >= 0.0:
            runtime.active_ratio_ema = float(snapshot.active_ratio_ema)
        runtime._stdp_prev_critic_score = float(snapshot.stdp_prev_critic_score)
        runtime._stdp_updates = int(snapshot.stdp_updates)
        if runtime.stdp_tracker is not None and snapshot.stdp_tracker is not None:
            runtime.stdp_tracker.load_state_dict(snapshot.stdp_tracker)
        return runtime
```
---
## File: `reality_stone/python/reality_stone/clarus/sleep.py`

```python
"""Wake/NREM/REM refinement for standalone CE artifacts."""

from __future__ import annotations

import argparse
from collections import deque
import importlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

try:
    from .engine import CEEngine, DEFAULT_PROMPTS, state_partition_counts
    from .ce_ops import pq_build_codebook
    from .utils import safe_print
except ImportError:
    from reality_stone.clarus.engine import CEEngine, DEFAULT_PROMPTS, state_partition_counts
    from reality_stone.clarus.ce_ops import pq_build_codebook
    from reality_stone.clarus.utils import safe_print


@dataclass
class SleepBatch:
    state_x: torch.Tensor
    prev_x: torch.Tensor
    target_y: torch.Tensor
    soft_y: torch.Tensor
    hard_mask: torch.Tensor
    top1_hits: torch.Tensor
    top50_hits: torch.Tensor
    target_ids: torch.Tensor
    top10_hits: torch.Tensor | None = None
    risk_scores: torch.Tensor | None = None
    phase_risk_scores: torch.Tensor | None = None
    teacher_top_ids: torch.Tensor | None = None
    teacher_top_probs: torch.Tensor | None = None


@dataclass
class DecoderTokenHead:
    token_ids: torch.Tensor
    state_proj: torch.Tensor | None
    prev_proj: torch.Tensor | None
    bias: torch.Tensor | None
    scale: float = 1.0


@dataclass
class PromptReplayBuffer:
    capacity: int
    prompts: deque[str] = field(default_factory=deque)

    def add(self, prompt: str):
        if not prompt:
            return
        self.prompts.append(prompt)
        while len(self.prompts) > self.capacity:
            self.prompts.popleft()

    def extend(self, prompts: list[str]):
        for prompt in prompts:
            self.add(prompt)

    def items(self) -> list[str]:
        return list(self.prompts)


DEFAULT_CORPUS_DATASET = "lcw99/wikipedia-korean-20221001"
DEFAULT_CORPUS_SPLIT = "train"
DEFAULT_CORPUS_TEXT_COLUMN = "text"


def _split_corpus_documents(text: str) -> list[str]:
    docs: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                docs.append(" ".join(current))
                current = []
            continue
        current.append(line)
    if current:
        docs.append(" ".join(current))
    if docs:
        return docs
    fallback = " ".join(part.strip() for part in text.split() if part.strip())
    return [fallback] if fallback else []


def _chunk_document(text: str, *, max_chars: int = 320, min_chars: int = 64) -> list[str]:
    cleaned = " ".join(part.strip() for part in text.split() if part.strip())
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    segments = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
        if part.strip()
    ]
    if not segments:
        segments = [cleaned]

    chunks: list[str] = []
    current = ""
    for segment in segments:
        if len(segment) > max_chars:
            words = segment.split()
            partial = ""
            for word in words:
                candidate = word if not partial else f"{partial} {word}"
                if len(candidate) <= max_chars:
                    partial = candidate
                    continue
                if len(partial) >= min_chars:
                    chunks.append(partial)
                    partial = word
                else:
                    overflow = candidate[:max_chars].strip()
                    if overflow:
                        chunks.append(overflow)
                    partial = candidate[max_chars:].strip()
            if partial:
                if len(partial) >= min_chars or not chunks:
                    chunks.append(partial)
            current = ""
            continue

        candidate = segment if not current else f"{current} {segment}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = segment
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def load_corpus_documents(
    data_path: str | None = None,
    *,
    dataset_name: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str = DEFAULT_CORPUS_SPLIT,
    text_column: str = DEFAULT_CORPUS_TEXT_COLUMN,
    doc_limit: int = 256,
    text_limit: int = 1_000_000,
) -> list[str]:
    if data_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing corpus file: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs: list[str] = []
        for raw_doc in _split_corpus_documents(text[: int(text_limit)]):
            docs.extend(_chunk_document(raw_doc))
            if len(docs) >= max(int(doc_limit), 1):
                break
        return docs[: max(int(doc_limit), 1)]

    dataset_name = dataset_name or DEFAULT_CORPUS_DATASET
    try:
        datasets = importlib.import_module("datasets")
        ds = datasets.load_dataset(dataset_name, dataset_config, split=dataset_split)
    except Exception as exc:
        raise RuntimeError(
            "Pass --data or install the 'datasets' package to load an external Korean corpus."
        ) from exc

    docs: list[str] = []
    total_chars = 0
    for row in ds:
        text = row.get(text_column)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        chunks = _chunk_document(text)
        if not chunks:
            continue
        docs.extend(chunks)
        total_chars += sum(len(chunk) for chunk in chunks)
        if len(docs) >= max(int(doc_limit), 1) or total_chars >= int(text_limit):
            break
    if not docs:
        raise RuntimeError("Loaded corpus is empty.")
    return docs


def _content_terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"[0-9A-Za-z가-힣]{2,}", text)}


def prioritize_documents_for_prompts(
    docs: list[str],
    prompts: list[str] | None,
) -> list[str]:
    if not docs or not prompts:
        return list(docs)

    prompt_weights: dict[str, int] = {}
    for prompt in prompts:
        for token in _content_terms(prompt):
            prompt_weights[token] = prompt_weights.get(token, 0) + 1

    if not prompt_weights:
        return list(docs)

    scored_docs: list[tuple[float, int, str]] = []
    for idx, doc in enumerate(docs):
        doc_tokens = _content_terms(doc)
        overlap = float(sum(prompt_weights.get(token, 0) for token in doc_tokens))
        scored_docs.append((overlap, idx, doc))

    prioritized = [
        doc
        for overlap, idx, doc in sorted(
            scored_docs,
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        if overlap > 0.0
    ]
    if not prioritized:
        return list(docs)

    ordered: list[str] = []
    seen: set[str] = set()
    for doc in [*prioritized, *docs]:
        if doc in seen:
            continue
        seen.add(doc)
        ordered.append(doc)
    return ordered


def ridge_solve(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    x = x.float()
    y = y.float()
    if weights is not None:
        w = weights.float().clamp_min(1e-6).sqrt().unsqueeze(1)
        x = x * w
        y = y * w
    xtx = x.T @ x
    xty = x.T @ y
    eye = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)
    return torch.linalg.solve(xtx + float(ridge) * eye, xty)


def fit_linear_with_bias(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge: float,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    x_aug = torch.cat([x.float(), ones], dim=1)
    out = ridge_solve(x_aug, y.float(), ridge, weights=weights)
    return out[:-1], out[-1]


def batch_weights(batch: SleepBatch, rem_weight: float) -> torch.Tensor:
    weights = torch.ones(
        batch.target_y.shape[0],
        dtype=batch.target_y.dtype,
        device=batch.target_y.device,
    )
    if batch.hard_mask.numel() and rem_weight > 1.0:
        weights[batch.hard_mask.bool()] = float(rem_weight)
    return weights


def fit_decoder_from_batch(
    batch: SleepBatch,
    *,
    prev_scale: float,
    ridge: float,
    rem_weight: float = 1.0,
    rem_mix: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = batch.target_y.clone()
    weights = batch_weights(batch, rem_weight)

    if batch.hard_mask.numel() and rem_mix > 0.0:
        hard = batch.hard_mask.bool()
        y[hard] = (1.0 - rem_mix) * y[hard] + rem_mix * batch.soft_y[hard]

    feat = torch.cat([batch.state_x.float(), float(prev_scale) * batch.prev_x.float()], dim=1)
    proj, bias = fit_linear_with_bias(feat, y, ridge=ridge, weights=weights)
    d = batch.state_x.shape[1]
    state_proj = proj[:d]
    prev_proj = proj[d : 2 * d]
    pred = batch.state_x @ state_proj + float(prev_scale) * (batch.prev_x @ prev_proj) + bias
    denom = float(pred.pow(2).sum().item())
    if denom > 1e-8:
        scale = float((pred * y).sum().item() / denom)
        state_proj = state_proj * scale
        prev_proj = prev_proj * scale
        bias = bias * scale
    return state_proj, prev_proj, bias


def fit_token_head_from_batch(
    batch: SleepBatch,
    *,
    prev_scale: float,
    ridge: float,
    rem_weight: float = 1.0,
    max_vocab: int = 2048,
    scale: float = 1.0,
) -> DecoderTokenHead | None:
    if batch.teacher_top_ids is None or batch.teacher_top_probs is None:
        return None

    top_ids = batch.teacher_top_ids.long()
    top_probs = batch.teacher_top_probs.float()
    if top_ids.numel() == 0 or top_probs.numel() == 0:
        return None

    flat_ids = top_ids.reshape(-1)
    flat_probs = top_probs.reshape(-1)
    uniq_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
    mass = torch.zeros(uniq_ids.shape[0], dtype=flat_probs.dtype, device=flat_probs.device)
    mass.scatter_add_(0, inverse, flat_probs)

    if max_vocab > 0 and uniq_ids.numel() > max_vocab:
        keep = torch.topk(mass, max_vocab).indices
        uniq_ids = uniq_ids.index_select(0, keep)
        uniq_ids, _ = torch.sort(uniq_ids)

    if uniq_ids.numel() == 0:
        return None

    token_map = {int(token_id): col for col, token_id in enumerate(uniq_ids.tolist())}
    y = torch.zeros(
        (top_ids.shape[0], uniq_ids.shape[0]),
        dtype=torch.float32,
        device=batch.state_x.device,
    )
    for row_idx, (row_ids, row_probs) in enumerate(zip(top_ids.tolist(), top_probs.tolist(), strict=False)):
        for token_id, prob in zip(row_ids, row_probs, strict=False):
            col_idx = token_map.get(int(token_id))
            if col_idx is not None:
                y[row_idx, col_idx] += float(prob)

    weights = batch_weights(batch, rem_weight).float()
    denom = weights.sum().clamp_min(1e-6)
    feat = torch.cat([batch.state_x.float(), float(prev_scale) * batch.prev_x.float()], dim=1)
    proj, bias = fit_linear_with_bias(feat, y, ridge=ridge, weights=weights)
    d = batch.state_x.shape[1]
    state_proj = proj[:d]
    prev_proj = proj[d : 2 * d]
    return DecoderTokenHead(
        token_ids=uniq_ids.long().cpu(),
        state_proj=state_proj.cpu(),
        prev_proj=prev_proj.cpu(),
        bias=bias.cpu(),
        scale=float(scale),
    )


def finetune_vocab_head_from_batch(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    lr: float = 1e-3,
    steps: int = 64,
    batch_size: int = 256,
    rem_weight: float = 1.0,
    weight_decay: float = 1e-4,
    soft_target_weight: float = 0.35,
) -> dict[str, float]:
    if batch.state_x.numel() == 0:
        return {"loss": 0.0, "top1_acc": 0.0, "top10_acc": 0.0, "steps": 0, "batch_size": 0}

    eng.ensure_vocab_head()
    assert eng.decoder_vocab_weight is not None

    state_x = batch.state_x.to(eng.device).float()
    prev_x = batch.prev_x.to(eng.device).float()
    target_ids = batch.target_ids.to(eng.device).long()
    sample_weights = batch_weights(batch, rem_weight).to(eng.device).float()
    teacher_top_ids = (
        None
        if batch.teacher_top_ids is None
        else batch.teacher_top_ids.to(eng.device).long()
    )
    teacher_top_probs = (
        None
        if batch.teacher_top_probs is None
        else batch.teacher_top_probs.to(eng.device).float()
    )
    soft_target_weight = min(max(float(soft_target_weight), 0.0), 1.0)

    weight = eng.decoder_vocab_weight.detach().clone().to(eng.device)
    bias = (
        torch.zeros(weight.shape[0], dtype=weight.dtype, device=eng.device)
        if eng.decoder_vocab_bias is None
        else eng.decoder_vocab_bias.detach().clone().to(eng.device)
    )
    weight.requires_grad_(True)
    bias.requires_grad_(True)

    optimizer = torch.optim.AdamW([weight, bias], lr=float(lr), weight_decay=float(weight_decay))
    total = int(state_x.shape[0])
    batch_size = max(1, min(int(batch_size), total))
    steps = max(1, int(steps))

    last_loss = 0.0

    for step in range(steps):
        start = (step * batch_size) % total
        end = start + batch_size
        if end <= total:
            idx = torch.arange(start, end, device=eng.device)
        else:
            tail = torch.arange(start, total, device=eng.device)
            head = torch.arange(0, end - total, device=eng.device)
            idx = torch.cat([tail, head], dim=0)

        query = eng.decoder_query(state_x.index_select(0, idx), prev_x.index_select(0, idx))
        logits = F.linear(query, weight, bias)
        target = target_ids.index_select(0, idx)
        hard_loss = F.cross_entropy(logits, target, reduction="none")
        loss = hard_loss
        if (
            teacher_top_ids is not None
            and teacher_top_probs is not None
            and soft_target_weight > 0.0
        ):
            top_ids = teacher_top_ids.index_select(0, idx).clamp(0, logits.shape[1] - 1)
            top_probs = teacher_top_probs.index_select(0, idx)
            top_probs = top_probs / top_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
            student_top_logits = logits.gather(1, top_ids)
            soft_loss = -(top_probs * F.log_softmax(student_top_logits, dim=1)).sum(dim=1)
            loss = (1.0 - soft_target_weight) * hard_loss + soft_target_weight * soft_loss
        weights = sample_weights.index_select(0, idx)
        loss = (loss * weights).sum() / weights.sum().clamp_min(1e-6)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    with torch.no_grad():
        query = eng.decoder_query(state_x, prev_x)
        logits = F.linear(query, weight, bias)
        top1 = float((logits.argmax(dim=-1) == target_ids).float().mean().item())
        top10_ids = torch.topk(logits, min(10, logits.shape[1]), dim=-1).indices
        top10 = float((top10_ids == target_ids.unsqueeze(1)).any(dim=1).float().mean().item())

    eng.apply_vocab_head(weight.detach().cpu(), bias=bias.detach().cpu(), scale=1.0)
    return {
        "loss": last_loss,
        "top1_acc": top1,
        "top10_acc": top10,
        "steps": float(steps),
        "batch_size": float(batch_size),
    }


def build_refresh_args(
    ce_args,
    *,
    steps: int,
    cb_topk: int,
    metric_rank: int,
    noise_scale: float,
):
    payload = vars(ce_args).copy()
    payload.update(
        steps=int(steps),
        cb_topk=int(cb_topk),
        metric_rank=int(metric_rank),
        noise_scale=float(noise_scale),
    )
    return argparse.Namespace(**payload)


def allocate_phase_sample_counts(
    total_samples: int,
    phase_profile: dict[str, float],
) -> dict[str, int]:
    total_samples = max(0, int(total_samples))
    if total_samples == 0:
        return {name: 0 for name in phase_profile}

    names = list(phase_profile)
    weights = [max(float(phase_profile[name]), 0.0) for name in names]
    weight_sum = sum(weights)
    if weight_sum <= 1e-8:
        base = total_samples // max(len(names), 1)
        counts = {name: base for name in names}
        for name in names[: total_samples - base * len(names)]:
            counts[name] += 1
        return counts

    raw = [total_samples * weight / weight_sum for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total_samples - sum(counts)
    order = sorted(
        range(len(names)),
        key=lambda idx: (raw[idx] - counts[idx], weights[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    positive = [idx for idx, weight in enumerate(weights) if weight > 0.0]
    if total_samples >= len(positive):
        missing = [idx for idx in positive if counts[idx] == 0]
        for receiver in missing:
            donors = sorted(
                (idx for idx in positive if counts[idx] > 1 and idx != receiver),
                key=lambda idx: (counts[idx] - raw[idx], counts[idx]),
                reverse=True,
            )
            if not donors:
                break
            donor = donors[0]
            counts[donor] -= 1
            counts[receiver] += 1

    return {name: int(count) for name, count in zip(names, counts, strict=False)}


def _build_sleep_batch(
    state_rows: list[torch.Tensor],
    prev_rows: list[torch.Tensor],
    target_rows: list[torch.Tensor],
    soft_rows: list[torch.Tensor],
    hard_rows: list[bool],
    top1_hits: list[bool],
    top10_hits: list[bool],
    top50_hits: list[bool],
    risk_rows: list[float],
    phase_risk_rows: list[float],
    target_ids: list[int],
    teacher_top_ids_rows: list[torch.Tensor],
    teacher_top_prob_rows: list[torch.Tensor],
) -> SleepBatch:
    return SleepBatch(
        state_x=torch.stack(state_rows, dim=0),
        prev_x=torch.stack(prev_rows, dim=0),
        target_y=torch.stack(target_rows, dim=0),
        soft_y=torch.stack(soft_rows, dim=0),
        hard_mask=torch.tensor(hard_rows, dtype=torch.bool),
        top1_hits=torch.tensor(top1_hits, dtype=torch.bool),
        top10_hits=torch.tensor(top10_hits, dtype=torch.bool),
        top50_hits=torch.tensor(top50_hits, dtype=torch.bool),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        risk_scores=torch.tensor(risk_rows, dtype=torch.float32),
        phase_risk_scores=torch.tensor(phase_risk_rows, dtype=torch.float32),
        teacher_top_ids=torch.stack(teacher_top_ids_rows, dim=0),
        teacher_top_probs=torch.stack(teacher_top_prob_rows, dim=0),
    )


def _mean_phase_grounding_risk(step_meta: dict[str, object]) -> float:
    risk = step_meta.get("phase_grounding_risk")
    if torch.is_tensor(risk) and risk.numel():
        return float(risk.float().mean().item())
    return 0.0


def _context_slice(full_ids: torch.Tensor, end_pos: int, window_tokens: int) -> torch.Tensor:
    start = max(0, int(end_pos) - max(int(window_tokens), 1))
    return full_ids[:, start:int(end_pos)]


def _target_distribution(
    eng: CEEngine,
    target_id: int,
    *,
    topk: int,
    teacher_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_emb = eng.token_embedding([target_id]).squeeze(0).detach()
    k = min(max(int(topk), 1), eng.vocab)
    if k <= 1:
        top_idx = torch.tensor([target_id], dtype=torch.long, device=eng.device)
        probs = torch.tensor([1.0], dtype=torch.float32, device=eng.device)
        return target_emb, top_idx.cpu(), probs.cpu(), target_emb

    if teacher_logits is not None:
        top_vals, top_idx_raw = torch.topk(teacher_logits.float(), min(k, teacher_logits.numel()))
        probs = F.softmax(top_vals, dim=0)
        ordered_ids = top_idx_raw.tolist()
        if target_id not in ordered_ids:
            ordered_ids = [target_id] + ordered_ids[:-1]
            probs_list = [0.5] + [p * 0.5 for p in probs.tolist()[:-1]]
            probs = torch.tensor(probs_list, dtype=torch.float32)
            probs = probs / probs.sum()
        top_idx = torch.tensor(ordered_ids, dtype=torch.long, device=eng.device)
        soft_target = (eng.token_embedding(ordered_ids).detach() * probs.unsqueeze(1).to(eng.device)).sum(0)
        return target_emb, top_idx.cpu(), probs.cpu(), soft_target

    soft_scores = eng.lexical_scores(target_emb)
    gather_k = min(max(k * 2, k + 1), soft_scores.numel())
    top_vals, top_idx = torch.topk(soft_scores, gather_k)
    neighbor_ids: list[int] = []
    neighbor_vals: list[float] = []
    seen = {int(target_id)}
    for token_id, score in zip(top_idx.tolist(), top_vals.tolist(), strict=False):
        token_int = int(token_id)
        if token_int in seen:
            continue
        seen.add(token_int)
        neighbor_ids.append(token_int)
        neighbor_vals.append(float(score))
        if len(neighbor_ids) >= k - 1:
            break

    ordered_ids = [int(target_id), *neighbor_ids]
    top_idx = torch.tensor(ordered_ids, dtype=torch.long, device=eng.device)
    probs = torch.zeros(top_idx.shape[0], dtype=torch.float32, device=eng.device)
    if top_idx.shape[0] == 1:
        probs[0] = 1.0
    else:
        target_mass = 0.85
        probs[0] = target_mass
        neighbor_scores = torch.tensor(neighbor_vals, dtype=torch.float32, device=eng.device)
        neighbor_probs = F.softmax(neighbor_scores, dim=0)
        probs[1:] = (1.0 - target_mass) * neighbor_probs
    soft_target = (probs.unsqueeze(1) * eng.token_embedding(top_idx).detach()).sum(dim=0)
    return target_emb, top_idx.detach().cpu(), probs.detach().cpu(), soft_target


def collect_sleep_batch(
    eng: CEEngine,
    prompts: list[str],
    ce_args,
    *,
    max_new_tokens: int,
    teacher_topk: int,
    refresh_interval: int = 0,
    refresh_steps: int = 48,
    refresh_cb_topk: int = 128,
    refresh_metric_rank: int = 0,
    refresh_noise_scale: float = 0.0,
    sample_budget: int | None = None,
    context_window: int = 64,
    seed_tokens: int = 8,
) -> SleepBatch:
    if not prompts:
        raise ValueError("collect_sleep_batch requires at least one prompt")
    prompts = [
        prompt
        for prompt in prompts
        if eng.tok.encode(prompt, return_tensors="pt").shape[1] > 1
    ]
    if not prompts:
        raise ValueError("collect_sleep_batch requires corpus entries with at least two tokens")

    state_rows: list[torch.Tensor] = []
    prev_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    soft_rows: list[torch.Tensor] = []
    hard_rows: list[bool] = []
    top1_hits: list[bool] = []
    top10_hits: list[bool] = []
    top50_hits: list[bool] = []
    risk_rows: list[float] = []
    phase_risk_rows: list[float] = []
    target_ids: list[int] = []
    teacher_top_ids_rows: list[torch.Tensor] = []
    teacher_top_prob_rows: list[torch.Tensor] = []
    refresh_args = None
    if refresh_interval > 0:
        refresh_args = build_refresh_args(
            ce_args,
            steps=refresh_steps,
            cb_topk=refresh_cb_topk,
            metric_rank=refresh_metric_rank,
            noise_scale=refresh_noise_scale,
        )

    from tqdm import tqdm as _tqdm

    target_samples = len(prompts) * max(1, int(max_new_tokens))
    if sample_budget is not None:
        target_samples = max(1, int(sample_budget))

    _pbar = _tqdm(total=target_samples, desc="    collect", unit="tok", ncols=80)
    prompt_idx = 0
    _collect_t0 = time.time()
    _collect_timeout = max(300.0, target_samples * 2.0)
    while len(state_rows) < target_samples:
        if time.time() - _collect_t0 > _collect_timeout:
            break
        prompt = prompts[prompt_idx % len(prompts)]
        prompt_idx += 1
        full_ids = eng.tok.encode(prompt, return_tensors="pt").to(eng.device)
        if full_ids.shape[1] <= 1:
            continue
        cursor = min(max(int(seed_tokens), 1), full_ids.shape[1] - 1)
        while cursor < full_ids.shape[1] and len(state_rows) < target_samples:
            ids = _context_slice(full_ids, cursor, context_window)
            if ids.shape[1] == 0:
                cursor += max(1, int(max_new_tokens))
                continue
            ctx = eng.context_from_ids(ids)
            if eng.model is not None:
                with torch.no_grad():
                    teacher_out = eng.model(ids, output_hidden_states=True)
                    ce_hidden = teacher_out.hidden_states[-1][0, -1].float().detach()
                phi_state = ctx.phi.detach()
            else:
                relax_result = eng.relax_context(ctx, ce_args)
                ce_hidden = eng.ce_hidden(relax_result["m_star"]).detach()
                phi_state = relax_result["phi_updated"].detach()
            init_layer = ctx.best_layer
            history_ids = ids[0].tolist()
            prev_hidden = None
            prev_prev_hidden = None
            context_anchor = ce_hidden.detach().clone()
            max_stop = min(full_ids.shape[1], cursor + max(1, int(max_new_tokens)))

            for target_pos in range(cursor, max_stop):
                target_id = int(full_ids[0, target_pos].item())
                prev_id = int(ids[0, -1].item())
                prev_emb = eng.token_embedding([prev_id]).squeeze(0).detach()

                teacher_logits = None
                if eng.model is not None:
                    with torch.no_grad():
                        teacher_out = eng.model(ids)
                        teacher_logits = teacher_out.logits[0, -1].detach().cpu()

                target_emb, top_idx, probs, soft_target = _target_distribution(
                    eng,
                    target_id,
                    topk=teacher_topk,
                    teacher_logits=teacher_logits,
                )

                standalone_logits, step_meta = eng.standalone_logits(
                    ce_hidden,
                    prev_id,
                    temperature=1.0,
                    history_ids=history_ids,
                    prev_hidden=prev_hidden,
                    prev_prev_hidden=prev_prev_hidden,
                    context_anchor=context_anchor,
                    return_meta=True,
                )
                top_ids = torch.topk(standalone_logits, min(50, standalone_logits.numel())).indices.tolist()
                stand_top1 = int(top_ids[0])
                hit1 = stand_top1 == target_id
                hit10 = target_id in top_ids[:10]
                hit50 = target_id in top_ids

                state_rows.append(ce_hidden.cpu())
                prev_rows.append(prev_emb.cpu())
                _pbar.update(1)
                target_rows.append(target_emb.cpu())
                soft_rows.append(soft_target.cpu())
                hard_rows.append(not hit50)
                top1_hits.append(hit1)
                top10_hits.append(hit10)
                top50_hits.append(hit50)
                risk_rows.append(float(step_meta["curvature_risk_score"]))
                phase_risk_rows.append(_mean_phase_grounding_risk(step_meta))
                target_ids.append(target_id)
                teacher_top_ids_rows.append(top_idx)
                teacher_top_prob_rows.append(probs)

                if len(state_rows) >= target_samples:
                    break

                step_hidden = ce_hidden.detach().clone()
                prev_prev_hidden = prev_hidden
                prev_hidden = step_hidden
                next_token = torch.tensor([[target_id]], device=eng.device)
                ids = torch.cat([ids, next_token], dim=1)
                if ids.shape[1] > int(context_window):
                    ids = ids[:, -int(context_window) :]
                history_ids.append(target_id)
                if (
                    refresh_args is not None
                    and target_pos + 1 < max_stop
                    and (target_pos - cursor + 1) % refresh_interval == 0
                ):
                    refresh_ctx = eng.context_from_ids(
                        ids,
                        init_layer=init_layer,
                        phi=phi_state,
                        need_teacher=False,
                    )
                    refresh_result = eng.relax_context(refresh_ctx, refresh_args)
                    ce_hidden = eng.ce_hidden(refresh_result["m_star"]).detach()
                    phi_state = refresh_result["phi_updated"].detach()
                    init_layer = refresh_ctx.best_layer
            cursor = max_stop

    _pbar.close()
    return _build_sleep_batch(
        state_rows,
        prev_rows,
        target_rows,
        soft_rows,
        hard_rows,
        top1_hits,
        top10_hits,
        top50_hits,
        risk_rows,
        phase_risk_rows,
        target_ids,
        teacher_top_ids_rows,
        teacher_top_prob_rows,
    )


def batch_stats(batch: SleepBatch) -> dict[str, float]:
    top1 = batch.top1_hits.float().mean().item() if batch.top1_hits.numel() else 0.0
    top10 = batch.top10_hits.float().mean().item() if batch.top10_hits is not None and batch.top10_hits.numel() else 0.0
    top50 = batch.top50_hits.float().mean().item() if batch.top50_hits.numel() else 0.0
    hard = batch.hard_mask.float().mean().item() if batch.hard_mask.numel() else 0.0
    risk = batch.risk_scores.float().mean().item() if batch.risk_scores is not None and batch.risk_scores.numel() else 0.0
    phase_risk = (
        batch.phase_risk_scores.float().mean().item()
        if batch.phase_risk_scores is not None and batch.phase_risk_scores.numel()
        else 0.0
    )
    return {
        "top1_acc": top1,
        "top10_acc": top10,
        "top50_acc": top50,
        "hard_ratio": hard,
        "curvature_risk": risk,
        "phase_grounding_risk": phase_risk,
        "samples": int(batch.state_x.shape[0]),
    }


def classify_state_dimensions(
    batch: SleepBatch,
    *,
    active_ratio: float,
    struct_ratio: float,
) -> dict[str, object]:
    scores = batch.state_x.abs().float().mean(dim=0)
    dim = int(scores.numel())
    active_k, struct_only_k, _ = state_partition_counts(dim, active_ratio, struct_ratio)
    struct_k = min(dim, active_k + struct_only_k)
    active_idx = torch.topk(scores, active_k).indices
    struct_idx = torch.topk(scores, struct_k).indices
    active_mask = torch.zeros(dim, dtype=torch.bool)
    struct_mask = torch.zeros(dim, dtype=torch.bool)
    active_mask[active_idx] = True
    struct_mask[struct_idx] = True
    struct_mask |= active_mask
    background_mask = ~struct_mask
    return {
        "active_mask": active_mask,
        "struct_mask": struct_mask,
        "background_mask": background_mask,
        "active_ratio": active_mask.float().mean().item(),
        "struct_ratio": (struct_mask & ~active_mask).float().mean().item(),
        "background_ratio": background_mask.float().mean().item(),
    }


def _weighted_covariance(x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    x = x.float()
    if weights is not None:
        scale = weights.float().clamp_min(1e-6).sqrt().unsqueeze(1)
        x = x * scale
    cov = x.T @ x
    return cov / max(int(x.shape[0]), 1)


def covariance_delta(batch: SleepBatch, *, emphasize_hard: float = 1.0) -> torch.Tensor:
    weights = batch_weights(batch, emphasize_hard)
    cov_state = _weighted_covariance(batch.state_x, weights=weights)
    cov_target = _weighted_covariance(batch.target_y, weights=weights)
    delta = cov_target - cov_state
    return 0.5 * (delta + delta.T)


def offdiag_density(mask: torch.Tensor) -> float:
    dim = int(mask.shape[0])
    if dim <= 1:
        return 0.0
    offdiag = mask.detach().bool().clone()
    offdiag.fill_diagonal_(False)
    return float(offdiag.sum().item()) / float(dim * (dim - 1))


def row_topk_mask(matrix: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    dim = int(matrix.shape[0])
    mask = torch.zeros_like(matrix, dtype=torch.bool)
    if dim <= 1 or float(keep_ratio) <= 0.0:
        return mask

    upper_i, upper_j = torch.triu_indices(dim, dim, offset=1, device=matrix.device)
    if upper_i.numel() == 0:
        return mask

    pair_scores = matrix.detach().abs()[upper_i, upper_j]
    keep_pairs = int(math.floor(float(keep_ratio) * float(upper_i.numel()) + 0.5))
    keep_pairs = max(1, min(int(upper_i.numel()), keep_pairs))
    top_idx = torch.topk(pair_scores, keep_pairs).indices
    keep_i = upper_i.index_select(0, top_idx)
    keep_j = upper_j.index_select(0, top_idx)
    mask[keep_i, keep_j] = True
    mask[keep_j, keep_i] = True
    return mask


def normalize_update(matrix: torch.Tensor) -> torch.Tensor:
    peak = float(matrix.abs().amax().item()) if matrix.numel() else 0.0
    if peak <= 1e-8:
        return torch.zeros_like(matrix)
    return matrix / peak


def smooth_weight_matrix(w: torch.Tensor, laplacian: torch.Tensor, eta: float) -> torch.Tensor:
    lap = laplacian.float()
    w = w.float()
    smoothed = w - float(eta) * (lap @ w + w @ lap) / 2.0
    return 0.5 * (smoothed + smoothed.T)


def apply_nrem_weight_update(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    smooth_eta: float = 0.02,
    plastic_lr: float = 0.01,
) -> dict[str, float]:
    partition = classify_state_dimensions(
        batch,
        active_ratio=eng.active_ratio,
        struct_ratio=eng.struct_ratio,
    )
    if eng.active_dim_mask is None or eng.struct_dim_mask is None:
        eng.apply_state_partition(partition["active_mask"], partition["struct_mask"])

    lap = eng.state_graph_laplacian().detach().cpu()
    delta = covariance_delta(batch, emphasize_hard=1.0)
    plastic_mask = row_topk_mask(delta, eng.active_ratio)
    update = normalize_update(delta * plastic_mask)
    base_w = eng.W.detach().cpu().float()
    candidate_w = smooth_weight_matrix(base_w, lap, smooth_eta)
    if update.abs().amax().item() > 0.0:
        candidate_w = candidate_w + float(plastic_lr) * update
    eng.apply_relax_matrix(candidate_w)
    return {
        "smooth_eta": float(smooth_eta),
        "plastic_lr": float(plastic_lr),
        "delta_norm": float(delta.norm().item()),
        "active_ratio": float(partition["active_ratio"]),
        "struct_ratio": float(partition["struct_ratio"]),
        "background_ratio": float(partition["background_ratio"]),
        "plastic_density": offdiag_density(plastic_mask),
        "w_offdiag_density_pct": eng.weight_density() * 100.0,
        "w_target_density_pct": eng.target_w_density * 100.0,
    }


def apply_rem_weight_update(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    rem_rank: int = 8,
    rem_lr: float = 0.005,
    noise_scale: float = 0.01,
) -> dict[str, float]:
    delta = covariance_delta(batch, emphasize_hard=2.5)
    selected_mask = row_topk_mask(delta, eng.active_ratio)
    residual_mask = ~selected_mask
    residual_mask.fill_diagonal_(False)
    residual = delta * residual_mask
    dim = int(residual.shape[0])
    if residual.abs().amax().item() <= 1e-8:
        return {
            "rem_rank": float(rem_rank),
            "rem_lr": float(rem_lr),
            "noise_scale": float(noise_scale),
            "residual_norm": 0.0,
            "residual_density": offdiag_density(residual_mask),
        }

    rank = max(1, min(int(rem_rank), dim))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    proj = torch.randn(dim, rank, generator=gen)
    remix = residual @ proj @ proj.T / float(rank)
    remix = 0.5 * (remix + remix.T)
    if noise_scale > 0.0:
        remix = remix + float(noise_scale) * residual.abs().mean().item() * torch.randn_like(remix, generator=gen)
    update = normalize_update(remix)
    candidate_w = eng.W.detach().cpu().float() + float(rem_lr) * update
    eng.apply_relax_matrix(candidate_w)
    return {
        "rem_rank": float(rank),
        "rem_lr": float(rem_lr),
        "noise_scale": float(noise_scale),
        "residual_norm": float(residual.norm().item()),
        "residual_density": offdiag_density(residual_mask),
        "w_offdiag_density_pct": eng.weight_density() * 100.0,
        "w_target_density_pct": eng.target_w_density * 100.0,
    }


def evaluate_guard_set(
    eng: CEEngine,
    prompts: list[str],
    ce_args,
    *,
    max_new_tokens: int,
    refresh_interval: int,
    refresh_steps: int,
    refresh_cb_topk: int,
    refresh_metric_rank: int,
    refresh_noise_scale: float,
    context_window: int = 64,
    seed_tokens: int = 8,
) -> dict[str, float]:
    if not prompts:
        return {
            "top1_acc": 0.0,
            "top10_acc": 0.0,
            "top50_acc": 0.0,
            "curvature_risk": 0.0,
            "phase_grounding_risk": 0.0,
            "samples": 0,
        }

    refresh_args = None
    if refresh_interval > 0:
        refresh_args = build_refresh_args(
            ce_args,
            steps=refresh_steps,
            cb_topk=refresh_cb_topk,
            metric_rank=refresh_metric_rank,
            noise_scale=refresh_noise_scale,
        )

    from tqdm import tqdm as _tqdm

    top1 = 0
    top10 = 0
    top50 = 0
    total = 0
    curvature_risk = 0.0
    phase_grounding_risk = 0.0

    for prompt in _tqdm(prompts, desc="    guard", unit="doc", ncols=80):
        full_ids = eng.tok.encode(prompt, return_tensors="pt").to(eng.device)
        if full_ids.shape[1] <= 1:
            continue
        cursor = min(max(int(seed_tokens), 1), full_ids.shape[1] - 1)
        while cursor < full_ids.shape[1]:
            ids = _context_slice(full_ids, cursor, context_window)
            ctx = eng.context_from_ids(ids, prompt=prompt)
            relax_result = eng.relax_context(ctx, ce_args)
            ce_hidden = eng.ce_hidden(relax_result["m_star"]).detach()
            phi_state = relax_result["phi_updated"].detach()
            init_layer = ctx.best_layer
            history_ids = ids[0].tolist()
            prev_hidden = None
            prev_prev_hidden = None
            context_anchor = ce_hidden.detach().clone()
            max_stop = min(full_ids.shape[1], cursor + max_new_tokens)

            for target_pos in range(cursor, max_stop):
                prev_id = int(ids[0, -1].item())
                target_id = int(full_ids[0, target_pos].item())
                logits, step_meta = eng.standalone_logits(
                    ce_hidden,
                    prev_id,
                    temperature=1.0,
                    history_ids=history_ids,
                    prev_hidden=prev_hidden,
                    prev_prev_hidden=prev_prev_hidden,
                    context_anchor=context_anchor,
                    return_meta=True,
                )
                top_ids = torch.topk(logits, min(50, logits.numel())).indices.tolist()
                top1 += int(top_ids[0] == target_id)
                top10 += int(target_id in top_ids[:10])
                top50 += int(target_id in top_ids)
                curvature_risk += float(step_meta["curvature_risk_score"])
                phase_grounding_risk += _mean_phase_grounding_risk(step_meta)
                total += 1

                step_hidden = ce_hidden.detach().clone()
                prev_prev_hidden = prev_hidden
                prev_hidden = step_hidden
                ids = torch.cat([ids, torch.tensor([[target_id]], device=eng.device)], dim=1)
                if ids.shape[1] > int(context_window):
                    ids = ids[:, -int(context_window) :]
                history_ids.append(target_id)
                if (
                    refresh_args is not None
                    and target_pos + 1 < max_stop
                    and (target_pos - cursor + 1) % refresh_interval == 0
                ):
                    refresh_ctx = eng.context_from_ids(
                        ids,
                        init_layer=init_layer,
                        phi=phi_state,
                        need_teacher=False,
                    )
                    refresh_result = eng.relax_context(refresh_ctx, refresh_args)
                    ce_hidden = eng.ce_hidden(refresh_result["m_star"]).detach()
                    phi_state = refresh_result["phi_updated"].detach()
                    init_layer = refresh_ctx.best_layer
            cursor = max_stop

    return {
        "top1_acc": top1 / max(total, 1),
        "top10_acc": top10 / max(total, 1),
        "top50_acc": top50 / max(total, 1),
        "curvature_risk": curvature_risk / max(total, 1),
        "phase_grounding_risk": phase_grounding_risk / max(total, 1),
        "samples": total,
    }


def should_accept_guard_update(
    before: dict[str, float],
    after: dict[str, float],
    *,
    min_top10_delta: float = 0.0,
    min_top50_delta: float = 0.0,
    max_top10_drop: float = 0.0,
    max_top50_drop: float = 0.0,
    max_phase_grounding_risk_increase: float | None = None,
) -> bool:
    top10_delta = float(after["top10_acc"]) - float(before["top10_acc"])
    top50_delta = float(after["top50_acc"]) - float(before["top50_acc"])
    phase_risk_delta = float(after.get("phase_grounding_risk", 0.0)) - float(
        before.get("phase_grounding_risk", 0.0)
    )
    if top10_delta < -float(max_top10_drop):
        return False
    if top50_delta < -float(max_top50_drop):
        return False
    if (
        max_phase_grounding_risk_increase is not None
        and phase_risk_delta > float(max_phase_grounding_risk_increase)
    ):
        return False
    return (
        top10_delta >= float(min_top10_delta)
        and top50_delta >= float(min_top50_delta)
    )


def run_guarded_microsleep_step(
    eng: CEEngine,
    buffer: PromptReplayBuffer,
    prompt: str,
    guard_prompts: list[str],
    ce_args,
    *,
    step_index: int,
    sleep_every: int,
    max_new_tokens: int,
    teacher_topk: int,
    ridge: float,
    rem_weight: float,
    rem_mix: float,
    token_head_max_vocab: int,
    token_head_scale: float,
    refresh_interval: int,
    refresh_steps: int,
    refresh_cb_topk: int,
    refresh_metric_rank: int,
    refresh_noise_scale: float,
    refresh_pq: bool,
    pq_subdim: int,
    pq_bits: int,
    pq_iters: int,
    pq_batch_size: int,
    pq_sample_size: int,
    guard_min_top10_delta: float = 0.0,
    guard_min_top50_delta: float = 0.0,
    guard_max_top10_drop: float = 0.0,
    guard_max_top50_drop: float = 0.0,
    guard_max_phase_grounding_risk_increase: float | None = None,
    context_window: int = 64,
    seed_tokens: int = 8,
) -> dict[str, object] | None:
    buffer.add(prompt)
    if sleep_every <= 0 or step_index % int(sleep_every) != 0:
        return None

    train_prompts = buffer.items()
    snapshot = eng.decoder_snapshot()
    before_guard = evaluate_guard_set(
        eng,
        guard_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    sleep_report = run_sleep_cycle(
        eng,
        train_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
        ridge=ridge,
        rem_weight=rem_weight,
        rem_mix=rem_mix,
        token_head_max_vocab=token_head_max_vocab,
        token_head_scale=token_head_scale,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        refresh_pq=refresh_pq,
        pq_subdim=pq_subdim,
        pq_bits=pq_bits,
        pq_iters=pq_iters,
        pq_batch_size=pq_batch_size,
        pq_sample_size=pq_sample_size,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    after_guard = evaluate_guard_set(
        eng,
        guard_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    accepted_update = should_accept_guard_update(
        before_guard,
        after_guard,
        min_top10_delta=guard_min_top10_delta,
        min_top50_delta=guard_min_top50_delta,
        max_top10_drop=guard_max_top10_drop,
        max_top50_drop=guard_max_top50_drop,
        max_phase_grounding_risk_increase=guard_max_phase_grounding_risk_increase,
    )
    if not accepted_update:
        eng.restore_decoder_snapshot(snapshot)

    effective_guard = after_guard if accepted_update else before_guard
    return {
        "step": step_index,
        "buffer_size": len(train_prompts),
        "train_prompts": train_prompts,
        "accepted": accepted_update,
        "sleep_report": sleep_report,
        "guard_before": before_guard,
        "guard_after": after_guard,
        "guard_effective": effective_guard,
        "guard_delta": {
            "top10_acc": after_guard["top10_acc"] - before_guard["top10_acc"],
            "top50_acc": after_guard["top50_acc"] - before_guard["top50_acc"],
        },
    }


def maybe_refresh_pq(
    eng: CEEngine,
    batch: SleepBatch,
    *,
    subdim: int,
    bits: int,
    iters: int,
    batch_size: int,
    sample_size: int,
):
    if eng.emb is None:
        return None

    emb = eng.emb.detach().cpu().float()
    freq = torch.bincount(batch.target_ids.cpu(), minlength=emb.shape[0]).float()
    hot = torch.nonzero(freq > 0, as_tuple=False).squeeze(1)
    if hot.numel() == 0:
        return None

    hot_emb = emb.index_select(0, hot)
    pool = torch.cat([emb, hot_emb, hot_emb], dim=0)
    pq = pq_build_codebook(
        pool,
        subdim=subdim,
        bits=bits,
        iters=iters,
        batch_size=batch_size,
        sample_size=min(sample_size, pool.shape[0]),
        seed=0,
    )
    centroids = pq["centroids"].cpu()

    codes = torch.empty((emb.shape[0], centroids.shape[0]), dtype=torch.uint8)
    for sub_idx in range(centroids.shape[0]):
        start = sub_idx * subdim
        stop = start + subdim
        dist = torch.cdist(emb[:, start:stop], centroids[sub_idx].float())
        codes[:, sub_idx] = dist.argmin(dim=1).to(torch.uint8)

    eng.pq_centroids = centroids.to(eng.device)
    eng.pq_codes = codes.to(eng.device)
    eng.data["pq_centroids"] = centroids
    eng.data["pq_codes"] = codes
    return {
        "pq_centroids_mb": centroids.numel() * centroids.element_size() / 1024 / 1024,
        "pq_codes_mb": codes.numel() * codes.element_size() / 1024 / 1024,
    }


def run_sleep_cycle(
    eng: CEEngine,
    prompts: list[str],
    ce_args,
    *,
    max_new_tokens: int,
    teacher_topk: int,
    ridge: float,
    rem_weight: float,
    rem_mix: float,
    token_head_max_vocab: int,
    token_head_scale: float,
    refresh_interval: int,
    refresh_steps: int,
    refresh_cb_topk: int,
    refresh_metric_rank: int,
    refresh_noise_scale: float,
    refresh_pq: bool,
    pq_subdim: int,
    pq_bits: int,
    pq_iters: int,
    pq_batch_size: int,
    pq_sample_size: int,
    guard_prompts: list[str] | None = None,
    guard_min_top10_delta: float = 0.0,
    guard_min_top50_delta: float = 0.0,
    guard_max_top10_drop: float = 0.0,
    guard_max_top50_drop: float = 0.0,
    guard_max_phase_grounding_risk_increase: float | None = None,
    context_window: int = 64,
    seed_tokens: int = 8,
    vocab_finetune_lr: float = 1e-3,
    vocab_finetune_steps: int = 64,
    vocab_finetune_batch_size: int = 256,
    vocab_finetune_soft_target_weight: float = 0.35,
) -> dict[str, object]:
    guard_snapshot = None
    guard_before = None
    if guard_prompts:
        guard_snapshot = eng.decoder_snapshot()
        guard_before = evaluate_guard_set(
            eng,
            guard_prompts,
            ce_args,
            max_new_tokens=max_new_tokens,
            refresh_interval=refresh_interval,
            refresh_steps=refresh_steps,
            refresh_cb_topk=refresh_cb_topk,
            refresh_metric_rank=refresh_metric_rank,
            refresh_noise_scale=refresh_noise_scale,
            context_window=context_window,
            seed_tokens=seed_tokens,
        )

    phase_profile = {
        "wake": float(eng.wake_ratio),
        "nrem": float(eng.nrem_ratio),
        "rem": float(eng.rem_ratio),
    }
    base_phase_samples = max(1, len(prompts) * max(1, int(max_new_tokens)))
    total_cycle_samples = max(len(phase_profile), base_phase_samples * len(phase_profile))
    phase_budget = allocate_phase_sample_counts(total_cycle_samples, phase_profile)
    sleep_total = max(phase_profile["nrem"] + phase_profile["rem"], 1e-8)
    phase_sleep_split = {
        "nrem": phase_profile["nrem"] / sleep_total,
        "rem": phase_profile["rem"] / sleep_total,
    }
    cycle_prompts = prioritize_documents_for_prompts(prompts, guard_prompts)
    wake = collect_sleep_batch(
        eng,
        cycle_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        sample_budget=phase_budget["wake"],
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    wake_stats = batch_stats(wake)
    nrem_weight_stats = apply_nrem_weight_update(
        eng,
        wake,
        smooth_eta=0.02 * phase_profile["nrem"],
        plastic_lr=0.01 * phase_profile["nrem"],
    )

    state_nrem, prev_nrem, bias_nrem = fit_decoder_from_batch(
        wake,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
    )
    eng.apply_decoder_refine(prev_nrem.cpu(), state_nrem.cpu(), query_bias=bias_nrem.cpu())
    vocab_nrem = finetune_vocab_head_from_batch(
        eng,
        wake,
        lr=vocab_finetune_lr,
        steps=vocab_finetune_steps,
        batch_size=vocab_finetune_batch_size,
        rem_weight=1.0,
        soft_target_weight=vocab_finetune_soft_target_weight,
    )
    token_nrem = fit_token_head_from_batch(
        wake,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        max_vocab=token_head_max_vocab,
        scale=token_head_scale,
    )
    if token_nrem is not None:
        eng.apply_token_head(
            token_nrem.token_ids,
            state_proj=token_nrem.state_proj,
            prev_proj=token_nrem.prev_proj,
            bias=token_nrem.bias,
            scale=token_nrem.scale,
        )

    nrem = collect_sleep_batch(
        eng,
        cycle_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        sample_budget=phase_budget["nrem"],
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    nrem_stats = batch_stats(nrem)

    rem_snapshot = eng.decoder_snapshot()
    rem_weight_stats = apply_rem_weight_update(
        eng,
        nrem,
        rem_lr=0.005 * phase_profile["rem"],
        noise_scale=max(float(refresh_noise_scale), 0.01) * phase_profile["rem"],
    )
    state_rem, prev_rem, bias_rem = fit_decoder_from_batch(
        nrem,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        rem_weight=rem_weight,
        rem_mix=rem_mix,
    )
    eng.apply_decoder_refine(prev_rem.cpu(), state_rem.cpu(), query_bias=bias_rem.cpu())
    vocab_rem = finetune_vocab_head_from_batch(
        eng,
        nrem,
        lr=vocab_finetune_lr,
        steps=vocab_finetune_steps,
        batch_size=vocab_finetune_batch_size,
        rem_weight=rem_weight,
        soft_target_weight=vocab_finetune_soft_target_weight,
    )
    token_rem = fit_token_head_from_batch(
        nrem,
        prev_scale=eng.decoder_prev_scale,
        ridge=ridge,
        rem_weight=rem_weight,
        max_vocab=token_head_max_vocab,
        scale=token_head_scale,
    )
    if token_rem is not None:
        eng.apply_token_head(
            token_rem.token_ids,
            state_proj=token_rem.state_proj,
            prev_proj=token_rem.prev_proj,
            bias=token_rem.bias,
            scale=token_rem.scale,
        )

    pq_stats = None
    if refresh_pq:
        pq_stats = maybe_refresh_pq(
            eng,
            nrem,
            subdim=pq_subdim,
            bits=pq_bits,
            iters=pq_iters,
            batch_size=pq_batch_size,
            sample_size=pq_sample_size,
        )

    rem = collect_sleep_batch(
        eng,
        cycle_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        teacher_topk=teacher_topk,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        sample_budget=phase_budget["rem"],
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    rem_stats = batch_stats(rem)
    rem_accepted = True
    if (
        rem_stats["top50_acc"] < nrem_stats["top50_acc"]
        and rem_stats["top1_acc"] < nrem_stats["top1_acc"]
    ):
        eng.restore_decoder_snapshot(rem_snapshot)
        rem = collect_sleep_batch(
            eng,
            cycle_prompts,
            ce_args,
            max_new_tokens=max_new_tokens,
            teacher_topk=teacher_topk,
            refresh_interval=refresh_interval,
            refresh_steps=refresh_steps,
            refresh_cb_topk=refresh_cb_topk,
            refresh_metric_rank=refresh_metric_rank,
            refresh_noise_scale=refresh_noise_scale,
            sample_budget=phase_budget["rem"],
            context_window=context_window,
            seed_tokens=seed_tokens,
        )
        rem_stats = batch_stats(rem)
        rem_accepted = False

    guard_after = None
    guard_effective = None
    guard_accepted = None
    if guard_prompts:
        guard_after = evaluate_guard_set(
            eng,
            guard_prompts,
            ce_args,
            max_new_tokens=max_new_tokens,
            refresh_interval=refresh_interval,
            refresh_steps=refresh_steps,
            refresh_cb_topk=refresh_cb_topk,
            refresh_metric_rank=refresh_metric_rank,
            refresh_noise_scale=refresh_noise_scale,
            context_window=context_window,
            seed_tokens=seed_tokens,
        )
        guard_accepted = should_accept_guard_update(
            guard_before,
            guard_after,
            min_top10_delta=guard_min_top10_delta,
            min_top50_delta=guard_min_top50_delta,
            max_top10_drop=guard_max_top10_drop,
            max_top50_drop=guard_max_top50_drop,
            max_phase_grounding_risk_increase=guard_max_phase_grounding_risk_increase,
        )
        if not guard_accepted and guard_snapshot is not None:
            eng.restore_decoder_snapshot(guard_snapshot)
            guard_effective = dict(guard_before)
        else:
            guard_effective = dict(guard_after)

    cycle_applied = guard_accepted is None or guard_accepted
    return {
        "phase_profile": phase_profile,
        "phase_sleep_split": phase_sleep_split,
        "phase_budget": {
            phase: {
                "samples": int(phase_budget[phase]),
                "ratio": float(phase_budget[phase]) / float(total_cycle_samples),
            }
            for phase in phase_profile
        },
        "phase_total_samples": int(total_cycle_samples),
        "wake": wake_stats,
        "nrem": nrem_stats,
        "rem": rem_stats,
        "nrem_weight": nrem_weight_stats if cycle_applied else None,
        "nrem_vocab_head": vocab_nrem if cycle_applied else None,
        "rem_weight": {**rem_weight_stats, "accepted": rem_accepted} if cycle_applied else None,
        "rem_vocab_head": vocab_rem if cycle_applied else None,
        "pq": pq_stats if cycle_applied else None,
        "token_head_vocab": (0 if token_rem is None else int(token_rem.token_ids.numel())) if cycle_applied else 0,
        "guard_before": guard_before,
        "guard_after": guard_after,
        "guard_effective": guard_effective,
        "guard_accepted": guard_accepted,
    }


def run_guarded_microsleep_session(
    eng: CEEngine,
    incoming_prompts: list[str],
    guard_prompts: list[str],
    ce_args,
    *,
    sleep_every: int,
    replay_capacity: int,
    max_new_tokens: int,
    teacher_topk: int,
    ridge: float,
    rem_weight: float,
    rem_mix: float,
    token_head_max_vocab: int,
    token_head_scale: float,
    refresh_interval: int,
    refresh_steps: int,
    refresh_cb_topk: int,
    refresh_metric_rank: int,
    refresh_noise_scale: float,
    refresh_pq: bool,
    pq_subdim: int,
    pq_bits: int,
    pq_iters: int,
    pq_batch_size: int,
    pq_sample_size: int,
    guard_min_top10_delta: float = 0.0,
    guard_min_top50_delta: float = 0.0,
    guard_max_top10_drop: float = 0.0,
    guard_max_top50_drop: float = 0.0,
    guard_max_phase_grounding_risk_increase: float | None = None,
    context_window: int = 64,
    seed_tokens: int = 8,
) -> dict[str, object]:
    buffer = PromptReplayBuffer(capacity=max(1, int(replay_capacity)))
    events: list[dict[str, object]] = []
    accepted = 0
    rejected = 0
    initial_guard = evaluate_guard_set(
        eng,
        guard_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )

    for idx, prompt in enumerate(incoming_prompts, start=1):
        event = run_guarded_microsleep_step(
            eng,
            buffer,
            prompt,
            guard_prompts,
            ce_args,
            step_index=idx,
            sleep_every=sleep_every,
            max_new_tokens=max_new_tokens,
            teacher_topk=teacher_topk,
            ridge=ridge,
            rem_weight=rem_weight,
            rem_mix=rem_mix,
            token_head_max_vocab=token_head_max_vocab,
            token_head_scale=token_head_scale,
            refresh_interval=refresh_interval,
            refresh_steps=refresh_steps,
            refresh_cb_topk=refresh_cb_topk,
            refresh_metric_rank=refresh_metric_rank,
            refresh_noise_scale=refresh_noise_scale,
            refresh_pq=refresh_pq,
            pq_subdim=pq_subdim,
            pq_bits=pq_bits,
            pq_iters=pq_iters,
            pq_batch_size=pq_batch_size,
            pq_sample_size=pq_sample_size,
            guard_min_top10_delta=guard_min_top10_delta,
            guard_min_top50_delta=guard_min_top50_delta,
            guard_max_top10_drop=guard_max_top10_drop,
            guard_max_top50_drop=guard_max_top50_drop,
            guard_max_phase_grounding_risk_increase=guard_max_phase_grounding_risk_increase,
            context_window=context_window,
            seed_tokens=seed_tokens,
        )
        if event is None:
            continue
        if event["accepted"]:
            accepted += 1
        else:
            rejected += 1
        events.append(event)

    final_guard = evaluate_guard_set(
        eng,
        guard_prompts,
        ce_args,
        max_new_tokens=max_new_tokens,
        refresh_interval=refresh_interval,
        refresh_steps=refresh_steps,
        refresh_cb_topk=refresh_cb_topk,
        refresh_metric_rank=refresh_metric_rank,
        refresh_noise_scale=refresh_noise_scale,
        context_window=context_window,
        seed_tokens=seed_tokens,
    )
    return {
        "initial_guard": initial_guard,
        "final_guard": final_guard,
        "accepted": accepted,
        "rejected": rejected,
        "events": events,
        "buffer_size": len(buffer.items()),
    }


def build_prompts(args) -> list[str]:
    prompts = list(args.prompts) if args.prompts else list(DEFAULT_PROMPTS)
    if args.prompt:
        prompts = [args.prompt] + prompts
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt and prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    return deduped


def main():
    ap = argparse.ArgumentParser(description="Sleep refinement for standalone CE artifacts")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--dataset-split", default=DEFAULT_CORPUS_SPLIT)
    ap.add_argument("--dataset-text-column", default=DEFAULT_CORPUS_TEXT_COLUMN)
    ap.add_argument("--doc-limit", type=int, default=256)
    ap.add_argument("--text-limit", type=int, default=1_000_000)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--online-prompts", nargs="*", default=None)
    ap.add_argument("--guard-prompts", nargs="*", default=None)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--sleep-every", type=int, default=4)
    ap.add_argument("--replay-capacity", type=int, default=16)
    ap.add_argument("--label-topk", "--teacher-topk", dest="teacher_topk", type=int, default=8)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--rem-weight", type=float, default=2.5)
    ap.add_argument("--rem-mix", type=float, default=0.35)
    ap.add_argument("--token-head-max-vocab", type=int, default=2048)
    ap.add_argument("--token-head-scale", type=float, default=1.0)
    ap.add_argument("--guard-min-top10-delta", type=float, default=0.0)
    ap.add_argument("--guard-min-top50-delta", type=float, default=0.0)
    ap.add_argument("--guard-max-top10-drop", type=float, default=0.0)
    ap.add_argument("--guard-max-top50-drop", type=float, default=0.0)
    ap.add_argument("--guard-max-phase-grounding-risk-increase", type=float, default=None)
    ap.add_argument("--refresh-interval", type=int, default=1)
    ap.add_argument("--refresh-steps", type=int, default=48)
    ap.add_argument("--refresh-cb-topk", type=int, default=128)
    ap.add_argument("--refresh-metric-rank", type=int, default=0)
    ap.add_argument("--refresh-noise-scale", type=float, default=0.0)
    ap.add_argument("--context-window", type=int, default=64)
    ap.add_argument("--seed-tokens", type=int, default=8)
    ap.add_argument("--refresh-pq", action="store_true")
    ap.add_argument("--pq-subdim", type=int, default=64)
    ap.add_argument("--pq-bits", type=int, default=8)
    ap.add_argument("--pq-iters", type=int, default=8)
    ap.add_argument("--pq-batch-size", type=int, default=4096)
    ap.add_argument("--pq-sample-size", type=int, default=16384)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--cb-topk", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--backend", default="torch", choices=["auto", "torch", "rust", "cuda"])
    ap.add_argument("--metric-rank", type=int, default=16)
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--lambda-phi", dest="lambda_phi", type=float, default=0.5)
    ap.add_argument("--lambda-var", dest="lambda_var", type=float, default=0.25)
    ap.add_argument("--noise-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    eng = CEEngine(args.engine, device=args.device, backend=args.backend)

    ce_args = argparse.Namespace(
        dt=args.dt,
        cb_weight=None,
        cb_topk=args.cb_topk,
        beta=args.beta,
        steps=args.steps,
        backend=args.backend,
        metric_rank=args.metric_rank,
        lambda0=args.lambda0,
        lambda_phi=args.lambda_phi,
        lambda_var=args.lambda_var,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    safe_print("=== CE Sleep Refinement ===")
    safe_print(f"  engine={args.engine}")
    safe_print(f"  model_source={eng.model_source}")

    if args.online_prompts:
        online_prompts = [prompt for prompt in args.online_prompts if prompt]
        guard_prompts = [prompt for prompt in (args.guard_prompts or list(DEFAULT_PROMPTS)) if prompt]
        safe_print(
            f"  online_prompts={len(online_prompts)}  guard_prompts={len(guard_prompts)}  "
            f"sleep_every={args.sleep_every}"
        )
        session = run_guarded_microsleep_session(
            eng,
            online_prompts,
            guard_prompts,
            ce_args,
            sleep_every=args.sleep_every,
            replay_capacity=args.replay_capacity,
            max_new_tokens=args.tokens,
            teacher_topk=args.teacher_topk,
            ridge=args.ridge,
            rem_weight=args.rem_weight,
            rem_mix=args.rem_mix,
            token_head_max_vocab=args.token_head_max_vocab,
            token_head_scale=args.token_head_scale,
            refresh_interval=args.refresh_interval,
            refresh_steps=args.refresh_steps,
            refresh_cb_topk=args.refresh_cb_topk,
            refresh_metric_rank=args.refresh_metric_rank,
            refresh_noise_scale=args.refresh_noise_scale,
            refresh_pq=args.refresh_pq,
            pq_subdim=args.pq_subdim,
            pq_bits=args.pq_bits,
            pq_iters=args.pq_iters,
            pq_batch_size=args.pq_batch_size,
            pq_sample_size=args.pq_sample_size,
            guard_min_top10_delta=args.guard_min_top10_delta,
            guard_min_top50_delta=args.guard_min_top50_delta,
            guard_max_top10_drop=args.guard_max_top10_drop,
            guard_max_top50_drop=args.guard_max_top50_drop,
            guard_max_phase_grounding_risk_increase=args.guard_max_phase_grounding_risk_increase,
            context_window=args.context_window,
            seed_tokens=args.seed_tokens,
        )
        safe_print(
            f"  microsleep: accepted={session['accepted']}  rejected={session['rejected']}  "
            f"final_guard_top50={session['final_guard']['top50_acc']:.3f}"
        )
        reports = session["events"]
        result_payload = {
            "engine": args.engine,
            "mode": "guarded_microsleep",
            "online_prompts": online_prompts,
            "guard_prompts": guard_prompts,
            "tokens": args.tokens,
            "session": session,
        }
    else:
        if args.data or args.dataset:
            prompts = load_corpus_documents(
                args.data,
                dataset_name=args.dataset,
                dataset_config=args.dataset_config,
                dataset_split=args.dataset_split,
                text_column=args.dataset_text_column,
                doc_limit=args.doc_limit,
                text_limit=args.text_limit,
            )
            safe_print(
                f"  corpus_docs={len(prompts)}  cycles={args.cycles}  tokens={args.tokens}  "
                f"context_window={args.context_window}"
            )
        else:
            prompts = build_prompts(args)
            safe_print(f"  prompts={len(prompts)}  cycles={args.cycles}  tokens={args.tokens}")
        reports = []
        for cycle in range(1, args.cycles + 1):
            report = run_sleep_cycle(
                eng,
                prompts,
                ce_args,
                max_new_tokens=args.tokens,
                teacher_topk=args.teacher_topk,
                ridge=args.ridge,
                rem_weight=args.rem_weight,
                rem_mix=args.rem_mix,
                token_head_max_vocab=args.token_head_max_vocab,
                token_head_scale=args.token_head_scale,
                refresh_interval=args.refresh_interval,
                refresh_steps=args.refresh_steps,
                refresh_cb_topk=args.refresh_cb_topk,
                refresh_metric_rank=args.refresh_metric_rank,
                refresh_noise_scale=args.refresh_noise_scale,
                refresh_pq=args.refresh_pq,
                pq_subdim=args.pq_subdim,
                pq_bits=args.pq_bits,
                pq_iters=args.pq_iters,
                pq_batch_size=args.pq_batch_size,
                pq_sample_size=args.pq_sample_size,
                context_window=args.context_window,
                seed_tokens=args.seed_tokens,
            )
            reports.append(report)
            safe_print(
                f"  cycle {cycle}: "
                f"wake top50={report['wake']['top50_acc']:.3f} -> "
                f"nrem {report['nrem']['top50_acc']:.3f} -> "
                f"rem {report['rem']['top50_acc']:.3f}  "
                f"token_vocab={report['token_head_vocab']}"
            )
        result_payload = {
            "engine": args.engine,
            "mode": "sleep_cycle",
            "prompts": prompts,
            "cycles": args.cycles,
            "reports": reports,
        }

    out_path = args.output or args.engine
    eng.save_runtime_artifact(out_path)
    result_path = os.path.join(os.path.dirname(out_path), "sleep_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)
    safe_print(f"  saved_engine={out_path}")
    safe_print(f"  saved_report={result_path}")


if __name__ == "__main__":
    main()
```
---
## File: `reality_stone/python/reality_stone/clarus/stdp.py`

```python
"""STDP learning with eligibility traces (17_AgentLoop.md F.14).

Implements spike-timing-dependent plasticity with 3-factor learning:
  e_ij[k+1] = r_e * e_ij[k] + (A+ * p_i * s_j - A- * s_i * q_j)
  dW_ij = lr * g[t] * e_ij
  W_{t+1} = Proj(W_t + dW_t)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

try:
    from .constants import (
        STDP_R_PLUS, STDP_R_MINUS, STDP_R_E,
        STDP_A_PLUS, STDP_A_MINUS, STDP_SPIKE_THRESHOLD,
        STDP_LR, STDP_ALPHA_G, ACTIVE_RATIO,
    )
except ImportError:
    from reality_stone.clarus.constants import (
        STDP_R_PLUS, STDP_R_MINUS, STDP_R_E,
        STDP_A_PLUS, STDP_A_MINUS, STDP_SPIKE_THRESHOLD,
        STDP_LR, STDP_ALPHA_G, ACTIVE_RATIO,
    )


@dataclass
class STDPConfig:
    dim: int
    r_plus: float = STDP_R_PLUS
    r_minus: float = STDP_R_MINUS
    r_e: float = STDP_R_E
    a_plus: float = STDP_A_PLUS
    a_minus: float = STDP_A_MINUS
    spike_threshold: float = STDP_SPIKE_THRESHOLD
    lr: float = STDP_LR
    alpha_g: float = STDP_ALPHA_G


class EligibilityTracker:
    """Tracks pre/post synaptic traces and eligibility matrix."""

    def __init__(self, config: STDPConfig, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.pre_trace = torch.zeros(config.dim, device=self.device)
        self.post_trace = torch.zeros(config.dim, device=self.device)
        self.eligibility = torch.zeros(config.dim, config.dim, device=self.device)

    def update(self, activation: torch.Tensor) -> None:
        """Update traces and eligibility from current activation (one R iteration)."""
        spike = (activation.abs() > self.config.spike_threshold).float()
        self.pre_trace = self.config.r_plus * self.pre_trace + spike
        self.post_trace = self.config.r_minus * self.post_trace + spike
        ltp = self.config.a_plus * torch.outer(self.pre_trace, spike)
        ltd = self.config.a_minus * torch.outer(spike, self.post_trace)
        self.eligibility = self.config.r_e * self.eligibility + (ltp - ltd)

    def reset(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.eligibility.zero_()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "pre_trace": self.pre_trace.detach().cpu(),
            "post_trace": self.post_trace.detach().cpu(),
            "eligibility": self.eligibility.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.pre_trace = state["pre_trace"].to(self.device).float()
        self.post_trace = state["post_trace"].to(self.device).float()
        self.eligibility = state["eligibility"].to(self.device).float()


def compute_learning_gate(
    critic_score: float,
    prev_critic_score: float,
    active_ratio: float,
    target_active: float = ACTIVE_RATIO,
    target_struct: float = 0.2623,
    target_bg: float = 0.6891,
    struct_ratio: float = 0.26,
    bg_ratio: float = 0.69,
    alpha_g: float = STDP_ALPHA_G,
    dt: float = 1.0,
) -> float:
    """g[t] = alpha_g * d(c_bar)/dt + (1-alpha_g) * bootstrap_deviation (F.14.2)."""
    critic_derivative = (critic_score - prev_critic_score) / max(dt, 1e-8)
    bootstrap_dev = (
        (active_ratio - target_active) ** 2
        + (struct_ratio - target_struct) ** 2
        + (bg_ratio - target_bg) ** 2
    )
    return alpha_g * critic_derivative + (1.0 - alpha_g) * bootstrap_dev


def structural_projection(
    weight: torch.Tensor,
    density: float = ACTIVE_RATIO,
    theta_on: float = 0.01,
    theta_off: float = 0.005,
) -> torch.Tensor:
    """Proj(W) = TopK(RowNorm(Hyst(W; theta_on, theta_off)), k) (F.14.3)."""
    mask = weight.abs() > theta_on
    below = weight.abs() < theta_off
    hyst = weight * mask.float()
    hyst[below] = 0.0

    row_norms = hyst.norm(dim=1, keepdim=True).clamp(min=1e-8)
    hyst = hyst / row_norms

    k = max(1, int(math.ceil(density * weight.shape[0])))
    for i in range(weight.shape[0]):
        row = hyst[i]
        if (row != 0).sum() > k:
            _, topk_idx = torch.topk(row.abs(), k)
            new_row = torch.zeros_like(row)
            new_row[topk_idx] = row[topk_idx]
            hyst[i] = new_row

    return hyst


def apply_stdp_update(
    weight: torch.Tensor,
    tracker: EligibilityTracker,
    gate: float,
    lr: float = STDP_LR,
    density: float = ACTIVE_RATIO,
) -> torch.Tensor:
    """Full STDP weight update: W_{t+1} = Proj(W + lr * g * e)."""
    dw = lr * gate * tracker.eligibility
    new_w = weight + dw
    return structural_projection(new_w, density=density)
```
---
## File: `reality_stone/python/reality_stone/clarus/utils.py`

```python
"""Shared utilities used across the clarus package.

Consolidates safe_print, normalize_vector, and resolve_device
which were previously duplicated in engine.py, runtime.py, and sleep.py.
"""

from __future__ import annotations

import sys

import torch

from .constants import NORM_EPS


def safe_print(text: object) -> None:
    """Print with UTF-8 fallback for Windows consoles."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        data = (str(text) + "\n").encode("utf-8", errors="replace")
        sys.stdout.buffer.write(data)
        sys.stdout.flush()


def normalize_vector(x: torch.Tensor) -> torch.Tensor:
    """Detach, cast to float, and L2-normalize. Returns zeros on degenerate input."""
    x = x.detach().float()
    norm = x.norm()
    if not torch.isfinite(norm) or norm.item() < NORM_EPS:
        return torch.zeros_like(x)
    return x / norm


def resolve_device(name: str) -> torch.device:
    """Resolve a device name to a torch.device with validation."""
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
```
---
## File: `reality_stone/python/reality_stone/core/__init__.py`

```python
from .mobius import *

__all__ = [
    'MobiusAdd', 
    'MobiusScalarMul'
]
```
---
## File: `reality_stone/python/reality_stone/core/mobius.py`

```python
from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd import Function

from .. import _has_cuda, _rust
from .._fallback import dynamic_curvature, mobius_add_torch, mobius_scalar_torch

_HAS_NATIVE = _rust is not None and not bool(getattr(_rust, "IS_FALLBACK", False))


class MobiusAdd(Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        y: Tensor,
        c: float = None,
        kappas: Tensor = None,
        layer_idx: int = None,
        c_min: float = -2.0,
        c_max: float = -0.1,
    ) -> Tensor:
        ctx.use_dynamic = kappas is not None and layer_idx is not None
        ctx.c = c if c is not None else 1.0
        ctx.layer_idx = layer_idx
        ctx.c_min = c_min
        ctx.c_max = c_max

        if ctx.use_dynamic:
            ctx.save_for_backward(x, y, kappas)
            if kappas.dim() == 0:
                kappas_list = [float(kappas.item())]
            else:
                kappas_list = [float(v) for v in kappas.detach().cpu().tolist()]
            if _HAS_NATIVE:
                output_np, c_val = _rust.mobius_add_layerwise_cpu(
                    x.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    kappas_list,
                    int(layer_idx),
                    float(c_min),
                    float(c_max),
                )
                ctx.c_val = float(c_val)
                return torch.from_numpy(output_np).to(device=x.device, dtype=x.dtype)
            ctx.c_val = dynamic_curvature(kappas_list[int(layer_idx)], c_min, c_max)
            return mobius_add_torch(x, y, ctx.c_val)

        ctx.save_for_backward(x, y)
        if x.is_cuda and _has_cuda and _HAS_NATIVE:
            output = torch.empty_like(x)
            _rust.mobius_add_cuda(
                x.data_ptr(),
                y.data_ptr(),
                output.data_ptr(),
                x.shape[0],
                x.shape[1],
                float(ctx.c),
            )
            return output
        if _HAS_NATIVE:
            out = _rust.mobius_add_cpu(
                x.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                float(ctx.c),
            )
            return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
        return mobius_add_torch(x, y, float(ctx.c))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.use_dynamic:
            x, y, kappas = ctx.saved_tensors
            if kappas.dim() == 0:
                kappas_list = [float(kappas.item())]
            else:
                kappas_list = [float(v) for v in kappas.detach().cpu().tolist()]
            if _HAS_NATIVE:
                gx_np, gy_np, gk = _rust.mobius_add_layerwise_backward_cpu(
                    grad_output.detach().cpu().numpy(),
                    x.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    kappas_list,
                    int(ctx.layer_idx),
                    float(ctx.c_min),
                    float(ctx.c_max),
                )
                gx = torch.from_numpy(gx_np).to(device=grad_output.device, dtype=grad_output.dtype)
                gy = torch.from_numpy(gy_np).to(device=grad_output.device, dtype=grad_output.dtype)
            else:
                gx = grad_output.clone()
                gy = grad_output.clone()
                gk = 0.0
            gk_tensor = torch.zeros_like(kappas)
            if kappas.dim() == 0:
                gk_tensor = torch.as_tensor(float(gk), device=kappas.device, dtype=kappas.dtype)
            else:
                gk_tensor[int(ctx.layer_idx)] = float(gk)
            return gx, gy, None, gk_tensor, None, None, None

        x, y = ctx.saved_tensors
        if _HAS_NATIVE and hasattr(_rust, "poincare"):
            gx_np, gy_np = _rust.poincare.mobius_add_vjp_cpu(
                grad_output.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                float(ctx.c),
            )
            gx = torch.from_numpy(gx_np).to(device=grad_output.device, dtype=grad_output.dtype)
            gy = torch.from_numpy(gy_np).to(device=grad_output.device, dtype=grad_output.dtype)
        else:
            gx = grad_output.clone()
            gy = grad_output.clone()
        return gx, gy, None, None, None, None, None


class MobiusScalarMul(Function):
    @staticmethod
    def forward(ctx, x: Tensor, r: float, c: float) -> Tensor:
        ctx.r = float(r)
        ctx.c = float(c)
        ctx.save_for_backward(x)
        if x.is_cuda and _has_cuda and _HAS_NATIVE:
            output = torch.empty_like(x)
            _rust.mobius_scalar_cuda(
                x.data_ptr(),
                output.data_ptr(),
                x.shape[0],
                x.shape[1],
                float(r),
                float(c),
            )
            return output
        if _HAS_NATIVE:
            out = _rust.mobius_scalar_cpu(x.detach().cpu().numpy(), float(r), float(c))
            return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
        return mobius_scalar_torch(x, float(r), float(c))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (x,) = ctx.saved_tensors
        if _HAS_NATIVE and hasattr(_rust, "poincare"):
            gx_np = _rust.poincare.mobius_scalar_vjp_cpu(
                grad_output.detach().cpu().numpy(),
                x.detach().cpu().numpy(),
                float(ctx.c),
                float(ctx.r),
            )
            gx = torch.from_numpy(gx_np).to(device=grad_output.device, dtype=grad_output.dtype)
        else:
            gx = grad_output * float(ctx.r)
        return gx, None, None
```
---
## File: `reality_stone/python/reality_stone/data.py`

```python
import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional, Union
import os

try:
    from reality_stone.utils.pre_segmenter import PreSegmenter
except ImportError:
    PreSegmenter = None


class SentenceTopicDataset(Dataset):
    """
    Dataset for Hierarchical Sentence-Topic LLM.
    Loads data from a JSON/JSONL file containing paragraphs or structured graph data.
    """
    def __init__(
        self, 
        data_path: str, 
        max_paragraphs: int = 10000,
        max_length: int = 128,
        k_neighbors: int = 3
    ):
        self.data_path = data_path
        self.max_paragraphs = max_paragraphs
        self.max_length = max_length
        self.k_neighbors = k_neighbors
        self.samples: List[Dict[str, Any]] = []
        
        self.segmenter = None
        if PreSegmenter is not None:
            self.segmenter = PreSegmenter(max_length=max_length, k_neighbors=k_neighbors)
        
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.data_path):
            print(f"Warning: Data file {self.data_path} not found.")
            return

        count = 0
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= self.max_paragraphs:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # If data is just a string (paragraph), wrap it
                    if isinstance(data, str):
                        self.samples.append({"paragraph": data})
                    # If data is dict, assume it has "paragraph" or pre-processed fields
                    elif isinstance(data, dict):
                        self.samples.append(data)
                    count += 1
                except json.JSONDecodeError:
                    # Fallback: treat line as raw text paragraph
                    if len(line) > 0:
                        self.samples.append({"paragraph": line})
                        count += 1
        
        print(f"[SentenceTopicDataset] Loaded {len(self.samples)} samples from {self.data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Case 1: Pre-processed data with graph structure (from generate_graph_data.py)
        if "tokens" in sample and "topo_idx" in sample:
            # Convert lists to tensors if they aren't already
            tokens = torch.tensor(sample["tokens"], dtype=torch.long)
            topo_idx = torch.tensor(sample["topo_idx"], dtype=torch.long)
            
            # Optional labels
            topic_labels = None
            if "topic_labels" in sample:
                topic_labels = torch.tensor(sample["topic_labels"], dtype=torch.long)

            # Replacement mask (default to all 1s if not present)
            if "replacement_mask" in sample:
                replacement_mask = torch.tensor(sample["replacement_mask"], dtype=torch.long)
            else:
                replacement_mask = torch.ones_like(tokens)
                
            return {
                "tokens": tokens,
                "topo_idx": topo_idx,
                "replacement_mask": replacement_mask,
                "sentences": sample.get("sentences", []),
                "paragraph": sample.get("paragraph", ""),
                "topic_labels": topic_labels
            }
            
        # Case 2: Raw paragraph text -> Segment on the fly
        elif "paragraph" in sample and self.segmenter is not None:
            text = sample["paragraph"]
            processed = self.segmenter(text)
            
            # PreSegmenter returns dict with tensors on CPU
            # Add missing keys if needed
            processed["paragraph"] = text
            
            # Check for explicit topic labels in raw data
            if "topic_label" in sample:
                # This needs a mapping from label str to int ID. 
                # For now, we skip or implement a simple hash if needed.
                pass
                
            return processed
            
        else:
            # Fallback / Error
            return {
                "tokens": torch.zeros(1, self.max_length, dtype=torch.long),
                "topo_idx": torch.zeros(1, self.k_neighbors, dtype=torch.long),
                "sentences": [],
                "paragraph": ""
            }

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.
    Pads tokens and topo_idx to the maximum number of sentences in the batch.
    """
    # 1. Find max T (number of sentences)
    max_t = 0
    max_l = 0
    for item in batch:
        if "tokens" in item:
            t, l = item["tokens"].shape
            max_t = max(max_t, t)
            max_l = max(max_l, l)
            
    # 2. Prepare batched tensors
    batch_size = len(batch)
    # tokens: [B, MaxT, L]
    batched_tokens = torch.zeros(batch_size, max_t, max_l, dtype=torch.long)
    
    # topo_idx: [B, MaxT, K]
    # We need to know K from the first non-empty sample
    k = 3
    for item in batch:
        if "topo_idx" in item:
            k = item["topo_idx"].shape[-1]
            break
            
    batched_topo_idx = torch.zeros(batch_size, max_t, k, dtype=torch.long)
    batched_replacement_mask = torch.zeros(batch_size, max_t, max_l, dtype=torch.long)
    
    # topic_labels: [B, MaxT] (Optional)
    has_labels = any("topic_labels" in item and item["topic_labels"] is not None for item in batch)
    batched_topic_labels = torch.full((batch_size, max_t), -1, dtype=torch.long) if has_labels else None
    
    batched_trees = []
    batched_paragraphs = []
    
    for i, item in enumerate(batch):
        tokens = item.get("tokens") # [T, L]
        topo = item.get("topo_idx") # [T, K]
        labels = item.get("topic_labels") # [T] or None
        
        if tokens is not None:
            t, l = tokens.shape
            batched_tokens[i, :t, :l] = tokens
            
        mask = item.get("replacement_mask")
        if mask is not None:
            t_m, l_m = mask.shape
            batched_replacement_mask[i, :t_m, :l_m] = mask
            
        if topo is not None:
            t_topo, k_topo = topo.shape
            # Adjust topology indices if they point to padding? 
            # Actually, topo indices are relative to the sentence list (0..T-1).
            # We just copy them. When using, we must mask out padding sentences.
            batched_topo_idx[i, :t_topo, :k_topo] = topo
            
            # If we pad sentences, we might need to adjust topo to point to self for padded areas?
            # For now, 0-padding is safe if we mask properly in the model.
            
        if has_labels and labels is not None:
            t_labels = labels.shape[0]
            batched_topic_labels[i, :t_labels] = labels
            
        if "tree" in item:
            batched_trees.append(item["tree"])
            
        if "paragraph" in item:
            batched_paragraphs.append(item["paragraph"])
            
    return {
        "tokens": batched_tokens,
        "topo_idx": batched_topo_idx,
        "replacement_mask": batched_replacement_mask,
        "topic_labels": batched_topic_labels,
        "tree": batched_trees if batched_trees else None,
        "paragraphs": batched_paragraphs
    }


class SimpleTextDataset(Dataset):
    """
    Simple dataset for list of text strings.
    Used in compression and finetuning scripts.
    """
    def __init__(self, texts: List[str], tokenizer, max_len: int = 128):
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            max_length=max_len,
            padding="max_length", 
            return_tensors="pt"
        )
    
    def __len__(self):
        return self.encodings.input_ids.size(0)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings.input_ids[idx],
            "attention_mask": self.encodings.attention_mask[idx],
            # For CausalLM training, labels are usually input_ids
            "labels": self.encodings.input_ids[idx] 
        }


class TextFileDataset(Dataset):
    """
    Simple dataset reading lines from a text file.
    Used in RS-ULF finetuning.
    """
    def __init__(self, path: str, tokenizer, max_len: int = 128):
        self.path = path
        self.tokenizer = tokenizer
        self.max_len = max_len
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        else:
            print(f"Warning: Data file {path} not found.")
            self.lines = []

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.lines[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        return {"input_ids": input_ids, "labels": input_ids.clone()}
```
---
## File: `reality_stone/python/reality_stone/layers/__init__.py`

```python
from .poincare import PoincareBallLayer, poincare_add, poincare_scalar_mul, poincare_distance, poincare_to_lorentz, poincare_to_klein, project_to_ball, HyperbolicLinear, GeodesicLinear, EquivalentHyperbolicLinear
from .lorentz import LorentzLayer, lorentz_add, lorentz_scalar_mul, lorentz_distance, lorentz_inner, lorentz_to_poincare, lorentz_to_klein, euclidean_to_lorentz
from .klein import KleinLayer, klein_add, klein_scalar_mul, klein_distance, klein_to_poincare, klein_to_lorentz
from .spline import SplineLinear
from .metric_attention import MetricAttention, SPDMetric, normalize, build_topo_topk, masked_gather, aggregate, get_default_topk_cfg
from .diffusion import RiemannianDiffusionStep, RiemannianDiffusionModule
from .rsulf_cuda import RSULFLayerCUDA, RSULFWrapperCUDA, RSULFLMHeadCUDA

__all__ = [
    "PoincareBallLayer",
    "poincare_add",
    "poincare_scalar_mul",
    "poincare_distance",
    "poincare_to_lorentz",
    "poincare_to_klein",
    "project_to_ball",
    "HyperbolicLinear",
    "GeodesicLinear",
    "EquivalentHyperbolicLinear",
    "LorentzLayer",
    "lorentz_add",
    "lorentz_scalar_mul",
    "lorentz_distance",
    "lorentz_inner",
    "lorentz_to_poincare",
    "lorentz_to_klein",
    "euclidean_to_lorentz",
    "KleinLayer",
    "klein_add",
    "klein_scalar_mul",
    "klein_distance",
    "klein_to_poincare",
    "klein_to_lorentz",
    "SplineLinear",
    "MetricAttention",
    "SPDMetric",
    "normalize",
    "build_topo_topk",
    "masked_gather",
    "aggregate",
    "get_default_topk_cfg",
    "RiemannianDiffusionStep",
    "RiemannianDiffusionModule",
    "RSULFLayerCUDA",
    "RSULFWrapperCUDA",
    "RSULFLMHeadCUDA",
]
```
---
## File: `reality_stone/python/reality_stone/layers/diffusion.py`

```python
import torch
import torch.nn as nn
import reality_stone as rs


class RiemannianDiffusionStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, flow, diffusion_engine, alpha, dt):
        h = h.contiguous()
        flow = flow.contiguous()

        h_next = torch.empty_like(h)
        batch_size, dim = h.shape

        if h.is_cuda and getattr(rs, "_has_cuda", False):
            diffusion_engine.step_cuda(
                h.data_ptr(),
                flow.data_ptr(),
                h_next.data_ptr(),
                batch_size,
                dim,
            )
        else:
            h_np = h.detach().cpu().numpy().astype("float32")
            flow_np = flow.detach().cpu().numpy().astype("float32")
            h_next_np = diffusion_engine.step_cpu(h_np, flow_np)
            h_next = torch.from_numpy(h_next_np).to(h.device)

        ctx.alpha = float(alpha)
        ctx.dt = float(dt)
        return h_next

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.alpha
        dt = ctx.dt
        a = 1.0 - (1.0 - alpha) * dt
        b = (1.0 - alpha) * dt

        grad_h = grad_output * a
        grad_flow = grad_output * b

        return grad_h, grad_flow, None, None, None


class RiemannianDiffusionModule(nn.Module):
    def __init__(self, dim, alpha=0.9, dt=0.1, num_steps=5):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.dt = dt
        self.num_steps = num_steps
        
        if rs.PyRiemannianDiffusion is not None:
            self.engine = rs.PyRiemannianDiffusion(dim, alpha, dt)
        else:
            self.engine = None
        
        self.flow_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h):
        if self.engine is None:
            return h
            
        for _ in range(self.num_steps):
            flow = self.flow_net(h)
            h = RiemannianDiffusionStep.apply(h, flow, self.engine, self.alpha, self.dt)
        return h
```
---
## File: `reality_stone/python/reality_stone/layers/klein.py`

```python
import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda
import math
from .poincare import poincare_to_klein
from .._fallback import klein_distance_torch, mobius_add_torch, mobius_scalar_torch

class KleinLayer(Function):
    @staticmethod
    def forward(
        ctx,
        u: Tensor,
        v: Tensor,
        c: float = None,
        t: float = 0.5,
        kappas: Tensor = None,
        layer_idx: int = None,
        c_min: float = 0.1,
        c_max: float = 5.0,
    ) -> Tensor:
        ctx.t = t
        if kappas is not None and layer_idx is not None:
            ctx.use_dynamic = True
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(u, v, kappas)
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
            if hasattr(_rust, "klein_layer_layerwise_cpu"):
                out_np, c_val = _rust.klein_layer_layerwise_cpu(
                    u.cpu().numpy(), v.cpu().numpy(), kappa_val, layer_idx, c_min, c_max, t
                )
                ctx.c_val = c_val
                return torch.from_numpy(out_np).to(u.device)
            sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val)))
            c_val = c_min + (c_max - c_min) * sig.item()
            ctx.c_val = c_val
            out_np = _rust.klein_layer_forward(u.cpu().numpy(), v.cpu().numpy(), float(c_val), t)
            return torch.from_numpy(out_np).to(u.device)
        ctx.use_dynamic = False
        ctx.c = c if c is not None else 1.0
        ctx.save_for_backward(u, v)
        if _rust is None:
            return (1.0 - float(t)) * u + float(t) * v
        out_np = _rust.klein_layer_forward(u.cpu().numpy(), v.cpu().numpy(), float(ctx.c), t)
        return torch.from_numpy(out_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        t = ctx.t
        if getattr(ctx, "use_dynamic", False):
            u, v, kappas = ctx.saved_tensors
            c_val = getattr(ctx, "c_val", None)
            if c_val is None:
                layer_idx = ctx.layer_idx
                c_min = ctx.c_min
                c_max = ctx.c_max
                if kappas.dim() == 0:
                    kappa_val = kappas.item()
                else:
                    kappa_val = kappas[layer_idx].item()
                sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val)))
                c_val = c_min + (c_max - c_min) * sig.item()
                ctx.c_val = c_val
            c = float(c_val)
        else:
            u, v = ctx.saved_tensors
            c = float(ctx.c)
        grad_u = grad_v = None
        if _rust is None:
            return grad_output * (1.0 - float(t)), grad_output * float(t), None, None, None, None, None, None
        
        if grad_output.is_cuda and _has_cuda:
            grad_u = torch.empty_like(u)
            grad_v = torch.empty_like(v)
            _rust.klein_ball_layer_backward_cuda(
                grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                grad_u.data_ptr(), grad_v.data_ptr(),
                float(c), t, u.shape[0], u.shape[1]
            )
        else:
            grad_u_np, grad_v_np = _rust.klein_ball_layer_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), float(c), t
            )
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
        if getattr(ctx, "use_dynamic", False):
            if kappas.dim() == 0:
                grad_kappas = torch.zeros_like(kappas)
            else:
                grad_kappas = torch.zeros_like(kappas)
            return grad_u, grad_v, None, None, grad_kappas, None, None, None
        return grad_u, grad_v, None, None, None, None, None, None

def klein_add(u: Tensor, v: Tensor, c: float) -> Tensor:
    if _rust is None:
        return mobius_add_torch(u, v, c)
    result_np = _rust.klein_add(u.cpu().numpy(), v.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(u.device)

def klein_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    if _rust is None:
        return mobius_scalar_torch(x, r, c)
    result_np = _rust.klein_scalar(x.cpu().numpy(), r, c)
    return torch.from_numpy(result_np).to(x.device)

def klein_distance(x: Tensor, y: Tensor, c: float) -> Tensor:
    if _rust is None:
        c_val = float(c.detach().cpu().item()) if isinstance(c, Tensor) else float(c)
        return klein_distance_torch(x, y, c_val)
    if isinstance(c, Tensor):
        eps = 1e-7
        c_t = c
        x2 = (x * x).sum(dim=-1)
        y2 = (y * y).sum(dim=-1)
        xy = (x * y).sum(dim=-1)
        den = (1.0 - c_t * x2) * (1.0 - c_t * y2)
        den = den.clamp_min(eps)
        arg = (1.0 - c_t * xy) / torch.sqrt(den)
        arg = arg.clamp_min(1.0 + eps)
        return torch.acosh(arg) / torch.sqrt(c_t)
    c_f = float(c)
    if x.is_cuda and _has_cuda:
        output = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
        _rust.klein_distance_cuda(output.data_ptr(), x.data_ptr(), y.data_ptr(), c_f, x.shape[0], x.shape[1])
        return output
    result_np = _rust.klein_distance(x.cpu().numpy(), y.cpu().numpy(), c_f)
    return torch.from_numpy(result_np).to(x.device)

def klein_to_poincare(x: Tensor, c: float) -> Tensor:
    if _rust is None:
        den = 1.0 + torch.sqrt((1.0 - float(c) * (x * x).sum(dim=-1, keepdim=True)).clamp_min(0.0))
        return x / den.clamp_min(1e-7)
    result_np = _rust.klein_to_poincare(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device)

def klein_to_lorentz(x: Tensor, c: float) -> Tensor:
    if _rust is None:
        x2 = (x * x).sum(dim=-1, keepdim=True)
        gamma = torch.rsqrt((1.0 - float(c) * x2).clamp_min(1e-7))
        return torch.cat([gamma, gamma * x], dim=-1)
    result_np = _rust.klein_to_lorentz(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device) 

class KleinFromPoincare(Function):
    @staticmethod
    def forward(ctx, x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
        if kappas is not None:
            ctx.use_dynamic = True
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(x, kappas)
            
            output_np, c_val = _rust.from_poincare_dynamic_cpu(
                x.cpu().numpy(), kappas.item(), c_min, c_max
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(x.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            # Delegate to poincare_to_klein for non-dynamic path
            output = poincare_to_klein(x, ctx.c)
            ctx.save_for_backward(x)
            return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.use_dynamic:
            x, kappas = ctx.saved_tensors
            grad_x_np, grad_kappa_val = _rust.from_poincare_dynamic_backward_cpu(
                grad_output.cpu().numpy(), x.cpu().numpy(), kappas.item(), ctx.c_min, ctx.c_max
            )
            grad_x = torch.from_numpy(grad_x_np).to(grad_output.device)
            grad_kappas = torch.tensor(grad_kappa_val, device=kappas.device)
            return grad_x, None, grad_kappas, None, None
        else:
            # VJP for non-dynamic version is not implemented yet.
            x, = ctx.saved_tensors
            grad_x = torch.zeros_like(x)
            return grad_x, None, None, None, None

def from_poincare(x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
    return KleinFromPoincare.apply(x, c, kappas, c_min, c_max) 


def project_to_klein(x: Tensor, c: float | Tensor, epsilon: float = 1e-5) -> Tensor:
    if isinstance(c, Tensor):
        radius = torch.rsqrt(c).clamp(min=epsilon)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = radius - float(epsilon)
        scale = torch.where(norm > max_norm, max_norm / norm.clamp_min(epsilon), torch.ones_like(norm))
        return x * scale
    radius = (1.0 / math.sqrt(c)) if c > 0 else 1.0
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    max_norm = radius - epsilon
    scale = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * scale
```
---
## File: `reality_stone/python/reality_stone/layers/lorentz.py`

```python
import torch
from torch import Tensor
from torch.autograd import Function
from .. import _rust, _has_cuda
from .poincare import poincare_to_lorentz
from .._fallback import lorentz_distance_torch, lorentz_inner_torch

class LorentzDistance(Function):
    """
    로렌츠 거리(Lorentz Distance)를 계산하는 Autograd Function입니다.
    """
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float) -> Tensor:
        ctx.c = c
        ctx.save_for_backward(u, v)
        
        if _rust is None:
            return lorentz_distance_torch(u, v, c)
        if u.is_cuda and _has_cuda:
            # CUDA 구현 사용
            output = torch.empty(u.shape[0], dtype=u.dtype, device=u.device)
            _rust.lorentz_distance_cuda(
                output.data_ptr(), u.data_ptr(), v.data_ptr(),
                c, u.shape[0], u.shape[1]
            )
            return output
        else:
            # CPU 구현 사용
            result_np = _rust.lorentz_distance(u.detach().cpu().numpy(), v.detach().cpu().numpy(), c)
            return torch.from_numpy(result_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        u, v = ctx.saved_tensors
        c = ctx.c
        
        # 로렌츠 거리의 해석적 역전파 구현
        # d(dist) = acosh(z) / sqrt(c) where z = c * <u,v>_L
        
        # 역전파를 위한 재계산
        # <u, v>_L = u0v0 - u.v
        inner = u[..., 0] * v[..., 0] - (u[..., 1:] * v[..., 1:]).sum(dim=-1)
        z = (c * inner).clamp(min=1.0 + 1e-7)
        
        # dist = acosh(z) / sqrt(c)
        # d(dist)/dz = 1 / (sqrt(c) * sqrt(z^2 - 1))
        sqrt_c = c**0.5
        d_dist_dz = 1.0 / (sqrt_c * torch.sqrt(z*z - 1.0))
        
        # d(z)/du = c * d(<u,v>)/du
        # 내적의 미분 (민코프스키 계량 고려)
        
        grad_z_u = torch.empty_like(v)
        grad_z_u[..., 0] = c * v[..., 0]
        grad_z_u[..., 1:] = -c * v[..., 1:]
        
        grad_z_v = torch.empty_like(u)
        grad_z_v[..., 0] = c * u[..., 0]
        grad_z_v[..., 1:] = -c * u[..., 1:]
        
        # 연쇄 법칙
        scale = (grad_output * d_dist_dz).unsqueeze(-1)
        grad_u = scale * grad_z_u
        grad_v = scale * grad_z_v
        
        return grad_u, grad_v, None

def lorentz_distance(x: Tensor, y: Tensor, c: float | Tensor) -> Tensor:
    if _rust is None:
        c_val = float(c.detach().cpu().item()) if isinstance(c, Tensor) else float(c)
        return lorentz_distance_torch(x, y, c_val)
    if isinstance(c, Tensor):
        eps = 1e-7
        inner = x[..., 0] * y[..., 0] - (x[..., 1:] * y[..., 1:]).sum(dim=-1)
        z = (c * inner).clamp(min=1.0 + eps)
        sqrt_c = torch.sqrt(c)
        return torch.acosh(z) / sqrt_c
    return LorentzDistance.apply(x, y, float(c))


class LorentzLayer(Function):
    """
    로렌츠 레이어 (지오데식 보간) Function
    """
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float, t: float) -> Tensor:
        ctx.c = c
        ctx.t = t
        ctx.save_for_backward(u, v)
        if _rust is None:
            return (1.0 - float(t)) * u + float(t) * v
        if u.is_cuda and _has_cuda:
            output = torch.empty_like(u)
            _rust.lorentz_layer_forward_cuda(
                output.data_ptr(), u.data_ptr(), v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
            return output
        else:
            output_np = _rust.lorentz_layer_forward(u.cpu().numpy(), v.cpu().numpy(), c, t)
            return torch.from_numpy(output_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        u, v = ctx.saved_tensors
        c, t = ctx.c, ctx.t
        grad_u = grad_v = None
        if _rust is None:
            return grad_output * (1.0 - float(t)), grad_output * float(t), None, None
        if grad_output.is_cuda and _has_cuda:
            grad_u = torch.empty_like(u)
            grad_v = torch.empty_like(v)
            _rust.lorentz_ball_layer_backward_cuda(
                grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                grad_u.data_ptr(), grad_v.data_ptr(),
                c, t, u.shape[0], u.shape[1]
            )
        else:
            grad_u_np, grad_v_np = _rust.lorentz_ball_layer_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
            )
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
        return grad_u, grad_v, None, None

def lorentz_add(u: Tensor, v: Tensor, c: float) -> Tensor:
    """
    로렌츠 덧셈 (자이로벡터 합)
    """
    if _rust is None:
        return u + v
    result_np = _rust.lorentz_add(u.cpu().numpy(), v.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(u.device)

def lorentz_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    """
    로렌츠 스칼라 곱
    """
    if _rust is None:
        return x * float(r)
    result_np = _rust.lorentz_scalar(x.cpu().numpy(), r, c)
    return torch.from_numpy(result_np).to(x.device)

def lorentz_inner(u: Tensor, v: Tensor) -> Tensor:
    """
    로렌츠 민코프스키 내적
    """
    if _rust is None:
        return lorentz_inner_torch(u, v)
    result_np = _rust.lorentz_inner(u.cpu().numpy(), v.cpu().numpy())
    return torch.from_numpy(result_np).to(u.device)

def lorentz_to_poincare(x: Tensor, c: float) -> Tensor:
    """
    로렌츠 -> 푸앵카레 변환
    """
    if _rust is None:
        denom = x[..., :1] + (1.0 / float(c)) ** 0.5
        return x[..., 1:] / denom.clamp_min(1e-7)
    result_np = _rust.lorentz_to_poincare(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device)

def lorentz_to_klein(x: Tensor, c: float) -> Tensor:
    """
    로렌츠 -> 클라인 변환
    """
    if _rust is None:
        return x[..., 1:] / x[..., :1].clamp_min(1e-7)
    result_np = _rust.lorentz_to_klein(x.cpu().numpy(), c)
    return torch.from_numpy(result_np).to(x.device) 

def euclidean_to_lorentz(x: Tensor, c: float = 1.0, epsilon: float = 1e-6) -> Tensor:
    """
    유클리드 벡터 x를 로렌츠 쌍곡면으로 들어올립니다(Lift).
    R^n의 x를 R^{n+1}의 부분집합인 H^n의 (sqrt(1/c + ||x||^2), x)로 매핑합니다.
    """
    # sq = ||x||^2
    sq = (x * x).sum(dim=-1, keepdim=True)
    # x0 = sqrt(1/c + ||x||^2)
    time_comp = torch.sqrt(torch.clamp(1.0 / c + sq, min=epsilon))
    return torch.cat([time_comp, x], dim=-1) 


class LorentzBallLayer(Function):
    """
    로렌츠 공 레이어 (Lorentz Ball Layer)
    
    동적 곡률(Dynamic Curvature)을 지원하며, 로렌츠 모델 위에서 두 점을 보간합니다.
    """
    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = 0.1, c_max: float = 5.0) -> Tensor:
        ctx.t = t
        if kappas is not None and layer_idx is not None:
            ctx.use_dynamic = True
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(u, v, kappas)
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
            # 네이티브 바인딩이 있다면 우선 사용
            if hasattr(_rust, 'lorentz_layer_layerwise_cpu'):
                out_np, c_val = _rust.lorentz_layer_layerwise_cpu(
                    u.cpu().numpy(), v.cpu().numpy(), kappa_val, layer_idx, c_min, c_max, t
                )
                ctx.c_val = c_val
                return torch.from_numpy(out_np).to(u.device)
            else:
                # Python fallback
                sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val)))
                c_val = c_min + (c_max - c_min) * sig.item()
                ctx.c_val = c_val
                out_np = _rust.lorentz_layer_forward(u.cpu().numpy(), v.cpu().numpy(), c_val, t)
                return torch.from_numpy(out_np).to(u.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            ctx.save_for_backward(u.clone(), v.clone())
            out_np = _rust.lorentz_layer_forward(u.cpu().numpy(), v.cpu().numpy(), ctx.c, t)
            return torch.from_numpy(out_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        t = ctx.t
        if ctx.use_dynamic:
            u, v, kappas = ctx.saved_tensors
            layer_idx = ctx.layer_idx
            c_min = ctx.c_min
            c_max = ctx.c_max
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
            c_val = getattr(ctx, 'c_val', None)
            if c_val is None:
                sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val)))
                c_val = (c_min + (c_max - c_min) * sig.item())
                ctx.c_val = c_val
            
            # 정적 역전파를 통한 u, v 그라디언트 계산
            if grad_output.is_cuda and _has_cuda:
                grad_u = torch.empty_like(u)
                grad_v = torch.empty_like(v)
                _rust.lorentz_ball_layer_backward_cuda(
                    grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    grad_u.data_ptr(), grad_v.data_ptr(),
                    float(c_val), t, u.shape[0], u.shape[1]
                )
            else:
                gu_np, gv_np = _rust.lorentz_ball_layer_backward_cpu(
                    grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), float(c_val), t
                )
                grad_u = torch.from_numpy(gu_np).to(grad_output.device)
                grad_v = torch.from_numpy(gv_np).to(grad_output.device)

            # kappa에 대한 정확한 그라디언트 (연쇄 법칙)
            def minkowski_inner(p: Tensor, q: Tensor) -> Tensor:
                return p[..., :1]*q[..., :1] - (p[..., 1:]*q[..., 1:]).sum(dim=-1, keepdim=True)

            eps = 1e-7
            inner = minkowski_inner(u, v)  # (B,1)
            z = torch.clamp_min(-float(c_val) * inner, 1.0 + eps)
            alpha = torch.acosh(z)
            sinh_a = torch.sinh(alpha).clamp_min(eps)
            cosh_a = torch.cosh(alpha)

            t1 = (1.0 - t) * alpha
            t2 = t * alpha
            w1 = torch.where(alpha.abs() < 1e-6, torch.full_like(alpha, 1.0 - t), torch.sinh(t1) / sinh_a)
            w2 = torch.where(alpha.abs() < 1e-6, torch.full_like(alpha, t), torch.sinh(t2) / sinh_a)

            num1 = (1.0 - t) * torch.cosh(t1) * sinh_a - torch.sinh(t1) * cosh_a
            num2 = t * torch.cosh(t2) * sinh_a - torch.sinh(t2) * cosh_a
            denom = (sinh_a * sinh_a).clamp_min(eps)
            dw1_da = torch.where(alpha.abs() < 1e-6, torch.zeros_like(alpha), num1 / denom)
            dw2_da = torch.where(alpha.abs() < 1e-6, torch.zeros_like(alpha), num2 / denom)

            dalpha_dz = 1.0 / (torch.sqrt(torch.clamp_min(z+1.0, 1.0+eps)) * torch.sqrt(torch.clamp_min(z-1.0, eps)))
            dz_dc = -inner
            dalpha_dc = dalpha_dz * dz_dc

            dw1_dc = dw1_da * dalpha_dc
            dw2_dc = dw2_da * dalpha_dc

            # dy/dc = dw1_dc * u + dw2_dc * v
            dy_dc = dw1_dc * u + dw2_dc * v
            grad_c_total = (grad_output * dy_dc).sum()
            sig = 1.0 / (1.0 + torch.exp(torch.tensor(-kappa_val, dtype=torch.float32, device=grad_output.device)))
            dc_dkappa = (c_max - c_min) * sig * (1.0 - sig)
            gk_val = (grad_c_total * dc_dkappa).item()

            if kappas.dim() == 0:
                grad_kappas = torch.tensor(gk_val, device=kappas.device)
            else:
                grad_kappas = torch.zeros_like(kappas)
                grad_kappas[layer_idx] = gk_val
            return grad_u, grad_v, None, None, grad_kappas, None, None, None
        else:
            u, v = ctx.saved_tensors
            c = ctx.c
            if grad_output.is_cuda and _has_cuda:
                grad_u = torch.empty_like(u)
                grad_v = torch.empty_like(v)
                _rust.lorentz_ball_layer_backward_cuda(
                    grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    grad_u.data_ptr(), grad_v.data_ptr(),
                    c, t, u.shape[0], u.shape[1]
                )
                return grad_u, grad_v, None, None, None, None, None, None
            else:
                gu_np, gv_np = _rust.lorentz_ball_layer_backward_cpu(
                    grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
                )
                grad_u = torch.from_numpy(gu_np).to(grad_output.device)
                grad_v = torch.from_numpy(gv_np).to(grad_output.device)
                return grad_u, grad_v, None, None, None, None, None, None


def lorentz_ball(u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = 0.1, c_max: float = 5.0) -> Tensor:
    """
    로렌츠 공 레이어 편의 함수
    """
    return LorentzBallLayer.apply(u, v, c, t, kappas, layer_idx, c_min, c_max)

class LorentzFromPoincare(Function):
    """
    푸앵카레 -> 로렌츠 변환 (Autograd 지원, 동적 곡률 지원)
    """
    @staticmethod
    def forward(ctx, x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
        if kappas is not None:
            ctx.use_dynamic = True
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(x, kappas)
            
            output_np, c_val = _rust.from_poincare_dynamic_cpu(
                x.cpu().numpy(), kappas.item(), c_min, c_max
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(x.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            # 정적 경로
            output = poincare_to_lorentz(x, ctx.c)
            ctx.save_for_backward(x)
            return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        if ctx.use_dynamic:
            x, kappas = ctx.saved_tensors
            grad_x_np, grad_kappa_val = _rust.from_poincare_dynamic_backward_cpu(
                grad_output.cpu().numpy(), x.cpu().numpy(), kappas.item(), ctx.c_min, ctx.c_max
            )
            grad_x = torch.from_numpy(grad_x_np).to(grad_output.device)
            grad_kappas = torch.tensor(grad_kappa_val, device=kappas.device)
            return grad_x, None, grad_kappas, None, None
        else:
            x, = ctx.saved_tensors
            grad_x = torch.zeros_like(x)
            return grad_x, None, None, None, None

def from_poincare(x: Tensor, c: float = None, kappas: Tensor = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
    """
    푸앵카레 -> 로렌츠 변환 편의 함수
    """
    return LorentzFromPoincare.apply(x, c, kappas, c_min, c_max)
```
---
## File: `reality_stone/python/reality_stone/layers/metric_attention.py`

```python
import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .poincare import poincare_distance
from .lorentz import lorentz_distance
from .klein import klein_distance

try:
    from reality_stone._rust import geodesic as _geodesic
    geodesic_topk_attention = _geodesic.geodesic_topk_attention
    HAS_CUDA_KERNEL = True
except Exception:
    HAS_CUDA_KERNEL = False

class SPDMetric(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 0, init_u_scale: float = 1e-3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = int(rank) if rank is not None else 0
        self.log_diag = nn.Parameter(torch.zeros(hidden_size))
        if self.rank > 0:
            u = torch.randn(hidden_size, self.rank) * float(init_u_scale)
            self.U = nn.Parameter(u)
        else:
            self.U = None

    def scale_q(self, q: Tensor) -> Tensor:
        d = F.softplus(self.log_diag).view(1, 1, 1, -1)
        return q * d

    def scale_k(self, k: Tensor) -> Tensor:
        d = F.softplus(self.log_diag).view(1, 1, 1, -1)
        return k * d

    def lowrank_proj(self, x: Tensor) -> Optional[Tensor]:
        if self.U is None:
            return None
        return torch.einsum("bhtd,dr->bhtr", x, self.U)


def _sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    z = logits
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    range_arange = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    range_view = [1] * z.dim()
    range_view[dim] = -1
    range_arange = range_arange.view(range_view)
    cssv = torch.cumsum(z_sorted, dim=dim) - range_arange * z_sorted
    nonzero = (z_sorted > (cssv / range_arange)).to(z.dtype)
    k = (nonzero * range_arange).max(dim=dim, keepdim=True).values.clamp(min=1.0)
    tau = (torch.gather(z_sorted, dim, k.long() - 1) - (cssv.gather(dim, k.long() - 1) / k)).detach()
    p = torch.clamp(z - tau, min=0.0)
    return p


def _sinkhorn(logits: Tensor, iters: int = 20, tau: float = 1.0, eps: float = 1e-9) -> Tensor:
    x = torch.exp(logits / max(tau, 1e-6))
    for _ in range(max(1, int(iters))):
        x = x / (x.sum(dim=-1, keepdim=True) + eps)
    return x


def normalize(scores: Tensor, method: str = "softmax", tau: float = 1.0) -> Tensor:
    scores = scores - scores.max(dim=-1, keepdim=True).values
    if method == "softmax":
        return torch.softmax(scores / max(tau, 1e-6), dim=-1)
    if method in {"entmax", "entmax15", "sparsemax"}:
        return _sparsemax(scores, dim=-1)
    if method == "sinkhorn":
        return _sinkhorn(scores, iters=20, tau=tau)
    return torch.softmax(scores / max(tau, 1e-6), dim=-1)


def build_topo_topk(topo_idx: Dict[str, Tensor], topk_cfg: Dict[str, int]) -> Tensor:
    if not isinstance(topo_idx, dict) or not topo_idx:
        raise ValueError("topo_idx must be a non-empty dict of relation -> indices (B,T,k_r)")

    idx_tensors = []
    for rel, k in topk_cfg.items():
        if k is None or k <= 0:
            continue
        if rel not in topo_idx:
            continue
        rel_idx = topo_idx[rel]
        if rel_idx.shape[-1] > k:
            rel_idx = rel_idx[..., :k]
        idx_tensors.append(rel_idx)

    if not idx_tensors:
        raise ValueError("No relations matched between topo_idx and topk_cfg")

    idx_all = torch.cat(idx_tensors, dim=-1)

    B, T, K_raw = idx_all.shape
    idx_flat = idx_all.reshape(B * T, K_raw)
    idx_sorted, _ = torch.sort(idx_flat, dim=-1)
    idx_uniq = torch.unique_consecutive(idx_sorted, dim=-1)
    K = idx_uniq.shape[-1]
    if K < K_raw:
        pad = idx_uniq[..., -1:].expand(B * T, K_raw - K)
        idx_uniq = torch.cat([idx_uniq, pad], dim=-1)
    elif K > K_raw:
        idx_uniq = idx_uniq[..., :K_raw]
    idx = idx_uniq.reshape(B, T, K_raw)
    return idx


def masked_gather(scores: Tensor, idx: Tensor) -> Tensor:
    B, H, T, S = scores.shape
    if idx.shape[0] != B or idx.shape[1] != T:
        raise ValueError("idx batch/time dims must match scores")
    idx_exp = idx.unsqueeze(1).expand(B, H, T, -1)
    return scores.gather(dim=3, index=idx_exp)


def aggregate(weights: Tensor, values: Tensor, idx: Tensor) -> Tensor:
    B, H, S, Dv = values.shape
    _, _, T, K = weights.shape
    if idx.shape[0] != B or idx.shape[1] != T:
        raise ValueError("idx batch/time dims must match weights")
    idx_h = idx.unsqueeze(1).expand(B, H, T, K)
    values_flat = values.reshape(B * H, S, Dv)
    idx_flat = idx_h.reshape(B * H, T * K)
    v_g = values_flat.gather(dim=1, index=idx_flat.unsqueeze(-1).expand(B * H, T * K, Dv))
    v_sel = v_g.reshape(B, H, T, K, Dv)
    y = (weights.unsqueeze(-1) * v_sel).sum(dim=3)
    return y


def get_default_topk_cfg() -> Dict[str, int]:
    return {"cell": 8, "row": 16, "col": 8, "pc": 4, "hdr": 12}


class MetricAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        normalizer: str = "softmax",
        rank: int = 0,
        tau: float = 1.0,
        mode: str = "dot",
        manifold: str = "poincare",
        c: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.normalizer = normalizer
        self.tau = float(tau)
        self.metric = SPDMetric(hidden_size, rank=rank)
        self.mode = str(mode)
        self.manifold = str(manifold)
        self.c_default = float(c)
        self._metric_cache: Dict[str, torch.Tensor] = {}

    def _apply_metric_factor(self, x: Tensor, l_factor: Tensor) -> Tensor:
        if x.dim() != 4 or l_factor.dim() != 2:
            return x
        l = l_factor.to(device=x.device, dtype=x.dtype)
        return torch.einsum("ij,bhtj->bhti", l, x)

    def _cholesky_from_keys(
        self,
        keys,
        masses,
        dim: int,
        min_lambda: float = 0.8,
        max_lambda: float = 1.2,
    ) -> torch.Tensor:
        """
        Build SPD metric from key(s) and return Cholesky factor L such that G = L L^T.
        Uses weighted average over per-key SPD metrics (still SPD).
        Cached by tuple(keys)+tuple(masses).
        """
        from .. import metrikey as _metrikey  # defer import to avoid circulars and allow no-extension mode
        if _metrikey is None:
            raise RuntimeError("MetriKey extension is not available. Omit metric_keys or use Python 3.11 to load the bundled extension.")
        import numpy as np  # lazy import to avoid hard dependency if SPD path is unused

        mk = tuple(keys), tuple(masses or [1.0] * len(keys)), dim, float(min_lambda), float(max_lambda)
        cache_key = f"{mk}"
        if cache_key in self._metric_cache:
            return self._metric_cache[cache_key]

        # accumulate SPD metrics
        m = np.array(mk[1], dtype="float32")
        m_sum = float(m.sum()) if m.size > 0 else 1.0
        g_accum = None
        for k, mass in zip(keys, (masses or [1.0] * len(keys))):
            g_k = _metrikey.spd_metric_from_key_weighted(k, dim, float(min_lambda), float(max_lambda), float(mass))
            g_accum = g_k if g_accum is None else (g_accum + g_k)
        g = g_accum / max(m_sum, 1e-6)
        l = _metrikey.metric_factor_cholesky(g)  # numpy array
        l_t = torch.from_numpy(np.asarray(l, dtype=np.float32))
        # cache on CPU to avoid device-specific duplication
        self._metric_cache[cache_key] = l_t
        return l_t

    def _geodesic_distance_pairs(self, q_pairs: Tensor, k_pairs: Tensor, c: float) -> Tensor:
        """
        Compute per-pair geodesic distance for flattened pairs.
        q_pairs, k_pairs: (N, d)
        Returns: (N,)
        """
        if self.manifold == "poincare":
            return poincare_distance(q_pairs, k_pairs, c)
        if self.manifold == "lorentz":
            return lorentz_distance(q_pairs, k_pairs, c)
        if self.manifold == "klein":
            return klein_distance(q_pairs, k_pairs, c)
        # Fallback: Euclidean
        return torch.norm(q_pairs - k_pairs, dim=-1)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        topo_idx: Optional[Dict[str, Tensor]] = None,
        rel_bias: Optional[Tensor] = None,
        topk_cfg: Optional[Dict[str, int]] = None,
        causal: bool = False,
        # RCE options
        metric_keys: Optional[list[str]] = None,   # e.g., ["dept:0","dept:3"]
        masses: Optional[list[float]] = None,      # same length as metric_keys
        metric_keys_b: Optional[list[str]] = None, # optional second context to mix
        alpha: Optional[float] = None,             # mix ratio for TCS: g' = a g1 + (1-a) g2
        c: Optional[float] = None,                 # curvature override for geodesic
    ) -> Tensor:
        # Optionally apply metric-key based SPD transform (security/context)
        qs, ks = q, k
        if metric_keys:
            d = q.shape[-1]
            # Compose single-context SPD
            l1 = self._cholesky_from_keys(metric_keys, masses, d)
            if metric_keys_b and alpha is not None:
                # Mix two contexts on the SPD: g' = a g1 + (1-a) g2
                l2 = self._cholesky_from_keys(metric_keys_b, masses, d)
                # Reconstruct G from L then mix and re-factor (CPU numpy for stability)
                import numpy as np  # local import
                from .. import metrikey as _metrikey  # local import to avoid circulars
                g1 = (l1 @ l1.t()).cpu().numpy()
                g2 = (l2 @ l2.t()).cpu().numpy()
                a = float(max(0.0, min(1.0, alpha)))
                g_mix = a * g1 + (1.0 - a) * g2
                if _metrikey is None:
                    raise RuntimeError("MetriKey extension is not available for mixed SPD context.")
                l_mix = _metrikey.metric_factor_cholesky(g_mix)
                l_used = torch.from_numpy(np.asarray(l_mix, dtype=np.float32))
            else:
                l_used = l1
            qs = self._apply_metric_factor(q, l_used)
            ks = self._apply_metric_factor(k, l_used)

        # Metric swap (learnable diag / optional low-rank) — applies after metric-key
        qs = self.metric.scale_q(qs)  # (B,H,T,d_h)
        ks = self.metric.scale_k(ks)  # (B,H,S,d_h)

        qu = self.metric.lowrank_proj(qs)
        ku = self.metric.lowrank_proj(ks)

        # Geodesic Top-k branch (preferred with topology selection)
        if self.mode == "geodesic" and topo_idx is not None and topk_cfg is not None:
            idx = build_topo_topk(topo_idx, topk_cfg)  # (B,T,K)
            B, H, T, Dh = qs.shape
            S = ks.shape[-2]
            K = idx.shape[-1]
            if HAS_CUDA_KERNEL and qs.is_cuda:
                if metric_keys:
                    l_factor = self._cholesky_from_keys(metric_keys, masses, Dh).to(qs.device)
                else:
                    diag = F.softplus(self.metric.log_diag).to(qs.device)
                    l_factor = torch.diag(diag.sqrt())
                c_used = float(self.c_default if c is None else c)
                try:
                    y = geodesic_topk_attention(qs, ks, v, idx, l_factor, c_used, self.tau)
                    return y
                except Exception:
                    pass
            
            # Python fallback (original implementation)
            # Flatten and gather selected keys: (B*H, S, Dh) -> (B*H, T*K, Dh)
            ks_flat = ks.reshape(B * H, S, Dh)
            idx_flat = idx.unsqueeze(1).expand(B, H, T, K).reshape(B * H, T * K)
            ks_sel_flat = ks_flat.gather(1, idx_flat.unsqueeze(-1).expand(B * H, T * K, Dh))
            # Replicate queries per K: (B*H, T, Dh) -> (B*H, T*K, Dh)
            q_flat = qs.reshape(B * H, T, Dh)
            q_rep = q_flat.unsqueeze(2).expand(B * H, T, K, Dh).reshape(B * H, T * K, Dh)
            # Compute geodesic distance per pair
            qf = q_rep.reshape(B * H * T * K, Dh)
            kf = ks_sel_flat.reshape(B * H * T * K, Dh)
            c_used = float(self.c_default if c is None else c)
            dist = self._geodesic_distance_pairs(qf, kf, c_used)  # (B*H*T*K,)
            d2 = dist.pow(2.0).reshape(B, H, T, K)
            # Convert to scores, then normalize per Top-k set
            s_sel = -d2 / max(self.tau, 1e-6)
            # Low-rank auxiliary term to ensure gradient flows to metric.U
            if qu is not None and ku is not None:
                s_lr_full = torch.einsum("bhtr,bhsr->bhts", qu, ku)  # (B,H,T,S)
                s_lr = masked_gather(s_lr_full, idx)  # (B,H,T,K)
                s_sel = s_sel + 1e-3 * s_lr
            if rel_bias is not None:
                # if bias is full (B,H,T,S), gather it
                if rel_bias.dim() == 4 and rel_bias.shape[-1] == S:
                    b_sel = masked_gather(rel_bias, idx)
                    s_sel = s_sel + b_sel
            a = normalize(s_sel, method=self.normalizer, tau=1.0)
            y = aggregate(a, v, idx)  # (B,H,T,d_v)
            return y

        # Dot-product path (default) or geodesic fallback without topo_idx
        s = torch.einsum("bhtd,bhsd->bhts", qs, ks) / math.sqrt(self.hidden_size)
        if qu is not None and ku is not None:
            s = s + torch.einsum("bhtr,bhsr->bhts", qu, ku)
        if rel_bias is not None:
            s = s + rel_bias
        if causal and s.size(2) == s.size(3):
            t = s.size(2)
            mask = torch.ones((t, t), device=s.device, dtype=torch.bool).triu(diagonal=1)
            s = s.masked_fill(mask.view(1, 1, t, t), float("-inf"))
        if topo_idx is not None and topk_cfg is not None:
            idx = build_topo_topk(topo_idx, topk_cfg)  # (B,T,K)
            s_sel = masked_gather(s, idx)  # (B,H,T,K)
            a = normalize(s_sel, method=self.normalizer, tau=self.tau)
            y = aggregate(a, v, idx)
            return y
        a_full = normalize(s, method=self.normalizer, tau=self.tau)
        y_full = torch.einsum("bhts,bhsd->bhtd", a_full, v)
        return y_full


__all__ = [
    "SPDMetric",
    "MetricAttention",
    "normalize",
    "build_topo_topk",
    "masked_gather",
    "aggregate",
    "get_default_topk_cfg",
]
```
---
## File: `reality_stone/python/reality_stone/layers/poincare.py`

```python
import torch
from torch import Tensor
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from .. import _rust, _has_cuda
from ..core.mobius import MobiusAdd, MobiusScalarMul
import math
from .._fallback import dynamic_curvature, mobius_add_torch

def project_to_ball(x: Tensor, epsilon: float = 1e-7) -> Tensor:
    """
    텐서를 푸앵카레 공으로 투영합니다. 모든 차원을 지원합니다.
    """
    # 순수 PyTorch 구현으로 대체 (Rust 바인딩이 2D만 지원하므로)
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    # norm이 1-epsilon보다 큰 경우만 스케일링
    scale = torch.where(
        norm > 1.0 - epsilon,
        (1.0 - epsilon) / norm,
        torch.ones_like(norm)
    )
    return x * scale

class PoincareBallLayer(Function):
    """
    푸앵카레 공 모델에서의 거리 및 덧셈 연산을 수행하는 레이어입니다.
    """

    @staticmethod
    def forward(ctx, u: Tensor, v: Tensor, c: float = None, t: float = 0.5, kappas: Tensor = None, layer_idx: int = None, c_min: float = -2.0, c_max: float = -0.1) -> Tensor:
        ctx.t = t
        
        if kappas is not None and layer_idx is not None:
            ctx.use_dynamic = True
            ctx.layer_idx = layer_idx
            ctx.c_min = c_min
            ctx.c_max = c_max
            ctx.save_for_backward(u, v, kappas)
            
            # kappas가 0차원 텐서면 바로 item(), 1차원 이상이면 인덱싱
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
                
            if _rust is None:
                c_val = dynamic_curvature(kappa_val, c_min, c_max)
                ctx.c_val = c_val
                return mobius_add_torch(u, (v - u) * float(t), c_val)
            output_np, c_val = _rust.poincare_ball_layer_layerwise_cpu(
                u.cpu().numpy(), v.cpu().numpy(), kappa_val, layer_idx, c_min, c_max, t
            )
            ctx.c_val = c_val
            return torch.from_numpy(output_np).to(u.device)
        else:
            ctx.use_dynamic = False
            ctx.c = c if c is not None else 1.0
            ctx.save_for_backward(u, v)
            # CUDA 경로는 실제 바인딩이 존재할 때만 사용 (안전 가드)
            if _rust is None:
                return mobius_add_torch(u, (v - u) * float(t), ctx.c)
            if (
                u.is_cuda
                and _has_cuda
                and hasattr(_rust, "poincare_ball_layer_cuda")
            ):
                output = torch.empty_like(u)
                _rust.poincare_ball_layer_cuda(
                    output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    ctx.c, t, u.shape[0], u.shape[1]
                )
                return output
            else:
                # CPU Path - 루트 레벨과 서브모듈 모두 시도
                try:
                    output_np = _rust.poincare_ball_layer_cpu(
                        u.cpu().numpy(), v.cpu().numpy(), ctx.c, t
                    )
                except AttributeError:
                    # Fallback to submodule
                    output_np = _rust.poincare.poincare_ball_layer_cpu(
                        u.cpu().numpy(), v.cpu().numpy(), ctx.c, t
                    )
                return torch.from_numpy(output_np).to(u.device)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        t = ctx.t
        
        if ctx.use_dynamic:
            u, v, kappas = ctx.saved_tensors
            layer_idx = ctx.layer_idx
            c_min = ctx.c_min
            c_max = ctx.c_max
            if _rust is None:
                grad_kappas = torch.zeros_like(kappas)
                return (
                    grad_output * (1.0 - float(t)),
                    grad_output * float(t),
                    None,
                    None,
                    grad_kappas,
                    None,
                    None,
                    None,
                )
            
            # kappas가 0차원 텐서면 바로 item(), 1차원 이상이면 인덱싱
            if kappas.dim() == 0:
                kappa_val = kappas.item()
            else:
                kappa_val = kappas[layer_idx].item()
                
            grad_u_np, grad_v_np, grad_kappa_val = _rust.poincare_ball_layer_layerwise_backward_cpu(
                grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), 
                kappa_val, layer_idx, c_min, c_max, t
            )
            
            grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
            grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
            
            # kappas와 같은 차원의 gradient 생성
            if kappas.dim() == 0:
                grad_kappas = torch.tensor(grad_kappa_val, device=kappas.device)
            else:
                grad_kappas = torch.zeros_like(kappas)
                grad_kappas[layer_idx] = grad_kappa_val
            
            return grad_u, grad_v, None, None, grad_kappas, None, None, None
        else:
            u, v = ctx.saved_tensors
            c = ctx.c
            grad_u = grad_v = None
            if _rust is None:
                return (
                    grad_output * (1.0 - float(t)),
                    grad_output * float(t),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            # CUDA 경로는 실제 바인딩이 존재할 때만 사용
            if (
                grad_output.is_cuda
                and _has_cuda
                and hasattr(_rust, "poincare_ball_layer_backward_cuda")
            ):
                grad_u = torch.empty_like(u)
                grad_v = torch.empty_like(v)
                _rust.poincare_ball_layer_backward_cuda(
                    grad_output.data_ptr(), u.data_ptr(), v.data_ptr(),
                    grad_u.data_ptr(), grad_v.data_ptr(),
                    c, t, u.shape[0], u.shape[1]
                )
            else:
                grad_u_np, grad_v_np = _rust.poincare_ball_layer_backward_cpu(
                    grad_output.cpu().numpy(), u.cpu().numpy(), v.cpu().numpy(), c, t
                )
                grad_u = torch.from_numpy(grad_u_np).to(grad_output.device)
                grad_v = torch.from_numpy(grad_v_np).to(grad_output.device)
            return grad_u, grad_v, None, None, None, None, None, None

def poincare_add(
    x: Tensor,
    y: Tensor,
    c: float = None,
    kappas: Tensor = None,
    layer_idx: int = None,
    c_min: float = -2.0,
    c_max: float = -0.1,
) -> Tensor:
    """
    뫼비우스 덧셈 (Mobius Addition)을 수행합니다.
    """
    return MobiusAdd.apply(x, y, c, kappas, layer_idx, c_min, c_max)

def poincare_scalar_mul(x: Tensor, r: float, c: float) -> Tensor:
    """
    뫼비우스 스칼라 곱 (Mobius Scalar Multiplication)을 수행합니다.
    """
    return MobiusScalarMul.apply(x, r, c)

def poincare_distance(x: Tensor, y: Tensor, c: float | Tensor, eps: float = 1e-7) -> Tensor:
    """
    푸앵카레 거리 (Poincare Distance)를 계산합니다.
    """
    if isinstance(c, (float, int)):
        if abs(c) < eps:
            return torch.norm(x - y, dim=1)
    elif isinstance(c, Tensor):
        if c.abs() < eps:
            return torch.norm(x - y, dim=1)
            
    x2 = (x * x).sum(dim=1)
    y2 = (y * y).sum(dim=1)
    diff2 = ((x - y) * (x - y)).sum(dim=1).clamp_min(0.0)
    den = (1.0 - c * x2) * (1.0 - c * y2)
    den = den.clamp_min(eps)
    frac = (c * diff2) / den
    frac = frac.clamp_min(0.0)
    
    # 정확한 공식: d = (2/sqrt(c)) * atanh(sqrt(frac / (1 + frac)))
    # 이는 d = (1/sqrt(c)) * arccosh(1 + 2*frac) 와 같습니다.
    
    arg = (frac / (1.0 + frac)).sqrt().clamp_max(1.0 - eps)
    if isinstance(c, Tensor):
        sqrtc = c.sqrt()
    else:
        sqrtc = torch.tensor(c, dtype=x.dtype, device=x.device).sqrt()
    return (2.0 / sqrtc) * torch.atanh(arg)

def poincare_to_lorentz(x: Tensor, c: float) -> Tensor:
    """
    푸앵카레 공 모델에서 로렌츠 모델로 변환합니다.
    """
    if _rust is None:
        c = float(c)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        den = (1.0 - c * x2).clamp_min(1e-7)
        time = (1.0 + c * x2) / (math.sqrt(c) * den)
        space = 2.0 * x / den
        return torch.cat([time, space], dim=-1)
    output_np = _rust.poincare_to_lorentz_cpu(x.cpu().numpy(), c)
    return torch.from_numpy(output_np).to(x.device)

def poincare_to_klein(x: Tensor, c: float) -> Tensor:
    """
    푸앵카레 공 모델에서 클라인 모델로 변환합니다.
    """
    if _rust is None:
        c = float(c)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        return (2.0 * x) / (1.0 + c * x2).clamp_min(1e-7)
    output_np = _rust.poincare_to_klein_cpu(x.cpu().numpy(), c)
    return torch.from_numpy(output_np).to(x.device)

# --- HyperbolicLinear 및 관련 함수 ---

def exp_map_zero(v: Tensor, c: float, eps: float = 1e-7) -> Tensor:
    """
    원점에서의 지수 맵 (접선 공간 -> 푸앵카레 공)

    NOTE: c 텐서를 항상 v 와 같은 device/dtype 으로 올려서
    CPU/GPU 혼합으로 인한 clamp/device 오류를 방지한다.
    """
    sqrt_c = torch.sqrt(torch.as_tensor(c, device=v.device, dtype=v.dtype))
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=eps)
    # 수치 안정성을 위해 norm을 제한
    v_norm_clipped = v_norm.clamp(max=1.0 / (sqrt_c * 1.1))  # 공의 경계에 너무 가까이 가지 않도록
    # tanh 계산
    sqrt_c_v_norm = sqrt_c * v_norm_clipped
    tanh_term = torch.tanh(sqrt_c_v_norm)
    # 결과 계산
    result = tanh_term / (sqrt_c * v_norm_clipped) * v
    # NaN 체크 및 처리
    result = torch.where(v_norm < eps, v, result)
    return result


def log_map_zero(y: Tensor, c: float, eps: float = 1e-7) -> Tensor:
    """
    원점에서의 로그 맵 (푸앵카레 공 -> 접선 공간)

    NOTE: c 텐서를 항상 y 와 같은 device/dtype 으로 올려서
    CPU/GPU 혼합 연산을 피한다.
    """
    sqrt_c = torch.sqrt(torch.as_tensor(c, device=y.device, dtype=y.dtype))
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True).clamp(min=eps)

    # 수치 안정성을 위해 norm을 제한 (공의 경계에서 멀리)
    y_norm_clipped = y_norm.clamp(max=1.0 - eps)

    # artanh 계산을 위한 안전한 범위 확인
    sqrt_c_y_norm = sqrt_c * y_norm_clipped
    sqrt_c_y_norm = sqrt_c_y_norm.clamp(max=1.0 - eps)  # artanh의 정의역 내로 제한

    # artanh 계산
    artanh_term = torch.atanh(sqrt_c_y_norm)

    # 결과 계산
    result = artanh_term / (sqrt_c * y_norm_clipped) * y

    # NaN 체크 및 처리
    result = torch.where(y_norm < eps, y, result)
    return result

class HyperbolicLinear(nn.Module):
    """
    쌍곡 공간에서의 선형 변환 레이어.
    두 가지 모드를 지원:
    1. 'tangent': 접선 공간에서 선형 변환 (기본값)
    2. 'mobius': Mobius 연산을 직접 사용
    """
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True, mode: str = 'tangent'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.mode = mode
        
        # 가중치는 항상 유클리드 공간에 저장
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 쌍곡 공간에 맞는 초기화
        # Xavier/Glorot 초기화를 사용하되, 쌍곡 공간의 특성을 고려하여 스케일 조정
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
        
        # 쌍곡 공간에서는 더 작은 값으로 시작하는 것이 안정적
        std = std * 0.5  
        
        with torch.no_grad():
            self.weight.uniform_(-std, std)
            
        if self.bias is not None:
            # 편향은 더 작게 초기화
            bound = 1 / torch.sqrt(torch.tensor(fan_in).float()) * 0.1
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # 3D 텐서 처리를 위해 차원 변환
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])

        if self.mode == 'tangent':
            # 기존 방식: 접선 공간에서 변환
            x_proj = project_to_ball(x, epsilon=1e-7)
            tangent_x = log_map_zero(x_proj, c=self.c, eps=1e-7)
            tangent_y = F.linear(tangent_x, self.weight, self.bias)
            hyperbolic_y = exp_map_zero(tangent_y, c=self.c, eps=1e-7)
            
        elif self.mode == 'mobius':
            # 개선된 방식: Mobius 연산 직접 사용
            x_proj = project_to_ball(x, epsilon=1e-7)
            
            # 가중치 행렬의 각 행을 푸앵카레 공으로 투영
            weight_proj = project_to_ball(self.weight, epsilon=1e-7)
            
            # Mobius 행렬-벡터 곱셈
            # y_i = sum_j (w_ij ⊗_c x_j)
            hyperbolic_y = []
            for i in range(self.out_features):
                # 각 출력 차원에 대해
                y_i = torch.zeros_like(x_proj[0])
                for j in range(self.in_features):
                    # Mobius 스칼라 곱셈과 덧셈
                    scaled = poincare_scalar_mul(x_proj[:, j:j+1], self.weight[i, j].item(), self.c)
                    if j == 0:
                        y_i = scaled
                    else:
                        y_i = poincare_add(y_i, scaled, self.c)
                hyperbolic_y.append(y_i)
            
            hyperbolic_y = torch.cat(hyperbolic_y, dim=1)
            
            # 편향 추가 (Mobius 덧셈)
            if self.bias is not None:
                bias_proj = project_to_ball(self.bias.unsqueeze(0), epsilon=1e-7)
                hyperbolic_y = poincare_add(hyperbolic_y, bias_proj.expand_as(hyperbolic_y), self.c)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # 원래 차원으로 복원
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            hyperbolic_y = hyperbolic_y.view(*output_shape)

        return hyperbolic_y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, c={self.c}'

    @classmethod
    def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):
        in_features, out_features, weight, has_bias = _extract_linear_like(linear_layer)
        hyperbolic_layer = cls(in_features=in_features, out_features=out_features, c=c, bias=has_bias)
        with torch.no_grad():
            weight_norm = torch.norm(weight, p='fro')
            target_norm = torch.sqrt(torch.tensor(weight.shape[0] * weight.shape[1]).float()) * 0.1
            scale_factor = target_norm / weight_norm
            hyperbolic_layer.weight.data.copy_(weight * scale_factor)
            if has_bias:
                hyperbolic_layer.bias.data.copy_(linear_layer.bias.data * 0.1)
        return hyperbolic_layer


# 중복 import 제거됨

class PoincareWrapper(nn.Module):
    """
    기존 선형 레이어를 감싸서 푸앵카레 공 위에서 동작하도록 만드는 래퍼(Wrapper)입니다.
    """

    def __init__(self, linear_layer: nn.Module):
        super().__init__()
        self.linear_layer = linear_layer

    def forward(self, x: Tensor) -> Tensor:
        # 1. 기존 선형 레이어 실행
        linear_output = self.linear_layer(x)
        
        # 2. 입력을 2D로 변환하여 푸앵카레 공으로 투영
        original_shape = linear_output.shape
        if linear_output.dim() > 2:
            linear_output = linear_output.view(-1, original_shape[-1])
            
        poincare_output = project_to_ball(torch.tanh(linear_output))
        
        # 3. 원래 shape으로 복원
        if len(original_shape) > 2:
            poincare_output = poincare_output.view(*original_shape)
        
        return poincare_output
    
    def __repr__(self):
        return f"PoincareWrapper({self.linear_layer})" 

class GeodesicLinear(nn.Module):
    """
    측지거리(Geodesic Distance)를 고려한 쌍곡 선형 레이어.
    HyperbolicLinear의 개선 버전으로, 더 안정적인 초기화와 스케일링을 사용합니다.
    """
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        # 가중치와 편향
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 매우 작은 값으로 초기화 (쌍곡 공간에 적합)
        with torch.no_grad():
            # Glorot uniform initialization with smaller scale
            limit = math.sqrt(6.0 / (self.in_features + self.out_features)) * 0.01
            self.weight.uniform_(-limit, limit)
            
        if self.bias is not None:
            # 편향은 0으로 초기화
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        # 입력 shape 저장
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])
            
        # 입력을 푸앵카레 공으로 투영 (더 작은 스케일)
        x_scaled = x * 0.1  # 입력 스케일 감소
        x_proj = project_to_ball(x_scaled, epsilon=1e-7)
        
        # 접선 공간으로 변환
        tangent_x = log_map_zero(x_proj, c=self.c, eps=1e-7)
        
        # 선형 변환 (가중치도 작게 스케일링)
        tangent_y = F.linear(tangent_x, self.weight * 0.1, self.bias)
        
        # 다시 푸앵카레 공으로 변환
        hyperbolic_y = exp_map_zero(tangent_y, c=self.c, eps=1e-7)
        
        # 원래 shape으로 복원
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            hyperbolic_y = hyperbolic_y.view(*output_shape)
            
        return hyperbolic_y
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):
        """기존 선형 레이어로부터 GeodesicLinear 생성"""
        in_features, out_features, weight, has_bias = _extract_linear_like(linear_layer)
        geodesic_layer = cls(in_features, out_features, c=c, bias=has_bias)
        with torch.no_grad():
            geodesic_layer.weight.data = weight * 0.01
            if has_bias:
                geodesic_layer.bias.data = linear_layer.bias.data * 0.01
        return geodesic_layer 

class EquivalentHyperbolicLinear(nn.Module):
    """
    유클리드 선형 레이어와 동등한 표현력을 가진 쌍곡 선형 레이어.
    원래 모델의 정확도를 최대한 보존하면서 쌍곡 기하학으로 변환합니다.
    """
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        # 유클리드 가중치를 저장
        self.euclidean_weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # 쌍곡 보정 파라미터 (학습 가능)
        self.scale_factor = nn.Parameter(torch.ones(1))
        self.output_scale = nn.Parameter(torch.ones(1))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 표준 초기화
        nn.init.kaiming_uniform_(self.euclidean_weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.euclidean_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # 보정 파라미터 초기화
        nn.init.ones_(self.scale_factor)
        nn.init.ones_(self.output_scale)

    def forward(self, x: Tensor) -> Tensor:
        # 이 변환은 이제 순수 유클리드 선형 변환과 동일합니다.
        # 쌍곡 변환 로직은 비활성화되었습니다.
        return F.linear(x, self.euclidean_weight, self.bias)

    @classmethod
    def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):
        """기존 선형 레이어로부터 동등한 쌍곡 레이어 생성"""
        in_features, out_features, weight, has_bias = _extract_linear_like(linear_layer)
        equiv_layer = cls(in_features, out_features, c=c, bias=has_bias)
        with torch.no_grad():
            equiv_layer.euclidean_weight.data.copy_(weight)
            if has_bias:
                equiv_layer.bias.data.copy_(linear_layer.bias.data)
        return equiv_layer 

class CompactEquivalentHyperbolicLinear(nn.Module):
    """
    메모리 효율적인 동등 쌍곡 선형 레이어.
    scale_factor를 모든 레이어가 공유하여 메모리를 절약합니다.
    """
    
    # 클래스 변수로 공유 스케일 팩터 정의
    shared_scale_factor = None
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        
        # 유클리드 가중치만 저장 (원본과 동일)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        # 공유 스케일 팩터 초기화
        if CompactEquivalentHyperbolicLinear.shared_scale_factor is None:
            CompactEquivalentHyperbolicLinear.shared_scale_factor = nn.Parameter(torch.tensor(100.0))
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 표준 초기화
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # 입력 shape 저장
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, original_shape[-1])
        
        # 1. 유클리드 공간에서 선형 변환 수행
        euclidean_output = F.linear(x, self.weight, self.bias)
        
        # 2. 공유 스케일 팩터로 쌍곡 변환
        scale = self.shared_scale_factor.abs() + 10.0
        hyperbolic_output = torch.tanh(euclidean_output / scale) * scale
        
        # 원래 shape으로 복원
        if len(original_shape) > 2:
            output_shape = list(original_shape[:-1]) + [self.out_features]
            hyperbolic_output = hyperbolic_output.view(*output_shape)
            
        return hyperbolic_output
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Module, c: float = 1.0):
        """기존 선형 레이어로부터 컴팩트 쌍곡 레이어 생성"""
        in_features, out_features, weight, has_bias = _extract_linear_like(linear_layer)
        compact_layer = cls(in_features, out_features, c=c, bias=has_bias)
        with torch.no_grad():
            compact_layer.weight.data.copy_(weight)
            if has_bias:
                compact_layer.bias.data.copy_(linear_layer.bias.data)
        return compact_layer 
def _extract_linear_like(linear_layer: nn.Module) -> tuple[int, int, torch.Tensor, bool]:
    """Linear 또는 Conv1D와 유사한 레이어에서 (in_features, out_features, weight_matrix, has_bias)를 추출합니다."""
    if 'Conv1D' in str(type(linear_layer)):
        in_features = linear_layer.weight.shape[0]
        out_features = linear_layer.weight.shape[1]
        weight = linear_layer.weight.t()
        has_bias = linear_layer.bias is not None
    else:
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        weight = linear_layer.weight.data
        has_bias = linear_layer.bias is not None
    return in_features, out_features, weight, has_bias
```
