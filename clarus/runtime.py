from __future__ import annotations
"""Reference brain runtime for the Python control plane.

This module intentionally keeps policy in Python while delegating reusable
numeric kernels to `clarus.ce_ops` / `clarus._rust`.

Concept layout from the refactor plan:
- `BrainRuntimeConfig`: global mode, lifecycle, and energy-budget policy
- `HippocampusMemory`: minimal fast-memory / replay subsystem
- `BrainRuntime`: sparse lifecycle + mode switching + snapshot continuity
"""

from dataclasses import dataclass, field
from enum import Enum
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
except ImportError:
    from clarus.ce_ops import pack_sparse
    from clarus.constants import (
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

try:
    from clarus._rust import nn_brain_step as _rust_brain_step
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
    from clarus.utils import normalize_vector as _normalize


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

    def __post_init__(self) -> None:
        self.dim = int(self.dim)
        if self.dim <= 0:
            raise ValueError("runtime dimension must be positive")
        self.active_ratio = min(max(float(self.active_ratio), 0.0), 1.0)
        self.memory_topk = max(1, int(self.memory_topk))
        self.memory_capacity = max(1, int(self.memory_capacity))

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
        import math as _math
        if not self._priority:
            return
        factor = _math.exp(-steps / FORGET_TAU)
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
        import math as _math
        self.arousal = float(external_norm)
        tau_w_inv = 1.0 / TAU_W_STEPS
        tau_s_inv = 1.0 / TAU_S_STEPS

        # Process C: circadian modulation
        self.circadian_phase += 1.0
        circadian = CIRCADIAN_BASE + CIRCADIAN_AMP * _math.cos(
            2.0 * _math.pi * self.circadian_phase / CIRCADIAN_PERIOD
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
        ext_np = external.detach().cpu().numpy().astype(np.float32)
        goal_np = self.goal.detach().cpu().numpy().astype(np.float32)
        replay_np = replay.detach().cpu().numpy().astype(np.float32)
        val_np = self.values.detach().cpu().numpy().astype(np.float32)
        col_np = self.col_idx.detach().cpu().numpy().astype(np.int32)
        row_np = self.row_ptr.detach().cpu().numpy().astype(np.int32)

        (new_act, new_ref, new_mem, new_adapt,
         new_su, new_sx, new_bit, active_count, energy) = _rust_brain_step(
            val_np, col_np, row_np,
            act_np, ref_np, mem_np, adapt_np, su_np, sx_np, bit_np,
            ext_np, goal_np, replay_np,
            mode_int, budget,
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

        active_mask = self._select_active(salience, self.config.energy_budget(mode))
        active_count = int(active_mask.sum().item())
        self.mode = mode
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
        return runtime
