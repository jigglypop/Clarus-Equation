"""Hierarchical state management for CE brain runtime.

Frontend state tree -- like React/Redux but for a brain.
Each layer has its own state, reducer, and dispatch.
Upper layers observe lower layers but don't mutate them directly.

Layer A: Cell state (activation, refractory, trace, bitfield)
Layer B: Field state (weight, coupling, topology)
Layer C: Mode state (wake/nrem/rem, sleep pressure, circadian)
Layer D: Memory state (hippocampus keys/values/priority)
Layer E: Self state (global summary, consciousness, identity)
Layer F: Agent state (critic, action, goal, working memory)

Basal ganglia = dispatcher = gate automaton between layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch

try:
    from .constants import (
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        BOOTSTRAP_CONTRACTION, NOISE_SIGMA, MEMORY_TRACE_DECAY,
        ADAPTATION_DECAY, ADAPTATION_COUPLING, ADAPTATION_CLAMP,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE,
        TAU_W_STEPS, TAU_S_STEPS, SLEEP_PRESSURE_MAX,
        REM_TAU_FACTOR, CIRCADIAN_PERIOD, CIRCADIAN_AMP,
        CIRCADIAN_BASE, FORGET_TAU, RECALL_SIMILARITY_THRESHOLD,
        NORM_EPS, CONSCIOUSNESS_TAU, CONSCIOUSNESS_CD,
    )
except ImportError:
    from clarus.constants import (
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        BOOTSTRAP_CONTRACTION, NOISE_SIGMA, MEMORY_TRACE_DECAY,
        ADAPTATION_DECAY, ADAPTATION_COUPLING, ADAPTATION_CLAMP,
        STP_TAU_FAC_INV, STP_TAU_REC, STP_U_BASE,
        TAU_W_STEPS, TAU_S_STEPS, SLEEP_PRESSURE_MAX,
        REM_TAU_FACTOR, CIRCADIAN_PERIOD, CIRCADIAN_AMP,
        CIRCADIAN_BASE, FORGET_TAU, RECALL_SIMILARITY_THRESHOLD,
        NORM_EPS, CONSCIOUSNESS_TAU, CONSCIOUSNESS_CD,
    )

import math
from collections import deque


# ---------------------------------------------------------------------------
# Action types (like Redux action types)
# ---------------------------------------------------------------------------
class ActionType(Enum):
    EXTERNAL_INPUT = auto()
    TICK = auto()
    ENCODE_MEMORY = auto()
    RECALL_MEMORY = auto()
    MODE_SWITCH = auto()
    TEACH = auto()
    QUERY = auto()
    THINK = auto()
    GATE_OPEN = auto()
    GATE_CLOSE = auto()
    SLEEP_ENTER = auto()
    SLEEP_EXIT = auto()
    STDP_UPDATE = auto()
    CHECKPOINT = auto()


@dataclass
class Action:
    type: ActionType
    payload: dict = field(default_factory=dict)
    source: str = ""


# ---------------------------------------------------------------------------
# Layer A: Cell State
# ---------------------------------------------------------------------------
@dataclass
class CellState:
    """Minimal per-cell state vector: (a, r, m, w, b, u, x)."""
    dim: int
    activation: torch.Tensor = None
    refractory: torch.Tensor = None
    memory_trace: torch.Tensor = None
    adaptation: torch.Tensor = None
    stp_u: torch.Tensor = None
    stp_x: torch.Tensor = None
    bitfield: torch.Tensor = None

    def __post_init__(self):
        if self.activation is None:
            self.activation = torch.zeros(self.dim)
            self.refractory = torch.zeros(self.dim)
            self.memory_trace = torch.zeros(self.dim)
            self.adaptation = torch.zeros(self.dim)
            self.stp_u = torch.full((self.dim,), STP_U_BASE)
            self.stp_x = torch.ones(self.dim)
            self.bitfield = torch.zeros(self.dim, dtype=torch.uint8)

    def update(self, drive: torch.Tensor, mode_params: dict) -> None:
        """Pure cell dynamics. No side effects on other layers."""
        gamma_a = mode_params.get("gamma_a", 0.18)
        kappa_a = mode_params.get("kappa_a", 0.82)
        gamma_r = mode_params.get("gamma_r", 0.12)
        kappa_r = mode_params.get("kappa_r", 0.24)

        self.activation = (
            (1.0 - gamma_a) * self.activation + kappa_a * torch.tanh(drive)
        ).clamp(-1, 1)
        self.refractory = (
            (1.0 - gamma_r) * self.refractory + kappa_r * self.activation.square()
        )
        self.memory_trace = (
            (1.0 - MEMORY_TRACE_DECAY) * self.memory_trace
            + MEMORY_TRACE_DECAY * self.activation
        )
        self.adaptation = (
            (1.0 - ADAPTATION_DECAY) * self.adaptation
            + ADAPTATION_DECAY * self.activation.square()
        ).clamp(0, ADAPTATION_CLAMP)

        self.bitfield[self.activation >= 0.30] = 1
        self.bitfield[self.activation <= 0.10] = 0

    @property
    def active_mask(self) -> torch.Tensor:
        return self.bitfield.bool()

    @property
    def active_count(self) -> int:
        return int(self.bitfield.sum().item())

    @property
    def active_ratio(self) -> float:
        return self.active_count / max(self.dim, 1)


# ---------------------------------------------------------------------------
# Layer B: Field State
# ---------------------------------------------------------------------------
@dataclass
class FieldState:
    """Coupling structure: sparse weight matrix + topology."""
    dim: int
    weight: torch.Tensor = None
    dale_sign: torch.Tensor = None

    def __post_init__(self):
        if self.weight is None:
            self.weight = torch.zeros(self.dim, self.dim)
        if self.dale_sign is None:
            n_exc = int(self.dim * 0.8)
            self.dale_sign = torch.ones(self.dim)
            self.dale_sign[n_exc:] = -4.0

    def recurrent(self, activation: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        pre = activation * active_mask.float()
        return self.weight @ pre

    def energy(self, activation: torch.Tensor) -> float:
        return float(-0.5 * torch.dot(activation, self.weight @ activation).item())


# ---------------------------------------------------------------------------
# Layer C: Mode State
# ---------------------------------------------------------------------------
class Mode(Enum):
    WAKE = "WAKE"
    NREM = "NREM"
    REM = "REM"

    @property
    def params(self) -> dict:
        return {
            Mode.WAKE: {"gamma_a": 0.18, "kappa_a": 0.82, "gamma_r": 0.12, "kappa_r": 0.24,
                        "noise": 1.0, "budget_scale": 1.0, "replay_mix": 0.08},
            Mode.NREM: {"gamma_a": 0.34, "kappa_a": 0.52, "gamma_r": 0.26, "kappa_r": 0.12,
                        "noise": 0.3, "budget_scale": 0.5, "replay_mix": 0.28},
            Mode.REM:  {"gamma_a": 0.22, "kappa_a": 0.68, "gamma_r": 0.18, "kappa_r": 0.18,
                        "noise": 0.7, "budget_scale": 0.75, "replay_mix": 0.35},
        }[self]


@dataclass
class ModeState:
    """Global operating mode + sleep pressure + circadian."""
    mode: Mode = Mode.WAKE
    sleep_pressure: float = 0.0
    circadian_phase: float = 0.0
    arousal: float = 0.0
    nrem_cycle: int = 0

    def update_pressure(self):
        tau_w_inv = 1.0 / TAU_W_STEPS
        tau_s_inv = 1.0 / TAU_S_STEPS
        if self.mode == Mode.WAKE:
            self.sleep_pressure += (SLEEP_PRESSURE_MAX - self.sleep_pressure) * tau_w_inv
        elif self.mode == Mode.NREM:
            self.sleep_pressure -= self.sleep_pressure * tau_s_inv
        else:
            self.sleep_pressure -= self.sleep_pressure * tau_s_inv * REM_TAU_FACTOR
        self.sleep_pressure = max(0.0, min(self.sleep_pressure, SLEEP_PRESSURE_MAX))
        self.circadian_phase += 1.0

    @property
    def circadian_value(self) -> float:
        return CIRCADIAN_BASE + CIRCADIAN_AMP * math.cos(
            2.0 * math.pi * self.circadian_phase / CIRCADIAN_PERIOD)

    def should_sleep(self, external_norm: float) -> bool:
        return self.sleep_pressure > 1.0 and external_norm < 0.18

    def should_wake(self, external_norm: float) -> bool:
        return external_norm > 0.27 or self.sleep_pressure < 0.15


# ---------------------------------------------------------------------------
# Layer D: Memory State
# ---------------------------------------------------------------------------
@dataclass
class MemoryState:
    """Hippocampal memory: keys, values, priorities."""
    dim: int
    capacity: int = 128
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    priorities: list = field(default_factory=list)

    def encode(self, key: torch.Tensor, value: torch.Tensor, priority: float = 1.0):
        k = key.detach().float()
        k = k / k.norm().clamp(min=NORM_EPS)
        v = value.detach().float()
        if len(self.priorities) >= self.capacity:
            drop = min(range(len(self.priorities)), key=self.priorities.__getitem__)
            self.keys.pop(drop)
            self.values.pop(drop)
            self.priorities.pop(drop)
        self.keys.append(k)
        self.values.append(v)
        self.priorities.append(max(priority, 1e-6))

    def recall(self, cue: torch.Tensor, topk: int = 4) -> torch.Tensor:
        if not self.keys:
            return torch.zeros(self.dim)
        cue_n = cue.detach().float()
        cue_n = cue_n / cue_n.norm().clamp(min=NORM_EPS)
        keys_t = torch.stack(self.keys)
        vals_t = torch.stack(self.values)
        prio_t = torch.tensor(self.priorities)
        sim = keys_t @ cue_n
        above = sim >= RECALL_SIMILARITY_THRESHOLD
        if not above.any():
            return torch.zeros(self.dim)
        score = sim + prio_t.log()
        score = score.masked_fill(~above, float("-inf"))
        k = min(topk, int(above.sum().item()))
        top_s, top_i = torch.topk(score, k)
        w = torch.softmax(top_s, dim=0)
        return (vals_t[top_i] * w.unsqueeze(1)).sum(0)

    def decay(self, steps: int = 1):
        if self.priorities:
            factor = math.exp(-steps / FORGET_TAU)
            self.priorities = [p * factor for p in self.priorities]

    def __len__(self) -> int:
        return len(self.priorities)


# ---------------------------------------------------------------------------
# Layer E: Self State
# ---------------------------------------------------------------------------
@dataclass
class SelfState:
    """Global self-awareness summary."""
    active_ratio: float = 0.0
    bootstrap_deviation: float = 0.0
    consciousness_depth: float = 0.0
    energy: float = 0.0
    mode: str = "WAKE"
    sleep_pressure: float = 0.0
    arousal: float = 0.0
    memory_count: int = 0
    tick: int = 0

    _deviation_history: deque = field(default_factory=lambda: deque(maxlen=int(CONSCIOUSNESS_TAU)))

    def update(self, cell: CellState, mode_state: ModeState, memory: MemoryState, field: FieldState):
        self.active_ratio = cell.active_ratio
        target = torch.tensor([ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO])
        current = torch.tensor([self.active_ratio, 0.26, 1.0 - self.active_ratio - 0.26])
        self.bootstrap_deviation = float((current - target).norm().item())
        self.energy = field.energy(cell.activation)
        self.mode = mode_state.mode.value
        self.sleep_pressure = mode_state.sleep_pressure
        self.arousal = mode_state.arousal
        self.memory_count = len(memory)
        self.tick += 1

        self._deviation_history.append(self.bootstrap_deviation)
        d_tau = sum(self._deviation_history) / max(len(self._deviation_history), 1)
        self.consciousness_depth = math.exp(-CONSCIOUSNESS_CD * d_tau)


# ---------------------------------------------------------------------------
# Layer F: Agent State
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    """Agent-level state: goal, working memory, critic history."""
    goal: torch.Tensor = None
    working_memory: deque = field(default_factory=lambda: deque(maxlen=7))
    critic_history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_action: Any = None
    last_observation: Any = None
    dim: int = 0

    def __post_init__(self):
        if self.goal is None and self.dim > 0:
            self.goal = torch.zeros(self.dim)


# ---------------------------------------------------------------------------
# Basal Ganglia: Gate Automaton (the dispatcher)
# ---------------------------------------------------------------------------
class GateState(Enum):
    CLOSED = 0
    OPEN = 1


class BasalGanglia:
    """Finite state automaton that gates information flow between layers.

    go/no-go decision for each channel:
    - Which cells receive external input (Layer A gate)
    - Which memories to encode/recall (Layer D gate)
    - Whether to switch mode (Layer C gate)
    - Whether to act (Layer F gate)
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.input_gate = torch.ones(dim, dtype=torch.bool)   # Layer A: who receives input
        self.memory_gate = GateState.OPEN                      # Layer D: encode/recall
        self.mode_gate = GateState.OPEN                        # Layer C: mode switch allowed
        self.action_gate = GateState.OPEN                      # Layer F: action allowed
        self.threshold = 0.22                                  # salience threshold for gating
        self.dopamine = 0.5                                    # DA level modulates thresholds

    def compute_gates(self, salience: torch.Tensor, mode: Mode, sleep_pressure: float):
        """Update all gates based on salience and global state."""
        effective_threshold = self.threshold / (1.0 + self.dopamine)

        self.input_gate = salience > effective_threshold

        if mode == Mode.WAKE:
            self.memory_gate = GateState.OPEN
            self.action_gate = GateState.OPEN
            self.mode_gate = GateState.OPEN if sleep_pressure > 1.0 else GateState.CLOSED
        elif mode == Mode.NREM:
            self.memory_gate = GateState.OPEN
            self.action_gate = GateState.CLOSED
            self.mode_gate = GateState.OPEN if sleep_pressure < 0.45 else GateState.CLOSED
        else:  # REM
            self.memory_gate = GateState.OPEN
            self.action_gate = GateState.CLOSED
            self.mode_gate = GateState.OPEN if sleep_pressure < 0.15 else GateState.CLOSED

    def gate_input(self, external: torch.Tensor) -> torch.Tensor:
        """Apply input gate: only pass through for open channels."""
        return external * self.input_gate.float()

    def can_encode(self) -> bool:
        return self.memory_gate == GateState.OPEN

    def can_act(self) -> bool:
        return self.action_gate == GateState.OPEN

    def can_switch_mode(self) -> bool:
        return self.mode_gate == GateState.OPEN

    @property
    def open_ratio(self) -> float:
        return float(self.input_gate.float().mean().item())


# ---------------------------------------------------------------------------
# Brain Store: the complete state tree
# ---------------------------------------------------------------------------
class BrainStore:
    """Centralized state store with hierarchical layers.

    Like Redux: single source of truth, state changes only through dispatch.
    Each layer's state is isolated. Cross-layer communication goes through
    the basal ganglia gate automaton.
    """

    def __init__(self, dim: int, weight: torch.Tensor | None = None):
        self.dim = dim
        self.cell = CellState(dim)
        self.field = FieldState(dim, weight=weight)
        self.mode = ModeState()
        self.memory = MemoryState(dim)
        self.self_state = SelfState()
        self.agent = AgentState(dim=dim)
        self.gate = BasalGanglia(dim)
        self._tick = 0
        self._subscribers: list = []

    def dispatch(self, action: Action) -> dict:
        """Process an action through the state tree. Returns result dict."""
        result = {}

        if action.type == ActionType.TICK:
            result = self._tick_step(action.payload.get("external"))

        elif action.type == ActionType.EXTERNAL_INPUT:
            ext = action.payload.get("input", torch.zeros(self.dim))
            result = self._tick_step(ext)

        elif action.type == ActionType.ENCODE_MEMORY:
            key = action.payload.get("key", self.cell.activation)
            value = action.payload.get("value", self.cell.memory_trace)
            priority = action.payload.get("priority", 1.0)
            if self.gate.can_encode():
                self.memory.encode(key, value, priority)
                result["encoded"] = True
            else:
                result["encoded"] = False

        elif action.type == ActionType.RECALL_MEMORY:
            cue = action.payload.get("cue", self.cell.activation)
            result["recalled"] = self.memory.recall(cue)

        elif action.type == ActionType.MODE_SWITCH:
            if self.gate.can_switch_mode():
                new_mode = action.payload.get("mode", Mode.WAKE)
                self.mode.mode = new_mode
                result["switched"] = True
            else:
                result["switched"] = False

        elif action.type == ActionType.TEACH:
            fact_vec = action.payload.get("vector", torch.zeros(self.dim))
            reps = action.payload.get("repetitions", 3)
            for r in range(reps):
                self.memory.encode(fact_vec, fact_vec, priority=3.0 + r * 0.5)
            result["encodings"] = reps

        elif action.type == ActionType.GATE_OPEN:
            channel = action.payload.get("channel", "input")
            if channel == "memory":
                self.gate.memory_gate = GateState.OPEN
            elif channel == "action":
                self.gate.action_gate = GateState.OPEN

        elif action.type == ActionType.GATE_CLOSE:
            channel = action.payload.get("channel", "input")
            if channel == "memory":
                self.gate.memory_gate = GateState.CLOSED
            elif channel == "action":
                self.gate.action_gate = GateState.CLOSED

        self._notify(action, result)
        return result

    def _tick_step(self, external: torch.Tensor | None = None) -> dict:
        """One complete tick through all layers, bottom-up."""
        if external is None:
            external = torch.zeros(self.dim)

        mode_params = self.mode.mode.params
        external_norm = float(external.norm().item())
        self.mode.arousal = external_norm

        # Layer B: compute recurrent input
        recurrent = self.field.recurrent(self.cell.activation, self.cell.active_mask)

        # Basal ganglia: compute gates based on current salience
        salience = (
            self.cell.activation.abs()
            + 0.35 * external.abs()
            - 0.15 * self.cell.refractory
        )
        self.gate.compute_gates(salience, self.mode.mode, self.mode.sleep_pressure)

        # Gate external input
        gated_external = self.gate.gate_input(external)

        # Layer D: recall from memory
        recalled = self.memory.recall(self.cell.activation)

        # Build drive (Layer A input)
        noise_scale = mode_params["noise"]
        noise = NOISE_SIGMA * noise_scale * torch.randn(self.dim)

        drive = (
            recurrent
            + 0.45 * gated_external
            + mode_params["replay_mix"] * recalled
            - 0.35 * self.cell.refractory
            - ADAPTATION_COUPLING * self.cell.adaptation
            + noise
        )

        # Layer A: cell dynamics
        self.cell.update(drive, mode_params)

        # Layer D: encode if in wake and input is significant
        if self.mode.mode == Mode.WAKE and external_norm > NORM_EPS and self.gate.can_encode():
            priority = float(salience[self.cell.active_mask].mean().item()) if self.cell.active_count > 0 else 0.0
            self.memory.encode(self.cell.activation, self.cell.memory_trace, priority + external_norm)

        # Layer D: decay priorities
        self.memory.decay()

        # Layer C: update sleep pressure
        self.mode.update_pressure()

        # Layer C: auto mode transition
        if self.gate.can_switch_mode():
            if self.mode.mode == Mode.WAKE and self.mode.should_sleep(external_norm):
                self.mode.mode = Mode.NREM
                self.mode.nrem_cycle += 1
            elif self.mode.mode == Mode.NREM and self.mode.sleep_pressure < 0.45:
                self.mode.mode = Mode.REM
            elif self.mode.mode == Mode.REM and self.mode.should_wake(external_norm):
                self.mode.mode = Mode.WAKE

        # Layer E: update self state
        self.self_state.update(self.cell, self.mode, self.memory, self.field)

        self._tick += 1

        return {
            "tick": self._tick,
            "mode": self.mode.mode.value,
            "active_count": self.cell.active_count,
            "active_ratio": self.cell.active_ratio,
            "energy": self.field.energy(self.cell.activation),
            "sleep_pressure": self.mode.sleep_pressure,
            "consciousness": self.self_state.consciousness_depth,
            "gate_open_ratio": self.gate.open_ratio,
            "memory_count": len(self.memory),
        }

    def subscribe(self, callback):
        self._subscribers.append(callback)

    def _notify(self, action: Action, result: dict):
        for cb in self._subscribers:
            cb(action, result, self)

    def get_state(self) -> dict:
        """Full state snapshot (read-only view)."""
        return {
            "tick": self._tick,
            "cell": {
                "active_ratio": self.cell.active_ratio,
                "active_count": self.cell.active_count,
                "act_norm": float(self.cell.activation.norm().item()),
            },
            "mode": {
                "current": self.mode.mode.value,
                "sleep_pressure": self.mode.sleep_pressure,
                "circadian": self.mode.circadian_value,
                "arousal": self.mode.arousal,
            },
            "memory": {
                "count": len(self.memory),
                "capacity": self.memory.capacity,
            },
            "self": {
                "consciousness": self.self_state.consciousness_depth,
                "bootstrap_dev": self.self_state.bootstrap_deviation,
                "energy": self.self_state.energy,
            },
            "gate": {
                "input_open": self.gate.open_ratio,
                "memory": self.gate.memory_gate.name,
                "action": self.gate.action_gate.name,
                "mode_switch": self.gate.mode_gate.name,
                "dopamine": self.gate.dopamine,
            },
        }
