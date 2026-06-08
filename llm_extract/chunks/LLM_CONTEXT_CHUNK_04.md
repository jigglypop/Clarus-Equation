# LLM Context Chunk

---
## File: `reality_stone/python/reality_stone/clarus/daemon.py`

```python
"""Persistent CE brain daemon -- always running, self-learning, minimal energy.

The brain never stops. When idle, it drops to NREM/REM and consolidates.
When input arrives, it wakes. Hippocampus encodes during wake, replays during sleep.
STDP updates weights continuously. Energy budget enforces 4.87% active ceiling.

Usage:
    daemon = BrainDaemon(engine_path="reality_stone/artifacts/skt_kogpt2-base-v2.ce.pt")
    daemon.start()                     # background loop begins
    response = daemon.query("prompt")  # wake, process, return
    daemon.stop()                      # graceful shutdown with checkpoint
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import torch

try:
    from .runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode, RuntimeStep
    from .engine import CEEngine, PromptContext
    from .stdp import STDPConfig, EligibilityTracker, compute_learning_gate, apply_stdp_update
    from .neuromod import NeuromodulatorState, step_neuromodulators, apply_modulation
    from .agent import (
        ConsciousnessMonitor, WorkingMemory, CerebellumPredictor,
        compute_critic, select_action_discrete, agent_step,
    )
    from .constants import (
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        BOOTSTRAP_CONTRACTION, NOISE_SIGMA,
    )
except ImportError:
    from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode, RuntimeStep
    from reality_stone.clarus.engine import CEEngine, PromptContext
    from reality_stone.clarus.stdp import STDPConfig, EligibilityTracker, compute_learning_gate, apply_stdp_update
    from reality_stone.clarus.neuromod import NeuromodulatorState, step_neuromodulators, apply_modulation
    from reality_stone.clarus.agent import (
        ConsciousnessMonitor, WorkingMemory, CerebellumPredictor,
        compute_critic, select_action_discrete, agent_step,
    )
    from reality_stone.clarus.constants import (
        ACTIVE_RATIO, STRUCT_RATIO, BACKGROUND_RATIO,
        BOOTSTRAP_CONTRACTION, NOISE_SIGMA,
    )


@dataclass
class DaemonConfig:
    tick_ms: float = 50.0
    idle_ticks_to_sleep: int = 20
    max_wake_ticks: int = 200
    checkpoint_interval: int = 1000
    checkpoint_path: str = "brain_checkpoint.pt"
    stdp_enabled: bool = True
    stdp_interval: int = 10
    consciousness_enabled: bool = True
    log_interval: int = 100


@dataclass
class DaemonStats:
    total_ticks: int = 0
    wake_ticks: int = 0
    nrem_ticks: int = 0
    rem_ticks: int = 0
    queries_processed: int = 0
    stdp_updates: int = 0
    hippocampus_encodes: int = 0
    hippocampus_replays: int = 0
    checkpoints_saved: int = 0
    avg_active_ratio: float = 0.0
    consciousness_depth: float = 0.0
    energy_total: float = 0.0


class BrainDaemon:
    """Persistent brain runtime that runs in background, self-learns, and sleeps."""

    def __init__(
        self,
        engine_path: str,
        *,
        config: DaemonConfig | None = None,
        device: str = "cpu",
        backend: str = "torch",
    ) -> None:
        self.config = config or DaemonConfig()
        self.eng = CEEngine(engine_path, device=device, backend=backend)
        self.eng._skip_ln_for_standalone = True
        self.eng.decoder_query_blend = 0.0

        dim = self.eng.d
        rt_cfg = BrainRuntimeConfig(
            dim=dim,
            active_ratio=ACTIVE_RATIO,
            noise_sigma=NOISE_SIGMA * 0.3,
            dale_law=False,
            axon_delay=False,
            memory_capacity=128,
        )
        w = self.eng.W.detach().cpu().float()
        self.runtime = BrainRuntime(w, config=rt_cfg, backend=backend, device=device)

        self.neuro = NeuromodulatorState()
        self.consciousness = ConsciousnessMonitor()
        self.wm = WorkingMemory(capacity=7)
        self.cerebellum = CerebellumPredictor(dim=dim)

        if self.config.stdp_enabled:
            self.stdp_cfg = STDPConfig(dim=dim, spike_threshold=0.15)
            self.stdp_tracker = EligibilityTracker(self.stdp_cfg)
        else:
            self.stdp_tracker = None

        self.stats = DaemonStats()
        self._running = False
        self._thread: threading.Thread | None = None
        self._query_queue: deque[tuple[str, threading.Event, list]] = deque()
        self._idle_counter = 0
        self._prev_critic_score = 0.0
        self._tick_energies: deque[float] = deque(maxlen=100)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="brain-daemon")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._save_checkpoint()

    def query(self, prompt: str, max_tokens: int = 30, timeout: float = 10.0) -> str:
        """Send a query to the brain. Wakes it up, processes, returns response."""
        event = threading.Event()
        result: list[str] = []
        self._query_queue.append(("query", prompt, event, result, max_tokens))
        event.wait(timeout=timeout)
        return result[0] if result else ""

    def teach(self, fact: str, repetitions: int = 3, timeout: float = 15.0) -> dict:
        """Teach the brain a fact. Encodes it into hippocampus with high priority.
        Repeats encoding to strengthen the memory trace."""
        event = threading.Event()
        result: list[dict] = []
        self._query_queue.append(("teach", fact, event, result, repetitions))
        event.wait(timeout=timeout)
        return result[0] if result else {}

    def think(self, topic: str, depth: int = 5, timeout: float = 15.0) -> list[str]:
        """Ask the brain to think about a topic. Runs multiple relaxation
        cycles with self-critique, generating a chain of thoughts."""
        event = threading.Event()
        result: list[list[str]] = []
        self._query_queue.append(("think", topic, event, result, depth))
        event.wait(timeout=timeout)
        return result[0] if result else []

    def recall(self, cue: str, timeout: float = 10.0) -> str:
        """Ask the brain to recall what it knows about a cue from hippocampus."""
        event = threading.Event()
        result: list[str] = []
        self._query_queue.append(("recall", cue, event, result, 20))
        event.wait(timeout=timeout)
        return result[0] if result else ""

    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()

            if self._query_queue:
                cmd = self._query_queue.popleft()
                cmd_type = cmd[0]
                if cmd_type == "query":
                    self._handle_query(cmd[1], cmd[2], cmd[3], cmd[4])
                elif cmd_type == "teach":
                    self._handle_teach(cmd[1], cmd[2], cmd[3], cmd[4])
                elif cmd_type == "think":
                    self._handle_think(cmd[1], cmd[2], cmd[3], cmd[4])
                elif cmd_type == "recall":
                    self._handle_recall(cmd[1], cmd[2], cmd[3], cmd[4])
            else:
                self._idle_tick()

            self.stats.total_ticks += 1

            if self.stats.total_ticks % self.config.checkpoint_interval == 0:
                self._save_checkpoint()

            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, self.config.tick_ms / 1000.0 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _make_ce_args(self, steps: int = 20, noise: float = 0.003):
        import argparse
        return argparse.Namespace(
            dt=0.01, cb_weight=None, cb_topk=4, beta=1.0, steps=steps,
            backend="torch", metric_rank=0, lambda0=1.0, lambda_phi=0.5,
            lambda_var=0.25, noise_scale=noise, seed=0,
        )

    def _relax_and_generate(self, prompt: str, max_tokens: int = 30,
                            temperature: float = 0.6, steps: int = 20) -> tuple[str, torch.Tensor]:
        ids = self.eng.tok.encode(prompt, return_tensors="pt")
        m0, phi = self.eng.runtime_prompt_state(ids)

        recalled = self.runtime.hippocampus.recall(m0[:self.runtime.config.dim])
        if recalled.norm().item() > 0.01:
            blend = 0.15
            m0_dim = min(m0.shape[0], recalled.shape[0])
            m0[:m0_dim] = (1.0 - blend) * m0[:m0_dim] + blend * recalled[:m0_dim]

        ce_args = self._make_ce_args(steps=steps)
        pc = PromptContext(
            prompt=prompt, prompt_ids=ids, h_true=None,
            m0=m0, phi=phi, best_layer=0, layer_scores={0: 0.0},
        )
        rr = self.eng.relax_context(pc, ce_args)
        text, tids, _ = self.eng.standalone_generate(
            ids, rr["m_star"],
            max_tok=max_tokens, temperature=temperature,
            top_k=40, repeat_penalty=2.0,
        )
        return text, rr["m_star"].detach()

    def _handle_query(self, prompt, event, result, max_tokens) -> None:
        self._idle_counter = 0
        ext = self._encode_prompt(prompt)
        for _ in range(3):
            step = self.runtime.step(external_input=ext, force_mode=RuntimeMode.WAKE)
            self._post_step(step, ext)

        text, m_star = self._relax_and_generate(prompt, max_tokens)

        self.runtime.hippocampus.encode(
            self.runtime.activation,
            value=m_star[:self.runtime.config.dim],
            priority=2.0,
        )
        self.stats.hippocampus_encodes += 1
        self.stats.queries_processed += 1
        self.wm.append(prompt, text)

        result.append(text)
        event.set()

    def _handle_teach(self, fact, event, result, repetitions) -> None:
        """Teach: encode the fact multiple times with increasing priority.
        Each repetition strengthens the trace through spaced encoding."""
        self._idle_counter = 0
        ext = self._encode_prompt(fact)
        ids = self.eng.tok.encode(fact, return_tensors="pt")
        m0, phi = self.eng.runtime_prompt_state(ids)
        ce_args = self._make_ce_args(steps=30)

        encodings = 0
        for rep in range(max(int(repetitions), 1)):
            self.runtime.step(external_input=ext, force_mode=RuntimeMode.WAKE)

            pc = PromptContext(
                prompt=fact, prompt_ids=ids, h_true=None,
                m0=m0, phi=phi, best_layer=0, layer_scores={0: 0.0},
            )
            rr = self.eng.relax_context(pc, ce_args)
            m_star = rr["m_star"].detach()

            priority = 3.0 + rep * 0.5
            self.runtime.hippocampus.encode(
                ext[:self.runtime.config.dim],
                value=m_star[:self.runtime.config.dim],
                priority=priority,
            )
            encodings += 1

            if self.stdp_tracker is not None:
                self.stdp_tracker.update(self.runtime.activation)

        self.stats.hippocampus_encodes += encodings
        self.wm.append(f"[LEARN] {fact}", f"encoded {encodings}x")

        result.append({
            "fact": fact,
            "encodings": encodings,
            "hippocampus_size": len(self.runtime.hippocampus),
            "final_priority": priority,
        })
        event.set()

    def _handle_think(self, topic, event, result, depth) -> None:
        """Think: chain of internal relaxation cycles with self-critique.
        Each thought feeds into the next, building on previous state."""
        self._idle_counter = 0
        thoughts: list[str] = []
        ext = self._encode_prompt(topic)

        self.runtime.step(external_input=ext, force_mode=RuntimeMode.WAKE)

        current_prompt = topic
        for step_i in range(max(int(depth), 1)):
            noise = 0.005 + 0.003 * step_i
            text, m_star = self._relax_and_generate(
                current_prompt, max_tokens=15,
                temperature=0.7 + 0.05 * step_i,
                steps=25 + step_i * 5,
            )
            thought = text.strip()
            if not thought:
                thought = "(silence)"
            thoughts.append(thought)

            self.runtime.hippocampus.encode(
                m_star[:self.runtime.config.dim],
                value=self.runtime.activation.detach(),
                priority=1.5 + step_i * 0.3,
            )

            key_words = [w for w in thought.split() if len(w) > 2][:3]
            if key_words:
                current_prompt = topic + " " + " ".join(key_words)
            else:
                current_prompt = topic

            if self.stdp_tracker is not None:
                self.stdp_tracker.update(self.runtime.activation)

            recalled = self.runtime.hippocampus.recall(self.runtime.activation)
            critic = compute_critic(
                self.runtime.activation.cpu(),
                self.cerebellum.predict(),
                self.runtime.activation.cpu(),
                recalled.cpu(),
            )
            self.neuro = step_neuromodulators(
                self.neuro, c_pred=critic.c_pred * 0.05,
            )

        self.stats.queries_processed += 1
        self.wm.append(f"[THINK] {topic}", " -> ".join(thoughts))

        result.append(thoughts)
        event.set()

    def _handle_recall(self, cue, event, result, max_tokens) -> None:
        """Recall: search hippocampus for memories related to the cue."""
        self._idle_counter = 0
        ext = self._encode_prompt(cue)

        recalled = self.runtime.hippocampus.recall(ext[:self.runtime.config.dim], topk=6)
        recall_norm = float(recalled.norm().item())

        if recall_norm < 0.01:
            result.append(f"(no memory of '{cue}')")
            event.set()
            return

        ids = self.eng.tok.encode(cue, return_tensors="pt")
        m0, phi = self.eng.runtime_prompt_state(ids)
        m0_dim = min(m0.shape[0], recalled.shape[0])
        m0[:m0_dim] = 0.5 * m0[:m0_dim] + 0.5 * recalled[:m0_dim]

        ce_args = self._make_ce_args(steps=25)
        pc = PromptContext(
            prompt=cue, prompt_ids=ids, h_true=None,
            m0=m0, phi=phi, best_layer=0, layer_scores={0: 0.0},
        )
        rr = self.eng.relax_context(pc, ce_args)
        text, tids, _ = self.eng.standalone_generate(
            ids, rr["m_star"],
            max_tok=max_tokens, temperature=0.5,
            top_k=30, repeat_penalty=2.0,
        )

        self.stats.queries_processed += 1
        result.append(text)
        event.set()

    def _idle_tick(self) -> None:
        """Idle tick: brain keeps running with minimal energy."""
        self._idle_counter += 1

        if self._idle_counter > self.config.idle_ticks_to_sleep:
            if self.runtime.sleep_pressure > 0.3:
                mode = RuntimeMode.NREM
            elif self.runtime.sleep_pressure < 0.1:
                mode = RuntimeMode.REM
            else:
                mode = RuntimeMode.NREM
        else:
            mode = RuntimeMode.WAKE

        step = self.runtime.step(force_mode=mode)
        self._post_step(step, torch.zeros(self.runtime.config.dim))

    def _post_step(self, step: RuntimeStep, external: torch.Tensor) -> None:
        """Common post-step processing: STDP, neuromod, consciousness, stats."""
        if step.mode == RuntimeMode.WAKE:
            self.stats.wake_ticks += 1
        elif step.mode == RuntimeMode.NREM:
            self.stats.nrem_ticks += 1
        else:
            self.stats.rem_ticks += 1

        self._tick_energies.append(step.energy)
        self.stats.energy_total += step.energy

        active_frac = step.active_modules / max(self.runtime.config.dim, 1)
        self.stats.avg_active_ratio = (
            0.99 * self.stats.avg_active_ratio + 0.01 * active_frac
        )

        if self.stdp_tracker is not None and self.stats.total_ticks % self.config.stdp_interval == 0:
            self.stdp_tracker.update(self.runtime.activation)
            if self.stats.total_ticks % (self.config.stdp_interval * 10) == 0:
                gate = compute_learning_gate(
                    critic_score=step.energy,
                    prev_critic_score=self._prev_critic_score,
                    active_ratio=active_frac,
                )
                self._prev_critic_score = step.energy
                if abs(gate) > 0.001:
                    self.runtime.weight = apply_stdp_update(
                        self.runtime.weight, self.stdp_tracker, gate,
                        lr=0.0001, density=ACTIVE_RATIO,
                    ).to(self.runtime.device)
                    self.runtime._rebuild_sparse()
                    self.stdp_tracker.reset()
                    self.stats.stdp_updates += 1

        cb_pred = self.cerebellum.predict()
        obs = self.runtime.activation[:cb_pred.shape[0]].detach().cpu()
        self.cerebellum.update(obs)

        recalled = self.runtime.hippocampus.recall(self.runtime.activation)
        critic = compute_critic(
            self.runtime.activation.cpu(), cb_pred,
            self.runtime.activation.cpu(), recalled.cpu(),
        )
        self.neuro = step_neuromodulators(
            self.neuro,
            c_pred=critic.c_pred * 0.01,
            c_nov=critic.c_nov * 0.01,
            salience=float(external.norm().item()) * 0.1,
        )

        if self.config.consciousness_enabled:
            self.consciousness.record_deviation(active_frac)
            self.stats.consciousness_depth = self.consciousness.consciousness_depth()

        if step.mode != RuntimeMode.WAKE and len(self.runtime.hippocampus) > 0:
            self.stats.hippocampus_replays += 1

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode prompt into an external input vector."""
        ids = self.eng.tok.encode(prompt, return_tensors="pt")
        emb = self.eng.prompt_embeddings(ids)
        return emb.mean(dim=0).detach().to(self.runtime.device)[:self.runtime.config.dim]

    def _save_checkpoint(self) -> None:
        snap = self.runtime.snapshot()
        torch.save({
            "runtime": snap,
            "stats": self.stats,
            "neuro": self.neuro,
            "consciousness_history": list(self.consciousness._deviation_history),
            "wm": self.wm.contents(),
        }, self.config.checkpoint_path)
        self.stats.checkpoints_saved += 1

    def status(self) -> dict:
        return {
            "running": self._running,
            "mode": self.runtime.mode.value,
            "total_ticks": self.stats.total_ticks,
            "wake/nrem/rem": f"{self.stats.wake_ticks}/{self.stats.nrem_ticks}/{self.stats.rem_ticks}",
            "queries": self.stats.queries_processed,
            "active_ratio": f"{self.stats.avg_active_ratio:.3f}",
            "consciousness": f"{self.stats.consciousness_depth:.3f}",
            "sleep_pressure": f"{self.runtime.sleep_pressure:.4f}",
            "hippocampus": len(self.runtime.hippocampus),
            "stdp_updates": self.stats.stdp_updates,
            "energy_avg": f"{sum(self._tick_energies) / max(len(self._tick_energies), 1):.4f}",
            "neuro_da": f"{self.neuro.da:.3f}",
            "checkpoints": self.stats.checkpoints_saved,
        }
```
---
## File: `reality_stone/python/reality_stone/clarus/device.py`

```python
"""CUDA / CPU device auto-detection and selection."""

from __future__ import annotations

import torch


def auto_device(preference: str = "auto") -> torch.device:
    """Return the best available device.

    Args:
        preference: "cuda", "cpu", or "auto" (default).
            "auto" picks CUDA if available, else CPU.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preference)


def device_summary(device: torch.device) -> str:
    """One-line summary of the compute device."""
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        mem_gb = torch.cuda.get_device_properties(idx).total_mem / 1e9
        return f"CUDA:{idx} {name} ({mem_gb:.1f} GB)"
    return "CPU"
```
---
## File: `reality_stone/python/reality_stone/clarus/dimensionless.py`

```python
"""Dimensionless bookkeeping for CE gates.

CE calculations should close first on dimensionless ratios.  This module
provides small exact-rational tools for checking that rule without pulling in
symbolic math dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod
from typing import Callable, Generic, Iterable, Sequence, TypeVar


DimVector = tuple[Fraction, ...]
T = TypeVar("T")
U = TypeVar("U")


def _frac(x: int | float | Fraction) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator()


def dim(*exponents: int | float | Fraction) -> DimVector:
    """Build a dimension vector, e.g. ``dim(1, 0, -2)`` for M L^0 T^-2."""

    return tuple(_frac(x) for x in exponents)


DIMENSIONLESS: DimVector = dim(0, 0, 0, 0)
MASS: DimVector = dim(1, 0, 0, 0)
LENGTH: DimVector = dim(0, 1, 0, 0)
TIME: DimVector = dim(0, 0, 1, 0)
TEMPERATURE: DimVector = dim(0, 0, 0, 1)
ENERGY: DimVector = MASS
CURVATURE: DimVector = dim(0, -2, 0, 0)
ACTION: DimVector = dim(1, 2, -1, 0)


@dataclass(frozen=True)
class Quantity:
    """Numeric value with a base-dimension exponent vector."""

    name: str
    value: float
    dims: DimVector = DIMENSIONLESS

    @property
    def dimensionless(self) -> bool:
        return is_dimensionless(self.dims)


@dataclass(frozen=True)
class GateResult(Generic[T]):
    """Small Result/Either value for composing CE gate checks."""

    value: T | None = None
    errors: tuple[str, ...] = ()

    @classmethod
    def ok(cls, value: T) -> "GateResult[T]":
        return cls(value=value)

    @classmethod
    def fail(cls, *errors: str) -> "GateResult[T]":
        return cls(errors=tuple(error for error in errors if error))

    @property
    def passed(self) -> bool:
        return not self.errors

    def map(self, transform: Callable[[T], U]) -> "GateResult[U]":
        if not self.passed:
            return GateResult.fail(*self.errors)
        return GateResult.ok(transform(self.unwrap()))

    def bind(self, transform: Callable[[T], "GateResult[U]"]) -> "GateResult[U]":
        if not self.passed:
            return GateResult.fail(*self.errors)
        return transform(self.unwrap())

    def unwrap(self) -> T:
        if self.errors:
            raise ValueError("; ".join(self.errors))
        if self.value is None:
            raise ValueError("gate result has no value")
        return self.value


def is_dimensionless(dims: Sequence[Fraction]) -> bool:
    return all(_frac(x) == 0 for x in dims)


def same_rank_dims(quantities: Iterable[Quantity]) -> int:
    lengths = {len(q.dims) for q in quantities}
    if len(lengths) != 1:
        raise ValueError(f"inconsistent dimension ranks: {sorted(lengths)}")
    return lengths.pop()


def require_dimensionless(quantity: Quantity, *, context: str = "") -> Quantity:
    """Raise if a quantity is not dimensionless."""

    if not quantity.dimensionless:
        where = f" for {context}" if context else ""
        raise ValueError(f"{quantity.name} must be dimensionless{where}; dims={quantity.dims}")
    return quantity


def check_dimensionless(quantity: Quantity, *, context: str = "") -> GateResult[Quantity]:
    """Return a composable gate result instead of raising on dimensional failure."""

    if quantity.dimensionless:
        return GateResult.ok(quantity)
    where = f" for {context}" if context else ""
    return GateResult.fail(f"{quantity.name} must be dimensionless{where}; dims={quantity.dims}")


def audit_dimensionless(
    quantities: Iterable[Quantity],
    *,
    context: str = "",
) -> GateResult[tuple[Quantity, ...]]:
    """Validate many quantities and accumulate every dimensional violation."""

    accepted: list[Quantity] = []
    errors: list[str] = []
    for quantity in quantities:
        check = check_dimensionless(quantity, context=context)
        if check.passed:
            accepted.append(check.unwrap())
        else:
            errors.extend(check.errors)
    if errors:
        return GateResult.fail(*errors)
    return GateResult.ok(tuple(accepted))


def _rref(matrix: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    rows = [row[:] for row in matrix]
    if not rows:
        return rows, []
    n_rows, n_cols = len(rows), len(rows[0])
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        pivot = next((i for i in range(r, n_rows) if rows[i][c] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        inv = Fraction(1, 1) / rows[r][c]
        rows[r] = [x * inv for x in rows[r]]
        for i in range(n_rows):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [x - factor * y for x, y in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
        if r == n_rows:
            break
    return rows, pivots


def nullspace(matrix: Sequence[Sequence[int | float | Fraction]]) -> list[list[Fraction]]:
    """Exact rational basis for ``matrix @ x = 0``."""

    rows = [[_frac(x) for x in row] for row in matrix]
    if not rows:
        return []
    n_cols = len(rows[0])
    if any(len(row) != n_cols for row in rows):
        raise ValueError("ragged matrix")
    rref, pivots = _rref(rows)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]
    basis: list[list[Fraction]] = []
    for free in free_cols:
        vec = [Fraction(0, 1) for _ in range(n_cols)]
        vec[free] = Fraction(1, 1)
        for row_idx, pivot_col in enumerate(pivots):
            vec[pivot_col] = -rref[row_idx][free]
        basis.append(vec)
    return basis


def buckingham_pi_groups(quantities: Sequence[Quantity]) -> list[dict[str, Fraction]]:
    """Return exponent maps for independent dimensionless Pi groups."""

    if not quantities:
        return []
    rank = same_rank_dims(quantities)
    matrix = [[q.dims[row] for q in quantities] for row in range(rank)]
    groups = []
    for vec in nullspace(matrix):
        groups.append({q.name: exponent for q, exponent in zip(quantities, vec) if exponent != 0})
    return groups


def evaluate_group(quantities: Sequence[Quantity], exponents: dict[str, Fraction]) -> float:
    by_name = {q.name: q for q in quantities}
    missing = sorted(set(exponents) - set(by_name))
    if missing:
        raise KeyError(f"unknown quantities in group: {missing}")
    return prod(by_name[name].value ** float(power) for name, power in exponents.items())


def group_dimension(quantities: Sequence[Quantity], exponents: dict[str, Fraction]) -> DimVector:
    by_name = {q.name: q for q in quantities}
    rank = same_rank_dims(quantities)
    out = [Fraction(0, 1) for _ in range(rank)]
    for name, power in exponents.items():
        q = by_name[name]
        for i, exponent in enumerate(q.dims):
            out[i] += exponent * power
    return tuple(out)


def nondimensionalize(quantity: Quantity, scales: Sequence[Quantity]) -> Quantity:
    """Divide ``quantity`` by dimensional scales that span its dimension vector."""

    all_q = [*scales, quantity]
    rank = same_rank_dims(all_q)
    matrix = [[scale.dims[row] for scale in scales] for row in range(rank)]
    target = [-quantity.dims[row] for row in range(rank)]
    augmented = [row + [rhs] for row, rhs in zip(matrix, target)]
    rref, pivots = _rref(augmented)
    if len(scales) in pivots:
        raise ValueError(f"{quantity.name} dimensions cannot be spanned by supplied scales")
    powers = [Fraction(0, 1) for _ in scales]
    for row_idx, pivot_col in enumerate(pivots):
        if pivot_col < len(scales):
            powers[pivot_col] = rref[row_idx][-1]
    value = quantity.value * prod(scale.value ** float(power) for scale, power in zip(scales, powers))
    scale_part = " ".join(f"{s.name}^{p}" for s, p in zip(scales, powers) if p)
    name = f"{quantity.name}*{scale_part}" if scale_part else quantity.name
    return Quantity(name=name, value=value, dims=DIMENSIONLESS)


def exp_argument(quantity: Quantity) -> float:
    """Return a value that is safe to place in exp/log-like CE kernels."""

    require_dimensionless(quantity, context="exponential/logarithmic kernel")
    return quantity.value


def exp_arguments(quantities: Iterable[Quantity]) -> GateResult[tuple[float, ...]]:
    """Validate a batch of exp/log arguments and return their raw values."""

    return audit_dimensionless(
        quantities,
        context="exponential/logarithmic kernel",
    ).map(lambda checked: tuple(quantity.value for quantity in checked))
```
---
## File: `reality_stone/python/reality_stone/clarus/engine.py`

```python
"""CE-AGI runtime aligned with the Equation spec.

Runtime mode is standalone-only and must remain teacher-free.
Any artifact carrying a teacher clone is rejected at load time.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    from .ce_ops import (
        build_metric_basis as ce_build_metric_basis,
        pack_sparse as ce_pack_sparse,
        pq_reconstruct_tokens,
        pq_scores,
        relax_packed as ce_relax_packed,
    )
    from .constants import AD, PORTAL, BYPASS, T_WAKE, NORM_EPS
    from .research import phase_grounding_suppression
    from .utils import safe_print, normalize_vector, resolve_device
except ImportError:
    from reality_stone.clarus.ce_ops import (
        build_metric_basis as ce_build_metric_basis,
        pack_sparse as ce_pack_sparse,
        pq_reconstruct_tokens,
        pq_scores,
        relax_packed as ce_relax_packed,
    )
    from reality_stone.clarus.constants import AD, PORTAL, BYPASS, T_WAKE, NORM_EPS
    from reality_stone.clarus.research import phase_grounding_suppression
    from reality_stone.clarus.utils import safe_print, normalize_vector, resolve_device

import re as _re

_REPEATED_CHAR = _re.compile(r"(.)\1{4,}")
_REPEATED_WORD = _re.compile(r"((?:\S{2,}\s*){1,3}?)(?:\s*\1){3,}")
_MULTI_SPACE = _re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = _re.compile(r"\n{3,}")


def postprocess_output(text: str) -> str:
    """Normalize generated text from the standalone CE decoder path.

    Collapses character-level and token-level repetition, trims whitespace,
    and limits newlines so outputs from different decoders look consistent.
    """
    text = text.strip()
    text = _REPEATED_CHAR.sub(lambda m: m.group(1) * 3, text)
    text = _REPEATED_WORD.sub(lambda m: m.group(1), text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text


DEFAULT_PROMPTS = (
    "인공지능의 미래는",
    "오늘 날씨가",
    "한국어로 대답해줘",
)


def _rounded_count(total: int, ratio: float) -> int:
    return int(math.floor(max(float(ratio), 0.0) * float(total) + 0.5))


def update_phi(phi: torch.Tensor, m_star: torch.Tensor, phi_var: torch.Tensor | None = None) -> torch.Tensor:
    v = normalize_vector(m_star)
    if phi_var is not None and phi_var.numel() == phi.numel():
        var_mean = phi_var.mean().clamp(min=1e-8)
        alpha = (BYPASS * var_mean / (var_mean + 1.0)).item()
    else:
        alpha = BYPASS
    return (1 - alpha) * phi + alpha * v


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _format_optional(value) -> str:
    value = _optional_float(value)
    return "n/a" if value is None else f"{value:.4f}"


def state_partition_counts(dim: int, active_ratio: float, struct_ratio: float) -> tuple[int, int, int]:
    dim = max(0, int(dim))
    if dim == 0:
        return 0, 0, 0

    active_ratio = max(float(active_ratio), 0.0)
    struct_ratio = max(float(struct_ratio), 0.0)
    active_k = min(dim, max(1, _rounded_count(dim, active_ratio)))
    remaining = max(dim - active_k, 0)
    if remaining == 0:
        return active_k, 0, 0

    background_ratio = max(0.0, 1.0 - active_ratio - struct_ratio)
    non_active_total = struct_ratio + background_ratio
    if non_active_total <= 1e-8:
        return active_k, remaining, 0

    struct_only_ratio = struct_ratio / non_active_total
    struct_only_k = min(remaining, max(0, _rounded_count(remaining, struct_only_ratio)))
    background_k = remaining - struct_only_k
    return active_k, struct_only_k, background_k


@dataclass
class PromptContext:
    prompt: str
    prompt_ids: torch.Tensor
    h_true: torch.Tensor | None
    m0: torch.Tensor
    phi: torch.Tensor
    best_layer: int
    layer_scores: dict[int, float]


class CEEngine:
    def __init__(self, path: str, device: str = "cpu", backend: str = "torch"):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.data = data
        self.device = resolve_device(device)
        self.backend = backend

        self.model_name = data["model_name"]
        self.d = int(data["d"])
        self.vocab = int(data["vocab"])
        self.tau = float(data["tau"])
        self.portal = float(data["portal"])
        self.bypass = float(data["bypass"])
        self.t_wake = float(data["t_wake"])
        self.n_layer = int(data["n_layer"])
        self.h_norm_ref = float(data.get("hidden_norm_ref", 50.0))
        self.decoder_prev_scale = float(data.get("decoder_prev_scale", 0.35))
        self.decoder_prev_proj = None
        if data.get("decoder_prev_proj") is not None:
            self.decoder_prev_proj = data["decoder_prev_proj"].float().to(self.device)
        self.decoder_state_proj = None
        if data.get("decoder_state_proj") is not None:
            self.decoder_state_proj = data["decoder_state_proj"].float().to(self.device)
        self.decoder_query_bias = None
        if data.get("decoder_query_bias") is not None:
            self.decoder_query_bias = data["decoder_query_bias"].float().to(self.device)
        self.decoder_vocab_weight = None
        if data.get("decoder_vocab_weight") is not None:
            self.decoder_vocab_weight = data["decoder_vocab_weight"].float().to(self.device)
        self.decoder_vocab_bias = None
        if data.get("decoder_vocab_bias") is not None:
            self.decoder_vocab_bias = data["decoder_vocab_bias"].float().to(self.device)
        self.decoder_vocab_scale = float(data.get("decoder_vocab_scale", 1.0))
        self.decoder_token_ids = None
        if data.get("decoder_token_ids") is not None:
            self.decoder_token_ids = data["decoder_token_ids"].long().to(self.device)
        self.decoder_token_state_proj = None
        if data.get("decoder_token_state_proj") is not None:
            self.decoder_token_state_proj = data["decoder_token_state_proj"].float().to(self.device)
        self.decoder_token_prev_proj = None
        if data.get("decoder_token_prev_proj") is not None:
            self.decoder_token_prev_proj = data["decoder_token_prev_proj"].float().to(self.device)
        self.decoder_token_bias = None
        if data.get("decoder_token_bias") is not None:
            self.decoder_token_bias = data["decoder_token_bias"].float().to(self.device)
        self.decoder_token_scale = float(data.get("decoder_token_scale", 1.0))
        self.decoder_query_blend = float(data.get("decoder_query_blend", 0.7))
        self.decoder_candidate_ratio = float(data.get("decoder_candidate_ratio", 0.04865))
        self.curvature_alpha = float(data.get("curvature_alpha", 1.5))
        self.curvature_lambda = float(data.get("curvature_lambda", 1.25))
        self.curvature_steepness = float(data.get("curvature_steepness", 8.0))
        self.curvature_eval_topk = int(data.get("curvature_eval_topk", 256))
        self.phase_grounding_lambda = float(data.get("phase_grounding_lambda", 0.25))
        self.data["phase_grounding_lambda"] = float(self.phase_grounding_lambda)
        self.repeat_window = int(data.get("repeat_window", 16))
        self.repeat_ngram = int(data.get("repeat_ngram", 3))
        self._terminal_ids_cache = None
        self.active_ratio = float(data.get("active_ratio", 0.0487))
        self.struct_ratio = float(data.get("struct_ratio", 0.2623))
        self.wake_ratio = float(data.get("wake_ratio", 0.6891))
        self.nrem_ratio = float(data.get("nrem_ratio", 0.2623))
        self.rem_ratio = float(data.get("rem_ratio", 0.0487))
        self.target_w_density = float(data.get("target_w_density", 0.0316))
        self.sparsity_radius = float(data.get("r_c", math.pi))
        self.active_dim_mask = None
        if data.get("active_dim_mask") is not None:
            self.active_dim_mask = data["active_dim_mask"].bool().to(self.device)
        self.struct_dim_mask = None
        if data.get("struct_dim_mask") is not None:
            self.struct_dim_mask = data["struct_dim_mask"].bool().to(self.device)
        self.background_dim_mask = None
        if data.get("background_dim_mask") is not None:
            self.background_dim_mask = data["background_dim_mask"].bool().to(self.device)
        self._state_graph_laplacian = None
        self._state_coords = None

        self.W = data["W"].float().to(self.device)
        self.W_pack = self._load_w_pack(data)
        self._dense_relax_w = None
        if self.W_pack[0].numel() == self.W.numel():
            self._dense_relax_w = self.W
        emb_weight = data.get("emb_weight")
        self.emb = emb_weight.float().to(self.device) if emb_weight is not None else None
        # Vocab pruning (V1): emb stores top-K rows in compact id space [0, K).
        # vocab_id_map maps a global tokenizer id -> compact id (or -1 if pruned).
        # kept_token_ids[k] is the global id of compact row k. unk_emb is the
        # fallback embedding used when a global id has been pruned.
        self.kept_token_ids = None
        self.vocab_id_map = None
        self.unk_emb = None
        if data.get("kept_token_ids") is not None and data.get("vocab_id_map") is not None:
            self.kept_token_ids = data["kept_token_ids"].long().to(self.device)
            self.vocab_id_map = data["vocab_id_map"].long().to(self.device)
            if data.get("pruned_unk_emb") is not None:
                self.unk_emb = data["pruned_unk_emb"].float().to(self.device)
            elif self.emb is not None:
                self.unk_emb = self.emb.mean(dim=0)
        pos_weight = data.get("pos_weight")
        self.pos = pos_weight.float().to(self.device) if pos_weight is not None else None
        self.ln_w = data["ln_f_weight"].float().to(self.device)
        self.ln_b = data["ln_f_bias"].float().to(self.device)
        self.context_first_proj = None
        if data.get("context_first_proj") is not None:
            self.context_first_proj = data["context_first_proj"].float().to(self.device)
        self.context_prev_proj = None
        if data.get("context_prev_proj") is not None:
            self.context_prev_proj = data["context_prev_proj"].float().to(self.device)
        self.context_last_proj = None
        if data.get("context_last_proj") is not None:
            self.context_last_proj = data["context_last_proj"].float().to(self.device)
        self.context_mean_proj = None
        if data.get("context_mean_proj") is not None:
            self.context_mean_proj = data["context_mean_proj"].float().to(self.device)
        self.context_decay_proj = None
        if data.get("context_decay_proj") is not None:
            self.context_decay_proj = data["context_decay_proj"].float().to(self.device)
        self.context_phi_proj = None
        if data.get("context_phi_proj") is not None:
            self.context_phi_proj = data["context_phi_proj"].float().to(self.device)
        self.context_len_proj = None
        if data.get("context_len_proj") is not None:
            self.context_len_proj = data["context_len_proj"].float().to(self.device)
        self.context_bias = None
        if data.get("context_bias") is not None:
            self.context_bias = data["context_bias"].float().to(self.device)
        self.allow_pretrained_fallback = bool(data.get("allow_pretrained_fallback", True))
        self.pq_centroids = None
        self.pq_codes = None
        if data.get("pq_centroids") is not None and data.get("pq_codes") is not None:
            self.pq_centroids = data["pq_centroids"].to(self.device)
            self.pq_codes = data["pq_codes"].to(self.device)

        self._stored_eigvecs = data.get("W_eigvecs")
        if self._stored_eigvecs is not None:
            self._stored_eigvecs = self._stored_eigvecs.float()
        self._eigvec_cache: dict[int, torch.Tensor] = {}
        if self.target_w_density > 0.0:
            self.apply_relax_matrix(self.W.detach().cpu())

        self.model = None
        self.tok = None
        self.pad_token_id = data.get("pad_token_id")
        self.eos_token_id = data.get("eos_token_id")
        self.model_memory_bytes = 0
        self._load_model()
        if self.active_dim_mask is None or self.struct_dim_mask is None:
            seed = None
            if self.decoder_state_proj is not None:
                seed = self.decoder_state_proj.abs().mean(dim=1)
            elif self.W is not None:
                seed = self.W.abs().mean(dim=1)
            if seed is not None:
                active_mask, struct_mask, _ = self.state_partition(seed, use_stored=False)
                self.apply_state_partition(active_mask, struct_mask)
        self._compress_runtime_projections()

    def _load_w_pack(self, data):
        values = data.get("W_values")
        col_idx = data.get("W_col_idx")
        row_ptr = data.get("W_row_ptr")
        if values is None or col_idx is None or row_ptr is None:
            values, col_idx, row_ptr = ce_pack_sparse(data["W"].float(), backend="torch")
        return (
            values.to(self.device),
            col_idx.to(self.device),
            row_ptr.to(self.device),
        )

    def _load_model(self):
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast

        clone_config = self.data.get("clone_config")
        clone_state = self.data.get("clone_state")
        clone_kind = self.data.get("clone_kind")
        tokenizer_json = self.data.get("tokenizer_json")
        tokenizer_specials = self.data.get("tokenizer_specials") or {}

        if tokenizer_json is not None:
            backend_tok = Tokenizer.from_str(tokenizer_json)
            tok_kwargs = {k: v for k, v in tokenizer_specials.items() if v is not None}
            self.tok = PreTrainedTokenizerFast(tokenizer_object=backend_tok, **tok_kwargs)
        else:
            raise RuntimeError("runtime artifact must embed tokenizer_json and may not fall back to pretrained assets")

        if clone_config is not None or clone_state is not None or clone_kind is not None:
            raise RuntimeError("teacher-bearing artifact is forbidden in runtime mode; rebuild as runtime-only")
        if self.allow_pretrained_fallback:
            raise RuntimeError("pretrained fallback is forbidden in runtime mode")

        self.model = None
        self.model_source = "runtime"
        self.model_memory_bytes = 0
        if self.eos_token_id is None:
            self.eos_token_id = self.tok.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tok.pad_token_id
        self.inject_layer = self.n_layer // 2

    def _build_state_graph_laplacian(self) -> torch.Tensor:
        coords = self.state_coords()
        dist = torch.cdist(coords, coords)
        adj = (dist > 0) & (dist <= 1.01)
        deg = adj.sum(dim=1).float()
        lap = -adj.float()
        lap[torch.arange(self.d, device=self.device), torch.arange(self.d, device=self.device)] = deg
        return lap

    def _build_state_coords(self) -> torch.Tensor:
        side = int(math.ceil(self.d ** (1.0 / 3.0)))
        idx = torch.arange(self.d, device=self.device, dtype=torch.long)
        x = idx // (side * side)
        y = (idx // side) % side
        z = idx % side
        return torch.stack([x, y, z], dim=1).float()

    def state_coords(self) -> torch.Tensor:
        if self._state_coords is None:
            self._state_coords = self._build_state_coords()
        return self._state_coords

    def state_graph_laplacian(self) -> torch.Tensor:
        if self._state_graph_laplacian is None:
            self._state_graph_laplacian = self._build_state_graph_laplacian()
        return self._state_graph_laplacian

    def weight_density(self, w: torch.Tensor | None = None) -> float:
        mat = self.W if w is None else w
        dim = int(mat.shape[0])
        if dim <= 1:
            return 0.0
        off = mat.detach().clone()
        off.fill_diagonal_(0)
        return float((off != 0).sum().item()) / float(dim * (dim - 1))

    def resparsify_relax_matrix(
        self,
        w: torch.Tensor,
        *,
        target_density: float | None = None,
        radius: float | None = None,
    ) -> torch.Tensor:
        target = self.target_w_density if target_density is None else float(target_density)
        if target <= 0.0 or target >= 1.0:
            return 0.5 * (w + w.T)

        w_cpu = 0.5 * (w.detach().cpu().float() + w.detach().cpu().float().T)
        dim = int(w_cpu.shape[0])
        diag = torch.diag(w_cpu).clone()
        radius_val = self.sparsity_radius if radius is None else float(radius)
        coords = self.state_coords().detach().cpu()
        dist = torch.cdist(coords, coords)
        candidate = (dist > 0) & (dist <= radius_val)
        upper = torch.triu(candidate, diagonal=1)
        pair_idx = upper.nonzero(as_tuple=False)
        if pair_idx.numel() == 0:
            sparse = torch.diag(diag)
            return sparse

        keep_edges = max(1, int(round(target * dim * (dim - 1) / 2)))
        keep_edges = min(keep_edges, int(pair_idx.shape[0]))
        pair_scores = w_cpu[pair_idx[:, 0], pair_idx[:, 1]].abs()
        top_idx = torch.topk(pair_scores, keep_edges).indices
        chosen = pair_idx.index_select(0, top_idx)
        mask = torch.zeros_like(w_cpu, dtype=torch.bool)
        mask[chosen[:, 0], chosen[:, 1]] = True
        mask = mask | mask.T
        sparse = torch.zeros_like(w_cpu)
        sparse[mask] = w_cpu[mask]
        sparse[torch.arange(dim), torch.arange(dim)] = diag
        return sparse

    def apply_relax_matrix(self, w: torch.Tensor):
        w_cpu = w.detach().cpu().float()
        w_sym = self.resparsify_relax_matrix(0.5 * (w_cpu + w_cpu.T))
        eigvals = torch.linalg.eigvalsh(w_sym)
        lam_max = float(eigvals[-1].item())
        if lam_max >= -1e-4:
            shift = lam_max + 1e-3
            w_sym = w_sym - shift * torch.eye(w_sym.shape[0], dtype=w_sym.dtype)
        self.data["W"] = w_sym
        values, col_idx, row_ptr = ce_pack_sparse(w_sym, backend="torch")
        self.data["W_values"] = values.cpu()
        self.data["W_col_idx"] = col_idx.cpu()
        self.data["W_row_ptr"] = row_ptr.cpu()
        self.data["W_eigvecs"] = None
        self.W = w_sym.to(self.device)
        self.W_pack = (
            values.to(self.device),
            col_idx.to(self.device),
            row_ptr.to(self.device),
        )
        self._dense_relax_w = self.W if values.numel() == self.W.numel() else None
        self._stored_eigvecs = None
        self._eigvec_cache.clear()

    def build_brain_runtime(
        self,
        *,
        active_ratio: float | None = None,
        backend: str | None = None,
    ):
        from reality_stone.clarus.runtime import BrainRuntime, BrainRuntimeConfig

        runtime_cfg = BrainRuntimeConfig(
            dim=self.d,
            active_ratio=self.active_ratio if active_ratio is None else float(active_ratio),
        )
        runtime_backend = "auto" if backend is None else backend
        return BrainRuntime(
            self.W.detach().cpu(),
            config=runtime_cfg,
            backend=runtime_backend,
            device=self.device,
        )

    def apply_state_partition(
        self,
        active_mask: torch.Tensor,
        struct_mask: torch.Tensor,
    ):
        active_mask = active_mask.bool().detach().cpu()
        struct_mask = struct_mask.bool().detach().cpu()
        if active_mask.shape != struct_mask.shape:
            raise ValueError("state partition masks must share shape")
        struct_mask = struct_mask | active_mask
        background_mask = ~(struct_mask)
        self.data["active_dim_mask"] = active_mask
        self.data["struct_dim_mask"] = struct_mask
        self.data["background_dim_mask"] = background_mask
        self.data["active_ratio"] = float(self.active_ratio)
        self.data["struct_ratio"] = float(self.struct_ratio)
        self.active_dim_mask = active_mask.to(self.device)
        self.struct_dim_mask = struct_mask.to(self.device)
        self.background_dim_mask = background_mask.to(self.device)

    def active_indices(self) -> torch.Tensor | None:
        if self.active_dim_mask is None:
            return None
        idx = torch.nonzero(self.active_dim_mask, as_tuple=False).squeeze(1)
        return idx if idx.numel() else None

    def struct_indices(self) -> torch.Tensor | None:
        if self.struct_dim_mask is None:
            return self.active_indices()
        idx = torch.nonzero(self.struct_dim_mask, as_tuple=False).squeeze(1)
        return idx if idx.numel() else None

    def _projection_indices(self) -> torch.Tensor | None:
        idx = self.struct_indices()
        return idx if idx is not None else self.active_indices()

    def _compress_state_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:
        if proj is None:
            return None
        proj_idx = self._projection_indices()
        if proj_idx is None:
            return proj.float().to(self.device)
        if proj.ndim == 2 and proj.shape[0] == self.d and proj.shape[1] == self.d:
            idx = proj_idx.to(proj.device)
            return proj.index_select(0, idx).index_select(1, idx).float().to(self.device)
        return proj.float().to(self.device)

    def _compress_prev_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:
        if proj is None:
            return None
        proj_idx = self._projection_indices()
        if proj_idx is None:
            return proj.float().to(self.device)
        if proj.ndim == 2 and proj.shape[0] == self.d and proj.shape[1] == self.d:
            idx = proj_idx.to(proj.device)
            return proj.index_select(1, idx).float().to(self.device)
        return proj.float().to(self.device)

    def _compress_token_state_proj(self, proj: torch.Tensor | None) -> torch.Tensor | None:
        if proj is None:
            return None
        proj_idx = self._projection_indices()
        if proj_idx is None:
            return proj.float().to(self.device)
        if proj.ndim == 2 and proj.shape[0] == self.d:
            idx = proj_idx.to(proj.device)
            return proj.index_select(0, idx).float().to(self.device)
        return proj.float().to(self.device)

    def _compress_runtime_projections(self):
        if self.decoder_state_proj is not None:
            if self.decoder_state_proj.shape[0] == self.d and self.decoder_state_proj.shape[1] == self.d:
                self.decoder_state_proj = self.decoder_state_proj.float().to(self.device)
            else:
                self.decoder_state_proj = self._compress_state_proj(self.decoder_state_proj.detach().cpu())
                self.data["decoder_state_proj"] = self.decoder_state_proj.detach().cpu()
        if self.decoder_prev_proj is not None:
            if self.decoder_prev_proj.shape[0] == self.d and self.decoder_prev_proj.shape[1] == self.d:
                self.decoder_prev_proj = self.decoder_prev_proj.float().to(self.device)
            else:
                self.decoder_prev_proj = self._compress_prev_proj(self.decoder_prev_proj.detach().cpu())
                self.data["decoder_prev_proj"] = self.decoder_prev_proj.detach().cpu()
        if self.decoder_token_state_proj is not None:
            self.decoder_token_state_proj = self._compress_token_state_proj(self.decoder_token_state_proj.detach().cpu())
            self.data["decoder_token_state_proj"] = self.decoder_token_state_proj.detach().cpu()

    def state_partition(
        self,
        x: torch.Tensor,
        *,
        use_stored: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if use_stored and self.active_dim_mask is not None and self.struct_dim_mask is not None:
            active_mask = self.active_dim_mask
            struct_mask = self.struct_dim_mask
            background_mask = (
                self.background_dim_mask
                if self.background_dim_mask is not None
                else ~struct_mask
            )
            return active_mask, struct_mask, background_mask

        dim = x.numel()
        active_k, struct_only_k, _ = state_partition_counts(dim, self.active_ratio, self.struct_ratio)
        struct_k = min(dim, active_k + struct_only_k)
        scores = x.detach().abs()
        active_idx = torch.topk(scores, active_k).indices
        struct_idx = torch.topk(scores, struct_k).indices
        active_mask = torch.zeros(dim, dtype=torch.bool, device=x.device)
        struct_mask = torch.zeros(dim, dtype=torch.bool, device=x.device)
        active_mask[active_idx] = True
        struct_mask[struct_idx] = True
        struct_mask = struct_mask | active_mask
        background_mask = ~struct_mask
        return active_mask, struct_mask, background_mask

    def masked_state(
        self,
        x: torch.Tensor,
        *,
        include_struct: bool = False,
        use_stored: bool = True,
    ) -> torch.Tensor:
        active_mask, struct_mask, _ = self.state_partition(x, use_stored=use_stored)
        mask = struct_mask if include_struct else active_mask
        return x * mask.to(dtype=x.dtype)

    def _project_state_query(self, state_hidden: torch.Tensor) -> torch.Tensor:
        proj_idx = self._projection_indices()
        if (
            self.decoder_state_proj is not None
            and proj_idx is not None
            and self.decoder_state_proj.ndim == 2
            and self.decoder_state_proj.shape[0] == proj_idx.numel()
            and self.decoder_state_proj.shape[1] == proj_idx.numel()
        ):
            state_in = state_hidden.index_select(-1, proj_idx)
            state_out = state_in @ self.decoder_state_proj
            query = torch.zeros_like(state_hidden)
            query[..., proj_idx] = state_out
            return query
        if self.decoder_state_proj is not None:
            return state_hidden @ self.decoder_state_proj
        return state_hidden

    def _project_prev_query(self, prev_emb: torch.Tensor) -> torch.Tensor:
        proj_idx = self._projection_indices()
        if (
            self.decoder_prev_proj is not None
            and proj_idx is not None
            and self.decoder_prev_proj.ndim == 2
            and self.decoder_prev_proj.shape[0] == self.d
            and self.decoder_prev_proj.shape[1] == proj_idx.numel()
        ):
            prev_out = prev_emb @ self.decoder_prev_proj
            query = torch.zeros_like(prev_emb)
            query[..., proj_idx] = prev_out
            return query
        if self.decoder_prev_proj is not None:
            return prev_emb @ self.decoder_prev_proj
        return prev_emb

    def _get_w_eigvecs(self, metric_rank: int) -> torch.Tensor | None:
        hess_rank = min(metric_rank // 2, 8)
        if hess_rank <= 0:
            return None
        if self._stored_eigvecs is not None and self._stored_eigvecs.shape[0] >= hess_rank:
            return self._stored_eigvecs[:hess_rank].to(self.device)
        if hess_rank not in self._eigvec_cache:
            _, eigvecs = torch.linalg.eigh(self.W.cpu())
            self._eigvec_cache[hess_rank] = eigvecs[:, :hess_rank].T.contiguous().to(self.device)
        return self._eigvec_cache[hess_rank]

    def memory_usage(self) -> dict[str, float]:
        values, col_idx, row_ptr = self.W_pack
        w_dense_bytes = self.W.numel() * self.W.element_size()
        w_packed_bytes = (
            values.numel() * values.element_size()
            + col_idx.numel() * col_idx.element_size()
            + row_ptr.numel() * row_ptr.element_size()
        )
        w_layers_bytes = sum(
            w.numel() * w.element_size() for w in self.data.get("W_layers", [])
        )
        emb_bytes = 0 if self.emb is None else self.emb.numel() * self.emb.element_size()
        pos_bytes = 0 if self.pos is None else self.pos.numel() * self.pos.element_size()
        pq_bytes = 0
        if self.pq_centroids is not None and self.pq_codes is not None:
            pq_bytes = (
                self.pq_centroids.numel() * self.pq_centroids.element_size()
                + self.pq_codes.numel() * self.pq_codes.element_size()
            )
        clone_artifact_bytes = 0
        clone_state = self.data.get("clone_state")
        if clone_state is not None:
            clone_artifact_bytes = sum(
                value.numel() * value.element_size() for value in clone_state.values()
            )
        prev_proj_bytes = (
            0 if self.decoder_prev_proj is None
            else self.decoder_prev_proj.numel() * self.decoder_prev_proj.element_size()
        )
        state_proj_bytes = (
            0 if self.decoder_state_proj is None
            else self.decoder_state_proj.numel() * self.decoder_state_proj.element_size()
        )
        query_bias_bytes = (
            0 if self.decoder_query_bias is None
            else self.decoder_query_bias.numel() * self.decoder_query_bias.element_size()
        )
        vocab_head_bytes = 0
        for tensor in (self.decoder_vocab_weight, self.decoder_vocab_bias):
            if tensor is not None:
                vocab_head_bytes += tensor.numel() * tensor.element_size()
        context_bytes = 0
        for tensor in (
            self.context_first_proj,
            self.context_prev_proj,
            self.context_last_proj,
            self.context_mean_proj,
            self.context_decay_proj,
            self.context_phi_proj,
            self.context_len_proj,
            self.context_bias,
        ):
            if tensor is not None:
                context_bytes += tensor.numel() * tensor.element_size()
        token_head_bytes = 0
        for tensor in (
            self.decoder_token_ids,
            self.decoder_token_state_proj,
            self.decoder_token_prev_proj,
            self.decoder_token_bias,
        ):
            if tensor is not None:
                token_head_bytes += tensor.numel() * tensor.element_size()
        partition_bytes = 0
        for tensor in (
            self.active_dim_mask,
            self.struct_dim_mask,
            self.background_dim_mask,
        ):
            if tensor is not None:
                partition_bytes += tensor.numel() * tensor.element_size()
        ln_bytes = (self.ln_w.numel() + self.ln_b.numel()) * self.ln_w.element_size()
        runtime_core = (
            w_packed_bytes
            + ln_bytes
            + context_bytes
            + prev_proj_bytes
            + state_proj_bytes
            + query_bias_bytes
            + vocab_head_bytes
            + token_head_bytes
            + partition_bytes
        )
        runtime_total = runtime_core + emb_bytes + pos_bytes + pq_bytes
        file_core = (
            w_dense_bytes
            + w_layers_bytes
            + ln_bytes
            + context_bytes
            + prev_proj_bytes
            + state_proj_bytes
            + query_bias_bytes
            + vocab_head_bytes
            + token_head_bytes
            + partition_bytes
        )
        file_total = file_core + emb_bytes + pos_bytes + pq_bytes + clone_artifact_bytes
        return {
            "W_dense_MB": w_dense_bytes / 1024 / 1024,
            "W_packed_MB": w_packed_bytes / 1024 / 1024,
            "W_layers_MB": w_layers_bytes / 1024 / 1024,
            "W_target_density_pct": self.target_w_density * 100.0,
            "W_offdiag_density_pct": self.weight_density() * 100.0,
            "Embedding_MB": emb_bytes / 1024 / 1024,
            "Positional_MB": pos_bytes / 1024 / 1024,
            "PQ_MB": pq_bytes / 1024 / 1024,
            "CloneArtifact_MB": clone_artifact_bytes / 1024 / 1024,
            "ContextProj_MB": context_bytes / 1024 / 1024,
            "PrevProj_MB": prev_proj_bytes / 1024 / 1024,
            "StateProj_MB": state_proj_bytes / 1024 / 1024,
            "QueryBias_MB": query_bias_bytes / 1024 / 1024,
            "VocabHead_MB": vocab_head_bytes / 1024 / 1024,
            "TokenHead_MB": token_head_bytes / 1024 / 1024,
            "StateMask_KB": partition_bytes / 1024,
            "ln_f_KB": ln_bytes / 1024,
            "runtime_core_MB": runtime_core / 1024 / 1024,
            "runtime_total_MB": runtime_total / 1024 / 1024,
            "file_total_MB": file_total / 1024 / 1024,
            "model_MB": self.model_memory_bytes / 1024 / 1024,
        }

    def save_artifact(self, path: str):
        torch.save(self.data, path)

    def save_runtime_artifact(self, path: str):
        runtime = dict(self.data)
        for key in ("clone_state", "clone_config", "clone_kind"):
            runtime.pop(key, None)
        runtime["allow_pretrained_fallback"] = False
        torch.save(runtime, path)

    def has_standalone_lexicon(self) -> bool:
        return self.emb is not None or (
            self.pq_centroids is not None and self.pq_codes is not None
        )

    def prompt_embeddings(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        token_ids = prompt_ids.to(device=self.device, dtype=torch.long).view(-1)
        emb = self.token_embedding(token_ids).view(prompt_ids.shape[1], self.d)
        if self.pos is not None:
            pos_idx = torch.arange(prompt_ids.shape[1], device=self.device, dtype=torch.long)
            pos_idx = pos_idx.clamp_max(self.pos.shape[0] - 1)
            emb = emb + self.pos.index_select(0, pos_idx)
        return emb

    def runtime_prompt_state(
        self,
        prompt_ids: torch.Tensor,
        *,
        phi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb_seq = self.prompt_embeddings(prompt_ids)
        first_emb = emb_seq[0]
        prev_emb = emb_seq[-2] if emb_seq.shape[0] > 1 else emb_seq[-1]
        last_emb = emb_seq[-1]
        mean_emb = emb_seq.mean(dim=0)
        weights = torch.arange(1, emb_seq.shape[0] + 1, device=self.device, dtype=emb_seq.dtype).unsqueeze(1)
        decay_emb = (emb_seq * weights).sum(dim=0) / weights.sum().clamp_min(1.0)
        phi_base = (
            torch.zeros_like(last_emb)
            if emb_seq.shape[0] <= 1
            else normalize_vector(emb_seq[:-1].mean(dim=0) - emb_seq[-1])
        )
        len_ratio = float(min(emb_seq.shape[0], 0 if self.pos is None else self.pos.shape[0]) or emb_seq.shape[0])
        if self.pos is not None and self.pos.shape[0] > 0:
            len_ratio /= float(self.pos.shape[0])
        else:
            len_ratio = 1.0
        state = torch.zeros_like(last_emb) if self.context_bias is None else self.context_bias.clone()
        if self.context_first_proj is not None:
            state = state + first_emb @ self.context_first_proj
        if self.context_prev_proj is not None:
            state = state + prev_emb @ self.context_prev_proj
        if self.context_last_proj is not None:
            state = state + last_emb @ self.context_last_proj
        else:
            state = state + last_emb
        if self.context_mean_proj is not None:
            state = state + mean_emb @ self.context_mean_proj
        if self.context_decay_proj is not None:
            state = state + decay_emb @ self.context_decay_proj
        if self.context_phi_proj is not None:
            state = state + phi_base @ self.context_phi_proj
        if self.context_len_proj is not None:
            state = state + len_ratio * self.context_len_proj
        return state, (phi_base if phi is None else phi.detach().float().to(self.device))

    def token_embedding(self, token_ids: int | list[int] | torch.Tensor) -> torch.Tensor:
        if self.emb is not None:
            if not torch.is_tensor(token_ids):
                token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
            token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
            if self.vocab_id_map is None:
                return self.emb.index_select(0, token_ids)
            compact_ids = self.vocab_id_map.index_select(0, token_ids)
            kept = compact_ids >= 0
            safe = compact_ids.clamp_min(0)
            out = self.emb.index_select(0, safe)
            if not bool(kept.all().item()):
                fallback = self.unk_emb if self.unk_emb is not None else torch.zeros_like(out[0])
                out = torch.where(kept.unsqueeze(1), out, fallback.unsqueeze(0).expand_as(out))
            return out
        if self.pq_centroids is not None and self.pq_codes is not None:
            if not torch.is_tensor(token_ids):
                token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
            token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
            return pq_reconstruct_tokens(
                self.pq_centroids,
                self.pq_codes,
                token_ids,
            ).to(self.device)
        raise RuntimeError("No lexical memory is available for token embedding lookup")

    def lexical_scores(self, query: torch.Tensor) -> torch.Tensor:
        if self.emb is not None:
            scores = self.emb @ query
            if self.kept_token_ids is None:
                return scores
            full = torch.full((self.vocab,), float("-inf"), dtype=scores.dtype, device=scores.device)
            full.index_copy_(0, self.kept_token_ids, scores)
            return full
        if self.pq_centroids is not None and self.pq_codes is not None:
            return pq_scores(query, self.pq_centroids, self.pq_codes)
        raise RuntimeError("No lexical memory is available for scoring")

    @staticmethod
    def _rescale_to_reference(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if query.ndim == 1:
            q_norm = float(query.norm().item())
            r_norm = float(reference.norm().item())
            if not math.isfinite(q_norm) or not math.isfinite(r_norm) or q_norm <= 1e-8 or r_norm <= 1e-8:
                return query
            return query * (r_norm / q_norm)
        q_norm = query.norm(dim=-1, keepdim=True)
        r_norm = reference.norm(dim=-1, keepdim=True)
        valid = torch.isfinite(q_norm) & torch.isfinite(r_norm) & (q_norm > 1e-8) & (r_norm > 1e-8)
        scale = torch.ones_like(q_norm)
        scale = torch.where(valid, r_norm / q_norm.clamp_min(1e-8), scale)
        return query * scale

    def decoder_query(self, state_hidden: torch.Tensor, prev_emb: torch.Tensor) -> torch.Tensor:
        state_query = self._project_state_query(state_hidden.float())
        prev_query = self._project_prev_query(prev_emb.float())
        query = state_query + float(self.decoder_prev_scale) * prev_query
        if self.decoder_query_bias is not None:
            query = query + self.decoder_query_bias

        blend = float(self.decoder_query_blend)
        if blend <= 0.0:
            query = state_hidden.float()
        elif blend < 1.0:
            query = blend * query + (1.0 - blend) * state_hidden.float()

        query = self._rescale_to_reference(query, state_hidden.float())
        finite_mask = torch.isfinite(query).all(dim=-1) if query.ndim > 1 else torch.isfinite(query).all()
        if query.ndim == 1:
            if not bool(finite_mask.item()) or query.abs().sum().item() <= 1e-8:
                return state_hidden.float()
            return query
        fallback = state_hidden.float()
        valid_rows = finite_mask & (query.abs().sum(dim=-1) > 1e-8)
        return torch.where(valid_rows.unsqueeze(-1), query, fallback)

    @staticmethod
    def _normalize_logits(logits: torch.Tensor) -> torch.Tensor:
        scores = logits.float()
        if scores.numel() <= 1:
            return scores.clone()
        mean = scores.mean()
        std = scores.std(unbiased=False).clamp_min(1e-6)
        return (scores - mean) / std

    def _merge_candidate_ids(self, *groups: torch.Tensor | None) -> torch.Tensor:
        merged: list[int] = []
        seen: set[int] = set()
        for group in groups:
            if group is None or group.numel() == 0:
                continue
            for token_id in group.reshape(-1).tolist():
                token_int = int(token_id)
                if token_int < 0 or token_int >= self.vocab or token_int in seen:
                    continue
                seen.add(token_int)
                merged.append(token_int)
        if not merged:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.tensor(merged, dtype=torch.long, device=self.device)

    def _sentence_terminal_ids(self) -> torch.Tensor:
        if self._terminal_ids_cache is not None:
            return self._terminal_ids_cache
        terminal_ids: list[int] = []
        if self.eos_token_id is not None:
            terminal_ids.append(int(self.eos_token_id))
        if self.tok is not None:
            for token_text in (".", "!", "?", "다", "요", "죠", "네", "까", "니다"):
                try:
                    token_ids = self.tok.encode(token_text, add_special_tokens=False)
                except TypeError:
                    token_ids = self.tok.encode(token_text)
                if len(token_ids) == 1:
                    terminal_ids.append(int(token_ids[0]))
        if terminal_ids:
            cache_src = torch.tensor(terminal_ids, dtype=torch.long, device=self.device)
            self._terminal_ids_cache = self._merge_candidate_ids(cache_src)
        else:
            self._terminal_ids_cache = torch.empty(0, dtype=torch.long, device=self.device)
        return self._terminal_ids_cache

    def _sentence_close_bonus(self, candidate_ids: torch.Tensor, *, generated_len: int) -> torch.Tensor:
        bonus = torch.zeros(candidate_ids.shape[0], dtype=torch.float32, device=self.device)
        if generated_len < 10 or candidate_ids.numel() == 0:
            return bonus
        terminal_ids = self._sentence_terminal_ids()
        if terminal_ids.numel() == 0:
            return bonus
        terminal_mask = (candidate_ids.unsqueeze(1) == terminal_ids.unsqueeze(0)).any(dim=1)
        if not terminal_mask.any():
            return bonus
        close_bonus = min(1.5, 0.35 + 0.08 * float(generated_len - 10))
        bonus[terminal_mask] = close_bonus
        if self.eos_token_id is not None:
            bonus[candidate_ids == int(self.eos_token_id)] += 0.15
        return bonus

    def _paper_candidate_count(self, vocab_size: int, top_k: int) -> int:
        ratio = min(max(float(self.decoder_candidate_ratio), 1e-6), 1.0)
        target = int(math.ceil(ratio * float(vocab_size)))
        target = max(target, int(top_k), 1)
        return min(int(vocab_size), target)

    def ensure_vocab_head(self):
        if self.decoder_vocab_weight is not None:
            if self.decoder_vocab_bias is None:
                self.decoder_vocab_bias = torch.zeros(
                    self.decoder_vocab_weight.shape[0],
                    dtype=self.decoder_vocab_weight.dtype,
                    device=self.device,
                )
                self.data["decoder_vocab_bias"] = self.decoder_vocab_bias.detach().cpu()
            self.data["decoder_vocab_scale"] = float(self.decoder_vocab_scale)
            return
        if self.emb is None:
            raise RuntimeError("cloned vocab head requires full embedding weights")
        weight = self.emb.detach().cpu().clone()
        bias = torch.zeros(weight.shape[0], dtype=weight.dtype)
        self.apply_vocab_head(weight, bias=bias, scale=1.0)

    def vocab_logits(self, query: torch.Tensor) -> torch.Tensor:
        if self.decoder_vocab_weight is not None:
            bias = None if self.decoder_vocab_bias is None else self.decoder_vocab_bias.float()
            logits = F.linear(query.float(), self.decoder_vocab_weight.float(), bias)
            return float(self.decoder_vocab_scale) * logits
        return self.lexical_scores(query.float())

    def _ngram_repeat_scores(self, history_ids: list[int] | None, candidate_ids: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros(candidate_ids.shape[0], dtype=torch.float32, device=self.device)
        if not history_ids:
            return scores
        ngram = max(int(self.repeat_ngram), 2)
        if len(history_ids) < ngram - 1:
            return scores
        prefix = tuple(int(x) for x in history_ids[-(ngram - 1):])
        seen = {
            tuple(int(x) for x in history_ids[idx : idx + ngram])
            for idx in range(max(len(history_ids) - ngram + 1, 0))
        }
        if not seen:
            return scores
        values = [1.0 if (*prefix, int(token_id)) in seen else 0.0 for token_id in candidate_ids.tolist()]
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def _curvature_adjust_logits(
        self,
        candidate_ids: torch.Tensor,
        candidate_logits: torch.Tensor,
        *,
        ce_hidden: torch.Tensor,
        prev_hidden: torch.Tensor | None = None,
        prev_prev_hidden: torch.Tensor | None = None,
        history_ids: list[int] | None = None,
        context_anchor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if candidate_ids.numel() == 0:
            return candidate_logits, {
                "candidate_ids": candidate_ids,
                "combined_risk": torch.empty(0, device=self.device),
                "phase_grounding_risk": torch.empty(0, device=self.device),
                "suppression": torch.empty(0, device=self.device),
                "threshold": None,
                "curvature_risk_score": 0.0,
                "phase_grounding_risk_score": 0.0,
                "suppressed_count": 0,
            }

        candidate_emb = self.token_embedding(candidate_ids).float()
        current_hidden = ce_hidden.float()
        step_next = candidate_emb - current_hidden.unsqueeze(0)

        k1 = torch.zeros(candidate_ids.shape[0], dtype=torch.float32, device=self.device)
        if prev_hidden is not None:
            prev_step = (current_hidden - prev_hidden.float()).unsqueeze(0).expand_as(step_next)
            k1 = 1.0 - F.cosine_similarity(prev_step, step_next, dim=1, eps=1e-6)
            k1 = k1.clamp_min(0.0)

        k2 = torch.zeros_like(k1)
        if prev_hidden is not None and prev_prev_hidden is not None:
            accel_prev = (current_hidden - 2.0 * prev_hidden.float() + prev_prev_hidden.float()).unsqueeze(0)
            accel_prev = accel_prev.expand_as(step_next)
            accel_next = candidate_emb - 2.0 * current_hidden.unsqueeze(0) + prev_hidden.float().unsqueeze(0)
            k2 = 1.0 - F.cosine_similarity(accel_prev, accel_next, dim=1, eps=1e-6)
            k2 = k2.clamp_min(0.0)

        lap = self.state_graph_laplacian().to(step_next.device, dtype=step_next.dtype)
        lbo = (step_next @ lap).pow(2).mean(dim=1)
        lbo = lbo / lbo.mean().clamp_min(1e-6)

        context_break = torch.zeros_like(k1)
        phase_reference = current_hidden
        if context_anchor is not None:
            anchor = context_anchor.float().unsqueeze(0).expand_as(candidate_emb)
            context_break = 1.0 - F.cosine_similarity(candidate_emb, anchor, dim=1, eps=1e-6)
            context_break = context_break.clamp_min(0.0)
            phase_reference = context_anchor.float()

        repeat = torch.zeros_like(k1)
        if history_ids:
            recent = history_ids[-max(int(self.repeat_window), 1) :]
            if recent:
                recent_ids = torch.tensor(recent, dtype=torch.long, device=self.device)
                repeat = (candidate_ids.unsqueeze(1) == recent_ids.unsqueeze(0)).float().sum(dim=1)
                repeat = repeat / float(max(len(recent), 1))
            repeat = repeat + 2.0 * self._ngram_repeat_scores(history_ids, candidate_ids)

        combined = k1 + 0.5 * k2 + 0.3 * lbo + 0.25 * context_break + 1.5 * repeat
        risk_mean = combined.mean()
        risk_std = combined.std(unbiased=False)
        threshold = risk_mean + float(self.curvature_alpha) * risk_std
        excess = (combined - threshold).clamp_min(0.0)
        gate = torch.sigmoid(float(self.curvature_steepness) * (combined - threshold))
        suppression = float(self.curvature_lambda) * gate * excess
        phase_risk = torch.zeros_like(suppression)
        if float(self.phase_grounding_lambda) > 0.0:
            _, phase_risk = phase_grounding_suppression(
                candidate_logits,
                candidate_emb,
                phase_reference,
                strength=0.0,
            )
            suppression = suppression + float(self.phase_grounding_lambda) * phase_risk
        adjusted = candidate_logits - suppression
        return adjusted, {
            "candidate_ids": candidate_ids,
            "combined_risk": combined,
            "phase_grounding_risk": phase_risk,
            "suppression": suppression,
            "threshold": float(threshold.item()),
            "curvature_risk_score": float((combined >= threshold).float().mean().item()),
            "phase_grounding_risk_score": float((phase_risk > 0.5).float().mean().item()),
            "suppressed_count": int((suppression > 1e-3).sum().item()),
        }

    def build_runtime_codebook(self, m_ref: torch.Tensor, top_k: int) -> torch.Tensor:
        if self.has_standalone_lexicon():
            query = self.masked_state(m_ref, include_struct=True)
            if query.abs().sum().item() <= 1e-8:
                query = m_ref
            scores = self.lexical_scores(query)
            top_ids = torch.topk(scores, min(top_k, scores.numel())).indices
            return self.token_embedding(top_ids)
        raise RuntimeError("legacy teacher-dependent codebook path was removed from reality_stone.clarus")

    def ce_hidden(self, m_star: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(m_star, (self.d,), self.ln_w, self.ln_b)

    def teacher_embedding(self, token_ids: torch.Tensor | list[int]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("teacher embedding is unavailable for a runtime-only artifact")
        if not torch.is_tensor(token_ids):
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        token_ids = token_ids.to(device=self.device, dtype=torch.long).view(-1)
        return self.model.transformer.wte.weight.index_select(0, token_ids)

    def teacher_hidden_and_logits(self, prompt_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("teacher path is disabled in runtime-only mode")

    def teacher_next_logits(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("teacher path is disabled in runtime-only mode")

    def decoder_token_correction(
        self,
        ce_hidden: torch.Tensor,
        prev_emb: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.decoder_token_ids is None:
            return None
        correction = None
        if self.decoder_token_state_proj is not None:
            proj_idx = self._projection_indices()
            if (
                proj_idx is not None
                and self.decoder_token_state_proj.ndim == 2
                and self.decoder_token_state_proj.shape[0] == proj_idx.numel()
            ):
                correction = ce_hidden.index_select(0, proj_idx) @ self.decoder_token_state_proj
            else:
                correction = ce_hidden @ self.decoder_token_state_proj
        if self.decoder_token_prev_proj is not None:
            prev_piece = self.decoder_prev_scale * (prev_emb @ self.decoder_token_prev_proj)
            correction = prev_piece if correction is None else correction + prev_piece
        if self.decoder_token_bias is not None:
            correction = self.decoder_token_bias if correction is None else correction + self.decoder_token_bias
        if correction is None:
            return None
        return self.decoder_token_scale * correction

    def standalone_logits(
        self,
        ce_hidden: torch.Tensor,
        prev_id: int,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        repeat_ids: list[int] | None = None,
        repeat_penalty: float = 3.0,
        history_ids: list[int] | None = None,
        prev_hidden: torch.Tensor | None = None,
        prev_prev_hidden: torch.Tensor | None = None,
        context_anchor: torch.Tensor | None = None,
        generated_len: int = 0,
        return_meta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        prev_emb = self.token_embedding([prev_id]).squeeze(0)
        state_hidden = ce_hidden.float()
        query = self.decoder_query(state_hidden, prev_emb)
        logits = self.vocab_logits(query)
        correction = self.decoder_token_correction(state_hidden, prev_emb)
        if correction is not None and self.decoder_token_ids is not None and self.decoder_token_ids.numel():
            logits = logits.clone()
            logits[self.decoder_token_ids] += correction

        candidate_k = self._paper_candidate_count(logits.numel(), top_k)
        candidate_top_ids = torch.topk(logits, min(candidate_k, logits.numel())).indices
        candidate_ids = self._merge_candidate_ids(candidate_top_ids, self._sentence_terminal_ids())
        if candidate_ids.numel() == 0:
            logits = logits.new_full(logits.shape, float("-inf"))
            if not return_meta:
                return logits
            return logits, {
                "candidate_ids": torch.empty(0, dtype=torch.long, device=self.device),
                "combined_risk": torch.empty(0, dtype=torch.float32, device=self.device),
                "phase_grounding_risk": torch.empty(0, dtype=torch.float32, device=self.device),
                "suppression": torch.empty(0, dtype=torch.float32, device=self.device),
                "threshold": None,
                "curvature_risk_score": 0.0,
                "phase_grounding_risk_score": 0.0,
                "suppressed_count": 0,
                "candidate_count": 0,
                "eval_candidate_count": 0,
            }

        candidate_logits = logits.index_select(0, candidate_ids)
        if repeat_ids:
            repeat_set = {int(token_id) for token_id in repeat_ids}
            if repeat_set:
                repeat_mask = torch.tensor(
                    [int(token_id) in repeat_set for token_id in candidate_ids.tolist()],
                    dtype=torch.bool,
                    device=self.device,
                )
                candidate_logits = candidate_logits.clone()
                candidate_logits[repeat_mask] -= repeat_penalty

        eval_k = min(
            int(candidate_ids.numel()),
            max(int(top_k) * 2, min(int(self.curvature_eval_topk), 96), 1),
        )
        eval_rank = torch.topk(candidate_logits, eval_k).indices
        eval_ids = candidate_ids.index_select(0, eval_rank)
        eval_logits = candidate_logits.index_select(0, eval_rank)
        adjusted_eval_logits, curvature_meta = self._curvature_adjust_logits(
            eval_ids,
            eval_logits,
            ce_hidden=ce_hidden,
            prev_hidden=prev_hidden,
            prev_prev_hidden=prev_prev_hidden,
            history_ids=history_ids,
            context_anchor=context_anchor,
        )
        candidate_logits = candidate_logits.clone()
        candidate_logits.index_copy_(0, eval_rank, adjusted_eval_logits)
        candidate_logits = candidate_logits + self._sentence_close_bonus(
            candidate_ids,
            generated_len=generated_len,
        )
        logits = logits.new_full(logits.shape, float("-inf"))
        logits[candidate_ids] = candidate_logits
        logits = logits / max(temperature, 1e-6)

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.numel()))
            logits = logits.clone()
            logits[logits < v[-1]] = float("-inf")
        if not return_meta:
            return logits
        curvature_meta["candidate_count"] = int(candidate_ids.numel())
        curvature_meta["eval_candidate_count"] = int(eval_k)
        return logits, curvature_meta

    def apply_vocab_head(
        self,
        weight: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
        scale: float | None = None,
    ):
        self.decoder_vocab_weight = weight.float().to(self.device)
        if bias is None:
            bias = torch.zeros(self.decoder_vocab_weight.shape[0], dtype=self.decoder_vocab_weight.dtype)
        self.decoder_vocab_bias = bias.float().to(self.device)
        if scale is not None:
            self.decoder_vocab_scale = float(scale)
        self.data["decoder_vocab_weight"] = self.decoder_vocab_weight.detach().cpu()
        self.data["decoder_vocab_bias"] = self.decoder_vocab_bias.detach().cpu()
        self.data["decoder_vocab_scale"] = float(self.decoder_vocab_scale)

    def apply_decoder_refine(
        self,
        prev_proj: torch.Tensor,
        state_proj: torch.Tensor,
        *,
        query_bias: torch.Tensor | None = None,
    ):
        self.decoder_prev_proj = self._compress_prev_proj(prev_proj.detach().cpu())
        self.decoder_state_proj = self._compress_state_proj(state_proj.detach().cpu())
        self.data["decoder_prev_proj"] = self.decoder_prev_proj.detach().cpu()
        self.data["decoder_state_proj"] = self.decoder_state_proj.detach().cpu()
        if query_bias is not None:
            self.decoder_query_bias = query_bias.detach().float().to(self.device)
            self.data["decoder_query_bias"] = self.decoder_query_bias.detach().cpu()

    def apply_token_head(
        self,
        token_ids: torch.Tensor,
        *,
        state_proj: torch.Tensor | None = None,
        prev_proj: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        scale: float | None = None,
    ):
        self.decoder_token_ids = token_ids.long().to(self.device)
        self.decoder_token_state_proj = None if state_proj is None else self._compress_token_state_proj(state_proj.detach().cpu())
        self.decoder_token_prev_proj = None if prev_proj is None else prev_proj.float().to(self.device)
        self.decoder_token_bias = None if bias is None else bias.float().to(self.device)
        if scale is not None:
            self.decoder_token_scale = float(scale)
        self.data["decoder_token_ids"] = token_ids.detach().cpu().long()
        self.data["decoder_token_state_proj"] = None if self.decoder_token_state_proj is None else self.decoder_token_state_proj.detach().cpu()
        self.data["decoder_token_prev_proj"] = None if prev_proj is None else prev_proj.detach().cpu()
        self.data["decoder_token_bias"] = None if bias is None else bias.detach().cpu()
        self.data["decoder_token_scale"] = float(self.decoder_token_scale)

    def decoder_snapshot(self) -> dict[str, torch.Tensor | float | None]:
        def clone_cpu(value):
            return None if value is None else value.detach().cpu().clone()

        return {
            "decoder_prev_proj": clone_cpu(self.decoder_prev_proj),
            "decoder_state_proj": clone_cpu(self.decoder_state_proj),
            "decoder_query_bias": clone_cpu(self.decoder_query_bias),
            "decoder_vocab_weight": clone_cpu(self.decoder_vocab_weight),
            "decoder_vocab_bias": clone_cpu(self.decoder_vocab_bias),
            "decoder_vocab_scale": float(self.decoder_vocab_scale),
            "decoder_token_ids": clone_cpu(self.decoder_token_ids),
            "decoder_token_state_proj": clone_cpu(self.decoder_token_state_proj),
            "decoder_token_prev_proj": clone_cpu(self.decoder_token_prev_proj),
            "decoder_token_bias": clone_cpu(self.decoder_token_bias),
            "decoder_token_scale": float(self.decoder_token_scale),
            "pq_centroids": clone_cpu(self.pq_centroids),
            "pq_codes": clone_cpu(self.pq_codes),
            "W": clone_cpu(self.W),
            "active_dim_mask": clone_cpu(self.active_dim_mask),
            "struct_dim_mask": clone_cpu(self.struct_dim_mask),
            "background_dim_mask": clone_cpu(self.background_dim_mask),
        }

    def restore_decoder_snapshot(self, snapshot: dict[str, torch.Tensor | float | None]):
        def load_tensor(name: str):
            value = snapshot.get(name)
            return None if value is None else value.to(self.device)

        w_tensor = snapshot.get("W")
        if w_tensor is not None:
            self.apply_relax_matrix(w_tensor)
        self.decoder_prev_proj = load_tensor("decoder_prev_proj")
        self.decoder_state_proj = load_tensor("decoder_state_proj")
        self.decoder_query_bias = load_tensor("decoder_query_bias")
        self.decoder_vocab_weight = load_tensor("decoder_vocab_weight")
        self.decoder_vocab_bias = load_tensor("decoder_vocab_bias")
        self.decoder_vocab_scale = float(snapshot.get("decoder_vocab_scale", self.decoder_vocab_scale))
        self.decoder_token_ids = load_tensor("decoder_token_ids")
        self.decoder_token_state_proj = load_tensor("decoder_token_state_proj")
        self.decoder_token_prev_proj = load_tensor("decoder_token_prev_proj")
        self.decoder_token_bias = load_tensor("decoder_token_bias")
        self.decoder_token_scale = float(snapshot.get("decoder_token_scale", self.decoder_token_scale))
        self.pq_centroids = load_tensor("pq_centroids")
        self.pq_codes = load_tensor("pq_codes")
        active_mask = snapshot.get("active_dim_mask")
        struct_mask = snapshot.get("struct_dim_mask")
        if active_mask is not None and struct_mask is not None:
            self.apply_state_partition(active_mask, struct_mask)
        else:
            self.active_dim_mask = None
            self.struct_dim_mask = None
            self.background_dim_mask = None

        for key in (
            "decoder_prev_proj",
            "decoder_state_proj",
            "decoder_query_bias",
            "decoder_vocab_weight",
            "decoder_vocab_bias",
            "decoder_token_ids",
            "decoder_token_state_proj",
            "decoder_token_prev_proj",
            "decoder_token_bias",
            "pq_centroids",
            "pq_codes",
            "active_dim_mask",
            "struct_dim_mask",
            "background_dim_mask",
        ):
            value = snapshot.get(key)
            self.data[key] = None if value is None else value.clone()
        self.data["decoder_vocab_scale"] = float(self.decoder_vocab_scale)
        self.data["decoder_token_scale"] = float(self.decoder_token_scale)

    def standalone_generate(
        self,
        prompt_ids: torch.Tensor,
        m_star: torch.Tensor,
        *,
        max_tok: int,
        temperature: float,
        top_k: int,
        repeat_penalty: float,
        refresh_interval: int = 0,
        refresh_args=None,
        refresh_init_layer: int | None = None,
        refresh_phi: torch.Tensor | None = None,
    ) -> tuple[str, list[int], dict[str, float | int | None]]:
        if not self.has_standalone_lexicon():
            raise RuntimeError("Standalone decoder requires embeddings or PQ lexical memory")

        refresh_interval = max(int(refresh_interval), 0)
        if getattr(self, '_skip_ln_for_standalone', False):
            h = m_star.float().to(self.device)
        else:
            h = F.layer_norm(m_star, (self.d,), self.ln_w, self.ln_b)
        prev_id = int(prompt_ids[0, -1].item())
        running_ids = prompt_ids.clone()
        out_ids: list[int] = []
        phi_state = None if refresh_phi is None else refresh_phi.detach().clone().to(self.device)
        init_layer = refresh_init_layer
        refresh_count = 0
        refresh_steps = 0
        refresh_time_s = 0.0
        refresh_cos: list[float] = []
        chosen_risk: list[float] = []
        chosen_phase_grounding_risk: list[float] = []
        chosen_suppression: list[float] = []
        step_risk_score: list[float] = []
        suppression_hits = 0
        history_ids = running_ids[0].tolist()
        prev_hidden = None
        prev_prev_hidden = None
        context_anchor = h.detach().clone()

        for _ in range(max_tok):
            logits, step_meta = self.standalone_logits(
                h,
                prev_id,
                temperature=temperature,
                top_k=top_k,
                repeat_ids=out_ids[-max(int(self.repeat_window), 1) :],
                repeat_penalty=repeat_penalty,
                history_ids=history_ids,
                prev_hidden=prev_hidden,
                prev_prev_hidden=prev_prev_hidden,
                context_anchor=context_anchor,
                generated_len=len(out_ids),
                return_meta=True,
            )
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            if self.eos_token_id is not None and next_id == self.eos_token_id:
                break
            candidate_ids = step_meta["candidate_ids"]
            if candidate_ids.numel():
                match = torch.nonzero(candidate_ids == next_id, as_tuple=False)
                if match.numel():
                    idx = int(match[0, 0].item())
                    risk_values = step_meta["combined_risk"]
                    phase_values = step_meta["phase_grounding_risk"]
                    suppression_values = step_meta["suppression"]
                    chosen_risk.append(float(risk_values[idx].item()))
                    chosen_phase_grounding_risk.append(float(phase_values[idx].item()))
                    chosen_suppression.append(float(suppression_values[idx].item()))
            step_risk_score.append(float(step_meta["curvature_risk_score"]))
            suppression_hits += int(step_meta["suppressed_count"])

            step_hidden = h.detach().clone()
            out_ids.append(next_id)
            prev_id = next_id
            next_token = torch.tensor([[next_id]], device=self.device)
            running_ids = torch.cat([running_ids, next_token], dim=1)
            history_ids.append(next_id)
            prev_prev_hidden = prev_hidden
            prev_hidden = step_hidden
            if (
                refresh_interval > 0
                and refresh_args is not None
                and init_layer is not None
                and len(out_ids) < max_tok
                and len(out_ids) % refresh_interval == 0
            ):
                refresh_ctx = self.context_from_ids(
                    running_ids,
                    init_layer=init_layer,
                    phi=phi_state,
                    need_teacher=False,
                )
                refresh_result = self.relax_context(refresh_ctx, refresh_args)
                h = self.ce_hidden(refresh_result["m_star"])
                phi_state = refresh_result["phi_updated"]
                init_layer = refresh_ctx.best_layer
                context_anchor = h.detach().clone()
                refresh_count += 1
                refresh_steps += int(refresh_result["steps"])
                refresh_time_s += float(refresh_result["elapsed_s"])
                if refresh_result["cos_ms_h"] is not None:
                    refresh_cos.append(float(refresh_result["cos_ms_h"]))

        meta = {
            "refresh_interval": refresh_interval,
            "refresh_count": refresh_count,
            "refresh_steps": refresh_steps,
            "refresh_time_s": refresh_time_s,
            "refresh_cos_mean": None if not refresh_cos else sum(refresh_cos) / len(refresh_cos),
            "refresh_phi_norm": None if phi_state is None else float(phi_state.norm().item()),
            "curvature_risk_score": None if not step_risk_score else sum(step_risk_score) / len(step_risk_score),
            "chosen_risk_mean": None if not chosen_risk else sum(chosen_risk) / len(chosen_risk),
            "chosen_phase_grounding_risk_mean": (
                None
                if not chosen_phase_grounding_risk
                else sum(chosen_phase_grounding_risk) / len(chosen_phase_grounding_risk)
            ),
            "chosen_suppression_mean": None if not chosen_suppression else sum(chosen_suppression) / len(chosen_suppression),
            "suppression_hits": int(suppression_hits),
        }
        return self.tok.decode(out_ids, skip_special_tokens=True), out_ids, meta

    def legacy_generate(
        self,
        token_ids: list[int],
        *,
        n_tokens: int,
        residual_scale: float,
        temperature: float,
    ) -> list[int]:
        if self.emb is None or not self.data.get("W_layers"):
            raise RuntimeError("Legacy generator requires full embedding table and W_layers")

        generated = list(token_ids)
        used = set()
        m = self.emb[token_ids].mean(dim=0)
        phi = self.emb.var(dim=0).clamp(min=1e-8).sqrt()
        m_hist: list[torch.Tensor] = [m.clone()]

        for _ in range(n_tokens):
            m_out = m.clone()
            for w_layer in self.data["W_layers"]:
                w_dev = w_layer.to(self.device)
                delta = w_dev @ m_out
                m_out = m_out + residual_scale * delta
                m_out = m_out / (m_out.norm() + 1e-8) * m.norm()
            m_out = m_out * (self.h_norm_ref / (m_out.norm() + 1e-8))

            phi_hat = F.normalize(phi, dim=0)
            m_out = m_out + self.portal * phi_hat * self.h_norm_ref

            # Bypass force per E20 (docs/7_AGI/12_Equation.md 1.5/3.1):
            #   F_bypass(k) = (C_k / alpha_b) * phi,   C_k = ||m_k - 2 m_{k-1} + m_{k-2}||.
            # Until 3 trajectory samples are available C_k = 0, matching ce_ops.relax.
            if len(m_hist) >= 3:
                c_k = float((m_hist[-1] - 2.0 * m_hist[-2] + m_hist[-3]).norm().item())
                m_out = m_out + (c_k * self.bypass) * phi

            h = F.layer_norm(m_out, (self.d,), self.ln_w, self.ln_b)
            logits = h @ self.emb.T
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            candidates = torch.topk(probs, 50)

            next_id = None
            for cid in candidates.indices.tolist():
                if cid not in used:
                    next_id = cid
                    break
            if next_id is None:
                next_id = int(candidates.indices[0].item())

            generated.append(next_id)
            used.add(next_id)
            new_emb = self.emb[next_id]
            m = 0.3 * m + 0.7 * new_emb
            m_hist.append(m.clone())
            if len(m_hist) > 3:
                m_hist.pop(0)

        return generated

    def prompt_context(self, prompt: str) -> PromptContext:
        prompt_ids = self.tok.encode(prompt, return_tensors="pt").to(self.device)
        return self.context_from_ids(prompt_ids, prompt=prompt)

    def _analyze_prompt_ids(
        self,
        prompt_ids: torch.Tensor,
        *,
        candidate_layers: list[int],
        need_teacher: bool,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor | None]:
        """Run a teacher forward pass and capture hidden states by layer.

        Returns ``(phi, captured, h_true)`` where ``phi`` is the normalized
        difference between the mean of all-but-last token hidden states and the
        last token hidden state (zero for single-token prompts), ``captured``
        is a dict of ``{layer_idx: hidden_state}`` for the requested layers,
        and ``h_true`` is the last-layer hidden state at the final position.

        Requires a teacher ``self.model`` to be present (e.g. for offline
        analysis or unit tests that inject a ``FakeModel``). Runtime-only
        artifacts must use :meth:`runtime_prompt_state` instead.
        """
        if self.model is None:
            raise RuntimeError(
                "_analyze_prompt_ids requires a teacher model; runtime artifacts must "
                "use runtime_prompt_state() / context_from_ids(need_teacher=False)"
            )

        ids = prompt_ids.to(device=self.device, dtype=torch.long)
        transformer = getattr(self.model, "transformer", None)
        if transformer is None:
            raise RuntimeError("teacher model is missing the .transformer attribute")

        with torch.no_grad():
            wte = transformer.wte(ids)
            pos_idx = torch.arange(ids.shape[1], device=self.device, dtype=torch.long)
            wpe = transformer.wpe(pos_idx).unsqueeze(0)
            h = wte + wpe

            seq = h[0]
            if seq.shape[0] <= 1:
                phi = torch.zeros(seq.shape[-1], device=self.device, dtype=seq.dtype)
            else:
                phi = normalize_vector(seq[:-1].mean(dim=0) - seq[-1])

            capture_set = sorted({int(layer) for layer in candidate_layers})
            captured: dict[int, torch.Tensor] = {}
            blocks = transformer.h
            target_layer = max(capture_set[-1] if capture_set else -1, len(blocks) - 1)
            for layer_idx in range(target_layer + 1):
                h = blocks[layer_idx](h)
                if isinstance(h, tuple):
                    h = h[0]
                if layer_idx in capture_set:
                    captured[layer_idx] = h[:, -1, :].squeeze(0).detach().float()

            h_true = transformer.ln_f(h)[:, -1, :].detach().float() if need_teacher else None
        return phi, captured, h_true

    def context_from_ids(
        self,
        prompt_ids: torch.Tensor,
        prompt: str | None = None,
        *,
        init_layer: int | None = None,
        phi: torch.Tensor | None = None,
        need_teacher: bool = True,
    ) -> PromptContext:
        if need_teacher and self.model is not None and not getattr(self, 'allow_pretrained_fallback', False):
            raise RuntimeError("teacher path is disabled in runtime-only mode")
        m0, phi_base = self.runtime_prompt_state(prompt_ids, phi=phi)
        best_layer = int(init_layer) if init_layer is not None else int(
            self.data.get("default_init_layer", max(self.n_layer - 1, 0))
        )
        phi_state = phi_base if phi is None else phi.detach().float().to(self.device)
        return PromptContext(
            prompt=prompt if prompt is not None else self.tok.decode(prompt_ids[0], skip_special_tokens=True),
            prompt_ids=prompt_ids,
            h_true=None,
            m0=m0,
            phi=phi_state,
            best_layer=best_layer,
            layer_scores={best_layer: float("nan")},
        )

    def relax_context(self, ctx: PromptContext, args):
        dt_eff = min(float(args.dt), 0.9 * self.tau)
        cb_weight = self.portal if args.cb_weight is None else float(args.cb_weight)
        codebook = self.build_runtime_codebook(ctx.m0, top_k=args.cb_topk)
        metric_basis = ce_build_metric_basis(
            codebook,
            ctx.m0,
            int(args.metric_rank),
            w_eigvecs=self._get_w_eigvecs(args.metric_rank),
            backend=args.backend,
        )
        t0 = time.time()
        m_star, hist, n_steps = ce_relax_packed(
            self.W_pack[0],
            self.W_pack[1],
            self.W_pack[2],
            ctx.m0,
            ctx.phi,
            ctx.m0,
            codebook,
            metric_basis,
            portal=self.portal,
            bypass=self.bypass,
            t_wake=self.t_wake,
            beta=args.beta,
            cb_w=cb_weight,
            tau=self.tau,
            dt=dt_eff,
            max_steps=args.steps,
            metric_rank=args.metric_rank,
            lambda0=args.lambda0,
            lambda_phi=args.lambda_phi,
            lambda_var=args.lambda_var,
            noise_scale=args.noise_scale,
            anneal_ratio=0.6,
            tol=1e-4,
            backend=args.backend,
            seed=args.seed,
            dense_w=self._dense_relax_w,
        )
        elapsed = time.time() - t0
        cos_ms = None
        if ctx.h_true is not None:
            cos_ms = F.cosine_similarity(m_star.unsqueeze(0), ctx.h_true).item()
        phi_var = hist.get("phi_var")
        if phi_var:
            phi_updated = update_phi(ctx.phi, m_star, phi_var=m_star.new_tensor(phi_var))
        else:
            phi_updated = update_phi(ctx.phi, m_star)
        return {
            "m_star": m_star,
            "hist": hist,
            "steps": n_steps,
            "elapsed_s": elapsed,
            "cos_m0_h": ctx.layer_scores[ctx.best_layer],
            "cos_ms_h": cos_ms,
            "phi_updated": phi_updated,
            "dt_eff": dt_eff,
        }

    def select_mode(self, phi_updated: torch.Tensor, args) -> str:
        mode = getattr(args, "decode_mode", "auto")
        if mode not in ("auto", "standalone"):
            raise RuntimeError("runtime-only mode supports standalone decoding")
        return "standalone"

    @staticmethod
    def _copy_args(args, **updates):
        payload = vars(args).copy()
        payload.update(updates)
        return SimpleNamespace(**payload)

    def decode_outputs(self, ctx: PromptContext, relax_result: dict, args):
        outputs: dict[str, str] = {}
        meta: dict[str, object] = {}
        chosen_mode = self.select_mode(relax_result["phi_updated"], args)

        def run_standalone():
            refresh_args = None
            if args.standalone_refresh_interval > 0:
                refresh_args = self._copy_args(
                    args,
                    steps=args.standalone_refresh_steps,
                    cb_topk=args.standalone_refresh_cb_topk,
                    metric_rank=args.standalone_refresh_metric_rank,
                    noise_scale=args.standalone_refresh_noise_scale,
                )
            text, token_ids, standalone_meta = self.standalone_generate(
                ctx.prompt_ids,
                relax_result["m_star"],
                max_tok=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                repeat_penalty=args.repeat_penalty,
                refresh_interval=args.standalone_refresh_interval,
                refresh_args=refresh_args,
                refresh_init_layer=ctx.best_layer,
                refresh_phi=relax_result["phi_updated"],
            )
            outputs["standalone"] = ctx.prompt + text
            meta["standalone_token_ids"] = token_ids
            meta["standalone_refresh_interval"] = standalone_meta["refresh_interval"]
            meta["standalone_refresh_count"] = standalone_meta["refresh_count"]
            meta["standalone_refresh_steps"] = standalone_meta["refresh_steps"]
            meta["standalone_refresh_time_s"] = standalone_meta["refresh_time_s"]
            meta["standalone_refresh_cos_mean"] = standalone_meta["refresh_cos_mean"]
            meta["standalone_refresh_phi_norm"] = standalone_meta["refresh_phi_norm"]
            meta["standalone_curvature_risk"] = standalone_meta["curvature_risk_score"]
            meta["standalone_chosen_risk_mean"] = standalone_meta["chosen_risk_mean"]
            meta["standalone_chosen_phase_grounding_risk_mean"] = (
                standalone_meta["chosen_phase_grounding_risk_mean"]
            )
            meta["standalone_chosen_suppression_mean"] = standalone_meta["chosen_suppression_mean"]
            meta["standalone_suppression_hits"] = standalone_meta["suppression_hits"]

        run_standalone()

        return chosen_mode, outputs, meta

    def reference_generate(self, prompt: str, max_new_tokens: int) -> str:
        raise RuntimeError("reference generation is disabled in runtime-only mode")


def build_prompt_list(args) -> list[str]:
    base = list(args.prompts) if args.prompts else [args.prompt, *DEFAULT_PROMPTS]
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in base:
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def build_guard_list(args) -> list[str]:
    base = list(args.microsleep_guard_prompts) if args.microsleep_guard_prompts else list(DEFAULT_PROMPTS)
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in base:
        if prompt and prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def load_microsleep_tools():
    if __package__:
        from .sleep import PromptReplayBuffer, evaluate_guard_set, run_guarded_microsleep_step
    else:
        from reality_stone.clarus.sleep import PromptReplayBuffer, evaluate_guard_set, run_guarded_microsleep_step
    return PromptReplayBuffer, evaluate_guard_set, run_guarded_microsleep_step


def main():
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="CE Equation-spec runtime")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--prompt", default="인공지능의 미래는")
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--tokens", type=int, default=15)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--multiround-steps", type=int, default=100)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="torch", choices=["auto", "torch", "rust", "cuda"])
    ap.add_argument("--compare-gpt2", action="store_true")
    ap.add_argument("--cb-topk", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--cb-weight", type=float, default=None)
    ap.add_argument("--metric-rank", type=int, default=16)
    ap.add_argument("--lambda0", type=float, default=1.0)
    ap.add_argument("--lambda-phi", dest="lambda_phi", type=float, default=0.5)
    ap.add_argument("--lambda-var", dest="lambda_var", type=float, default=0.25)
    ap.add_argument("--noise-scale", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ce-strength", type=float, default=0.3)
    ap.add_argument(
        "--decode-mode",
        default="standalone",
        choices=["auto", "standalone"],
    )
    ap.add_argument("--phi-threshold", type=float, default=1.0)
    ap.add_argument("--sleep-threshold", type=float, default=2.0)
    ap.add_argument("--sleep-decay", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--repeat-penalty", type=float, default=3.0)
    ap.add_argument("--phase-grounding-lambda", type=float, default=None)
    ap.add_argument("--standalone-refresh-interval", type=int, default=1)
    ap.add_argument("--standalone-refresh-steps", type=int, default=48)
    ap.add_argument("--standalone-refresh-cb-topk", type=int, default=128)
    ap.add_argument("--standalone-refresh-metric-rank", type=int, default=0)
    ap.add_argument("--standalone-refresh-noise-scale", type=float, default=0.0)
    ap.add_argument("--microsleep-every", type=int, default=0)
    ap.add_argument("--microsleep-replay-capacity", type=int, default=16)
    ap.add_argument("--microsleep-guard-prompts", nargs="*", default=None)
    ap.add_argument("--microsleep-tokens", type=int, default=4)
    ap.add_argument("--microsleep-label-topk", "--microsleep-teacher-topk", dest="microsleep_teacher_topk", type=int, default=8)
    ap.add_argument("--microsleep-ridge", type=float, default=1e-3)
    ap.add_argument("--microsleep-rem-weight", type=float, default=2.5)
    ap.add_argument("--microsleep-rem-mix", type=float, default=0.35)
    ap.add_argument("--microsleep-token-head-max-vocab", type=int, default=2048)
    ap.add_argument("--microsleep-token-head-scale", type=float, default=1.0)
    ap.add_argument("--microsleep-guard-min-top10-delta", type=float, default=0.0)
    ap.add_argument("--microsleep-guard-min-top50-delta", type=float, default=0.0)
    ap.add_argument("--microsleep-guard-max-top10-drop", type=float, default=0.0)
    ap.add_argument("--microsleep-guard-max-top50-drop", type=float, default=0.0)
    ap.add_argument("--microsleep-guard-max-phase-grounding-risk-increase", type=float, default=None)
    ap.add_argument("--microsleep-output", default=None)
    ap.add_argument(
        "--residual",
        type=float,
        default=0.15,
        help="Legacy sequential residual argument retained for CLI compatibility.",
    )
    args = ap.parse_args()
    if args.compare_gpt2:
        raise RuntimeError("teacher/reference comparison is disabled in runtime-only mode")

    eng = CEEngine(args.engine, device=args.device, backend=args.backend)
    if eng.model is not None or eng.model_source != "runtime":
        raise RuntimeError("runtime-only execution requires a clone-free runtime artifact")
    if args.phase_grounding_lambda is not None:
        eng.phase_grounding_lambda = float(args.phase_grounding_lambda)
        eng.data["phase_grounding_lambda"] = float(eng.phase_grounding_lambda)

    mem = eng.memory_usage()
    prompts = build_prompt_list(args)
    microsleep_events: list[dict[str, object]] = []
    microsleep_guard_initial = None
    microsleep_guard_final = None
    microsleep_guard_prompts: list[str] = []
    microsleep_buffer = None
    run_microsleep_step = None
    evaluate_guard_set = None
    microsleep_accepted = 0
    microsleep_rejected = 0

    safe_print("\n=== CE Hopfield Engine (Equation Spec Runtime) ===")
    safe_print(f"  model={eng.model_name}")
    safe_print(f"  d={eng.d}  layers={eng.n_layer}  vocab={eng.vocab}")
    safe_print(f"  tau={eng.tau:.4f}  hidden_norm_ref={eng.h_norm_ref:.2f}")
    safe_print(f"  backend={args.backend}  device={eng.device}  model_source={eng.model_source}")

    safe_print("\n--- Memory ---")
    for key, value in mem.items():
        safe_print(f"  {key}: {value:.2f}")

    safe_print(
        f"\n--- Generation (steps={args.steps}, dt={min(args.dt, 0.9 * eng.tau):.4f}, "
        f"temp={args.temperature}, mode={args.decode_mode}, "
        f"phase_lambda={eng.phase_grounding_lambda:.4f}) ---"
    )

    if args.microsleep_every > 0:
        PromptReplayBuffer, evaluate_guard_set, run_microsleep_step = load_microsleep_tools()
        microsleep_guard_prompts = build_guard_list(args)
        microsleep_buffer = PromptReplayBuffer(capacity=max(1, args.microsleep_replay_capacity))
        microsleep_guard_initial = evaluate_guard_set(
            eng,
            microsleep_guard_prompts,
            args,
            max_new_tokens=args.microsleep_tokens,
            refresh_interval=args.standalone_refresh_interval,
            refresh_steps=args.standalone_refresh_steps,
            refresh_cb_topk=args.standalone_refresh_cb_topk,
            refresh_metric_rank=args.standalone_refresh_metric_rank,
            refresh_noise_scale=args.standalone_refresh_noise_scale,
        )
        safe_print("\n--- Microsleep ---")
        safe_print(
            f"  every={args.microsleep_every}  replay_capacity={args.microsleep_replay_capacity}  "
            f"tokens={args.microsleep_tokens}  guard_prompts={len(microsleep_guard_prompts)}"
        )
        safe_print(
            f"  guard_top10={microsleep_guard_initial['top10_acc']:.3f}  "
            f"guard_top50={microsleep_guard_initial['top50_acc']:.3f}"
        )

    results = []
    for idx, prompt in enumerate(prompts, start=1):
        ctx = eng.prompt_context(prompt)
        relax_result = eng.relax_context(ctx, args)
        chosen_mode, outputs, decode_meta = eng.decode_outputs(ctx, relax_result, args)

        safe_print(f"\n  [{prompt}]")
        safe_print(
            f"    init_layer={ctx.best_layer}  cos(m0,h)={_format_optional(relax_result['cos_m0_h'])}  "
            f"cos(m*,h)={_format_optional(relax_result['cos_ms_h'])}"
        )
        safe_print(
            f"    relax_steps={relax_result['steps']}  "
            f"time={relax_result['elapsed_s']:.2f}s  "
            f"phi={ctx.phi.norm().item():.2f}->{relax_result['phi_updated'].norm().item():.2f}"
        )
        if relax_result["hist"]["E"]:
            safe_print(
                f"    energy={relax_result['hist']['E'][0]:.4f}"
                f"->{relax_result['hist']['E'][-1]:.4f}"
            )

        output_text = outputs.get(chosen_mode, "")
        safe_print(f"    [{chosen_mode}] -> {output_text}")
        if "standalone_refresh_count" in decode_meta:
            safe_print(
                f"    [standalone-refresh] interval={decode_meta['standalone_refresh_interval']}  "
                f"count={decode_meta['standalone_refresh_count']}  "
                f"time={decode_meta['standalone_refresh_time_s']:.2f}s"
            )
        if "standalone_curvature_risk" in decode_meta:
            safe_print(
                f"    [standalone-guard] risk={_format_optional(decode_meta['standalone_curvature_risk'])}  "
                f"phase={_format_optional(decode_meta.get('standalone_chosen_phase_grounding_risk_mean'))}  "
                f"suppression_hits={int(decode_meta.get('standalone_suppression_hits', 0))}"
            )

        results.append(
            {
                "prompt": prompt,
                "best_init_layer": ctx.best_layer,
                "init_layer_sweep": {
                    str(k): (_optional_float(v) if _optional_float(v) is not None else None)
                    for k, v in ctx.layer_scores.items()
                },
                "mode": chosen_mode,
                "cos_m0_h": _optional_float(relax_result["cos_m0_h"]),
                "cos_ms_h": _optional_float(relax_result["cos_ms_h"]),
                "phi_norm": {
                    "initial": round(ctx.phi.norm().item(), 4),
                    "updated": round(relax_result["phi_updated"].norm().item(), 4),
                },
                "relax_steps": relax_result["steps"],
                "relax_time_s": round(relax_result["elapsed_s"], 4),
                "energy_start": round(relax_result["hist"]["E"][0], 4) if relax_result["hist"]["E"] else None,
                "energy_end": round(relax_result["hist"]["E"][-1], 4) if relax_result["hist"]["E"] else None,
                "outputs": outputs,
                "standalone_curvature_risk": _optional_float(decode_meta.get("standalone_curvature_risk")),
                "standalone_chosen_risk_mean": _optional_float(decode_meta.get("standalone_chosen_risk_mean")),
                "standalone_chosen_phase_grounding_risk_mean": _optional_float(
                    decode_meta.get("standalone_chosen_phase_grounding_risk_mean")
                ),
                "standalone_chosen_suppression_mean": _optional_float(
                    decode_meta.get("standalone_chosen_suppression_mean")
                ),
                "standalone_suppression_hits": int(decode_meta.get("standalone_suppression_hits", 0)),
                "multiround_phi_norms": [
                    round(v, 4) for v in decode_meta.get("multiround_phi_norms", [])[:32]
                ],
                "multiround_energies": [
                    round(v, 4) for v in decode_meta.get("multiround_energies", [])[:32]
                ],
                "gpt2_reference": None,
            }
        )

        if run_microsleep_step is not None and microsleep_buffer is not None:
            event = run_microsleep_step(
                eng,
                microsleep_buffer,
                prompt,
                microsleep_guard_prompts,
                args,
                step_index=idx,
                sleep_every=args.microsleep_every,
                max_new_tokens=args.microsleep_tokens,
                teacher_topk=args.microsleep_teacher_topk,
                ridge=args.microsleep_ridge,
                rem_weight=args.microsleep_rem_weight,
                rem_mix=args.microsleep_rem_mix,
                token_head_max_vocab=args.microsleep_token_head_max_vocab,
                token_head_scale=args.microsleep_token_head_scale,
                refresh_interval=args.standalone_refresh_interval,
                refresh_steps=args.standalone_refresh_steps,
                refresh_cb_topk=args.standalone_refresh_cb_topk,
                refresh_metric_rank=args.standalone_refresh_metric_rank,
                refresh_noise_scale=args.standalone_refresh_noise_scale,
                refresh_pq=False,
                pq_subdim=64,
                pq_bits=8,
                pq_iters=8,
                pq_batch_size=4096,
                pq_sample_size=16384,
                guard_min_top10_delta=args.microsleep_guard_min_top10_delta,
                guard_min_top50_delta=args.microsleep_guard_min_top50_delta,
                guard_max_top10_drop=args.microsleep_guard_max_top10_drop,
                guard_max_top50_drop=args.microsleep_guard_max_top50_drop,
                guard_max_phase_grounding_risk_increase=(
                    args.microsleep_guard_max_phase_grounding_risk_increase
                ),
            )
            if event is not None:
                if event["accepted"]:
                    microsleep_accepted += 1
                else:
                    microsleep_rejected += 1
                microsleep_events.append(event)
                safe_print(
                    f"    [microsleep] accepted={event['accepted']}  buffer={event['buffer_size']}  "
                    f"guard_top10={event['guard_before']['top10_acc']:.3f}"
                    f"->{event['guard_effective']['top10_acc']:.3f}  "
                    f"guard_top50={event['guard_before']['top50_acc']:.3f}"
                    f"->{event['guard_effective']['top50_acc']:.3f}  "
                    f"phase={event['guard_before'].get('phase_grounding_risk', 0.0):.3f}"
                    f"->{event['guard_effective'].get('phase_grounding_risk', 0.0):.3f}"
                )

    mem_after = mem
    microsleep_report = None
    if evaluate_guard_set is not None:
        microsleep_guard_final = evaluate_guard_set(
            eng,
            microsleep_guard_prompts,
            args,
            max_new_tokens=args.microsleep_tokens,
            refresh_interval=args.standalone_refresh_interval,
            refresh_steps=args.standalone_refresh_steps,
            refresh_cb_topk=args.standalone_refresh_cb_topk,
            refresh_metric_rank=args.standalone_refresh_metric_rank,
            refresh_noise_scale=args.standalone_refresh_noise_scale,
        )
        mem_after = eng.memory_usage()
        microsleep_report = {
            "sleep_every": args.microsleep_every,
            "replay_capacity": args.microsleep_replay_capacity,
            "tokens": args.microsleep_tokens,
            "guard_prompts": microsleep_guard_prompts,
            "initial_guard": microsleep_guard_initial,
            "final_guard": microsleep_guard_final,
            "accepted": microsleep_accepted,
            "rejected": microsleep_rejected,
            "events": microsleep_events,
        }
        safe_print("\n--- Microsleep Summary ---")
        safe_print(
            f"  accepted={microsleep_accepted}  rejected={microsleep_rejected}  "
            f"guard_top10={microsleep_guard_initial['top10_acc']:.3f}"
            f"->{microsleep_guard_final['top10_acc']:.3f}  "
            f"guard_top50={microsleep_guard_initial['top50_acc']:.3f}"
            f"->{microsleep_guard_final['top50_acc']:.3f}  "
            f"phase={microsleep_guard_initial.get('phase_grounding_risk', 0.0):.3f}"
            f"->{microsleep_guard_final.get('phase_grounding_risk', 0.0):.3f}"
        )
        if args.microsleep_output and microsleep_accepted > 0:
            eng.save_runtime_artifact(args.microsleep_output)
            microsleep_report["saved_engine"] = args.microsleep_output
            safe_print(f"  microsleep_engine={args.microsleep_output}")

    result_path = os.path.join(os.path.dirname(args.engine), "engine_results.json")
    payload = {
        "engine": os.path.basename(args.engine),
        "model_name": eng.model_name,
        "device": str(eng.device),
        "backend": args.backend,
        "decode_mode": args.decode_mode,
        "tokens": args.tokens,
        "steps": args.steps,
        "temperature": args.temperature,
        "memory": mem,
        "prompts": results,
    }
    if microsleep_report is not None:
        payload["memory_after"] = mem_after
        payload["microsleep"] = microsleep_report
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    safe_print(f"\n  Results -> {result_path}")


if __name__ == "__main__":
    main()
```
---
## File: `reality_stone/python/reality_stone/clarus/evidence.py`

```python
"""Evidence manifest checks for empirical Clarus gates.

The evolution ladder uses external datasets to promote theoretical terms into
reproducible gates. This module keeps the readiness judgment mechanical: a
dataset moves forward only when the required manifest fields are present.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


READINESS_ORDER = (
    "metadata-only",
    "download-ready",
    "field-ready",
    "gate-ready",
    "reproducible",
)

METADATA_FIELDS = (
    "source_id",
    "species",
    "paper_or_dataset_url",
    "license_or_access",
)

FIELD_FIELDS = (
    "neural_fields",
    "behavior_fields",
    "stimulus_fields",
    "timebase_fields",
    "subject_id_fields",
)

GATE_FIELDS = (
    "train_test_split_rule",
    "baseline_models",
    "null_models",
    "candidate_terms",
    "expected_gate",
)

REPRODUCIBLE_FIELDS = (
    "local_gate_command",
    "local_artifacts",
)

ARTIFACT_FIELDS = (
    "artifact_type",
    "artifact_version",
    "source_id",
    "criteria",
    "gate_passed",
    "result",
)

LOCOMOTION_ARTIFACT_TYPES = (
    "clarus_locomotion_gate",
    "clarus_locomotion_control_gate",
)


@dataclass(frozen=True)
class EvidenceCheck:
    """Result of checking one external evidence manifest."""

    source_id: str
    readiness: str
    missing: tuple[str, ...]
    next_action: str

    @property
    def is_gate_ready(self) -> bool:
        return READINESS_ORDER.index(self.readiness) >= READINESS_ORDER.index("gate-ready")

    @property
    def is_reproducible(self) -> bool:
        return self.readiness == "reproducible"


@dataclass(frozen=True)
class ArtifactCheck:
    """Result of validating a gate artifact."""

    source_id: str
    artifact_type: str
    passed: bool
    missing: tuple[str, ...]
    next_action: str

    @property
    def is_reproducible(self) -> bool:
        return self.passed and not self.missing

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "artifact_type": self.artifact_type,
            "passed": self.passed,
            "is_reproducible": self.is_reproducible,
            "missing": self.missing,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class LinearDecoderGate:
    """Held-out linear decoder result for one neural-to-behavior target."""

    n_train: int
    n_test: int
    r2_model: float
    r2_baseline: float
    delta_r2: float
    p_value: float | None

    @property
    def passed(self) -> bool:
        return self.delta_r2 > 0.0 and (self.p_value is None or self.p_value < 0.05)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_train": self.n_train,
            "n_test": self.n_test,
            "r2_model": self.r2_model,
            "r2_baseline": self.r2_baseline,
            "delta_r2": self.delta_r2,
            "p_value": self.p_value,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class LocomotionGatePanel:
    """Panel-level C. elegans locomotion decoder gate result."""

    results: Mapping[str, Mapping[str, LinearDecoderGate]]

    @property
    def recording_count(self) -> int:
        return len(self.results)

    def pass_count(self, target: str) -> int:
        return sum(1 for result in self.results.values() if result[target].passed)

    def pass_rate(self, target: str) -> float:
        if not self.results:
            return 0.0
        return self.pass_count(target) / len(self.results)

    def summary(self, targets: tuple[str, ...] = ("velocity", "curvature")) -> dict[str, object]:
        target_summary: dict[str, object] = {}
        for target in targets:
            gates = [result[target] for result in self.results.values() if target in result]
            if not gates:
                target_summary[target] = {
                    "pass_count": 0,
                    "pass_rate": 0.0,
                    "mean_delta_r2": 0.0,
                    "mean_r2_model": 0.0,
                    "mean_r2_baseline": 0.0,
                }
                continue
            target_summary[target] = {
                "pass_count": sum(1 for gate in gates if gate.passed),
                "pass_rate": sum(1 for gate in gates if gate.passed) / len(gates),
                "mean_delta_r2": float(np.mean([gate.delta_r2 for gate in gates])),
                "mean_r2_model": float(np.mean([gate.r2_model for gate in gates])),
                "mean_r2_baseline": float(np.mean([gate.r2_baseline for gate in gates])),
            }
        return {
            "recording_count": self.recording_count,
            "targets": target_summary,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "recordings": {
                recording_id: {
                    target: gate.to_dict()
                    for target, gate in target_results.items()
                }
                for recording_id, target_results in self.results.items()
            },
        }

    def passed(
        self,
        *,
        min_pass_rate: float,
        targets: tuple[str, ...] = ("velocity", "curvature"),
    ) -> bool:
        return all(self.pass_rate(target) >= min_pass_rate for target in targets)


@dataclass(frozen=True)
class LocomotionControlComparison:
    """Treatment-vs-control comparison for a locomotion decoder gate."""

    treatment: LocomotionGatePanel
    control: LocomotionGatePanel
    min_pass_rate: float
    min_control_delta: float
    targets: tuple[str, ...] = ("velocity", "curvature")

    def target_summary(self, target: str) -> dict[str, object]:
        treatment_rate = self.treatment.pass_rate(target)
        control_rate = self.control.pass_rate(target)
        delta = treatment_rate - control_rate
        return {
            "treatment_pass_rate": treatment_rate,
            "control_pass_rate": control_rate,
            "pass_rate_delta": delta,
            "passed": treatment_rate >= self.min_pass_rate and delta >= self.min_control_delta,
        }

    @property
    def passed(self) -> bool:
        return all(bool(self.target_summary(target)["passed"]) for target in self.targets)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "min_pass_rate": self.min_pass_rate,
            "min_control_delta": self.min_control_delta,
            "targets": {target: self.target_summary(target) for target in self.targets},
            "treatment": self.treatment.to_dict(),
            "control": self.control.to_dict(),
        }


def assess_manifest(manifest: Mapping[str, object]) -> EvidenceCheck:
    """Assess how far an external dataset can move through the gate pipeline."""

    source_id = str(manifest.get("source_id") or "unknown")
    missing_metadata = _missing(manifest, METADATA_FIELDS)
    if missing_metadata:
        return EvidenceCheck(
            source_id=source_id,
            readiness="metadata-only",
            missing=missing_metadata,
            next_action="fill required source metadata before using this dataset",
        )

    missing_download = _missing(manifest, ("raw_files",))
    if missing_download:
        return EvidenceCheck(
            source_id=source_id,
            readiness="metadata-only",
            missing=missing_download,
            next_action="locate public raw file URLs or document the API access path",
        )

    missing_fields = _missing(manifest, FIELD_FIELDS)
    if missing_fields:
        return EvidenceCheck(
            source_id=source_id,
            readiness="download-ready",
            missing=missing_fields,
            next_action="inspect files and map neural, behavior, stimulus, timebase, and subject fields",
        )

    missing_gate = _missing(manifest, GATE_FIELDS)
    if missing_gate:
        return EvidenceCheck(
            source_id=source_id,
            readiness="field-ready",
            missing=missing_gate,
            next_action="define split rule, baselines, nulls, candidate terms, and expected gate",
        )

    missing_repro = _missing(manifest, REPRODUCIBLE_FIELDS)
    if missing_repro:
        return EvidenceCheck(
            source_id=source_id,
            readiness="gate-ready",
            missing=missing_repro,
            next_action="add a local command and generated artifacts after running the gate",
        )

    return EvidenceCheck(
        source_id=source_id,
        readiness="reproducible",
        missing=(),
        next_action="rerun the local gate before promoting or editing equations",
    )


def validate_locomotion_artifact(artifact: Mapping[str, object]) -> ArtifactCheck:
    """Validate whether a locomotion JSON artifact can support promotion."""

    source_id = str(artifact.get("source_id") or "unknown")
    artifact_type = str(artifact.get("artifact_type") or "unknown")
    missing = _missing(artifact, ARTIFACT_FIELDS)
    if missing:
        return ArtifactCheck(
            source_id=source_id,
            artifact_type=artifact_type,
            passed=False,
            missing=missing,
            next_action="rerun the gate to produce a complete artifact",
        )
    if artifact_type not in LOCOMOTION_ARTIFACT_TYPES:
        return ArtifactCheck(
            source_id=source_id,
            artifact_type=artifact_type,
            passed=False,
            missing=("artifact_type",),
            next_action="use a supported locomotion artifact type",
        )
    if artifact.get("gate_passed") is not True:
        return ArtifactCheck(
            source_id=source_id,
            artifact_type=artifact_type,
            passed=False,
            missing=(),
            next_action="gate artifact is complete but did not pass",
        )
    return ArtifactCheck(
        source_id=source_id,
        artifact_type=artifact_type,
        passed=True,
        missing=(),
        next_action="attach this artifact as local_artifacts before promotion",
    )


def validate_locomotion_artifact_file(path: str | Path) -> ArtifactCheck:
    """Read and validate a locomotion gate artifact JSON file."""

    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        return ArtifactCheck(
            source_id="unknown",
            artifact_type="unknown",
            passed=False,
            missing=("artifact",),
            next_action="artifact JSON must be an object",
        )
    return validate_locomotion_artifact(artifact)


def celegans_elife_66135_manifest() -> dict[str, object]:
    """Known manifest facts for the first C. elegans locomotion target."""

    return {
        "source_id": "celegans_elife_66135_locomotion",
        "species": "C. elegans",
        "paper_or_dataset_url": "https://elifesciences.org/articles/66135",
        "license_or_access": "public OSF data and GPL-2.0 analysis code; verify file-level terms before redistribution",
        "raw_files": (
            "https://osf.io/dpr3h/",
            "https://github.com/leiferlab/PredictionCode",
        ),
        "recording_files": (
            "centerline.mat",
            "heatData.mat",
            "heatDataMS.mat",
            "pointStatsNew.mat",
            "positionDataMS.mat",
        ),
        "neural_fields": (
            "Neurons.I_smooth_interp_crop_noncontig",
            "derived neuron_derivatives = d/dt(Neurons.I_smooth_interp_crop_noncontig)",
        ),
        "behavior_fields": (
            "Behavior_crop_noncontig.CMSVelocity",
            "Behavior_crop_noncontig.Curvature",
        ),
        "stimulus_fields": (
            "strain_condition directory, e.g. AML310_moving, AML32_moving, AML18_moving",
            "BFP cutoff volume for AML310 identity recordings",
        ),
        "timebase_fields": (
            "Neurons.I_Time_crop_noncontig",
            "heatDataMS.hasPointsTime",
            "heatDataMS.clTime",
        ),
        "subject_id_fields": (
            "strain_condition",
            "recording folder key",
        ),
        "field_mapping_source": "leiferlab/PredictionCode utility/get_all_recordings.py and utility/data_handler.py",
        "train_test_split_rule": "held-out recording or blocked time split",
        "baseline_models": ("behavior autocorrelation baseline",),
        "null_models": ("time-shuffled neural activity", "recording-label permutation"),
        "candidate_terms": ("neural population -> velocity/curvature decoder",),
        "expected_gate": "positive held-out decoding over baseline and nulls",
    }


def linear_decoder_gate(
    features: object,
    target: object,
    *,
    train_fraction: float = 0.7,
    ridge: float = 1.0,
    n_permutations: int = 0,
    seed: int = 0,
) -> LinearDecoderGate:
    """Run a minimal held-out ridge decoder gate.

    ``features`` may be shaped as ``time x features`` or ``features x time``.
    The baseline is the training-set mean target. Permutations shuffle the
    training target, giving a one-sided p-value for ``delta_r2``.
    """

    x, y = _clean_xy(features, target)
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    n_train = int(x.shape[0] * train_fraction)
    if n_train < 2 or x.shape[0] - n_train < 2:
        raise ValueError("need at least two train and two test samples after cleaning")

    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    x_train, x_test = _standardize_train_test(x_train, x_test)

    model_pred = _fit_predict_ridge(x_train, y_train, x_test, ridge)
    baseline_pred = np.full_like(y_test, float(np.mean(y_train)), dtype=float)
    r2_model = _r2(y_test, model_pred)
    r2_baseline = _r2(y_test, baseline_pred)
    delta_r2 = r2_model - r2_baseline

    p_value = None
    if n_permutations > 0:
        rng = np.random.default_rng(seed)
        null_delta = np.empty(n_permutations, dtype=float)
        for idx in range(n_permutations):
            shuffled = rng.permutation(y_train)
            shuffled_pred = _fit_predict_ridge(x_train, shuffled, x_test, ridge)
            null_delta[idx] = _r2(y_test, shuffled_pred) - r2_baseline
        p_value = float((np.sum(null_delta >= delta_r2) + 1) / (n_permutations + 1))

    return LinearDecoderGate(
        n_train=n_train,
        n_test=x.shape[0] - n_train,
        r2_model=float(r2_model),
        r2_baseline=float(r2_baseline),
        delta_r2=float(delta_r2),
        p_value=p_value,
    )


def celegans_locomotion_gate(
    recordings: Mapping[str, Mapping[str, object]],
    *,
    targets: tuple[str, ...] = ("velocity", "curvature"),
    include_derivatives: bool = True,
    train_fraction: float = 0.7,
    ridge: float = 1.0,
    n_permutations: int = 0,
    seed: int = 0,
) -> LocomotionGatePanel:
    """Run velocity/curvature gates on PredictionCode preprocessed recordings."""

    results: dict[str, dict[str, LinearDecoderGate]] = {}
    for recording_id, recording in recordings.items():
        features = _celegans_features(recording, include_derivatives=include_derivatives)
        target_results: dict[str, LinearDecoderGate] = {}
        for target in targets:
            if target not in recording:
                raise KeyError(f"recording {recording_id!r} is missing target {target!r}")
            target_results[target] = linear_decoder_gate(
                features,
                recording[target],
                train_fraction=train_fraction,
                ridge=ridge,
                n_permutations=n_permutations,
                seed=seed,
            )
        results[recording_id] = target_results
    return LocomotionGatePanel(results=results)


def celegans_locomotion_gate_from_pickle(
    path: str | Path,
    *,
    targets: tuple[str, ...] = ("velocity", "curvature"),
    include_derivatives: bool = True,
    train_fraction: float = 0.7,
    ridge: float = 1.0,
    n_permutations: int = 0,
    seed: int = 0,
) -> LocomotionGatePanel:
    """Load a PredictionCode ``*_recordings.dat`` pickle and run the panel gate."""

    with Path(path).open("rb") as handle:
        recordings = pickle.load(handle, encoding="latin1")
    return celegans_locomotion_gate(
        recordings,
        targets=targets,
        include_derivatives=include_derivatives,
        train_fraction=train_fraction,
        ridge=ridge,
        n_permutations=n_permutations,
        seed=seed,
    )


def build_locomotion_gate_artifact(
    panel: LocomotionGatePanel,
    *,
    source_id: str = "celegans_elife_66135_locomotion",
    recordings_pickle: str | None = None,
    min_pass_rate: float = 0.0,
    permutations: int = 0,
    ridge: float = 1.0,
    train_fraction: float = 0.7,
    include_derivatives: bool = True,
) -> dict[str, object]:
    """Build a self-describing JSON artifact for one locomotion gate run."""

    return {
        "artifact_type": "clarus_locomotion_gate",
        "artifact_version": 1,
        "source_id": source_id,
        "recordings_pickle": recordings_pickle,
        "criteria": {
            "min_pass_rate": min_pass_rate,
            "permutations": permutations,
            "ridge": ridge,
            "train_fraction": train_fraction,
            "include_derivatives": include_derivatives,
        },
        "gate_passed": panel.passed(min_pass_rate=min_pass_rate),
        "result": panel.to_dict(),
    }


def build_locomotion_control_artifact(
    comparison: LocomotionControlComparison,
    *,
    source_id: str = "celegans_elife_66135_locomotion",
    treatment_pickle: str | None = None,
    control_pickle: str | None = None,
    permutations: int = 0,
    ridge: float = 1.0,
    train_fraction: float = 0.7,
    include_derivatives: bool = True,
) -> dict[str, object]:
    """Build a self-describing JSON artifact for a treatment-vs-control gate."""

    return {
        "artifact_type": "clarus_locomotion_control_gate",
        "artifact_version": 1,
        "source_id": source_id,
        "treatment_pickle": treatment_pickle,
        "control_pickle": control_pickle,
        "criteria": {
            "min_pass_rate": comparison.min_pass_rate,
            "min_control_delta": comparison.min_control_delta,
            "permutations": permutations,
            "ridge": ridge,
            "train_fraction": train_fraction,
            "include_derivatives": include_derivatives,
        },
        "gate_passed": comparison.passed,
        "result": comparison.to_dict(),
    }


def compare_locomotion_to_control(
    treatment: LocomotionGatePanel,
    control: LocomotionGatePanel,
    *,
    min_pass_rate: float,
    min_control_delta: float,
    targets: tuple[str, ...] = ("velocity", "curvature"),
) -> LocomotionControlComparison:
    """Compare an experimental locomotion panel against a matched control panel."""

    return LocomotionControlComparison(
        treatment=treatment,
        control=control,
        min_pass_rate=min_pass_rate,
        min_control_delta=min_control_delta,
        targets=targets,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the C. elegans locomotion decoder gate on a PredictionCode pickle."
    )
    parser.add_argument(
        "recordings_pickle",
        nargs="?",
        help="Path to gcamp_recordings.dat or gfp_recordings.dat",
    )
    parser.add_argument("--validate-artifact", help="Validate an existing gate artifact JSON")
    parser.add_argument("--control-pickle", help="Optional GFP/control recordings pickle")
    parser.add_argument("--permutations", type=int, default=0, help="Permutation count for p-values")
    parser.add_argument("--ridge", type=float, default=1.0, help="Ridge penalty")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Blocked train fraction")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for permutations")
    parser.add_argument("--output", help="Optional path to write the JSON result artifact")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.0,
        help="Required pass rate for every target; returns exit code 2 on failure",
    )
    parser.add_argument(
        "--min-control-delta",
        type=float,
        default=0.0,
        help="Required treatment minus control pass-rate margin when --control-pickle is used",
    )
    parser.add_argument(
        "--no-derivatives",
        action="store_true",
        help="Use only neural activity, without neuron_derivatives",
    )
    args = parser.parse_args(argv)
    if args.validate_artifact:
        check = validate_locomotion_artifact_file(args.validate_artifact)
        print(json.dumps(check.to_dict(), indent=2, sort_keys=True))
        return 0 if check.is_reproducible else 2
    if not args.recordings_pickle:
        parser.error("recordings_pickle is required unless --validate-artifact is provided")
    if not 0.0 <= args.min_pass_rate <= 1.0:
        parser.error("--min-pass-rate must be between 0 and 1")
    if not 0.0 <= args.min_control_delta <= 1.0:
        parser.error("--min-control-delta must be between 0 and 1")
    panel = celegans_locomotion_gate_from_pickle(
        args.recordings_pickle,
        include_derivatives=not args.no_derivatives,
        train_fraction=args.train_fraction,
        ridge=args.ridge,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    if args.control_pickle:
        control = celegans_locomotion_gate_from_pickle(
            args.control_pickle,
            include_derivatives=not args.no_derivatives,
            train_fraction=args.train_fraction,
            ridge=args.ridge,
            n_permutations=args.permutations,
            seed=args.seed,
        )
        comparison = compare_locomotion_to_control(
            panel,
            control,
            min_pass_rate=args.min_pass_rate,
            min_control_delta=args.min_control_delta,
        )
        payload = build_locomotion_control_artifact(
            comparison,
            treatment_pickle=args.recordings_pickle,
            control_pickle=args.control_pickle,
            permutations=args.permutations,
            ridge=args.ridge,
            train_fraction=args.train_fraction,
            include_derivatives=not args.no_derivatives,
        )
        passed = comparison.passed
    else:
        payload = build_locomotion_gate_artifact(
            panel,
            recordings_pickle=args.recordings_pickle,
            min_pass_rate=args.min_pass_rate,
            permutations=args.permutations,
            ridge=args.ridge,
            train_fraction=args.train_fraction,
            include_derivatives=not args.no_derivatives,
        )
        passed = panel.passed(min_pass_rate=args.min_pass_rate)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if passed else 2


def _missing(manifest: Mapping[str, object], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(field for field in fields if _empty(manifest.get(field)))


def _empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return len(value) == 0
    return False


def _clean_xy(features: object, target: object) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if x.shape[0] != y.shape[0] and x.shape[1] == y.shape[0]:
        x = x.T
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and target must share the time dimension")

    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    return x[finite], y[finite]


def _celegans_features(
    recording: Mapping[str, object],
    *,
    include_derivatives: bool,
) -> np.ndarray:
    if "neurons" not in recording:
        raise KeyError("recording is missing neural activity field 'neurons'")

    neurons = np.asarray(recording["neurons"], dtype=float)
    if not include_derivatives:
        return neurons

    derivatives = recording.get("neuron_derivatives")
    if derivatives is None:
        return neurons
    return np.vstack([neurons, np.asarray(derivatives, dtype=float)])


def _standardize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std


def _fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    ridge: float,
) -> np.ndarray:
    x_train_i = np.column_stack([np.ones(x_train.shape[0]), x_train])
    x_test_i = np.column_stack([np.ones(x_test.shape[0]), x_test])
    penalty = np.eye(x_train_i.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x_train_i.T @ x_train_i + penalty, x_train_i.T @ y_train)
    return x_test_i @ weights


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0.0:
        return 0.0
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom


__all__ = [
    "ArtifactCheck",
    "EvidenceCheck",
    "LinearDecoderGate",
    "LocomotionControlComparison",
    "LocomotionGatePanel",
    "READINESS_ORDER",
    "assess_manifest",
    "build_locomotion_control_artifact",
    "build_locomotion_gate_artifact",
    "celegans_elife_66135_manifest",
    "celegans_locomotion_gate",
    "celegans_locomotion_gate_from_pickle",
    "compare_locomotion_to_control",
    "linear_decoder_gate",
    "main",
    "validate_locomotion_artifact",
    "validate_locomotion_artifact_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
```
---
## File: `reality_stone/python/reality_stone/clarus/neuromod.py`

```python
"""Four neuromodulator system (17_AgentLoop.md F.19 / F.24.4).

g_DA[t+1] = g_DA[t] + (1/tau_DA)(g0_DA - g_DA[t]) + alpha_DA * c_pred[t]
g_NE[t+1] = g_NE[t] + (1/tau_NE)(g0_NE - g_NE[t]) + alpha_NE * c_nov[t]
g_5HT[t+1] = g_5HT[t] + (1/tau_5HT)(g0_5HT - g_5HT[t]) + alpha_5HT * (-discount[t])
g_ACh[t+1] = g_ACh[t] + (1/tau_ACh)(g0_ACh - g_ACh[t]) + alpha_ACh * salience[t]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from .constants import (
        NEURO_TAU_DA, NEURO_TAU_NE, NEURO_TAU_5HT, NEURO_TAU_ACH,
        NEURO_BASELINE_DA, NEURO_BASELINE_NE, NEURO_BASELINE_5HT, NEURO_BASELINE_ACH,
        NEURO_ALPHA_DA, NEURO_ALPHA_NE, NEURO_ALPHA_5HT, NEURO_ALPHA_ACH,
    )
except ImportError:
    from reality_stone.clarus.constants import (
        NEURO_TAU_DA, NEURO_TAU_NE, NEURO_TAU_5HT, NEURO_TAU_ACH,
        NEURO_BASELINE_DA, NEURO_BASELINE_NE, NEURO_BASELINE_5HT, NEURO_BASELINE_ACH,
        NEURO_ALPHA_DA, NEURO_ALPHA_NE, NEURO_ALPHA_5HT, NEURO_ALPHA_ACH,
    )


@dataclass
class NeuromodulatorState:
    """Four neuromodulator levels."""
    da: float = NEURO_BASELINE_DA
    ne: float = NEURO_BASELINE_NE
    sht: float = NEURO_BASELINE_5HT
    ach: float = NEURO_BASELINE_ACH

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.da, self.ne, self.sht, self.ach)


def step_neuromodulators(
    state: NeuromodulatorState,
    c_pred: float = 0.0,
    c_nov: float = 0.0,
    discount: float = 0.0,
    salience: float = 0.0,
) -> NeuromodulatorState:
    """Update 4 neuromodulator levels for one timestep (F.24.4)."""
    da = state.da + (1.0 / NEURO_TAU_DA) * (NEURO_BASELINE_DA - state.da) + NEURO_ALPHA_DA * c_pred
    ne = state.ne + (1.0 / NEURO_TAU_NE) * (NEURO_BASELINE_NE - state.ne) + NEURO_ALPHA_NE * c_nov
    sht = state.sht + (1.0 / NEURO_TAU_5HT) * (NEURO_BASELINE_5HT - state.sht) + NEURO_ALPHA_5HT * (-discount)
    ach = state.ach + (1.0 / NEURO_TAU_ACH) * (NEURO_BASELINE_ACH - state.ach) + NEURO_ALPHA_ACH * salience
    return NeuromodulatorState(
        da=max(0.0, min(da, 2.0)),
        ne=max(0.0, min(ne, 2.0)),
        sht=max(0.0, min(sht, 2.0)),
        ach=max(0.0, min(ach, 2.0)),
    )


@dataclass
class ModulationEffect:
    """Effects of neuromodulation on runtime parameters (F.19)."""
    n_iter_boost: float
    encode_threshold_scale: float
    temperature_scale: float
    exploration_boost: float


def apply_modulation(
    state: NeuromodulatorState,
    base_n_iter: int = 60,
    delta_n: int = 30,
    base_encode_threshold: float = 0.3,
    base_temperature: float = 0.3148,
    beta_5ht: float = 0.5,
) -> ModulationEffect:
    """Map neuromodulator levels to runtime parameter modulations (F.19)."""
    sigmoid_ne = 1.0 / (1.0 + math.exp(-2.0 * (state.ne - 0.5)))
    n_iter = base_n_iter + int(delta_n * sigmoid_ne)
    theta_encode = base_encode_threshold / (1.0 + state.ach)
    t_eff = base_temperature * (1.0 + beta_5ht * state.sht)
    exploration = state.da * 0.5
    return ModulationEffect(
        n_iter_boost=float(n_iter),
        encode_threshold_scale=theta_encode,
        temperature_scale=t_eff,
        exploration_boost=exploration,
    )
```
---
## File: `reality_stone/python/reality_stone/clarus/ops.py`

```python
"""Tensor-friendly facade over the Rust ``reality_stone.clarus._rust`` neural ops.

Runtime integrations consume these wrappers via
``from reality_stone.clarus.ops import topk_silu, lbo_fused_fwd, power_iter_step,
gauge_lattice_fwd, ops_backend``. The wrappers convert between
PyTorch tensors and the contiguous f32 ``numpy`` arrays expected by the
Rust pyfunctions.

CUDA dispatch lives in :mod:`reality_stone.clarus.kernels`. CPU dispatch (the path
this module covers) routes through Rust when ``reality_stone.clarus._rust`` is built;
otherwise we fall back to a pure-PyTorch implementation that matches the
Rust kernel's mathematical contract.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import _rust as _r
    _HAS_RUST = True
except ImportError:
    _r = None
    _HAS_RUST = False

from .ce_ops import _as_cpu_numpy_flat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_flat_f32(t: torch.Tensor) -> np.ndarray:
    """Float32 view used by ``nn_*`` Rust pyfunctions. Wraps the dtype-preserving
    helper from :mod:`reality_stone.clarus.ce_ops` and adds the f32 cast Rust expects."""
    arr = _as_cpu_numpy_flat(t)
    return arr if arr.dtype == np.float32 else arr.astype(np.float32, copy=False)


def _from_flat(arr: np.ndarray, shape: Tuple[int, ...], device, dtype) -> torch.Tensor:
    out = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    if shape:
        out = out.view(*shape)
    return out.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# TopK SiLU
# ---------------------------------------------------------------------------
def topk_silu(x: torch.Tensor, k: int, ratio: float, threshold: float = 0.0) -> torch.Tensor:
    """Fused SiLU + per-row Top-K masking.

    The Rust kernel ignores ``threshold`` and recomputes the per-row threshold
    from ``ratio``. ``threshold`` is accepted for API parity with older callers.
    """
    dim = int(x.shape[-1])
    if k >= dim or ratio >= 1.0:
        return F.silu(x)

    if _HAS_RUST and not x.is_cuda:
        flat = _as_flat_f32(x)
        out_flat, _mask = _r.nn_topk_silu_fwd(flat, dim, float(ratio))
        return _from_flat(out_flat, x.shape, x.device, x.dtype)

    h = F.silu(x)
    abs_h = h.abs()
    thr = abs_h.kthvalue(dim - k + 1, dim=-1, keepdim=True).values
    return h.masked_fill(abs_h < thr, 0.0)


# ---------------------------------------------------------------------------
# LBO fused forward
# ---------------------------------------------------------------------------
def lbo_fused_fwd(
    x_normed: torch.Tensor,
    V: torch.Tensor,
    h: float,
    scale: torch.Tensor,
    bias: torch.Tensor,
    alpha_conf: float,
    dim: int,
    rank: int,
    *,
    need_curvature: bool = True,
) -> Tuple[torch.Tensor, float]:
    """Fused (post-LayerNorm) Laplace-Beltrami diffusion.

    out = ((1-h)*x + h*x @ V_eff^T @ V_eff) * scale + bias
    where V_eff = V * exp(-alpha_conf * mean(x^2)).
    """
    if _HAS_RUST and not x_normed.is_cuda:
        flat_x = _as_flat_f32(x_normed)
        flat_v = _as_flat_f32(V)
        out_flat, curv = _r.nn_lbo_fused_fwd(
            flat_x,
            flat_v,
            float(h),
            _as_flat_f32(scale),
            _as_flat_f32(bias),
            float(alpha_conf),
            int(dim),
            int(rank),
        )
        out = _from_flat(out_flat, x_normed.shape, x_normed.device, x_normed.dtype)
        return out, float(curv) if need_curvature else 0.0

    phi_sq = x_normed.detach().pow(2).mean()
    conformal = torch.exp(-abs(alpha_conf) * phi_sq)
    v_eff = V * conformal
    proj = x_normed @ v_eff.t()
    xw = proj @ v_eff
    pre = torch.lerp(x_normed, xw, h)
    out = torch.addcmul(bias, pre, scale)
    curvature = 0.0
    if need_curvature:
        lx = x_normed - xw
        curvature = float((lx.detach().pow(2).sum() / lx.numel()).item())
    return out, curvature


# ---------------------------------------------------------------------------
# Power iteration step
# ---------------------------------------------------------------------------
def power_iter_step(
    V: torch.Tensor,
    spectral_v: torch.Tensor,
    dim: int,
    rank: int,
) -> Tuple[torch.Tensor, float]:
    """One step of power iteration to estimate sigma_max(V)."""
    if _HAS_RUST and not V.is_cuda:
        new_v_flat, sigma = _r.nn_power_iter(
            _as_flat_f32(V),
            _as_flat_f32(spectral_v),
            int(dim),
            int(rank),
        )
        new_v = _from_flat(new_v_flat, (dim,), V.device, V.dtype)
        return new_v, float(sigma)

    u = F.normalize(V @ spectral_v, dim=0)
    new_v = F.normalize(V.t() @ u, dim=0)
    sigma = float((V @ new_v).norm().item())
    return new_v, sigma


# ---------------------------------------------------------------------------
# Gauge lattice forward
# ---------------------------------------------------------------------------
def gauge_lattice_fwd(
    x: torch.Tensor,
    su3_up: torch.Tensor, su3_down: torch.Tensor,
    su2_up: torch.Tensor, su2_down: torch.Tensor,
    u1_up: torch.Tensor, u1_down: torch.Tensor,
    mix_down: torch.Tensor | None,
    mix_up: torch.Tensor | None,
    *,
    d3: int, d2: int, d1: int,
    h3: int, h2: int, h1: int,
    mix_rank: int,
    ratio: float,
    dim: int,
) -> torch.Tensor:
    """3x3+1 gauge lattice forward with optional cross-channel mixing."""
    if _HAS_RUST and not x.is_cuda:
        empty = np.zeros(0, dtype=np.float32)
        out_flat = _r.nn_gauge_lattice_fwd(
            _as_flat_f32(x),
            _as_flat_f32(su3_up), _as_flat_f32(su3_down),
            _as_flat_f32(su2_up), _as_flat_f32(su2_down),
            _as_flat_f32(u1_up), _as_flat_f32(u1_down),
            empty if mix_down is None else _as_flat_f32(mix_down),
            empty if mix_up is None else _as_flat_f32(mix_up),
            int(d3), int(d2), int(d1),
            int(h3), int(h2), int(h1),
            int(mix_rank),
            float(ratio),
            int(dim),
        )
        return _from_flat(out_flat, x.shape, x.device, x.dtype)

    s3 = d3
    s32 = d3 + d2
    k3 = max(1, math.ceil(ratio * h3))
    k2 = max(1, math.ceil(ratio * h2))
    k1 = max(1, math.ceil(ratio * h1))

    def channel(x_part, up, down, k, hid):
        h = F.silu(F.linear(x_part, up))
        if k < hid:
            abs_h = h.abs()
            thr = abs_h.kthvalue(hid - k + 1, dim=-1, keepdim=True).values
            h = h.masked_fill(abs_h < thr, 0.0)
        return F.linear(h, down)

    y3 = channel(x[..., :s3], su3_up, su3_down, k3, h3)
    y2 = channel(x[..., s3:s32], su2_up, su2_down, k2, h2)
    y1 = channel(x[..., s32:], u1_up, u1_down, k1, h1)
    y = torch.cat([y3, y2, y1], dim=-1)
    if mix_rank > 0 and mix_down is not None and mix_up is not None:
        y = y + F.linear(F.linear(y, mix_down), mix_up)
    return y


# ---------------------------------------------------------------------------
# Backend introspection
# ---------------------------------------------------------------------------
def ops_backend() -> str:
    """Identifier of the active CPU dispatch backend ('rust' or 'torch')."""
    return "rust" if _HAS_RUST else "torch"


__all__ = [
    "topk_silu",
    "lbo_fused_fwd",
    "power_iter_step",
    "gauge_lattice_fwd",
    "ops_backend",
]
```
---
## File: `reality_stone/python/reality_stone/clarus/quantum.py`

```python
"""Quantum phase evolution (12_Equation.md 1.5 / 4.1).

psi_{k+1} = exp(-i E dt) psi_k  (quantum form)
Wick rotation t -> -i*tau yields real relaxation.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

ALPHA_B_DEFAULT = math.exp(1.0 / 3.0) * (math.pi ** (1.0 / 3.0))


def quantum_phase_step(
    psi: torch.Tensor,
    energy: float,
    dt: float = 0.01,
) -> torch.Tensor:
    """psi_{k+1} = exp(-i*E*dt) * psi_k. Complex state evolution."""
    if not psi.is_complex():
        psi = torch.complex(psi, torch.zeros_like(psi))
    phase = torch.tensor(-energy * dt)
    rotation = torch.complex(torch.cos(phase), torch.sin(phase))
    return psi * rotation


def wick_rotate(
    psi: torch.Tensor,
    energy: float,
    dt: float = 0.01,
) -> torch.Tensor:
    """Euclidean rotation t -> -i*tau: exp(-E*dt) * psi (real damping)."""
    if psi.is_complex():
        return psi * torch.exp(torch.tensor(-energy * dt))
    return psi * torch.exp(torch.tensor(-energy * dt))


def quantum_to_real(psi: torch.Tensor) -> torch.Tensor:
    """Project complex state to real for classical relaxation."""
    if psi.is_complex():
        return psi.real
    return psi


def check_norm_conservation(psi_before: torch.Tensor, psi_after: torch.Tensor, tol: float = 1e-6) -> bool:
    """Verify unitary evolution preserves norm."""
    norm_before = psi_before.abs().norm()
    norm_after = psi_after.abs().norm()
    return bool(torch.abs(norm_before - norm_after).item() < tol)


def convergence_inequality(
    grad_norm: float,
    c_k: float,
    phi_norm: float,
    alpha_b: float = ALPHA_B_DEFAULT,
) -> bool:
    """Pointwise sufficient condition: ||grad E|| > C_k * ||phi|| / alpha_b.

    Reference: docs/7_AGI/12_Equation.md 4.7 (gate F2). Scope: pointwise
    monotone-energy region; not a global convergence guarantee.
    """
    return grad_norm > c_k * phi_norm / alpha_b


def time_curvature(m_history: Sequence[torch.Tensor]) -> float:
    """C_k = ||m_k - 2 m_{k-1} + m_{k-2}|| (docs/7_AGI/12_Equation.md 1.5).

    Returns 0.0 when fewer than 3 samples are available.
    """
    if len(m_history) < 3:
        return 0.0
    m_k, m_km1, m_km2 = m_history[-1], m_history[-2], m_history[-3]
    return float((m_k - 2.0 * m_km1 + m_km2).norm().item())


def estimate_mu(
    residuals: Sequence[float],
    *,
    dt_over_tau: float,
    skip: int = 0,
) -> float:
    """Estimate local Hessian floor mu from observed residual contraction.

    Near an attractor, the gradient flow gives
        ||m_{k+1} - m*|| approx (1 - mu * dt/tau) * ||m_k - m*||,
    so a least-squares fit on log residuals returns
        mu approx (1 - mean(r_{k+1}/r_k)) * tau / dt.

    Args:
        residuals: sequence of ||m_k - m*|| values (must be > 0).
        dt_over_tau: integration step ratio dt/tau used by `relax`.
        skip: optional warm-up steps to drop from the head of the sequence.

    Returns:
        mu estimate. Returns 0.0 if the sequence is too short or expanding.
    """
    if dt_over_tau <= 0.0:
        raise ValueError("dt_over_tau must be positive")
    seq = [float(r) for r in residuals[skip:] if r > 0.0]
    if len(seq) < 3:
        return 0.0
    log_ratios = []
    for prev, curr in zip(seq[:-1], seq[1:]):
        ratio = curr / prev
        if ratio <= 0.0 or ratio >= 1.0:
            continue
        log_ratios.append(math.log(ratio))
    if not log_ratios:
        return 0.0
    contraction = math.exp(sum(log_ratios) / len(log_ratios))
    mu = (1.0 - contraction) / dt_over_tau
    return max(mu, 0.0)


def iss_ball_radius(
    *,
    c_k_max: float,
    phi_inf_norm: float,
    mu: float,
    alpha_b: float = ALPHA_B_DEFAULT,
) -> float:
    """ISS ball radius for the bypass-driven memory dynamics (gate F2).

    Reference: docs/7_AGI/12_Equation.md appendix A.1. With dm/dt =
    -nabla_m E / tau + d(t) and ||d||_inf <= C_k_max * ||phi||_inf / (tau * alpha_b),
    the Hessian floor mu yields the closed-form bound

        limsup ||m - m*|| <= C_k_max * ||phi||_inf / (mu * alpha_b).

    Args:
        c_k_max: empirical maximum of trajectory time-curvature C_k.
        phi_inf_norm: empirical sup-norm of the residue field phi.
        mu: local Hessian floor (use `estimate_mu` from contraction history).
        alpha_b: bypass denominator e^{1/3} pi^{1/3}.

    Returns:
        ISS ball radius. Returns +inf if mu == 0.
    """
    if mu <= 0.0:
        return float("inf")
    return c_k_max * phi_inf_norm / (mu * alpha_b)


def pci_regression(
    stability: Sequence[float],
    pci: Sequence[float],
) -> dict:
    """OLS regression PCI ~ alpha * stability + beta (gate F4, A.4).

    Returns alpha, beta, R^2, sample count and Pearson r. With less than 3
    samples or zero stability variance, returns NaNs and zero R^2 so callers
    can detect insufficient data without exception handling.

    Stability is the metacognitive scalar exp(-c_d * d_tau); see
    reality_stone.clarus.agent.ConsciousnessMonitor and 12_Equation.md A.4.
    """
    xs = [float(v) for v in stability]
    ys = [float(v) for v in pci]
    n = min(len(xs), len(ys))
    if n < 3:
        return {"n": n, "alpha": float("nan"), "beta": float("nan"), "r2": 0.0, "pearson_r": float("nan")}
    xs = xs[:n]
    ys = ys[:n]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sx2 = sum((x - mean_x) ** 2 for x in xs)
    sy2 = sum((y - mean_y) ** 2 for y in ys)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    if sx2 <= 0.0:
        return {"n": n, "alpha": float("nan"), "beta": float("nan"), "r2": 0.0, "pearson_r": float("nan")}
    alpha = sxy / sx2
    beta = mean_y - alpha * mean_x
    r2 = (sxy * sxy) / (sx2 * sy2) if sy2 > 0.0 else 0.0
    pearson_r = sxy / math.sqrt(sx2 * sy2) if sy2 > 0.0 else float("nan")
    return {"n": n, "alpha": alpha, "beta": beta, "r2": r2, "pearson_r": pearson_r}


def iss_report(
    m_history: Sequence[torch.Tensor],
    phi: torch.Tensor,
    *,
    dt_over_tau: float,
    m_star: torch.Tensor | None = None,
    alpha_b: float = ALPHA_B_DEFAULT,
) -> dict:
    """End-to-end gate F2 measurement from a relaxation trajectory.

    Args:
        m_history: list of m_k tensors recorded during `relax`.
        phi: residue field tensor at end of relaxation (or any reference point).
        dt_over_tau: integrator step ratio.
        m_star: optional fixed point estimate. Defaults to mean of last
            quarter of the trajectory.
        alpha_b: bypass denominator.

    Returns:
        Dict with `c_k_max`, `phi_inf_norm`, `mu`, `iss_ball_radius`,
        `residuals`, `samples`. Suitable for direct logging.
    """
    if not m_history:
        return {
            "samples": 0,
            "c_k_max": 0.0,
            "phi_inf_norm": 0.0,
            "mu": 0.0,
            "iss_ball_radius": float("inf"),
        }

    if m_star is None:
        tail = m_history[max(1, 3 * len(m_history) // 4):]
        if not tail:
            tail = m_history[-1:]
        m_star = torch.stack(list(tail)).mean(dim=0)

    residuals = [float((m_k - m_star).norm().item()) for m_k in m_history]
    c_k_max = 0.0
    for k in range(2, len(m_history)):
        c_k = float(
            (m_history[k] - 2.0 * m_history[k - 1] + m_history[k - 2]).norm().item()
        )
        if c_k > c_k_max:
            c_k_max = c_k
    phi_inf_norm = float(phi.detach().abs().max().item())
    mu = estimate_mu(residuals, dt_over_tau=dt_over_tau, skip=0)
    radius = iss_ball_radius(
        c_k_max=c_k_max, phi_inf_norm=phi_inf_norm, mu=mu, alpha_b=alpha_b
    )
    return {
        "samples": len(m_history),
        "c_k_max": c_k_max,
        "phi_inf_norm": phi_inf_norm,
        "mu": mu,
        "iss_ball_radius": radius,
        "residuals": residuals,
    }
```
---
## File: `reality_stone/python/reality_stone/clarus/reality.py`

```python
"""Optional Reality Stone backend bridge.

This module is the integration boundary between Clarus policy code and the
Reality Stone geometry/RSULF backend. It intentionally keeps imports lazy so
Clarus remains usable when the local `reality_stone` package has not been
built yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


_RS: ModuleType | None = None
_LOAD_ERROR: Exception | None = None


def _ensure_local_source_path() -> None:
    src = Path(__file__).resolve().parents[2]
    if src.exists():
        src_s = str(src)
        if src_s not in sys.path:
            sys.path.insert(0, src_s)


def load_reality_stone() -> ModuleType:
    global _RS, _LOAD_ERROR
    if _RS is not None:
        return _RS
    _ensure_local_source_path()
    try:
        import reality_stone as rs
    except Exception as exc:  # pragma: no cover - reported by status()
        _LOAD_ERROR = exc
        raise
    _RS = rs
    _LOAD_ERROR = None
    return rs


def has_reality_stone() -> bool:
    try:
        load_reality_stone()
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class RealityStoneStatus:
    available: bool
    version: str | None
    rust: bool
    cuda: bool
    error: str | None = None


def status() -> RealityStoneStatus:
    try:
        rs = load_reality_stone()
    except Exception as exc:
        return RealityStoneStatus(False, None, False, False, str(exc))
    return RealityStoneStatus(
        available=True,
        version=getattr(rs, "__version__", None),
        rust=bool(getattr(rs, "_has_rust_ext", False)),
        cuda=bool(getattr(rs, "_has_cuda", False)),
        error=None,
    )


def metric_attention(*args: Any, **kwargs: Any):
    rs = load_reality_stone()
    return rs.MetricAttention(*args, **kwargs)


def unified_riemannian_layer(*args: Any, **kwargs: Any):
    rs = load_reality_stone()
    layer_cls = getattr(rs, "UnifiedRiemannianLayer", None)
    if layer_cls is None:
        raise RuntimeError("Reality Stone UnifiedRiemannianLayer is not available")
    return layer_cls(*args, **kwargs)


def convert_transformer_to_rsulf(model: Any, *args: Any, **kwargs: Any):
    rs = load_reality_stone()
    converter = getattr(getattr(rs, "models", None), "convert_transformer_to_rsulf", None)
    if converter is None:
        raise RuntimeError("Reality Stone transformer converter is not available")
    return converter(model, *args, **kwargs)


__all__ = [
    "RealityStoneStatus",
    "convert_transformer_to_rsulf",
    "has_reality_stone",
    "load_reality_stone",
    "metric_attention",
    "status",
    "unified_riemannian_layer",
]
```
---
## File: `reality_stone/python/reality_stone/clarus/research.py`

```python
"""Research helpers for comparing HF language models with CE runtime probes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode


@dataclass
class RuntimeProbeResult:
    mse: float
    cosine: float
    active_ratio: float
    stdp_updates: int
    stdp_gate: float


@dataclass
class PhaseLockProbeResult:
    initial_alignment: float
    final_alignment: float
    initial_coherence: float
    final_coherence: float
    steps: int


@dataclass
class PhaseNetworkProbeResult:
    initial_coherence: float
    final_coherence: float
    initial_alignment: float | None
    final_alignment: float | None
    steps: int


def pca_projection(hidden: torch.Tensor, max_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Project hidden states to a compact orthonormal basis for runtime probes."""
    hidden = hidden.float()
    if hidden.ndim != 2:
        raise ValueError("hidden must be a 2D tensor")
    max_dim = max(1, int(max_dim))
    if hidden.shape[1] <= max_dim:
        basis = torch.eye(hidden.shape[1], dtype=hidden.dtype, device=hidden.device)
        return hidden, basis

    centered = hidden - hidden.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:max_dim].T.contiguous()
    return centered @ basis, basis


def apply_projection(hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    hidden = hidden.float()
    if hidden.shape[1] == basis.shape[0]:
        centered = hidden - hidden.mean(dim=0, keepdim=True)
        return centered @ basis.to(hidden.device)
    if hidden.shape[1] == basis.shape[1]:
        return hidden
    raise ValueError("hidden dimension does not match projection basis")


def hopfield_from_hidden(hidden: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """Build a stable symmetric coupling matrix from hidden-state covariance."""
    hidden = hidden.float()
    if hidden.ndim != 2:
        raise ValueError("hidden must be a 2D tensor")
    centered = hidden - hidden.mean(dim=0, keepdim=True)
    n, dim = centered.shape
    cov = (centered.T @ centered) / max(n - 1, 1)
    cov = 0.5 * (cov + cov.T)
    cov.fill_diagonal_(0.0)
    lam_max = float(torch.linalg.eigvalsh(cov)[-1].item())
    if lam_max >= -float(ridge):
        cov = cov - (lam_max + float(ridge)) * torch.eye(dim, dtype=cov.dtype, device=cov.device)
    return cov


def normalized_drive(hidden: torch.Tensor, gain: float = 0.4) -> torch.Tensor:
    hidden = hidden.float()
    return F.normalize(hidden, dim=1) * float(gain)


def periodic_stimulus(
    steps: int,
    dim: int,
    *,
    frequency: float,
    amplitude: float = 1.0,
    phase_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a protist-scale periodic cue as complex oscillator modes."""
    steps = max(1, int(steps))
    dim = max(1, int(dim))
    if phase_offsets is None:
        phase_offsets = torch.linspace(0.0, math.pi, dim)
    phase_offsets = phase_offsets.float()
    if phase_offsets.numel() != dim:
        raise ValueError("phase_offsets must match dim")
    time = torch.arange(steps, dtype=torch.float32).unsqueeze(1)
    phase = 2.0 * math.pi * float(frequency) * time + phase_offsets.unsqueeze(0)
    radius = torch.full_like(phase, float(amplitude))
    return torch.polar(radius, phase)


def phase_coherence(state: torch.Tensor, *, eps: float = 1e-8) -> float:
    """Kuramoto order parameter for internal phase agreement."""
    state = _as_complex(state)
    mask = state.abs() > float(eps)
    if not bool(mask.any().item()):
        return 0.0
    phasor = state[mask] / state[mask].abs().clamp_min(float(eps))
    return float(phasor.mean().abs().item())


def phase_alignment(state: torch.Tensor, reference: torch.Tensor, *, eps: float = 1e-8) -> float:
    """Mean cosine alignment against an external grounding phase."""
    state = _as_complex(state)
    reference = _as_complex(reference).to(state.device)
    if state.shape != reference.shape:
        raise ValueError("state and reference must have the same shape")
    mask = (state.abs() > float(eps)) & (reference.abs() > float(eps))
    if not bool(mask.any().item()):
        return 0.0
    delta = torch.angle(state[mask]) - torch.angle(reference[mask])
    return float(torch.cos(delta).mean().item())


def phase_grounding_risk(
    state: torch.Tensor,
    reference: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> float:
    """Risk is high when internal phase agreement lacks external grounding."""
    coherence = phase_coherence(state, eps=eps)
    alignment = phase_alignment(state, reference, eps=eps)
    grounding = min(max((alignment + 1.0) * 0.5, 0.0), 1.0)
    return float(coherence * (1.0 - grounding))


def phase_grounding_suppression(
    logits: torch.Tensor,
    candidate_states: torch.Tensor,
    reference: torch.Tensor,
    *,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Suppress candidates whose coherent phase is not grounded in the cue."""
    if candidate_states.ndim != 2:
        raise ValueError("candidate_states must be a 2D tensor")
    logits = logits.float()
    if logits.shape[0] != candidate_states.shape[0]:
        raise ValueError("logits and candidate_states must have the same first dimension")
    risks = torch.tensor(
        [phase_grounding_risk(candidate, reference) for candidate in candidate_states],
        dtype=logits.dtype,
        device=logits.device,
    )
    return logits - float(strength) * risks, risks


def phase_coupling_step(
    state: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    neighbor_coupling: float,
    external: torch.Tensor | None = None,
    external_coupling: float = 0.0,
) -> torch.Tensor:
    """Kuramoto-style graph coupling for early network synchronization."""
    state = _as_complex(state)
    adjacency = adjacency.to(state.device).float()
    if adjacency.shape != (state.numel(), state.numel()):
        raise ValueError("adjacency must be square with size matching state")

    theta = torch.angle(state)
    pair_delta = theta.unsqueeze(0) - theta.unsqueeze(1)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    neighbor_pull = (adjacency * torch.sin(pair_delta)).sum(dim=1) / degree
    phase = theta + float(neighbor_coupling) * neighbor_pull

    if external is not None:
        external = _as_complex(external).to(state.device)
        if external.shape != state.shape:
            raise ValueError("external must match state shape")
        phase = phase + float(external_coupling) * torch.angle(external * state.conj())

    return torch.polar(state.abs(), phase)


def phase_network_probe(
    initial_state: torch.Tensor,
    adjacency: torch.Tensor,
    *,
    steps: int,
    neighbor_coupling: float,
    external: torch.Tensor | None = None,
    external_coupling: float = 0.0,
) -> PhaseNetworkProbeResult:
    """Measure whether graph coupling improves collective phase coherence."""
    state = _as_complex(initial_state)
    initial_alignment = None if external is None else phase_alignment(state, external)
    initial_coherence = phase_coherence(state)
    for _ in range(max(0, int(steps))):
        state = phase_coupling_step(
            state,
            adjacency,
            neighbor_coupling=neighbor_coupling,
            external=external,
            external_coupling=external_coupling,
        )
    return PhaseNetworkProbeResult(
        initial_coherence=initial_coherence,
        final_coherence=phase_coherence(state),
        initial_alignment=initial_alignment,
        final_alignment=None if external is None else phase_alignment(state, external),
        steps=max(0, int(steps)),
    )


def phase_lock_step(
    state: torch.Tensor,
    reference: torch.Tensor,
    *,
    coupling: float,
    amplitude_rate: float = 0.0,
) -> torch.Tensor:
    """Move oscillator phases toward an external cue without changing topology."""
    state = _as_complex(state)
    reference = _as_complex(reference).to(state.device)
    if state.shape != reference.shape:
        raise ValueError("state and reference must have the same shape")

    coupling = min(max(float(coupling), 0.0), 1.0)
    amplitude_rate = min(max(float(amplitude_rate), 0.0), 1.0)
    delta = torch.angle(reference * state.conj())
    phase = torch.angle(state) + coupling * delta
    radius = (1.0 - amplitude_rate) * state.abs() + amplitude_rate * reference.abs()
    return torch.polar(radius, phase)


def phase_lock_probe(
    stimulus: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    coupling: float,
    amplitude_rate: float = 0.0,
) -> PhaseLockProbeResult:
    """Measure whether a simple agent phase-locks to a periodic cue."""
    stimulus = _as_complex(stimulus)
    if stimulus.ndim != 2:
        raise ValueError("stimulus must be a 2D tensor")
    state = _as_complex(initial_state).to(stimulus.device)
    if state.shape != stimulus[0].shape:
        raise ValueError("initial_state must match one stimulus frame")

    initial_alignment = phase_alignment(state, stimulus[0])
    initial_coherence = phase_coherence(state)
    for reference in stimulus:
        state = phase_lock_step(
            state,
            reference,
            coupling=coupling,
            amplitude_rate=amplitude_rate,
        )
    return PhaseLockProbeResult(
        initial_alignment=initial_alignment,
        final_alignment=phase_alignment(state, stimulus[-1]),
        initial_coherence=initial_coherence,
        final_coherence=phase_coherence(state),
        steps=int(stimulus.shape[0]),
    )


def _as_complex(state: torch.Tensor) -> torch.Tensor:
    if state.is_complex():
        return state
    return torch.complex(state.float(), torch.zeros_like(state).float())


def build_runtime(
    weight: torch.Tensor,
    *,
    active_ratio: float,
    stdp_enabled: bool,
    stdp_lr: float,
    stdp_apply_interval: int,
    stdp_density: float,
) -> BrainRuntime:
    return BrainRuntime(
        weight.detach().cpu().float(),
        config=BrainRuntimeConfig(
            dim=weight.shape[0],
            active_ratio=active_ratio,
            active_threshold=0.0,
            noise_sigma=0.0,
            dale_law=False,
            axon_delay=False,
            stdp_enabled=stdp_enabled,
            stdp_apply_interval=stdp_apply_interval,
            stdp_lr=stdp_lr,
            stdp_density=stdp_density,
            stdp_gate_threshold=0.0,
            stdp_spike_threshold=0.0,
        ),
        backend="torch",
        device="cpu",
    )


def train_runtime_stdp(runtime: BrainRuntime, train_drive: torch.Tensor, steps: int) -> BrainRuntime:
    if train_drive.ndim != 2:
        raise ValueError("train_drive must be a 2D tensor")
    steps = max(0, int(steps))
    for idx in range(steps):
        external = train_drive[idx % train_drive.shape[0]].cpu()
        runtime.step(external_input=external, force_mode=RuntimeMode.WAKE)
    return runtime


def evaluate_transition_probe(runtime: BrainRuntime, eval_drive: torch.Tensor) -> RuntimeProbeResult:
    """Measure whether runtime activations predict the next hidden-state drive."""
    if eval_drive.ndim != 2:
        raise ValueError("eval_drive must be a 2D tensor")
    if eval_drive.shape[0] < 2:
        raise ValueError("eval_drive must contain at least two states")

    preds = []
    targets = []
    active = []
    last_step = None
    for idx in range(eval_drive.shape[0] - 1):
        step = runtime.step(external_input=eval_drive[idx].cpu(), force_mode=RuntimeMode.WAKE)
        preds.append(F.normalize(runtime.activation.detach().cpu(), dim=0))
        targets.append(F.normalize(eval_drive[idx + 1].detach().cpu(), dim=0))
        active.append(step.active_modules / max(runtime.config.dim, 1))
        last_step = step

    pred = torch.stack(preds)
    target = torch.stack(targets)
    mse = float(F.mse_loss(pred, target).item())
    cosine = float(F.cosine_similarity(pred, target, dim=1).mean().item())
    return RuntimeProbeResult(
        mse=mse,
        cosine=cosine,
        active_ratio=float(sum(active) / max(len(active), 1)),
        stdp_updates=0 if last_step is None else int(last_step.stdp_updates),
        stdp_gate=0.0 if last_step is None else float(last_step.stdp_gate),
    )
```
