"""Persistent CE brain daemon -- always running, self-learning, minimal energy.

The brain never stops. When idle, it drops to NREM/REM and consolidates.
When input arrives, it wakes. Hippocampus encodes during wake, replays during sleep.
STDP updates weights continuously. Energy budget enforces 4.87% active ceiling.

Usage:
    daemon = BrainDaemon(engine_path="clarus/gpt2_v2.ce.pt")
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
    from clarus.runtime import BrainRuntime, BrainRuntimeConfig, RuntimeMode, RuntimeStep
    from clarus.engine import CEEngine, PromptContext
    from clarus.stdp import STDPConfig, EligibilityTracker, compute_learning_gate, apply_stdp_update
    from clarus.neuromod import NeuromodulatorState, step_neuromodulators, apply_modulation
    from clarus.agent import (
        ConsciousnessMonitor, WorkingMemory, CerebellumPredictor,
        compute_critic, select_action_discrete, agent_step,
    )
    from clarus.constants import (
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
