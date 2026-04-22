"""Stateful AGI-style tests for the text intent path."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def intent_module():
    sys.modules.setdefault("clarus", types.ModuleType("clarus"))
    _load("clarus.constants", ROOT / "clarus" / "constants.py")
    _load("clarus.neuromod", ROOT / "clarus" / "neuromod.py")
    _load("clarus.stdp", ROOT / "clarus" / "stdp.py")
    _load("clarus.text_topology", ROOT / "clarus" / "text_topology.py")
    return _load("clarus.text_intent", ROOT / "clarus" / "text_intent.py")


def test_predict_step_uses_single_topology_pass(intent_module, monkeypatch):
    classifier = intent_module.TopologyIntentClassifier(dim=24)
    calls = {"count": 0}
    real_analyze = classifier.engine.analyze

    def counting(text):
        calls["count"] += 1
        return real_analyze(text)

    monkeypatch.setattr(classifier.engine, "analyze", counting)
    classifier.predict_step("이 구조를 분석해줘.")
    assert calls["count"] == 1


def test_replay_bias_can_flip_ambiguous_prediction(intent_module):
    prototypes = (
        intent_module.IntentPrototype("alpha", ()),
        intent_module.IntentPrototype("beta", ()),
    )
    classifier = intent_module.TopologyIntentClassifier(dim=16, prototypes=prototypes)
    text = "plain neutral text"
    base = classifier.predict(text)
    assert base.label == "alpha"

    feature = classifier.feature_vector(text)
    classifier.state.replay_buffer.append(
        intent_module.IntentReplayItem(
            feature_vector=feature,
            label="beta",
            confidence=1.0,
            salience=1.0,
            priority=2.0,
        )
    )
    stepped = classifier.predict_step(text)
    assert stepped.label == "beta"
    assert stepped.replay_scores["beta"] > stepped.replay_scores["alpha"]


def test_self_measure_updates_and_bridge_report(intent_module):
    classifier = intent_module.TopologyIntentClassifier(dim=24)
    before = classifier.state.active_ratio_ema
    classifier.predict_step("이 구조를 설명해줘.")
    after = classifier.state.active_ratio_ema
    report = classifier.bridge_gate_report()
    assert after != before
    assert "F1_active_ratio_ema" in report
    assert "F1_deviation" in report


def test_neuromodulation_changes_effective_temperature(intent_module):
    classifier = intent_module.TopologyIntentClassifier(dim=24, temperature=0.35)
    classifier.state.neuromod = intent_module.NeuromodulatorState(
        da=1.2,
        ne=1.4,
        sht=1.8,
        ach=1.1,
    )
    prediction = classifier.predict_step("이 구조를 설명해줘.")
    assert prediction.effective_temperature > classifier.temperature


def test_stdp_lite_bounded_and_serializable(intent_module):
    prototypes = (
        intent_module.IntentPrototype("alpha", ()),
        intent_module.IntentPrototype("beta", ()),
    )
    classifier = intent_module.TopologyIntentClassifier(
        dim=16,
        prototypes=prototypes,
        stdp_lr=0.4,
        stdp_bias_clip=0.5,
    )
    for _ in range(10):
        classifier.predict_step("same text")
        classifier.observe_feedback("beta", reward=1.0, text="same text")

    for value in classifier.state.label_bias.values():
        assert abs(value) <= 0.5 + 1e-8

    snapshot = classifier.session_state_dict()
    restored = intent_module.TopologyIntentClassifier(dim=16, prototypes=prototypes)
    restored.load_session_state_dict(snapshot)
    assert restored.state.label_bias == classifier.state.label_bias
    assert restored.state.last_label == classifier.state.last_label


def test_consolidate_decays_replay_priority(intent_module):
    classifier = intent_module.TopologyIntentClassifier(dim=24)
    classifier.predict_step("replay target text")
    before = classifier.state.replay_buffer[0].priority
    report = classifier.consolidate(mode="REM")
    after = classifier.state.replay_buffer[0].priority
    assert report["consolidated"] >= 1.0
    assert after < before


def test_daemon_checkpoint_roundtrip_preserves_intent_state(tmp_path, monkeypatch):
    package = sys.modules.setdefault("clarus", types.ModuleType("clarus"))

    runtime_mod = types.ModuleType("clarus.runtime")

    class RuntimeMode(str, Enum):
        WAKE = "WAKE"
        NREM = "NREM"
        REM = "REM"

    @dataclass
    class BrainRuntimeConfig:
        dim: int
        active_ratio: float = 0.1
        noise_sigma: float = 0.0
        dale_law: bool = False
        axon_delay: bool = False
        memory_capacity: int = 16

    @dataclass
    class RuntimeStep:
        step: int = 0
        mode: RuntimeMode = RuntimeMode.WAKE
        energy: float = 0.0
        active_modules: int = 0
        replay_norm: float = 0.0
        sleep_pressure: float = 0.0
        arousal: float = 0.0
        lifecycle_counts: dict[str, int] | None = None

    class _DummyHippocampus:
        def __len__(self):
            return 0

    class BrainRuntime:
        def __init__(self, weight, *, config, backend="torch", device="cpu"):
            self.weight = weight
            self.config = config
            self.backend = backend
            self.device = device
            self.mode = RuntimeMode.WAKE
            self.sleep_pressure = 0.0
            self.goal = torch.zeros(config.dim)
            self.hippocampus = _DummyHippocampus()

        def set_goal(self, goal):
            self.goal = goal

        def snapshot(self):
            return {"goal": self.goal.clone(), "dim": self.config.dim}

        @classmethod
        def from_snapshot(cls, snapshot, *, backend="torch", device="cpu"):
            cfg = BrainRuntimeConfig(dim=int(snapshot["dim"]))
            obj = cls(torch.eye(cfg.dim), config=cfg, backend=backend, device=device)
            obj.goal = snapshot["goal"].clone()
            return obj

    runtime_mod.BrainRuntime = BrainRuntime
    runtime_mod.BrainRuntimeConfig = BrainRuntimeConfig
    runtime_mod.RuntimeMode = RuntimeMode
    runtime_mod.RuntimeStep = RuntimeStep
    sys.modules["clarus.runtime"] = runtime_mod
    setattr(package, "runtime", runtime_mod)

    engine_mod = types.ModuleType("clarus.engine")

    class CEEngine:
        def __init__(self, *_args, device="cpu", backend="torch", **_kwargs):
            self.device = device
            self.backend = backend
            self.d = 8
            self.W = torch.eye(8)

    @dataclass
    class PromptContext:
        prompt: str
        prompt_ids: object
        h_true: object
        m0: object
        phi: object
        best_layer: int
        layer_scores: dict[int, float]

    engine_mod.CEEngine = CEEngine
    engine_mod.PromptContext = PromptContext
    sys.modules["clarus.engine"] = engine_mod
    setattr(package, "engine", engine_mod)

    stdp_mod = types.ModuleType("clarus.stdp")

    @dataclass
    class STDPConfig:
        dim: int
        spike_threshold: float = 0.15

    class EligibilityTracker:
        def __init__(self, *_args, **_kwargs):
            pass

        def update(self, *_args, **_kwargs):
            pass

        def reset(self):
            pass

    def compute_learning_gate(**_kwargs):
        return 0.0

    def apply_stdp_update(weight, *_args, **_kwargs):
        return weight

    stdp_mod.STDPConfig = STDPConfig
    stdp_mod.EligibilityTracker = EligibilityTracker
    stdp_mod.compute_learning_gate = compute_learning_gate
    stdp_mod.apply_stdp_update = apply_stdp_update
    sys.modules["clarus.stdp"] = stdp_mod
    setattr(package, "stdp", stdp_mod)

    neuromod_mod = types.ModuleType("clarus.neuromod")

    @dataclass
    class NeuromodulatorState:
        da: float = 0.0
        ne: float = 0.0
        sht: float = 0.0
        ach: float = 0.0

    def step_neuromodulators(state, **_kwargs):
        return state

    def apply_modulation(*_args, **_kwargs):
        return None

    neuromod_mod.NeuromodulatorState = NeuromodulatorState
    neuromod_mod.step_neuromodulators = step_neuromodulators
    neuromod_mod.apply_modulation = apply_modulation
    sys.modules["clarus.neuromod"] = neuromod_mod
    setattr(package, "neuromod", neuromod_mod)

    agent_mod = types.ModuleType("clarus.agent")

    class ConsciousnessMonitor:
        def __init__(self):
            self._deviation_history = deque(maxlen=8)

        def consciousness_depth(self):
            return 0.0

    class WorkingMemory:
        def __init__(self, capacity=7):
            self._buffer = deque(maxlen=capacity)

        def append(self, action, observation):
            self._buffer.append((action, observation))

        def contents(self):
            return list(self._buffer)

    class CerebellumPredictor:
        def __init__(self, dim):
            self._prediction = torch.zeros(dim)

        def predict(self):
            return self._prediction.clone()

    def compute_critic(*_args, **_kwargs):
        return types.SimpleNamespace(c_pred=0.0, c_nov=0.0)

    def select_action_discrete(*_args, **_kwargs):
        return 0

    def agent_step(*_args, **_kwargs):
        return {}

    agent_mod.ConsciousnessMonitor = ConsciousnessMonitor
    agent_mod.WorkingMemory = WorkingMemory
    agent_mod.CerebellumPredictor = CerebellumPredictor
    agent_mod.compute_critic = compute_critic
    agent_mod.select_action_discrete = select_action_discrete
    agent_mod.agent_step = agent_step
    sys.modules["clarus.agent"] = agent_mod
    setattr(package, "agent", agent_mod)

    constants_mod = types.ModuleType("clarus.constants")
    constants_mod.ACTIVE_RATIO = 0.0487
    constants_mod.STRUCT_RATIO = 0.2623
    constants_mod.BACKGROUND_RATIO = 0.6891
    constants_mod.BOOTSTRAP_CONTRACTION = 0.5
    constants_mod.NOISE_SIGMA = 0.01
    sys.modules["clarus.constants"] = constants_mod
    setattr(package, "constants", constants_mod)

    text_intent_mod = types.ModuleType("clarus.text_intent")

    class TopologyIntentClassifier:
        def __init__(self, dim=32):
            self.dim = dim
            self._state = {"count": 0}
            self.state = types.SimpleNamespace(replay_buffer=[])

        def session_state_dict(self):
            return dict(self._state)

        def load_session_state_dict(self, state):
            self._state = dict(state)

        def bridge_gate_report(self):
            return {"F1_deviation": 0.0}

    text_intent_mod.TopologyIntentClassifier = TopologyIntentClassifier
    sys.modules["clarus.text_intent"] = text_intent_mod
    setattr(package, "text_intent", text_intent_mod)

    daemon_module = _load("clarus.daemon", ROOT / "clarus" / "daemon.py")
    saved_payload: dict[str, object] = {}

    def fake_save(obj, path):
        saved_payload[str(path)] = obj

    def fake_load(path, **_kwargs):
        return saved_payload[str(path)]

    monkeypatch.setattr(daemon_module.torch, "save", fake_save)
    monkeypatch.setattr(daemon_module.torch, "load", fake_load)
    checkpoint_path = tmp_path / "daemon.pt"
    daemon = daemon_module.BrainDaemon(
        "dummy.pt",
        config=daemon_module.DaemonConfig(checkpoint_path=str(checkpoint_path)),
    )
    daemon.intent._state = {"count": 7}
    daemon._save_checkpoint()

    restored = daemon_module.BrainDaemon(
        "dummy.pt",
        config=daemon_module.DaemonConfig(checkpoint_path=str(checkpoint_path)),
    )
    restored.load_checkpoint()
    assert restored.intent._state == {"count": 7}
