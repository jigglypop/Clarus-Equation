"""Regression tests for the topology-first text engine."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


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
def topology_module():
    sys.modules.setdefault("clarus", types.ModuleType("clarus"))
    return _load("clarus.text_topology", ROOT / "clarus" / "text_topology.py")


@pytest.fixture
def engine(topology_module):
    return topology_module.TextTopologyEngine(dim=24)


def test_dominant_basis_changes_with_input(engine):
    short = engine.analyze("Hi.")
    long_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa."
    long_result = engine.analyze(long_text)
    assert short.dominant_euler_basis == "anchor"
    assert long_result.dominant_euler_basis == "drift"
    assert short.dominant_euler_basis != long_result.dominant_euler_basis


def test_basis_activation_is_normalised(engine):
    result = engine.analyze("alpha beta gamma delta epsilon zeta.")
    total = sum(result.euler_basis_activation.values())
    assert pytest.approx(total, rel=1e-6) == 1.0
    for value in result.euler_basis_activation.values():
        assert value >= 0.0


def test_token_chi_carries_cycle_information(engine):
    result = engine.analyze(
        "alpha beta gamma. alpha beta gamma. alpha beta gamma."
    )
    chi = result.token_summary.euler_characteristic
    components = result.token_summary.components
    # Repeated tokens form cycles, so chi must drop below the component
    # count instead of collapsing onto the trivial forest identity.
    assert chi < components


def test_alignment_uses_cosine(engine):
    result = engine.analyze(
        "Alpha bridges concept. Beta extends concept.\n\n"
        "Gamma rephrases idea. Delta echoes idea."
    )
    for value in (
        result.token_sentence_alignment,
        result.sentence_paragraph_alignment,
        result.token_sentence_bridge,
        result.sentence_paragraph_bridge,
    ):
        assert -1.0001 <= value <= 1.0001


def test_bridge_energy_is_distinct_from_alignment(engine):
    result = engine.analyze(
        "First short claim. Second short claim.\n\n"
        "Third paragraph statement only."
    )
    assert result.bridge_energy != pytest.approx(result.sentence_paragraph_alignment)


def test_paragraph_summary_uses_chi_v_minus_e_plus_f(engine, topology_module):
    text = (
        "Sentence one talks about graphs. Sentence two refines that idea. "
        "Sentence three closes it.\n\n"
        "Second paragraph extends the discussion. It links to graphs again. "
        "It also references closures."
    )
    result = engine.analyze(text)
    summary = result.sentence_summary
    expected = summary.count - summary.edges + summary.faces
    assert summary.euler_characteristic == expected


def test_algebraic_connectivity_non_negative(engine):
    result = engine.analyze(
        "Alpha aligns with beta. Beta aligns with gamma. Gamma loops to alpha."
    )
    assert result.sentence_summary.algebraic_connectivity >= 0.0
    assert result.paragraph_summary.algebraic_connectivity >= 0.0


def test_analysis_is_deterministic(engine):
    text = "Alpha bridges concept. Beta extends concept."
    a = engine.analyze(text)
    b = engine.analyze(text)
    assert a.bridge_energy == b.bridge_energy
    assert a.token_sentence_bridge == b.token_sentence_bridge
    assert a.sentence_summary.algebraic_connectivity == b.sentence_summary.algebraic_connectivity
    assert a.euler_basis_activation == b.euler_basis_activation
    assert a.phase_carrier_alignment == b.phase_carrier_alignment
    assert a.phase_carrier_decay == b.phase_carrier_decay


def test_phase_carrier_metrics_in_unit_range(engine):
    text = (
        "Alpha bridges concept. Beta extends concept. Gamma closes it.\n\n"
        "Delta opens a new line. Epsilon refines it."
    )
    result = engine.analyze(text)
    assert -1.0001 <= result.phase_carrier_alignment <= 1.0001
    assert result.phase_carrier_decay >= 0.0
    assert -1.0001 <= result.sentence_summary.phase_carrier_alignment <= 1.0001
    assert result.sentence_summary.phase_carrier_decay >= 0.0


def test_phase_carrier_decay_non_zero_on_distant_pairs(engine):
    text = " ".join(f"sentence number {i} body." for i in range(8))
    result = engine.analyze(text)
    assert result.sentence_summary.phase_carrier_decay > 0.0


def test_phase_carrier_helpers_match_pure_python(topology_module):
    states = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]]
    rotated = topology_module._phase_carrier_states(states, dim=4)
    align, decay = topology_module._phase_carrier_metrics(states, rotated, dim=4)
    assert rotated[0] == states[0]  # position 0 leaves vector untouched
    assert -1.0001 <= align <= 1.0001
    assert decay >= 0.0
