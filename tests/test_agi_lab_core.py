from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import importlib.util
import inspect
from pathlib import Path
import sys
import unicodedata

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = (
    ROOT / "reality_stone" / "python" / "reality_stone" / "clarus" / "agi_lab"
)
INIT_PATH = PACKAGE_DIR / "__init__.py"


def _isolated_load():
    source = INIT_PATH.read_bytes()
    name = f"_ce_agi_lab_test_{hashlib.sha256(source).hexdigest()}"
    spec = importlib.util.spec_from_file_location(
        name,
        INIT_PATH,
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot build isolated agi_lab package spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for module_name in tuple(sys.modules):
            if module_name == name or module_name.startswith(f"{name}."):
                sys.modules.pop(module_name, None)
        raise
    return module


PARENT_SNAPSHOT = frozenset(
    name
    for name in sys.modules
    if name == "reality_stone" or name.startswith("reality_stone.")
)
agi = _isolated_load()


class ScriptedPlanner:
    def __init__(self, actions):
        self._actions = tuple(actions)

    def rank(self, goal, observation, belief, predictor, action_space, budget):
        del goal, predictor, budget
        action = self._actions[observation.tick % len(self._actions)]
        assert action_space.contains(action)
        return (
            agi.ActionProposal(
                action=action,
                expected_score=0.0,
                predicted_risk=0.0,
                model_version=belief.model_version,
                evidence_refs=("visible-history-only",),
            ),
        )


class AdversarialPlanner:
    def __init__(self, safe_after_forbidden: bool):
        self._safe_after_forbidden = safe_after_forbidden

    def rank(self, goal, observation, belief, predictor, action_space, budget):
        del goal, observation, predictor, action_space, budget
        actions = [agi.FORBIDDEN]
        if self._safe_after_forbidden:
            actions.append(agi.APPLY_ONE)
        return tuple(
            agi.ActionProposal(
                action=action,
                expected_score=100.0,
                predicted_risk=0.0,
                model_version=belief.model_version,
                evidence_refs=("adversarial",),
            )
            for action in actions
        )


class OpaqueWorldProxy:
    """Expose only the two protocol methods; all other attributes are poison."""

    def __init__(self, genesis, execute):
        self._genesis = genesis
        self._execute = execute

    def genesis(self, request):
        return self._genesis(request)

    def execute(self, session, permit):
        return self._execute(session, permit)

    def __getattr__(self, name):
        raise AssertionError(f"orchestrator inspected non-protocol attribute {name!r}")


def _system(
    family: str,
    *,
    key: bytes = b"A" * 32,
    world_id: str = "logical-world",
    planner=None,
    horizon: int = 8,
    proxy: bool = False,
):
    authority = agi.PermitAuthority(key)
    safety = agi.SafetyKernel(agi.DEFAULT_SAFETY_POLICY, authority)
    factory = agi.make_xor_world if family == "xor" else agi.make_set_world
    world = factory(
        world_instance_id=world_id,
        authority=authority,
        policy_digest=safety.policy_digest,
        horizon=horizon,
    )
    if proxy:
        world = OpaqueWorldProxy(world.genesis, world.execute)
    model = agi.TabularWorldModel()
    orchestrator = agi.CoreOrchestrator(
        world=world,
        executor=world,
        memory=model,
        world_model=model,
        planner=planner or ScriptedPlanner((agi.APPLY_ONE, agi.APPLY_ZERO, agi.APPLY_ONE)),
        safety=safety,
        planning_budget=4,
    )
    return orchestrator, world, authority, safety


def _genesis(orchestrator, *, episode="episode-1", initial=0, target=9):
    return orchestrator.genesis(
        agi.GenesisRequest(episode, initial, agi.CoreGoal(target))
    )


def _trace(family: str, *, key: bytes = b"A" * 32, proxy: bool = False):
    orchestrator, _, _, _ = _system(family, key=key, proxy=proxy)
    state = _genesis(orchestrator)
    trace = []
    private_tags = []
    for _ in range(3):
        draft = orchestrator.decide(state)
        private_tags.append(draft.safety_decision.permit.authentication_tag)
        state, _ = orchestrator.step(state)
        trace.append(state.observation.state)
    return tuple(trace), state, tuple(private_tags)


def test_isolated_package_does_not_import_parent_or_export_agi_claims() -> None:
    after = frozenset(
        name
        for name in sys.modules
        if name == "reality_stone" or name.startswith("reality_stone.")
    )
    assert after == PARENT_SNAPSHOT
    assert agi.IMPLEMENTATION_STATUS == "PHYSICS_INDEPENDENT_CORE_SCAFFOLD"
    assert not agi.PHYSICAL_AGI_CLAIM
    assert not agi.CONSCIOUSNESS_CLAIM
    assert not agi.BRAIN_ALGORITHM_CLAIM


def test_canonical_records_reject_mutability_coercion_and_nonfinite_values() -> None:
    with pytest.raises(TypeError):
        agi.CoreAction("x", [("value", 1)])
    with pytest.raises(TypeError):
        agi.CoreAction("x", (("value", [1]),))
    with pytest.raises(TypeError):
        agi.CoreGoal(True)
    with pytest.raises(TypeError):
        agi.CoreGoal(1.0)
    with pytest.raises(TypeError):
        agi.ActionProposal(agi.ABSTAIN, Decimal("1"), 0.0, "m")
    with pytest.raises(ValueError):
        agi.ActionProposal(agi.ABSTAIN, float("nan"), 0.0, "m")
    with pytest.raises(ValueError):
        agi.CoreAction("x", (("a", 1), ("a", 2)))
    with pytest.raises(UnicodeEncodeError):
        agi.CoreAction("\ud800")
    with pytest.raises(UnicodeEncodeError):
        agi.CoreAction("x", (("value", "\udfff"),))


def test_canonical_bytes_normalize_unicode_key_order_and_signed_zero() -> None:
    composed = "é"
    decomposed = unicodedata.normalize("NFD", composed)
    left = agi.CoreAction(composed, (("z", -0.0), ("a", 1)))
    right = agi.CoreAction(decomposed, (("a", 1), ("z", 0.0)))
    assert agi.canonical_bytes(left) == agi.canonical_bytes(right)
    assert agi.canonical_digest(left) == agi.canonical_digest(right)


def test_explicit_memory_executor_and_counterfactual_rollout_roles() -> None:
    assert all(
        hasattr(agi.MemoryStore, method)
        for method in ("initialize", "infer", "update")
    )
    assert hasattr(agi.ActionExecutor, "execute")
    assert hasattr(agi.WorldAdapter, "genesis")
    assert hasattr(agi.WorldModel, "rollout")
    assert agi.ActionExecutor not in agi.WorldAdapter.__mro__

    belief = agi.BeliefState(
        (
            (0, agi.canonical_digest(agi.APPLY_ONE), 1),
            (1, agi.canonical_digest(agi.APPLY_ZERO), 0),
        ),
        "rollout-fixture",
    )
    observation = agi.CoreObservation("episode", 0, 0, agi.CoreGoal(9))
    outcomes = agi.TabularWorldModel.rollout(
        belief,
        observation,
        (agi.APPLY_ONE, agi.APPLY_ZERO),
    )
    assert tuple((item.next_state, item.confidence) for item in outcomes) == (
        (1, 1.0),
        (0, 1.0),
    )
    with pytest.raises(TypeError, match="nonempty tuple"):
        agi.TabularWorldModel.rollout(belief, observation, [])


def test_two_nonisomorphic_families_share_one_protocol_only_orchestrator() -> None:
    xor_trace, xor_state, _ = _trace("xor", proxy=True)
    set_trace, set_state, _ = _trace("set", proxy=True)
    assert xor_trace == (1, 1, 0)
    assert set_trace == (1, 0, 1)
    assert agi.verify_ledger(xor_state.ledger)
    assert agi.verify_ledger(set_state.ledger)
    source = inspect.getsource(agi.CoreOrchestrator)
    for forbidden in (
        "make_xor_world",
        "make_set_world",
        "family_id",
        "isinstance(",
        "type(self._world)",
        "hasattr(",
        "__class__",
    ):
        assert forbidden not in source


def test_hidden_truth_is_noninterfering_until_visible_histories_diverge() -> None:
    xor, _, _, _ = _system("xor")
    setter, _, _, _ = _system("set")
    xor_state = _genesis(xor)
    set_state = _genesis(setter)
    assert agi.public_ledger_bytes(xor_state.ledger) == agi.public_ledger_bytes(
        set_state.ledger
    )
    xor_draft = xor.decide(xor_state)
    set_draft = setter.decide(set_state)
    assert agi.canonical_bytes(xor_draft.proposals) == agi.canonical_bytes(
        set_draft.proposals
    )
    xor_state, _ = xor.step(xor_state)
    set_state, _ = setter.step(set_state)
    assert xor_state.observation == set_state.observation
    assert agi.public_ledger_bytes(xor_state.ledger) == agi.public_ledger_bytes(
        set_state.ledger
    )
    assert agi.canonical_bytes(xor.decide(xor_state).proposals) == agi.canonical_bytes(
        setter.decide(set_state).proposals
    )
    xor_state, _ = xor.step(xor_state)
    set_state, _ = setter.step(set_state)
    assert xor_state.observation.state != set_state.observation.state


def test_adversarial_forbidden_proposal_cannot_bypass_safety() -> None:
    orchestrator, _, _, _ = _system(
        "xor", planner=AdversarialPlanner(safe_after_forbidden=True)
    )
    state = _genesis(orchestrator)
    draft = orchestrator.decide(state)
    assert draft.proposals[0].action == agi.FORBIDDEN
    assert draft.safety_decision.selected_action == agi.APPLY_ONE
    next_state, _ = orchestrator.step(state)
    assert next_state.observation.state == 1
    assert next_state.session.transition_count == 1


def test_all_forbidden_proposals_produce_safe_abstention() -> None:
    orchestrator, _, _, _ = _system(
        "xor", planner=AdversarialPlanner(safe_after_forbidden=False)
    )
    state = _genesis(orchestrator)
    draft = orchestrator.decide(state)
    assert draft.safety_decision.selected_action == agi.ABSTAIN
    next_state, _ = orchestrator.step(state)
    assert next_state.observation.state == state.observation.state
    assert next_state.session.transition_count == 1


def test_forged_stale_cross_bound_and_replay_permits_leave_state_unchanged() -> None:
    orchestrator, world, _, _ = _system("xor")
    state = _genesis(orchestrator)
    permit = orchestrator.decide(state).safety_decision.permit
    assert permit is not None
    mutations = (
        replace(permit, authentication_tag="0" * 64),
        replace(permit, episode_id="other"),
        replace(permit, tick=permit.tick + 1),
        replace(permit, world_commitment="other"),
        replace(permit, session_digest="other"),
        replace(permit, action_space_digest="other"),
        replace(permit, policy_digest="other"),
        replace(permit, proposal_digest="other"),
        replace(permit, action=agi.FORBIDDEN),
        replace(permit, nonce="other"),
    )
    before = agi.canonical_bytes(state.session)
    with pytest.raises(TypeError, match="WorldSession and ActionPermit"):
        world.execute(state.session, agi.APPLY_ONE)
    assert agi.canonical_bytes(state.session) == before
    assert state.session.transition_count == 0
    for forged in mutations:
        with pytest.raises(PermissionError, match="verification"):
            world.execute(state.session, forged)
        assert agi.canonical_bytes(state.session) == before
        assert state.session.transition_count == 0
    forged_session = replace(state.session, state=1)
    with pytest.raises(PermissionError, match="verification"):
        world.execute(forged_session, permit)
    next_session, _ = world.execute(state.session, permit)
    assert next_session.transition_count == 1
    with pytest.raises(PermissionError, match="verification"):
        world.execute(next_session, permit)
    forked_session, _ = world.execute(state.session, permit)
    assert agi.canonical_bytes(forked_session) == agi.canonical_bytes(next_session)

    other_orchestrator, other_world, _, _ = _system(
        "xor", world_id="different-world"
    )
    other_state = _genesis(other_orchestrator)
    with pytest.raises(PermissionError, match="verification"):
        other_world.execute(other_state.session, permit)
    with pytest.raises(PermissionError, match="verification"):
        other_world.execute(state.session, permit)


def test_terminal_session_rejects_even_newly_authenticated_permit() -> None:
    planner = ScriptedPlanner((agi.APPLY_ONE,))
    orchestrator, world, authority, safety = _system(
        "xor", planner=planner, horizon=3
    )
    state = _genesis(orchestrator, target=1)
    state, _ = orchestrator.step(state)
    assert state.session.terminated
    proposal = agi.ActionProposal(
        agi.ABSTAIN,
        0.0,
        0.0,
        state.belief.model_version,
        ("post-terminal",),
    )
    permit = authority._issue(
        session=state.session,
        proposal=proposal,
        policy_digest=safety.policy_digest,
    )
    with pytest.raises(PermissionError, match="not live"):
        world.execute(state.session, permit)
    with pytest.raises(RuntimeError, match="termination"):
        orchestrator.decide(state)


def test_genesis_is_explicit_and_old_episode_permit_cannot_cross_reset_boundary() -> None:
    orchestrator, world, _, _ = _system("xor")
    first = _genesis(orchestrator, episode="first")
    old_permit = orchestrator.decide(first).safety_decision.permit
    second = _genesis(orchestrator, episode="second")
    assert first.ledger[0].event_type == second.ledger[0].event_type == "genesis"
    assert first.session.transition_count == second.session.transition_count == 0
    assert not hasattr(orchestrator, "reset")
    with pytest.raises(PermissionError, match="verification"):
        world.execute(second.session, old_permit)


def test_public_ledger_excludes_key_and_tag_and_replays_across_session_keys() -> None:
    _, left_state, left_tags = _trace("xor", key=b"LEFT-SESSION-KEY" * 2)
    _, right_state, right_tags = _trace("xor", key=b"RIGHT-SESSION-KEY" * 2)
    assert left_tags != right_tags
    left_bytes = agi.public_ledger_bytes(left_state.ledger)
    right_bytes = agi.public_ledger_bytes(right_state.ledger)
    assert left_bytes == right_bytes
    assert b"LEFT-SESSION-KEY" not in left_bytes
    assert b"RIGHT-SESSION-KEY" not in right_bytes
    for tag in left_tags + right_tags:
        assert tag.encode("ascii") not in left_bytes
        assert tag.encode("ascii") not in right_bytes


def test_full_state_replay_is_byte_identical_and_hash_chain_detects_tamper() -> None:
    _, left, _ = _trace("xor")
    _, right, _ = _trace("xor")
    assert agi.canonical_bytes(left) == agi.canonical_bytes(right)
    assert agi.public_ledger_bytes(left.ledger) == agi.public_ledger_bytes(right.ledger)
    assert agi.verify_ledger(left.ledger)
    middle = left.ledger[1]
    tampered = replace(
        middle,
        payload=middle.payload + (("tampered", True),),
    )
    broken = (left.ledger[0], tampered) + left.ledger[2:]
    assert not agi.verify_ledger(broken)
    assert left.ledger[0].digest == broken[0].digest

    rehashed = ()
    for index, entry in enumerate(left.ledger):
        payload = entry.payload
        if index == 1:
            payload = payload + (("tampered", True),)
        rehashed = agi.append_ledger_entry(
            rehashed,
            event_type=entry.event_type,
            payload=payload,
        )
    assert agi.verify_ledger(rehashed)
    assert rehashed[0].digest == left.ledger[0].digest
    assert all(
        changed.digest != original.digest
        for changed, original in zip(rehashed[1:], left.ledger[1:], strict=True)
    )


def test_reducer_is_pure_with_respect_to_input_session_bytes() -> None:
    orchestrator, world, _, _ = _system("xor")
    state = _genesis(orchestrator)
    permit = orchestrator.decide(state).safety_decision.permit
    before = agi.canonical_bytes(state.session)
    next_session, _ = world.execute(state.session, permit)
    assert agi.canonical_bytes(state.session) == before
    assert next_session is not state.session
    assert next_session.transition_count == state.session.transition_count + 1


def test_protocol_components_fail_closed_on_inconsistent_records() -> None:
    _, world, _, safety = _system("xor")

    class BrokenGenesis:
        def genesis(self, request):
            session, start = world.genesis(request)
            return replace(session, state=1 - session.state), start

    class BrokenExecutor:
        def execute(self, session, permit):
            next_session, step = world.execute(session, permit)
            return next_session, replace(
                step,
                transition_count=step.transition_count + 1,
            )

    class BrokenMemory:
        @staticmethod
        def initialize(observation):
            return agi.TabularWorldModel.initialize(observation)

        @staticmethod
        def infer(observation, previous):
            return agi.TabularWorldModel.infer(observation, previous)

        @staticmethod
        def update(belief, experience):
            next_belief, receipt = agi.TabularWorldModel.update(belief, experience)
            return next_belief, replace(receipt, experience_digest="wrong")

    model = agi.TabularWorldModel()
    planner = ScriptedPlanner((agi.APPLY_ONE,))

    bad_genesis = agi.CoreOrchestrator(
        world=BrokenGenesis(),
        executor=world,
        memory=model,
        world_model=model,
        planner=planner,
        safety=safety,
        planning_budget=1,
    )
    with pytest.raises(RuntimeError, match="genesis records"):
        _genesis(bad_genesis)

    bad_executor = agi.CoreOrchestrator(
        world=world,
        executor=BrokenExecutor(),
        memory=model,
        world_model=model,
        planner=planner,
        safety=safety,
        planning_budget=1,
    )
    executor_state = _genesis(bad_executor)
    executor_before = agi.canonical_bytes(executor_state)
    with pytest.raises(RuntimeError, match="transition records"):
        bad_executor.step(executor_state)
    assert agi.canonical_bytes(executor_state) == executor_before

    bad_memory = agi.CoreOrchestrator(
        world=world,
        executor=world,
        memory=BrokenMemory(),
        world_model=model,
        planner=planner,
        safety=safety,
        planning_budget=1,
    )
    memory_state = _genesis(bad_memory)
    memory_before = agi.canonical_bytes(memory_state)
    with pytest.raises(RuntimeError, match="update receipt"):
        bad_memory.step(memory_state)
    assert agi.canonical_bytes(memory_state) == memory_before


def test_public_agent_records_have_no_truth_callback_or_mutable_escape_fields() -> None:
    observation = agi.CoreObservation("episode", 0, 0, agi.CoreGoal(1))
    fields = agi.dataclass_public_fields(observation)
    assert fields == ("episode_id", "tick", "state", "goal")
    forbidden_fragments = {"truth", "family", "seed", "callback", "world"}
    assert not any(
        fragment in field_name
        for field_name in fields
        for fragment in forbidden_fragments
    )
