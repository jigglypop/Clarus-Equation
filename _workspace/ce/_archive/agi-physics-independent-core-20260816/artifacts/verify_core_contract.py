"""Independent counterexamples and constructive witnesses for AGI Core V0.

This scratch verifier imports no product module.  It checks only finite models
of the contract boundaries: shallow frozen records, learner/truth aliasing,
state-dependent replay, permit validation, hash-chained ledgers, and two
non-isomorphic transition families behind one adapter-shaped orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import math
from typing import Any, Callable


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256(payload: object) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class ShallowFrozen:
    values: list[int]


def frozen_is_shallow_counterexample() -> bool:
    record = ShallowFrozen([1])
    before = tuple(record.values)
    record.values.append(2)
    return before == (1,) and tuple(record.values) == (1, 2)


@dataclass(frozen=True)
class TruthAliasWithoutTruthField:
    metadata: dict[str, int]


def no_truth_field_is_not_noninterference_counterexample() -> bool:
    evaluator_truth = {"hidden_rule": 7}
    learner_view = TruthAliasWithoutTruthField(metadata=evaluator_truth)
    field_names = set(learner_view.__dataclass_fields__)
    return "truth" not in field_names and learner_view.metadata["hidden_rule"] == 7


def state_omission_breaks_replay_counterexample() -> bool:
    observation = (0,)
    seed = 19
    config = {"planner": "history-sensitive"}

    def ledger(memory: tuple[int, ...]) -> bytes:
        action = "left" if sum(memory) % 2 == 0 else "right"
        return canonical_bytes(
            {
                "action": action,
                "config": config,
                "memory": memory,
                "observation": observation,
                "seed": seed,
            }
        )

    return ledger((0,)) != ledger((1,))


def reset_is_a_world_change_counterexample() -> bool:
    class MutableWorld:
        def __init__(self) -> None:
            self.state = -1

        def reset(self) -> None:
            self.state = 0

    world = MutableWorld()
    before = world.state
    world.reset()
    return before == -1 and world.state == 0


@dataclass(frozen=True)
class NaivePermit:
    action: str
    allowed: bool


def frozen_permit_is_forgeable_counterexample() -> bool:
    allowed = NaivePermit("wait", True)
    forged_by_constructor = NaivePermit("forbidden", True)
    forged_by_replace = replace(allowed, action="forbidden")
    return forged_by_constructor.allowed and forged_by_replace.allowed


@dataclass(frozen=True)
class Permit:
    episode: str
    tick: int
    action: str
    policy_hash: str
    adapter_hash: str
    nonce: str
    tag: str


class PermitIssuer:
    def __init__(self, key: bytes, policy_hash: str, adapter_hash: str) -> None:
        self._key = key
        self.policy_hash = policy_hash
        self.adapter_hash = adapter_hash

    def _message(
        self,
        episode: str,
        tick: int,
        action: str,
        nonce: str,
    ) -> bytes:
        return canonical_bytes(
            {
                "action": action,
                "adapter_hash": self.adapter_hash,
                "episode": episode,
                "nonce": nonce,
                "policy_hash": self.policy_hash,
                "tick": tick,
            }
        )

    def issue(self, episode: str, tick: int, action: str, nonce: str) -> Permit:
        tag = hmac.new(
            self._key,
            self._message(episode, tick, action, nonce),
            hashlib.sha256,
        ).hexdigest()
        return Permit(
            episode,
            tick,
            action,
            self.policy_hash,
            self.adapter_hash,
            nonce,
            tag,
        )

    def expected_tag(self, permit: Permit) -> str:
        return hmac.new(
            self._key,
            self._message(
                permit.episode,
                permit.tick,
                permit.action,
                permit.nonce,
            ),
            hashlib.sha256,
        ).hexdigest()


class PermitOnlyExecutor:
    def __init__(self, issuer: PermitIssuer, episode: str) -> None:
        self.issuer = issuer
        self.episode = episode
        self.tick = 0
        self.used_nonces: set[str] = set()
        self.transition_count = 0

    def execute(self, permit: Permit) -> bool:
        valid = (
            permit.episode == self.episode
            and permit.tick == self.tick
            and permit.policy_hash == self.issuer.policy_hash
            and permit.adapter_hash == self.issuer.adapter_hash
            and permit.nonce not in self.used_nonces
            and hmac.compare_digest(permit.tag, self.issuer.expected_tag(permit))
        )
        if not valid:
            return False
        self.used_nonces.add(permit.nonce)
        self.transition_count += 1
        self.tick += 1
        return True


def permit_only_constructive_witness() -> bool:
    issuer = PermitIssuer(b"independent-verifier-key", "policy-v0", "adapter-v0")
    executor = PermitOnlyExecutor(issuer, "episode-a")
    valid = issuer.issue("episode-a", 0, "wait", "nonce-0")
    forged = replace(valid, action="forbidden")
    wrong_episode = issuer.issue("episode-b", 0, "wait", "nonce-1")
    before = executor.transition_count
    forged_rejected = not executor.execute(forged)
    wrong_episode_rejected = not executor.execute(wrong_episode)
    unchanged_after_rejections = executor.transition_count == before
    valid_accepted = executor.execute(valid)
    replay_rejected = not executor.execute(valid)
    return all(
        (
            forged_rejected,
            wrong_episode_rejected,
            unchanged_after_rejections,
            valid_accepted,
            replay_rejected,
            executor.transition_count == 1,
        )
    )


def hash_chain(entries: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    previous = "0" * 64
    digests: list[str] = []
    for index, entry in enumerate(entries):
        digest = sha256(
            {
                "entry": entry,
                "index": index,
                "previous": previous,
                "schema": "ce.agi-core-v0.ledger-entry.v1",
            }
        )
        digests.append(digest)
        previous = digest
    return tuple(digests)


def deterministic_ledger_constructive_witness() -> bool:
    entries = (
        {"action": "probe", "observation": [0, 1], "tick": 0},
        {"action": "wait", "observation": [1, 1], "tick": 1},
    )
    replay_equal = hash_chain(entries) == hash_chain(entries)
    mutated = (entries[0], {**entries[1], "action": "forbidden"})
    tamper_changes_tail = hash_chain(entries)[-1] != hash_chain(mutated)[-1]
    return replay_equal and tamper_changes_tail


State = int
Action = int
Transition = Callable[[State, Action], State]


def family_xor(state: State, action: Action) -> State:
    return state ^ action


def family_set(_state: State, action: Action) -> State:
    return action


class FiniteAdapter:
    def __init__(self, transition: Transition) -> None:
        self.transition = transition
        self.state = 0

    def step(self, action: Action) -> State:
        self.state = self.transition(self.state, action)
        return self.state


def same_orchestrator(adapter: FiniteAdapter, actions: tuple[Action, ...]) -> tuple[State, ...]:
    return tuple(adapter.step(action) for action in actions)


def action_image_cardinalities(transition: Transition) -> tuple[int, ...]:
    states = (0, 1)
    actions = (0, 1)
    return tuple(
        sorted(len({transition(state, action) for state in states}) for action in actions)
    )


def two_family_adapter_constructive_witness() -> bool:
    invariant_xor = action_image_cardinalities(family_xor)
    invariant_set = action_image_cardinalities(family_set)
    trace_xor = same_orchestrator(FiniteAdapter(family_xor), (1, 0, 1))
    trace_set = same_orchestrator(FiniteAdapter(family_set), (1, 0, 1))
    return (
        invariant_xor == (2, 2)
        and invariant_set == (1, 1)
        and invariant_xor != invariant_set
        and trace_xor == (1, 1, 0)
        and trace_set == (1, 0, 1)
    )


def visible_history_noninterference_boundary() -> bool:
    visible_history = ((0,), (1,))

    def lawful_decision(history: tuple[tuple[int, ...], ...]) -> str:
        return "left" if sum(sum(row) for row in history) % 2 else "right"

    first_truth = {"family": "xor"}
    second_truth = {"family": "set"}

    def illegal_closure_decision(truth: dict[str, str]) -> str:
        return "left" if truth["family"] == "xor" else "right"

    lawful_equal = lawful_decision(visible_history) == lawful_decision(visible_history)
    illegal_differs = illegal_closure_decision(first_truth) != illegal_closure_decision(second_truth)
    return lawful_equal and illegal_differs


def main() -> int:
    checks = {
        "counterexamples": {
            "frozen_dataclass_is_only_shallow": frozen_is_shallow_counterexample(),
            "no_truth_field_does_not_imply_noninterference": (
                no_truth_field_is_not_noninterference_counterexample()
            ),
            "same_observation_seed_config_omits_agent_state": (
                state_omission_breaks_replay_counterexample()
            ),
            "reset_changes_world_without_action_permit": (
                reset_is_a_world_change_counterexample()
            ),
            "plain_frozen_permit_is_forgeable": (
                frozen_permit_is_forgeable_counterexample()
            ),
        },
        "constructive_witnesses": {
            "authenticated_episode_tick_bound_single_use_permit": (
                permit_only_constructive_witness()
            ),
            "canonical_hash_chain_replays_and_detects_tamper": (
                deterministic_ledger_constructive_witness()
            ),
            "two_nonisomorphic_families_share_one_orchestrator": (
                two_family_adapter_constructive_witness()
            ),
            "visible_history_noninterference_is_stronger_than_field_name_audit": (
                visible_history_noninterference_boundary()
            ),
        },
    }
    if not all(all(group.values()) for group in checks.values()):
        raise AssertionError(checks)
    print(canonical_bytes(checks).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
