"""Finite DAG decision slices with feedback committed across time steps.

This is an engineering abstraction of recurrent action-selection circuits.  It
does not assert that basal-ganglia anatomy is itself acyclic or tree-shaped.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DagNode:
    node_id: int
    depth: int
    kind: str


@dataclass(frozen=True)
class DagEdge:
    source: int
    target: int


@dataclass(frozen=True)
class RecurrentDagConfig:
    action_count: int = 8
    context_masks: tuple[int, ...] = (0, 3, 5, 6)
    state_decay: float = 0.82
    cue_gain: float = 1.0
    feedback_gain: float = 1.35
    policy_temperature: float = 0.35
    inhibition_strength: float = 0.20
    state_norm_cap: float = 6.0

    def __post_init__(self) -> None:
        if self.action_count < 2 or self.action_count & (self.action_count - 1):
            raise ValueError("action_count must be a power of two")
        if not self.context_masks:
            raise ValueError("at least one context mask is required")
        if any(mask < 0 or mask >= self.action_count for mask in self.context_masks):
            raise ValueError("context masks must index the action space")
        if not 0.0 <= self.state_decay < 1.0:
            raise ValueError("state_decay must lie in [0, 1)")
        for name, value in (
            ("cue_gain", self.cue_gain),
            ("feedback_gain", self.feedback_gain),
            ("policy_temperature", self.policy_temperature),
            ("state_norm_cap", self.state_norm_cap),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.inhibition_strength) or self.inhibition_strength < 0.0:
            raise ValueError("inhibition_strength must be finite and nonnegative")


@dataclass(frozen=True)
class RecurrentDagOutput:
    action: int
    probabilities: tuple[float, ...]
    context_probabilities: tuple[float, ...]
    base_action: int
    predicted_actions_by_context: tuple[int, ...]
    evaluated_nodes: int
    evaluated_edges: int


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values / temperature
    scaled = scaled - float(np.max(scaled))
    weights = np.exp(scaled)
    return weights / float(np.sum(weights))


def context_action_topology(config: RecurrentDagConfig) -> tuple[tuple[DagNode, ...], tuple[DagEdge, ...]]:
    nodes = [DagNode(0, 0, "input")]
    context_offset = 1
    action_offset = context_offset + len(config.context_masks)
    nodes.extend(
        DagNode(context_offset + index, 1, "context")
        for index in range(len(config.context_masks))
    )
    nodes.extend(
        DagNode(action_offset + index, 2, "action")
        for index in range(config.action_count)
    )
    edges = [
        DagEdge(0, context_offset + index)
        for index in range(len(config.context_masks))
    ]
    edges.extend(
        DagEdge(context_offset + context, action_offset + action)
        for context in range(len(config.context_masks))
        for action in range(config.action_count)
    )
    return tuple(nodes), tuple(edges)


def validate_topology(nodes: Sequence[DagNode], edges: Sequence[DagEdge]) -> None:
    node_map = {node.node_id: node for node in nodes}
    if len(node_map) != len(nodes):
        raise ValueError("DAG node identifiers must be unique")
    for edge in edges:
        if edge.source not in node_map or edge.target not in node_map:
            raise ValueError("DAG edge references an unknown node")
        source = node_map[edge.source]
        target = node_map[edge.target]
        if source.depth >= target.depth:
            raise ValueError("every within-step edge must increase topological depth")


class RecurrentDecisionDag:
    """Conditional context-to-action DAG with delayed feedback recurrence."""

    def __init__(self, config: RecurrentDagConfig = RecurrentDagConfig()) -> None:
        self.config = config
        self.nodes, self.edges = context_action_topology(config)
        validate_topology(self.nodes, self.edges)
        self._state = np.zeros(len(config.context_masks), dtype=np.float64)
        self._pending: RecurrentDagOutput | None = None
        self.commit_count = 0
        self.nonfinite_count = 0

    @property
    def state(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._state)

    def reset(self) -> None:
        self._state.fill(0.0)
        self._pending = None
        self.commit_count = 0
        self.nonfinite_count = 0

    def forward_step(
        self,
        content_evidence: Sequence[float],
        context_logits: Sequence[float],
    ) -> RecurrentDagOutput:
        bit_count = int(math.log2(self.config.action_count))
        content = np.asarray(content_evidence, dtype=np.float64)
        cues = np.asarray(context_logits, dtype=np.float64)
        if content.shape != (bit_count,):
            raise ValueError("content evidence has the wrong bit dimension")
        if cues.shape != self._state.shape:
            raise ValueError("context logits have the wrong dimension")
        if not np.all(np.isfinite(content)) or not np.all(np.isfinite(cues)):
            raise ValueError("DAG inputs must be finite")

        base_action = sum(int(value >= 0.0) << index for index, value in enumerate(content))
        context_probabilities = _softmax(
            self.config.cue_gain * cues + self._state
        )
        predicted = tuple(base_action ^ mask for mask in self.config.context_masks)
        proposal = np.full(self.config.action_count, 1e-12, dtype=np.float64)
        for context, action in enumerate(predicted):
            proposal[action] += context_probabilities[context]
        promotion = np.log(proposal)

        inhibition = np.zeros_like(promotion)
        for action in range(self.config.action_count):
            competitors = np.delete(promotion, action)
            maximum = float(np.max(competitors))
            log_mean_exp = maximum + math.log(
                float(np.mean(np.exp(competitors - maximum)))
            )
            inhibition[action] = self.config.inhibition_strength * log_mean_exp
        probabilities = _softmax(
            promotion - inhibition,
            self.config.policy_temperature,
        )
        if not np.all(np.isfinite(probabilities)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite DAG policy")
        output = RecurrentDagOutput(
            action=int(np.argmax(probabilities)),
            probabilities=tuple(float(value) for value in probabilities),
            context_probabilities=tuple(float(value) for value in context_probabilities),
            base_action=base_action,
            predicted_actions_by_context=predicted,
            evaluated_nodes=len(self.nodes),
            evaluated_edges=len(self.edges),
        )
        self._pending = output
        return output

    def commit_feedback(
        self,
        signed_feedback: float,
        *,
        eligibility_permutation: Sequence[int] | None = None,
        flip_sign: bool = False,
    ) -> None:
        if self._pending is None:
            raise RuntimeError("feedback requires one preceding forward step")
        if not math.isfinite(signed_feedback) or abs(signed_feedback) > 1.0:
            raise ValueError("signed feedback must be finite and lie in [-1, 1]")
        feedback = -signed_feedback if flip_sign else signed_feedback
        chosen = self._pending.action
        count = len(self.config.context_masks)
        eligibility = np.asarray(
            [
                1.0 if action == chosen else -1.0 / max(1, count - 1)
                for action in self._pending.predicted_actions_by_context
            ],
            dtype=np.float64,
        )
        if eligibility_permutation is not None:
            permutation = np.asarray(tuple(eligibility_permutation), dtype=np.int64)
            if sorted(permutation.tolist()) != list(range(count)):
                raise ValueError("eligibility permutation must be a full permutation")
            eligibility = eligibility[permutation]
        update = self.config.feedback_gain * feedback * eligibility
        self._state = self.config.state_decay * self._state + update
        self._state -= float(np.mean(self._state))
        norm = float(np.linalg.norm(self._state))
        if norm > self.config.state_norm_cap:
            self._state *= self.config.state_norm_cap / norm
        if not np.all(np.isfinite(self._state)):
            self.nonfinite_count += 1
            raise FloatingPointError("nonfinite recurrent state")
        self.commit_count += 1
        self._pending = None


__all__ = [
    "DagEdge",
    "DagNode",
    "RecurrentDagConfig",
    "RecurrentDagOutput",
    "RecurrentDecisionDag",
    "context_action_topology",
    "validate_topology",
]
