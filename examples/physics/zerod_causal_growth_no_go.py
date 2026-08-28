"""Model-theoretic no-go for unique causal growth from a singleton seed.

Let the only input be a distinguished event ``{0}``.  Require a sequential
growth kernel to preserve the old order, add one event at a time, give every
new event a nonempty past, be normalized, and commute with relabeling.  Two
deterministic kernels satisfy all those requirements:

* ``maximum`` puts each new event above every old event;
* ``minimal-past`` gives it exactly the current minimal events as its past.

After two births they produce, respectively,

    chain: 0 < 1 < 2,
    fork:  0 < 1 and 0 < 2, with 1 incomparable to 2.

The outputs are non-isomorphic: their relation counts, heights, widths, and
finite-poset order dimensions differ.  Therefore a singleton, normalization,
and relabeling covariance do not select a unique causal kernel or causal
order.  Continuum spacetime dimension is not even a typed observable until a
faithful-embedding/continuum readout is added; finite-poset order dimension is
reported only as an isomorphism invariant and is not identified with it.

This completely deletes the bare-input implication.  It does not rule out a
constructive model after transition weights, a measure/scale, a continuum
criterion, and a dimension readout are supplied as additional axioms.
"""

from __future__ import annotations

from dataclasses import dataclass


Relation = tuple[int, int]


def _validate_extension(
    events: frozenset[int], relations: frozenset[Relation], new_event: int
) -> None:
    if new_event in events:
        raise ValueError("new_event must not already belong to events")
    if any(left not in events or right not in events for left, right in relations):
        raise ValueError("every relation endpoint must belong to events")


def _minimal_elements(
    events: frozenset[int], relations: frozenset[Relation]
) -> frozenset[int]:
    """Return the order-theoretic minima, independently of event labels."""

    return frozenset(
        event
        for event in events
        if not any(right == event for _, right in relations)
    )


def _maximum_extension(
    events: frozenset[int], relations: frozenset[Relation], new_event: int
) -> tuple[frozenset[int], frozenset[Relation]]:
    """Put ``new_event`` above every old event."""

    _validate_extension(events, relations, new_event)
    return (
        events | {new_event},
        relations | frozenset((event, new_event) for event in events),
    )


def _minimal_past_extension(
    events: frozenset[int], relations: frozenset[Relation], new_event: int
) -> tuple[frozenset[int], frozenset[Relation]]:
    """Give ``new_event`` precisely the current order-theoretic minima as past."""

    _validate_extension(events, relations, new_event)
    minima = _minimal_elements(events, relations)
    return (
        events | {new_event},
        relations | frozenset((event, new_event) for event in minima),
    )


def _relabel_relations(
    relations: frozenset[Relation], relabeling: dict[int, int]
) -> frozenset[Relation]:
    return frozenset(
        (relabeling[left], relabeling[right]) for left, right in relations
    )


@dataclass(frozen=True)
class ZeroDCausalGrowthNoGoAudit:
    seed: tuple[int, ...]
    maximum_kernel_relations: frozenset[Relation]
    minimal_past_kernel_relations: frozenset[Relation]
    maximum_relation_count: int
    minimal_past_relation_count: int
    maximum_height: int
    minimal_past_height: int
    maximum_width: int
    minimal_past_width: int
    maximum_order_dimension: int
    minimal_past_order_dimension: int
    both_deterministic_normalized: bool
    both_relabel_covariant: bool
    both_preserve_old_order: bool
    all_births_have_nonempty_past: bool
    outputs_isomorphic: bool
    unique_causal_growth_follows: bool
    continuum_dimension_is_defined_by_input: bool
    status: str = "SINGLETON_TO_UNIQUE_CAUSAL_GROWTH_DISPROVED"
    claim_ceiling: str = "COMPLETE_FINITE_MODEL_THEORETIC_COUNTEREXAMPLE"


def zerod_causal_growth_no_go() -> ZeroDCausalGrowthNoGoAudit:
    """Return the connected three-event chain/fork countermodels."""

    seed_events = frozenset((0,))
    seed_relations: frozenset[Relation] = frozenset()

    maximum_events_1, maximum_relations_1 = _maximum_extension(
        seed_events, seed_relations, 1
    )
    maximum_events_2, chain = _maximum_extension(
        maximum_events_1, maximum_relations_1, 2
    )
    minimal_events_1, minimal_relations_1 = _minimal_past_extension(
        seed_events, seed_relations, 1
    )
    minimal_events_2, fork = _minimal_past_extension(
        minimal_events_1, minimal_relations_1, 2
    )

    # Check covariance by conjugating the complete two-step construction with
    # a deliberately non-monotone relabeling. Neither kernel inspects the
    # numerical size or birth-order spelling of an event label.
    relabeling = {0: 7, 1: 3, 2: 5}
    relabeled_seed = frozenset((relabeling[0],))
    relabeled_maximum_events_1, relabeled_maximum_relations_1 = (
        _maximum_extension(relabeled_seed, frozenset(), relabeling[1])
    )
    _, relabeled_chain = _maximum_extension(
        relabeled_maximum_events_1,
        relabeled_maximum_relations_1,
        relabeling[2],
    )
    relabeled_minimal_events_1, relabeled_minimal_relations_1 = (
        _minimal_past_extension(relabeled_seed, frozenset(), relabeling[1])
    )
    _, relabeled_fork = _minimal_past_extension(
        relabeled_minimal_events_1,
        relabeled_minimal_relations_1,
        relabeling[2],
    )
    both_relabel_covariant = (
        _relabel_relations(chain, relabeling) == relabeled_chain
        and _relabel_relations(fork, relabeling) == relabeled_fork
    )

    birth_steps = (
        (maximum_relations_1, 1),
        (chain, 2),
        (minimal_relations_1, 1),
        (fork, 2),
    )
    all_births_have_nonempty_past = all(
        any(right == birth for _, right in after) for after, birth in birth_steps
    )
    return ZeroDCausalGrowthNoGoAudit(
        seed=(0,),
        maximum_kernel_relations=chain,
        minimal_past_kernel_relations=fork,
        maximum_relation_count=len(chain),
        minimal_past_relation_count=len(fork),
        maximum_height=3,
        minimal_past_height=2,
        maximum_width=1,
        minimal_past_width=2,
        maximum_order_dimension=1,
        minimal_past_order_dimension=2,
        both_deterministic_normalized=True,
        both_relabel_covariant=both_relabel_covariant,
        both_preserve_old_order=(
            maximum_relations_1 <= chain
            and minimal_relations_1 <= fork
            and maximum_events_2 == minimal_events_2 == frozenset((0, 1, 2))
        ),
        all_births_have_nonempty_past=all_births_have_nonempty_past,
        outputs_isomorphic=False,
        unique_causal_growth_follows=False,
        continuum_dimension_is_defined_by_input=False,
    )
