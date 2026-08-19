"""Locked synthetic temporal-memory protocol and scenario generator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class TemporalMemoryBenchConfig:
    dim: int = 64
    people: int = 8
    teams: int = 4
    cities: int = 16
    interference_per_session: int = 8
    capacity: int = 24
    unseen_queries: int = 4
    seeds: int = 32
    key_noise: float = 0.025
    bootstrap_draws: int = 3000

    def __post_init__(self) -> None:
        slots = 2 * self.people + self.teams + 2 * self.interference_per_session
        if self.people < 4 or self.teams < 2 or self.cities < 8:
            raise ValueError("locked protocol minimums were violated")
        if self.people % self.teams:
            raise ValueError("people must be divisible by teams")
        if self.dim <= slots + self.unseen_queries:
            raise ValueError("dim must leave an unseen orthogonal subspace")
        if self.capacity < 2 * self.people + self.teams:
            raise ValueError("capacity must fit the merged active core")
        if self.seeds < 1 or self.bootstrap_draws < 100:
            raise ValueError("seeds and bootstrap_draws must be positive")


@dataclass(frozen=True)
class _Event:
    op: Literal["UPSERT", "DELETE"]
    slot: str | None
    key: torch.Tensor | None
    value: int | None
    evidence: str
    priority: float
    timestamp: int


@dataclass(frozen=True)
class _Recall:
    value: int | None
    evidence: str | None
    abstained: bool


@dataclass(frozen=True)
class _Scenario:
    events: tuple[_Event, ...]
    bases: dict[str, torch.Tensor]
    current: dict[str, tuple[int, str]]
    stale: dict[str, frozenset[int]]
    temporal_slots: tuple[str, ...]
    deleted_slots: tuple[str, ...]
    deleted_team: int
    person_teams: tuple[int, ...]
    unseen: tuple[torch.Tensor, ...]
    expected_updates: int
    expected_deletes: int


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / value.norm().clamp(min=1e-12)


def _lcb(values: list[float], *, seed: int, draws: int) -> float:
    rng = random.Random(seed)
    means = [
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(draws)
    ]
    return sorted(means)[max(0, int(0.025 * draws) - 1)]


def _build_scenario(seed: int, cfg: TemporalMemoryBenchConfig) -> _Scenario:
    generator = torch.Generator().manual_seed(seed)
    core = [f"loc:p{i}" for i in range(cfg.people)]
    core += [f"team:p{i}" for i in range(cfg.people)]
    core += [f"hq:t{i}" for i in range(cfg.teams)]
    distractors = [
        f"d:s{session}:i{i}"
        for session in (2, 3)
        for i in range(cfg.interference_per_session)
    ]
    slots = core + distractors
    q = torch.linalg.qr(torch.randn(cfg.dim, len(slots), generator=generator)).Q.T
    bases = {slot: q[i] for i, slot in enumerate(slots)}
    span = torch.stack(list(bases.values()))

    events: list[_Event] = []
    current: dict[str, tuple[int, str]] = {}
    stale: dict[str, set[int]] = {}
    timestamp = 0

    def add(session: int, slot: str, value: int, label: str, priority: float) -> str:
        nonlocal timestamp
        evidence = f"seed{seed}:{label}:{slot}"
        noise = cfg.key_noise * torch.randn(cfg.dim, generator=generator)
        events.append(
            _Event(
                "UPSERT",
                slot,
                _unit(bases[slot] + noise),
                value,
                evidence,
                priority,
                timestamp,
            )
        )
        current[slot] = (value, evidence)
        timestamp += 1
        return evidence

    person_teams = tuple(i % cfg.teams for i in range(cfg.people))
    old_loc: dict[str, int] = {}
    old_hq: dict[str, int] = {}
    for person in range(cfg.people):
        slot = f"loc:p{person}"
        old = (3 * person + seed) % cfg.cities
        old_loc[slot] = old
        stale[slot] = {old}
        add(1, slot, old, "s1-old", 4.0)
    for person, team in enumerate(person_teams):
        add(1, f"team:p{person}", cfg.cities + team, "s1-team", 4.0)
    for team in range(cfg.teams):
        slot = f"hq:t{team}"
        old = (5 * team + seed + 1) % cfg.cities
        old_hq[slot] = old
        stale[slot] = {old}
        add(1, slot, old, "s1-old", 4.0)

    temporal_people = range(cfg.people // 2)
    temporal_teams = range(cfg.teams // 2)
    for person in temporal_people:
        slot = f"loc:p{person}"
        mid = (old_loc[slot] + 5) % cfg.cities
        stale[slot].add(mid)
        add(2, slot, mid, "s2-mid", 4.5)
    for team in temporal_teams:
        slot = f"hq:t{team}"
        mid = (old_hq[slot] + 6) % cfg.cities
        stale[slot].add(mid)
        add(2, slot, mid, "s2-mid", 4.5)
    for i in range(cfg.interference_per_session):
        add(2, f"d:s2:i{i}", (i + seed) % cfg.cities, "s2-distractor", 0.25)

    final_evidence: dict[str, str] = {}
    for person in range(cfg.people):
        slot = f"loc:p{person}"
        final = (old_loc[slot] + 9) % cfg.cities
        if final in stale[slot]:
            final = (final + 1) % cfg.cities
        final_evidence[slot] = add(3, slot, final, "s3-final", 5.0)
    for team in range(cfg.teams):
        slot = f"hq:t{team}"
        final = (old_hq[slot] + 10) % cfg.cities
        if final in stale[slot]:
            final = (final + 1) % cfg.cities
        final_evidence[slot] = add(3, slot, final, "s3-final", 5.0)
    for i in range(cfg.interference_per_session):
        add(3, f"d:s3:i{i}", (i + seed + 7) % cfg.cities, "s3-distractor", 0.25)

    deleted_team = cfg.teams - 1
    deleted_slots = (f"loc:p{cfg.people - 1}", f"hq:t{deleted_team}")
    for slot in deleted_slots:
        events.append(
            _Event("DELETE", None, None, None, final_evidence[slot], 1.0, timestamp)
        )
        current.pop(slot)
        timestamp += 1

    unseen: list[torch.Tensor] = []
    for _ in range(cfg.unseen_queries):
        vector = torch.randn(cfg.dim, generator=generator)
        vector = vector - (span @ vector) @ span
        if vector.norm().item() <= 1e-8:
            raise RuntimeError("could not construct unseen orthogonal cue")
        unseen.append(_unit(vector))

    temporal_slots = tuple(
        [f"loc:p{i}" for i in temporal_people]
        + [f"hq:t{i}" for i in temporal_teams]
    )
    return _Scenario(
        tuple(events),
        bases,
        current,
        {slot: frozenset(values) for slot, values in stale.items()},
        temporal_slots,
        deleted_slots,
        deleted_team,
        person_teams,
        tuple(unseen),
        cfg.people + cfg.teams + cfg.people // 2 + cfg.teams // 2,
        len(deleted_slots),
    )


__all__ = [
    "TemporalMemoryBenchConfig",
    "_Event",
    "_Recall",
    "_Scenario",
    "_build_scenario",
    "_lcb",
    "_unit",
]
