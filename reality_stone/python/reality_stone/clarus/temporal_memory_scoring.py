"""Scoring and per-seed execution for the locked temporal benchmark."""

from __future__ import annotations

import random

import torch

from .temporal_memory_backends import _Backend
from .temporal_memory_protocol import (
    TemporalMemoryBenchConfig,
    _Recall,
    _Scenario,
    _build_scenario,
    _unit,
)


def _query(
    backend: _Backend,
    scenario: _Scenario,
    slot: str,
    generator: torch.Generator,
    cfg: TemporalMemoryBenchConfig,
) -> _Recall:
    noise = cfg.key_noise * torch.randn(cfg.dim, generator=generator)
    return backend.recall(slot, _unit(scenario.bases[slot] + noise))


def _score(
    backend: _Backend,
    scenario: _Scenario,
    seed: int,
    cfg: TemporalMemoryBenchConfig,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 420_000)
    factual = sorted(slot for slot in scenario.current if not slot.startswith("d:"))
    updated = [slot for slot in factual if slot in scenario.stale]

    latest = provenance = stale = 0
    for slot in factual:
        expected_value, expected_evidence = scenario.current[slot]
        result = _query(backend, scenario, slot, generator, cfg)
        latest += int(not result.abstained and result.value == expected_value)
        provenance += int(result.evidence == expected_evidence)
        if slot in scenario.stale:
            stale += int(not result.abstained and result.value in scenario.stale[slot])

    temporal = 0
    temporal_slots = [slot for slot in scenario.temporal_slots if slot in scenario.current]
    for slot in temporal_slots:
        expected, _ = scenario.current[slot]
        result = _query(backend, scenario, slot, generator, cfg)
        temporal += int(not result.abstained and result.value == expected)

    multihop = multihop_total = deleted_hop = deleted_hop_residual = 0
    for person, team in enumerate(scenario.person_teams):
        first = _query(backend, scenario, f"team:p{person}", generator, cfg)
        if first.abstained or first.value != cfg.cities + team:
            if team != scenario.deleted_team:
                multihop_total += 1
            continue
        second = _query(backend, scenario, f"hq:t{team}", generator, cfg)
        if team == scenario.deleted_team:
            deleted_hop += int(second.abstained)
            deleted_hop_residual += int(not second.abstained)
        else:
            expected, _ = scenario.current[f"hq:t{team}"]
            multihop += int(not second.abstained and second.value == expected)
            multihop_total += 1

    unknown = sum(int(backend.recall(None, cue).abstained) for cue in scenario.unseen)
    deleted_direct = deleted_direct_residual = 0
    for slot in scenario.deleted_slots:
        result = _query(backend, scenario, slot, generator, cfg)
        deleted_direct += int(result.abstained)
        deleted_direct_residual += int(not result.abstained)

    deleted_people = sum(team == scenario.deleted_team for team in scenario.person_teams)
    latest_acc = latest / max(len(factual), 1)
    provenance_acc = provenance / max(len(factual), 1)
    temporal_acc = temporal / max(len(temporal_slots), 1)
    multihop_acc = multihop / max(multihop_total, 1)
    deleted_hop_acc = deleted_hop / max(deleted_people, 1)
    abstention_acc = (unknown + deleted_direct) / max(
        len(scenario.unseen) + len(scenario.deleted_slots), 1
    )
    stale_rate = stale / max(len(updated), 1)
    delete_residual = (deleted_direct_residual + deleted_hop_residual) / max(
        len(scenario.deleted_slots) + deleted_people, 1
    )
    components = (
        latest_acc,
        provenance_acc,
        temporal_acc,
        multihop_acc,
        deleted_hop_acc,
        abstention_acc,
        1.0 - stale_rate,
        1.0 - delete_residual,
    )
    return {
        "latest_accuracy": latest_acc,
        "provenance_accuracy": provenance_acc,
        "temporal_order_accuracy": temporal_acc,
        "multihop_accuracy": multihop_acc,
        "deleted_multihop_accuracy": deleted_hop_acc,
        "abstention_accuracy": abstention_acc,
        "stale_error_rate": stale_rate,
        "delete_residual_rate": delete_residual,
        "composite": sum(components) / len(components),
        "storage_count": float(len(backend)),
    }


def _seed_result(seed: int, cfg: TemporalMemoryBenchConfig) -> dict[str, dict[str, float]]:
    scenario = _build_scenario(seed, cfg)
    names = (
        "candidate",
        "existing",
        "full_context",
        "fifo",
        "update_off",
        "abstention_off",
        "temporal_order_shuffle",
        "evidence_id_removed",
        "no_memory",
    )
    backends = {name: _Backend.build(name, cfg) for name in names}
    for name, backend in backends.items():
        events = list(scenario.events)
        if name == "temporal_order_shuffle":
            random.Random(seed + 202_608_19).shuffle(events)
        for event in events:
            backend.apply(event)
    metrics = {name: _score(backend, scenario, seed, cfg) for name, backend in backends.items()}
    candidate = backends["candidate"].audited
    if candidate is None:
        raise AssertionError("candidate memory was not initialized")
    metrics["candidate"].update(
        {
            "capacity_ok": float(len(candidate) <= cfg.capacity),
            "update_audit_count": float(
                sum(row["operation"] == "UPDATE" for row in candidate.audit_log)
            ),
            "delete_audit_count": float(
                sum(row["operation"] == "DELETE" for row in candidate.audit_log)
            ),
            "expected_update_audit_count": float(scenario.expected_updates),
            "expected_delete_audit_count": float(scenario.expected_deletes),
        }
    )
    return metrics


__all__ = ["_seed_result"]
