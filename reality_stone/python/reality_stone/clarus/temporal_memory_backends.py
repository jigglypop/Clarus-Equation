"""Memory backends and locked ablations for the temporal benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .episodic_memory import AuditedEpisodicMemory
from .runtime import HippocampusMemory
from .temporal_memory_protocol import (
    TemporalMemoryBenchConfig,
    _Event,
    _Recall,
    _unit,
)


@dataclass
class _Backend:
    name: str
    cfg: TemporalMemoryBenchConfig
    audited: AuditedEpisodicMemory | None = None
    legacy: HippocampusMemory | None = None
    fifo: list[tuple[torch.Tensor, int, str]] = field(default_factory=list)
    history: dict[str, list[tuple[int, int | None, str | None]]] = field(
        default_factory=dict
    )
    evidence_slot: dict[str, str] = field(default_factory=dict)
    event_count: int = 0

    @classmethod
    def build(cls, name: str, cfg: TemporalMemoryBenchConfig) -> "_Backend":
        result = cls(name, cfg)
        if name in {
            "candidate",
            "update_off",
            "abstention_off",
            "temporal_order_shuffle",
            "evidence_id_removed",
        }:
            result.audited = AuditedEpisodicMemory(
                cfg.dim,
                cfg.capacity,
                merge_updates=name != "update_off",
                abstention_enabled=name != "abstention_off",
            )
        elif name == "existing":
            result.legacy = HippocampusMemory(cfg.dim, capacity=cfg.capacity)
        return result

    def apply(self, event: _Event) -> None:
        if self.name == "no_memory":
            return
        if event.op == "DELETE":
            if self.audited is not None:
                self.audited.delete(event.evidence)
            elif self.name == "full_context":
                slot = self.evidence_slot.get(event.evidence)
                if slot is not None:
                    self.history.setdefault(slot, []).append((event.timestamp, None, None))
                self.event_count += 1
            return
        if event.slot is None or event.key is None or event.value is None:
            raise AssertionError("invalid UPSERT event")
        if self.audited is not None:
            self.audited.upsert(
                event.key,
                event.value,
                event.evidence,
                priority=event.priority,
                timestamp=event.timestamp,
            )
        elif self.legacy is not None:
            value_dim = self.cfg.cities + self.cfg.teams
            value = torch.nn.functional.one_hot(
                torch.tensor(event.value), value_dim
            ).float()
            self.legacy.encode(event.key, value, priority=event.priority)
        elif self.name == "fifo":
            if len(self.fifo) >= self.cfg.capacity:
                self.fifo.pop(0)
            self.fifo.append((_unit(event.key), event.value, event.evidence))
        elif self.name == "full_context":
            self.history.setdefault(event.slot, []).append(
                (event.timestamp, event.value, event.evidence)
            )
            self.evidence_slot[event.evidence] = event.slot
            self.event_count += 1

    def recall(self, slot: str | None, cue: torch.Tensor) -> _Recall:
        if self.audited is not None:
            result = self.audited.recall(cue)
            evidence = None if self.name == "evidence_id_removed" else result.evidence_id
            return _Recall(result.value, evidence, result.abstained)
        if self.legacy is not None:
            vector = self.legacy.recall(cue, topk=1)
            expected = self.cfg.cities + self.cfg.teams
            if vector.numel() != expected or vector.norm().item() == 0.0:
                return _Recall(None, None, True)
            return _Recall(int(vector.argmax().item()), None, False)
        if self.name == "fifo":
            ranked = sorted(
                ((float(key @ _unit(cue)), i) for i, (key, _, _) in enumerate(self.fifo)),
                reverse=True,
            )
            if not ranked:
                return _Recall(None, None, True)
            top, index = ranked[0]
            second = ranked[1][0] if len(ranked) > 1 else -1.0
            if top < 0.60 or top - second < 0.05:
                return _Recall(None, None, True)
            _, value, evidence = self.fifo[index]
            return _Recall(value, evidence, False)
        if self.name == "full_context" and slot in self.history:
            _, value, evidence = max(self.history[slot], key=lambda row: row[0])
            if value is not None:
                return _Recall(value, evidence, False)
        return _Recall(None, None, True)

    def __len__(self) -> int:
        if self.audited is not None:
            return len(self.audited)
        if self.legacy is not None:
            return len(self.legacy)
        if self.name == "fifo":
            return len(self.fifo)
        if self.name == "full_context":
            return self.event_count
        return 0


__all__ = ["_Backend"]
