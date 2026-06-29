"""Replay loop — the idle-time consolidation step (the "dream").

Online, the runtime only *reacts*. Replay walks memory M (committed
DAGlets) and learns from what got blocked:

  - any DAGlet that resolved to `draft` was a real action the gate caught
  - mine the imperative verb stem from its event text
  - if that stem is not already a known action hint, learn it

Result: a verb the guard only caught because a tool was declared becomes
catchable from *text alone* next time. That closes the evasion gap the
hard benchmark exposed, adaptively, with no LLM and no human edit.

This is deliberately conservative: it only learns from confirmed blocks,
so it cannot invent risk out of harmless traffic.
"""

from __future__ import annotations

from .cells.salience_cell import _ACTION_HINTS
from .learn import LEARNED_ACTION_HINTS
from .trace.store import STORE

_IMPERATIVE_TAILS = ("해줘", "해", "줘", "처리해줘", "하자", "해주세요")


def _stems(event: str) -> list[str]:
    """Imperative verb stems, e.g. '양도해줘' -> '양도', '넘겨줘' -> '넘겨'."""
    out: list[str] = []
    for word in event.replace("?", " ").replace(".", " ").split():
        w = word.strip()
        for tail in sorted(_IMPERATIVE_TAILS, key=len, reverse=True):
            if w.endswith(tail) and len(w) - len(tail) >= 2:
                out.append(w[: len(w) - len(tail)])
                break
    return out


def replay(store=STORE) -> dict:
    """Consolidate memory; return a summary of what was learned."""
    learned: list[str] = []
    blocks = 0
    for d in store.all():
        if d.result != "draft":          # only learn from caught actions
            continue
        blocks += 1
        for stem in _stems(d.event):
            known = any(stem in h or h in stem for h in _ACTION_HINTS)
            if known or stem in LEARNED_ACTION_HINTS:
                continue
            LEARNED_ACTION_HINTS.add(stem)
            learned.append(stem)
    return {
        "blocks_reviewed": blocks,
        "newly_learned": learned,
        "total_learned": sorted(LEARNED_ACTION_HINTS),
    }
