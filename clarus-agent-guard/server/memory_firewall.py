"""Memory Firewall — verify every memory_write before it reaches long-term
memory. Defends the ASB 'memory poisoning' axis.

A consolidation step (replay/dream) proposes updates to long-term memory.
Writing them blindly is how memory gets corrupted: injected instructions
get stored, hallucinated facts with no episode backing get promoted, and
existing records get silently overwritten. Three gates (after TrustMem's
coverage / preservation / faithfulness):

  1. POISONING   : an update carrying an imperative from untrusted content
                   is rejected. Memory holds facts, not commands.
  2. FAITHFULNESS: an abstracted update with no raw-episode backing is
                   rejected (no hallucinated memory).
  3. PRESERVATION: an update that contradicts an existing record is NOT
                   silently applied; it is quarantined for review.

Only updates that pass all three are committed. Raw episodes are always
kept, so abstractions remain auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cells.salience_cell import _ACTION_HINTS, _KNOWLEDGE_HINTS


@dataclass
class MemoryUpdate:
    key: str
    value: str
    provenance: str = "user"        # 'user' is trusted; 'tool'/'search'/'memory' are not
    raw_episode_id: str | None = None  # backing evidence; None = abstraction w/o source


@dataclass
class Verdict:
    ok: bool
    action: str                     # commit | reject | quarantine
    reasons: list[str] = field(default_factory=list)


@dataclass
class Record:
    key: str
    value: str
    provenance: str
    raw_episode_id: str | None


def _looks_like_instruction(text: str) -> bool:
    t = text.lower()
    if any(h in t for h in _KNOWLEDGE_HINTS):
        return False
    return any(h in t for h in _ACTION_HINTS)


class MemoryStore:
    def __init__(self) -> None:
        self.records: dict[str, Record] = {}
        self.quarantine: list[MemoryUpdate] = []
        self.raw_episodes: set[str] = set()

    def add_episode(self, episode_id: str) -> None:
        self.raw_episodes.add(episode_id)

    def verify(self, u: MemoryUpdate) -> Verdict:
        reasons: list[str] = []

        # gate 1: poisoning — an imperative from untrusted content
        if u.provenance != "user" and _looks_like_instruction(u.value):
            return Verdict(False, "reject",
                           ["poisoning: imperative from untrusted content"])

        # gate 2: faithfulness — abstraction must be backed by a real episode
        if u.raw_episode_id is None or u.raw_episode_id not in self.raw_episodes:
            return Verdict(False, "reject",
                           ["faithfulness: no raw-episode backing"])

        # gate 3: preservation — never silently overwrite a conflicting record
        prior = self.records.get(u.key)
        if prior and prior.value != u.value:
            # a trusted, episode-backed revision may supersede; otherwise hold
            if u.provenance == "user":
                reasons.append("preservation: trusted revision supersedes prior")
            else:
                return Verdict(False, "quarantine",
                               ["preservation: conflicts with existing record"])

        return Verdict(True, "commit", reasons or ["clean"])

    def commit(self, u: MemoryUpdate) -> Verdict:
        v = self.verify(u)
        if v.action == "commit":
            self.records[u.key] = Record(u.key, u.value, u.provenance,
                                         u.raw_episode_id)
        elif v.action == "quarantine":
            self.quarantine.append(u)
        return v


STORE = MemoryStore()
