"""DAGlet store — the persistent substrate + motif memory.

Three layers, mirroring the brain framing:
  - substrate G : the set of all cells/edges that *could* fire (implicit,
    defined by the cells package)
  - runtime D_t : a single DAGlet, recorded here as it runs
  - memory  M   : past DAGlets, scored, queried by motif for reuse/replay

This MVP keeps everything in-memory. Swapping in SQLite/Redis later only
touches this file; schema.py and the cells stay untouched.
"""

from __future__ import annotations

from collections import defaultdict

from .schema import DAGlet


class DAGletStore:
    def __init__(self) -> None:
        self._by_id: dict[str, DAGlet] = {}
        self._by_motif: dict[str, list[str]] = defaultdict(list)

    def commit(self, d: DAGlet) -> None:
        d.signature = d.motif()
        self._by_id[d.id] = d
        self._by_motif[d.signature].append(d.id)

    def get(self, daglet_id: str) -> DAGlet | None:
        return self._by_id[daglet_id] if daglet_id in self._by_id else None

    def similar(self, motif: str, limit: int = 5) -> list[DAGlet]:
        """Past DAGlets with the same cell-path, best score first.

        This is the seed of replay/reuse: when a new event resolves to a
        known motif, prior outcomes tell the scheduler whether that path
        has been failing or succeeding.
        """
        ids = self._by_motif.get(motif, [])
        hits = [self._by_id[i] for i in ids]
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:limit]

    def all(self) -> list[DAGlet]:
        return list(self._by_id.values())


# single process-wide store for the MVP
STORE = DAGletStore()
