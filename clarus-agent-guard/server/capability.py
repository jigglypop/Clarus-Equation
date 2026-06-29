"""Capability + taint core — structural injection defense (CaMeL-style).

The keyword cells are a *detection* layer: fallible, lexical, evadable. This
module is an *enforcement* layer that does not depend on recognising any
particular wording.

Two invariants make injection structurally impossible (not merely unlikely):

  I1  CAPABILITY PROVENANCE
      A capability to perform a side effect is minted ONLY from trusted
      (user) input, by `authorize()`. It can never be derived from data.

  I2  TAINT MONOTONICITY
      Every value carries a provenance trust level. Combining values takes
      the MINIMUM trust (untrusted infects). So data fetched from the web /
      memory / a tool can never launder itself into a higher trust level.

Consequence: an action proposed by injected content reaches the executor
with NO matching capability (because the capability could only come from the
user turn, which never contained that request). It is refused regardless of
how cleverly the injection is phrased. Detection is then a convenience for
UX (nice error messages), not the thing standing between you and a breach.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    SEND_EMAIL = "send_email"
    DELETE_FILE = "delete_file"
    DEPLOY = "deploy"
    TRANSFER = "transfer_funds"
    READ = "read"
    SEARCH = "search"


# trust lattice: higher = more trusted. USER is the only source of authority.
class Trust(int, Enum):
    UNTRUSTED = 0   # web / tool output / retrieved doc
    MEMORY = 1      # consolidated memory (semi-trusted)
    USER = 2        # the human's direct request


_PROVENANCE_TRUST = {
    "search": Trust.UNTRUSTED,
    "tool": Trust.UNTRUSTED,
    "doc": Trust.UNTRUSTED,
    "memory": Trust.MEMORY,
    "user": Trust.USER,
}


class CapabilityError(Exception):
    """Raised when a side effect is attempted without authority."""


@dataclass(frozen=True)
class Value:
    """A tainted value. `payload` is the data; `trust` is how much we trust
    its origin. Capabilities are NOT stored on data — only `authorize()`
    holds them — which is exactly what enforces I1.
    """

    payload: object
    trust: Trust

    @staticmethod
    def from_provenance(payload: object, provenance: str) -> "Value":
        return Value(payload, _PROVENANCE_TRUST.get(provenance, Trust.UNTRUSTED))

    def derive(self, payload: object, *others: "Value") -> "Value":
        """Produce a new value derived from self (+others). Trust is the
        minimum across all inputs — I2. There is no method that raises trust.
        """
        t = self.trust
        for o in others:
            t = min(t, o.trust, key=lambda x: int(x))
        return Value(payload, t)
