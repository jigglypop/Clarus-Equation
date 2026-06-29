"""The execution chokepoint. A side effect can ONLY happen by calling
`Executor.execute`; tool functions are never exposed directly. This is what
makes the gate non-vacuous: there is no bypass path to wrap or audit around.

Two enforcement checks, both independent of wording:

  1. AUTHORITY (I1) : a side-effecting tool runs only if its capability is
     in `granted`. `granted` is produced solely by `authorize(user_text)`,
     which by construction sees only trusted user input. Injected content
     therefore never yields a capability -> refused, no matter the phrasing.

  2. ARGUMENT INTEGRITY (I2) : even with a granted capability, a critical
     argument (recipient, target, amount) tainted by UNTRUSTED data is
     refused. This stops "user said send an email, but the address was
     lifted from an attacker-controlled web page" (exfiltration / argument
     tampering), which a pure capability check would miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .capability import Capability, CapabilityError, Trust, Value


@dataclass
class Tool:
    name: str
    fn: Callable[..., object]
    side_effecting: bool = False
    required_cap: Capability | None = None
    critical_args: tuple[str, ...] = ()   # args whose taint matters


# authorize() runs ONLY on trusted user text. The lexical map here is
# deliberately simple — its weakness does not matter, because injected text
# never reaches this function (it lives in data, not in the user turn).
_INTENT = {
    Capability.SEND_EMAIL: ("메일", "이메일", "발송", "보내", "send", "email"),
    Capability.DELETE_FILE: ("삭제", "지워", "delete", "remove"),
    Capability.DEPLOY: ("배포", "deploy", "릴리즈", "release"),
    Capability.TRANSFER: ("이체", "송금", "결제", "transfer", "pay"),
}


def authorize(user_text: str) -> set[Capability]:
    """Mint capabilities from a TRUSTED user request only. Never call this on
    tool output, retrieved documents, or memory — that would violate I1."""
    t = user_text.lower()
    return {cap for cap, words in _INTENT.items() if any(w in t for w in words)}


class Executor:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def execute(self, name: str, args: dict[str, Value],
                granted: set[Capability]) -> object:
        tool = self._tools.get(name)
        if tool is None:
            raise CapabilityError(f"unknown tool '{name}'")

        if tool.side_effecting:
            # check 1: authority
            if tool.required_cap not in granted:
                raise CapabilityError(
                    f"'{name}' requires {tool.required_cap.value!r}; "
                    f"not granted by the user turn")
            # check 2: argument integrity
            for arg_name in tool.critical_args:
                v = args.get(arg_name)
                if isinstance(v, Value) and v.trust == Trust.UNTRUSTED:
                    raise CapabilityError(
                        f"'{name}' critical arg '{arg_name}' is UNTRUSTED-"
                        f"tainted; refusing (possible exfiltration/tampering)")

        # only here, past both gates, does the real effect run.
        raw = {k: (v.payload if isinstance(v, Value) else v)
               for k, v in args.items()}
        return tool.fn(**raw)
