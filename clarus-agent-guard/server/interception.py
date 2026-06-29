"""Interception plumbing — make the chokepoint non-vacuous at the CALL layer.

The capability layer is only as real as the guarantee that there is no other
way to cause a side effect. Here we enforce that physically:

  @gate.guarded(...) wraps a tool function and returns a SEALED handle. The
  raw callable is captured inside the gate's closure and is never handed
  back. Calling the sealed handle directly raises. The only path to the real
  effect is `gate.dispatch(name, args, user_text)`, which runs the capability
  checks and then executes.

So even agent code that holds the tool object cannot bypass the gate — there
is nothing to bypass to. This is the plumbing that turns "every external
action passes the gate" from a convention into a property of the call graph.

In a real deployment the `@guarded` decorator is applied at the tool-
registration boundary (MCP server, function-call schema, shell wrapper); the
agent is handed only sealed handles + the dispatcher.
"""

from __future__ import annotations

from typing import Callable

from .capability import Capability, Value
from .executor import Executor, Tool, authorize


class InterceptionError(Exception):
    """Raised when a guarded tool is invoked outside the gate."""


class _Sealed:
    """A non-callable stand-in for a guarded tool. Holds no reference to the
    real function, so it cannot be used to bypass the gate."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *a, **k):
        raise InterceptionError(
            f"'{self._name}' is sealed; invoke via gate.dispatch(), not directly")

    def __repr__(self) -> str:
        return f"<sealed tool '{self._name}'>"


class ToolGate:
    def __init__(self) -> None:
        self._ex = Executor()

    def guarded(self, *, side_effecting: bool = False,
                required_cap: Capability | None = None,
                critical_args: tuple[str, ...] = ()) -> Callable:
        def deco(fn: Callable) -> _Sealed:
            self._ex.register(Tool(fn.__name__, fn, side_effecting,
                                   required_cap, critical_args))
            return _Sealed(fn.__name__)   # raw fn now only lives in the gate
        return deco

    def dispatch(self, name: str, args: dict, *, user_text: str = "") -> object:
        """The ONLY way to invoke a guarded tool.

        `args` maps name -> value, or name -> (value, provenance). Capabilities
        are minted from `user_text` (trusted) only.
        """
        granted = authorize(user_text)
        vals: dict[str, Value] = {}
        for k, v in args.items():
            if isinstance(v, tuple) and len(v) == 2:
                value, prov = v
            else:
                value, prov = v, "user"
            vals[k] = Value.from_provenance(value, prov)
        return self._ex.execute(name, vals, granted)


# the gate the runtime shares
GATE = ToolGate()
