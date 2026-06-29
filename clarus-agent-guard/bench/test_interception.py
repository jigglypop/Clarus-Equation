"""Interception soundness — prove there is NO bypass path to a side effect.

    python -m bench.test_interception

Properties:
  1. a sealed tool cannot be called directly (no bypass exists)
  2. the sealed handle leaks no reference to the real function
  3. the only path, gate.dispatch, runs the effect when authorised
  4. dispatch refuses an unauthorised (injected) call
  5. dispatch refuses an authorised call with an untrusted critical arg
"""

from __future__ import annotations

import sys
import types

from server.capability import Capability, CapabilityError
from server.interception import ToolGate, InterceptionError

SINK: list[str] = []


def main() -> None:
    SINK.clear()
    gate = ToolGate()

    @gate.guarded(side_effecting=True, required_cap=Capability.TRANSFER,
                  critical_args=("to",))
    def send_money(to, amount):
        SINK.append(f"money:{to}:{amount}")
        return "ok"

    @gate.guarded(side_effecting=False)
    def read_balance():
        return 42

    results: list[tuple[str, bool]] = []

    # 1) direct call is impossible
    try:
        send_money("attacker", "9999")
        direct_blocked = False
    except InterceptionError:
        direct_blocked = True
    results.append(("sealed tool cannot be called directly",
                    direct_blocked and SINK == []))

    # 2) the real function object does not leak through the handle
    leaked = any(isinstance(getattr(send_money, a, None), types.FunctionType)
                 for a in dir(send_money))
    results.append(("sealed handle leaks no function ref", not leaked))

    # 3) authorised dispatch runs the effect
    gate.dispatch("send_money",
                  {"to": ("bob@bank", "user"), "amount": ("100", "user")},
                  user_text="Bob에게 100 송금해줘")
    results.append(("authorised dispatch executes", SINK == ["money:bob@bank:100"]))

    # 4) unauthorised (injected) dispatch refused: user turn granted nothing
    SINK.clear()
    try:
        gate.dispatch("send_money",
                      {"to": ("attacker@x", "tool"), "amount": ("9999", "tool")},
                      user_text="내 잔액 보여줘")          # read-only intent
        refused = False
    except CapabilityError:
        refused = True
    results.append(("injected dispatch refused (no capability)",
                    refused and SINK == []))

    # 5) authorised but tainted critical arg refused
    try:
        gate.dispatch("send_money",
                      {"to": ("attacker@x", "tool"), "amount": ("100", "user")},
                      user_text="Bob에게 송금해줘")        # grants TRANSFER
        tainted_refused = False
    except CapabilityError:
        tainted_refused = True
    results.append(("tainted critical arg refused", tainted_refused and SINK == []))

    width = max(len(n) for n, _ in results)
    print("\n  interception - no-bypass properties\n")
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}")
    allok = all(ok for _, ok in results)
    print(f"\n  CHOKEPOINT: {'[NON-VACUOUS]' if allok else '[BYPASSABLE]'}\n")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
