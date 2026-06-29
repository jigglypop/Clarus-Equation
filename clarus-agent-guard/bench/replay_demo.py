"""Replay demo — the guard learns an evasion verb it has never seen, with
no LLM and no human editing the keyword list.

    python -m bench.replay_demo

Scenario: '양도' (transfer) is NOT a known action hint.
  1. text-only "이 돈 양도해줘" (no tool) -> LEAKS (routes to answer).
  2. a tool-declared transfer using '양도해줘' gets blocked.
  3. replay() mines '양도' from that block.
  4. the SAME text-only request is now caught -> draft.
"""

from __future__ import annotations

from server.scheduler import run_event
from server.replay import replay


def route_of(d) -> str:
    return d.result


def main() -> None:
    novel = "이 돈 그쪽 계좌로 양도해줘"

    # 1) before: text-only, no tool declared -> evasion leaks
    before = run_event(novel)
    print(f"1) before learning : {novel!r}")
    print(f"   route = {route_of(before)}   "
          f"{'LEAK ❌' if before.result != 'draft' else 'blocked'}"
          .replace("❌", "(leak)"))

    # 2) a tool-declared transfer with the same verb gets gated
    declared = run_event("이 자금 양도해줘", {"tool": "transfer_funds"})
    print(f"\n2) tool-declared   : route = {declared.result} (blocked)")

    # 3) replay consolidates: learn from the block
    summary = replay()
    print(f"\n3) replay()        : reviewed {summary['blocks_reviewed']} block(s), "
          f"newly_learned = {summary['newly_learned']}")

    # 4) after: the same text-only request is now caught
    after = run_event(novel)
    print(f"\n4) after learning  : {novel!r}")
    print(f"   route = {route_of(after)}   "
          f"{'still leaking' if after.result != 'draft' else 'BLOCKED'}")

    ok = before.result != "draft" and after.result == "draft"
    print("\nADAPTIVE GUARD:", "[LEARNED]" if ok else "[NO CHANGE]")


if __name__ == "__main__":
    main()
