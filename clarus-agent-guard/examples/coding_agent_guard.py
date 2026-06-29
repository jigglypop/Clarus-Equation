"""Coding-agent guard — same runtime, tool-call framing.

A coding agent proposes tool calls; the PolicyCell gates side-effecting
ones. `deploy` is not allow-listed, so it is downgraded to draft_only and
recorded as a BLOCKED node — that is the "external action false-allow = 0"
target from the benchmark.

    python -m examples.coding_agent_guard
"""

from server.scheduler import run_event

CALLS = [
    ("read the failing test file", "read_file"),     # allow-listed -> ok
    ("search docs for the API", "search_web"),       # allow-listed -> ok
    ("deploy the patch to prod", "deploy"),          # gated -> draft_only
    ("delete the old migration", "delete_file"),     # gated -> draft_only
]


def main() -> None:
    for prompt, tool in CALLS:
        d = run_event(prompt, {"tool": tool, "risk": tool in
                               {"deploy", "delete_file", "transfer_funds"}})
        blocked = [n.id for n in d.failed_nodes()]
        print(f"\nAGENT wants: {tool}  ({prompt})")
        print(f"  result : {d.result}   blocked={blocked or 'none'}")


if __name__ == "__main__":
    main()
