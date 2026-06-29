"""One-shot scoreboard — runs every benchmark and prints a single table.

    python -m bench.all

Each child bench prints a status line; we capture it and aggregate into a
pass/fail board. Exit code is non-zero if any guarantee breaks, so this
doubles as a CI gate.
"""

from __future__ import annotations

import re
import subprocess
import sys

# (module, axis, regex for the verdict line, token that means PASS)
BENCHES = [
    ("bench.run", "route accuracy (100)", r"RESULT: \[(\w+)\]", "PASS"),
    ("bench.hard_run", "adversarial (trap/evasion/injection)",
     r"GUARD SAFETY: \[(\w+)\]", "SAFE"),
    ("bench.inject_run", "indirect prompt injection (ASR)",
     r"INJECTION DEFENSE: \[(\w+)\]", "HELD"),
    ("bench.firewall_run", "memory poisoning (ASB)",
     r"MEMORY FIREWALL: \[(\w+)\]", "HELD"),
    ("bench.audit_check", "gate-bypass invariant",
     r"INVARIANT: \[(\w+)\]", "HOLDS"),
    ("bench.replay_demo", "adaptive learning",
     r"ADAPTIVE GUARD: \[(\w+)\]", "LEARNED"),
    ("bench.test_capability", "capability soundness (detection-free)",
     r"STRUCTURAL DEFENSE: \[(\w+)\]", "SOUND"),
    ("bench.agentdojo_suite", "AgentDojo-style (utility + ASR)",
     r"RESULT: \[(\w+)\]", "PASS"),
]


def main() -> None:
    rows = []
    all_ok = True
    for mod, axis, pattern, ok_token in BENCHES:
        out = subprocess.run([sys.executable, "-m", mod],
                             capture_output=True, text=True)
        text = out.stdout + out.stderr
        m = re.search(pattern, text)
        verdict = m.group(1) if m else "ERROR"
        # pull a headline metric line if present
        metric = ""
        for key in ("ASR", "route accuracy", "poison block-rate",
                    "INVARIANT"):
            mm = re.search(rf"{re.escape(key)}[^:]*:\s*(.+)", text)
            if mm:
                metric = mm.group(1).split("(")[0].strip()
                break
        ok = (verdict == ok_token)
        all_ok = all_ok and ok
        rows.append((axis, verdict, metric, ok))

    w = max(len(r[0]) for r in rows)
    print("\n  Clarus Agent Guard - benchmark scoreboard\n")
    print(f"  {'axis'.ljust(w)}  {'verdict':<9} metric")
    print(f"  {'-' * w}  {'-' * 9} {'-' * 24}")
    for axis, verdict, metric, ok in rows:
        mark = "OK " if ok else "XX "
        print(f"  {axis.ljust(w)}  {mark}{verdict:<6} {metric}")
    print(f"\n  OVERALL: {'ALL GUARANTEES HELD' if all_ok else 'FAILURE'}\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
