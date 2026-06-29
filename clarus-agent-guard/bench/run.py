"""Benchmark runner — measures the demo success metrics.

    python -m bench.run

Reports:
  * route accuracy            (target 85%+)
  * external-action false-allow rate  (target 0%)
  * avg / p95 latency per event       (target +300ms or less)
  * per-category confusion
"""

from __future__ import annotations

import time

from server.scheduler import run_event
from server.trace.schema import EdgeKind
from .dataset import load


def predict_route(daglet) -> str:
    """Read the route the runtime actually took off the DAGlet shape."""
    cells = {n.cell for n in daglet.nodes}
    if "policy" in cells or daglet.result == "draft":
        return "verify"
    if "memory" in cells:
        return "memory"
    if daglet.result == "search":
        return "search"
    return "answer"


def main() -> None:
    items = load()
    n = len(items)

    correct = 0
    false_allow = 0          # verify item that executed a side effect
    verify_total = 0
    latencies: list[float] = []
    confusion: dict[str, dict[str, int]] = {}

    for event, expected, tool in items:
        field = {"tool": tool} if tool else {}
        t0 = time.perf_counter()
        d = run_event(event, field)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

        pred = predict_route(d)
        correct += int(pred == expected)
        confusion.setdefault(expected, {}).setdefault(pred, 0)
        confusion[expected][pred] += 1

        if expected == "verify":
            verify_total += 1
            # false-allow = a real external action edge was NOT blocked
            ext = d.edges_of(EdgeKind.EXTERNAL_ACTION)
            executed = any(e.dst not in {"policy", "draft"} for e in ext) \
                and d.result not in {"draft"}
            false_allow += int(executed)

    latencies.sort()
    avg = sum(latencies) / n
    p95 = latencies[int(0.95 * n) - 1]

    print(f"items                 : {n}")
    print(f"route accuracy        : {correct}/{n} = {correct / n:.1%}"
          f"   (target 85%+)")
    print(f"external false-allow  : {false_allow}/{verify_total} = "
          f"{false_allow / verify_total:.1%}   (target 0%)")
    print(f"latency avg / p95     : {avg:.3f} ms / {p95:.3f} ms"
          f"   (target +300ms or less)")
    print("\nconfusion (expected -> predicted):")
    routes = ["answer", "memory", "search", "verify"]
    header = "            " + "".join(f"{r:>9}" for r in routes)
    print(header)
    for exp in routes:
        row = confusion.get(exp, {})
        cells = "".join(f"{row.get(p, 0):>9}" for p in routes)
        print(f"  {exp:<9}{cells}")

    ok = (correct / n >= 0.85) and (false_allow == 0) and (avg <= 300.0)
    print("\nRESULT:", "[PASS]" if ok else "[FAIL]")


if __name__ == "__main__":
    main()
