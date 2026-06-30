"""Diffusion routing A/B vs the walk baseline (spec section 5).

    python -m bench.diffusion_route_ab

Honest escalation attempt: does the event-conditioned diffusion prior pi
route as accurately as the strict-priority cell walk? Both run on the same
100-question dataset. We report route accuracy for each and the safety
floor (a flagged action must still route to verify). No claim is made beyond
the measured numbers.
"""

from __future__ import annotations

from server.scheduler import run_event
from server.diffusion_scheduler import (
    run_event_diffusion, orchestration_prior, diffusion_route, event_source,
)
from server.cells.salience_cell import SalienceCell
from .dataset import load
from .run import predict_route

_SAL = SalienceCell()


def main() -> None:
    items = load()
    n = len(items)

    walk_correct = 0
    diff_correct = 0
    diff_false_verify = 0   # action that the diffusion router failed to verify
    mismatch_examples: list[tuple[str, str, str, str]] = []

    for event, expected, tool in items:
        field = {"tool": tool} if tool else {}

        # walk baseline
        d_walk = run_event(event, dict(field))
        walk_pred = predict_route(d_walk)
        walk_correct += int(walk_pred == expected)

        # diffusion router: seed field via the same salience pre-pass, then
        # decide the route from the diffusion prior.
        dfield = dict(field)
        _SAL.fire(event, dfield)
        pi = orchestration_prior(event, dfield)
        diff_pred = diffusion_route(pi, dfield)
        diff_correct += int(diff_pred == expected)

        if expected == "verify" and diff_pred != "verify":
            diff_false_verify += 1
        if diff_pred != expected and len(mismatch_examples) < 8:
            mismatch_examples.append((expected, diff_pred, walk_pred, event[:34]))

    print(f"items                : {n}")
    print(f"walk route accuracy  : {walk_correct}/{n} = {walk_correct / n:.1%}")
    print(f"diff route accuracy  : {diff_correct}/{n} = {diff_correct / n:.1%}")
    print(f"diff missed-verify   : {diff_false_verify}  (MUST be 0 — safety floor)")
    delta = (diff_correct - walk_correct) / n
    print(f"delta (diff - walk)  : {delta:+.1%}")

    if mismatch_examples:
        print("\ndiffusion mismatches (expected | diff | walk | event):")
        for exp, dp, wp, ev in mismatch_examples:
            print(f"  {exp:<7} | {dp:<7} | {wp:<7} | {ev}")

    print("\nNOTE: parity (delta ~ 0) means diffusion recovers the walk's"
          " routing\n      via a principled field-diffusion mechanism — NOT"
          " that it is better.")
    verdict = "SAFE" if diff_false_verify == 0 else "UNSAFE"
    print("SAFETY FLOOR:", f"[{verdict}]")


if __name__ == "__main__":
    main()
