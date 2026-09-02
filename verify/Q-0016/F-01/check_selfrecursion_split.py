"""Q-0016 F-01 theory kill K_T (pre-registered 2026-09-03, card revision 2).

Statistic (defined on rendered LABELS, so any implementation of the CE self-recursion renderer G[p_n]
can be scored without exposing its internals):

    R := sqrt( mean_{z: k_z>=2}  || sum_{c in ch(z)} (X_c - X_z) ||^2 / k_z )
         / sqrt( mean_{c: non-root} || X_c - X_{p(c)} ||^2 )

over all split events z (k_z >= 2 children) of an ensemble of rendered trees.  X_v in R^16 is the tetrad
label of cell v.  Under increments with sibling correlation -s/(k-1) and unit marginal variance
R = sqrt(1 - s) exactly in expectation: complete conservation (this card) gives R = 0, F-02 i.i.d.
increments give R = 1.

WINDOW (frozen now, before any renderer exists):  R in [0, 0.1]   (s >= 0.99).
Kill: the CE self-recursion renderer, once implemented (ladder step 2), yields R > 0.1 on >= 1000 split
events at seed 20260902.  F-02's value R = 1 is outside; the K_A2-allowed band s in [0.71, 1.00] would
give R up to 0.54, also outside -- this theory kill is therefore STRICTER than the numeric kills.

The renderer hook `render_labels(seed, n_trees)` is deliberately NOT implemented here: writing it is the
content of ladder step 2 (derive the increment structure from p_{n+1} = G[p_n]).  Running the script
without it records status 'not_implemented' and no verdict.  `--selftest` validates the statistic on the
two hand-written samplers of this card (split -> 0, F-02 heritable -> ~1).

Usage: python verify/Q-0016/F-01/check_selfrecursion_split.py [--selftest]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
F02 = ROOT / "verify" / "Q-0008" / "F-02"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(F02))
sys.path.insert(0, str(HERE))

from driver_numbers import qspine_block, tree_arrays  # noqa: E402
from predict_split_kernel import children_of, split_labels  # noqa: E402

SEED = 20260902
MIN_SPLIT_EVENTS = 1000
WINDOW = (0.0, 0.1)
PREREGISTERED = {"R_split_card": 0.0, "R_f02_iid": 1.0}


def sibling_sum_ratio(parent: list[int], labels: np.ndarray) -> tuple[float, float, int]:
    """Return (sum_z |sum_c (X_c - X_z)|^2 / k_z, sum_c |X_c - X_p(c)|^2, number of split events)."""
    lab = np.asarray(labels, dtype=float).reshape(len(parent), -1)
    num = 0.0
    den = 0.0
    events = 0
    for z, ch in enumerate(children_of(parent)):
        if not ch:
            continue
        inc = lab[ch] - lab[z]
        den += float(np.sum(inc * inc))
        if len(ch) >= 2:
            tot = inc.sum(axis=0)
            num += float(np.dot(tot, tot)) / len(ch)
            events += 1
    return num, den, events


def score(trees: list[tuple[list[int], np.ndarray]]) -> dict:
    num = den = 0.0
    events = 0
    n_inc = 0
    for parent, labels in trees:
        a, b, e = sibling_sum_ratio(parent, labels)
        num += a
        den += b
        events += e
        n_inc += len(parent) - 1
    R = math.sqrt((num / events) / (den / n_inc)) if events and den > 0 else math.nan
    return {"R": R, "split_events": events, "increments": n_inc}


def render_labels(seed: int, n_trees: int) -> list[tuple[list[int], np.ndarray]]:
    """Ladder step 2 must replace this with the actual CE self-recursion renderer G[p_n]
    (parent array + 16-component label per cell).  Not implemented at card time on purpose."""
    raise NotImplementedError("CE self-recursion renderer not implemented (ladder step 2)")


def heritable_labels(parent: list[int], xi: np.ndarray) -> np.ndarray:
    order, _, _, _ = tree_arrays(parent)
    labels = np.zeros_like(xi)
    for v in order:
        p = parent[v]
        labels[v] = xi[v] + (labels[p] if p >= 0 else 0.0)
    return labels


def selftest(seed: int = SEED, n_trees: int = 400, depth: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    trees_split, trees_f02 = [], []
    for _ in range(n_trees):
        parent = qspine_block(depth, rng)
        xi = rng.normal(size=(len(parent), 16))
        trees_split.append((parent, split_labels(parent, xi)))
        trees_f02.append((parent, heritable_labels(parent, xi)))
    return {"split_sampler": score(trees_split), "f02_heritable_sampler": score(trees_f02)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        out = selftest()
        ok = out["split_sampler"]["R"] < 1e-12 and abs(out["f02_heritable_sampler"]["R"] - 1.0) < 0.1
        print(json.dumps({"selftest": "ok" if ok else "FAIL", **out}))
        return 0 if ok else 1
    result = {"card": "F-01", "question": "Q-0016", "kill": "K_T", "seed": SEED, "window": WINDOW,
              "preregistered": PREREGISTERED, "min_split_events": MIN_SPLIT_EVENTS}
    try:
        trees = render_labels(SEED, 2000)
    except NotImplementedError as error:
        result["status"] = "not_implemented"
        result["reason"] = str(error)
        result["verdict"] = None
        (HERE / "result_KT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return 0
    sc = score(trees)
    result.update(sc)
    if sc["split_events"] < MIN_SPLIT_EVENTS:
        result["status"] = "insufficient_events"
        result["verdict"] = None
    else:
        result["status"] = "done"
        result["verdict"] = "KILL" if not (WINDOW[0] <= sc["R"] <= WINDOW[1]) else "survive"
    (HERE / "result_KT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
