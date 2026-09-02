"""adversary: (A) fine∩coarse 위 holonomy 이중정의 정량화 + 두 갈래 위상 계산,
(B) K2 (2,2) vs (3,1) leapfrog 증폭 기울기 독립 실행 (카드 규약 그대로).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent))

from examples.physics.causal_face_simplicity import composition_faces  # noqa: E402
import check_carrier as CC  # noqa: E402
from rerun_k1_homology import betti1_independent, undirected  # noqa: E402


# ---------------------------------------------------------------- (A)
def holonomy_consistency():
    rng = np.random.default_rng(CC.SEED)
    rows = []
    for n in CC.SIZES:
        fine = CC.split_merge_dag(n, rng)
        coarse = CC.two_step_coarse(fine)
        faces = composition_faces(fine, coarse)
        # check_carrier 와 같은 순서로 rng 를 소비하되 edge_hol 을 직접 본다
        edge_hol = {e: CC.random_su2(rng) for e in sorted(fine)}
        by_coarse = {}
        for f in faces:
            by_coarse.setdefault((f.source, f.target), []).append(f.middle)
        gaps = []
        for (u, v), mids in by_coarse.items():
            if (u, v) not in fine:
                continue
            m0 = sorted(mids)[0]
            u_coarse = edge_hol[(u, m0)] @ edge_hol[(m0, v)]
            gaps.append(float(np.linalg.norm(edge_hol[(u, v)] - u_coarse)))
        # rng 소비를 check_carrier 와 맞추기 위해 남은 호출을 흉내낸다
        CC.split_only_tree(n, rng)

        V = len({x for e in (fine | coarse) for x in e})
        b1_identified, comps = betti1_independent(fine | coarse, {x for e in (fine | coarse) for x in e})
        b1_distinct = (len(fine) + len(coarse)) - V + comps
        rows.append({
            "n": n,
            "edges_in_fine_and_coarse": len(fine & coarse),
            "double_defined_with_face": len(gaps),
            "min_gap": min(gaps) if gaps else None,
            "median_gap": float(np.median(gaps)) if gaps else None,
            "max_gap": max(gaps) if gaps else None,
            "b1_identified_cells": b1_identified,
            "b1_distinct_fine_coarse_cells": b1_distinct,
            "faces": len(faces),
        })
    return rows


# ---------------------------------------------------------------- (B)
def d2_operator(shape, axis, h):
    """주기 격자의 2차 중심차분 (축 하나)."""
    return lambda u: (np.roll(u, -1, axis=axis) - 2.0 * u + np.roll(u, 1, axis=axis)) / (h * h)


def amplification(N, signature, seed=20260902):
    h = 1.0 / N
    dt = 0.4 * h
    steps = int(round(1.0 / dt))
    rng = np.random.default_rng(seed)
    u0 = rng.normal(size=(N, N, N))
    n0 = float(np.linalg.norm(u0))
    signs = (1.0, 1.0, 1.0) if signature == "31" else (1.0, 1.0, -1.0)
    prev = u0.copy()
    cur = u0.copy()
    best = 1.0
    for _ in range(steps):
        lap = np.zeros_like(cur)
        for ax, s in enumerate(signs):
            lap += s * (np.roll(cur, -1, axis=ax) - 2.0 * cur + np.roll(cur, 1, axis=ax)) / (h * h)
        nxt = 2.0 * cur - prev + dt * dt * lap
        prev, cur = cur, nxt
        val = float(np.linalg.norm(cur)) / n0
        if not math.isfinite(val):
            best = float("inf")
            break
        best = max(best, val)
    return best


def k2():
    Ns = (8, 16, 32)
    out = {}
    for sig in ("22", "31"):
        amps = [amplification(N, sig) for N in Ns]
        logs = [math.log(a) if math.isfinite(a) and a > 0 else float("inf") for a in amps]
        if all(math.isfinite(x) for x in logs):
            slope = float(np.polyfit(np.array(Ns, dtype=float), np.array(logs), 1)[0])
        else:
            slope = float("inf")
        out[sig] = {"N": list(Ns), "A": amps, "lnA": logs, "slope": slope}
    c = 4 * 0.4 ** 2
    lam = ((2 + c) + math.sqrt((2 + c) ** 2 - 4)) / 2
    out["closed_form_slope"] = math.log(lam) / 0.4
    out["card_prereg_slope_22"] = 1.9502
    out["card_window_22"] = [1.90, 2.00]
    out["in_window_22"] = 1.90 <= out["22"]["slope"] <= 2.00
    out["in_window_31"] = abs(out["31"]["slope"]) <= 0.02
    out["amp_31_N32"] = out["31"]["A"][-1]
    out["amp_31_le_3"] = out["31"]["A"][-1] <= 3
    return out


def main() -> int:
    res = {"holonomy_consistency": holonomy_consistency(), "K2": k2()}
    (HERE / "holonomy_and_k2.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
