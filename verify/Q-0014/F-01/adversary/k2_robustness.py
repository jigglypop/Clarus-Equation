"""adversary: K2 기울기의 구현 민감도와 유한 N 편향.

카드 사전등록: slope_22 = 1.9502 +- 0.05 (창 [1.90, 2.00]), 격자 N in {8,16,32}.
닫힌 형태 (T/CFL)*ln(lambda) = 1.9501765989 는 N->infty 점근값이다.
무작위 초기자료에서 가장 불안정한 mode는 단일 Nyquist mode (0,0,N/2)이므로
ln A(N) = s*N - (3/2)ln N + c 가 기대되고, 카드의 유한 격자에서 최소제곱
기울기는 s 보다 체계적으로 낮다. 그 편향 크기를 잰다.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def amplification(N, signature, seed=20260902, include_n0=True, steps_mode="round"):
    h = 1.0 / N
    dt = 0.4 * h
    raw = 1.0 / dt
    steps = int(round(raw)) if steps_mode == "round" else int(math.ceil(raw))
    rng = np.random.default_rng(seed)
    u0 = rng.normal(size=(N, N, N))
    n0 = float(np.linalg.norm(u0))
    signs = (1.0, 1.0, 1.0) if signature == "31" else (1.0, 1.0, -1.0)
    prev, cur = u0.copy(), u0.copy()
    best = 1.0 if include_n0 else 0.0
    for _ in range(steps):
        lap = np.zeros_like(cur)
        for ax, s in enumerate(signs):
            lap += s * (np.roll(cur, -1, axis=ax) - 2.0 * cur + np.roll(cur, 1, axis=ax)) / (h * h)
        cur, prev = 2.0 * cur - prev + dt * dt * lap, cur
        best = max(best, float(np.linalg.norm(cur)) / n0)
    return best, steps


def lsq_slope(Ns, ys):
    return float(np.polyfit(np.asarray(Ns, dtype=float), np.asarray(ys, dtype=float), 1)[0])


def main() -> int:
    c = 4 * 0.4 ** 2
    lam = ((2 + c) + math.sqrt((2 + c) ** 2 - 4)) / 2
    asymptote = math.log(lam) / 0.4

    variants = {}
    for name, kw in {
        "card_literal": {},
        "exclude_n0": {"include_n0": False},
        "steps_ceil": {"steps_mode": "ceil"},
        "seed_shift_1": {"seed": 20260903},
        "seed_shift_2": {"seed": 20260901},
    }.items():
        Ns = (8, 16, 32)
        amps = [amplification(N, "22", **kw)[0] for N in Ns]
        ys = [math.log(a) for a in amps]
        s = lsq_slope(Ns, ys)
        variants[name] = {"slope_22": s, "in_window": 1.90 <= s <= 2.00, "lnA": ys}

    # 유한 N 편향: 큰 N 까지 확장하고 lnA = s*N + b*lnN + c 로 적합
    big_Ns = (8, 16, 32, 64, 96)
    big = []
    for N in big_Ns:
        a, st = amplification(N, "22")
        big.append({"N": N, "steps": st, "A": a, "lnA": math.log(a)})
    X = np.array([[r["N"], math.log(r["N"]), 1.0] for r in big])
    y = np.array([r["lnA"] for r in big])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    # 카드 격자만으로 낸 기울기와 점근값 차이
    slope_card_grid = lsq_slope([r["N"] for r in big[:3]], [r["lnA"] for r in big[:3]])
    slope_big_grid = lsq_slope([r["N"] for r in big], [r["lnA"] for r in big])

    out = {
        "closed_form_asymptote": asymptote,
        "card_prereg": {"value": 1.9502, "uncertainty": 0.05, "window": [1.90, 2.00],
                        "grid": [8, 16, 32]},
        "variants_on_card_grid": variants,
        "extended": big,
        "fit_lnA_eq_sN_plus_b_lnN_plus_c": {"s": float(coef[0]), "b": float(coef[1]),
                                            "c": float(coef[2])},
        "slope_on_card_grid": slope_card_grid,
        "slope_on_extended_grid": slope_big_grid,
        "bias_card_grid_vs_asymptote": slope_card_grid - asymptote,
    }
    (HERE / "k2_robustness.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "extended"}, ensure_ascii=False, indent=2))
    print("extended:", [(r["N"], round(r["lnA"], 4)) for r in big])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
