"""adversary: 카드 verify[0..17]을 독립 재계산하고, 각 항목이 '실패 가능한가'를 분류한다.

분류
  TAUTOLOGY : 카드의 정의/대수만으로 항상 참 (반증 불가)
  ARITHMETIC: 하드코딩 상수의 수치 재평가 (외부 자료 대조 아님)
  EXTERNAL  : 카드 밖의 독립 사실을 실제로 대조
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
a, d, k, M = sp.symbols("a d k M", positive=True)

MU = sp.Float("2.1777584234", 30)


def poisson_tail(thr, lam):
    return 1 - sp.exp(-lam) * sum(lam**j / sp.factorial(j) for j in range(thr))


def main() -> int:
    rows = []

    def add(i, desc, ok, kind, detail=""):
        rows.append({"index": i, "desc": desc, "reproduced": bool(ok), "kind": kind,
                     "detail": detail})

    add(0, "a**(d-2)/a == a**(d-3)",
        sp.simplify(a**(d - 2) / a - a**(d - 3)) == 0, "TAUTOLOGY",
        "지수법칙. d, a 무관하게 항등")
    add(1, "lim_{d->3} a**(d-3) == 1",
        sp.limit(a**(d - 3), d, 3) == 1, "TAUTOLOGY",
        "a**0=1. '지수 0 => d=3'의 역방향 대입일 뿐, 'a 의존성 소멸 => d=3'을 시험하지 않는다")
    add(2, "(-1)**(k*(2k-k))*(-1) == (-1)**(k**2+1)",
        sp.simplify(sp.expand((-1)**(k * (2 * k - k)) * (-1) - (-1)**(k**2 + 1))) == 0,
        "TAUTOLOGY", "k*(2k-k)=k^2 재서술")
    add(3, "k=2: (-1)**4*(-1)+1 == 0", ((-1)**(2 * (4 - 2)) * (-1) + 1) == 0, "ARITHMETIC", "")
    add(4, "k=3: (-1)**9*(-1)-1 == 0", ((-1)**(3 * (6 - 3)) * (-1) - 1) == 0, "ARITHMETIC", "")
    add(5, "lim_{k->2} (2k-2)/2 == 1", sp.limit((2 * k - 2) / 2, k, 2) == 1, "TAUTOLOGY", "")
    add(6, "(4*3/2)/2 - 3 == 0", abs((4 * 3 / 2) / 2 - 3) < 1e-12, "ARITHMETIC", "")
    add(7, "(6*5/2)/2 > 5", (6 * 5 / 2) / 2 > 5, "ARITHMETIC", "")
    add(8, "(2M+1)-(M+2)+1 == M",
        sp.simplify((2 * M + 1) - (M + 2) + 1 - M) == 0, "TAUTOLOGY", "선형 항등식")
    add(9, "M=0: (1)-(2)+1 == 0", ((2 * 0 + 1) - (0 + 2) + 1) == 0, "ARITHMETIC", "")
    add(10, "(2k-1)+1 == 2k", sp.simplify((2 * k - 1) + 1 - 2 * k) == 0, "TAUTOLOGY",
        "장부 항등식; n_time=1을 이미 대입한 뒤의 재서술")
    for i, (thr, b, target) in enumerate(
        [(4, 1, "0.1762918862"), (4, 5, "0.9946550103"), (2, 1, "0.6399752045"),
         (2, 2, "0.9312576369"), (2, 3, "0.9890448472"), (2, 4, "0.9984000309")], start=11):
        val = poisson_tail(thr, b * MU)
        add(i, f"P(F_{b}>= {thr}) == {target}", abs(float(val) - float(target)) < 1e-9,
            "ARITHMETIC", "상속 상수 D=3.1777584234의 닫힌 형태 재평가")
    c = 4 * sp.Rational(4, 10)**2
    lam = ((2 + c) + sp.sqrt((2 + c)**2 - 4)) / 2
    slope = sp.log(lam) / sp.Rational(4, 10)
    add(17, "(T/CFL)*ln(lambda) == 1.9501765989",
        abs(float(slope) - 1.9501765989) < 1e-9, "ARITHMETIC",
        "von Neumann 증폭인자의 N->infty 점근값. 유한 격자 최소제곱 기울기와 다르다")

    summary = {
        "all_reproduced": all(r["reproduced"] for r in rows),
        "counts": {kind: sum(1 for r in rows if r["kind"] == kind)
                   for kind in ("TAUTOLOGY", "ARITHMETIC", "EXTERNAL")},
        "falsifiable_against_external_data": sum(1 for r in rows if r["kind"] == "EXTERNAL"),
    }
    out = {"summary": summary, "rows": rows}
    (HERE / "recheck_verify18.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    bad = [r for r in rows if not r["reproduced"]]
    print("not reproduced:", bad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
