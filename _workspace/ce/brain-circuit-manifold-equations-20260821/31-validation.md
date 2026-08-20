# 검증

Status: COMPLETE

## A6 deterministic math witness

명령:

```powershell
.codex\hooks\python.cmd python _workspace\ce\brain-circuit-manifold-equations-20260821\artifacts\a6_math_witness.py
```

결과: PASS.

핵심 수치:

| 증인 | 결과 |
|---|---|
| frozen-$p$ Jacobian | `0.6536849812854421` |
| true activity-dependent Jacobian | `0.871579975047256` |
| passive principal stretches | `[0.5, 2.0]` |
| passive log-volume change | `0.0` |
| rank-loss metric rank | `1` |
| nonnormal $E^*(e_1)$ | `0.0181` |
| analytic / finite-difference $\dot g$ | `0.2797950506157409` / `0.2797950505034619` |
| analytic / finite-difference $\dot E$ | `-0.148720999405116` / `-0.14872099945995032` |

## 무차원 checker

명령:

```powershell
.codex\hooks\python.cmd pytest tests\test_dimensionless.py -q
```

결과: `17 passed in 0.44s`.

추가 실행:

```powershell
.codex\hooks\python.cmd python reality_stone\python\reality_stone\clarus\dimensionless.py
```

결과: exit code 0.

## 판정 범위

위 결과는 식의 미분·선형대수·차원 정합만 검증한다. nonlinear global reachability, 실제 회로 형성, AGI 기능, 물리 cortical folding은 검증하지 않는다.
