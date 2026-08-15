# 31-validation — universe-simulator life-climb

Status: COMPLETE

기계 검사 문자열 PASS는 이론 지위가 아니다. 이 파일은 실행한 명령과 원문 출력만 기록한다. P-E1·P-H1을 정리로 올리지 않는다.

## 1. 명령

저장소 루트에서:

```
python -m pytest tests/test_universe_life_kernel.py -q
```

전체 repo pytest는 돌리지 않았다. dimensionless checker에 식을 넣지 않았다. G-DIMENSIONLESS는 계약 기호 `m,b,q,E,r,λ,ρ,δ,s,μ,η,θ_D,K,κ,ν`에 이미 걸려 있다.

## 2. 원문 출력

```
.........................                                                [100%]
25 passed in 9.12s
```

exit code 0.

## 3. 회귀 / 범위

| 게이트 | 기계 결과 | 이론 지위 |
|---|---|---|
| G-HOST | 8꼭짓점 + 6고정점, `T=8`, 커널 vs 로컬 `F_0` Fraction 항등 | P-C3 구성. 생물 주장 아님 |
| G-COUPLE `κ=0` | 같은 `(m,b)` 씨드에서 `q`가 `(m,b)`를 바꾸지 않음 | `F_0` 채널 없음. P-C2 인용과 정합 |
| G-COUPLE `κ=1/4` | `q=1/4`와 `q=3/4`의 `(m,b)`가 갈림 | 상자 구성. P-E1 미완성 |
| G-COUPLE `{1/2, 1}` | `I_r ∪ {0}` 밖이라 거절. `Δ_r(q=1/4)`는 `-95/64`, `-359/16` | 죽이는 시험. `I_r` 원소 아님 |
| 소멸 면적 | 출처 공식 `1/10 ≥ 1/20`. `q=1/2`에서 `r` 명목 | 새 면적 정리 아님 |
| G-DIMENSIONLESS | checker 미변경 | 계약 기호만 |

회귀: 이 파일 묶음 밖 테스트는 실행하지 않음. 공개 export는 기존 lazy optional 패턴을 유지했다.
