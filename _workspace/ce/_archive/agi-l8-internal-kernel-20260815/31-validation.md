# 31-validation — L8 내부 커널 (등록 쌍 $S$)

Status: COMPLETE

기계 검사 문자열 PASS는 이론 지위가 아니다. 이 파일은 실행한 명령과 원문 출력만 기록한다. $L8$-$E1$--$E3$와 $L8$-$H1$을 정리로 승격하지 않는다. 「비트값 $K$면 충분하다」를 산 주장으로 쓰지 않는다. 닫힘·유도됨·제1원리·자율·BrainRuntime·셋째 큐브·AGI를 쓰지 않는다.

## 1. 명령

저장소 루트에서:

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py tests/test_l5_role_split.py tests/test_l6_activity_closure.py tests/test_l7_region_loop.py tests/test_l8_internal_kernel.py -q
```

전체 repo pytest는 돌리지 않았다. dimensionless checker에 식을 넣지 않았다.

## 2. 원문 출력

```
........................................................................ [ 85%]
............                                                             [100%]
84 passed in 5.81s
```

exit code 0.

## 3. 회귀 / 범위

| 잠금 | 기계 결과 | 이론 지위 |
|---|---|---|
| 등록 $S$ | $H_{\star}$, $H_{\circ}$가 L6 점과 $e^{(2)}$를 씀 | 구성 |
| 구동 | $W=I$, $I=1$에서 $(u^{\mathrm{S}},u^{\mathrm{A}})=(0,1)$ | 구성. L5 곱과 L7 곱이 $S$에서 같음 |
| $K=\Phi$ | `internal_kernel`이 독립 한 스텝 조립과 같음 | $L8$-$E1$ 구성 잠금. 새 맵 아님 |
| 센서 $u=0$ | $m'=0$, $1-\lambda(1-49/99)=-26/99<0$ | 한 스텝 소멸. $F^{32}$ 없음 |
| 작용 $u=1$ | $(m',b')$가 L6 잠금 분수 | 인용. 재풀이 없음 |
| $o^{\mathrm{A}}$ | $S$에서 둘 다 $1$ | $L8$-$E2$ 구성 잠금 |
| 작용 차 | $\Delta m=-1487/950400$, $\Delta b=1/297$ | $L8$-$E2$ 구성 잠금. L6 인용 |
| 맵 불일치 | $K\neq o^{\mathrm{A}}$ as maps from $S$ | $L8$-$E3$ 구성 잠금. 유한 쌍 |
| 공역 | 비트 $\neq$ `HostTuple` | $L8$-$H1$ 구성 잠금. 형 검사 |
| 선행 L3--L7 | `test_l3_*.py` … `test_l7_*.py` 유지 | 선행을 재유도하지 않음 |
| 기존 커널 | `test_universe_life_kernel.py` 회귀 유지 | `drive` 기본 $1$ |

회귀: 이 여덟 파일 묶음 밖 테스트는 실행하지 않음. `docs/7_AGI/`는 변경하지 않았다. 기계 통과는 $L8$-$E1$--$E3$, $L8$-$H1$ 정리의 대체 증명이 아니다.
