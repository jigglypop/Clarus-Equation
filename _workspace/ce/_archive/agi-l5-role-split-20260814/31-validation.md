# 31-validation — L5 역할 분리 (wash + $\sigma$)

Status: COMPLETE

기계 검사 문자열 PASS는 이론 지위가 아니다. 이 파일은 실행한 명령과 원문 출력만 기록한다. $L5$-$E1$--$E3$를 정리로 승격하지 않는다. $L5$-$H1$을 정리로 올리지 않는다. 닫힘·유도됨·제1원리·자율·Drosophila / C. elegans·$L6$·AGI를 쓰지 않는다.

## 1. 명령

저장소 루트에서:

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py tests/test_l5_role_split.py -q
```

전체 repo pytest는 돌리지 않았다. dimensionless checker에 식을 넣지 않았다.

## 2. 원문 출력

```
........................................................                 [100%]
56 passed in 5.96s
```

exit code 0.

## 3. 회귀 / 범위

| 잠금 | 기계 결과 | 이론 지위 |
|---|---|---|
| `drive=1` 성장 괄호 | 기존 `source_hybrid_step`과 같음 | 구성. $L0$ 환원 |
| $u=0$ 한 스텝 소멸 | $1-\lambda(1-b_{\mathrm{hi}})=-53/297<0$, $\widetilde m=0$ | 구성. 선행 산술 재현 |
| wash+$\sigma$ 판독 | $\tau^{(1)}\mapsto 1$, $\tau^{(2)}\mapsto 0$ | $L5$-$E1$ 구성 잠금. $u=1$은 선행 $U_0$ 헐 인용 |
| no-store wash 판독 | 두 과제에서 같음 | $L5$-$E2$ 구성 잠금. 공통 값은 미채점 |
| 연산자 불일치 | 역할 분리 맵 $\neq$ no-store 맵 | $L5$-$E3$ 구성 잠금. 유한 $\{\tau^{(1)},\tau^{(2)}\}$ |
| no-wash 둘째 창 | 채점하지 않음 | unfinished. $L5$-$H1$ 정리가 아님 |
| 선행 L3--L4 | `test_l3_*.py`, `test_l4_weighted_routing.py` 유지 | 선행 상자·헐을 재유도하지 않음 |
| 기존 커널 | `test_universe_life_kernel.py` 회귀 유지 | `drive` 기본 $1$ |

회귀: 이 다섯 파일 묶음 밖 테스트는 실행하지 않음. `docs/7_AGI/`는 변경하지 않았다. 기계 통과는 $L5$-$E1$--$E3$ 정리의 대체 증명이 아니다.
