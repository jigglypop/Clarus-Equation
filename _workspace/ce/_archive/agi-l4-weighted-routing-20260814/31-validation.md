# 31-validation — L4 두 채널 가중 라우팅

Status: COMPLETE

기계 검사 문자열 PASS는 이론 지위가 아니다. 이 파일은 실행한 명령과 원문 출력만 기록한다. $L4$-$E1$--$E3$를 정리로 승격하지 않는다. 닫힘·유도됨·제1원리·자율·C. elegans·$L5$·AGI를 쓰지 않는다.

## 1. 명령

저장소 루트에서:

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py -q
```

전체 repo pytest는 돌리지 않았다. dimensionless checker에 식을 넣지 않았다.

## 2. 원문 출력

```
............................................                             [100%]
44 passed in 5.20s
```

exit code 0.

## 3. 회귀 / 범위

| 잠금 | 기계 결과 | 이론 지위 |
|---|---|---|
| `drive=1` 성장 괄호 | 기존 `source_hybrid_step`과 같음 | 구성. $L0$ 환원 |
| $u=0$ 한 스텝 소멸 | $1-\lambda(1-b_{\mathrm{hi}})=-53/297<0$, $\widetilde m=0$ | 구성. $U_0$ 전칭 산술의 재현 |
| $W=I$ 점유 쌍 | $(1,0)$ 대 $(0,1)$ | $L4$-$E1$ 구성 잠금. $u=1$은 선행 $U_0$ 헐 인용 |
| $A_{\mathbf 1}$ 점유 쌍 | 두 플럭스에서 같음 | $L4$-$E2$ 구성 잠금. 공통 값은 미채점 |
| 연산자 불일치 | $I$ 맵 $\neq$ $A_{\mathbf 1}$ 맵 | $L4$-$E3$ 구성 잠금. 그래프 전칭 아님 |
| $I$와 스왑 | 둘 다 갈림 | killing test. $L4$-$H1$ 전칭이 아님 |
| 선행 L3 | `test_l3_nonlinear_las.py`, `test_l3_ne2_open_set.py` 유지 | 선행 상자·헐을 재유도하지 않음 |
| 기존 커널 | `test_universe_life_kernel.py` 회귀 유지 | `drive` 기본 $1$ |

회귀: 이 네 파일 묶음 밖 테스트는 실행하지 않음. `docs/7_AGI/`는 변경하지 않았다. 기계 통과는 $L4$-$E1$--$E3$ 정리의 대체 증명이 아니다.
