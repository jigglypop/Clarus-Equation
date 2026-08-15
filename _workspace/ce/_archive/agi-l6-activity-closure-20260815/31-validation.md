# 31-validation — L6 활동 한 스텝 (등록 쌍)

Status: COMPLETE

기계 검사 문자열 PASS는 이론 지위가 아니다. 이 파일은 실행한 명령과 원문 출력만 기록한다. $L6$-$E1$--$E3$를 정리로 승격하지 않는다. $L6$-$H1$을 새 헐 정리로 올리지 않는다. 닫힘·유도됨·제1원리·자율·zebrafish·$L7$·AGI를 쓰지 않는다.

## 1. 명령

저장소 루트에서:

```
python -m pytest tests/test_universe_life_kernel.py tests/test_l3_nonlinear_las.py tests/test_l3_ne2_open_set.py tests/test_l4_weighted_routing.py tests/test_l5_role_split.py tests/test_l6_activity_closure.py -q
```

전체 repo pytest는 돌리지 않았다. dimensionless checker에 식을 넣지 않았다.

## 2. 원문 출력

```
...............................................................          [100%]
63 passed in 5.72s
```

exit code 0.

## 3. 회귀 / 범위

| 잠금 | 기계 결과 | 이론 지위 |
|---|---|---|
| `drive=1` 성장 괄호 | 기존 `source_hybrid_step`과 같음 | 구성. $L0$ 환원 |
| 등록 쌍 (L6.1) | $P_{\star},P_{\circ}\in U_0\times\{3/4\}$, $\sigma=1$ | 구성. 기하 인용 |
| 한 스텝 $(m',b')$ | 잠금 분수와 일치, $\Delta m\neq 0$, $\Delta b\neq 0$ | $L6$-$E1$ 구성 잠금 |
| 비트 예측기 | $\sigma=1$에서 한 값. 두 참 다음 상태를 동시 불일치 | $L6$-$E2$ 구성 잠금 |
| 연산자 불일치 | $\{P\mapsto(m',b')\}\neq\{\sigma\mapsto$ 한 쌍$\}$ | $L6$-$E3$ 구성 잠금. 유한 $\{P_{\star},P_{\circ}\}$ |
| $U_0$ 소속 | 두 점 모두 열린 상자 내부 | 인용. $O$-$E1$ 적용 자리. 새 헐 아님. $T=32$ 궤적 없음 |
| 선행 L3--L5 | `test_l3_*.py`, `test_l4_*.py`, `test_l5_*.py` 유지 | 선행 상자·헐을 재유도하지 않음 |
| 기존 커널 | `test_universe_life_kernel.py` 회귀 유지 | `drive` 기본 $1$ |

회귀: 이 여섯 파일 묶음 밖 테스트는 실행하지 않음. `docs/7_AGI/`는 변경하지 않았다. 기계 통과는 $L6$-$E1$--$E3$ 정리의 대체 증명이 아니다.
