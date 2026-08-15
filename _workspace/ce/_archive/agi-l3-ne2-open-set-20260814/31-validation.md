# 31-validation — $N$-$E2$ 열린 집합 점유 갈림

Status: COMPLETE

기계 검사 문자열 PASS는 이론 지위가 아니다. 이 파일은 실행한 명령과 원문 출력만 기록한다. $R_0$ 전체 갈림이나 횟수 갈림을 정리로 올리지 않는다. 닫힘·유도됨·자율·AGI를 쓰지 않는다.

## 1. 명령

저장소 루트에서:

```
python -m pytest tests/test_l3_ne2_open_set.py tests/test_l3_nonlinear_las.py tests/test_universe_life_kernel.py -q
```

전체 repo pytest는 돌리지 않았다. dimensionless checker에 식을 넣지 않았다.

## 2. 원문 출력

```
.................................                                        [100%]
33 passed in 4.88s
```

exit code 0.

## 3. 회귀 / 범위

| 잠금 | 기계 결과 | 이론 지위 |
|---|---|---|
| $U_0=\operatorname{int}(B_c)$ 기하 | $B_c\subset R_0$, 중심 $(1/2,49/99)$, $U_1$ 선형 $1/3$ | 사전등록 기하의 구성 검사. 새 상자 아님 |
| $U_0$ 1보 $\widetilde m$ | $q=1/4$ 하한 $1098217/1425600>3/4$, $q=3/4$도 분열 | $O$-$E1$ 1보 분기의 재현. 대체 증명 아님 |
| $q=1/4$ $T=32$ 헐 | $m\le 48924156634417547/125000000000000000<2/5$, $R_0$과 서로소 | $U_0$ 점유 거짓의 구성 잠금. $R_0$ 전체 아님 |
| $q=3/4$ $T=32$ 헐 | $2/5\le m\le 3/5$, $4/9\le b\le 6/11$ | $U_0$ 점유 참의 구성 잠금 |
| 분열 횟수 | $32$ 대 $32$, `count_split is False` | 횟수 갈림은 거짓. 횟수 정리 아님 |
| 선행 L3 상자 | `tests/test_l3_nonlinear_las.py` 유지 | 선행 $5\times5$ 증인을 정리로 올리지 않음 |
| 기존 커널 | `tests/test_universe_life_kernel.py` 회귀 유지 | 커널 사상 변경 없음 |

회귀: 이 세 파일 묶음 밖 테스트는 실행하지 않음. 커널 모듈·공개 export·`docs/7_AGI/`는 변경하지 않았다. 기계 통과는 $O$-$E1$ 정리의 대체 증명이 아니다.
