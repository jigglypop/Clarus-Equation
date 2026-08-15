# 31-validation — L3 boxed map, nonlinear LAS and $T=32$

Status: COMPLETE

기계 검사 문자열 PASS는 이론 지위가 아니다. 이 파일은 실행한 명령과 원문 출력만 기록한다. N-E2를 정리로 올리지 않는다. 닫힘·유도됨·AGI를 쓰지 않는다.

## 1. 명령

저장소 루트에서:

```
python -m pytest tests/test_l3_nonlinear_las.py tests/test_universe_life_kernel.py -q
```

전체 repo pytest는 돌리지 않았다. dimensionless checker에 식을 넣지 않았다.

## 2. 원문 출력

```
..............................                                           [100%]
30 passed in 4.87s
```

exit code 0.

## 3. 회귀 / 범위

| 잠금 | 기계 결과 | 이론 지위 |
|---|---|---|
| $Z_-$ 고정점 | `source_hybrid_step`·커널 한 스텝이 $(7/18,7/16,1/4)$를 유지. $\widetilde m=7/9$ | N-D2 정의의 기계 잠금. N-E1 상자 정리의 대체 증명이 아님 |
| $Z_+$ 좌표 | $\mathbb Q(\sqrt{18601})$에서 $F_{1/4}(Z_+)=Z_+$, 판별식 $18601/256$ | N-D2 정확 좌표. 커널 Fraction 상태가 아님 |
| $Q_-$ 행합 | $\nu=1/200$, $(w,u)=(1,2)$, $\mathrm{lip}=16861/18000<1$ | N-E1 증인 숫자의 재현. $I_r$ 전칭 LAS 아님 |
| N-E3 면적 | $r(1/2)=9/2$, 면적 $1/10\ge 1/20$ | 산출. 새 면적 정리 아님 |
| $R_0$ $5\times5$ | $\|G\|=25$, 부동소수 $T=32$ 점유 $0$ 대 $25$ | 구성 검사·증인. N-E2는 미완성. 열린 집합 정리 아님 |
| 기존 커널 | `tests/test_universe_life_kernel.py` 회귀 유지 | 커널 사상 변경 없음 |

회귀: 이 두 파일 묶음 밖 테스트는 실행하지 않음. 커널 모듈·공개 export·`docs/7_AGI/`는 변경하지 않았다.
