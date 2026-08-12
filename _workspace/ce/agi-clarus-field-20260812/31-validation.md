# 31-validation — agi-clarus-field-20260812

Status: COMPLETE

## 결과 요약

| 범위 | 명령 요약 | 원래 결과 | 회귀 |
|---|---|---|---|
| focused | `pytest tests/test_clarus_field.py -q -p no:cacheprovider` | `17 passed in 3.35s` | 없음 |
| dimensionless + field | `pytest tests/test_dimensionless.py tests/test_clarus_field.py ...` | `28 passed in 3.47s` | 없음 |
| CE core slice | bootstrap/dimensionless/layer-A/bridge/field | `71 passed, 2 warnings in 4.19s` | 없음 |
| runtime/public slice | runtime-contracts/reality-bridge/field | `34 passed, 2 warnings in 3.17s` | 없음 |
| local-cloud compatibility | V10 kernel/learnable-small-gain/V13/field | `72 passed in 9.09s` | 없음 |
| style | `ruff check` on changed Python | `All checks passed!` | 없음 |
| deterministic demo | `.venv/Scripts/python.exe examples/agi/clarus_field_demo.py` | tick 32, max memory norm `0.9999998953`, field norm `9.4243143777`, mean occupancy `(0.125,0.765625,0.109375)` | smoke only |

PyTorch sparse invariant/beta 경고 2건은 기존 `runtime.py` 경로의 경고다. 첫 runtime slice 시도는 시스템 `%TEMP%/pytest-of-dongh` 접근 거부로 `29 passed, 4 errors`였고, 작업공간 안의 검증된 `--basetemp`로 같은 테스트를 재실행해 `34 passed`를 얻었다. 코드 실패와 환경 setup 실패를 분리한다.

## 수학·수치 재검산

다음 기존 독립 스크립트 6개를 `.venv/Scripts/python.exe`로 재실행했고 모두 exit 0이었다.

1. `verify_cf1_cf2.py`: 동결 비트 항등, $s$ bound 포화, $\sup\lVert\phi\rVert_2=11.3031\le11.4286$, $phi\ge0$.
2. `verify_cf3.py`: A-E2R형 공통 write fixture에서 두 초기조건 정확 합류, 40,000 tick 점유율 차 $2.031\times10^{-5}$.
3. `verify_cf4.py`: 스칼라 $a^*=0.0487077473$, 옛 3성분 $B$의 고정점 이탈 $0.5344$, 입력률 $0.049/0.120/0.300$에 활성률 $0.0488/0.1207/0.2994$ 추적.
4. `verify_cf5.py`: 명시한 공통 잡음 law fixture에서 interlacing 위반 0.
5. `route_a_candidates.py`: R-A1 toy 재현, baseline에는 미편입.
6. `route_b_corrections.py`: 스칼라 축소와 $D_{\text{eff}}a^*=0.154727$ 재현.

focused 회귀는 추가로 다음을 고정한다.

- 닫힌 gate 256 tick 후 memory bit pattern 완전 동일.
- one-node field step이 정확 damped solution과 상대오차 $10^{-13}$ 이내 일치.
- 500 tick 동안 field 비음수·CF-1 2-norm bound 유지.
- 공통 외생 write 10회에서 초기조건 차이가 정확히 $(1-0.75)^{10}$ 배.
- prediction-error gate가 부호 반전과 기준 scale 재조정에 불변.
- 외생 활성 비율 $0.1,0.3,0.7$을 phase occupancy가 그대로 추적. 이는 $p^*$ 자기수렴의 **음성 대조**다.
- public certificate가 `p_star_self_convergence=False`, `v14_route_l_inherited=False`를 유지.

## CE 검증 하네스

- `bootstrap_solver.py`: 잔차 $2.08\times10^{-17}$, 기존 해 $0.0486466333$. 이 하네스는 다른 $D_{\text{eff}}=3.17776$ 원천을 쓰므로 field run의 $0.0487077$과 동일 수치라고 합치지 않는다.
- `tests/scorecard.py`: 총 23, 채점 12, PASS 11, CAUTION 1, 외부입력 1, open test 1; 최대 주의 잔차는 $\Omega_bh^2=-1.80\sigma$.
- `tests/run_validation.py`: bootstrap PASS, scorecard CAUTION, 차원 7/7, 전체 `CAUTION`.
- `proof_completion_attempt.py`: exit 0. 후보 수치 일치는 원 스크립트의 bridge/후보 지위를 유지한다.

위 하네스 결과는 수치 무결성이지 클라루스장, 물리 이론 또는 AGI의 증명이 아니다. 저장소 전체 pytest는 실행하지 않았다. 선행 기록상 전체 suite에는 외부 fixture·정책 drift 실패가 남아 있으며, 이번 고립 변경의 관련 슬라이스와 구분한다.

Status: COMPLETE
