# 신경 계량 후보 전수 검증

Status: COMPLETE

## 실행 검증

과학 runtime은 workspace 외부 임시 cache의 `uv run --offline --no-project --with scipy python`으로 고정했다. V2.2 runner는 registry와 freeze를 먼저 검증한 뒤 102초에 완료됐고 다음 단계가 모두 성공했다: 11개 input byte 검증, session 준비, fit-only dynamics, uncertainty, deformation, condition field, graph, directed action, distribution, exact tuple audit, exclusive-create result write.

수학 fixture는 `PASS`다. time-varying innovation-to-endpoint transport는 정답 52와 잘못된 initial-product 33을 구별했고, chart quadratic error는 0이었다. `S13`은 SPD였고, extreme `S14` zero ridge는 `INELIGIBLE_SINGULAR`, positive ridge는 finite SPD였다. `S7-H,H=1`은 `INELIGIBLE_TAUTOLOGY`, 2차원 `S8/S9`의 $\lambda_G=0$은 모두 `INELIGIBLE_SINGULAR`였다. Gaussian NLPD는 log determinant를 포함해 inflated covariance가 zero residual에서 더 나쁜 점수였고, graph disconnect와 simple $W_2$ fixture도 통과했다.

사후 validator `artifacts/validate_e17_candidate_tournament.py`는 `PASS`다. 결과/동결/runner/validator/원장/fixture 해시, 27개 unique ID, 11개 input, 3개 animal, 표준 JSON finite number, boolean schema, 후보 coverage, field source와 생물학적 승자 금지를 재검증했다. 추가로 적격 uncertainty 2,992개, deformation 2,024개, decoder 44개, graph 168개, D1 88개, Wasserstein 22개의 primary score를 확인하고, 모든 outer 선택·animal 집계와 uncertainty scoreboard 4개를 raw 기록에서 독립 재계산했다.

Runner의 exclusive-create는 기존 경로 덮어쓰기를 막지만 crash-safe atomic rename은 아니다. 이 용어 경계는 `artifacts/e17-candidate-tournament-v2.2-wording-clarification.json`에 기록했다. 현재 결과는 실행 완료 뒤 별도 result SHA-256 lock과 validator를 모두 통과했으므로 이 표현 보정이 현재 artifact 무결성을 바꾸지 않는다.

## Tuple 완전성

| 계열 | cell | 예상 tuple | 관측 tuple | 누락 | 추가 |
|---|---:|---:|---:|---:|---:|
| uncertainty | 1,056 | 3,024 | 3,024 | 0 | 0 |
| deformation | 352 | 2,112 | 2,112 | 0 | 0 |
| condition decoder | 11 | 44 | 44 | 0 | 0 |
| condition field | 88 | 352 | 352 | 0 | 0 |
| graph | 88 | 264 | 264 | 0 | 0 |
| directed action | 22 | 88 | 88 | 0 | 0 |
| distribution | 11 | 22 | 22 | 0 | 0 |

총 5,906개 raw tuple key가 정확히 일치했다. `S7-H`는 `ELIGIBLE` 264, `INELIGIBLE_TAUTOLOGY` 88이다. `S8`과 `S9`는 각각 positive-ridge `ELIGIBLE` 132와 zero-ridge `INELIGIBLE_SINGULAR` 44이며, 독립 metric outer endpoint는 0이다. `G1/G2/G3a/G3b`는 raw tuple을 전부 계산했지만 공통 LOAO tuple이 없어 outer evaluation 0으로 `UNTESTABLE_UNDER_FROZEN_LOAO_INTERSECTION`이다. `D1`과 `P1/P2`는 각 3개 outer fold를 평가했다.

## Discovery 결과

동물별 동일가중 평균 NLPD에서 $H=1$은 persistence baseline이 가장 낮았다. $H=5,15,30$에서는 `S2`가 각각 1.539538, 2.855627, 3.520984로 평균 1위였지만 `S3/S4-H`와 차이가 작고 동물별 방향이 일치하지 않았다. 예를 들어 $H=30$에서 `S2`는 DCO1에 유리했지만 DCO2와 DCO4는 `S4-H`가 더 낮았다. $n=3$의 열린 자료에서 population winner나 안정적 direct/flexible-baseline 우월을 선언할 수 없다.

Condition decoder held-out balanced log loss는 DCO1 0.643892, DCO2 0.684471, DCO4 0.716548이었다. 이는 saline/DCZ label 정보이며 task geometry가 아니다. `D1` reverse-minus-forward는 0.005975, 0.012106, 0.008545로 작았고 shuffle-minus-forward만 크게 증가해 temporal structure는 보이지만 directed geometry는 통과하지 못했다. $W_2$는 세 animal 모두 condition distribution shift를 보였지만 treatment-label exchangeability가 release metadata로 닫히지 않아 permutation 위치는 descriptive다.

## Kill-test 판정

`K2`의 V2.2 split 규율만 통과했다. `K3`은 population advantage가 없어 triggered/unresolved이고, `K5`는 reversal이 forward와 가까워 directionality를 clear하지 못했다. 직접 $W$가 없는 gain/noise 비식별성 `K6`, same-unit chain 부재 `K7`, longitudinal cell identity 부재 `K8`, 열린 E17 재사용 금지 `K10`은 해당 강한 주장에 대해 triggered다. `K1`은 synthetic chart check만 있으며 confirmatory cell-resampling은 없고, `K4`의 path/hitting-time endpoint도 없다.

직접 `python` fixture 시도는 base interpreter에 SciPy가 없어 실패했고, 고정한 `uv --with scipy` runtime에서 통과했다. 이 환경 차이는 결과의 실패가 아니라 dependency provenance이며 최종 결과에 package/build 정보를 남겼다.
