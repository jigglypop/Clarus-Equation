# 신경 리만 계량 후보식 전수 검증

Status: COMPLETE

## 초록

연결 변화가 유효 계량을 바꾸고 이후 신경 궤적을 예측한다는 통합 가설을 검증하기 위해 서로 다른 입력과 기하 타입을 대표하는 27개 후보 ID를 유한 원장으로 닫았다. 공식 E17 자료의 11개 session, 3개 animal에서 입력이 있는 5,906개 tuple을 동일한 fit/inner/test와 leave-one-animal-out 규칙으로 전부 계산했다. 수학 감사로 `S7-H,H=1` 항등식과 rank-one field의 수치 오판정을 제거하고, 11개 input byte, runner, 원장과 tuple 완전성을 강제한 V2.2를 다시 실행했다. 장기 horizon 평균에서는 increment-covariance 후보 `S2`가 근소하게 앞섰지만 동물별 방향이 엇갈리고 직접 동역학 및 다른 SPD 후보를 안정적으로 이기지 못했다. E17에는 direct $W$, same-unit longitudinal chain과 독립 lock cohort가 없으므로 뇌의 유일한 리만 계량이나 $\Delta W^s\to\Delta g\to\Delta x$를 확인하지 못했다. 이번 산출은 후보식 전수 실행과 반증 경계의 완결이며, 생물학적 승자 판정은 새 cohort의 결정적 실험으로 남는다.

## 판정

1. `NRM-D1`은 [정의]다. $W^s$, effective $J$, gain/delay state, process noise $Q$, chart와 horizon은 서로 구별해야 한다.
2. `NRM-D2`는 [정의: 모델 선택]다. 이번 세대의 "후보식 전부"는 원장에 고정된 27개 ID와 완전히 열거된 tuple이다.
3. `NRM-T1/T2`는 조건부 수학 정리로 fixture를 통과했다. covariance ridge와 metric reference는 반대 chart law를 가지며, time-varying reachability는 각 innovation을 공통 종점까지 운반해야 한다.
4. raw $W$가 유일한 생물학적 $g$를 정한다는 `NRM-N1`, representation 변화만으로 $\Delta W\to\Delta g$를 안다는 `NRM-N2`, held-out prediction만으로 causal spatial mechanism을 안다는 `NRM-N3`는 완전 반례 때문에 삭제된 강한 부모 주장이다.
5. `NRM-E17D`는 [산출: retrospective discovery]로 완결됐다. `NRM-H1A`는 same-unit direct-$W$ chain 부재로 `UNTESTABLE`; `NRM-H1B/H2--H5`는 미완성이다.

## 결과

모든 계산 가능 tuple은 실행됐고 모든 계산 불가능 식은 필요한 입력과 함께 기록됐다. Uncertainty leaderboard에서 `S2`의 동물평균 NLPD는 $H=5,15,30$에서 각각 1.539538, 2.855627, 3.520984였으나 `S3/S4-H`와 근소한 차이였고 동물별 승패가 뒤집혔다. $H=1$에서는 persistence baseline이 1위였다. 따라서 E17 population winner는 `PROHIBITED`다.

`S7-H,H=1` 88개 tuple은 항등식으로 제거됐고 $H\ge2$도 identity-observation technical proxy다. `S8/S9`는 saline/DCZ decoder를 공유하는 fit-only field gate이며 독립 trajectory metric 결과가 아니다. 모든 graph 후보는 local 계산은 됐지만 frozen LOAO intersection에서 outer fold 0이므로 테스트 불가다. `D1`은 shuffle에 민감했지만 reversal과 forward 차이가 작아 directed action 증거가 약하다. `P1/P2`는 condition distribution shift만 기술한다.

## 관측 비교와 한계

E17 Figure 2의 activity trials, Figure 3의 synapse summary와 Figure 4의 dendrite summary는 같은 unit을 시간축으로 연결하지 않는다. released array order는 미래 chronology로 검증되지 않았고, 독립 단위는 session이나 trial이 아니라 animal 3마리다. session-local fit calibration을 허용했으므로 결과는 hyperparameter 선택 규칙의 LOAO feasibility이지 학습된 metric 자체의 cross-animal transport가 아니다. $S4-H$는 process-noise reachability covariance의 역이지 controllability Gramian이 아니고, predictive covariance가 잘 맞는다는 사실만으로 공간적·인과적 뇌 기제를 얻지 못한다.

## 다음 결정적 실험

새 cohort에서 같은 세포 또는 synapse의 pre/post structural $W^s$, 동일 chart의 calibration activity, 이후 trajectory와 behavior를 연결한다. Intervention arm은 plasticity/connectivity를 직접 조작하고 gain-only, noise-only, replay/sham과 anatomy-fixed context control을 둔다. 후보와 하나의 primary horizon, ridge, score, exclusion, cell matching, animal-level sample size와 Holm family를 source lock 전에 고정한다. Discovery에서 선택한 식 하나와 direct-dynamics, parameter-matched SPD, covariance, identity, label-only baseline을 새 동물에 한 번만 적용한다. 확인 조건은 $\Delta W^s\to\Delta g\to\Delta x$ 방향 일치, animal-level proper-score 우월, chart/cell-resampling 안정성, reversal/shuffle/label permutation 붕괴와 gain/noise 대안 배제다.

## 재현

최종 결과는 `artifacts/e17-candidate-tournament-results-v2.2.json`, SHA-256 `fff9e93c1711341a5a77a5ba4f15996535279fccc687b3bf0fadd9ed7a4b9271`이다. Freeze는 `artifacts/e17-candidate-tournament-freeze-v2.2.json`, result lock은 `artifacts/e17-candidate-tournament-result-lock-v2.2.json`, 독립 validator 출력은 `artifacts/e17-candidate-tournament-validation-v2.2.json`이다. validator는 primary score finiteness, outer 선택·집계와 4개 uncertainty scoreboard를 raw 기록에서 재계산했다. `20-audit.md`는 당시 원장 해시를 기록한 역사적 구현 전 감사이고, 현재 V2.2 지위와 해시는 30/31단계 및 이 보고서가 정한다. 세부 구현과 검증은 `30-implementation.md`, `31-validation.md`, 전체 식과 eligibility는 `artifacts/candidate-equation-registry.md`에 있다.

## 참고

E17 원 논문과 공식 자료는 DOI `10.1126/science.adx4358`, `10.12751/g-node.etlk5k`이며 2026-08-18에 접근했다. 후보식별 1차 문헌과 데이터별 Tier 판정은 `10-sources.md`, 수학 반례와 증명은 `11-math.md`, 결정적 실험 경로는 `12-routes.md`에 기록했다.
