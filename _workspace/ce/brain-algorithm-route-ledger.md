# Brain-algorithm route ledger

Status: ACTIVE

이 원장은 BrainRuntime 실험에서 어떤 알고리즘 후보가 살아 있고 어떤 경로가 기각·퇴역했는지를 빠르게 감사하기 위한 문서다. 실제 뇌를 설명하는 논문이나 의식 이론의 서사가 아니다.

## 정의와 경계

**[정의]** 이 원장의 `CONFIRMED_SIMULATOR`, `STOP`, `APPARATUS_INVALID`, `BLOCKED_NOT_IDENTIFIED`, `OPEN`은 실행 경로의 관리 상태다. CE의 형식 지위나 생물학적 진리 판정이 아니다.

**[공리: 모델 경계]** 고정된 BrainRuntime 코드·설정·seed에서 재현한 결과만 해당 simulator 주장에 사용한다. simulator 결과를 실제 뇌의 기억, 학습, connectome, SCC 또는 의식과 동일시하지 않는다.

**[정의]** 새 후보는 이전 실패와 다른 mechanism 또는 개입 seam을 가지며, 그 차이를 죽일 독립 falsifier와 matched control을 사전에 포함해야 한다. threshold·seed·endpoint·decoder만 바꾼 반복은 새 후보가 아니다.

## 실행 증거 원장

| ID | 후보·mechanism | 실행 상태 | 형식 지위 | 보존되는 좁은 주장 | 증거 경로와 SHA-256 | 재시도·재개 조건 |
|---|---|---|---|---|---|---|
| BA-M0 | supervised rank-4 recurrent write | `CONFIRMED_SIMULATOR` 32/32 | **[산출]** 고정 simulator 계산 | rank 4에 32개 연상의 supervised capacity가 있다. 획득 규칙의 증거는 아니다 | `brain-mechanism-alternative-routes-20260819/artifacts/m0-m1-confirmation-results.json`, `536590a9d38669c5c7fc7485b388f7c4af2d413e213de1cfa68613c287c8f8bb` | brain-algorithm 후보로 재실행하지 않는다. capability ceiling으로만 사용 |
| BA-M1 | fixed-clock delayed local eligibility + replay | `CONFIRMED_SIMULATOR` 32/32 | **[산출]** 고정 simulator 계산 | zero-store cue/value binding acquisition과 여섯 adverse-control 분리를 확인했다 | 같은 confirmation artifact와 SHA | factor transfer·prediction·biological consolidation을 별도 입증해야 한다 |
| BA-T1 | frozen M1 factorized held-out composition | `STOP` 11/16 | **[산출]** 부정 계산 | M1 binding은 안정된 조합 전이를 보장하지 않는다 | `brain-memory-contrastive-predictive-routes-20260819/artifacts/t1-development-results-v2-audited.json`, `1c1914b952ead084a21a88a35abca983314dccb013e57c333d1d1075436841fa` | 같은 불균형 schedule의 threshold/seed retune 금지. factor-balanced mechanism이 새로 필요 |
| BA-M2 | positive-minus-negative lag contrastive write | `STOP` 0/16 | **[산출]** 부정 계산 | frozen schedule에서 negative phase가 정확히 0이고 positive write도 recall을 만들지 못했다 | binding `brain-memory-contrastive-predictive-routes-20260819/artifacts/m2-binding-development-results-v2-frozen.json`, `0ddbe12b5b78c6c3e9a4f1d4d14a1b5b24690807a0d6f45646f84ec368535567`; factor `brain-memory-contrastive-predictive-routes-20260819/artifacts/m2-factor-development-results-v2-frozen.json`, `f52a2ee5bf5424c27823eb2441514357b0236d939c704f3dea2cfb0ab9764d11` | nonzero negative phase를 만드는 독립 mechanism 없이는 퇴역 |
| BA-M3 | teacher-forced replay residual write | `STOP` | **[산출]** 부정 계산 | binding capability는 있으나 predictor가 persistence를 이기지 못했고 transition-order shuffle이 binding을 재현했다 | predictor `brain-memory-contrastive-predictive-routes-20260819/artifacts/m3-predictor-development-results-v2-frozen.json`, `4e865af022bc7e8ac33a11861a83816b7e4b94fce097ed382f88fc3d45fbdaff`; binding `brain-memory-contrastive-predictive-routes-20260819/artifacts/m3-binding-development-results-v2-frozen.json`, `80db6e32ce50a84a716f9c989441fdea8f5dce19b3f011113bf46188a8bd0879`; factor `brain-memory-contrastive-predictive-routes-20260819/artifacts/m3-factor-development-results-v2-frozen.json`, `2b89b82ffb757e9d6cbb97f7a52ebeafbfba79998092845dfb9ab8cf6f7db8b3` | predictor→policy 인과 seam처럼 학습 write와 독립된 새 시험만 허용 |
| BA-G1 | directed $\operatorname{do}(W)$ → SPD response + endpoint | `STOP` 0/16 | **[산출]** 부정 계산 | 평균 효과는 양수였지만 preregistered per-circuit effect gate를 통과하지 못했다 | `brainruntime-weight-metric-dynamics-intervention-20260819/artifacts/g1-development-results-v1.json`, `3e2a69b22bce0ae906bdf4b3fd3a2830421df6f385969e19577a0ca6b1cca6f3` | 효과크기 문턱·noise control을 결과 후 바꾼 재시도 금지 |
| BA-G2 | compressed SPD metric feature for fixed-W prediction | `STOP` 0/16 | **[산출]** 부정 계산 | $g$는 유효한 압축 표현이지만 raw horizon $B_h$, direct quadratic, $C$ terms보다 유용하지 않았다 | `brainruntime-weight-metric-dynamics-intervention-20260819/artifacts/g2-development-results-v1.json`, `c4fbeeee6cc3e71f596238e902007c30f942cad389d9bf6e45884cdd0437b489` | “metric sufficiency/고유 정보” 경로는 퇴역. 새 독립 mediator 개입 없이는 재개 금지 |
| BA-G3D-v1 | response/recall diagnostic, first apparatus | `APPARATUS_INVALID` | **[미완성]** | 과학 결과로 사용하지 않는다 | `brainruntime-weight-metric-dynamics-intervention-20260819/artifacts/g3-diagnostic-development-results-v1.json`, `2075e3516ffab8c21691535540443f8b4743609b227416bbd420b40c15e2ee9e` | seed `97701..97716` 영구 퇴역 |
| BA-G3D | independent response-summary/recall co-change diagnostic | `STOP`; mediation `BLOCKED_NOT_IDENTIFIED` | **[산출]** 부정 계산 + **[미완성]** mediation | M1 continuous recall advantage는 강하지만 global SPD-change 우위와 same-arm correlation은 실패했다 | `brainruntime-weight-metric-dynamics-intervention-20260819/artifacts/g3-diagnostic-development-results-v2.json`, `a0a9321aca3366d1c6c7d4e12f7cfe97d1387e86c6e05799ee90543a01a665b7` | independently manipulable mediator 또는 충분한 causal state model 없이는 mediation 재개 금지 |

## 활성 후보 순위

| 순위 | ID | 상태 | 선택 이유 | 필수 falsifier / stop 조건 |
|---:|---|---|---|---|
| 1 | BA-C1 | `OPEN` | 기존 Loop10의 “예측 readout만 존재” 경계를 직접 넘어서, frozen forecast가 실제 정책 action을 고르는지를 개입할 수 있다 | predictor→policy port shuffle은 손실·행동을 바꾸고, 표시 readout만 shuffle한 arm은 결과가 동등해야 한다. guided policy가 persistence·random·error-magnitude-only·reactive control을 이기지 못하면 STOP |
| 2 | BA-S1 | `OPEN_CONDITIONAL` | recurrence topology 가설을 직접 lesion하기 전에 현재 M1 구조에서 공정한 SCC matched control이 존재하는지 판정한다 | support가 dense/한 giant SCC라 outside·SCC-preserving matched lesion이 불가능하면 `STRUCTURE_UNDEFINED_STOP`; threshold를 outcome 후 조정하지 않는다 |
| 3 | BA-BIO-LONG | `BLOCKED_INPUT` | 실제 뇌 주장에 필요한 same-cell/synapse longitudinal $W$, activity, intervention 자료 경로다 | 직접 구조 측정·독립 metric calibration·held-out dynamics·animal-level intervention이 없으면 열지 않는다 |

## 오케스트레이터 선택 규칙

1. 활성 후보 1번부터 capability dependency를 검사한다. 앞 후보가 STOP이어도 다음 후보가 독립 mechanism이면 진행할 수 있다.
2. 새 결과는 해당 run의 12/31/40과 artifact hash가 안정된 뒤 이 원장에 한 번만 반영한다.
3. 양성 결과를 찾기 위해 seed·threshold·endpoint를 순회하지 않는다. apparatus 결함은 과학 결과와 분리해 전 seed block을 퇴역시키고 새 contract로만 교체한다.
4. `CONFIRMED_SIMULATOR`는 실제 뇌 증거가 아니다. 생물학적 승격은 BA-BIO-LONG의 독립 입력과 개입 gate가 있어야 한다.

## 현재 인계

**[미완성]** BA-C1 계약·구현과 BA-S1 구조 적격성 preflight는 아직 실행 결과가 없다. 다음 오케스트레이터는 이 두 후보를 같은 “의식” 주장으로 합치지 않고 독립 판정해야 한다.
