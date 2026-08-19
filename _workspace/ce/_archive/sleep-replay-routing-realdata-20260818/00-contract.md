# 실제 뇌 데이터 기반 수면·replay·라우팅 기하 검증 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/_archive/agi-learning-geometry-sleep-20260818
PREDECESSOR: _workspace/ce/_archive/agi-causal-recurrent-geometry-phase-a-20260816

## 1. 질문

공개된 실제 신경 데이터에서 다음 세 축을 독립적으로 재현하고 연결 가능한 범위를 판정한다.

1. 수면박탈 또는 수면 상태에 따른 hippocampal replay fidelity 변화;
2. 같은 동물·세션·시간창에서 추정한 effective branching, transition entropy 또는 routing geometry와 replay fidelity의 관계;
3. 학습 및 REM/SWS 뒤 neural representation geometry와 item/category structure의 변화.

최종 목표는 `꿈`, `라우팅`, `학습된 기하`를 하나의 설명으로 묶는 것이 아니라, 실제 데이터가 허용하는 연결과 허용하지 않는 연결을 수치로 가르는 것이다. 합성 데이터는 분석 코드의 unit fixture로만 허용하며 경험 결과를 대신할 수 없다.

## 2. 우선 자료

| 축 | 1순위 1차 출처 | 요구 자료 |
|---|---|---|
| replay·branching | Giri et al., *Nature* (2024), `s41586-024-07538-2` | animal/session/time-window 수준 spike, SWR, replay score, sleep-deprivation condition |
| learning geometry | Mazurek et al., *Nature Neuroscience* (2026), `s41593-026-02333-w` | training stage별 population activity 또는 공개 source-data geometry summaries |
| REM/SWS transformation | *Communications Biology* (2025), `s42003-025-08812-3` | participant-level item/category representation change, REM/SWS measures |
| criticality replication | Xu et al., *Nature Neuroscience* (2023), `s41593-023-01536-9` | cortical spike/event time series와 wake/sleep label, 공개 analysis code |

[공리: 자료 계층] event-level raw/processed data가 공개되면 동일 시간창 분석을 수행한다. 이 분석에는 animal, session, timestamp, unit quality, condition, SWR/replay endpoint가 연결되어 있어야 한다. figure source-data 또는 group summary만 공개되면 해당 figure/statistic 재현까지만 허용하며 개체 수준 연관, 예측 또는 인과 주장으로 확장하지 않는다. 필요한 granularity가 없으면 해당 주장은 음성 결과가 아니라 `UNTESTABLE`이다. 논문 수치의 수기 전사는 데이터 분석으로 세지 않는다.

## 3. 데이터 무결성과 provenance

각 획득 객체에 다음을 기록한다.

- 공식 article, repository 또는 저자 GitHub/OSF/Dryad/Zenodo URL;
- release/version, 접근일, license와 파일 크기;
- 원본 SHA-256과 local relative path;
- 압축 해제 전후 manifest;
- animal, session, condition, time unit과 exclusion;
- raw, processed, source-data table 또는 code-only 등급.

네트워크 접근 실패, 접근 승인 필요, 공개되지 않은 raw object와 라이선스 제한을 구분한다. hash와 schema가 확인되지 않은 파일은 채점에 사용하지 않는다.

## 4. 등록 주장

| ID | 주장 | 시작 지위 | 필수 증거 |
|---|---|---|---|
| `RR-D1` | 공개 자료와 코드를 공식 출처에서 결정적으로 재획득할 수 있다 | [미완성] | URL, version, license, size, SHA-256 manifest |
| `RR-R1` | E15 공개 자료에서 수면박탈 조건의 replay/reactivation 감소를 독립 재현한다 | [미완성] | paper-matched unit, effect direction, uncertainty와 exclusion |
| `RR-H1` | 같은 E15 animal/session/window에서 $B_{\rm eff}$ 또는 transition entropy가 클수록 replay fidelity가 낮다 | [예측 후보] | preregistered estimator, covariates, held-out animal/session |
| `RR-H2` | 임계 sigmoid가 선형 branching-replay model보다 held-out prediction에서 낫다 | [예측 후보] | generally non-nested model의 outer animal/session group-held-out proper score, training-only parameter fit, linear/constant baseline, threshold stability |
| `RR-H3` | E02에서 학습이 task-relevant geometry를 바꾸며 그 변화가 behavior를 추가 예측한다 | [미완성] | participant/animal split, raw/source-data reproduction, behavior baseline |
| `RR-H4` | E19에서 REM/SWS balance가 item 감소와 category 보존·강화를 추가 예측한다 | [미완성] | participant-level model, sleep-duration/baseline covariates, uncertainty |
| `RR-H5` | replay/criticality, learning geometry와 REM transformation 결과가 하나의 $\Delta W\to\Delta g\to\Delta x(t)$ 사슬을 이룬다 | [강한 통합 주장] | 같은 개체·측정 사슬이 없으면 기각 또는 분리 유지 |
| `RR-H6` | REM이 `새 조합을 샘플링하는 꿈 알고리즘`임이 자료로 확인된다 | [강한 부모 주장] | generative recombination의 직접 event-level measure와 REM-specific causal control |
| `RR-X1` | 위 결과가 AGI architecture의 우위를 증명한다 | 활성 제외 | matched AGI benchmark bridge 없음 |

## 5. 사전등록 분석

### 5.1 E15 replay 재현

논문의 공개 code가 정의한 replay/reactivation endpoint를 먼저 byte-level 수정 없이 재현한다. 통계 단위는 frame이나 SWR event가 아니라 animal 또는 독립 session이다. primary는 sleep-deprivation minus control의 replay fidelity 차이이며, 방향은 control보다 작음이다. 논문 figure와 차이가 나면 단위, exclusion, seed와 normalization을 먼저 대조하고 결과에 맞춰 정의를 바꾸지 않는다.

### 5.2 동일 시간창 branching/라우팅 분석

event-level spike data가 있을 때만 실행한다. window 길이, bin 크기와 threshold는 training animals에서 고정한다. 최소 두 estimator를 사용한다.

1. branching proxy: 연속 bin population event count의 회귀 기울기 또는 공개 criticality code와 같은 estimator;
2. transition entropy: state discretization을 training split에서 고정한 뒤 다음-state conditional entropy.

Primary model은

$$
Q_{a,s,t}=\alpha_a+\beta B_{a,s,t}+\gamma_1R_{a,s,t}
+\gamma_2N^{\rm SWR}_{a,s,t}+\gamma_3T^{\rm awake}_{a,s,t}+\epsilon_{a,s,t},
$$

여기서 $Q$는 replay fidelity, $B$는 branching/entropy estimator, $R$은 firing-rate proxy다. primary prediction은 $\beta<0$이다. animal/session group split을 사용하고 window를 독립 표본처럼 세어 유의성을 부풀리지 않는다.

임계 모델은

$$
\widehat Q(B)=\alpha+\frac{A}{1+\exp[k(B-B^*)]}
$$

이며 $k>0$과 $B^*$를 포함한 모든 parameter와 hyperparameter는 각 outer split의 training animal/session에서만 fit한다. sigmoid와 linear model은 일반적으로 non-nested이므로 likelihood-ratio test를 쓰지 않는다. 미리 정한 proper score로 outer animal/session held-out prediction을 linear와 constant baseline에 비교하고, $B^*$가 training 범위 내부에 있으며 split 사이에서 안정적이어야 `RR-H2`가 살아남는다. window 수는 confirmatory degrees of freedom을 늘리지 않는다.

### 5.3 E02와 E19 재현

raw population activity가 있으면 논문의 geometry statistic을 먼저 재현하고, source-data table만 있으면 공개된 figure effect와 uncertainty만 재계산한다. E19 participant-level 자료에서는 item-change와 category-change를 분리하며 REM/SWS ratio, total sleep, baseline score와 가능한 age/condition covariates를 함께 둔다. group mean 표만 있으면 participant-level 추가 예측은 `UNTESTABLE`로 남긴다.

## 6. 대조와 kill test

| 주장 | 필수 대조 | kill test |
|---|---|---|
| RR-R1 | paper code, condition-label shuffle, unit/exclusion ledger | 공개 자료로 방향 또는 figure를 재현하지 못함 |
| RR-H1 | firing rate, SWR count, time awake, window autocorrelation, animal/session random effect | held-out $\beta$ 방향 실패 또는 baseline 대비 추가 예측 없음 |
| RR-H2 | linear, constant, monotone spline; equal split | held-out 우위 없음 또는 $B^*$ 불안정 |
| RR-H3 | trial count, behavior-only, raw activity dimension, label shuffle | geometry statistic이 behavior에 추가 정보 없음 |
| RR-H4 | total sleep, baseline, SWS/REM 각각, ratio permutation | participant-level 추가 예측 없음 또는 자료 부재 |
| RR-H5 | dataset/object identity audit | 다른 종·개체·자료의 부분 결과만 존재 |
| RR-H6 | NREM/wake control, direct recombination metric, causal REM manipulation | category 변화만 있고 generative recombination 직접 측정 없음 |

다중 window, bin, state dimension과 metric 후보의 선택 수를 기록한다. target을 본 뒤 선택한 분석은 exploratory로 표시하고 confirmatory 결과에 합치지 않는다.

## 7. 구현 경계

감사 전에는 외부 데이터나 code를 제품 경로에 넣지 않는다. 승인 뒤 `artifacts/realdata/` 아래에 manifest, acquisition receipt와 가능한 작은 공개 자료만 보관하고, 라이선스나 크기 때문에 저장소에 둘 수 없는 raw data는 workspace 밖의 승인된 cache 경로와 SHA-256으로 참조한다. 분석 코드는 run `artifacts/`에 두고, 재사용 가치와 license가 확인된 경우에만 후속 정본 구현으로 승격한다.

대용량 다운로드, 계정 로그인, data-use agreement 수락 또는 외부 compute가 필요하면 정확한 객체와 크기를 확인한 뒤 승인 요청한다. 공개 source-data와 code의 소규모 다운로드는 이 실데이터 검증의 정상 구현 단계다.

## 8. 완료 조건

1. 네 우선 자료 각각에 공식 접근 경로, 공개 등급, license/제약과 재현 가능 판정이 있다.
2. 접근 가능한 실제 데이터는 checksum·schema 검사 후 분석하며 합성 자료로 대체하지 않는다.
3. 최소 하나의 실제 뇌 source-data 또는 event-level dataset에서 원 논문 결과를 수치 재현한다.
4. event-level E15가 공개되면 동일 시간창 `RR-H1/H2`를 실행한다. 공개되지 않으면 접근 부재를 증명하고 가능한 가장 가까운 실제-data 분석을 수행한다.
5. E02/E19는 공개 granularity가 허용하는 최대 수준까지 재현하고 `UNTESTABLE`과 실패를 숨기지 않는다.
6. 꿈 알고리즘, routing geometry와 통합 사슬에 반례/kill-test 판정이 있다.
7. 실행 code, manifest, 원문 log, focused test와 최종 보고서가 재현 가능하다.
