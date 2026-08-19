# 학습된 계산 기하와 수면 재정렬 연구 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/_archive/agi-connectome-geometric-memory-20260816
PREDECESSOR: _workspace/ce/_archive/agi-causal-recurrent-geometry-phase-a-20260816
PREDECESSOR: docs/7_AGI/1_AGI.md

## 1. 질문과 목표

사용자 가설의 핵심 연쇄

$$
\Delta W \longrightarrow \Delta g \longrightarrow \Delta x(t)
$$

를 수학 정리, 모델 정의, 경험 가설로 분해한다. 구체적으로 (a) 국소 연결 또는 비용 변화가 전역 최단경로 비용을 어떻게 바꾸는지 증명하고, (b) 그래프 변화와 계량 변화가 어떤 관측 수준에서만 동치인지 경계를 밝히며, (c) 학습 전후 유효 계산 기하가 이후 신경·에이전트 궤적을 추가로 예측하는지, (d) 수면을 그 기하의 재정렬로 보는 해석이 기존 연결·활동·표현 대조군보다 강한 예측을 내는지 판정한다.

목표는 직관 전체를 참으로 만드는 것이 아니다. 완전 반례가 있는 부모 주장은 삭제하고, 증명 가능한 핵심과 반증 가능한 실험 프로그램을 보존하여 CE-AGI 정본을 수정할 수 있는 수준까지 닫는 것이다.

## 2. 정의와 대상

[정의] 유한 상태 집합 $V$와 양의 비용을 가진 유향 간선 집합 $E$로 이루어진 계산 그래프를 $G=(V,E,w)$라 한다. 도달 가능한 두 상태의 유향 최단경로 비용을 $d_G(i,j)$라 하며, 도달 불가능하면 $+\infty$로 둔다. 대칭 무향 그래프가 아닌 일반 경우 $d_G$는 metric이 아니라 extended quasi-metric일 수 있다.

[정의] 문맥 $c$에서 정책 또는 동역학이 실제로 사용하는 전이 비용을 $w_c$라 하고, 이에 따른 task-conditioned effective distance를 $d_c$라 한다. 신경 연결 강도 $W$에서 비용 $w_c$로 가는 사상 $\Phi_c$는 자동으로 주어지지 않으며 모델 또는 측정 절차로 명시해야 한다.

$$
w_c=\Phi_c(W,A,c),
\qquad
d_c=\operatorname{ShortestPath}(G,w_c),
$$

여기서 $A$는 활동, gain, 지연, 억제와 가용성 등 연결 강도만으로 결정되지 않는 상태를 포함한다.

[정의] 빠른 상태 궤적을 $x_{0:T}$, 학습 전후 느린 연결 상태를 $W^{\rm pre},W^{\rm post}$, 그로부터 사전 고정한 절차로 추정한 유효 기하를 $g^{\rm pre},g^{\rm post}$ 또는 $d^{\rm pre},d^{\rm post}$라 쓴다. $g$는 연속 SPD tensor일 수도 있고, 그래프 shortest-path cost 또는 control-energy quadratic form일 수도 있으므로 서로 다른 typed object를 동일시하지 않는다.

[정의] 이 run에서 수면 재정렬은 수면 전후 연결·gain·표현의 변화가 다음 날의 task-conditioned transition cost, trajectory fidelity 또는 일반화를 바꾸는 경험적 가설이다. NREM, REM, replay, criticality를 하나의 직렬 알고리즘으로 선결하지 않는다.

## 3. 등록 주장

| ID | 주장 | 시작 지위 | 필요한 판정 |
|---|---|---|---|
| `LGS-T1` | 간선 $(u,v)$를 비용 $a>0$로 추가한 뒤 모든 쌍 최단거리는 증가하지 않으며, 음의 cycle이 없는 조건에서 $d'(i,j)=\min\{d(i,j),d(i,u)+a+d(v,j)\}$다 | [정리 후보] | 엄밀 증명과 유한 반례 검사 |
| `LGS-T2` | 국소 간선 변화가 많은 쌍의 최단거리와 최적 경로를 바꿀 수 있지만 반드시 모든 쌍을 바꾸지는 않는다 | [정리 후보] | 영향 집합의 필요충분 조건과 경계 예시 |
| `LGS-N1` | shortcut edge 추가와 공간의 계량 변형은 구조적으로 동일하다 | [강한 부모 주장] | topology, direction, multiplicity, dynamics를 보존하지 않는 완전 반례가 있으면 삭제 |
| `LGS-T3` | 고정된 상태 집합과 선택된 cost observable에 대해서는 두 모델의 pairwise cost와 induced policy가 같을 때 operationally 동치라 부를 수 있다 | [조건부 정리 후보] | 동치관계와 식별 불가능 범위를 명시 |
| `LGS-N2` | 연결 강도 변화 $\Delta W$가 별도 가정 없이 유일한 계량 변화 $\Delta g$를 결정한다 | [강한 부모 주장] | gauge, activity/gain dependence와 비식별성 반례 |
| `LGS-H1` | 사전 고정한 $\Phi_c$로 계산한 학습 전후 유효거리 변화가 raw $W$, activity, latency, Euclidean latent distance 이후에도 이후 궤적·반응시간·오류를 추가 예측한다 | [미완성] | held-out longitudinal/interventional 비교 |
| `LGS-N3` | 인지시간은 보편적으로 $T_{\rm cognition}=L_{\rm effective}/v_{\rm neural}$이다 | [강한 부모 주장] | 병렬 처리, synaptic delay, integration overhead 반례 |
| `LGS-H2` | 라우팅 깊이 또는 유효 geodesic cost는 적절한 고정비용 항과 병렬성 항을 포함한 모델에서 반응시간의 독립 예측변수다 | [미완성] | equal-budget baseline, mediation와 intervention |
| `LGS-H3` | 숙련은 자주 필요한 상태 사이의 task-conditioned effective cost를 선택적으로 줄인다 | [미완성] | 학습 전후 변화, 사용빈도 dose-response, untrained-pair control |
| `LGS-H4` | 문맥 전환은 고정 topology 위에서 cost/gain을 바꿔 서로 다른 effective geometry를 선택한다 | [미완성] | 동일 anatomy 내 context intervention과 trajectory prediction |
| `LGS-H5` | 수면은 학습으로 변한 계산 기하를 단순 전역 축소가 아니라 선택적으로 재정렬하며 다음 날 궤적 충실도와 일반화를 개선한다 | [미완성] | sleep/wake matched control, local/global metric, replay coupling |
| `LGS-N4` | NREM은 잡음 곡률 제거이고 REM은 새 조합 샘플링이라는 고정 직렬 알고리즘이다 | [강한 부모 주장] | 동시성·지역성·다기능성과 직접 인과증거 경계로 판정 |
| `LGS-H6` | $\Delta W\to\Delta g\to\Delta x(t)$의 매개 경로는 학습 전후, 문맥 전환, 수면 전후에서 방향성과 out-of-sample 예측을 반복한다 | [통합 예측] | 세 조건별 독립 split과 mediation/ablation |
| `LGS-X1` | 위 가설이 참이면 AGI가 증명된다 | 활성 주장 제외 | 좁은 계산·신경 가설과 AGI 충분조건 사이 bridge 부재 |

## 4. 수학 레인 요구사항

1. `LGS-T1`을 extended distance와 도달 불가능 쌍까지 포함해 증명하고, directed/undirected 및 edge decrease/edge addition을 구분한다.
2. 영향 집합

$$
S_{uv}(a)=\{(i,j):d(i,u)+a+d(v,j)<d(i,j)\}
$$

을 정의하여 전역 변화가 가능한 조건과 변화가 전혀 없는 조건을 모두 제시한다.
3. pairwise distance 보존, optimal-policy 보존, trajectory-law 보존, topology 보존을 서로 다른 동치로 분리하고 함의가 역으로 성립하지 않는 반례를 만든다.
4. $W\mapsto g$의 비식별성, directed drift와 symmetric metric의 차이, shortest-path quasi-metric과 Riemannian metric의 type 차이를 감사한다.
5. 인지시간 식에 최소한 경로 길이, edge별 지연, 병렬 critical path, integration/decision overhead가 들어가는 좁은 대안을 제시한다.
6. 작은 유한 그래프의 exhaustive 또는 property-based 검산으로 정리와 반례를 재현한다.

## 5. 외부 근거 레인 요구사항

사용자 첨부의 20개 인용과 수면 스케줄러 노트의 10개 인용을 DOI, 저널 원문 또는 공식 데이터 페이지에서 확인한다. 중복은 합치고 다음 항목을 추출한다.

- 실제 출판 연도와 peer-review 상태
- 종, 표본, 기록/개입 방식과 독립 통계 단위
- 직접 측정한 변수와 저자가 주장한 범위
- geometry, routing, cache, unit test, renormalization이라는 해석이 원 논문의 결과인지 본 연구의 비유인지
- 공개 데이터·코드와 $\Delta W\to\Delta g\to\Delta x(t)$ 재분석 가능성

우선 근거군은 learning-induced neural geometry, communication subspace, 숙련과 대사비용, sleep local ON/OFF induction, sleep/criticality, sleep deprivation과 replay, NREM microstate replay, slow-wave/spindle/ripple coupling, 수면 후 표현 변환이다. 미래 시점·서지 오류·원문이 확인되지 않는 인용은 활성 근거에서 제외한다.

## 6. 경험 검증 설계

### 6.1 최소 계산 benchmark

동일한 state graph에서 학습 전후 edge/cost perturbation을 만들고 다음 세 모델을 비교한다.

1. raw edge/weight feature baseline;
2. local activity/latency feature baseline;
3. preregistered effective-distance feature를 추가한 모델.

held-out graph와 perturbation에서 future trajectory next-state NLL, goal hitting time, path fidelity와 energy proxy를 평가한다. geometry 추가 이득이 없으면 독립 기전 주장을 철회한다.

### 6.2 생물 자료 단계

한 데이터셋이 연결 변화, geometry와 이후 궤적을 모두 제공하지 않으면 서로 다른 데이터의 상관을 하나의 인과 사슬로 합산하지 않는다. 우선순위는 다음과 같다.

1. 같은 개체·세션의 학습 전후 연결 또는 effective connectivity와 neural trajectory;
2. 같은 개체·시간창의 sleep pressure/criticality 또는 routing entropy와 replay fidelity;
3. 구조 변화가 없더라도 context intervention으로 gain/cost와 trajectory가 함께 변하는 자료.

### 6.3 primary endpoint와 kill test

| 축 | primary endpoint | 필수 대조 | kill test |
|---|---|---|---|
| edge-to-distance | all-pairs distance 변화의 exact 재현 | 직접 APSP 재계산 | closed-form mismatch 한 건 |
| distance-to-trajectory | held-out trajectory NLL 또는 hitting-time error | raw $W$, activity, Euclidean, parameter-matched model | geometry의 paired CI가 0을 넘지 못함 |
| context geometry | held-out context trajectory prediction | one-metric, context-label-only, gain-only | geometry shuffle 후 성능 유지 |
| sleep rearrangement | post-sleep replay fidelity/generalization | time-awake, firing-rate, global scaling, no-sleep control | 선택적 재정렬이 global scaling과 구별되지 않음 |
| mediation chain | preregistered indirect effect와 intervention direction | direct $W\to x$, confounder-adjusted alternatives | $g$ 제거가 예측을 악화시키지 않음 |

## 7. 구현과 문서 수정 경계

감사 전에는 제품 runtime 또는 AGI 정본을 수정하지 않는다. gate 통과 뒤 다음만 허용한다.

- 정리와 반례를 재현하는 작은 검증 스크립트 및 focused test;
- `docs/7_AGI/1_AGI.md`, `3_Sleep.md`, `15_Equations.md`의 관련 부모 주장 교정;
- 새 정본 장 또는 기존 장에 operational geometry, 경험 예측, kill test를 추가;
- 후속 데이터 재분석을 위한 preregistration 초안.

기존 CE 물리 metric, V16 SPD cost learner, V17 sign no-go, CRGM의 controllability geometry를 자동으로 신경 geodesic과 동일시하지 않는다.

## 8. 완료 조건

1. 모든 등록 주장에 수학·출처·형식 지위 판정이 있다.
2. 완전 반례가 있는 `LGS-N1`--`N4` 부모 주장은 최종 보고서와 활성 AGI 정본에서 제거되거나 명시적 반례 경계로 대체된다.
3. `LGS-T1`--`T3`의 증명 또는 반증이 재현 가능한 검산과 함께 제공된다.
4. 30개 사용자 인용의 중복 제거 후 각 고유 자료에 검증 상태가 기록된다.
5. 통합 경험 예측 `LGS-H6`은 데이터 요구사항, baseline, split, primary endpoint와 중단 규칙을 갖는다.
6. 승인된 문서/코드 변경은 focused validation을 통과한다.
7. 최종 보고서는 증명된 핵심, 기각된 직관, 살아남은 AGI 설계 원리, 가장 값싼 다음 실험을 독립적으로 설명한다.
