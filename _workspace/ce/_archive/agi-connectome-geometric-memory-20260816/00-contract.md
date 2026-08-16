# 인과적 재귀 기하 복원 연구 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/_archive/agi-v15-unified-metric-20260813
PREDECESSOR: docs/7_AGI/1_AGI.md

## 1. 연구 제목과 한 문장 명제

**연구 제목:** 관측 동역학에서 숨은 인과 구조와 재귀 모듈을 복원하고, 학습으로 변한 접근 기하가 기억 회상을 예측하는지 검증하는 인과적 재귀 기하 복원.

**한 문장 명제:** 부분 관측과 개입이 주어진 동적 시스템에서 비교적 불변인 상호작용 법칙, 시간가변 문맥, 재귀 모듈과 접근 기하를 분리해 학습하면, 같은 계산량의 비기하·비재귀 대조군보다 보지 않은 개입의 장기 rollout과 기억 회상을 더 잘 예측할 수 있는가.

이 계약은 사용자가 제공한 `Connectome / SCC / Riemannian Memory` 가설 보고서를 연구 프로그램으로 바꾼다. 의식이나 자아를 결과로 선결하지 않고, 구조 복원, SCC 압축, 기하 기억이라는 세 개의 독립 검증축부터 시험한다.

## 2. 대상과 범위

[정의] 관측 시계열을 $X_{0:T}$, 알려진 개입 또는 행동을 $u_{0:T}$, 숨은 비교적 불변인 유향 상호작용 구조를 $G$, 시간가변 문맥을 $z_t$, 상태 접근 비용을 나타내는 대칭 양의 정부호 계량을 $g_t$, 전이 법칙을 $F$라 쓴다. 후보 생성 모형은 다음과 같다.

$$
x_{t+1}=F_\theta(x_t,z_t,G,g_t,u_t)+\eta_t.
\tag{1}
$$

[정의] $R(G)$는 고정된 유향 그래프 $G$의 maximal strongly connected component 분할이고, $\operatorname{Cond}(G)$는 그 분할을 축약한 condensation DAG다. SCC를 곧바로 의식, 기억 또는 생물학적 기관과 동일시하지 않는다.

[정의] 기억 가설에서 빠른 상태는 $x_t$, 느린 학습 상태는 $g_t$와 필요한 경우 $G_t$다. cue $c$가 학습 전후의 느린 상태를 통해 과제 관련 recurrent trajectory를 재생성하는지를 회상으로 측정한다.

$$
(x_t,g_t,G_t)
\xrightarrow{\text{learning}}
(x_{t+1},g_{t+1},G_{t+1}),
\qquad
c+ (g_T,G_T)\longrightarrow \widehat\tau_c.
\tag{2}
$$

이번 run은 연구 질문, 식별성 경계, 대조군, 평가 프로토콜과 논문 순서를 고정하는 이론·설계 run이다. 외부 대용량 자료를 내려받거나 실제 MICrONS 전량을 학습시키거나 제품 AGI runtime을 승격하지 않는다.

## 3. 최소 주장

| Claim ID | 계약 주장 | 시작 지위 | 판정 요구 |
|---|---|---|---|
| `CGM-D1` | 유한 고정 유향 그래프의 SCC 분할은 유일하며 condensation은 DAG다 | [정리 후보] | 정의와 표준 증명 대조 |
| `CGM-N1` | 일반적인 관측 시계열만으로 잠재 좌표와 정확한 구조 $G$를 유일 복원할 수 있다 | [미완성] | 관측 동치·잠재 재매개변수 반례 |
| `CGM-H1` | 개입과 관측 모형 제약이 있으면 $G/F$와 $z_t$의 분리가 보지 않은 개입 rollout을 개선한다 | [미완성] | 식별 조건, 대조군, held-out 환경 |
| `CGM-H2` | 복원 그래프의 SCC quotient가 같은 압축률의 community·spectral·learned-module 대조군보다 장기 예측과 개입 전달을 개선한다 | [미완성] | compute/parameter-matched 비교 |
| `CGM-H3` | 학습 전후 $g_t$의 변화가 단순 weight·activity 통계 이후에도 회상 성공과 basin 접근성을 추가로 예측한다 | [미완성] | nested baseline과 ablation |
| `CGM-H4` | 장기기억은 활성 SCC의 영구 보존이 아니라 cue에서 관련 recurrent trajectory를 재생성하는 느린 제약으로 모델링할 수 있다 | [미완성] | encoding/recall 불변량과 인과 조작 |
| `CGM-H5` | 하나의 구조가 아니라 문맥별 SPD basis의 혼합이 단일 metric보다 OOD 문맥 전환을 더 잘 설명한다 | [미완성] | rank·parameter-matched 단일 metric 대조 |
| `CGM-H6` | 같은 복원 규약이 합성 동역학과 신경 자료에서 모두 작동한다 | [미완성] | domain별 재튜닝 회계와 교차 도메인 시험 |
| `CGM-X1` | SCC가 존재하면 의식이 존재한다 | 활성 주장 제외 | 충분조건 근거가 없으며 이번 연구 대상 아님 |
| `CGM-X2` | 계량 하나가 비가역 전이, 의미 목표와 기억을 무입력으로 모두 결정한다 | 활성 주장 제외 | 선행 V15의 대칭·방향성 no-go 보존 |

## 4. 핵심 식별성 질문

수학 레인은 다음을 구분해야 한다.

1. 완전 관측 선형계, 부분 관측 비선형계, 잠재 교란이 있는 계에서 복원 가능한 대상이 정확한 edge인지, Markov/동역학 동치류인지, 예측 충분통계인지 구분한다.
2. 좌표변환 $h$에 따른 잠재 재매개변수 $x'_t=h(x_t)$가 같은 관측 분포를 만드는 경우 $G$와 $g$의 비식별성을 반례로 고정한다.
3. SCC는 고정 그래프의 위상 성질이므로 threshold, time window와 effective-edge 정의가 바뀌면 분할도 바뀐다는 점을 회계한다.
4. 대칭 Riemannian metric $g_t$와 유향 drift/transition $F$를 동일시하지 않는다. 비가역성은 $F$, control, time orientation 또는 비대칭 구조에서 온다.
5. $g_t$의 좌표 공변성, SPD 보존, gauge freedom과 regularization이 예측 개선을 가장해 parameter count를 늘리지 않는지 검사한다.

## 5. 1차 연구 프로토콜

### 5.1 단계 A: 합성 식별성 시험

첫 논문은 ground truth가 완전히 알려진 유한 동역학으로 제한한다. 선형 VAR, 비선형 recurrent network, context-switching system을 사용하고 다음 축을 교차한다.

- 완전 관측 대 부분 관측
- 수동 관측 대 single-node·module intervention
- 고정 $G$ 대 느리게 변하는 $G_t$
- 하나의 metric 대 context mixture
- 알려진 SCC 구조 대 SCC가 거의 없는 DAG 구조

train/validation/test는 trajectory가 아니라 생성 graph와 environment seed 단위로 분리한다. test graph와 intervention target은 모델 선택에 사용하지 않는다.

### 5.2 단계 B: blind structure/function 시험

구조와 기능이 함께 있는 공개 자료에서는 구조를 끝까지 숨긴다. activity, stimulus, behavior와 허용된 intervention만으로 $\widehat G$와 $R(\widehat G)$를 만든 뒤 freeze된 평가 코드로 공개 구조와 비교한다. 해부학적 edge와 effective causal edge가 같다는 가정은 두지 않으며, 직접 edge 일치와 interventional rollout을 별도 endpoint로 둔다.

MICrONS는 후보 검증 도메인이지 계약 시점의 승인된 실행 자산이 아니다. source 레인은 논문, 공개 자료 범위, functional/structural 정합과 실제 intervention 유무를 확인해야 한다.

### 5.3 단계 C: geometric memory 시험

학습, partial cue, interference, delayed recall이 있는 자료 또는 합성 recurrent task를 사용한다. 같은 encoder와 parameter budget에서 다음 모델을 비교한다.

1. activity-only recurrent baseline
2. weight/state-statistics baseline
3. Euclidean latent dynamics
4. single learned SPD metric
5. context-mixture SPD metric
6. metric 외에 drift를 명시한 directed geometric model

회상 성공뿐 아니라 cue에서 회상 trajectory까지의 접근 비용, basin 진입시간, encoding/recall trajectory의 quotient-level similarity를 평가한다.

## 6. 사전 고정 endpoint와 대조군

| 축 | 1차 endpoint | 필수 대조군 | 실패 조건 |
|---|---|---|---|
| 구조 복원 | held-out intervention NLL 또는 normalized rollout error | identity/empty, degree-matched, VAR/Neural ODE, 같은 parameter 수 latent model | 최선 비구조 baseline 대비 독립 seed CI가 0을 넘지 못함 |
| edge 비교 | directed edge AUPRC와 calibration | prevalence, degree-matched random, transitive-reduction sensitivity | 구조 일치가 baseline 수준이고 rollout 이득도 없음 |
| SCC 압축 | long-horizon error 대 compression ratio 곡선 | community, spectral, balanced random, learned soft modules | 같은 압축률에서 Pareto 우위가 없음 |
| causal module | module lesion/intervention effect prediction | SCC label shuffle, edge direction shuffle, cut/reset | SCC 정보 제거가 성능을 악화시키지 않음 |
| 기하 기억 | held-out recall NLL/accuracy와 basin-entry time | Euclidean, weight-only, activity-only, parameter-matched MLP | geometry의 독립 개선 CI가 0을 넘지 못함 |
| 재고착 | recall 전후 느린 상태 변화가 다음 recall을 예측 | no-update, replay-only, random update | geometry update가 후속 변화와 무관 |

통계 단위는 frame이 아니라 독립 graph, animal, session 또는 simulation seed다. 여러 endpoint를 탐색한 뒤 가장 좋은 것만 보고하지 않으며, 단계별 primary endpoint 하나와 secondary endpoint 목록을 preregistration에 고정한다.

## 7. 성공·중단 규칙

첫 단계의 성공은 AGI 또는 생물학적 기억의 증명이 아니다. 다음 세 조건을 모두 만족할 때만 두 번째 단계로 간다.

1. 합성 test graph에서 구조·문맥 분리 모델이 parameter-matched 최선 대조군보다 보지 않은 intervention rollout을 개선하고, 독립 seed bootstrap $95\%$ 신뢰구간이 0보다 크다.
2. SCC quotient가 같은 압축률의 최선 대조군보다 장기 예측 또는 intervention propagation에서 하나 이상의 사전 고정 primary endpoint를 개선한다.
3. metric 모델의 이득이 단순 parameter 증가에서 오지 않음을 rank, FLOP, train budget을 맞춘 대조와 geometry shuffle로 확인한다.

다음 결과가 나오면 해당 강한 경로를 중단하거나 좁힌다.

- 수동 관측에서 정확한 $G$를 유일 복원한다는 주장은 관측 동치 반례가 나오면 삭제하고 동치류 또는 예측 충분구조로 축소한다.
- SCC가 generic learned module보다 우위가 없으면 SCC를 필수 계산 단위가 아니라 해석 도구로 내린다.
- geometry가 Euclidean/weight-only baseline을 이기지 못하면 기억의 중심 기전이 아니라 시각화·regularization 후보로 축소한다.
- 구조 일치 없이 rollout만 좋아지면 이를 connectome recovery로 부르지 않고 effective predictive graph로 한정한다.
- recurrence 조작 없이 상관만 있으면 의식이나 기억의 필요조건을 주장하지 않는다.

## 8. 논문 프로그램

이번 run이 권고할 논문 순서는 다음 우선순위를 따른다.

1. **Paper A — Identifiability of Causal Recurrent Geometry.** 합성계에서 무엇이 복원 가능한지, 개입이 어떤 비식별성을 깨는지, $G$, $z_t$, $g_t$ 분리가 OOD rollout에 주는 효과를 검증한다.
2. **Paper B — SCC Quotients as Predictive Coarse-Graining.** SCC가 압축·개입·장기예측에서 generic clustering을 이기는지 검증한다.
3. **Paper C — Geometric Constraints for Memory Regeneration.** 학습·회상·간섭·재고착을 하나의 느린 geometry update로 묶되 weight-only baseline을 반드시 포함한다.
4. **Paper D — Blind Structure/Function Recovery.** 공개 connectome 자료의 구조를 숨겨 empirical bridge를 시험한다. 자료 적합성과 권한이 확인된 뒤에만 실행한다.
5. **Paper E — Recursive Self-Model.** 앞선 네 단계가 살아남은 뒤 self-prediction과 metacognition을 다룬다. 의식 충분조건은 범위 밖이다.

## 9. 외부 근거와 provenance 요구

source 레인은 사용자 노트의 MICrONS, engram, consciousness adversarial-test, light-microscopy connectomics 인용을 원 논문과 공식 자료에서 확인한다. 추가 문헌은 네트워크 식별성, nonlinear latent identifiability, causal discovery with interventions, neural manifold geometry, attractor memory와 reconsolidation에 한정하고 1차 논문 또는 공식 dataset 문서만 사용한다.

각 자료 후보에는 다음을 기록한다.

- 공개 접근 경로와 라이선스
- activity, anatomy, stimulus, behavior, intervention의 실제 필드
- 개체·session 수와 독립 통계 단위
- 구조와 기능의 정합 범위
- train/validation/test blind split 가능 여부
- 다운로드 크기와 계산 자원 경계

## 10. 구현 경계

이번 run은 제품 코드를 수정하지 않는다. 구현 레인은 연구 계약과 감사가 허용한 경우에도 preregistration, dataset manifest, synthetic benchmark specification까지만 제안하며, 대용량 다운로드나 model training은 별도 승인된 후속 run으로 분리한다.

기존 V15의 `UnifiedMetricState`, SCC atlas와 후속 V16/V17 결과는 재사용 가능성을 조사하되 다음 경계를 보존한다.

- 정적 metric은 비가역 전이 법칙이 아니다.
- finite metric graph는 continuum Riemannian manifold의 증거가 아니다.
- SCC 수 증가나 중첩은 지능 증가의 증거가 아니다.
- 합성 과제의 개선은 생물학, 기억, 의식 또는 AGI의 검증이 아니다.

## 11. 최종 산출물

최종 보고서는 다음을 제공해야 한다.

1. 18개 원 가설을 독립 주장으로 다시 분류한 판정표
2. 가장 먼저 실행할 단일 연구주제와 논문 제목
3. 식별성 no-go와 살아남는 좁은 명제
4. 최소 모델, baseline, ablation, primary endpoint와 blind split
5. 자료 적합성·출처 표
6. 12주 또는 이에 준하는 단계별 실행 로드맵
7. 후속 구현 run의 명확한 개시 조건

Status: COMPLETE
