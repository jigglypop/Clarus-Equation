Status: COMPLETE

# 인과적 재귀 기하 기억: AGI 가설의 연구 프로그램화

연구 기준일: 2026-08-16  
권고 프로그램명: **Causal Recurrent Geometry of Memory (CRGM)**  
첫 논문 권고 제목: **Identifiability of Causal Recurrent Geometry: Intervention-Aware Recovery and SCC Predictive Coarse-Graining**

## 초록

사용자 가설의 중심 통찰은 지능과 기억을 단순한 연결 가중치의 집합이 아니라, 비교적 안정한 유향 상호작용 구조, 문맥에 따라 변하는 동역학, 재귀 모듈과 상태 접근 기하의 결합으로 본다는 데 있다. 이 통찰은 연구 가치가 있지만, 관측 시계열만으로 정확한 잠재 인과그래프를 유일 복원한다는 주장, 같은 그래프에서 SCC 축약을 반복하면 자동으로 비자명한 계층이 생긴다는 주장, SPD 계량 하나가 방향성과 기억을 모두 결정한다는 주장은 각각 완전 반례를 가진다. 반례를 제거한 뒤에는 개입 조건부 구조 식별, SCC의 예측적 coarse-graining, controllability energy에 의한 기억 접근성이라는 세 개의 독립적이고 반증 가능한 연구축이 남는다. 첫 논문은 ground truth가 알려진 합성계에서 개입이 관측 동치류를 어디까지 깨는지 검증하고, 성공한 경우에만 SCC quotient의 예측 효용을 같은 압축률의 대조군과 비교한다. 기억 기하는 고정된 representation에서 최소 접근 energy가 weight·activity baseline 이후에도 부분단서 회상을 예측하는지 묻는 두 번째 논문으로 분리한다. 의식·자아·AGI 달성은 이 단계의 결론이 아니라 장기 동기다.

## 1. 최종 연구주제

### 1.1 핵심 질문

부분 관측과 알려진 개입이 있는 재귀 동역학에서 다음 세 대상을 분리해 복원할 수 있는가.

1. 비교적 불변인 유향 상호작용 구조 $\mathcal G$와 전이 법칙 $F$;
2. 시간에 따라 변하는 문맥 $z_t$;
3. cue에서 목표 recurrent trajectory로 들어가는 접근 비용 $g_t$.

그리고 이렇게 분리한 모델이 같은 정보와 계산량을 가진 monolithic recurrent model보다 보지 않은 intervention, 장기 rollout과 partial-cue recall을 더 잘 예측하는가.

권고하는 한 문장 가설은 다음과 같다.

> 선언된 관측·개입 조건 아래에서 식별한 유향 재귀 동역학을 먼저 고정하면, 그 동역학의 예측적으로 충분한 SCC quotient와 controllability-derived access geometry가 보지 않은 개입 및 기억 접근성을 추가로 예측할 수 있다.

이 문장은 “SCC가 기억이다” 또는 “기억이 곧 Riemannian metric이다”라는 존재론적 동일시가 아니다. 각 표현이 등록된 과제에서 추가 예측력을 갖는지를 묻는 운영적 가설이다.

### 1.2 연구의 새로움

SCC 정리, 선형계 식별, controllability Gramian 자체는 표준 도구다. 이 연구의 새로움은 도구의 이름에 있지 않고 다음 연결 규약에 있다.

- 관측-only 비식별성을 음성 정리로 먼저 고정하고 개입으로 깨지는 동치만 주장한다.
- topology, predictive quotient, access geometry와 biological memory를 서로 다른 typed object와 gate로 분리한다.
- SCC를 시각화가 아니라 compression-matched 장기예측 가설로 시험한다.
- metric을 자유로운 설명변수가 아니라 고정된 control interface의 energy로 시작한다.
- 합성 causal truth, 구조–기능 bridge, longitudinal memory를 하나의 데이터셋에 억지로 합치지 않고 순차적으로 삼각검증한다.

## 2. 수학적으로 닫힌 기반과 정확한 경계

### 2.1 살아남는 정리

| ID | 지위 | 내용 |
|---|---|---|
| T1 | [정리] | 유한 고정 유향 그래프의 maximal SCC 분할은 유일하며 condensation은 DAG다. |
| T2 | [정리: no-go] | 자유로운 잠재 similarity와 관측 Markov 동치 아래 observation-only exact latent support와 causal direction은 유일 식별되지 않는다. |
| T3 | [정리: 조건부] | 고정 좌표, 완전관측, 알려진 입력, hidden confounding 없음, full-rank design인 LTI 계에서는 $[A\ B]$가 유일하게 식별된다. |
| T4 | [정리: 필요충분] | 선형 quotient가 정확히 닫힐 조건은 $QA=\bar A Q$와 $QB=\bar B$다. SCC membership 자체는 이 조건을 보장하지 않는다. |
| T5 | [정리: gauge] | loss가 $W^TgW$만 보면 $W$와 $g$는 각각 식별되지 않는다. |
| T6 | [정리: no-go] | SPD metric의 대칭성만으로 비가역 drift, 시간방향 또는 유일 목표를 정할 수 없다. |
| T7 | [정리: 조건부] | finite-horizon controllability Gramian $W_T\succ0$이면 목표 $x$에 이르는 최소 control energy는 $x^TW_T^{-1}x$다. |

완전관측 LTI에서

$$
x_{t+1}=Ax_t+Bu_t
$$

이고 stacked design $Z$가 full row rank이면

$$
[A\ B]=YZ^T(ZZ^T)^{-1}.
\tag{1}
$$

식 (1)은 첫 benchmark의 positive anchor다. 부분관측 비선형계 전체의 식별 정리가 아니며, 우수한 예측 성능을 자동 보장하지도 않는다.

### 2.2 반복 SCC hierarchy의 교정

고정된 한 그래프에서

$$
\mathcal G
\xrightarrow{\operatorname{Cond}}
\operatorname{Cond}(\mathcal G)
$$

를 수행하면 오른쪽은 DAG다. 같은 edge semantics로 다시 SCC를 계산하면 모든 component가 singleton이므로 비자명한 hierarchy는 첫 단계 뒤 끝난다. 계층 가설을 살리려면 각 scale $\ell$에서 coarse trajectory와 선언된 시간창으로 새 effective-edge operator를 정의해야 한다.

$$
\mathcal G_\ell
\xrightarrow{\operatorname{SCC}}Q_\ell
\xrightarrow{\Phi_\ell(\text{coarse dynamics},W_\ell,\tau_\ell)}
\mathcal G_{\ell+1}.
\tag{2}
$$

$\Phi_\ell$, window $W_\ell$와 threshold $\tau_\ell$은 정리의 산출이 아니라 모델 선택이다. confirmation 전에 고정하고 선택 자유도를 회계해야 한다.

### 2.3 기하 기억의 좁은 정식화

첫 operational metric은 자유롭게 학습한 manifold가 아니라 고정된 dynamics와 control interface에서 유도한다.

$$
W_T=\sum_{k=0}^{T-1}A^kBB^T(A^k)^T,
\qquad
g_T=W_T^{-1}.
\tag{3}
$$

$W_T$가 SPD일 때 $x^Tg_Tx$는 최소 접근 energy다. 이는 recall accessibility의 후보 predictor이지 기억의 독립 물질이나 보편 기전이 아니다. $W_T$가 rank deficient이면 reachable subspace와 pseudoinverse로 범위를 좁힌다. $A,B$를 직접 사용하는 weight baseline이 같은 예측을 내면 metric은 유용한 재표현일 수는 있어도 독립 기억기전은 아니다.

## 3. 원 가설 H01--H18의 최종 연구 지위

원 가설은 버리지 않고 서로 독립적으로 실패할 수 있는 프로그램으로 보존했다.

| ID | 최종 지위 | 연구에서 남기는 형태 |
|---|---|---|
| H01 Dynamics-to-Structure | 부모 삭제 + 조건부 정리 + [미완성] | observation-only exact graph는 제외하고, 고정 chart의 interventional identification과 predictive equivalence class를 시험 |
| H02 Mechanism Separation | [미완성] | factorized $\mathcal G/F/z$와 matched monolithic model의 unseen-intervention rollout 비교 |
| H03 SCC Unit | [미완성] | SCC가 같은 압축률의 community·spectral·random·learned partition보다 유용한지 비교 |
| H04 SCC Hierarchy | 부모 삭제 + scale operator [공리] + [미완성] | scale별 새 effective-edge semantics가 있을 때만 계층 효용 시험 |
| H05 Structural/Effective Gap | [정의] | anatomy, observational effective graph와 causal intervention object를 별도 type으로 유지 |
| H06 Metric-over-Topology | [미완성] | topology와 weight 이후 access cost의 incremental prediction |
| H07 Metric Ensemble | 보편 우위 삭제 + [미완성] | 이질 문맥에서 matched single/low-rank metric과 비교 |
| H08 Geometric Memory | [미완성] | frozen representation의 invariant geometry가 recall을 추가 예측하는지 시험 |
| H09 Regeneration Constraint | [미완성] | partial cue에서 target recurrent trajectory와 basin으로의 접근성 |
| H10 Generative Recall | [미완성] | exact replay가 아니라 사전 고정한 trajectory invariant의 재생성 |
| H11 Accessibility Loss | [미완성] | trace 소실과 cue/access failure를 rescue intervention으로 분리 |
| H12 Geometry Rewrite | [미완성] | recall 후 느린 상태 update가 다음 recall을 인과적으로 바꾸는지 시험 |
| H13 Attractor-Geometry Unit | [미완성: 고위험] | local edge model보다 landscape 설명이 더 간결하고 intervention-predictive한지 비교 |
| H14 Recursive Self-Model | [미완성] | 앞 단계 통과 뒤 self-state prediction과 metacognitive calibration으로 분리 연구 |
| H15 Recurrent Conscious Access | [미완성: 별도 신경과학] | operational consciousness predicate와 recurrence 억제 실험이 있을 때만 재개 |
| H16 Consciousness Trajectory | 활성 결론 제외 | manifold와 의식을 동일시할 독립 bridge가 생기기 전까지 보류 |
| H17 Metric-Space Dynamics | 두 시간척도 [공리] + [미완성] | 빠른 $x_t$와 느린 $g_t,\mathcal G_t$가 matched one-state model보다 유용한지 비교 |
| H18 Domain-General Recovery | [미완성: 최종 replication] | freeze된 protocol이 합성계와 복수 독립 생물 자료에 재튜닝 없이 전이되는지 시험 |

현재 새로 확인된 empirical [예측]은 0개다. 이는 연구주제가 실패했다는 뜻이 아니라, 아직 seed·표본수·모델 hash와 confirmation split을 고정하지 않았으므로 결과를 미리 예측으로 승격하지 않았다는 뜻이다.

## 4. 첫 논문 설계

### 4.1 논문 명제

첫 논문은 H01--H04의 축소형만 다룬다.

> 알려진 node/module intervention과 사전 선언한 관측모형이 있을 때, factorized recurrent model이 관측 동치류 일부를 깨고 보지 않은 intervention rollout을 개선하는가. 그리고 그 복원 그래프의 SCC quotient가 exact/approximate lumpability를 가질 때 같은 압축률의 다른 partition보다 장기예측과 개입 전달에 유리한가.

기억, MICrONS causal recovery, 의식과 AGI를 이 논문의 성과로 합산하지 않는다.

### 4.2 최소 생성계

$$
x_{t+1}
=\tanh\!\left(A(z_t)x_t+Bu_t+b(z_t)\right)+\epsilon_t,
\qquad
y_t=Cx_t+\nu_t.
\tag{4}
$$

다음 세 관측 regime을 분리한다.

- known identity: $C=I$, node 좌표가 알려짐;
- known mask: 일부 node만 관측;
- unknown mix: 미지의 가역 혼합 $C$.

exact edge recovery는 known identity에서만 primary가 될 수 있다. unknown mix에서는 latent graph가 아니라 prediction/intervention equivalence class를 평가한다.

generator는 선형 VAR, nonlinear recurrent와 context-switch family를 포함하고 graph seed와 environment seed를 모두 blind split한다. train에 없던 intervention target, dose와 context 조합을 confirmation에 둔다.

### 4.3 비교군

| 축 | 제안 모델의 필수 대조 |
|---|---|
| 구조 | ridge/sparse VAR, parameter-matched MLP·GRU, graph transition model |
| 문맥 | switching state-space model, monolithic recurrent latent model |
| 압축 | community, spectral clustering, balanced random partition, learned soft module |
| 그래프 음성 대조 | degree-preserving rewire, direction shuffle, SCC-label shuffle |
| 개입 음성 대조 | target shuffle, time shuffle, intervention-tag 제거 |

모든 arm은 같은 encoder information, training example, optimizer budget과 blind split을 사용한다. parameter 수가 다르면 단일 점수가 아니라 performance–parameter/FLOP Pareto curve를 보고한다.

### 4.4 단계별 primary와 gate

| 단계 | primary endpoint | GO 조건 | STOP 조건 |
|---|---|---|---|
| A1 mechanism recovery | graph-seed paired confirmation intervention NLL 차이 | 95% paired CI가 0보다 크고 intervention shuffle에서 이득 소실 | matched baseline과 동률/열위, exact-edge 주장이 unknown mix에 의존 |
| A2 SCC quotient | error-versus-compression curve 면적 차이 | 최선 compression-matched partition보다 우위, lumpability/paired-microstate test 통과 | Pareto 우위 없음 또는 label·internal-state ablation에 민감하지 않음 |
| A3 scale hierarchy | scale별 held-out transfer·latency prediction | one-level 신호가 먼저 살아 있고 고정된 $\Phi_\ell$이 blind scale에서 재현 | window 한 bin 변화로 계층 전면 교체 또는 one-level과 동률 |

A1이 실패하면 A2의 topology 결과를 causal recovery의 증거로 쓰지 않는다. A2가 실패하면 SCC를 필수 계산단위가 아니라 해석용 topology로 축소한다.

## 5. 두 번째 논문: geometric memory

권고 제목은 **Geometric Constraints for Memory Regeneration: Control Energy, Partial Cues, and Recall Accessibility**다.

첫 질문은 “기억이 metric인가”가 아니라 다음과 같다.

> 고정하거나 cross-fit한 representation에서 학습 전후 controllability-derived access energy가 raw weight, activity, covariance와 equal-budget quadratic-form baseline 이후에도 partial-cue recall success, basin-entry time과 필요한 cue magnitude를 함께 예측하는가.

필수 kill test는 다음과 같다.

1. weight-only 모델이 같은 성능이면 독립 geometry 효능을 철회한다.
2. encoder를 바꿨을 때만 이득이 생기면 metric 귀속을 철회한다.
3. trace-matched SPD null 또는 eigenvector shuffle에서도 성능이 유지되면 geometry ordering 설명을 철회한다.
4. $W_T$가 singular이면 전체공간 Riemannian metric 문구를 철회한다.
5. manifold 변화가 있어도 attractor signature가 없으면 H08과 H09를 분리한다.

context metric mixture는 이 single-metric gate를 통과한 뒤에만 연다. context 수, rank와 gate family를 test에서 선택하지 않고 nested validation에서 고정한다.

## 6. 외부 데이터 전략

어느 한 공개 데이터셋도 anatomy, longitudinal dynamics, learning, neural intervention와 partial-cue recall을 모두 제공하지 않는다. 따라서 세 단계를 분리한다.

| 단계 | 자료 | 가능한 질문 | 금지되는 결론 |
|---|---|---|---|
| A causal identifiability | ground-truth synthetic controlled dynamics | intervention이 어떤 관측 동치를 깨는가 | 생물학·기억·AGI 일반화 |
| B structure/function bridge | MICrONS frozen matched subset | activity·stimulus에서 숨긴 anatomical synapse/graph feature를 예측하는가 | EM edge=causal edge, neural intervention, memory |
| C memory accessibility | 별도 longitudinal partial-cue/intervention 자료 또는 합성 memory task | access energy가 recall을 추가 예측하는가 | connectome 복원과 의식 |

[MICrONS](https://www.nature.com/articles/s41586-025-08790-w)은 한 마우스의 시각피질 기능 기록과 사후 EM을 정합하므로 blind structure/function benchmark에는 강하지만, neural-node intervention이나 기억 과제가 없다. 그러므로 anatomical-edge prediction과 stimulus-response prediction을 분리하고 single-mouse 내부 검증으로 보고한다.

[Science의 hippocampal engram EM 연구](https://doi.org/10.1126/science.ado8316)는 기억 관련 구조적 remodeling의 근거지만 소수 동물의 파괴적 cross-sectional SBEM이며 동일 개체의 pre/post dynamics와 partial-cue recall trajectory가 없다. [H01](https://doi.org/10.1126/science.adk4858)과 [LICONN](https://www.nature.com/articles/s41586-025-08985-1)은 구조 stress-test 또는 획득기술 근거이지 동역학·기억·인과 자료가 아니다.

[Cogitate adversarial test](https://www.nature.com/articles/s41586-025-08888-1)는 의식 이론의 예측을 엄격히 비교한 경계 근거지만 SCC나 recurrence의 충분조건을 지지하지 않는다. [Hu et al.](https://pubmed.ncbi.nlm.nih.gov/39605658/)은 학습과 neural-manifold separation의 연결을 지지하는 preprint지만 명백한 attractor signature를 찾지 못했으므로 H08 geometry와 H09 attractor를 합칠 근거가 아니다.

## 7. 12주 실행계획

| 주 | 목표 | 종료 판정 |
|---:|---|---|
| 1--2 | observation-only 반례, full-observation positive fixture, generator와 matched baseline | truth replay, no-go와 compute ledger 통과 |
| 3--4 | factorized $\mathcal G/F/z$ estimator, node/module intervention evaluator | unknown-mix exact-edge fail-closed, shuffle controls 통과 |
| 5--6 | validation 후 model freeze와 Phase A confirmation | GO-A 또는 STOP-A |
| 7--8 | SCC/community/spectral/random/learned partition의 동일 압축률 비교 | GO-B 또는 SCC 해석도구로 축소 |
| 9--10 | Gramian memory task와 Euclidean/weight/covariance/quadratic baselines | reachable-subspace와 analytic energy fixture 통과 |
| 11--12 | geometry family freeze와 Phase C confirmation | GO-C 또는 metric 설명 축소 |

confirmation을 연 뒤 generator, loss와 primary endpoint를 바꾸지 않는다. model author와 evaluator 역할은 가능하면 분리하고, 과거 V9/V16/V17 sealed seed를 재사용하지 않는다. 상세 실행 원장은 artifacts/twelve-week-roadmap.md와 artifacts/benchmark-preregistration-blueprint.md에 고정했다.

## 8. 논문 프로그램

1. **Paper A — Identifiability of Causal Recurrent Geometry.** 합성계의 개입 조건부 식별성과 음성 정리.
2. **Paper B — SCC Quotients as Predictive Coarse-Graining.** A의 freeze된 dynamics에서 SCC가 matched partition을 이기는지 독립 평가.
3. **Paper C — Geometric Constraints for Memory Regeneration.** 접근 energy, partial cue, interference와 recall.
4. **Paper D — Blind Structure/Function Recovery.** MICrONS 등에서 anatomy를 숨긴 bridge 검증.
5. **Paper E — Recursive Self-Model.** 앞선 네 단계가 통과한 뒤 self-prediction과 metacognitive calibration.

의식 가설은 Paper E와도 자동 연결하지 않는다. operational predicate, matched report/task와 neural recurrence manipulation이 있는 별도 신경과학 연구로 남긴다.

## 9. 기존 저장소와의 통합 경계

현재 저장소는 새 프로그램을 시작할 부품과 반례를 모두 갖고 있다.

- 즉시 재사용: clean finite SCC atlas, brain scale schema, V4의 paired-intervention generator와 leakage guard.
- 패턴만 재사용: manifest·receipt·one-shot confirmation, no-future/no-hidden tests.
- hash-bound characterization 뒤 재사용: dirty V15 metric, untracked V16/V17, current infinite-tail 후보.
- 재사용 금지: V9의 실패한 candidate architecture와 sealed seed, same-graph repeated-SCC hierarchy, strict metric-only signed memory 주장.

현재 nested SCC 코드·테스트에는 활성 정본과 CodeMap보다 넓은 infinite-tail certificate 후보가 존재한다. 이는 새 정리가 아니라 repository-level P0 불일치다. 이번 연구는 그 자산을 dependency와 근거에서 격리했으므로 설계 gate를 통과하지만, 후속 구현에서 이를 import하려면 먼저 exact file hash, norm/domain/schedule와 boundary assumption을 별도 closure run으로 조정해야 한다.

## 10. 예상 심사 공격과 답변

| 공격 | 연구 설계의 답 |
|---|---|
| “full-rank LTI 회귀는 알려진 결과다.” | 맞다. novelty는 표준 정리가 아니라 nonlinear/context OOD에서 식별 가능·불가능 영역을 intervention과 blind benchmark로 지도화하는 데 둔다. |
| “SCC가 좋아 보이는 것은 threshold 선택 효과다.” | threshold/window를 freeze하고 compression curve 전체, lumpability residual과 matched partition을 보고한다. |
| “generator가 정답 factorization을 심었다.” | adverse generator, topology/context family holdout, monolithic baseline과 misspecification regime을 둔다. |
| “MICrONS는 causal connectome이 아니다.” | 맞다. anatomy prediction과 stimulus response만 평가하고 causal endpoint는 합성 또는 실제 intervention 자료에 둔다. |
| “metric은 weight의 재표현이다.” | weight-only, frozen encoder, gauge/null shuffle을 통과하지 못하면 독립 기전 주장을 철회한다. |
| “이것이 왜 AGI 연구인가.” | AGI 달성의 증명이 아니라, 환경 개입에서 재사용되는 인과 구조·재귀 압축·기억 접근성을 분리 측정하는 기반 연구다. domain-general transfer는 마지막 H18 gate다. |

## 11. 최종 판정

이 가설은 **충분히 연구주제가 될 수 있다.** 다만 가장 강한 형태를 한 번에 주장하면 관측 비식별성, SCC 축약의 즉시 종료와 metric gauge 때문에 반례를 맞는다. 연구로 살아남는 핵심은 “하나의 이론이 전부 설명한다”가 아니라 다음 세 문장을 순서대로 시험하는 것이다.

1. 개입은 관측만으로 구분할 수 없는 recurrent dynamics의 일부를 식별 가능하게 만든다.
2. 식별된 dynamics에서 SCC quotient는 lumpability와 blind prediction으로 효용을 입증해야 한다.
3. 고정된 dynamics와 control interface에서 유도한 접근 geometry는 weight·activity 이후 recall의 추가 predictor여야 한다.

형식 gate는 이 연구주제와 반증 설계에 대해 PASS다. 이는 구현, empirical confirmation, 기억기전, 의식 또는 AGI 승인이 아니다. 후속 구현의 개시 조건은 실제 preregistration, fresh seed와 model/evaluator hash, parameter/FLOP ledger, unopened confirmation receipt, 그리고 필요 시 infinite-tail repository reconciliation이다.
