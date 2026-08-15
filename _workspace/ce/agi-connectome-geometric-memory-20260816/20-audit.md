# 인과적 재귀 기하 복원 형식 지위 감사

Status: COMPLETE

Gate: PASS

이 PASS는 **연구주제·주장 범위·반증 설계를 확정하는 현재 이론/설계 run**에만 적용된다. 구현, confirmation split 개봉, MICrONS 인과복원, 기억기전, 의식, 자아 또는 AGI 달성을 승인하지 않는다. 완전 반례가 있는 넓은 부모 문장은 아래와 같이 활성 결론에서 제거했고, 실행 전 조건이 필요한 항목은 정리나 예측으로 올리지 않고 `[미완성]`으로 남겼다.

## 1. 감사 입력과 Claim ID 정규화

감사 입력은 `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`와 직접 연결된 `artifacts/hypothesis-decomposition.md`, `artifacts/benchmark-preregistration-blueprint.md`, `artifacts/repository-reuse-map.md`, `artifacts/twelve-week-roadmap.md`다. 네 root 입력 레인은 모두 `Status: COMPLETE`였다.

계약의 `CGM-H1`--`CGM-H6`와 사용자 원문 분해의 `H1`--`H18`은 번호가 같지만 뜻이 다르다. 이 번호 충돌을 그대로 두면 증거와 kill test가 다른 주장을 같은 ID로 인용하게 된다. 따라서 이 감사 이후에는 다음 namespace를 쓴다.

| 감사 정본 ID | 원 위치 | 의미 |
|---|---|---|
| `CGM-D1`, `CGM-N1`, `CGM-X1`, `CGM-X2` | `00-contract.md:44-53` | 정의/정리 경계, 관측-only 부모 주장, 제외 주장 |
| `CGM-C1`--`CGM-C6` | 계약의 옛 `CGM-H1`--`CGM-H6`, `00-contract.md:46-51` | 이번 run의 여섯 최소 비교 주장 |
| `CGM-H01`--`CGM-H18` | 사용자 원문 분해 `H1`--`H18`, `artifacts/hypothesis-decomposition.md:9-26` | 원 가설 18개를 보존하는 장기 연구 원장 |

향후 문서에서 namespace 없이 `CGM-H1`처럼 쓰는 것은 허용하지 않는다. 이번 감사는 이 crosswalk로 충돌을 닫았으므로 별도 열린 P1으로 세지 않는다.

## 2. P0 완결성 판정

| P0 ID | 반례·경계 | 활성 부모 범위에서의 조치 | 살아남는 좁은 범위 | 근거 |
|---|---|---|---|---|
| `A-P0-01` | latent similarity와 Gaussian causal-direction 관측 동치 | 관측 시계열만으로 정확한 latent chart와 edge support/direction을 유일 복원한다는 부모 문장 삭제 | 고정 chart, 완전관측, 알려진 개입, full-rank design의 LTI 계수 식별; 그 밖에는 동치류 또는 predictive effective graph | `11-math.md:75-168` |
| `A-P0-02` | true model이 대조군에 포함되거나 문맥이 상수이면 strict improvement가 0 | 개입이 존재하면 factorized model이 항상 더 낫다는 문장 삭제 | frozen split의 OOD rollout 비교 가설 | `11-math.md:168`, `12-routes.md:23-39` |
| `A-P0-03` | 하나의 SCC 안에서도 같은 aggregate state가 다른 다음 aggregate state를 가짐 | SCC membership 또는 block sum이 자동 predictive sufficient라는 문장 삭제 | `QA=\bar A Q`, `QB=\bar B`를 만족하는 exact-lumpable quotient; 그 밖에는 residual을 측정하는 경험 가설 | `11-math.md:170-198` |
| `A-P0-04` | maximal SCC condensation은 DAG이고 다음 SCC는 singleton | 같은 edge semantics의 반복 SCC가 비자명 hierarchy를 자동 생성한다는 문장 삭제 | scale마다 사전 고정한 새 effective-edge operator `\Phi_\ell`를 두는 모델 | `11-math.md:51-73`, `12-routes.md:77-93` |
| `A-P0-05` | `W'=SW`, `g'=S^{-T}gS^{-1}`가 같은 관측 비용을 만듦 | joint encoder 성능을 metric의 독립 기억효과로 귀속하는 문장 삭제 | frozen/cross-fitted chart의 invariant cost 또는 독립 측정 energy 비교 | `11-math.md:200-226` |
| `A-P0-06` | 모든 context metric이 같으면 mixture는 단일 metric과 동일 | context mixture가 항상 strict 우위라는 문장 삭제 | 이질 문맥, matched rank/dof/FLOP, outcome-independent gate의 OOD 비교 | `11-math.md:296-304`, `12-routes.md:95-111` |
| `A-P0-07` | SPD quadratic form과 Riemannian distance는 방향 반전에 대칭 | metric 하나가 비가역 전이·시간방향·유일 목표를 결정한다는 문장 삭제 | 별도 drift, control, oriented state 또는 time orientation을 둔 directed model | `11-math.md:227-242` |
| `A-P0-08` | 그래프 공리에는 의식 predicate와의 bridge가 없어 SCC가 있어도 predicate를 false로 둘 수 있음 | SCC 존재가 의식의 충분조건이라는 활성 결론 삭제 | 독립 operational predicate와 조작 가능한 bridge를 가진 별도 연구만 재개 가능 | `11-math.md:48`, `11-math.md:321` |
| `A-P0-09` | MICrONS의 EM edge는 causal/effective edge가 아니며 node intervention과 memory protocol이 없음 | MICrONS를 causal-connectome 또는 기억 ground truth로 쓰는 범위 삭제 | 고정 snapshot에서 anatomical-edge prediction과 stimulus-response prediction을 분리한 단일-mouse structure/function bridge | `10-sources.md:12-13`, `10-sources.md:24-25`, `10-sources.md:33-39` |

완전 반례를 받은 부모 범위 아홉 개는 활성 주장으로 남지 않는다. 반례를 피하는 좁은 명제는 새 전제와 정의역을 함께 기록했으며, 원래 부모의 성공으로 계산하지 않는다.

## 3. 핵심 계약 주장 감사

| Claim ID | 계약 시작 지위 | 실제 지위 | 활성 범위와 판정 |
|---|---|---|---|
| `CGM-D1` | [정리 후보] | **[정리]** | 유한 고정 유향 그래프에서 maximal SCC 분할은 유일하고 self-edge를 제거한 condensation은 DAG다. 동치관계와 cycle contradiction 증명이 닫혔다(`11-math.md:51-64`). 같은 semantics의 반복 hierarchy는 이 정리의 산출이 아니다. |
| `CGM-N1` | [미완성] | **부모 삭제 + 대체 [정리]** | 관측-only exact latent graph 유일복원은 식 (5)--(8)의 완전 반례로 삭제한다. 대체 no-go와 full-rank LTI 조건부 식별 정리만 활성이다(`11-math.md:75-168`). |
| `CGM-C1` | [미완성] | **[미완성: 경험 비교]** | 알려진 개입과 선언한 관측모형 아래 `G/F`와 `z` 분리가 unseen-intervention rollout을 개선하는지 시험할 수 있다. strict 우위 정리는 없다. 실제 seed, 표본수, model hash가 아직 고정되지 않아 `[예측]` 승격 전이다(`artifacts/benchmark-preregistration-blueprint.md:5`). |
| `CGM-C2` | [미완성] | **[미완성: 경험 비교]** | SCC quotient를 같은 압축률의 community/spectral/random/learned partition과 비교한다. SCC의 자동 충분성은 삭제했고 lumpability residual과 paired-microstate rollout을 함께 측정한다. |
| `CGM-C3` | [미완성] | **[미완성: 경험 bridge]** | frozen representation에서 Gramian/invariant cost가 weight·activity baseline 이후 recall을 추가 예측하는지만 남는다. joint encoder의 metric 귀속은 금지한다. |
| `CGM-C4` | [미완성] | **[공리: 모델 선택] + [미완성]** | 기억을 cue에서 recurrent trajectory를 재생성하는 느린 제약으로 모델링할 수는 있다. 그것이 보편 기억기전이라는 결론은 없으며 partial cue, delayed recall, causal update가 필요하다. |
| `CGM-C5` | [미완성] | **[미완성: 경험 비교]** | heterogeneous context와 matched dof에서만 mixture 대 single metric OOD 비교를 허용한다. 항상 우위인 부모는 삭제한다. |
| `CGM-C6` | [미완성] | **[미완성: blind replication]** | 같은 복원 규약의 합성/신경 교차도메인 성공은 마지막 replication gate다. domain별 retuning과 독립단위를 회계해야 한다(`12-routes.md:113-121`). |
| `CGM-X1` | 활성 제외 | **부모 삭제** | SCC와 의식을 잇는 bridge가 없고 이번 endpoint에도 의식 predicate가 없다. Cogitate는 경계 근거일 뿐 SCC 충분조건의 근거가 아니다(`10-sources.md:113`). |
| `CGM-X2` | 활성 제외 | **부모 삭제 + no-go [정리]** | 정적 SPD metric은 방향·비가역 drift·source-free 유일 목표를 주지 못한다. 방향을 주는 별도 typed object가 필요하다. |

## 4. 사용자 원 가설 H01--H18 판정

| Claim ID | 실제 지위 | 활성 연구 범위 | 삭제·축소 또는 재개 조건 |
|---|---|---|---|
| `CGM-H01` Dynamics-to-Structure | 부모 삭제 + 조건부 [정리] + [미완성] | fixed-chart interventional LTI 식별 정리와, 더 넓은 nonlinear predictive recovery 비교 | observation-only exact `G`는 삭제. 부분관측/unknown mix에서는 동치류 endpoint를 먼저 선언해야 함 |
| `CGM-H02` Mechanism Separation | [미완성] | factorized `G/F/z` 대 monolithic matched baseline의 unseen-intervention rollout | `z`의 관측/숨은 지위, interaction feature와 design rank를 freeze해야 함 |
| `CGM-H03` SCC Unit | [미완성] | SCC가 compression-matched 대조군보다 predictive coarse-graining에 유용한가 | SCC=충분통계는 삭제. `QA=\bar A Q` residual, paired microstate, lesion/propagation을 함께 시험 |
| `CGM-H04` SCC Hierarchy | 부모 삭제 + [공리: scale operator] + [미완성] | scale별 새 edge semantics가 해당 horizon의 양방향 intervention transfer를 예측하는가 | same-semantics 반복은 삭제. `\Phi_\ell`, window, threshold와 선택 수를 사전 고정한 뒤에만 재개 |
| `CGM-H05` Structural/Effective Gap | [정의] | anatomical graph `G_s`, observational/effective graph `G_e`, causal intervention object를 별도 type으로 유지 | 두 graph를 같은 정답으로 채점하지 않음. MICrONS에서는 anatomy endpoint만 허용 |
| `CGM-H06` Metric-over-Topology | [미완성] | topology/weight 이후 operational access cost의 incremental prediction | representation, metric family, direction source와 null model을 먼저 고정 |
| `CGM-H07` Metric Ensemble | 부모의 항상-우위 범위 삭제 + [미완성] | H06 통과 뒤 context mixture 대 matched single/low-rank metric | context 이질성, gate independence, `K/rank` nested validation 필요 |
| `CGM-H08` Geometric Memory | [미완성] | 학습 전후 invariant access geometry가 recall을 추가 예측하는가 | joint encoder/metric 귀속 삭제. weight/activity baseline 이후 독립값이 남아야 함 |
| `CGM-H09` Regeneration Constraint | [미완성] | partial cue에서 target recurrent trajectory/basin으로의 접근성 | manifold geometry와 attractor evidence를 분리. Hu preprint는 attractor 증거가 아님(`10-sources.md:84-91`) |
| `CGM-H10` Generative Recall | [미완성] | exact replay가 아니라 사전 고정 quotient/trajectory invariant의 재생성 | 어떤 invariant인지, 허용 변형과 null distribution을 confirmation 전에 고정 |
| `CGM-H11` Accessibility Loss | [미완성] | trace 보존과 cue/access 실패를 강한 cue·control 조작으로 구분 | trace의 독립 측정과 rescue intervention이 있는 memory dataset 필요 |
| `CGM-H12` Geometry Rewrite | [미완성] | recall 뒤 느린 상태 변화가 다음 recall을 예측/변경하는가 | no-update, replay-only, random-update 대조와 longitudinal 반복측정 필요 |
| `CGM-H13` Attractor-Geometry Unit | [미완성: 고위험] | local edge/weight model보다 landscape description이 더 간결하고 개입예측적인가 | attractor의 존재·basin·stability부터 독립 검정; 실패하면 해석 단위로만 유지 |
| `CGM-H14` Recursive Self-Model | [미완성] | 앞선 복원 연구가 통과한 뒤 self-state prediction과 metacognitive calibration | self-state의 조작 가능한 정의, matched ablation과 독립 task 필요; Paper E 전에는 비활성 |
| `CGM-H15` Recurrent Conscious Access | [미완성: 별도 신경과학 가설] | recurrence 인과 억제와 matched task/report의 관계 | AGI 성능과 분리하고 의식 operational predicate·neural intervention이 있을 때만 재개 |
| `CGM-H16` Consciousness Trajectory | [미완성], 활성 결론 제외 | 없음 | deforming manifold와 의식을 동일시할 bridge와 반증 가능한 predicate가 없다. 정의와 독립 조작이 생길 때까지 제외 |
| `CGM-H17` Metric-Space Dynamics | [공리: 두 시간척도 모델] + [미완성] | 빠른 `x_t`와 느린 `g_t/G_t` 분리가 matched one-state model보다 예측적인가 | 두 상태의 update law, gauge, time-scale separation, parameter budget을 명시해야 함 |
| `CGM-H18` Domain-General Recovery | [미완성: 마지막 replication] | 한 번 freeze한 protocol이 합성 및 복수 독립 biological dataset에서 재현되는가 | domain-specific encoder/loss/tuning 변경을 숨기지 않고 independent animal/session을 확보해야 함 |

## 5. 정리·공리·산출·예측 회계

### 5.1 닫힌 수학 명제

| ID | 지위 | 명제와 한계 |
|---|---|---|
| `T1` | [정리] | finite fixed directed graph의 maximal SCC partition 유일성과 condensation DAG |
| `T2` | [정리: no-go] | free latent similarity/관측 Markov 동치 아래 observation-only exact latent support/direction 비식별 |
| `T3` | [정리: 조건부] | fixed chart, full observation, known input, no hidden confounding, full-rank `Z`에서 LTI `[A B]` 유일 식별 |
| `T4` | [정리: 필요충분] | 선형 block-sum quotient closure와 `QA=\bar A Q`, `QB=\bar B`의 동치 |
| `T5` | [정리: gauge] | joint factorization에서 `W^TgW`만 보는 loss는 `W`와 `g`를 각각 식별하지 못함 |
| `T6` | [정리: no-go] | SPD metric의 부호대칭/거리대칭만으로 directed irreversible dynamics를 정할 수 없음 |
| `T7` | [정리: 조건부] | finite-horizon controllability와 `W_T\succ0` 아래 minimum control energy가 `x^TW_T^{-1}x` |

`T7`에서 `g_T=W_T^{-1}`은 고정 `A,B,T`, control norm과 chart를 대입해 얻는 **[산출]**이다. recall이나 생물학적 effort와 동일하다는 정리가 아니다. rank가 부족하면 global SPD 산출은 없고 reachable subspace의 pseudoinverse cost만 남는다(`11-math.md:244-291`).

### 5.2 숨은 공리·외부 선택

다음 13개는 정리의 결론이 아니라 반드시 기록할 공리 또는 설계 선택이다.

1. graph edge의 방향·support·sampling interval과 관측 chart;
2. anatomy, effective prediction, causal intervention graph의 서로 다른 type;
3. intervention target, dose, timing, off-target와 reset 의미론;
4. LTI/noise class와 hidden confounder 부재;
5. excitation/full-rank 하한;
6. SCC estimator, threshold, time window와 transitive-reduction 규칙;
7. quotient state map `Q`와 block aggregate;
8. scale별 effective-edge operator `\Phi_\ell`와 schedule;
9. encoder freeze/cross-fit 또는 gauge-fixing 규칙;
10. control interface, norm, horizon와 reachable target;
11. context label/gate의 outcome independence;
12. parameter, rank, FLOP, train-budget matching;
13. independent graph/animal/session 단위, blind split와 multiple-selection 회계.

이 중 하나라도 사후 선택되면 대응 empirical claim은 무입력 산출이나 정리가 아니다.

### 5.3 수량 요약

| 항목 | 수 | 해석 |
|---|---:|---|
| 감사 record | 28 | core 계약 10개와 사용자 원 가설 18개; 의미 중복은 namespace crosswalk로 분리 |
| 닫힌 정리/no-go | 7 | `T1`--`T7`; 모두 정의역과 전제가 붙은 좁은 명제 |
| 명시할 공리/선택 | 13 | 위 5.2의 설계 자유도 |
| 원 가설 중 empirical/bridge 미완성 성분 | 17 | H05의 graph-type 구분만 [정의]; 나머지는 전부 적어도 하나의 [미완성] 성분 보유 |
| 활성 부모 범위에서 삭제 | 9 | `A-P0-01`--`A-P0-09`; 좁은 대체 명제는 별도 ID로 보존 |
| 현재 획득한 [예측] | 0 | 청사진은 실제 seed·표본수·model hash를 아직 freeze하지 않았음 |
| 사전등록 후 [예측] 승격 후보 | 3 | Phase A intervention NLL, Phase B error-compression AUC, Phase C partial-cue recall endpoint(`artifacts/benchmark-preregistration-blueprint.md:133-166`) |

## 6. 첫 논문의 허용 범위

권고 제목은 **Identifiability of Causal Recurrent Geometry: Intervention-Aware Recovery and SCC Predictive Coarse-Graining**이다. 다만 하나의 통계적 성공으로 세 주장을 묶지 않고 다음 순서를 강제한다.

| 단계 | 활성 명제 | primary/secondary | 통과와 중단 |
|---|---|---|---|
| `A1` | `T2`, `T3`, `CGM-H01`, `CGM-H02`의 조건부 축소형 | primary: graph-seed paired confirmation intervention NLL 차이 | full-rank coefficient 복원과 unseen target/dose rollout이 matched baseline을 이겨야 `GO-A`; 아니면 exact-graph 서술을 중단 |
| `A2` | `T1`, `T4`, `CGM-H03` | `GO-A` 뒤 분리한 primary: compression ratio 전체의 error curve 면적 | lumpability residual과 paired-microstate test가 나쁘거나 최선 partition 대조에 Pareto 우위가 없으면 SCC를 해석 topology로 내림 |
| `A3` | `CGM-H04`의 scale-dependent 축소형 | exploratory/후속 원고 | one-level SCC 신호가 살아 있고 `\Phi_\ell` schedule이 사전 고정된 뒤에만 개시 |

따라서 첫 실행은 `R1`이고, `R2`는 freeze된 `A,B`를 소비하는 조건부 두 번째 stage다(`12-routes.md:119-121`). `R3` memory energy는 수학 fixture로는 유지하되 첫 논문의 성과로 합치지 않는다. H06--H13의 기억 기하, H14의 self-model, H15--H16의 의식, H18의 domain-general 주장은 첫 논문의 결론에서 제외한다.

## 7. 데이터와 저장소 경계

| 대상 | 현재 허용 지위 | 금지 범위 | 재개 조건 |
|---|---|---|---|
| synthetic Phase A | 첫 논문의 유일한 causal-identifiability ground truth | generator를 본 test tuning, 과거 sealed seed 재사용 | fresh graph/environment seed, known intervention, frozen manifest/evaluator |
| MICrONS | 조건부 anatomy/function bridge | causal edge truth, held-out neural intervention, memory/biological replication | v1300/table/root IDs/query date, anatomy-blind evaluator, single-mouse 한계와 stimulus-response/anatomy endpoint 분리 |
| engram EM | cross-sectional structural motivation | same-animal pre/post dynamics, partial-cue trajectory | raw repository/license 확인 및 별도 longitudinal memory dataset |
| Cogitate | 의식 이론의 경계 근거 | SCC/recurrence 충분조건 근거 | 별도 operational consciousness study 없이는 재개 불가 |
| current infinite-tail code/test | **격리** | 새 CGM 전제·증명·정본으로 인용 | base/working-tree characterization, norm/domain/schedule 감사, hash-bound tests, 별도 closure decision(`artifacts/repository-reuse-map.md:20-55`) |
| dirty/untracked V15/V16/V17 | hash-bound predecessor 또는 대조 패턴 | stable API, 승격된 metric-memory 증거 | exact file hash characterization과 isolated adapter |

저장소의 infinite-tail 구현과 활성 정본/CodeMap의 불일치는 여전히 repository-level P0다. 그러나 현재 run은 그 코드를 실행·수입하거나 그 정리를 전제로 삼지 않고 격리했으므로 **현재 연구주제 설계 gate의 열린 P0는 아니다**. 해당 자산을 후속 구현 dependency로 여는 순간 이 P0가 다시 활성화되고 구현 gate는 자동 `BLOCKED`다.

## 8. Referee 공격과 출판 준비도

첫 논문 개념의 등급은 **needs-closure**다. 수학적 경계와 연구 질문은 정리됐지만 실제 preregistration과 결과가 없어 arXiv-ready empirical paper가 아니다.

| 공격 | 현재 방어 | 출판 전 필요한 폐쇄 |
|---|---|---|
| 1. “LTI full-rank 식별은 표준 회귀이고 AGI 기여가 아니다.” | novelty를 표준 정리가 아니라 nonlinear/context OOD에서 어떤 동치가 intervention으로 깨지는지와 fail-closed benchmark에 둠 | 새로운 비선형 식별 조건 또는 엄격한 negative/positive empirical map을 결과로 제시 |
| 2. “SCC가 좋아 보이는 것은 threshold/window/compression 선택 효과다.” | threshold/window를 freeze하고 전체 compression curve, lumpability residual, matched partitions와 label shuffle을 요구 | 선택 수 ledger, nested validation, blind paired-microstate confirmation |
| 3. “합성 generator가 제안 factorization을 정답으로 심었다.” | 여러 topology/context family와 monolithic/graph baselines, graph/environment OOD를 분리 | generator family holdout, misspecification regime, independent evaluator와 adverse generator |
| 4. “MICrONS로 causal graph나 기억을 검증하지 못한다.” | MICrONS를 single-mouse anatomical-edge/stimulus-response bridge로만 제한 | causal 주장은 synthetic/새 intervention 자료에 남기고 anatomy bridge를 별도 원고/section으로 보고 |
| 5. “geometry memory는 `A,B`의 재표현이며 encoder gauge다.” | Gramian energy를 derived operational readout으로 명시하고 weight-only/gauge shuffle을 필수화 | 별도 Paper C에서 frozen representation, longitudinal recall, incremental-value와 causal perturbation을 통과 |

출판 재개 조건은 (a) seed·표본수·alpha/CI·model/evaluator hash를 고정한 실제 preregistration, (b) negative fixtures와 no-future/no-hidden test, (c) parameter/FLOP ledger, (d) unopened confirmation receipt, (e) 주장별 failure result를 그대로 보존하는 것이다.

## 9. Gate 최종 판정

- **열린 P0: 0 (현재 이론/설계 범위).** 아홉 부모 범위는 삭제·축소되었고 repository P0 자산은 dependency에서 격리됐다.
- **열린 P1: 0 (현재 이론/설계 범위).** 실제 preregistration, 데이터 확보와 구현은 미완성을 숨기지 않은 후속 run의 개시 조건이지 현재 정리의 누락된 증명이 아니다.
- **P2 경계:** Claim ID namespace를 final report와 후속 manifest에서 유지하고, empirical 결과가 생기기 전 `[예측]` 또는 `[경험식]`으로 승격하지 않는다.
- **허용:** 첫 연구주제, 정리/no-go, 반증가능한 benchmark 설계, 12주 로드맵의 최종 집필.
- **불허:** 제품/정본 수정, 대용량 자료 다운로드, confirmation 개봉, current infinite-tail 승격, 기억·의식·AGI 달성 주장.

따라서 좁혀진 연구 프로그램은 형식 지위와 근거가 일치하며 `Gate: PASS`다. 후속 구현 gate는 새 manifest와 repository reconciliation을 별도로 통과해야 한다.

