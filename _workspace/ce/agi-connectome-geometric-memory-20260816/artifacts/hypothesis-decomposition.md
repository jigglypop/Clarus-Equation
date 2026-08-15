# 사용자 가설 H1--H18 연구 분해 초안

Status: COMPLETE

이 문서는 사용자 원문의 가설을 삭제하지 않고, 서로 독립적으로 실패할 수 있는 검증 단위로 분해한다. 최종 지위는 `20-audit.md`가 판정한다.

| 원 가설 | 연구 단위 | 현재 취급 | 가장 가까운 kill test |
|---|---|---|---|
| H1 Dynamics-to-Structure | observation/intervention에서 $G$ 복원 | 강한 observation-only 유일복원은 반례 대상; 개입 조건부 형태를 1차 연구로 보존 | 관측 동치인 서로 다른 $G$가 같은 $X$ 분포 생성 |
| H2 Mechanism Separation | $G,F$와 $z_t$ 분리 | 1차 핵심 경험 가설 | monolithic latent model이 OOD intervention에서 동률 또는 우위 |
| H3 SCC Unit | SCC의 predictive utility | 1차 핵심 경험 가설 | 같은 압축률의 learned/community module이 동률 또는 우위 |
| H4 SCC Hierarchy | scale별 recurrent quotient | 단순 반복 condensation과 scale-dependent 재학습을 분리 | 고정 edge semantics에서는 첫 condensation 뒤 DAG라 추가 SCC가 singleton |
| H5 Structural/Effective Gap | anatomy $G_s$와 effective $G_e$ 구분 | 연구 설계의 필수 정의 | 두 graph를 같은 정답으로 채점해 결론이 뒤집힘 |
| H6 Metric-over-Topology | topology 위 접근 비용 | 2차 핵심 가설; metric operational definition 필요 | 동일 parameter budget Euclidean/weight model이 동률 또는 우위 |
| H7 Metric Ensemble | context별 basis metric | H6 통과 뒤 확장 | single metric 또는 shared low-rank model이 동률 또는 우위 |
| H8 Geometric Memory | 학습 전후 접근 geometry 변화 | 3차 핵심 가설; 인과 기전이 아니라 operational model로 시작 | weight/activity feature를 조건화하면 geometry의 독립 예측력이 사라짐 |
| H9 Regeneration Constraint | cue에서 attractor 재형성 | 3차 핵심 가설 | trace는 존재해도 cue-energy/접근성이 recall을 예측하지 못함 |
| H10 Generative Recall | exact replay가 아닌 invariant 재생성 | secondary endpoint | encoding/recall 사이 사전 고정 불변량이 없음 |
| H11 Accessibility Loss | 망각과 trace/accessibility 분리 | 3차 실험 가설 | 강한 cue나 control-energy 조작으로 회복되지 않고 trace 자체만 소실 |
| H12 Geometry Rewrite | 회상 뒤 느린 상태 갱신 | 후속 인과 가설 | no-update/replay control과 다음 recall 변화가 같음 |
| H13 Attractor-Geometry Unit | 개별 edge보다 landscape가 최소 단위 | 고위험 확장 | edge/weight local feature가 같은 intervention 효과를 더 간결하게 설명 |
| H14 Recursive Self-Model | self-state 예측 폐쇄 | Paper E 후보 | self-state 제거가 metacognitive calibration을 악화시키지 않음 |
| H15 Recurrent Conscious Access | recurrence와 conscious access | 별도 신경과학 가설; AGI 성능과 분리 | recurrence 인과 억제 뒤 matched task/report가 유지 |
| H16 Consciousness Trajectory | deforming manifold와 의식 동일시 | 활성 결론 제외 | 조작 가능한 독립 operational definition 부재 |
| H17 Metric-Space Dynamics | $x_t$와 $g_t$의 두 시간척도 | 형식 모델 후보 | 단일-state recurrent model이 같은 예측을 더 적은 자유도로 달성 |
| H18 Domain-General Recovery | 여러 도메인의 공통 복원 규약 | 마지막 일반화 가설 | 도메인별 encoder·loss·tuning을 바꾸지 않으면 성능 붕괴 |

## 가장 먼저 묶을 주장

첫 연구는 H1의 조건부 축소형, H2, H3와 H4의 scale-dependent 축소형만 묶는다. 제목 후보는 **Identifiability of Causal Recurrent Geometry: Intervention-Aware Recovery and SCC Predictive Coarse-Graining**이다. 이 연구의 결과가 살아남기 전에는 H8 이후의 기억 기하와 H14 이후의 자기모델을 같은 논문의 성과로 세지 않는다.

두 번째 연구는 H6, H8, H9, H10, H11을 하나의 operational metric 아래 묶는다. 가장 명확한 첫 metric 후보는 regularized inverse controllability Gramian 또는 사전 고정된 local transition-cost metric이다. encoder와 metric을 동시에 자유롭게 학습해 생기는 좌표 gauge를 피하려면 representation을 freeze하거나 chart-covariant 평가를 함께 둔다.

## 단순 반복 SCC hierarchy의 경계

고정된 $G$의 maximal SCC를 축약하면 $\operatorname{Cond}(G)$는 DAG다. 따라서 같은 edge semantics로 다시 maximal SCC를 구하면 모든 SCC는 singleton이고 비자명한 hierarchy는 끝난다. 사용자 원문의 반복식이 살아남으려면 각 scale $k$에서 coarse trajectory로부터 새로운 effective edge relation $E_{k+1}$을 정의해야 한다.

$$
G_k
\xrightarrow{\operatorname{SCC}}
Q_k
\xrightarrow{\text{coarse dynamics + declared window}}
G_{k+1}.
$$

가운데의 새 edge 정의는 정리가 아니라 모델 선택이며, time window, threshold, directionality와 intervention semantics를 사전에 고정해야 한다.

## 기억 기하의 경계

관측 vector field만으로 potential과 metric을 동시에 유일하게 정할 수 없고, 자유로운 latent diffeomorphism 아래 같은 거리·trajectory를 여러 $(h,g)$ 표현이 재현할 수 있다. 그러므로 “기억은 metric이다”를 직접 시험하지 않는다. 대신 다음 좁은 문장을 시험한다.

> 사전 고정된 representation과 metric family에서 학습 전후의 접근비용 변화가 weight/activity baseline 이후에도 partial-cue recall과 intervention energy를 추가로 예측하는가.

이 문장이 실패해도 기억·회상 현상 자체는 부정되지 않으며, metric 설명만 축소된다.

