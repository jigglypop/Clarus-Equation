# 학습된 계산 기하와 수면 재정렬: 증명 가능한 핵심과 AGI 연구 프로그램

Status: COMPLETE

## 초록

학습을 “자주 필요한 상태 사이의 유효거리를 줄이는 과정”으로 보는 직관을 유한 유향 계산 그래프에서 검토했다. 양의 비용 간선 하나를 추가할 때 모든 쌍 최단거리의 정확한 갱신식과 영향을 받는 쌍의 필요충분조건은 증명된다. 그러나 shortcut 추가와 리만 계량 변형의 구조적 동일시, raw 연결 변화에서 유일한 계량 변화의 산출, 인지시간의 보편적 길이/속도 공식은 각각 완전 반례로 무너진다. 살아남는 이론은 연결, 활동, gain, 지연과 문맥을 사전 고정한 비용 사상에 넣고 그 산출이 이후 궤적을 추가 예측하는지 묻는 operational geometry다. 수면 자료는 국소 ON/OFF dynamics, replay, 장거리 동기화와 표현 변환을 각각 지지하지만, 이들을 고정된 NREM 정리·REM 재조합 알고리즘이나 전체 $\Delta W\to\Delta g\to\Delta x(t)$ 사슬의 증거로 합칠 수는 없다. 따라서 현재 지위는 두 그래프 정리와 네 개의 실행 가능한 경험 경로이며, AGI 충분조건이나 생물학적 통합 이론은 아니다.

## 1. 문제와 핵심 기여

이 연구는 연결 변화가 계산공간을 바꾼다는 직관에서 출발한다. 이 직관의 강점은 학습, 숙련, 문맥 전환과 수면을 모두 “어떤 상태가 다른 상태에 얼마나 싸게 도달하는가”라는 공통 질문으로 바꾼다는 데 있다. 약점은 graph topology, 유향 전이 비용, 대칭 SPD 계량, neural representation과 실제 trajectory를 모두 `geometry`라는 한 단어로 묶기 쉽다는 데 있다.

핵심 기여는 그 객체들을 분리한 것이다. 간선 추가가 shortest-path cost에 주는 효과는 정확한 정리로 닫았다. 반면 구조적 동일성은 삭제하고, 학습·수면 이론은 사전 고정한 비용 추정기가 raw 연결·활동 대조군 이후에도 미래 궤적을 예측하는지 묻는 경험 프로그램으로 바꿨다.

## 2. 정의와 표기

[정의] 유한 상태 집합 $V$와 양의 비용을 가진 유향 multigraph를 $G=(V,E,w)$라 한다. $d_G(i,j)$는 $i$에서 $j$로 가는 유향 최단경로 비용이고, 도달할 수 없으면 $+\infty$다. 일반적으로 $d_G(i,j)\ne d_G(j,i)$이므로 이는 extended quasi-distance이며 리만 거리와 같은 typed object가 아니다.

[정의] raw 연결 상태를 $W$, 활동과 gain을 $A$, edge 지연을 $\tau$, 문맥을 $q$라 한다. 문맥별 유향 비용은 사전 고정한 사상으로만 정의한다.

$$
c^{(q)}=\Phi_q(W,A,\tau,q),
\qquad
d_q=\operatorname{ShortestPath}(G,c^{(q)}).
\tag{1}
$$

$\Phi_q$를 선언하지 않으면 $W\to d_q$는 정리되지 않은 것이 아니라 함수 자체가 정의되지 않은 상태다. 연속 SPD tensor $g$, control-energy quadratic form과 식 (1)의 graph quasi-distance도 서로 바꾸어 부르지 않는다.

[정의] 두 모델의 operational cost equivalence는 고정된 상태·행동 집합, cost observable, tie rule, policy class와 intervention protocol에서 pairwise cost와 induced policy가 같은 경우를 뜻한다. 이 정의는 topology, optimal path multiplicity 또는 trajectory law의 동일성을 뜻하지 않는다.

## 3. 공리와 모델 선택

[공리: 모델 선택] 학습된 계산 기하를 시험할 때 $\Phi_q$, chart, normalization, action set, tie rule, missingness 처리와 평가 window를 training 전에 고정한다. 이 선택은 뇌가 특정 metric을 구현한다는 물리 명제가 아니다.

[공리: 모델 선택] CE-AGI의 WAKE/NREM/REM label은 online accumulation, offline regularization과 탐색을 호출하는 software schedule이다. 생물학적 NREM과 REM의 고정 계산 순서나 효능은 이 명칭에서 따라오지 않는다.

[공리: 외부 입력] 신경과학 논문에서 측정한 representation, functional connectivity, replay, synchrony와 행동은 각 논문의 species, task와 intervention 범위 안에서만 사용한다. 서로 다른 자료의 상관을 하나의 개체에서 측정한 인과 사슬처럼 합산하지 않는다.

## 4. 단일 간선 최단거리 정리

[정리] 비용 $a>0$의 새 유향 간선 $e=(u,v)$를 추가한 그래프를 $G'$라 하면 모든 $i,j$에 대해

$$
d_{G'}(i,j)
=\min\{d_G(i,j),\ d_G(i,u)+a+d_G(v,j)\}.
\tag{2}
$$

증명. 새 간선을 쓰지 않는 경로의 최솟값은 첫 항이다. 새 간선을 쓰는 최단 walk가 그 간선을 두 번 이상 지난다면 두 사용 사이에 양의 비용 closed subwalk가 생긴다. 이를 제거하면 비용이 엄격히 작아지므로 최단 walk와 모순이다. 따라서 새 간선을 쓰는 최단 walk는 기존 그래프의 $i\leadsto u$ 부분, 새 간선, 기존 그래프의 $v\leadsto j$ 부분으로 분해되고 그 최솟값이 둘째 항이다. 두 경우의 최솟값이 식 (2)다. □

[따름정리] 영향을 받는 ordered pair 집합을

$$
S_{uv}(a)
=\{(i,j):d_G(i,u)+a+d_G(v,j)<d_G(i,j)\}
\tag{3}
$$

로 두면 $d_{G'}(i,j)<d_G(i,j)$일 필요충분조건은 $(i,j)\in S_{uv}(a)$다. 등호인 경우 거리값은 같아도 새 optimal route가 생길 수 있다. 따라서 국소 연결 변화는 많은 쌍의 거리를 바꿀 수 있지만 반드시 모든 쌍을 바꾸지는 않으며, distance와 route-set 변화도 구별해야 한다.

[산출: 유한 검산] 세 labelled vertex의 모든 non-loop arc가 absent, 비용 1, 비용 2인 $729$개 base graph에 모든 새 간선과 두 비용을 적용한 8,748 cases에서 식 (2)가 exact arithmetic으로 일치했다. 기존에 더 싼 $u\to v$ arc가 있는 1,458 cases도 포함했다. 이 검산은 증명의 대체물이 아니라 구현 회귀 증거다.

## 5. 삭제된 동일시와 살아남는 경계

[정리: no-go] shortcut 추가와 metric deformation은 구조적으로 동일하지 않다. 간선 추가는 incidence와 multiplicity를 바꿀 수 있고, 유향 graph distance는 비대칭이거나 무한일 수 있으며, 같은 pairwise distance라도 parallel action과 transition noise가 다르면 trajectory law가 달라진다. 보존 가능한 문장은 선택한 cost observable에서 같은 pairwise cost를 산출하는 표현들이 operationally 서로를 emulate할 수 있다는 정의뿐이다.

[정리: 비식별성] raw $W$는 유일한 $g$를 정하지 않는다. 예를 들어 $c_q=1/(WA_q)$에서는 같은 $W$가 문맥별 $A_q$에 따라 다른 비용을 만들고, $W$와 $A_q$를 반대로 바꾸면 $\Delta W\ne0$인데 비용은 그대로일 수 있다. 또한 $S\in GL(r)$에 대해 $W'=SW$, $g'=S^{-T}gS^{-1}$이면

$$
(W')^Tg'W'=W^TgW
\tag{4}
$$

이므로 이 quadratic observable만으로 $W$와 $g$를 개별 식별할 수 없다. 따라서 올바른 연구식은 자동 사슬이 아니라 다음의 측정 계약이다.

$$
(W,A,\tau,q)
\xrightarrow{\ \Phi_q\ }
c^{(q)}
\xrightarrow{\ \operatorname{ShortestPath}\ }
d_q
\xrightarrow{\ \text{registered dynamics}\ }
\widehat x(t).
\tag{5}
$$

[정리: 시간 모형의 경계] 인지시간은 보편적으로 $L_{\rm effective}/v_{\rm neural}$가 아니다. edge delay, queue, integration overhead와 병렬 branch가 있으면 등록 가능한 좁은 모형은

$$
T=t_0+max_{p\in\mathcal P_{\rm required}}
\left(\sum_{e\in p}\tau_e\right)+t_{\rm integrate},
\qquad
\tau_e=\ell_e/v_e+s_e+q_e.
\tag{6}
$$

단일 직렬 경로, 균일 속도, zero synaptic/queue delay와 zero overhead일 때만 식 (6)이 $L/v$로 줄어든다.

## 6. 관측 근거가 허용하는 범위

22개 고유 1차 출처를 대조한 결과, 학습과 수면의 여러 하위 현상은 확인되지만 전체 이론을 직접 측정한 자료는 없다.

| 근거군 | 직접 지지되는 내용 | 지지되지 않는 확장 |
|---|---|---|
| E01--E03 | 문맥별 communication subspace와 학습에 따른 representation geometry 변화 | unique router, cache key, $W\to g$ |
| E04--E10 | 숙련의 대사비용 감소, 지능과 network reconfiguration, 조류의 추상 규칙·주의 | branching 억제의 인과성, AGI 알고리즘 충분성 |
| E11--E14 | PFC synaptic state와 sleep pressure, 국소 ON/OFF 조작, critical regime 복구, 수면박탈 중 connectivity 변화 | 전역 metric renormalization, 고정 maintenance scheduler |
| E15--E18, E21 | replay 붕괴, 새/옛 기억 replay 분리, spindle-ripple coupling과 timing intervention | literal unit/regression/integration test 알고리즘 |
| E19--E20, E22 | 수면 뒤 representation·sequence 변화와 NREM/REM transition gating | REM 전용 compile, 꿈의 조합 알고리즘 |

[미완성] 어느 자료도 같은 개체와 시간창에서 연결 변화, 사전 고정한 기하 변화와 이후 trajectory를 모두 측정하지 않는다. 따라서 $\Delta W\to\Delta g\to\Delta x(t)$는 현재 관측 결론이 아니다.

## 7. 살아남은 네 연구 경로

첫째, context-conditioned cost 경로는 training 전에 $\Phi_q$를 고정하고 unseen context의 cost ordering, next-state likelihood와 hitting time을 함께 예측한다. raw $W$, activity, latency, Euclidean distance와 parameter-matched unrestricted model이 필수 대조다. geometry shuffle 뒤 성능이 유지되거나 paired held-out gain이 없으면 독립 geometry 효능을 철회한다.

둘째, critical-path timing 경로는 task precedence DAG와 edge delay를 독립 calibration에서 고정한다. 병렬 branch와 join을 조작한 새 task의 반응시간을 식 (6)으로 예측하고 단순 $L/v$ 및 parameter-matched latency model과 비교한다.

셋째, selective-practice 경로는 동일 초기 비용의 trained pair와 untrained matched pair를 둔다. usage dose가 $\Delta d_q$, energy proxy와 error/response time을 같은 방향으로 예측하는지 시험한다. global scaling이나 exposure-only model로 설명되면 “숙련은 선택적 거리 단축”이라는 가설을 기각한다.

넷째, sleep-selective-rearrangement 경로는 sleep/wake, time-awake, firing-rate, global scaling과 replay-sham 대조를 함께 둔다. 같은 밤의 선택적 변화가 다음 날 trajectory fidelity와 generalization을 모두 예측하고 replay-coupling shuffle에서 이득이 사라질 때만 살아남는다. 이 경로는 NREM 뒤 REM이라는 고정 직렬 계산을 전제하지 않으며 local asynchronous maintenance와 periodic synchronization을 경쟁 모델로 남긴다.

## 8. 가장 값싼 다음 반증 실험

[예측] 가장 먼저 할 분석은 공개 가능성이 있는 sleep-loss hippocampal recording(E15)의 동일 시간창에서 trajectory replay fidelity와 effective branching 또는 transition-entropy proxy를 함께 계산하는 것이다. window, unit exclusion, event detector와 proxy를 training/analysis 전에 고정하고 animal/session을 독립 통계 단위로 둔다.

Primary model은

$$
Q_{\rm replay}(t)
=\alpha+\beta B_{\rm eff}(t)+\gamma^Tz(t)+\epsilon(t),
\qquad \beta<0,
\tag{7}
$$

이고, $z(t)$에는 firing rate, SWR count, time awake와 motion/arousal proxy를 넣는다. Secondary로 임계 sigmoid를 선형식과 nested validation에서 비교한다. branching proxy가 replay를 추가 예측하지 못하거나 firing/SWR baseline과 동률이면 통합 경로의 중심 다리를 기각한다. E13의 cortical criticality 자료는 별도 replication domain이지 E15와 개체 수준 인과 사슬로 합칠 수 없다.

## 9. AGI 정본의 수정된 결론

[정의] 연결은 가능한 전이와 제약을 제공하고, 학습은 문맥별 전이 비용과 표현을 바꿀 수 있으며, 사고는 그 동역학이 만든 trajectory다. 이 문장은 객체의 역할을 구분하는 연구 언어이지 하나의 존재론적 동일시가 아니다.

[미완성] 지능을 “좋은 계산 기하를 빠르고 싸게 만드는 능력”으로 운영화하려면 task family, 비용, 일반화, adaptation budget과 safety constraint를 함께 정의해야 한다. shortest path가 짧은 것만으로 정확성, compositionality, robustness 또는 목표의 타당성이 보장되지 않는다.

[미완성] 수면형 offline phase가 AGI에 필요한지, replay와 regularization을 결합한 schedule이 동일 계산량의 기존 continual-learning 방법보다 나은지, 생물학적 수면 자료가 그 설계를 제약하는지는 아직 열려 있다. 현재 정본은 해당 schedule을 software proposal과 kill test로만 유지한다.

## 10. 한계와 재개 조건

출처 ledger의 일부 논문은 full text, exact sample size 또는 public code 접근을 추가 확인해야 하므로 정량 재분석 전에 source manifest를 다시 고정해야 한다. 이번 run은 외부 데이터를 내려받거나 model을 훈련하지 않았고, 기존 synthetic causal recurrent geometry의 development result를 수면·기억 증거로 재사용하지 않았다.

통합 가설을 다시 열 최소 조건은 같은 animal/session의 pre/post 측정, frozen $\Phi_q$, independent split, raw-$W$/activity/latency baseline, context 또는 sleep intervention과 trajectory endpoint다. 이 조건 없이 서로 다른 논문의 부분 결과를 이어 붙인 설명은 연구 동기일 수는 있어도 검증된 계산이론은 아니다.

## 11. 재현성

수학 검산:

```powershell
python _workspace/ce/agi-learning-geometry-sleep-20260818/artifacts/verify_lgs_math.py
```

연구 체인과 문서 수정 기록은 같은 run의 `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`, `20-audit.md`, `30-implementation.md`, `31-validation.md`에 있다. 제품 runtime과 public API는 수정하지 않았으며 전체 test suite는 실행하지 않았다.

## 12. 참조

접근일: 2026-08-18. 아래는 사용자 노트의 중복을 제거한 22개 1차 출처다.

1. E01, [A communication subspace relays context-dependent actions from human prefrontal to motor cortex](https://www.nature.com/articles/s41593-026-02290-4).
2. E02, [Learning shapes neural geometry in the primate prefrontal cortex](https://www.nature.com/articles/s41593-026-02333-w).
3. E03, [Dynamically shifting from compositional to conjunctive brain representations supports cognitive task learning](https://www.nature.com/articles/s41467-025-65041-2).
4. E04, [Extended practice of a motor skill is associated with reduced metabolic activity in M1](https://www.nature.com/articles/nn.3477).
5. E05, [Multitask brain network reconfiguration is inversely associated with human intelligence](https://academic.oup.com/cercor/advance-article-abstract/doi/10.1093/cercor/bhab473/6523266).
6. E06, [Structural-functional brain network coupling during cognitive demand reveals intelligence-relevant communication strategies](https://www.nature.com/articles/s42003-025-08231-4).
7. E07, [The network architecture of general intelligence in the human connectome](https://www.nature.com/articles/s41467-026-68698-5).
8. E08, [Genetic mapping and evolutionary analysis of human-expanded cognitive networks](https://www.nature.com/articles/s41467-019-12764-8).
9. E09, [Abstract rule neurons in the endbrain support intelligent behaviour in corvid songbirds](https://www.nature.com/articles/ncomms3878).
10. E10, [Neuronal correlates of endogenous selective attention in the endbrain of crows](https://www.nature.com/articles/s42003-025-07914-2).
11. E11, [Prefrontal synaptic regulation of homeostatic sleep pressure revealed through synaptic chemogenetics](https://doi.org/10.1126/science.adl3043).
12. E12, [Induction of cortical on/off periods in awake mice fulfills sleep functions](https://www.nature.com/articles/s41593-026-02318-9).
13. E13, [Sleep restores an optimal computational regime in cortical networks](https://www.nature.com/articles/s41593-023-01536-9).
14. E14, [Breakdown of thalamocortical connectivity under sleep deprivation](https://academic.oup.com/sleepadvances/article/6/4/zpaf065/8269852).
15. E15, [Sleep loss diminishes hippocampal reactivation and replay](https://www.nature.com/articles/s41586-024-07538-2).
16. E16, [Sleep microstructure organizes memory replay](https://www.nature.com/articles/s41586-024-08340-w).
17. E17, [Spindle-locked ripples mediate memory reactivation during human NREM sleep](https://www.nature.com/articles/s41467-024-49572-8).
18. E18, [Augmenting hippocampal-prefrontal neuronal synchrony during sleep enhances memory consolidation in humans](https://www.nature.com/articles/s41593-023-01324-5).
19. E19, [Slow-wave sleep and REM sleep differentially contribute to memory representational transformation](https://www.nature.com/articles/s42003-025-08812-3).
20. E20, [Sleep selectively and durably enhances memory for the sequence of real-world experiences](https://www.nature.com/articles/s41562-025-02117-5).
21. E21, [How coupled slow oscillations, spindles and ripples coordinate neuronal processing and communication during human sleep](https://www.nature.com/articles/s41593-023-01381-w).
22. E22, [Infraslow noradrenergic locus coeruleus activity fluctuations are gatekeepers of the NREM-REM sleep cycle](https://www.nature.com/articles/s41593-024-01822-0).
