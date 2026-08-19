# 신경 기하 후보식 전수 검증 계약

Status: COMPLETE

PREDECESSOR: _workspace/ce/_archive/agi-learning-geometry-sleep-20260818
PREDECESSOR: _workspace/ce/_archive/sleep-replay-routing-realdata-20260818
PREDECESSOR: docs/7_AGI/15_Equations.md

AMENDMENT: artifacts/candidate-equation-registry.md
REGISTRY_SHA256: 3491e665d00c0f2eec0c423d162dc52aa69ca917865f25fb86926e3293737d8a
ELIGIBILITY_LEDGER: artifacts/candidate-equation-registry.json
ELIGIBILITY_LEDGER_SHA256: f321ae5263695082ac8ff12448bfed3299e32ed5afb9bfd015511bbcf1ec93ae

V2_VALIDITY_AMENDMENT: V1은 `S7-H,H=1` 항등식, zero-ridge rank gate의 부동소수 오판정, 입력 byte와 tuple 완전성 freeze 부족 때문에 무효다. V2는 후보 계열·유한 grid·점수 방향을 결과에 맞춰 바꾸지 않고 이 구조적 eligibility와 provenance 결함만 고친다. V1과 V2 모두 열린 E17의 retrospective discovery이며 봉인 확인이 아니다.

## 1. 질문과 판정 목표

다음 통합 가설을 실제 신경 자료에서 반증 가능하게 만든다.

$$
\Delta W\longrightarrow\Delta g\longrightarrow\Delta x_{0:T}.
\tag{1}
$$

여기서 목표는 학습 뒤의 신경활동으로 계량을 사후 정의해 같은 활동을 재설명하는 것이 아니다. 연결 또는 국소 동역학을 독립 자료에서 먼저 측정하고, 사전 고정한 사상으로 계량 변화를 계산한 뒤, 보지 않은 미래 trial의 궤적 분포를 예측해야 한다. 학습 전후, 문맥 전환, 수면 전후의 세 조건은 각각 독립 검정이며 서로 다른 연구의 효과를 이어 붙여 하나의 인과 사슬로 세지 않는다.

이번 amendment는 한 operational 식만 시험하지 않는다. `artifacts/candidate-equation-registry.md`에 닫힌 SPD, graph, directed action/Finsler, distribution geometry와 readout 계열을 모두 자격 판정하고, 서로 같은 관측량을 예측하는 타입 안에서만 비교한다. 연속적인 모든 함수가 아니라 입력 객체·변환법칙·극한에서 서로 동치가 아닌 대표 계열과 완전히 열거한 tuple을 "전부"로 정의한다. E17의 단일 `S4-H=5` 예비 결과는 이 registry를 닫기 전에 이미 열렸으므로 E17 전수 비교는 retrospective discovery이며 봉인 확인이 아니다.

## 2. typed object와 조작적 계량

[정의] $W^s$는 해부학적·시냅스 연결, $W^e$는 독립 학습 구간에서 추정한 directed effective connectivity, $z_t\in\mathbb R^r$는 세포 identity 또는 training-only chart가 고정된 신경상태다. $W^s$와 $W^e$를 같은 측정량으로 부르지 않는다.

[정의] 조건 $c$의 국소 확률 동역학을

$$
z_{t+1}=F(z_t;W,A,c)+B(z_t)u_t+\varepsilon_t,
\qquad
\operatorname{Cov}(\varepsilon_t)=Q(z_t)
\tag{2}
$$

로 둔다. $A$는 gain, inhibition, delay와 neuromodulatory state를 포함한다. $J_t=\partial F/\partial z|_{z_t}$와 $Q_t$는 trajectory 평가 trial과 분리된 calibration/train 자료에서만 추정한다.

[정의: 후보 universe] 등록 집합을

$$
\mathcal U=\{S0,\ldots,S16;\ G1,G2,G3a,G3b;\ D1,D2,D3;\ P1/P2;\ R3,R4\}
\tag{3a}
$$

로 둔다. 각 ID의 정확한 식, 입력, 자격 조건과 유한 격자는 registry의 고정 bytes가 정의한다. 서로 다른 typed object를 하나의 scalar leaderboard에 합치지 않는다.

[정의: process-noise 후보] horizon $H$의 local process-noise reachability covariance와 `S4-H` SPD 계량 후보는 time-invariant 구간에서

$$
C_H(z)=\sum_{k=0}^{H-1}J_z^k Q_z(J_z^k)^\top,
\qquad
g_H(z)=\left(C_H(z)+\lambda_C R_C(z)\right)^{-1}
\tag{3}
$$

로 정한다. time-varying 식에서는 $\varepsilon_{t+k}$가 들어간 시점 $t+k+1$부터 공통 종점 $t+H$까지의 곱 $\Psi_{t,H,k}=J_{t+H-1}\cdots J_{t+k+1}$을 사용해 $C_{Q,H}(t)=\sum_k\Psi_{t,H,k}Q_{t+k}\Psi_{t,H,k}^\top$로 계산한다. 초기시점에서 앞으로 가는 곱을 각 $Q_{t+k}$ 양쪽에 붙이지 않는다.

$R_C\succ0$는 covariance ridge이고 선형 chart 변화 $z'=Pz$에서 $R'_C=PR_CP^\top$로 변환한다. 직접 metric 식에 더하는 $G_0\succ0$는 별도 객체이며 $G'_0=P^{-\top}G_0P^{-1}$로 변환한다. 둘을 같은 $R_0$로 쓰지 않는다. $R_C=G_0=I$를 고정하는 E17 구현은 등록된 표준화 chart와 직교변환으로 제한한다. $H$, $r$, ridge, chart, smoothing, estimator, rank, edge cost, optimizer와 missingness 처리는 test trajectory를 열기 전에 고정한다. 식 (3)은 제어가능성 Gramian도, 뇌의 유일한 계량이라는 공리도 아니다. 제어 계량을 시험하려면 별도로 측정한 input channel $B$와 control cost $R_u$를 가진 `S10-H`를 사용한다.

[정의] Riemannian 후보가 예측하는 곡선 비용은

$$
L_g[\gamma]=\int_0^1\sqrt{\dot\gamma(s)^\top g_H(\gamma(s))\dot\gamma(s)}\,ds.
\tag{4}
$$

유향 drift $F$는 대칭 SPD tensor에 흡수하지 않는다. 확률 궤적의 비교 모델은 drift를 분리한 action

$$
S[x]=\frac12\sum_t
(x_{t+1}-F(x_t))^\top Q_t^{-1}(x_{t+1}-F(x_t))
\tag{5}
$$

을 사용한다. 식 (4)는 reversible Riemannian length이고 식 (5)는 drift-dependent directed transition action이라는 서로 다른 typed object다. 어느 한쪽의 양성 결과를 다른 쪽의 검증으로 세지 않으며, 식 (4)가 방향성, 비가역성 또는 transition law 전체를 표현한다는 부모 주장은 등록하지 않는다.

[정의] graph node metric/quasi-metric `G*`, path action/Finsler `D*`, 시행 분포의 Wasserstein `P1/P2`, smooth field가 있어야 정의되는 curvature/flow `R3/R4`도 상태 접공간 SPD와 별도 타입이다. `P1`과 `P2`는 같은 $W_2$의 static/dynamic 표현이므로 두 독립 후보로 세지 않는다. sessionwise constant SPD matrix는 고정 chart에서 flat하므로 E17에서 비자명한 Ricci curvature를 산출하지 않는다.

## 3. 등록 주장

| ID | 주장 | 시작 지위 | 승격 조건 |
|---|---|---|---|
| `NRM-D1` | 식 (2)--(5)는 $W,A,Q$, chart와 horizon이 고정됐을 때 계산 가능한 operational geometry다 | [정의] | typed input과 좌표변환 법칙 유지 |
| `NRM-D2` | registry의 유한 집합 $\mathcal U$가 이번 세대의 후보식 전부다 | [정의: 모델 선택] | registry hash, tuple, eligibility와 type별 endpoint 고정 |
| `NRM-T1` | $C_H+\lambda_C R_C\succ0$이면 $g_H$는 SPD이며, $\lambda_C=0$인 full-rank 경우 또는 $R_C$를 공변 변환한 경우에만 일반 선형 chart change에서 길이가 불변이다 | [정리 후보] | 제한 조건의 증명, isotropic-ridge 반례와 수치 검산 |
| `NRM-T2` | time-varying $C_{Q,H}$는 각 innovation을 그 시점에서 공통 종점까지 보내는 $\Psi_{t,H,k}$ 합이다 | [정리 후보] | covariance recursion 유도와 time-varying 반례 fixture |
| `NRM-N1` | raw $W$만으로 유일한 생물학적 리만 계량이 결정된다 | [강한 부모 주장] | gain/noise/gauge 완전 반례가 있으면 삭제 |
| `NRM-N2` | representational geometry 변화만 관찰하면 $\Delta W\to\Delta g$가 검증된다 | [강한 부모 주장] | 재표집·chart drift·noise covariance 반례가 있으면 삭제 |
| `NRM-H1A` | 같은 세포의 직접 연결 변화 $\Delta W^s$로 계산한 $\Delta g_H$가 이후 trial trajectory를 외부 예측한다 | [핵심 미완성] | Tier A 자료의 nested holdout과 intervention |
| `NRM-H1B` | 독립 calibration에서 얻은 $\Delta W^e$ 기반 $\Delta g_H$가 이후 trial trajectory를 외부 예측한다 | [대리 미완성] | Tier B 자료; H1A와 구분 표기 |
| `NRM-H2` | geometry는 raw $W$, firing rate, covariance, Euclidean latent distance, direct dynamics보다 parameter-matched held-out 점수를 추가 개선한다 | [미완성] | animal/session-held-out proper-score 우위 |
| `NRM-H3` | 학습으로 짧아질 것으로 예측된 방향에서 실제 경로 action, hitting time 또는 오류가 선택적으로 감소한다 | [미완성] | trained-pair 대 matched untrained-pair interaction |
| `NRM-H4` | 문맥 조작은 anatomy가 고정된 채 $A,c$를 통해 계량을 바꾸고 새 문맥 trajectory를 예측한다 | [미완성] | context-label/gain-only 대조와 causal switch |
| `NRM-H5` | 수면 후 선택적 $\Delta g_H$가 global scaling과 time-awake보다 다음 날 trajectory fidelity를 잘 예측한다 | [미완성] | paired sleep/wake 및 replay/sham 대조 |
| `NRM-E17D` | E17 Figure 2에서 자격 있는 모든 tuple을 같은 split 규칙으로 계산할 수 있다 | [산출 후보: discovery only] | machine eligibility ledger, raw tuple score와 animal-level aggregation |
| `NRM-N3` | H1B 또는 표현기하 양성만으로 “연결이 공간을 만들고 학습이 공간을 휜다”는 인과 명제가 확정된다 | [강한 부모 주장] | 식별성 감사; 불충분하면 삭제 |

## 4. 증거 등급

| Tier | 필요한 동시 측정 | 허용 결론 |
|---|---|---|
| A | 같은 식별 세포/시냅스의 pre/post $W^s$, 독립 $A,Q$, 이후 single-trial trajectory, 학습 또는 plasticity 개입 | `H1A`의 직접 검정 |
| B | 같은 세포의 longitudinal activity, calibration 구간의 $W^e,J,Q$, 분리된 future trials, 행동 endpoint | effective-dynamics 대리검정만 |
| C | 세션별 다른 세포 또는 condition-mean representational geometry와 행동 | 기하 상관/현상 재현만; (1)의 검증 금지 |

Tier를 자료를 본 뒤 올리지 않는다. 동물 수가 3 미만인 자료의 window/trial 수는 독립 표본 수로 세지 않으며 confirmatory population claim을 금지한다.

## 5. primary analysis와 누수 차단

1. E17 discovery는 animal leave-one-out 세 fold다. held-out animal의 앞 50% trial은 session-local calibration에 쓸 수 있지만 그 animal의 inner/test outcome으로 tuple을 고르지 않는다. session score를 먼저 animal 안에서 평균해 반복 session이 많은 animal에 가중치가 몰리지 않게 한다.
2. 각 session의 fit block에서만 cell filter, chart, $F,J,Q$와 session-local scalar calibration을 정한다. candidate tuple은 outer-train animals의 inner block에서만 고른다. registry §9의 fit/inner/test, horizon, ridge, rank, optimizer, seed와 tie rule을 그대로 쓴다. outer-train raw tuple 결과와 선택된 held-out test 결과를 보존한다.
3. uncertainty SPD의 primary endpoint는 held-out Gaussian NLPD이고 energy score는 secondary다. deformation/observation, condition-information, graph, directed action과 distribution family는 registry §10의 서로 다른 endpoint를 쓴다. 타입 사이 점수를 합치지 않는다.
4. 비교군은 persistence, Euclidean/isotropic, firing-rate/state covariance, raw effective-$W$, direct full state-space covariance, diagonal covariance, 가장 가까운 자유도의 `S12/S13` flexible SPD와 label-only decoder다.
5. E17은 3 animals이고 release 전체가 이미 열린 자료다. 모든 p-value와 candidate rank는 탐색량이며 population winner, lock success, future-trial chronology 또는 biological metric 증거로 부르지 않는다.
6. geometry의 독립 기전 주장은 별도 잠금 코호트에서 direct dynamics 대비 paired outer-test improvement의 confidence interval이 0을 넘고, geometry shuffle 및 time reversal에서 사라지며, 새로운 task/target endpoint도 같은 frozen estimator가 맞힐 때만 허용한다.
7. mediation은 $W$ intervention이 있고 sequential ignorability 또는 그 대체 식별 가정이 충족될 때만 confirmatory로 부른다. 관측 자료의 product-of-coefficients는 기술량이다.

## 6. 사전등록 kill tests

| ID | 즉시 기각 또는 강등 조건 |
|---|---|
| `K1` | $g_H$의 eigenvalue/length가 bootstrap cell resampling 또는 admissible chart 변화에 불안정 |
| `K2` | test trajectory로 chart, $H$, $\lambda$, smoothing 또는 target pair를 선택해야 양성 |
| `K3` | raw-$W$/direct-dynamics/parameter-matched SPD가 geometry와 동률 또는 우월 |
| `K4` | predicted shortening direction과 held-out path/hitting-time 변화의 방향 불일치 |
| `K5` | geometry shuffle, condition-label permutation 또는 time reversal 뒤 점수가 유지됨 |
| `K6` | global gain/covariance scaling만으로 동일한 $\Delta g$와 trajectory 효과 설명 가능 |
| `K7` | 동일 개체·동일 시간창 없이 서로 다른 연구의 $\Delta W$, $\Delta g$, $\Delta x$를 연결해야 함 |
| `K8` | 세션 사이 세포 identity가 바뀌는데 이를 연결 변화로 해석함 |
| `K9` | SPD, graph, action, distribution distance 또는 curvature를 같은 점수표에서 서로 대체 가능한 metric으로 취급함 |
| `K10` | 이미 열린 E17 결과로 후보를 고른 뒤 같은 E17 test를 봉인 확인으로 재사용함 |

## 7. 자료 탐색 및 실행 범위

외부 근거 레인은 2026-08-18 현재의 공식 논문·저장소만 사용해 다음을 찾는다.

- 동일 세포 longitudinal population activity와 학습 행동;
- 직접 synapse/connectivity pre/post 측정과 기능 activity의 결합;
- causal plasticity 또는 connectivity perturbation;
- 학습 전후 representational geometry 공개자료;
- 수면 전후 동일 세포 trajectory와 connectivity/effective-connectivity 자료.

각 자료에 종, 동물 수, 세포 identity 보존, 연결의 측정 종류, intervention, trajectory 해상도, 공개 파일·라이선스·크기를 기록한다. 가장 높은 가용 Tier 자료 하나 이상을 실제로 내려받아 manifest와 SHA-256을 남긴다. 각 candidate tuple은 `ELIGIBLE`, `CONDITIONAL`, `UNTESTABLE_MISSING_INPUT`, `INELIGIBLE_MATH`, `INSUFFICIENT_PAIRS` 또는 실행 실패 사유 중 하나를 반드시 받는다. 계산 가능한 식은 registry의 동일 split로 전부 실행하고, 입력이 없는 식은 임의 proxy를 넣지 않는다. 접근 차단이나 필수 변수가 없으면 오류 body를 데이터로 세지 않고 `ACCESS_BLOCKED` 또는 `UNTESTABLE`로 판정한다.

## 8. 수학·구현 요구사항

1. 식 (3)의 SPD 조건, horizon/regularization dependence와 제한된 좌표 공변성을 증명하고, covariance ridge $R_C$와 metric reference $G_0$의 반대 변환법칙을 검산한다.
2. time-varying `S4/S10`에서 innovation/input 시점부터 공통 종점까지의 $\Psi_{t,H,k}$ 방향을 유도하고, 잘못된 초기시점 곱과 값이 달라지는 fixture를 둔다.
3. raw $W$ 유일성, symmetric metric의 drift 흡수, representational geometry의 causal 식별, Fisher/pullback/Hessian의 rank 결손과 타입 혼합에 대한 반례 또는 gate를 제시한다.
4. synthetic known-ground-truth dynamics에서 $\Delta W\to\Delta g\to\Delta x$ estimator가 양성·null·confounded case를 구분하고 후보별 SPD/coordinate/type gate가 작동하는지 검산한다.
5. real-data 분석은 registry hash, runner hash, 11개 입력 MAT의 사전 고정 SHA-256, candidate manifest, random seed, split ledger, schema, 독립 단위, raw tuple score와 실패 조건을 machine-readable artifact로 남긴다. 후보별 예상 tuple key와 실제 key를 모든 session/condition/horizon cell에서 대조하고 하나라도 빠지거나 추가되면 결과 파일을 쓰지 않는다. 기존 결과 파일은 덮어쓰지 않는다.
6. canonical 문서는 최종 감사에서 허용된 지위와 검증법만 반영하며 양성 discovery를 핵심식의 증명이나 E17 winner로 승격하지 않는다.

## 9. 완료 조건

1. 모든 `NRM-*` 주장과 registry candidate에 정의·정리·산출·경험식·미완성·기각 또는 입력부재 지위가 있다.
2. 공식 E17 자료의 byte-level provenance와 field eligibility를 기록하고, 자격 있는 모든 tuple을 같은 split 규칙으로 실행하며 자격 없는 모든 tuple의 결측 입력을 machine-readable하게 남긴다.
3. 시간가변 방향, SPD, chart, likelihood normalization, type gate 수학 fixture와 real-data candidate fixture가 focused validation을 통과한다.
4. raw score, animal-level discovery 요약, direct/flexible baselines와 kill-test 결과를 분리하고 E17에서 population winner를 선언하지 않는다.
5. Tier A/B/C 결론을 섞지 않고 식 (1)이 현재 검증됐는지 명시한다.
6. 다음 결정적 실험은 측정, intervention, split, primary endpoint, 표본 단위와 kill test까지 구체화한다.
