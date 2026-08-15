# Causal Recurrent Geometry benchmark 사전등록 청사진

Status: COMPLETE

이 문서는 후속 구현 run에서 freeze할 설계를 정의한다. 아직 seed, 표본수와 모델 hash를 고정한 실제 preregistration은 아니며, 외부 자료나 confirmation split을 열지 않았다.

## 1. 연구 질문

첫 benchmark는 세 질문을 순서대로 검사한다.

1. 알려진 개입이 있을 때 불변 구조 $G$와 문맥 $z_t$를 분리한 모형이 보지 않은 개입의 rollout을 개선하는가.
2. 복원된 $G$의 SCC quotient가 같은 압축률의 다른 모듈화보다 장기예측과 개입효과 전파에 유리한가.
3. 고정된 representation에서 control-energy metric이 단순 weight·activity 특징 이후에도 partial-cue recall 접근성을 예측하는가.

앞 질문이 실패하면 뒤 질문의 결과를 앞 질문의 증거로 사용하지 않는다.

## 2. 합성 생성계

### 2.1 구조와 문맥

각 simulation instance는 유한 유향 graph $G=(V,E)$, context $z_t$, control $u_t$와 state $x_t$를 갖는다. graph는 명시된 SCC 내부 edge와 SCC 사이의 acyclic edge로 생성한다. 내부 spectral radius는 안정 영역에 두고, instability regime은 별도 stratum으로 둔다.

$$
x_{t+1}
=\tanh\!\left(A(z_t)x_t+Bu_t+b(z_t)\right)+\epsilon_t.
\tag{1}
$$

$A(z_t)$는 기본 실험에서 support를 공유하고 edge gain만 문맥별로 바꾼다. support까지 바뀌는 실험은 별도 regime이며 같은 결론에 합치지 않는다. latent context는 유한-state Markov chain과 연속 drift 두 종류로 나눈다.

### 2.2 관측 조건

관측은 세 조건을 분리한다.

$$
y_t=Cx_t+\nu_t.
\tag{2}
$$

- `known_identity`: $C=I$이고 node 좌표가 알려진다.
- `known_mask`: 알려진 일부 node만 관측한다.
- `unknown_mix`: 미지의 가역 혼합 $C$를 사용한다.

정확한 edge recovery는 `known_identity`에서만 primary가 될 수 있다. `unknown_mix`에서는 similarity/diffeomorphism gauge 때문에 예측 동치류와 intervention response를 primary로 둔다.

### 2.3 개입 조건

- 수동 관측만 있는 regime
- 한 node에 pulse 또는 clamp를 가하는 regime
- SCC 내부와 SCC 사이 edge를 cut하는 regime
- train에 없던 node·module·context 조합을 confirmation에 두는 regime

개입 target은 graph 생성 seed와 독립적으로 뽑고, confirmation target 목록은 model freeze 전 공개하지 않는다.

## 3. 모형과 대조군

### 3.1 제안 모형

제안 모형은 다음 persistent 부분을 분리한다.

$$
\widehat x_{t+1}
=F_\theta(x_t,u_t;\widehat G,\widehat z_t),
\qquad
\widehat z_{t+1}=Z_\phi(\widehat z_t,x_t,u_t).
\tag{3}
$$

SCC는 $\widehat G$에서 계산하는 deterministic readout이다. SCC label을 정답에서 주입하지 않는다. scale $k+1$ graph를 만들 때에는 quotient trajectory 사이의 새 effective edge estimator를 사용하고, 그 estimator의 window와 threshold를 validation 전에 고정한다.

### 3.2 필수 대조군

| 역할 | 대조군 |
|---|---|
| 선형 구조 | ridge VAR, sparse VAR |
| 비선형 구조 | parameter-matched MLP/GRU, graph neural transition model |
| latent 문맥 | switching state-space model, monolithic recurrent latent model |
| 모듈 압축 | weak/strong community, spectral clustering, balanced random partition, learned soft modules |
| edge 음성 대조 | degree-preserving rewire, direction shuffle, SCC label shuffle |
| intervention 음성 대조 | target shuffle, time shuffle, no-intervention tag |

모든 비교는 encoder, hidden width, optimizer step, training examples와 wall-clock 또는 FLOP budget을 기록한다. parameter 수가 정확히 같지 않으면 성능-parameter Pareto curve를 함께 보고한다.

## 4. SCC predictive coarse-graining

고정 graph의 condensation을 다시 SCC로 축약하는 연산은 사용하지 않는다. 첫 quotient의 state는 component 내부 상태의 사전 고정 summary $q_C(x_t)$로 정의한다. 다음 scale의 effective graph는 coarse state intervention response로 새로 추정한다.

$$
q_{t+1}
=F^{(1)}(q_t,u_t),
\qquad
E^{(1)}_{ij}=1
\iff
\text{declared intervention score}_{j\to i}>\tau_1.
\tag{4}
$$

$q_C$ 후보는 mean, first fixed number of PCA coordinates와 learned equal-budget encoder로 제한한다. SCC라는 label만으로 내부 phase와 slow mode가 보존된다고 가정하지 않는다.

Primary는 compression ratio에 따른 held-out intervention rollout error curve의 면적이다. SCC가 한 operating point에서만 좋고 전체 curve에서 지배하지 못하면 보편적 우위로 쓰지 않는다.

## 5. 기하 기억 route

첫 geometry는 자유로운 신경망 metric이 아니라 제어 가능한 선형·국소선형계의 controllability metric으로 고정한다.

$$
x_{t+1}=Ax_t+Bu_t,
\qquad
W_T=\sum_{k=0}^{T-1}A^kBB^T(A^k)^T.
\tag{5}
$$

$W_T$가 SPD일 때 원점에서 target $x$로 가는 최소 제어에너지는 다음 quadratic form이다.

$$
E_T(x)=x^TW_T^{-1}x.
\tag{6}
$$

따라서 첫 operational metric은 $g_T=W_T^{-1}$이다. rank-deficient $W_T$에서는 reachable subspace와 Moore--Penrose inverse를 별도로 표시하고 전체공간 Riemannian metric이라고 부르지 않는다.

학습은 $A$, $B$ 또는 둘의 제한된 update로 구현한다. recall은 partial cue에서 target attractor neighborhood로 들어가는 최소 intervention energy와 basin-entry time으로 측정한다. metric은 $A,B$의 함수이므로 독립 물리 기전이라고 주장하지 않고, weight-only baseline보다 더 좋은 sufficient readout인지 시험한다.

필수 대조는 다음과 같다.

- Euclidean norm
- raw $A,B$ 또는 synaptic weight summary
- state covariance와 inverse covariance
- Fisher/local-Jacobian metric
- 같은 자유도의 learned quadratic form
- $W_T^{-1}$의 eigenvector shuffle와 trace-matched SPD null

## 6. primary endpoint

각 단계는 primary endpoint 하나만 가진다.

| 단계 | primary endpoint | pair 단위 |
|---|---|---|
| A mechanism recovery | confirmation intervention NLL의 최선 matched baseline 대비 차이 | graph seed |
| B SCC quotient | error-versus-compression curve 면적의 차이 | graph seed |
| C memory accessibility | held-out partial-cue recall NLL 또는 success의 차이 | task/graph seed |

edge AUPRC, SCC overlap, condensation edge F1, calibration, free-rollout error, basin-entry time과 trajectory similarity는 secondary다. secondary가 좋아도 primary 실패를 뒤집지 않는다.

통계는 paired graph-seed bootstrap을 사용하고 frame을 독립 표본으로 세지 않는다. empirical data에서는 animal 또는 독립 session을 pair 단위로 사용한다. 한 connectome의 수백만 edge를 수백만 독립 표본으로 취급하지 않는다.

## 7. 데이터 분리와 누수 방지

- 생성 graph seed 단위 train/validation/confirmation 분리
- topology family 일부와 intervention target 조합을 confirmation 전용으로 유지
- hyperparameter 선택은 validation까지만 사용
- source와 evaluation code hash를 model training 전에 기록
- structure/function dataset에서는 anatomy 파일을 별도 경로에 두고 evaluator만 읽음
- empirical edge prevalence와 spatial distance는 baseline에 명시적으로 제공하여 제안 모형만의 숨은 이점이 되지 않게 함

실제 sample size는 scoring하지 않는 pilot seed의 paired-effect variance로 정한다. pilot graph와 empirical session은 confirmation에 재사용하지 않는다.

## 8. 판정

`GO-A`는 단계 A의 primary CI가 0보다 크고, unknown-mix에서 정확 edge가 아니라 예측 동치류라고 올바르게 보고하며, intervention tag shuffle에서 이득이 사라질 때만 허용한다.

`GO-B`는 단계 B의 primary curve가 최선 압축 대조군보다 좋고, SCC label shuffle과 internal-state ablation에서 예상한 방향의 성능 저하가 재현될 때만 허용한다.

`GO-C`는 단계 C의 primary가 weight/activity/parameter-matched quadratic baseline보다 좋고, control-energy ordering을 뒤집는 metric null에서 이득이 사라질 때만 허용한다.

셋 중 하나의 `GO`를 다른 단계로 전이하지 않는다. 특히 `GO-A/B`는 기억, 생물학 또는 AGI 증거가 아니고, `GO-C`도 의식이나 자아의 증거가 아니다.
