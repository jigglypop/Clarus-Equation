# CloudCell-모나드 경험적 감사

> 작성 기준: 2026-07-31
> 결과 갱신: 2026-08-04
> 범위: 실제 뉴런 시계열에서 `stateful local-cell + population-cloud` 가설을
> 검증하는 통계 게이트와 인간 MTL coded-memory 교차 검증
> 현재 상태: `formal PASS / C. elegans primary 0/4 FAIL /
> human MTL Sternberg operational proxy 0/3 FAIL /
> concept-coded H1+H2 NOT EVALUATED / 1 subject NOT EVALUABLE`

## 0. 현재 결론과 증명의 범위

이 문서가 검증하려는 가장 강한 관측 명제는 다음이다.

$$
\boxed{
\text{개별 뉴런의 국소 기억과 집단 잠재상태를 함께 쓴 모델이}
\quad
\text{각 요소만 쓴 모델보다 미래 신경상태를 더 잘 예측한다.}
}
$$

이를 통과하면 허용되는 표현은 다음과 같다.

> CloudCell은 교체 가능한 뉴런 집단 위에서 유지되는 상태적
> 확률 전이 모듈이라는 경험적 지지를 얻었다.

반면 다음 문장은 관측 시계열만으로 증명되지 않는다.

- 뉴런 하나가 문자 그대로 범주론적 모나드다.
- CloudCell이 생물학적 실체의 유일한 분해다.
- population latent가 행동의 원인이다.
- C. elegans에서 얻은 동역학이 인간 기억의 기전과 동일하다.
- 이 구조가 AGI 성능 우위를 자동으로 보장한다.

따라서 `증명`은 세 층으로 나눈다.

| 층 | 질문 | 통과 뒤 허용되는 주장 |
|---|---|---|
| 형식 | 확률 kernel과 합성이 올바르게 정의되는가 | Markov/Kleisli 계산 구조 |
| 예측 | untouched future에서 full model이 모든 축약형을 이기는가 | operational CloudCell |
| 인과 | 정의된 ensemble/state를 교란하면 예측 방향으로 상태와 행동이 변하는가 | causal CloudCell 후보 |

현재 C. elegans 원자료는 네 recording의
`heatDataMS.mat`, `centerline.mat`까지 확보된 상태다. 이 네 recording은
실제 pilot에는 사용할 수 있지만, 서로 다른 subject에서 나온 독립 recording이라고
확인되더라도 네 개가 모두 같은 방향인 단측 exact sign test는
$p=1/16=0.0625$다. 같은 subject의 반복 recording이면 독립 단위 수는 더
작다. 따라서 네 recording만으로 population-level 확증을 선언하지 않는다.
현재 단계의 최대 판정은
`within-dataset replicated pilot`이며, confirmatory 승격에는 별도의 untouched
recording panel이 필요하다.

---

## 1. 형식 정의: 뉴런이 모나드가 아니라 전이가 monadic하다

### 1.1 관측과 상태

Recording $r$에서 다음을 관측한다.

$$
X_t^{(r)}
=
\left(x_{1t}^{(r)},\ldots,x_{N_rt}^{(r)}\right)^\top,
\qquad
U_t^{(r)}
=
\left(
v_t^{(r)},\kappa_t^{(r)},\ldots
\right).
$$

- $x_{it}$: 뉴런 $i$의 causal neural feature
- $U_t$: 속도, 곡률 등 행동 및 공통 nuisance
- $h_{it}$: 뉴런 $i$의 국소 기억상태
- $z_t$: 뉴런 집단의 cloud 상태
- $S_t=(h_{1t},\ldots,h_{N_rt},z_t)$: 전체 CloudCell 상태

CloudCell의 최소 형식은 다음 확률 kernel이다.

$$
K_\Delta:
\mathcal S\times\mathcal C
\longrightarrow
\mathcal D
\left(
\mathcal S\times\mathcal Y
\right),
$$

여기서 $\mathcal D$는 확률분포 monad이고, $\mathcal C$는 관측 가능한 control,
$\mathcal Y$는 다음 신경상태 또는 행동이다. 따라서 모나드라는 말은 뉴런의
생물학적 종류가 아니라 계산이 갖는 확률 효과의 타입을 가리킨다.

두 전이의 Kleisli 합성은 다음이다.

$$
\left(K_{\Delta_2}\star K_{\Delta_1}\right)(s,A)
=
\int_{\mathcal S}
K_{\Delta_2}(s',A)
K_{\Delta_1}(s,ds').
$$

확률 kernel을 올바르게 구현하면 identity와 associativity는 구조적으로
따른다. 데이터가 추가로 검사할 수 있는 것은 다음 Chapman-Kolmogorov
일관성이다.

$$
D
\left[
K_{2\Delta}(\cdot\mid s),
\int
K_\Delta(\cdot\mid s')
K_\Delta(ds'\mid s)
\right]
\le \varepsilon_{\mathrm{comp}}.
$$

이 equivalence gate는 `monad 법칙의 발견`이 아니라, 학습된 한 단계 상태가
두 단계 합성에서도 충분한지를 보는 경험적 Markov closure 검사다.

### 1.2 Operational CloudCell 가설

검증할 가설은 다음 세 부분이다.

$$
\begin{aligned}
H_{\mathrm{local}} &: h_{i,t-1}\ \text{가}\ x_{i,t+1}\ \text{예측에 남는다},\\
H_{\mathrm{cloud}} &: z_{t-1}\ \text{가 local history 뒤에도 남는다},\\
H_{\mathrm{full}} &: (h_{i,t-1},z_{t-1})\ \text{가 각각의 축약형을 이긴다}.
\end{aligned}
$$

세 번째 항이 이 문서의 핵심이다. Population decoder 하나가 잘 되는 것만으로는
CloudCell을 증명하지 않는다. 느린 calcium autocorrelation 하나가 잘 되는
것도 local-state와 cloud의 결합을 증명하지 않는다.

---

## 2. C. elegans 원자료 계약

### 2.1 현재 확보 자료

원자료 archive는 OSF 공개본 `AML310_moving.tar.gz`이고 실제 내부 root는
`AKS297.51_moving`이다. 다운로드 크기는 `348,444,164 B`, SHA-256은
`144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a`로
OSF API 값과 일치했다. OSF node에는 명시 라이선스가 없으므로 원자료는
재배포하지 않고 URL·크기·해시만 보존한다.

| recording id | neurons | cut frames | evaluated targets | median usable transitions |
|---|---:|---:|---:|---:|
| `BrainScanner20200130_105254_MS` | 128 | 1,525 | 122 | 910 |
| `BrainScanner20200130_110803_MS` | 134 | 1,466 | 134 | 1,318 |
| `BrainScanner20200310_141211_MS` | 116 | 1,495 | 109 | 751 |
| `BrainScanner20200310_142022_MS` | 97 | 1,480 | 97 | 1,303 |

각 recording은 최소한 다음 내부 schema로 변환한다.

```text
recording_id
subject_id
condition
source_file_sha256
time[T]
neural[T, N]
behavior[T, Q]
segment_id[T]
neuron_quality[N]
field_mapping
```

`time`, `neural`, `behavior`의 길이가 다르거나 timebase mapping이 명시되지
않으면 해당 recording은 실행 전에 실패한다. 행동 field가 없더라도 neural
transition은 실행할 수 있지만, 그 경우 behavior-conditioned 분석과 보조
decoder는 `NOT_EVALUABLE`로 기록한다.

### 2.2 Causal feature만 허용한다

미래 $x_{i,t+1}$을 예측할 때 feature는 시각 $t$까지의 값만 사용할 수 있다.

허용되는 차분은 다음 backward difference다.

$$
\dot x^-_{i,t}
=
\frac{x_{i,t}-x_{i,t-1}}{t_t-t_{t-1}}.
$$

중앙차분, 대칭 Gaussian derivative, 전체 시계열을 앞뒤로 보는 smoothing은
금지한다. Smoothing이 필요하면 train/test 각각의 연속 segment 안에서
one-sided kernel만 사용한다.

$$
\tilde x_{i,t}
=
\sum_{\ell=0}^{L_f-1}w_\ell x_{i,t-\ell},
\qquad
w_\ell\ge0,
\quad
\sum_\ell w_\ell=1.
$$

이미 대칭 smoothing된 field밖에 없으면 두 선택지만 허용한다.

1. raw field에서 causal feature를 다시 만든다.
2. 해당 분석을 `offline descriptive`로 낮추고 future-prediction 증거로 쓰지
   않는다.

`centerline.mat`에서 속도나 방향 변화를 새로 만들 때도 같은 원칙을 쓴다.
시간 미분은 backward difference로 계산하고, 결측 구간을 미래 frame으로
interpolate하지 않는다. 같은 frame의 공간 좌표만으로 계산되는 곡률은 허용하지만,
그 값을 neural timebase에 맞출 때는 future-aware interpolation을 금지한다.

### 2.3 불연속 timebase 처리

`crop_noncontig` 계열은 제거된 구간 때문에 이웃 index의 실제 $\Delta t$가
일정하지 않을 수 있다. 다음 조건에서 새 segment를 시작한다.

$$
t_{j+1}\le t_j
\quad\text{or}\quad
t_{j+1}-t_j
>
3\,\operatorname{median}(\Delta t).
$$

새 segment에서는 모든 lag window, causal filter, recurrent state를 reset한다.
Window가 segment 경계를 건너면 해당 표본을 버린다. Gap 양쪽은 최소

$$
E
\ge
\max
\left(
L_{\mathrm{history}},
L_f,
h
\right)
$$

frame만큼 embargo한다.

---

## 3. 핵심 2×2 모델 비교

### 3.1 공통 입력

모든 모델은 같은 target, timepoints, behavior history와 nuisance를 사용한다.
행동 history는

$$
B_t
=
\left[
1,
U_t,
U_{t-1},
\ldots,
U_{t-L_u+1}
\right]
$$

로 둔다. 미래 $U_{t+1}$은 입력에 들어가지 않는다. Primary neural gate는
movement/reafference만으로 population effect가 생기는 반례를 막기 위해
$B_t$를 공통 조건으로 넣는다.

뉴런 $i$의 국소 history는

$$
L_{i,t}
=
\left[
x_{i,t},
x_{i,t-1},
\ldots,
x_{i,t-L_x+1},
\dot x^-_{i,t}
\right].
$$

Cloud feature는 outer-train에서만 fit한 PCA로 만든다.

$$
z_t
=
P_{k,\mathrm{train}}^\top
\left(
X_t-\mu_{\mathrm{train}}
\right),
\qquad
Z_t
=
\left[
z_t,z_{t-1},\ldots,z_{t-L_z+1}
\right].
$$

$\mu_{\mathrm{train}}$, scale, PCA loading, rank $k$는 test data를 보지 않는다.

### 3.2 네 모델

모든 식은 $x_{i,t+1}$을 예측한다.

Memoryless/current-state model:

$$
M_0:
\qquad
\widehat x_{i,t+1}
=
\alpha_i
+a_i x_{i,t}
+\beta_i^\top B_t.
$$

Local-only state model:

$$
M_L:
\qquad
\widehat x_{i,t+1}
=
\alpha_i
+a_i^\top L_{i,t}
+\beta_i^\top B_t.
$$

Population-cloud-only model:

$$
M_P:
\qquad
\widehat x_{i,t+1}
=
\alpha_i
+c_i^\top Z_t
+\beta_i^\top B_t.
$$

Full local-cell + cloud model:

$$
M_F:
\qquad
\widehat x_{i,t+1}
=
\alpha_i
+a_i^\top L_{i,t}
+c_i^\top Z_t
+\beta_i^\top B_t.
$$

이 비교에서 $M_F$가 $M_L$을 이기는 것은 cloud의 조건부 증분,
$M_F$가 $M_P$를 이기는 것은 local state의 조건부 증분이다. 둘 다 통과해야
`local + cloud`라고 쓴다.

추가적인 비선형 interaction

$$
\sum_\ell
d_{i\ell}^\top
\left(
x_{i,t-\ell}z_{t-\ell}
\right)
$$

은 secondary model로만 둔다. Primary 결과를 본 뒤 interaction을 추가하면
post-hoc model이므로 confirmatory claim에 쓰지 않는다.

### 3.3 Ridge와 자유도 공정성

각 모델은 같은 squared-error family와 같은 hyperparameter budget을 쓴다.
Full model의 목적함수는 다음 group-ridge다.

$$
\min_\theta
\sum_{(i,t)\in\mathrm{train}}
\left(
x_{i,t+1}-\widehat x_{i,t+1}
\right)^2
+
\lambda_L\lVert a\rVert_2^2
+
\lambda_P\lVert c\rVert_2^2
+
\lambda_U\lVert\beta\rVert_2^2.
$$

각 $\lambda$, history 길이와 PCA rank는 outer-train 안의 inner blocked CV에서만
고른다. 모델마다 최적 regularization을 별도로 허용하되, 탐색 grid와 validation
예산은 동일하게 한다.

복잡도 반례를 막기 위해 다음 sensitivity도 같이 기록한다.

$$
\operatorname{df}_\lambda
=
\operatorname{tr}
\left[
D
\left(
D^\top D+\lambda I
\right)^{-1}
D^\top
\right].
$$

$M_F$와 비슷한 effective degrees of freedom를 갖도록 lag/rank를 늘린
parameter-matched $M_L^\ast$, $M_P^\ast$가 $M_F$를 이기면 구조적 우위 주장은
실패한다.

---

## 4. Leakage-safe split

### 4.1 Split 순서

분할은 반드시 다음 순서다.

1. subject와 recording을 먼저 고정한다.
2. 각 recording을 timestamp 기준 연속 segment로 나눈다.
3. segment 안에서 chronological outer train/validation/embargo/test를 만든다.
4. outer train 안에서만 inner blocked CV를 만든다.
5. split을 고정한 뒤 lag window를 생성한다.

Random frame split은 금지한다. Trial 자료에서는 frame보다 trial을 먼저
분할한다. 같은 trial, 같은 overlapping window, filter support가 양쪽 split에
동시에 들어가면 leakage다.

권장 기본 분할은 다음이다.

```text
outer train: first 60%
inner validation: next 20%
embargo/warm-up: protocol-fixed buffer
outer test: final 20%
```

Rolling-origin split은 robustness로 추가할 수 있으나, primary final test를 본
뒤 fold 수나 경계를 바꾸지 않는다.

### 4.2 Train-only 연산

다음은 모두 outer-train에서만 fit한다.

- 결측 대치값
- neuron mean/scale
- behavior mean/scale
- PCA와 rank
- neuron quality threshold 중 neural value-dependent 부분
- ridge penalty
- lag/history 길이
- feature 또는 latent-axis 선택
- early stopping과 model seed 선택

Test 구간의 timestamp로 segment 경계를 찾는 것은 허용한다. Timestamp는 target
값이 아니기 때문이다. Test 값으로 normalization하거나 PCA axis의 부호,
순서, rank를 다시 맞추는 것은 금지한다.

### 4.3 Cross-recording 판정

Recording은 독립 반복 단위이고 frame은 독립 표본이 아니다. 현재 네 recording은
각각 별도 effect를 낸다.

Confirmatory cross-recording claim은 다음을 모두 요구한다.

1. untouched recording 또는 subject가 최소 8개다.
2. full-vs-best-simple effect의 exact sign test가 단측 $p<0.05$다.
3. 최소 7/8 recording에서 효과 방향이 양수다.
4. recording-equal-weight aggregate CI의 하한이 사전등록 SESOI보다 크다.
5. leave-one-recording-out aggregate의 부호가 바뀌지 않는다.
6. 여러 lab 자료라면 leave-one-lab-out에서도 부호가 유지된다.

현재 4개는 이 조건을 만족시킬 수 없으므로 숫자가 좋아도 `pilot`이다.

---

## 5. Primary metric, effect와 신뢰구간

### 5.1 Neural transition metric

뉴런별 scale 차이를 막기 위해 outer-train baseline variance로 정규화한다.

$$
\mathcal L_{x,r}(M)
=
\frac1{N_r}
\sum_{i=1}^{N_r}
\frac{
\sum_{t\in\mathrm{test}}
\left(
x_{i,t+1}-\widehat x^{(M)}_{i,t+1}
\right)^2
}{
\sum_{t\in\mathrm{test}}
\left(
x_{i,t+1}-\overline x_{i,\mathrm{train}}
\right)^2
+\epsilon
}.
$$

Recording $r$, comparator $c$에 대한 full-model 상대 개선은

$$
d_{r,c}
=
\frac{
\mathcal L_{x,r}(M_c)-\mathcal L_{x,r}(M_F)
}{
\mathcal L_{x,r}(M_c)
}.
$$

세 축약형을 모두 이기는 보수적 효과는

$$
d_r^{\min}
=
\min_{c\in\{0,L,P\}}
d_{r,c}.
$$

Primary SESOI는 scoring 전에 고정한다. 별도 pilot noise estimate가 없으면
기본값은 상대 loss 감소 $1\%$다.

$$
\delta_x=0.01.
$$

단순 유의성뿐 아니라 simultaneous 95% CI의 하한이 $\delta_x$를 넘어야 한다.

### 5.2 계층적 불확실성

시간 frame을 i.i.d. bootstrap하지 않는다.

1. 각 recording test에서 autocorrelation time보다 긴 contiguous block을
   stationary bootstrap한다.
2. recording 안에서 paired prediction 차이를 계산한다.
3. subject를 최상위 cluster로 resample하고, 그 아래 recording과 time block을
   resample한다.
4. 세 comparator에 대한 max-$t$ simultaneous CI를 계산한다.

Recording 길이에 비례해 가중하면 긴 worm 하나가 결론을 지배하므로 aggregate는
recording 또는 subject equal weight를 primary로 둔다. Frame-weighted 수치는
secondary로만 보고한다.

### 5.3 Secondary metrics

- neuron-median held-out $R^2$
- Gaussian predictive NLL 또는 calibrated variance를 쓴 NLL
- $h\in\{1,5,10\}$ frame multi-horizon loss
- calcium autocorrelation time 이후 horizon의 effect
- two-step rollout 대 direct two-step transition의 composition error
- neuron dropout 10%, 30%, 50%에서의 loss curve

한 frame 예측만 통과하고 calcium autocorrelation time 이후에는 cloud 증분이
사라지면 `slow indicator persistence` 이상으로 승격하지 않는다.

---

## 6. Null과 shuffled control

Null은 test score만 한 번 섞는 것이 아니라 preprocessing, PCA, inner selection,
ridge fit을 포함한 전체 pipeline을 매 permutation마다 다시 실행한다.
Permutation 수는 confirmatory 실행에서 최소 1000회다.

| control | 보존하는 것 | 깨뜨리는 것 | 필요한 결과 |
|---|---|---|---|
| cell-wise independent circular shift | 각 뉴런의 분포와 autocorrelation | 뉴런 간 동시성/cloud | full-vs-local gain이 null 수준으로 하락 |
| population common shift vs behavior | neural population dynamics | neural-behavior alignment | behavior decoder gain이 하락 |
| block/phase surrogate | 저주파 power와 장기 autocorrelation | 정확한 전이 위상 | real transition이 우세 |
| neuron dropout | 남은 뉴런의 실제 시계열 | 일부 population support | full이 완만하게 저하 |
| parameter-matched local | 자유도와 history budget | cloud 구조 | full이 여전히 우세 |
| behavior-history conditioning | 운동과 task history | 단순 reafference 설명 | cloud 증분 유지 |
| GFP/motion control | 영상·운동 artifact | calcium-specific neural signal | GCaMP에서 더 큰 gain |

Circular shift는 gap을 건너지 않고 각 continuous segment 안에서 수행한다.
Shift 크기는

$$
|s_i|
>
\max
\left(
L_x,L_z,h,2\tau_{\mathrm{AC}}
\right)
$$

를 만족해야 한다. Pointwise target permutation은 autocorrelation 구조를
파괴하므로 primary null로 쓰지 않는다.

---

## 7. C. elegans 결과 표

### 7.1 Recording별 neural transition

아래 값은 target-neuron별 held-out $R^2$의 recording 중앙값이다. $M_0$는
현재 target 상태 $x_i(t)$만, $M_L$은 $x_i(t),x_i(t-1),x_i(t-2)$, $M_P$는
target을 제외한 population PCA, $M_F$는 $M_L+M_P$를 쓴다.

| recording | $R^2(M_0)$ | $R^2(M_L)$ | $R^2(M_P)$ | $R^2(M_F)$ | $R^2_F-R^2_L$ | positive target fraction | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `105254` | 0.978552 | 0.999397 | -0.100001 | 0.999271 | -0.0000977 | 0.139 | `FAIL` |
| `110803` | 0.970132 | 0.999134 | 0.184907 | 0.999084 | -0.0000224 | 0.351 | `FAIL` |
| `141211` | 0.940050 | 0.997805 | -0.857539 | 0.997469 | -0.0002658 | 0.211 | `FAIL` |
| `142022` | 0.961289 | 0.998722 | -0.311110 | 0.998582 | -0.0001252 | 0.196 | `FAIL` |

### 7.2 Aggregate

| contrast | equal-recording mean $\Delta R^2$ | simultaneous 95% CI | positive recordings | exact sign $p$ | status |
|---|---:|---|---:|---:|---|
| full vs current-only | +0.036096 | not estimated | 4/4 | 0.0625 | local-history increment only |
| full vs local-only | -0.0001278 | not needed for rejection | 0/4 | 1.0 | `FAIL` |
| full vs cloud-only | +1.267254 | not estimated | 4/4 | 0.0625 | local state required |
| full vs best simple | -0.0001278 | not needed for rejection | 0/4 | 1.0 | `FAIL` |

뉴런을 독립 반복으로 세지 않았다. 따라서 neuron-level 표본 수가 커도
recording-level 유의성으로 바꾸지 않는다. 더 중요한 점은 full-vs-local의
방향이 네 recording 모두 음수라서 CI나 더 많은 permutation을 기다리지
않고도 현재 primary joint-state gate가 실패한다는 것이다.

### 7.3 Robustness와 null

| check | real effect | null/control effect | CI or $p$ | pass |
|---|---:|---:|---|---|
| causal split/train-only transforms | four chronological held-out blocks | -- | deterministic audit | `PASS` |
| held-out common cloud shift | $\Delta R^2_{\rm align}>0$ in 4/4 at $h=1$ | inference-time shift only | confirmatory whole-pipeline null not run | diagnostic only |
| behavior-conditioned | not evaluated | not evaluated | -- | `OPEN` |
| parameter-matched local | ridge-selected local baseline | exact df matching not run | -- | `OPEN` |
| multi-horizon full-vs-local | $h=6$: -0.0096 to -0.0835; $h=30$: -0.094 to -0.242 | -- | 0/4 positive at both horizons | `FAIL` |
| neuron dropout | not evaluated | not evaluated | -- | `OPEN` |
| formal composition | finite $\mathcal D$ and $T_S$ laws exact in tests | malformed mass/state rejected | 10 focused tests | `PASS` |

### 7.4 현재 판정

```text
FORMAL KERNEL:              PASS / Exact finite state-probability contract
WITHIN-RECORDING DYNAMICS:  FAIL / 0 of 4 recordings
LOCAL HISTORY INCREMENT:    PASS direction / 4 of 4 recordings
CLOUD GIVEN LOCAL:          FAIL direction / 0 of 4 recordings
FULL JOINT INCREMENT:       FAIL / local-only is sufficient here
SHUFFLE SPECIFICITY:        diagnostic only; confirmatory null not needed for rejection
CROSS-RECORDING STATUS:     not eligible; only four-recording within-dataset pilot
CURRENT CLAIM LEVEL:        formal software Exact; biological joint CloudCell rejected on this dataset
```

해석은 “population activity가 정보가 없다”가 아니다. $M_P$가 일부 기록에서
신호를 가지며 time-aligned cloud가 shifted cloud보다 나은 경우도 있다.
그러나 사전 핵심 질문은 cloud가 이미 강한 local history 위에 독립 예측력을
더하느냐였고, 그 답은 세 horizon 모두 `아니오`였다.

### 7.5 실패 뒤 식 변형 loop

Primary test를 본 뒤의 변형은 모두 `exploratory`로 동결한다. 같은 test에서
가장 좋은 식을 고른 뒤 증명이라고 부르지 않는다.

먼저 additive 식

$$
V_0:\qquad \widehat x_{i,t+h}=f_L(L_{i,t})+c_i^\top Z_t
$$

을 local innovation 식으로 바꿨다.

$$
V_1:\qquad
e_{i,t+h}=x_{i,t+h}-f_L(L_{i,t}),
\qquad
\widehat e_{i,t+h}=c_i^\top Z_t.
$$

그다음 local state가 cloud gain을 조절하도록 했다.

$$
V_2:\qquad
\widehat e_{i,t+h}
=
c_i^\top Z_t
+d_i^\top\!\left(x_{i,t}Z_t\right)
+q_i^\top\!\left(\Delta x_{i,t}Z_t\right).
$$

$V_3$은 여기에 $Z_t^2$와 $\tanh Z_t$를 추가했다. $h=1$에서 recording별
median $\Delta R^2_{\rm cloud|local}$은 다음과 같다.

| variant | `110803` | `105254` | `141211` | `142022` | positive recordings |
|---|---:|---:|---:|---:|---:|
| $V_0$ additive | -0.0000224 | -0.0000977 | -0.0002658 | -0.0001252 | 0/4 |
| $V_1$ innovation | +0.0000007 | -0.0000188 | -0.0000417 | -0.0000057 | 1/4 |
| $V_2$ interaction | -0.0000395 | -0.0001485 | -0.0002882 | -0.0000633 | 0/4 |
| $V_3$ nonlinear | -0.0000881 | -0.0003298 | -0.0005360 | -0.0000520 | 0/4 |

항을 늘릴수록 개선되지 않았으므로 per-neuron 식 확장은 여기서 중단했다.
마지막으로 ontology를 individual neuron에서 population latent로 낮춰

$$
z_{t+1}=Az_t,
\qquad
z_{t+h}\stackrel{?}{=}A^h z_t
$$

의 Markov closure를 검사했다. persistence와 비교한 direct-transition
$\Delta R^2$와 composition gap을 함께 요구했다.

| latent state | passed recordings | 결론 |
|---|---:|---|
| $s_t=z_t$ | 1/4 | `FAIL` |
| $s_t=(z_t,z_{t-1})$ | 2/4 | `FAIL` |
| order 3 | 2/4 | `FAIL` |
| order 4 | 2/4 | `FAIL` |

order 2 이후 개선이 포화됐고 같은 두 recording이 계속 실패했다. 따라서
더 높은 차수나 더 큰 비선형식으로 이 dataset을 맞추는 loop는 중단한다.
현재 선택 가능한 정직한 식은 joint CloudCell이 아니라

$$
\boxed{
x_{i,t+h}=f_i(x_{i,\le t})+\varepsilon_{i,t+h},
\qquad
Z_t\ \text{is optional population context, not a demonstrated sufficient state.}
}
$$

이다. 다음 식 승격은 아직 보지 않은 recording에서만 가능하다.

---

## 8. 행동 decoding은 보조 게이트다

행동을 neural feature로 같은 시각에 복원하는 것은 population code의 존재를
보일 수 있지만, stateful CloudCell의 핵심 증거는 아니다. Primary는
$X_t\rightarrow X_{t+1}$ transition이다.

보조 행동 게이트는 다음 미래식으로 둔다.

$$
\widehat U_{t+1}
=
g
\left(
U_{\le t},
L_t,
Z_t
\right).
$$

Behavior-only autoregressive baseline

$$
\widehat U_{t+1}^{(B)}
=
g_B(U_{\le t})
$$

을 모든 neural model보다 먼저 비교한다. Velocity와 curvature는 별도 endpoint로
보고하고 Holm correction을 적용한다.

| target | behavior-only | local-only | cloud-only | full | full-vs-best effect | corrected $p$ | status |
|---|---:|---:|---:|---:|---:|---:|---|
| velocity $t+1$ | not evaluated | not evaluated | not evaluated | not evaluated | -- | -- | `OPEN/secondary` |
| curvature $t+1$ | not evaluated | not evaluated | not evaluated | not evaluated | -- | -- | `OPEN/secondary` |

같은 시각 decoding을 함께 계산할 수는 있지만 `offline association`으로
명시한다. Neural derivative가 미래 sample을 사용하거나 full-series
normalization이 들어간 결과는 이 표의 future gate에 넣지 않는다.

---

## 9. 왜 기존 최소 decoder만으로 부족한가

현재 `reality_stone.clarus.evidence.linear_decoder_gate`는 외부 데이터 연결을
확인하는 최소 도구로는 유용하지만 CloudCell 확증에는 부족하다.

1. 한 recording을 앞 70%, 뒤 30%로 한 번만 나눈다.
2. ridge가 고정되어 있고 nested tuning이 없다.
3. permutation이 train target을 pointwise shuffle하여 시계열
   autocorrelation을 보존하지 않는다.
4. gap reset과 filter-support embargo가 없다.
5. local-only, cloud-only, full의 2×2 ablation이 없다.
6. recording을 독립 반복 단위로 취급하는 계층적 CI가 없다.
7. 동시 행동 decoding이 중심이고 미래 neural transition을 직접 묻지 않는다.

Upstream PredictionCode 방식도 그대로 confirmatory pipeline으로 쓰지 않는다.
검사된 upstream 경로에는 전체 시계열 normalization 뒤 split하는 방식,
대칭 Gaussian derivative, 불연속 timepoints 제거 뒤의 nonuniform timebase가
포함될 수 있다. 원 논문의 offline decoding을 재현하는 데는 적합할 수 있지만,
causal future-state proof에는 train-only normalization, backward derivative,
gap reset이 별도로 필요하다.

---

## 10. Human MTL coded-memory 교차 게이트

C. elegans dynamics가 통과해도 그것만으로 `coded memory`를 증명하지 않는다.
인간 medial temporal lobe 자료에서는 local concept coding과 population cloud를
분리해 두 단계로 검사한다.

### 10.1 필요한 데이터 계약

```text
subject_id
session_id
unit_id
brain_region
trial_id
stimulus_id
stimulus_variant_or_repeat_id
event_time
spike_counts[trial, unit, causal_time_bin]
recognition_or_recall_outcome
confidence
reaction_time
task_covariates
```

동일 subject/session에서 동시에 기록된 unit만 population으로 묶는다. 서로 다른
session의 unit을 한 시각에 기록된 것처럼 합친 pseudo-population은 primary
CloudCell 근거로 금지한다.

### 10.2 Gate H1: local coded-cell 재현

Training trial에서만 unit-concept pair와 response window를 선택한다.

$$
\Delta_{i,c}^{\mathrm{train}}
=
\mathbb E
\left[
x_{ij}^{\mathrm{evoked}}
\mid c_j=c
\right]
-
\mathbb E
\left[
x_{ij}^{\mathrm{evoked}}
\mid c_j\ne c
\right].
$$

선택된 pair는 untouched variant/repeat에서 다음을 통과해야 한다.

$$
\Delta_{i,c}^{\mathrm{test}}>0,
\qquad
\mathrm{AUC}_{i,c}^{\mathrm{test}}>0.5,
\qquad
p_{\mathrm{stimulus\ block}}<0.05.
$$

단순히 $\mathrm{AUC}>0.5$인 것만으로는 실질 효과를 보장하지 않는다.
Confirmatory 실행 전 $\mathrm{AUC}_{\min}>0.5$인 SESOI와 unit-concept pair
다중검정의 FWER 또는 FDR 보정법을 고정한다.

다른 사진, 글자, 음성 등 같은 개념의 variant가 있다면 leave-variant-out을
primary로 둔다. 모든 trial에서 가장 잘 반응한 뉴런을 고른 뒤 같은 trial로
평가하는 것은 circular selection이므로 무효다.

| dataset/subject | selected pairs in train | held-out variants | held-out AUC/effect | block-null $p$ | status |
|---|---:|---:|---|---:|---|
| Sternberg NWB panel | N/E — concept identity label 없음 | N/E — concept variant 없음 | -- | -- | `NOT EVALUATED` |

### 10.3 Gate H2: local code 뒤의 population-cloud 증분

Trial $j$의 later memory outcome를 $Y_j$라 두고 네 모델을 다시 비교한다.

$$
\begin{aligned}
M_0 &: Y_j\sim\text{stimulus/task/history},\\
M_L &: Y_j\sim M_0+\text{train-selected local coded-cell state},\\
M_P &: Y_j\sim M_0+\text{train-PCA population state},\\
M_F &: Y_j\sim M_0+\text{local state}+\text{population state}.
\end{aligned}
$$

Primary metric은 held-out log loss, secondary는 Brier score와 balanced accuracy다.
Log-loss effect는

$$
\Delta\mathrm{LL}_{F-B}
=\mathrm{LL}(M_B)-\mathrm{LL}(M_F),
\qquad B\in\{0,L,P\},
$$

로 정의해 양수가 full의 개선을 뜻하게 한다.
Split은 최소 두 종류를 모두 보고한다.

- leave-stimulus-variant/repeat-out
- leave-session-out 또는 leave-subject-out aggregate

Population rank, local-cell selection, regularization은 inner train에서만 결정한다.
Human MTL의 neuron identity가 subject 사이에서 공유되지 않으므로 named PCA axis를
강제로 맞추지 않는다. Subject별 effect를 낸 뒤 계층적으로 합치거나, unit order에
불변인 set encoder를 사전등록해 사용한다.

| contrast | held-out log-loss effect | simultaneous 95% CI | positive subjects | corrected $p$ | status |
|---|---:|---|---:|---:|---|
| full vs task/history | -- | -- | -- | -- | `NOT EVALUATED` |
| full vs local coded-cell | -- | -- | -- | -- | `NOT EVALUATED` |
| full vs population-only | -- | -- | -- | -- | `NOT EVALUATED` |

현재 Sternberg panel 실행에는 concept-selected cell과 later-memory outcome가 없고,
PCA population 및 위 log-loss contrast도 사용하지 않았다. 따라서 아래 proxy
결과를 이 H2 표에 대입하지 않는다.

### 10.4 Human MTL null

- stimulus label을 subject/session block 안에서 permutation
- firing-rate와 isolation quality가 맞는 non-selected unit
- best-single-neuron baseline
- 동시성을 깨는 unit-wise trial permutation
- trial 수와 parameter 수가 맞는 local-only model
- previous exposure, reaction time, confidence, stimulus frequency를 포함한
  task/history baseline

H1만 통과하면 `local sparse/concept code`다. H2에서 full이 task/history,
local-only, population-only를 모두 이겨야 `coded CloudCell bridge`로 올린다.

### 10.5 완료된 Sternberg operational proxy

이 분석은 concept-cell H1/H2의 대체 검증이 아니다. Memory-load cardinality와
probe membership의 population-vs-best-single decoding, 그리고 maintenance
early-to-late firing-rate persistence를 평가한 exploratory operational proxy다.
`n_shifts=1000` 요청에서 가능한 모든 서로 다른 5-trial block offset이 사용되어
subject별 유효 shift 수는 26 또는 21이었다.

| subject | trials | MTL units | load $\Delta$BA / null $p$ | probe $\Delta$BA / null $p$ | persistence full-vs-best $\Delta R^2$ | status |
|---|---:|---:|---:|---:|---:|---|
| 18 | 135 | 64 | -0.104843 / 0.851852 | +0.223684 / 0.111111 | -0.693539 | `FAIL` |
| 19 | 108 | 11 | +0.019048 / 0.545455 | +0.051923 / 0.318182 | -0.053921 | `FAIL` |
| 7 | 108 | 16 | +0.039881 / 0.454545 | -0.076923 / 0.909091 | -0.830916 | `FAIL` |
| 20 | 135 | 1 | -- | -- | -- | `NOT EVALUABLE` |

평가 가능한 subject는 3명이고 operational pass는 `0/3`이다. Subject 18의 probe
population gain은 기술적으로 크지만 block-shift null $p=0.111111$이므로
통과가 아니다. 이 convenience panel에서 subject-independent population 추론은
하지 않는다. 2026-08-04 all-shift 결과는 현재
`.tmp/cloudcell-data/human_mtl_recheck_all_shifts.json`에만 있으므로 tracked
artifact로 승격하기 전까지 `exploratory/non-durable`이다.

---

## 11. 인과 한계와 필요한 개입

관측 시계열에서 full model이 앞서도 다음 대안은 남는다.

- 측정되지 않은 stimulus feature
- proprioception과 motor efference copy
- 공통 arousal 또는 neuromodulation
- calcium indicator의 느린 impulse response
- 행동이 neural population을 되먹임하는 역방향
- electrode/unit sampling bias

따라서 temporal precedence와 conditional prediction은 Granger형 예측 근거이지
생물학적 인과 증명이 아니다.

Causal claim에는 train data에서 정의한 ensemble 또는 cloud-state 방향을
closed-loop로 교란해야 한다.

$$
\operatorname{do}
\left(
z_t\leftarrow z_t+\delta v
\right)
\Longrightarrow
\begin{cases}
\Delta X_{t+h}\ \text{가 사전 예측 방향과 일치},\\
\Delta U_{t+h}\ \text{가 사전 예측 방향과 일치}.
\end{cases}
$$

같은 neuron 수, 총 spike/light power, 해부학 위치를 맞춘 random ensemble,
sham, 반대 방향 $-\delta v$를 control로 둔다. 결과가 전체 각성 변화로만
설명되거나 random ensemble과 차이가 없으면 causal CloudCell은 실패한다.

---

## 12. Claim ladder

| level | 필요한 gate | 현재 값 | 허용 문장 |
|---|---|---|---|
| `L0 FORMAL` | 확률 kernel, state type, 합성 법칙 | `PASS / Exact` | CloudCell을 Markov/Kleisli module로 정의 |
| `L1 IMPLEMENTED` | 네 모델, split, leakage tests, artifact hash | `PARTIAL PASS` — primary gate·hash 구현, whole-pipeline bootstrap null은 미구현 | 검증기가 현재 명세 범위에서 동작 |
| `L2 WITHIN-RECORDING` | 네 C. elegans recording의 future transition | `FAIL 0/4` — full이 local-only를 이기지 못함 | joint CloudCell 문장 불허 |
| `L3 CROSS-RECORDING` | 최소 8 untouched recording, exact sign/CI | `NOT EVALUATED` | 불허 |
| `L4 CROSS-SPECIES/CODED` | human MTL concept H1+H2 | `NOT EVALUATED` — Sternberg proxy `FAIL 0/3`, sub-20 N/E | coded-memory/CloudCell bridge 불허 |
| `L5 CAUSAL` | matched perturbation | `NOT EVALUATED` | 불허 |
| `L6 AGI EFFICACY` | 동일 parameter/FLOP AGI ablation | `NOT EVALUATED` | 불허 |

낮은 level 통과를 높은 level 문장으로 바꾸지 않는다.

---

## 13. Kill rules

다음 중 하나면 해당 claim을 즉시 낮춘다.

1. Causal filter가 미래 sample을 사용한다.
2. normalization, PCA, rank, lag, ridge가 outer test를 본다.
3. lag/filter window가 train-test 또는 gap 경계를 건넌다.
4. full이 $M_0$, $M_L$, $M_P$ 중 하나라도 simultaneous CI 기준으로 이기지
   못한다.
5. full-vs-best-simple CI 하한이 사전등록 SESOI를 넘지 못한다.
6. parameter-matched local/population model에서 우위가 사라진다.
7. behavior history를 조건으로 넣으면 cloud 증분이 사라진다.
8. cell-wise shift null에서도 같은 cloud gain이 나온다.
9. one-step에서만 통과하고 indicator autocorrelation 이후 horizon에서 전부
   사라진다.
10. 결과가 한 recording 또는 한 subject 제거 시 반전된다.
11. confirmatory panel이 exact sign test 또는 hierarchical CI를 통과하지
    못한다.
12. Human MTL local-cell 선택이 holdout trial까지 사용한다.
13. 서로 다른 session의 unit을 동시 population처럼 합쳐야만 H2가 통과한다.
14. raw hash, split manifest, test access log가 없거나 변경된다.
15. 예측 결과를 causal 또는 ontological monad 증명으로 확대 해석한다.

분석 도중 kill rule을 수정하면 기존 자료는 exploratory로 고정하고, 새 protocol과
새 untouched holdout으로 다시 시작한다.

---

## 14. 구현 가능한 API와 산출물

### 14.1 최소 Python API

```python
@dataclass(frozen=True)
class NeuralRecording:
    recording_id: str
    subject_id: str
    time: NDArray
    neural: NDArray       # [time, neuron]
    behavior: NDArray     # [time, target]
    segment_id: NDArray
    source_hashes: dict[str, str]


@dataclass(frozen=True)
class BlockedSplit:
    train_index: NDArray
    validation_index: NDArray
    warmup_index: NDArray
    test_index: NDArray
    embargo_frames: int


def load_recording_from_mat(...) -> NeuralRecording: ...
def build_continuous_segments(...) -> NDArray: ...
def make_blocked_split(...) -> BlockedSplit: ...
def fit_causal_train_preprocessor(...) -> object: ...
def build_local_and_cloud_features(...) -> dict[str, NDArray]: ...
def fit_nested_ridge(model_kind, ...) -> object: ...
def score_neural_transition(...) -> dict: ...
def run_shift_null(...) -> dict: ...
def aggregate_recording_effects(...) -> dict: ...
def validate_cloudcell_artifact(...) -> dict: ...
```

`model_kind`는 정확히 다음 네 값을 갖는다.

```text
memoryless
local_only
population_only
full
```

### 14.2 실행 전 invariant

실행기는 다음을 assertion으로 막아야 한다.

```text
all timestamps finite and strictly increasing within segment
no lag window crosses segment_id
no fit index occurs in validation/warmup/test
normalizer.fit_indices == outer_train_indices
pca.fit_indices == outer_train_indices
ridge_selection_indices subset outer_train_indices
max feature timestamp <= target timestamp - horizon
all four models share identical scored targets
all source hashes match manifest
```

### 14.3 결과 artifact

```text
artifact_type: clarus_cloudcell_dynamics_gate
artifact_version
protocol_hash
code_commit
source_hashes
field_mapping
causal_filter
segment_rules
split_indices_hash
model_grids
selected_hyperparameters_by_recording
per_recording_predictions_hash
per_recording_metrics
paired_effects
bootstrap_configuration
null_configuration
aggregate_result
claim_level
violated_kill_rules
test_access_log
```

최종 보고서는 aggregate 평균 하나만 저장하지 않고 각 recording의 loss,
prediction hash, hyperparameter, effect를 반드시 남긴다.

### 14.4 실행 명령

2026-07-31 primary artifact는 다음 명령으로 생성했다.

```powershell
.tmp/cloudcell-venv/Scripts/python.exe `
  reality_stone/python/reality_stone/clarus/cloudcell_dynamics.py `
  .tmp/cloudcell-data/extracted/AKS297.51_moving `
  --archive .tmp/cloudcell-data/AML310_moving.tar.gz `
  --expected-sha256 144126ee9a49d311c3393deea434e1a0963d55de35318e25d98d48f9c175250a `
  --output artifacts/agi/cloudcell_celegans_dynamics.json `
  --quiet
```

`h=6`, `h=30` 진단은 `--horizon-steps 6`과 `--horizon-steps 30`으로
별도 artifact를 만들었다. 현재 CLI는 artifact 재검증만 하는 독립 모드를 아직
제공하지 않으므로 `L1`을 부분 통과로 유지한다.

---

## 15. 현재 최종 판정

```text
FORMAL SOFTWARE CONTRACT:       PASS / Exact
C. ELEGANS CLOUD INCREMENT:     FAIL / 0 of 4 recordings
HUMAN MTL STERNBERG PROXY:      FAIL / 0 of 3 evaluable subjects
HUMAN CONCEPT-CODE H1+H2:       NOT EVALUATED
CAUSAL / ONTOLOGICAL MONAD:     NOT IDENTIFIED
OVERALL STRONG CODED-MONAD:     NOT SUPPORTED
WEAK LOCAL TEMPORAL FEATURE:    SUPPORTED / AML32 7 of 7 at h=1 and h=6
```

> Causal held-out transition에서 full model은 local-only를 네 recording 모두에서
> 이기지 못했다. equal-recording mean $\Delta R^2_{\rm full-local}$은
> $-1.28\times10^{-4}$이고 방향 통과는 `0/4`였다. 약 1초와 5초 진단에서도
> 각각 네 recording 모두 음수였다. 따라서 현재 C. elegans 자료는 target의
> 현재 상태를 넘는 local history는 지지하지만, 그 위의 독립 population-cloud
> 증분은 지지하지 않는다. `뉴런 = coded monadic CloudCell`이라는 생물학적
> 동일성은 이 관측 자료로 식별되지 않았고, 사전 정의한 C. elegans operational
> joint-state gate는 이 데이터 경계에서 `FAIL`; 남는 것은
> `stateful local units embedded in a population`이라는 더 약한 설계 가설이다.

형식 결과와 경험 결과를 섞지 않는다. 확률·상태 monad와 완전 snapshot에 대한
소프트웨어 정리는 `Exact/PASS`지만, 그것은 실제 뉴런이 그 객체라는 증명이
아니다.

---

## 16. 2026-07-31 후속 루프 결론

후속 우선순위 루프의 완전한 수치와 판정은 다음 문서로 분리했다.

- `23_Graph_Dynamics_Loop.md`: directed graph, AML310 0/4
- `24_Manifold_Diffusion_Loop.md`: diffusion, AML310 최대 1/4
- `25_Local_Temporal_Memory_Confirmation.md`: local temporal memory,
  untouched AML32에서 $h=1$ 7/7 및 $h=6$ 7/7

따라서 원래 CloudCell 명제는 두 부분으로 분해된다.

$$
\underbrace{\text{뉴런 측정 신호가 시간적 local state를 가진다}}
_{\text{확인 패널에서 지지}}
\quad\not\Rightarrow\quad
\underbrace{\text{뉴런이 coded monadic CloudCell이다}}
_{\text{현재 미증명}}.
$$

AGI에 의미 있는 최소 가정은 local temporal-history/state feature이며,
population cloud, graph, diffusion은 이번 자료에서 그 위의 필수 구조로
확인되지 않았다.

> **후기 (2026-08-12).** `25_Local_Temporal_Memory_Confirmation.md`의 계산
> 게이트 PASS는 유지되나, 동일 게이트를 활동 비의존 GFP 대조군
> AML18(11 recording)에 사전등록 후 적용한 결과 $h=1$·$h=6$ 모두 11/11
> PASS였고 효과 크기가 AML32보다 컸다. 따라서 위 분해에서 "뉴런 측정
> 신호가 시간적 local state를 가진다"는 문장은 "전처리된 형광 시계열이
> lag 예측정보를 가진다"로 좁혀 읽어야 하며, 신경 기억이라는 생물학적
> 해석은 강등되었다(25번 문서 머리 지위 변경 블록 참조). 이 경계는
> §12 claim ladder의 L2 계열 해석에도 동일하게 적용된다. headroom 교정
> 통계량($\delta'$, median-of-ratios) 기반 사전등록 본 실행에서도
> GCaMP·GFP 분포 겹침이 확정되었다(주검정 $h=6$ exact one-sided
> Mann–Whitney $p=0.535$; `artifacts/agi/local_memory_gfp_matched_result.json`).

2026-08-04 AML32 원자료 재실행에서는 $h=1$과 $h=6$ 결과 JSON이 기존 커밋의
두 confirmatory artifact와 각각 정확히 일치했고, 구현 정규화 SHA-256
`6032a76d33a0a2adc2671972248c550b00c600be418919c3c296924106475f86`에 대해
독립 verifier가 `proof_passed=true`, 오류 0개를 반환했다. 이것은 measured
same-unit history의 held-out 예측 정보에 대한 재현이지, 인과적 recurrence의
증명이 아니다.

---

## 17. 비증거적 설계 힌트: private inference와 shared intervention

수행자에게만 물이 땅처럼 성립한다는 서사와 다른 사람도 건너는 다리의 대비는
물리적 증거가 아니라 다음 AGI 판별식을 주는 설계 은유로만 사용한다. 세계상태를
$s$, agent $i$의 내부상태를 $z_i$, 관측 kernel을 $O_i$, 환경 전이를 $T$라 두면

$$
\begin{aligned}
\text{private inference}:\quad
&z_i\to z_i',\quad O_i,\pi_i\ \text{변화},\quad T'=T,\\
\text{shared intervention}:\quad
&s'\sim T(\cdot\mid s,a_i),\quad
\exists j\ne i:\ \Delta\mathrm{utility}_j>0.
\end{aligned}
$$

CloudCell의 경험적 핵심은 내부 표상의 인상적인 변화가 아니라 다른 단위의 정보를
썼을 때 untouched future가 더 잘 예측되는지다.

$$
G_{i\leftarrow -i}
=R^2(M_F)-R^2(M_L)>\varepsilon.
$$

현재 C. elegans에서는 이 shared gain이 네 recording 모두 음수였고, local-history
gain만 양성이었다. 따라서 AGI 구현에서도 우선 local temporal state를 유지하되,
shared workspace는 parameter/FLOP-matched 외부 과제와 cross-module ablation에서
전이 가능한 증분을 보일 때만 필수 구조로 승격한다.
