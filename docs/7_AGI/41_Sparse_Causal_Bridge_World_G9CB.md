# G9-CB: 희소 인과 브리지 세계모형

> 상태: V1/V3/V5/V6/V7 validation `FAIL` 보존, V2/V4 locked test `PASS`,
> V5/V6/V7 locked test 미개봉
>
> 사전등록: `../../experiments/preregistration/sparse_causal_bridge_v1.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v2.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v3.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v4.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v5.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v6.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v7.json`
>
> 구현: `../../reality_stone/python/reality_stone/clarus/sparse_causal_bridge.py`
> 및 `../../reality_stone/python/reality_stone/clarus/latent_causal_bridge.py`,
> `../../reality_stone/python/reality_stone/clarus/free_rollout_bridge.py`,
> `../../reality_stone/python/reality_stone/clarus/consensus_rollout_bridge.py`,
> `../../reality_stone/python/reality_stone/clarus/invariant_prior_rollout_bridge.py`
>
> 실행: `../../examples/agi/sparse_causal_bridge_gate.py`,
> `../../examples/agi/latent_causal_bridge_gate.py`,
> `../../examples/agi/free_rollout_bridge_gate.py`,
> `../../examples/agi/consensus_rollout_bridge_gate.py`,
> `../../examples/agi/invariant_prior_rollout_bridge_gate.py`
>
> 테스트: `../../tests/test_sparse_causal_bridge.py`,
> `../../tests/test_latent_causal_bridge.py`,
> `../../tests/test_free_rollout_bridge.py`,
> `../../tests/test_consensus_rollout_bridge.py`,
> `../../tests/test_invariant_prior_rollout_bridge.py`,
> `../../tests/test_invariant_prior_bridge_integrity.py`

## 1. 왜 방향을 바꿨는가

G9-F는 고랑 기하가 Yeo 대규모 기능망 경계를 직접 나눈다는 단순 명제를
지지하지 않았다. G9-B는 반대로 접힘이 일부 표면상 원거리 쌍에 희소한
3차원 지름길 **후보**를 만들 수 있음을 보였지만, 낙관적 white chord일 뿐
실제 U-fiber나 정보 전달을 입증하지 못했다.

따라서 계산 가설을 다음처럼 축소했다.

> 기하는 연결을 확정하지 않는다. 이산 Laplace--Beltrami 열핵과 접촉
> 점수는 검사할 후보만 줄인다. 연결의 방향ㆍ계수는 무작위 개입으로
> 독립 확인하고, 최종 채택은 관측 holdout gain과 개입 기준을 함께 쓴다.

이는 사용자가 제안한 “고랑이 영역을 나누기보다 멀리 있던 계산 영역의
접촉을 강화할 수 있다”는 생각의 가장 작은 계산 대응실험이다. 실제 뇌
자료를 쓰지 않으므로 생물학적 검증은 아니다.

## 2. 최소 세계와 식

차트 identity 발견 문제를 섞지 않고 네 차트 $A,B,C,D$를 고정한다.
숨은 공통원인 $h_t$와 관측 상태 $x_t\in\mathbb R^4$는

\[
h_{t+1}=\rho h_t+\sigma_h\xi_t,
\]

\[
x_{t+1}
=D x_t+B\tanh(x_t)+\Lambda_e h_t+\sigma_x\epsilon_t
\]

를 따른다. 행렬 방향은 $B_{ij}=B[\text{target}=i,\text{source}=j]$다.
V2의 실제 방향성 bridge는

\[
B_{CA}=0.52,\qquad B_{DC}=-0.48
\]

뿐이다. $H\to A,H\to B$는 존재하지만 $A\leftrightarrow B$ 직접 edge는
없다. OOD에서는

\[
\Lambda_{\mathrm{train}}=(1.15,1.25,0,0)^\top,
\qquad
\Lambda_{\mathrm{ood}}=(1.15,-1.25,0,0)^\top
\]

처럼 $H\to B$의 부호만 바꾼다. 진짜 $B$는 고정이다. 결합계의 최대
고윳값 절댓값은 $0.96<0.98$이므로 등록된 안정성 gate를 만족한다.

### 2.1 관측 상관이 틀릴 수 있는 이유

선형화한 관측 회귀는 일반적으로

\[
\widehat W_e
\approx D+B+
\Lambda_e\operatorname{Cov}(h_t,x_t)
\operatorname{Cov}(x_t)^{-1}
\]

를 포함한다. 마지막 항은 환경 $e$에 따라 변하는 nuisance다. V2 train의

\[
|\operatorname{corr}(A,B)|=0.96554
\]

는 매우 크지만 직접 edge는 없다. 실제 raw-correlation top-2는 거짓
$B\to A$를 선택했고 OOD에서 깨졌다.

## 3. Laplace--Beltrami는 후보만 제안한다

네 차트의 표면 인접행렬을 $W$, degree를 $Q$라 두고 이산
Laplace--Beltrami 대용 연산자를

\[
L=Q-W
\]

로 둔다. $L$은 대칭 positive semidefinite이고 열핵은

\[
K_\tau=e^{-\tau L},\qquad \tau=0.5
\]

다. 사전고정한 extrinsic white-contact affinity를 $C_{ij}$, 열핵의
off-diagonal 정규화를 \(\widetilde K_{ij}\)라 하면 fold 후보 점수는

\[
S_{ij}=C_{ij}(1-\widetilde K_{ij})
\]

다. 즉 white-contact는 가깝지만 표면 열확산으로는 가깝지 않은 쌍을 먼저
검사한다. 상위 네 undirected pair는

\[
\{A,C\},\{A,B\},\{C,D\},\{B,D\}
\]

이고 양방향으로 펼쳐 여덟 후보를 만든다. 이 구성은 true pair를 사전에
높은 affinity로 포함한 양성대조다. 따라서 “LB가 연결을 발견했다”거나
표본 효율을 높였다는 주장은 허용되지 않는다.

## 4. 방향과 계수의 개입 식별

각 source $j$에 대해 같은 base state, 같은 $h_t$, 같은 process noise를
보존한 채

\[
x_{j,t}^{+}=+\delta,\qquad x_{j,t}^{-}=-\delta,
\qquad \delta=0.75
\]

로 surgical replacement한다. $i\ne j$이면 paired difference는

\[
Y_i^+-Y_i^-
=2B_{ij}\tanh(\delta)+\eta_i
\]

다. 숨은 원인과 공통 process noise는 정확히 소거되고, 독립 sensor noise
$\eta_i$만 남는다. 따라서

\[
\widehat B_{ij}
=\frac{1}{n}\sum_{r=1}^{n}
\frac{Y_{i,r}^{+}-Y_{i,r}^{-}}
{2\tanh(\delta)}
\]

는 이 합성 family에서 $B_{ij}$의 불편 추정량이다.

선택 규칙은 truth label을 보지 않고 다음 네 조건을 모두 요구한다.

1. $i\leftrightarrow j$가 LB proposal에 포함된다.
2. observational holdout의 one-edge gain이 1% 이상이다.
3. $|\widehat B_{ij}|\ge0.15$, $z\ge3$이다.
4. probe를 반으로 나눈 두 추정치의 부호가 같고 directed budget 2 안이다.

최종 causal 모델은 개입으로 추정한 bridge 계수를 고정하고, 그 효과를 뺀
관측 train residual에서 local basis $[1,x_i,x_i^3]$만 적합한다.

## 5. 데이터 역할과 정보비용

| 역할 | V2 seed | 크기 | 선택에 사용 |
|---|---|---:|---|
| observational train | 15100--15107 | 8×520 step | 계수 적합 |
| observational selector holdout | 15600--15603 | 4×360 step | 관측 gain |
| topology intervention probe | 16100--16103 | 512 paired do | causal edgeㆍ계수 |
| validation OOD | 17100--17119 | 20×360 step | 평가만 |
| locked test OOD | 18100--18129 | 30×360 step | 평가만 |

`causal_bridge`와 `dense_probe`만 512개의 topology probe를 추가로 받는다.
따라서 이는 동일 데이터 예산에서의 순수 알고리즘 우월성 비교가 아니라
**개입 정보의 가치와 그 정보를 희소화하는 효과**를 시험한다. 같은 probe를
받는 dense-probe가 공정성 대조군이다.

## 6. 기준선

- `local_only`: cross-chart edge 0
- `dense_observational`: 관측으로 12개 방향 edge 모두 적합
- `raw_correlation_top2`: raw association 상위 2개
- `predictive_gain_top2`: local residual 예측 gain 상위 2개
- `geometry_observational_top2`: LB 후보 안에서 관측 gain 상위 2개
- `dense_probe`: 같은 개입 probe로 12개 계수를 추정
- `causal_bridge`: LB 후보를 개입으로 prune/orient한 최대 2개 edge
- `oracle_diagnostic`: truth 사용 상한이며 경쟁 모델이 아님

## 7. 루프 장부

### V1 — `FAIL`, test 미개봉

V1은 $H$가 실제 bridge target $C$에도 직접 들어가게 만들었다. bridge
두 개는 정확히 복원했다.

- 선택: $A\to C,C\to D$
- precision/recall: `1.00 / 1.00`
- coefficient MAE: `0.00148`
- intervention NRMSE: `0.08485`

그러나 local mechanism은 관측 회귀로 남겨 숨은 원인이 local 계수에
흡수됐다. OOD에서 causal global RMSE `0.19481`은 local `0.16899`보다
15.28% 나빴고, downstream도 41.78% 나빴다. bridge를 제거하는 lesion이
오히려 오차를 줄였으므로 예측 gate를 명확히 실패했다.

별도로 permutation control은 label을 실제 estimator 입력에서 끊지 못한
구현 결함 때문에 실패했다. V1 artifact는 수정하거나 덮지 않았다. V1 test는
열지 않았다.

### V2 — 가설 축소와 구현 수정

V2는 bridge target의 직접 hidden loading을 0으로 두어 “bridge 식별”만
격리했다. 모든 train/probe/validation/test/control seed를 새로 바꿨고,
permutation은 paired outcome arm을 균형 있게 뒤집도록 수정했다. 또한
강한 observation-only selector가 같은 graph를 찾을 수 있으므로 우월성
gate를 비열등성 gate로 바꾼 사실을 사전등록했다.

## 8. V2 결과

| 지표 | validation 20 seed | locked test 30 seed | 판정 |
|---|---:|---:|---|
| 선택 edge | `A→C`, `C→D` | 동일 | PASS |
| precision / recall | 1.00 / 1.00 | 1.00 / 1.00 | PASS |
| probe-seed exact recovery | 4/4 | 4/4 | PASS |
| common-cause / reverse false edge | 0 / 0 | 0 / 0 | PASS |
| bridge coefficient MAE | 0.00403 | 0.00403 | PASS |
| intervention NRMSE | 0.08884 | 0.08867 | PASS |
| causal global RMSE | 0.15507 | 0.15594 | PASS |
| global 감소 vs local | 11.14% | 10.68% | PASS |
| global 감소 vs dense observational | 55.20% | 56.20% | PASS |
| global 감소 vs raw correlation | 44.26% | 45.25% | PASS |
| ratio vs predictive-gain | 0.999996 | 0.999998 | 비열등 |
| ratio vs same-probe dense | 1.000149 | 1.000516 | 비열등 |
| downstream 감소 vs local | 33.49% | 33.15% | PASS |
| downstream 감소 vs dense observational | 0.47% | 0.41% | 비열등 |
| lesion direct-target MSE 최소 증가 | 259.43% | 359.44% | PASS |
| lesion non-target 즉시 변화 | 0 | 0 | 구조상 PASS |
| no-edge / permuted control 선택 | 0 / 0 | 0 / 0 | PASS |
| wall time | 0.37 s | 0.49 s | PASS |

validation과 test는 같은 합성 train/probe에서 만든 하나의 모델을 서로 다른
OOD seed로 평가한다. config SHA는 둘 다
`be128195e8504a09cbf2ec58bfc2191a47fb9c7b7ef895d514729253b3c6c704`다.
평가 seed만 바꿔도 selection과 intervention estimate가 불변인 누설 방지
테스트를 통과했다.

test lock은 passing validation artifact와 config SHA를 확인하는 **soft
lock**이다. 코드ㆍDGP hash나 git commit까지 암호학적으로 고정하지 않으며
API에서 명시적으로 우회할 수 있다. 이번 실행에서는 validation 뒤 코드를
바꾸지 않고 test를 한 번 실행했지만, 이를 강한 cryptographic blind로
부르지 않는다. 실행 뒤 현재 소스ㆍ설정ㆍ결과의 hash는
`../../artifacts/agi/sparse_causal_bridge_integrity_v2.json`에 별도로 고정했다.
이 파일도 사후 attestation이지 사전 blind의 대체물이 아니다.

## 9. V3--V4: 직접 잠재교란과 문맥 필터

V2는 bridge target의 hidden loading을 제거해 구조 선택만 격리했다. 다음
루프는 V1의 어려운 조건

\[
\Lambda_{\mathrm{train}}=(1.15,1.25,0.55,0)^\top,
\qquad
\Lambda_{\mathrm{ood}}=(1.15,-1.25,-0.55,0)^\top
\]

을 복원했다. 이제 숨은 원인 $H$는 source $A$와 bridge target $C$에
동시에 들어간다.

### 9.1 대각 기전까지 개입으로 식별

source와 target이 같으면 paired difference는

\[
Y_j^+-Y_j^-=2\delta D_{jj}+\eta_j
\]

이므로 대각 자기동역학도

\[
\widehat D_{jj}=\frac{\overline{Y_j^+-Y_j^-}}{2\delta}
\]

로 추정한다. off-diagonal은 앞 절의 $\tanh$ 분모를 그대로 쓴다. 이렇게
얻은 불변 기전을

\[
\widehat f(x_t)=\widehat D x_t+\widehat B\tanh(x_t)
\]

라 두면 관측 residual은

\[
r_t=x_{t+1}-\widehat f(x_t)\approx\Lambda_e h_t+\epsilon_t
\]

다.

### 9.2 rank-1 residual filter

OOD seed마다 처음 80 transition만 calibration에 사용한다. 그 residual
공분산의 첫 고유벡터를 $u_e$라 하면 바뀐 loading subspace를 label 없이
추정할 수 있다. 현재 outcome을 미리 쓰지 않고 직전 residual만 사용해

\[
z_t=u_e^\top(r_t-\bar r_e),
\qquad
\widehat r_{t+1}
=\bar r_e+u_e(c_e+\gamma z_t)
\]

로 다음 문맥 residual을 예측한다. 최종 one-step 예측은

\[
\widehat x_{t+2}
=\widehat f(x_{t+1})+\widehat r_{t+1}
\]

다. 평가 prefix는 점수에서 제외하고, filter의
$\left(u_e,\bar r_e,c_e\right)$는 prefix
뒤에 동결한다.

### 9.3 V3 — `FAIL`, test 미개봉

V3는 $\gamma$까지 각 80-step OOD prefix에서 다시 추정했다. graph는
exact였고 기전계수 오차와 subspace 오차는 작았지만, 일부 seed의 짧은 AR
추정이 불안정했다.

- graph precision/recall: `1.00 / 1.00`
- self / bridge coefficient MAE: `0.00099 / 0.00471`
- loading-subspace $|\cos|$: `0.99876`
- rank-1 설명분산: `88.97%`
- global 감소 vs V1 방식 / no-latent: `13.30% / 57.45%`
- global 감소 vs fixed-local: `0.59%` — 등록 기준 5% 실패
- downstream 감소 vs fixed-local: `13.55%` — 등록 기준 15% 실패
- mean AR error: `0.10935` — 등록 상한 0.08 실패
- paired CI95 lower vs fixed-local: `-0.00849` — 실패

V3 artifact와 구현 SHA는 보존했고 test는 열지 않았다.

### 9.4 V4 — 공유 시간상수, 변하는 loading

V4의 알고리즘 변경은 하나이며 protocol상 모든 seed도 새로 배정했다.
$\gamma$는 긴 observational-train residual 전체에서 episode 경계를 넘지
않고 한 번만 적합한다. OOD prefix는 residual 중심, rank-1 subspace,
scalar intercept만 적응한다. 즉

\[
\underbrace{D,B,\gamma}_{\text{environment-invariant}}
\quad+\quad
\underbrace{u_e,\bar r_e,c_e}_{\text{environment-adaptive}}
\]

로 분리한다. train에서 얻은 $\widehat\gamma=0.93693$이고 생성값 0.96과의
절대오차는 0.02307이다.

| 지표 | V4 validation 20 seed | V4 locked test 30 seed | 판정 |
|---|---:|---:|---|
| train/probe에서 동결한 edge | `A→C`, `C→D` | 같은 frozen model | PASS |
| frozen graph precision / recall | 1.00 / 1.00 | 같은 1.00 / 1.00 | PASS |
| self coefficient MAE | 0.00128 | 0.00128 | PASS |
| bridge coefficient MAE | 0.00240 | 0.00240 | PASS |
| intervention NRMSE | 0.08838 | 0.08799 | PASS |
| loading-subspace $|\cos|$ mean | 0.99918 | 0.99921 | PASS |
| loading-subspace $|\cos|$ seed 최소 | 0.99710 | 0.99536 | PASS |
| rank-1 설명분산 | 89.98% | 89.93% | PASS |
| causal latent global RMSE | 0.15712 | 0.15578 | PASS |
| global 감소 vs fixed-local | 7.14% | 6.58% | PASS |
| global 감소 vs V1 방식 | 18.73% | 20.22% | PASS |
| global 감소 vs no-latent mechanism | 58.73% | 61.35% | PASS |
| global 감소 vs adaptive-dense prefix | 13.10% | 13.52% | PASS |
| downstream 감소 vs fixed-local | 15.48% | 15.07% | PASS |
| paired CI95 lower vs fixed-local | +0.01109 | +0.01001 | PASS |
| lesion direct-target MSE 최소 증가 | 75.30% | 46.31% | PASS |
| wall time | 2.03 s | 2.98 s | PASS |

validation/test는 동일한 고정 OOD loading과 하나의 frozen train/probe 모델을
공유하고 평가 noise seed만 다르다. 따라서 test graph를 다시 식별한 것도,
새 loading family로 일반화한 것도 아니다.

V4 validation/test의 merged registration SHA는
`f9f8a0d9c3a9203e91a675db01775564f05f360557c814bf31e2770017c98a91`다.
V4 lock은 이 SHA뿐 아니라 실제 알고리즘/DGP 두 파일의 SHA도 함께 검사했다.

- `latent_causal_bridge.py`:
  `40306162c5d266a8ecb80d882202afb92fbc45aa5bc467a1004721706e57eef7`
- `sparse_causal_bridge.py`:
  `0885d7244c3ea35367987ec59538d15c081d3ba6009897e5d0e5e42a24538ca7`

따라서 V2의 soft lock보다 강하지만, git revisionㆍPython/NumPy binaryㆍOS까지
고정한 완전한 재현 환경 hash는 아니다.

## 10. V5: 실제 상태 재입력을 끊은 자유전개

### 10.1 누설 없는 단일 origin 정의

V4는 매 step의 실제 $x_t$를 다시 받아 one-step 예측했다. V5는 OOD
prefix $x_0,\ldots,x_{80}$만 관측하고, $C=80$에서 한 번 시작한다.
마지막 관측 residual로 초기 score를 만들고

\[
r_{C-1}=x_C-\widehat f(x_{C-1}),
\qquad z_{C-1}=u^\top(r_{C-1}-\bar r),
\]

그 뒤에는 실제 미래 상태나 residual을 읽지 않고

\[
\widehat z_{C+k}=c+\gamma\widehat z_{C+k-1},
\quad
\widehat r_{C+k}=\bar r+u\widehat z_{C+k},
\quad
\widehat x_{C+k+1}
=\widehat f(\widehat x_{C+k})+\widehat r_{C+k}
\]

만 20번 반복한다. H5는 이 하나의 H20 trajectory 앞 다섯 row다.
rollout API는 `Episode`, future outcome, hidden state를 인자로 받지 않으며,
$x_{81:100}$과 hidden을 변조해도 비-oracle 예측이 bit-identical인 테스트를
추가했다.

### 10.2 V5 validation — `FAIL`, test 미개봉

V5는 V4의 frozen mechanism을 사용하고 20개의 새 validation seed에서
평가했다. 비교의 독립 단위는 step이 아니라 seed이며 Student-$t$ CI를 썼다.

| 지표 | H5 | H20 | 판정 |
|---|---:|---:|---|
| causal-latent free path RMSE | 0.20678 | 0.33308 | 유한ㆍ안정 |
| persistence RMSE | 0.23166 | 0.38913 | 기준선 |
| no-latent mechanism RMSE | 0.39476 | 0.44420 | 기준선 |
| fixed-local RMSE | 0.23665 | 0.42220 | 기준선 |
| stable adaptive-dense RMSE | 0.21488 | 0.31270 | 기준선 |
| same-probe dense+latent RMSE | 0.20697 | 0.33402 | 동일 정보예산 대조 |
| 감소 vs persistence | 10.74% | 14.40% | 평균 기준 PASS |
| 감소 vs no-latent | 47.62% | 25.02% | PASS |
| 감소 vs fixed-local | 12.62% | 21.11% | PASS |
| persistence 대비 seed 승률 | 0.65 | 0.75 | H5 FAIL |
| persistence 대비 paired CI95 lower | -0.00679 | -0.01481 | 둘 다 FAIL |
| stable adaptive-dense 대비 | 3.77% 우수 | 6.52% 열세 | H20 FAIL |
| ratio vs same-probe dense+latent | 0.99910 | 0.99721 | 비열등 PASS |

따라서 평균 성능만 보면 자유전개가 작동했지만, seed 간 변동성과 강한
prefix-adaptive 기준선을 포함한 등록 계약은 통과하지 못했다. 실패 check는
정확히 네 개다.

- `h5_seed_wins_persistence`
- `h5_ci_persistence`
- `h20_ci_persistence`
- `h20_vs_stable_adaptive_dense`

예측은 모두 finite였고 최대 절댓값 1.04922, learned-mechanism Jacobian
spectral radius 최대 0.78142, H20/H5 RMSE ratio 1.61081이었다. 즉 실패
원인은 발산이나 미래 누설이 아니라 모델 간ㆍseed 간 예측 불확실성이다.
V5 validation artifact와 구현 hash를 보존했고 locked test는 열지 않았다.

### 10.3 V6 사전등록과 결과 비개봉 수정

V5 validation은 이후 후보 설계용 development data로만 사용했다. prefix
전체로 scalar Kalman posterior를 추정한 후보는 H5/H20 RMSE
`0.20730 / 0.33406`으로 V5보다 오히려 나빠 폐기했다.

V6 사전등록 후보는 세 독립 expert를 prefix 내부에서 backtest해
불확실성 가중치를 정하는 방식이다. pseudo-origin 60에서 causal-latent,
stable adaptive-dense, persistence를 각각 20-step 자유전개한다. 각 expert의
prefix 오차는 $E_j=(20d)^{-1}$ 곱하기 20개 leadㆍ4개 chart 제곱오차의 합으로
두고, $w_j$는 $(E_j+10^{-12})^{-1/2}$에 비례하도록 합 1로 정규화한다.
이후 전체 $x_{0:80}$으로 expert를 다시 적합하고, 세 expert가 각자 만든
20-step trajectory의 lead별 Euclidean barycenter만 취한다. 합의값을 다시
component 입력으로 넣지는 않는다.

V5 development에서 이 후보의 평균 weight는 causal/adaptive/persistence
`0.3552 / 0.3196 / 0.3253`, H5/H20 RMSE는
`0.20442 / 0.31930`이었다. 이 수치는 후보 선택 자료이지 증거가 아니다.
원래 V6 merged registration SHA는
`b73245484d1a8ff1e385cceb08cbf99105ef7db7d672531da677daba7fbc4eed`였다.
결과를 열기 전 `active_models_path=consensus.models`가 가리키는 동일한 9개
모델 목록을 누락 없이 복사하는 행정적 보정을 기록했다. 모델ㆍseedㆍ수식ㆍ
문턱ㆍ자료 역할은 바꾸지 않았고 보정 뒤 SHA는
`53c1cd432943573b4de227bf474718ad610bd5a22c2d4c82cd6e9cda482600be`다.

### 10.4 V6 validation — `FAIL`, test 미개봉

**[예측]** prefix backtest로 정한 세 expert의 convex consensus가 fresh V6
validation seed에서 V5 오차와 seed-level 변동성을 줄이는지를 등록된 51개
check로 판정했다.

**[산출]** validation 20 seed에서 34개 check는 통과하고 17개는 실패했다.
따라서 performance와 전체 판정은 모두 `FAIL`이며 locked test는 열지 않았다.

| 모델 | H5 path RMSE | H20 path RMSE |
|---|---:|---:|
| V6 consensus | 0.21635 | 0.38910 |
| 동일 V6 seed의 V5 | 0.20878 | 0.38424 |
| adaptive dense | 0.22730 | 0.40765 |
| persistence | 0.25067 | 0.42844 |
| same-probe dense consensus | 0.21646 | 0.38939 |
| no-latent | 0.40157 | 0.44871 |

V6는 V5보다 H5에서 3.63%, H20에서 1.26% 나빴다. persistence보다
13.69%/9.18% 좋았지만 H20 등록 기준 10%에는 못 미쳤다. V5 대비 seed
승률은 0.30/0.40이고, 평균 causal/adaptive/persistence weight는
`0.3663 / 0.3366 / 0.2971`이었다. prefix에서 잘한 expert가 미래에도
잘한다는 신호가 fresh seed로 안정적으로 전이되지 않았다.

비유한 예측과 미래 관측 read는 모두 0이고 최대 절댓값은 1.94437이었다.
다만 adaptive self coefficient 최대가 `0.98263 > 0.98`이어서 안정성 check도
실패했다. 즉 결과를 단순 폭주나 누설로 설명할 수 없고, 고정 arbitration의
전이 실패와 seed 변동성이 함께 남는다.

### 10.5 V7 사전등록 — 비영시차 불변 시간 prior

**[공리: 모델 선택]** V7은 V6 arbitration을 활성 후보에서 제거하고 V5의
희소 인과 expert를 다시 anchor로 둔다. 바꾸는 항목은 공유 scalar AR 추정기
하나다. train residual의 첫 주성분 score에서 episode 경계를 넘지 않는
자기공분산 \(\gamma_1,\ldots,\gamma_{20}\)을 계산하고

\[
\widehat\rho=
\frac{\sum_{k=1}^{19}\gamma_k\gamma_{k+1}}
{\sum_{k=1}^{19}\gamma_k^2}
\]

로 둔다. 독립 관측잡음이 분산을 더하는 \(\gamma_0\)는 사용하지 않고,
결과는 안정성 구간 `[-0.98, 0.98]`로 투영한다.
residual score의 단위를 \([x]\)라 하면 각 \(\gamma_k\)는 \([x]^2\), 분자와
분모는 모두 \([x]^4\)이므로 \(\widehat\rho\)는 무차원이다.

**[산출]** frozen train에서 V5 lag-zero OLS는 `0.936927`, 비영시차 추정은
`0.957925`다. programmed `0.96`은 진단에만 쓰며 candidate 함수가 읽을 수
없다. V6 validation을 개발 자료로 재사용한 pilot은 H5/H20
`0.20708 / 0.37849`였지만 이는 V7 증거가 아니다.

**[예측: 사전등록 당시]** 새 validation 80 seed `69100..69179`와 locked test 100 seed
`70100..70199`에서 V6 대비 H5/H20 2% 이상 개선, V5 평균 비열등, seed/paired
CIㆍpersistenceㆍsame-probe denseㆍ안정성 기준을 함께 판정한다. validation이
통과하기 전 test는 열지 않는다. merged registration SHA는
`44ad51cc5279581568e254617db2aa63b2adf42095f97aaaa94fabf7e9a54e02`다.

### 10.6 V7 validation — `FAIL`, test 미개봉

**[산출]** 고정 구현 SHA
`f7a64266ad167b58eecb5d3b27907f3c4f30b66fa660e50f52b88b27c72fc90f`로
validation seed 80개를 한 번 실행했다. 42개 성능 check 중 35개가 통과하고
7개가 실패했으며 resource check는 모두 통과했다.

| 모델 | H5 path RMSE | H20 path RMSE |
|---|---:|---:|
| V7 비영시차 prior | 0.22532 | 0.36797 |
| 동일 seed의 V6 consensus | 0.22838 | 0.37099 |
| 동일 seed의 V5 lag-zero prior | 0.22580 | 0.36182 |
| persistence | 0.25039 | 0.41854 |

V7은 V6보다 H5에서 1.34%, H20에서 0.81% 좋아졌지만 등록한 2% 문턱에는
도달하지 못했다. V6 대비 paired CI95 lower는 각각 `-0.00254`, `-0.00727`이고
seed win fraction은 `0.575`, `0.525`였다. V5 대비 H5 ratio는 `0.99786`이지만
H20 ratio는 `1.01701`, H20 seed win fraction은 `0.4125`였다. 따라서 더 정확한
scalar AR 추정이 이 misspecified rollout의 장기 오차를 자동으로 줄인다는
예측은 지지되지 않았다.

비영시차 AR은 `0.957925`로 programmed diagnostic `0.96`과의 절대오차가
`0.002075`였고, V5 OLS 오차의 `0.0899`배였다. leave-one-episode-out 표준편차와
범위는 `0.002273`, `0.006382`로 estimator gate를 통과했다. 예측의 미래 관측
read와 비유한 값은 모두 0, 최대 절댓값은 `1.59873`, H20/H5 오차비는
`1.63311`이었다. 즉 실패 원인은 AR 식별ㆍ수치 폭주ㆍtarget-window 누설이
아니라, 더 정확한 한 개의 시간상수가 전체 rollout 오차로 전이되지 않은 데
있다.

validation artifact SHA는
`cc8d0b6945499a374306cd1269322184b9f4d49bb315c22e90a4a77865bea06d`,
사후 integrity SHA는
`0354782ed6cb8a99ec4111a67a1ade4a7cd68c724f98934d5159229561a46108`다.
V6와 V7 locked test artifact는 모두 생성하지 않았다.

## 11. 무엇이 입증됐고 무엇이 아닌가

현재 V4까지 지지하는 가장 강한 문장은 다음뿐이다.

> 등록된 4-chart rank-1 합성 family에서 paired intervention은 programmed
> $A\to C,\ C\to D$와 대각 기전을 복원하고 거짓
> $A\leftrightarrow B$를 배제했다. $D,B,\gamma$를 불변 기전으로,
> 80-step OOD residual 방향을
> 변하는 문맥으로 분리한 V4는 bridge target 직접 잠재교란과 loading-sign
> shift에서 등록 기준선보다 낮은 sequential one-step 오차를 보였다.

**[산출]** V5--V7은 실제 미래 상태를 읽지 않고 단일 origin에서 finite
20-step trajectory를 만들었다. 그러나 세 버전 모두 등록된 강건성 계약을
통과하지 못했다. V6 consensus는 fresh seed에서 V5를 개선하지 못했고,
V7의 더 정확한 scalar AR도 H20에서 V5보다 1.70% 나빴다.

다음 주장은 금지한다.

- 개입이 `predictive_gain_top2`보다 우월했다: 둘은 같은 graph와 사실상
  같은 RMSE다.
- LB가 bridge를 발견했다: affinity와 4/6 pair budget에 true pair가
  설계상 들어 있다.
- 임의의 latent target confounding을 해결했다: V4는 rank 1, 같은 latent
  AR, loading 부호변화, 80-step calibration이라는 등록 family에 한정된다.
- lesion locality가 생물학적 모듈성을 발견했다: one-step 행렬의 다른
  output row가 변하지 않는 것은 구성상 자명하다.
- multi-step planning, adaptive chart, 실제 고랑/U-fiber, 뉴런 알고리즘,
  뇌 세계시뮬레이터 또는 AGI를 입증했다.
- V6가 arbitration의 강건성을 입증했다거나 persistence 성능이 invariant
  world prior의 존재를 입증했다.
- V7이 보편적 invariant world prior를 발견했다거나 더 정확한 AR이 더 좋은
  world model을 뜻한다.
- V5--V7이 기억 회상ㆍ꿈ㆍ행동 계획ㆍvalid-world manifold를 구현했다.
- 희소 인과 expert가 same-probe dense consensus보다 우월했다.

또한 생성기와 learner가 같은 `tanh(source)` basis를 쓴다. 20-step free
rollout 구현은 완료됐지만 V5--V7 robustness gate는 실패했다. 행동조건부
반사실 계획, episodic retrieval, world-validity constraint와 함수족ㆍlatent
rank 변화는 여전히 미구현이다.
paired-do도 같은 hidden/process noise가 정확히 소거되는 이상적 합성
randomized experiment다.

## 12. 재현

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_sparse_causal_bridge.py -q --basetemp .tmp/g9cb-unit
.\.venv\Scripts\python.exe examples/agi/sparse_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v2.json --split validation
.\.venv\Scripts\python.exe examples/agi/sparse_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v2.json --split test
.\.venv\Scripts\python.exe -m pytest tests/test_latent_causal_bridge.py -q --basetemp .tmp/g9cb-v4
.\.venv\Scripts\python.exe examples/agi/latent_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v4.json --split validation
# 위 validation artifact의 configㆍimplementation SHA가 모두 같은 경우에만
.\.venv\Scripts\python.exe examples/agi/latent_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v4.json --split test
.\.venv\Scripts\python.exe -m pytest tests/test_consensus_rollout_bridge.py tests/test_sparse_causal_bridge_integrity.py -q
.\.venv\Scripts\python.exe examples/agi/consensus_rollout_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v6.json --split validation
# V6 validation FAIL이므로 V6 locked test는 실행하지 않는다.
.\.venv\Scripts\python.exe -m pytest tests/test_invariant_prior_rollout_bridge.py tests/test_invariant_prior_bridge_integrity.py -q --basetemp .tmp/g9cb-v7
.\.venv\Scripts\python.exe examples/agi/invariant_prior_rollout_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v7.json --split validation
# V7 validation FAIL이므로 V7 locked test는 실행하지 않는다.
```

결과:

- `../../artifacts/agi/sparse_causal_bridge_validation_v1.json` — 보존된 실패
- `../../artifacts/agi/sparse_causal_bridge_validation_v2.json` — PASS
- `../../artifacts/agi/sparse_causal_bridge_test_v2.json` — PASS
- `../../artifacts/agi/sparse_causal_bridge_integrity_v2.json` — 사후 파일 hash
- `../../artifacts/agi/sparse_causal_bridge_validation_v3.json` — 보존된 실패
- `../../artifacts/agi/sparse_causal_bridge_validation_v4.json` — PASS
- `../../artifacts/agi/sparse_causal_bridge_test_v4.json` — PASS
- `../../artifacts/agi/sparse_causal_bridge_integrity_v4.json` — V4 파일 hash
- `../../artifacts/agi/sparse_causal_bridge_validation_v5.json` — 보존된 실패
- `../../artifacts/agi/sparse_causal_bridge_validation_v6.json` — 보존된 실패
- `../../artifacts/agi/sparse_causal_bridge_integrity_v6.json` — test 미개봉 증명
- `../../artifacts/agi/sparse_causal_bridge_validation_v7.json` — 보존된 실패
- `../../artifacts/agi/sparse_causal_bridge_integrity_v7.json` — V6/V7 test 미개봉 증명

외부 다운로드, GPU, trajectory 파일은 모두 0이다.

## 13. 다음 반증 순서

1. V7 실패를 시간상수 하나의 문제가 아니라 source-basisㆍresidual-model
   misspecification 반례로 보존하고, 같은 validation에 문턱을 맞추지 않는다.
2. episodic attractorㆍreplayㆍconstrained imagination은 42장의 독립 실험으로
   분리해 recall, prediction, planning을 섞지 않고 사전등록한다.
3. 여러 intervention amplitude로 local 선형ㆍ비선형 항을 분리하고,
   source basis를 일부러 misspecify한다.
4. rank-2 latent, latent AR 자체가 바뀌는 환경, 부분관측 chart에서 실패
   경계를 찾는다.
5. 고정 chart gate를 통과한 뒤에만 G6 adaptive chart discovery와 G7 replay를
   결합한다.
6. 실제 U-fiber tractography matched-control은 현재 자료ㆍ비용 한계 때문에
   `SKIPPED_COST`로 둔다. 저비용 공개 파생치가 생기기 전에는 생물학적
   연결 주장으로 승격하지 않는다.
