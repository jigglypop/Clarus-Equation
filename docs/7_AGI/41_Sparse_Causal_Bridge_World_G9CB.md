# G9-CB: 희소 인과 브리지 세계모형

> 상태: V1/V3 validation `FAIL` 보존, V2/V4 validation 및 locked test `PASS`
>
> 사전등록: `../../experiments/preregistration/sparse_causal_bridge_v1.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v2.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v3.json`,
> `../../experiments/preregistration/sparse_causal_bridge_v4.json`
>
> 구현: `../../reality_stone/python/reality_stone/clarus/sparse_causal_bridge.py`
> 및 `../../reality_stone/python/reality_stone/clarus/latent_causal_bridge.py`
>
> 실행: `../../examples/agi/sparse_causal_bridge_gate.py`,
> `../../examples/agi/latent_causal_bridge_gate.py`
>
> 테스트: `../../tests/test_sparse_causal_bridge.py`,
> `../../tests/test_latent_causal_bridge.py`

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

V3는 $\gamma$까지 각 80-step OOD prefix에서 다시 추정했다. 기전과
subspace는 정확했지만 일부 seed의 짧은 AR 추정이 불안정했다.

- graph precision/recall: `1.00 / 1.00`
- self / bridge coefficient MAE: `0.00099 / 0.00471`
- loading cosine: `0.99876`
- rank-1 설명분산: `88.97%`
- global 감소 vs V1 방식 / no-latent: `13.30% / 57.45%`
- global 감소 vs fixed-local: `0.59%` — 등록 기준 5% 실패
- downstream 감소 vs fixed-local: `13.55%` — 등록 기준 15% 실패
- mean AR error: `0.10935` — 등록 상한 0.08 실패
- paired CI95 lower vs fixed-local: `-0.00849` — 실패

V3 artifact와 구현 SHA는 보존했고 test는 열지 않았다.

### 9.4 V4 — 공유 시간상수, 변하는 loading

V4의 변경은 하나다. $\gamma$는 긴 observational-train residual 전체에서
episode 경계를 넘지 않고 한 번만 적합한다. OOD prefix는 residual 중심,
rank-1 방향, scalar intercept만 적응한다. 즉

\[
\underbrace{D,B,\gamma}_{\text{environment-invariant}}
\quad+\quad
\underbrace{u_e,\bar r_e,c_e}_{\text{environment-adaptive}}
\]

로 분리한다. train에서 얻은 $\widehat\gamma=0.93693$이고 생성값 0.96과의
절대오차는 0.02307이다.

| 지표 | V4 validation 20 seed | V4 locked test 30 seed | 판정 |
|---|---:|---:|---|
| 선택 edge | `A→C`, `C→D` | 동일 | PASS |
| graph precision / recall | 1.00 / 1.00 | 1.00 / 1.00 | PASS |
| self coefficient MAE | 0.00128 | 0.00128 | PASS |
| bridge coefficient MAE | 0.00240 | 0.00240 | PASS |
| intervention NRMSE | 0.08838 | 0.08799 | PASS |
| loading cosine mean | 0.99918 | 0.99921 | PASS |
| loading cosine seed 최소 | 0.99710 | 0.99536 | PASS |
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

V4 validation/test의 merged registration SHA는
`f9f8a0d9c3a9203e91a675db01775564f05f360557c814bf31e2770017c98a91`다.
V4 lock은 이 SHA뿐 아니라 실제 알고리즘/DGP 두 파일의 SHA도 함께 검사했다.

- `latent_causal_bridge.py`:
  `40306162c5d266a8ecb80d882202afb92fbc45aa5bc467a1004721706e57eef7`
- `sparse_causal_bridge.py`:
  `0885d7244c3ea35367987ec59538d15c081d3ba6009897e5d0e5e42a24538ca7`

따라서 V2의 soft lock보다 강하지만, git revisionㆍPython/NumPy binaryㆍOS까지
고정한 완전한 재현 환경 hash는 아니다.

## 10. 무엇이 입증됐고 무엇이 아닌가

현재 V4까지 지지하는 가장 강한 문장은 다음뿐이다.

> 등록된 4-chart rank-1 합성 family에서 paired intervention은 programmed
> $A\to C,\ C\to D$와 대각 기전을 복원하고 거짓
> $A\leftrightarrow B$를 배제했다. $D,B,\gamma$를 불변 기전으로,
> 80-step OOD residual 방향을
> 변하는 문맥으로 분리한 V4는 bridge target 직접 잠재교란과 loading-sign
> shift에서 등록 기준선보다 낮은 sequential one-step 오차를 보였다.

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

또한 생성기와 learner가 같은 `tanh(source)` basis를 쓴다. V4는 직전
residual만 쓰는 순차 one-step으로 누설은 막았지만, 매 step의 실제 상태를
다시 받는다. 함수족 불일치와 free multi-step rollout은 아직 열려 있다.
paired-do도 같은 hidden/process noise가 정확히 소거되는 이상적 합성
randomized experiment다.

## 11. 재현

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_sparse_causal_bridge.py -q --basetemp .tmp/g9cb-unit
.\.venv\Scripts\python.exe examples/agi/sparse_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v2.json --split validation
.\.venv\Scripts\python.exe examples/agi/sparse_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v2.json --split test
.\.venv\Scripts\python.exe -m pytest tests/test_latent_causal_bridge.py -q --basetemp .tmp/g9cb-v4
.\.venv\Scripts\python.exe examples/agi/latent_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v4.json --split validation
# 위 validation artifact의 configㆍimplementation SHA가 모두 같은 경우에만
.\.venv\Scripts\python.exe examples/agi/latent_causal_bridge_gate.py --config experiments/preregistration/sparse_causal_bridge_v4.json --split test
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

외부 다운로드, GPU, trajectory 파일은 모두 0이다.

## 12. 다음 반증 순서

1. 실제 상태 재입력을 끊은 5/20-step free rollout에서 latent filter가
   오차 누적을 늦추는지 사전등록한다.
2. 여러 intervention amplitude로 local 선형ㆍ비선형 항을 분리하고,
   source basis를 일부러 misspecify한다.
3. rank-2 latent, latent AR 자체가 바뀌는 환경, 부분관측 chart에서 실패
   경계를 찾는다.
4. 고정 chart gate를 통과한 뒤에만 G6 adaptive chart discovery와 G7 replay를
   결합한다.
5. 실제 U-fiber tractography matched-control은 현재 자료ㆍ비용 한계 때문에
   `SKIPPED_COST`로 둔다. 저비용 공개 파생치가 생기기 전에는 생물학적
   연결 주장으로 승격하지 않는다.
