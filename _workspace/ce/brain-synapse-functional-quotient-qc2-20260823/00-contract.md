# BA-SRM3 — source-corrected synapse functional quotient contract

Date: 2026-08-23

Status: `ADAPTIVE_NEW_CANDIDATE / PREREG_FROZEN_BEFORE_RESPONSE_ONLY_SUPPORT / TRAIN_SUMMARIES_KNOWN / DEVELOPMENT_CONFIRMATION_SEALED`

## 왜 별도 후보인가

이 실행은 원래 BA-SRM2를 고쳐서 성공으로 바꾸는 작업이 아니다. BA-SRM2는 multipatch
importer가 채우지 않는 `stim_pulse.qc_pass=1`을 12개 펄스 모두에 요구했고, frozen train
manifest에서 E/I 모두 통과 sequence가 0개였으므로 `STOP_TRAIN_SUPPORT`다. 그 결과와
receipt는 그대로 보존한다.

BA-SRM3는 공식 Allen dynamics가 실제로 사용하는 sign-matched
`pulse_response.ex_qc_pass` 또는 `pulse_response.in_qc_pass`를 적용하는 새 적응적 후보다.
이 변경은 BA-SRM2의 STOP과 pinned source를 본 뒤 만들었으므로 사전 독립 후보라고
부르지 않는다. 다만 development와 confirmation outcome은 아직 열지 않았고, 구조
manifest도 결과를 보기 전에 이미 고정됐으므로 이후 예측 검사는 이 계약에서 다시 잠근다.

## 이미 알려진 것과 아직 모르는 것

[관측 비교] medium DB의 SHA-256은
`dbf19786f9e0d0d73c26351dc29d69ef8c10a2e67e32e19ac73034a5624d48c5`이고,
전체 SQLite integrity/FK/schema/order gate를 통과했다.

[관측 비교] target-blind train manifest는 정확히 3,000개이며 E/I 각각 1,500개다.
manifest SHA-256은
`4ddb4a52294a55b011c5118a02432ca28c057ca5b5ebb63d8d7c945923aa62c2`다.
BA-SRM3는 이 파일을 그대로 사용하며 다시 뽑지 않는다.

[관측 비교] 이 계약을 쓰기 전에 BA-SRM2 train support에서 알려진 집계는 다음뿐이다.

- 16-target complete: ex 903/1,500, in 937/1,500;
- 12개 type-matched response QC all-pass: ex 986/1,500, in 889/1,500;
- 12개 `stim_pulse.qc_pass=1` all-pass: 양쪽 모두 0;
- complete와 response-QC의 교집합, 그 slice 수, target MAD, target 값의 크기 및 모델 성능은
  아직 계산하지 않았다.

이 이미 알려진 집계를 숨기지 않는다. BA-SRM3의 첫 실행은 교집합 support만 계산하며,
그 결과가 부족하면 모델을 만들지 않는다.

## 데이터와 순서

source cohort는 mouse V1의 `mouse V1 coarse matrix`와
`mouse V1 pre-production`, `pair.has_synapse=1`, current clamp,
positive induction frequency/recovery delay다. E/I는 `synapse.synapse_type`에서 읽고 별도
모델로 유지한다.

medium DB는 펄스를 0부터 센다. 따라서 물리적 첫 8개 펄스는 SQL index `0..7`, 다음
4개는 `8..11`이다. sequence는 같은 pair, pre/post recording, stimulus name,
induction frequency, recovery delay 안에서 정확히 12행이어야 하며 pulse index와 onset이
엄격히 증가해야 한다.

slice split은 기존 frozen rule을 그대로 쓴다.

```text
salt = BA-SRM2-MEDIUM-R21-20260823-V1
bucket = uint64_be(SHA256(salt + ':' + slice.ext_id)[0:8]) mod 10
train = 0..5, development = 6..7, confirmation = 8..9
```

동일 slice의 모든 pair/recording/sequence는 같은 split에 남는다. donor가 여러 slice를
가질 가능성은 해소되지 않으므로 claim ceiling에 남긴다.

## QC와 complete-target estimand

[공리: 외부 입력] multipatch sequence의 event QC는 각 sequence의 12개 행 모두에서
synapse type과 맞는 `PulseResponse` QC가 1일 때만 통과한다.

```text
ex synapse: pulse_response.ex_qc_pass = 1 for all 12 rows
in synapse: pulse_response.in_qc_pass = 1 for all 12 rows
```

`stim_pulse.qc_pass`는 multipatch importer가 설정하지 않으므로 selection gate로 쓰지 않는다.
non-NULL 값이 발견되면 개수와 provenance를 별도 진단으로만 보고한다. 반응 QC는 공식
`pulse_response_qc_pass`가 postsynaptic recording, presynaptic spike 존재, 인접 pulse,
noise/artifact와 holding-potential 조건을 이미 반영한 source field다.

primary estimand은 QC를 통과하고 SQL index `8..11`의 네 fit field가 모두 finite인
complete-case conditional population이다.

$$
Y=left(
\frac{A_r}{V_0},\frac{\ell_r}{T_0},
\frac{\rho_r}{T_0},\frac{\tau_r}{T_0}
\right)_{r=9}^{12}\in\mathbb R^{16}.
$$

각 성분은 `dec_fit_reconv_amp`, `dec_fit_latency`, `dec_fit_rise_time`,
`dec_fit_decay_tau`다. NULL, NaN, infinity만 incomplete로 분류한다. finite 0과 음수는
그대로 보존하고 log, clipping, epsilon, winsorization을 금지한다. target complete 여부는
이미 frozen manifest를 바꾸지 않는다.

## 인과 이력과 무차원화

입력은 펄스 `0..7`까지만 사용한다.

$$
H_8=\sigma\{c,z_0,\ldots,z_7\},
\qquad M:H_8\to\mathbb R^{16}.
$$

각 $z_r$는 stimulus interval/amplitude/duration/spike count/first-spike timing과 과거의
여섯 fit field를 포함한다. $c$는 첫 펄스 전에 정해진 protocol, bath temperature,
postsynaptic baseline, soma distance, nearest test-pulse membrane quantity, layer와 source
cell class만 포함한다. target `8..11`의 값, 결측 flag, fit success 또는 QC summary는 입력,
normalizer, basis와 hyperparameter에 들어가지 않는다.

고정 기준은 $T_0=1\,\mathrm{ms}$, $V_0=1\,\mathrm{mV}$,
$I_0=1\,\mathrm{pA}$, $R_0=1\,\mathrm{M\Omega}$,
$C_0=1\,\mathrm{pF}$, $L_0=100\,\mu\mathrm m$,
$\Theta_0=310\,\mathrm K$다. frequency는 $fT_0$, 나머지 물리량은 대응 기준으로
나눈 뒤 train median/MAD를 적용한다. missing value는 train median으로 채우고 같은 위치의
binary missing mask를 붙인다. categorical vocabulary는 train에서만 만들고 새로운 level은
`UNK`로 보낸다.

source의 SI/metadata 단위를 계산 좌표로 옮기는 규칙은 다음과 같다. bath temperature는
source 값이 섭씨이므로 먼저 $\Theta_K=\Theta_{^\circ C}+273.15$로 바꾼다.

| source field family | source unit | dimensionless value |
|---|---|---|
| response/baseline amplitude, IC baseline/noise | V | value / $10^{-3}\,\mathrm V$ |
| latency, rise, decay, duration, interval, spike timing | s | value / $10^{-3}\,\mathrm s$ |
| induction frequency | Hz | value $\times10^{-3}\,\mathrm s$ |
| stimulus/baseline current | A | value / $10^{-12}\,\mathrm A$ |
| test-pulse input resistance | $\Omega$ | value / $10^6\,\Omega$ |
| test-pulse capacitance | F | value / $10^{-12}\,\mathrm F$ |
| soma distance | m | value / $10^{-4}\,\mathrm m$ |
| bath temperature | $^\circ$C | $(value+273.15)/310\,\mathrm K$ |
| spike count, NRMSE, missing mask | 1 | unchanged |

physical normalization 뒤 continuous input의 train center는 median이다. scale은 순서대로
$1.482602218505602\times\mathrm{MAD}$, $(Q_{0.75}-Q_{0.25})/1.3489795003921634$,
sample standard deviation을 사용하며 앞 단계가 0 또는 nonfinite일 때만 다음 단계로 간다.
세 값이 모두 0이면 scale을 1로 두고, composite design에서 exact-constant channel로 제거한다.
이 fallback 순서는 fold마다 fit subset에서만 계산한다.

## 첫 support gate

모델 unlock은 E/I 각각 다음을 모두 만족할 때만 허용한다.

1. response-QC와 complete target을 함께 만족한 distinct train slice group이 160개 이상;
2. 16개 target coordinate 각각의 train MAD가 finite이고 0보다 큼;
3. 정확한 12-pulse relation/order 위반이 0;
4. manifest hash, sequence key, slice split과 DB hash가 frozen receipt와 동일함.

하나라도 실패하면 `STOP_TRAIN_SUPPORT`, model/geometry/rank를 계산하지 않는다.

## train-only sieve와 response operator

support가 통과하면 continuous numeric input과 missing mask를 fold마다 다시 정규화하고,
categorical one-hot을 포함한 train-only PCA sieve를 쓴다.

$$
d\in\{2,4,8,16,32\},\qquad
k_\ell(a,b)=\exp\!\left[-\frac{\lVert a-b\rVert^2}{2\ell^2}\right].
$$

grid는 다음으로 고정한다.

```text
d       = 2, 4, 8, 16, 32
ell     = 0.5, 1, 2, 4
ridge   = 1e-6, 1e-4, 1e-2, 1
gamma_R = 0.25, 0.5, 0.75, 1
rho     = 0.5, 1, 2, 4
```

fold-fit coordinate를 $a_i$, dimensionless target의 fold-train median/MAD 표준화를
$\widetilde Y_i=(Y_i-m_Y)\oslash s_Y$라고 한다. exact multi-output KRR은

$$
K_{ij}=k_\ell(a_i,a_j),\qquad
\alpha=(K+\lambda I)^{-1}\widetilde Y,
$$

$$
M_o(a)=m_{Y,o}+s_{Y,o}\sum_i k_\ell(a,a_i)\alpha_{io}.
$$

여기서 grid의 `ridge`가 바로 $\lambda$이며 $n\lambda$로 재해석하지 않는다. output-by-input
Jacobian의 부호와 방향은

$$
J_{oj}(a)=\frac{\partial M_o}{\partial a_j}
=s_{Y,o}\sum_i\alpha_{io}k_\ell(a,a_i)
\frac{a_{i,j}-a_j}{\ell^2}
$$

로 고정한다. finite difference 상대오차 $10^{-5}$를 넘으면 구현 gate 실패다.

각 fold의 PCA numerical eigenvalue가 최대값의 $10^{-12}$ 이하이면 그 방향은 사용할 수 없다.
candidate $d$는 모든 평가 fold에서 사용 가능한 grid 원소의 교집합으로 제한하고 제외된 $d$를
receipt에 기록한다. 교집합에 $d\ge2$가 하나도 없으면 STOP하며 ridge나 artificial PCA floor로
차원을 만들지 않는다.

outer fold는
`SHA256('BA-SRM3-OUTER-FOLD-V1:' + slice.ext_id) mod 5`, outer $f$ 내부의 covariance
cross-fit fold는
`SHA256('BA-SRM3-INNER-R-V1:' + f + ':' + slice.ext_id) mod 4`다. fold마다 input
scaler, categorical vocabulary, target median/MAD와 PCA를 그 fold의 fit group에만 맞춘다.

16개 standardized target MSE의 동일 가중 평균으로 $(d,\ell,\mathrm{ridge})$를 고른다.
동점 허용오차 $10^{-12}$ 안에서는 작은 $d$, 큰 ridge, 큰 $\ell$ 순으로 고른다.
각 outer $f$의 $R_{-f,\gamma}$는 $-f$ 안의 four-fold cross-fit residual만 사용한다.
gamma 동점은 큰 gamma를 고른다. 최종 $R$은 선택된 operator의 full-train OOF residual
covariance이며 development/confirmation과 $G$에 그대로 고정한다.

정확한 nesting은 다음과 같다. 각 outer $f$에서 $-f$의 four-fold CV만으로
$(d,\ell,\lambda)$를 다시 고르고, 그 tuple로 $-f$ 내부 OOF residual과 $M_{-f}$를 만든다.
각 gamma 후보의 $R_{-f,\gamma}$는 그 inner OOF residual만 사용하고, 같은 $-f$ OOF residual
training criterion의 Gaussian negative log likelihood로 gamma를 고른다. 이 값은 성능 추정치가
아니라 covariance tuning loss다. 따라서 어떤 $f$의 target도 $M_{-f}$, preprocessing,
gamma 또는 $R_{-f,\gamma}$에 들어가지 않는다. rho는 그 뒤 outer $f$ prediction을 사용해
고르는 train-only tuning parameter다. gamma training loss와 rho outer score를 모두 held-out
성능으로 보고하지 않는다. 첫 성능 평가는 frozen model의 development score다. 최종 gamma도
full-train OOF residual에 같은 training criterion을 적용해 한 번 고른다.

$$
R_\gamma=(1-\gamma)S+\gamma\operatorname{diag}(S).
$$

OOF residual을 $e_i=Y_i-\widehat Y_i^{(-fold)}$, $\bar e=n^{-1}\sum_i e_i$라고 하면

$$
S=\frac1{n-1}\sum_{i=1}^n(e_i-\bar e)(e_i-\bar e)^T.
$$

Cholesky 계산에만 $10^{-8}\operatorname{median}(\operatorname{diag}R)I$ floor를 더하고,
그 floor를 관측 rank의 증거로 세지 않는다. $R$의 spectrum과 condition number를 보고한다.

## 관측가능 몫기하

선택된 KRR의 analytic Jacobian을 $J(a)=DM(a)$라고 한다.

$$
G(a)=J(a)^TR^{-1}J(a),
\qquad \operatorname{rank}G(a)\le\min(16,d).
$$

[정리] 유한 출력이므로 이것은 전체 고차원 또는 무한차원 상태공간의 SPD metric을 식별하지
않는다. 식별되는 것은 $T_aH_d/\ker J(a)$의 pointwise PSD pullback뿐이다. local support에서
rank가 일정하고 kernel subbundle이 매끄럽다는 조건을 별도로 통과해야 local quotient
manifold라고 부를 수 있다.

native whitened PCA chart의 reference tensor는 $g_{\rm ref}=I_d$로 고정한다. 이는 관측
$G$가 아니라 kernel과 coordinate transport의 기준이다.

rank threshold는 $\sigma_j(J)/\sigma_1(J)\ge10^{-4}$다. 모든 eligible train sequence를
rank anchor로 쓴다. train pointwise rank의 minimum과 maximum이 같은 정수
$r_{\rm train}\ge5$여야 constant-rank/high-dimensional train gate를 통과한다. development의
supported anchor에서도 minimum과 maximum이 서로 같고 그 정수가 정확히 $r_{\rm train}$일 때만
constant-rank support라고 보고한다.

별도 descriptive margin으로 각 anchor의 $q_5=\sigma_5/\sigma_1$을 계산하고 distinct slice
group을 replacement로 1,000회 resample해 slice-weighted median $q_5$의 2.5/50/97.5 percentile을
보고한다. KRR은 refit하지 않으며, 이 bootstrap은 pass gate나 독립 안정성 증거가 아니고
fit uncertainty도 측정하지 않는다. PRNG seed는
`uint64_be(SHA256('BA-SRM3-RANK-BOOTSTRAP-V1:' + synapse_type)[0:8])`로 고정한다.

affine rechart $a'=Aa+b$에서는

$$
G'=A^{-T}GA^{-1},\qquad g'_{\rm ref}=A^{-T}g_{\rm ref}A^{-1}
$$

로 모든 object를 운반한다. reverse permutation, fixed diagonal scale, shear와 translation에서
line element, prediction, generalized spectrum 상대오차가 $10^{-6}$ 이하여야 한다. transformed
chart에서 kernel/PCA를 다시 맞추면 실패다.

검사 anchor는
`SHA256('BA-SRM3-GAUGE-ANCHOR-V1:' + sequence_key)`가 작은 순서의 최대 256개다.
fixed maps는 reverse permutation $P$, $D=\operatorname{diag}(\exp(t_j))$ with
$t_j$ equally spaced from $-\log2$ to $\log2$, $S=I+0.2e_1e_2^T$와
$b_j=0.1(-1)^j$다. 각 $A\in\{P,D,S\}$와 translation에서

$$
M'(a')=M(A^{-1}(a'-b)),\qquad J'(a')=J(a)A^{-1}
$$

를 직접 운반한다. generalized spectrum은 정렬한
$\operatorname{eig}(g_{\rm ref}^{-1/2}Gg_{\rm ref}^{-1/2})$다. $d=1$일 때 shear는 생략하며,
이 route의 후보 $d$는 모두 2 이상이다.

## 예측 falsifier와 controls

local secant diagnostic은

$$
D_G^2(a,b)=\frac12(a-b)^T\{G(a)+G(b)\}(a-b)
$$

이고 $\exp[-D_G^2/(2\rho^2)]$로 train neighbor의 future response mean을 만든다.
effective neighbor count가 10 미만이면 abstain한다. rho는 outer-fold train-only ELPD로 고르고
동점이면 큰 rho를 택한다. rho는 $M,R,G$를 바꾸지 않는다.

controls는 constant, linear ridge, direct RBF $M$, Euclidean neighbor, constant-full
$\bar G$, pointwise diagonal $G$, deterministic pulse-order shuffle, past-response/stimulus/static
세 channel-group ablation이다. 모든 mean predictor는 동일한 frozen $R$로 점수화한다. linear
ridge는 같은 ridge grid를 fold 안에서 고른다. shuffle은
`SHA256('BA-SRM3-PULSE-SHUFFLE-V1:' + sequence_key)`로 `0..7` pulse-local channel과 mask를
공동 permutation하며 static과 target은 유지한다.

constant는 fold-train target mean, linear ridge는 fold-train에서 같은 ridge grid를 고른
multi-output linear model이다. neighbor/control과 세 ablation도 outer $-f$에서만 scaler,
vocabulary, PCA, parameter와 mean을 맞춘다. ablation마다 남은 nonconstant support 안에서 같은
$d$ grid를 다시 적용하되 원 model의 $d$보다 큰 차원으로 보상하지 않는다. 모든 predictor의
Gaussian log score는 같은 fold-specific 또는 최종 frozen $R$로

$$
\log p(Y_i\mid\mu_i,R)=-\frac12\left[
(Y_i-\mu_i)^TR^{-1}(Y_i-\mu_i)+\log\det R+16\log(2\pi)
\right]
$$

를 사용한다.

held-out score는 slice마다 sequence log predictive density 차이의 평균을 먼저 구하고,
slice 평균들의 평균과 $SE=sd/\sqrt{n_{slice}}$를 쓴다. direct RBF가 constant와 linear를,
full-$G$ neighbor가 나머지 모든 개별 geometry/control predictor를 각각
$\Delta\mathrm{ELPD}>2SE$로 이겨야 한다. 하나의 control만 골라 비교하지 않는다.

## unlock 순서와 주장 상한

1. 이 계약과 source receipt를 고정한다.
2. 동일 frozen train manifest에서 response-QC support 교집합만 한 번 계산한다.
3. support PASS일 때 train-only model과 모든 hyperparameter를 동결한다.
4. development structural manifest를 결과 없이 먼저 고정하고 outcome을 한 번 연다.
5. support, rank, gauge, missingness와 모든 control gate가 통과할 때만 confirmation을 한 번 연다.
6. development 실패 뒤 model, subset, dimension, QC, threshold를 바꾸면 confirmation은 열지 않고
   또 다른 후보로 분리한다.

[미완성] 현재는 2단계 전이다. 따라서 기하, rank, 예측 향상은 아직 주장하지 않는다.
성공하더라도 Allen mouse-V1의 complete fitted-event population에서 finite observable quotient가
예측적으로 유용하다는 L2/L3 수준까지만 허용한다. conductance, $Npq$, vesicle/receptor,
STDP, homeostasis, memory, 전체 뇌 시공간, 무한차원 SPD 다양체 또는 AGI mechanism은
이 계약의 관측량이 아니다.
