# Revision 01 — medium event-level preaccess preregistration

Date: 2026-08-23

Status: `PREACCESS_FROZEN / ACQUISITION_IN_PROGRESS / MEDIUM_OUTCOMES_UNREAD / CONFIRMATION_UNTOUCHED`

Parent: `../00-contract.md` (`BA-SRM2`)

이 문서는 Allen-SynPhys r2.1 medium 파일의 event outcome을 열기 전에 후속 시험을
고정한다. 다운로드 중인 partial file은 분석 입력이 아니며, 정확한 byte count,
SHA-256, SQLite integrity와 schema-only gate를 모두 통과하기 전에는 어떤
`pulse_response_fit` 값도 읽지 않는다. small DB의 잘못된 12-slot aggregate는 이
시험에 사용하지 않는다.

## 1. 질문, 주 분석군과 주장 단위

질문은 시냅스 상태의 차원을 4로 고정하는 것이 아니라, 과거 8-pulse event history가
미래 4-pulse response를 예측할 때 데이터가 식별하는 관측가능 몫공간의 rank가
얼마인지 묻는 것이다.

주 분석군은 결과값과 무관하게 다음 source metadata로 고정한다.

- `slice.species = 'mouse'`;
- `experiment.project_name IN ('mouse V1 coarse matrix', 'mouse V1 pre-production')`;
- `pair.has_synapse = 1`, postsynaptic `patch_clamp_recording.clamp_mode = 'ic'`;
- source `synapse.synapse_type`의 `ex`와 `in`은 섞지 않고 별도 층화한다;
- `multi_patch_probe.induction_frequency > 0`, `recovery_delay > 0`;
- pulse stimulus와 response QC는 각각 `stim_pulse.qc_pass = 1` 및 synapse type에
  맞는 `pulse_response.ex_qc_pass = 1` 또는 `in_qc_pass = 1`을 쓴다.

전체 mouse V1이나 모든 시냅스가 아니라 이 조건부 event-fit 모집단만 추론 대상이다.
흥분성과 억제성 가운데 하나만 통과하면 그 층화에 대해서만 보고한다.

## 2. source-locked event join

공식 schema와 `aisynphys/dynamics.py`의 join을 다음처럼 고정한다.

```text
pair pa -> experiment ex -> slice sl
pa -> pulse_response pr -> pulse_response_fit prf
pr -> stim_pulse sp
pr.recording_id -> recording post_r -> patch_clamp_recording post_pcr
post_pcr -> multi_patch_probe mpp
sp.recording_id -> recording pre_r
post_r/pre_r -> sync_rec post_sr/pre_sr
```

`pr.recording_id`는 postsynaptic recording이고 `sp.recording_id`는 presynaptic
recording이므로 둘의 동일성을 요구하지 않는다. 오히려 다음 consistency guard를
모두 요구한다.

```text
post_r.id != pre_r.id
post_r.sync_rec_id = pre_r.sync_rec_id
post_sr.experiment_id = pre_sr.experiment_id = pa.experiment_id
sp.cell_id IS NULL OR sp.cell_id = pa.pre_cell_id
pre_r.electrode_id = pre_cell.electrode_id
post_r.electrode_id = post_cell.electrode_id
```

official multipatch importer는 `StimPulse(recording=rec_entry, ...)`를 만들 때 `cell`을
설정하지 않으므로 multipatch row의 `sp.cell_id`는 NULL일 수 있다. 이 경우 presynaptic
identity의 source-locked guard는 `pre_r.electrode_id = pre_cell.electrode_id`다. NULL을
임의 cell ID로 채우지 않으며, `sp.cell_id`가 실제로 non-NULL인데 pair pre-cell과
다르면 그 sequence만 제외한다.

protocol parameter는 공식 source 분석과 같이 postsynaptic chain의 `mpp`에서 읽는다.
schema-only audit가 실제 column/FK/cardinality를 먼저 확인하기 전에는 위 join을
source-verified medium fact로 승격하지 않는다. sequence key는
`(pair_id, post_r.id, pre_r.id, post_r.stim_name, induction_frequency,
recovery_delay)`이고, 그 안에서 `sp.pulse_number`와 `sp.onset_time`이 엄격히
증가해야 한다. pulse number 중복, 역전, 서로 다른 protocol metadata 또는 위
consistency guard 위반이 하나라도 있으면 그 sequence를 제외하고 위반 수를 공개한다.
`stim_pulse.data`, `pulse_response.data`, baseline BLOB은 medium 시험에서 읽지 않는다.

## 3. 인과 표본과 16차원 미래 출력

주 경로는 pulse 1--8의 prefix만 입력으로 쓰고 pulse 9--12를 예측한다.

$$
H_8=\sigma\{c, z_1,\ldots,z_8\},
$$

$$
Y_8=
\left(
\frac{A_r}{V_0},
\frac{\ell_r}{T_0},
\frac{\rho_r}{T_0},
\frac{\tau_r}{T_0}
\right)_{r=9}^{12}
\in\mathbb R^{16},
$$

여기서 source field는 각각 `dec_fit_reconv_amp`, `dec_fit_latency`,
`dec_fit_rise_time`, `dec_fit_decay_tau`다. amplitude는 signed raw value를 유지하며
어떤 target에도 log를 적용하지 않는다. latency, rise와 decay도 저장된 raw unit을
먼저 $T_0$로 나눌 뿐, 이름만 보고 양수라고 가정하거나 log를 적용하지 않는다.
NULL, NaN과 $\pm\infty$는 incomplete target으로 분류한다. finite zero와 negative
fit 값은 censor, epsilon, clipping 또는 winsorization 없이 그대로 보존한다.

pulse 9--12의 값, fit 성공 여부, NULL flag, QC-derived summary와 그 값으로 계산한
normalizer는 $H_8$, 표본 선택, basis 또는 hyperparameter 선택에 들어갈 수 없다.
16개 target이 전부 finite인 표본에서만 $\mathbb R^{16}$ geometry를 말한다. 일부만
관측된 표본은 후보 모집단의 missingness 통계에는 남기지만 primary rank/ELPD에는
들어가지 않으며, 이 결과는 complete-fit conditional estimand다.

병렬 sensitivity route는 처음부터 다음으로 고정한다.

$$
H_4\longmapsto
Y_4=
\left(
\frac{A_r}{V_0},
\frac{\ell_r}{T_0},
\frac{\rho_r}{T_0},
\frac{\tau_r}{T_0}
\right)_{r=5}^{8}.
$$

이 route는 $H_8$ 실패 뒤 고르는 fallback이 아니며 어떤 경우에도 primary claim을
구제하지 않는다. confirmation은 두 고정 route에 대해 같은 한 번의 unlock에서만
평가한다. 승격 판단은 $H_8$에만 의존하고 $H_4$는 sensitivity로만 보고하므로 둘 중
좋은 결과를 고르는 다중선택은 없다.

## 4. 입력 history와 무차원화

각 prefix pulse $r$의 $z_r$에는 그 pulse 시점까지 실제로 관측된 다음 필드만 쓴다.

1. stimulus: `previous_pulse_dt`, `amplitude`, `duration`, `n_spikes`,
   `first_spike_time - onset_time`;
2. past response fit: `dec_fit_reconv_amp`, `baseline_dec_fit_reconv_amp`,
   `dec_fit_latency`, `dec_fit_rise_time`, `dec_fit_decay_tau`, `dec_fit_nrmse`;
3. pulse 1 전에 정해진 protocol/static covariate: induction frequency, recovery delay,
   bath temperature, postsynaptic baseline potential/current/noise, pair soma distance,
   postsynaptic input resistance/capacitance/membrane time constant, pre/post layer와
   source-locked cell class.

관측되지 않은 conductance, vesicle count, $Npq$, receptor identity, STDP eligibility,
homeostatic state는 만들거나 대입하지 않는다. categorical vocabulary는 train에만
맞추고 dev/confirmation의 새 level은 단일 `UNK`로 보낸다. input missing value는
physical normalization 뒤 train median으로 채우고 같은 위치의 binary missing mask를
추가한다. exact-constant input channel은 train에서만 제거하고 목록을 기록한다.

고정 SI reference는 다음과 같다.

| quantity | reference |
|---|---:|
| time | $T_0=1\,\mathrm{ms}$ |
| voltage/IC response | $V_0=1\,\mathrm{mV}$ |
| injected/current covariate | $I_0=1\,\mathrm{pA}$ |
| resistance | $R_0=1\,\mathrm{M\Omega}$ |
| capacitance | $C_0=1\,\mathrm{pF}$ |
| distance | $L_0=100\,\mathrm{\mu m}$ |
| temperature | $\Theta_0=310\,\mathrm K$ |

모든 exp/kernel/log/probability core는 이 비율과 dimensionless count만 받는다. frequency는
$fT_0$, recovery와 interval은 $\Delta t/T_0$로 넣는다. 물리 기준척도 뒤에 적용하는
median/MAD standardization과 target centering은 train에서만 적합한다. target channel의
train MAD가 0이거나 finite하지 않으면 그 층화의 16D geometry를 STOP하고 차원을
사후 축소하지 않는다.

정확 KRR의 resource ceiling도 outcome 전에 고정한다. slice split을 먼저 배정한 뒤
train에 속한 structural candidate만 대상으로 하며, E/I label은 source
`synapse.synapse_type`에서만 읽는다. target 값이나 target availability를 읽기 전에
candidate sequence를 slice별로 나누고, 각 slice 안에서는
`SHA256('BA-SRM2-MEDIUM-SEQUENCE-CAP-V1:' + sequence_key)`로 정렬한다. 모든 slice의
첫 번째 sequence, 그 다음 모든 slice의 두 번째 sequence 순으로 round-robin하여 E/I
층화별 최대 1,500개를 선택한다. 1,500개보다 적으면 전부 쓴다. 이 cap은 큰 slice가
표본을 독점하는 것을 막기 위한 고정 계산 한계이며 completion 여부에 따라 다시 뽑지
않는다. 선택된 `(slice.ext_id, sequence_key)` 정렬 manifest의 SHA-256과 층화별 수를
target unlock 전에 receipt로 고정한다. development와 confirmation은 train cap에
영향을 주지 않으며 자기 unlock 뒤 eligible sequence 전부를 평가한다.

## 5. split과 confirmation 봉인

최상위 독립 group은 `slice.ext_id`로 고정한다. donor ID가 source에 없으므로 같은 donor의
여러 slice 가능성은 해소되지 않으며 claim ceiling에 남긴다. split key는

```text
salt = BA-SRM2-MEDIUM-R21-20260823-V1
bucket = uint64_be(SHA256(salt + ':' + slice.ext_id)[0:8]) mod 10
train = 0..5, development = 6..7, confirmation = 8..9
```

이다. pair, recording 또는 sequence는 자기 slice의 split을 그대로 상속한다. schema-only
단계에서는 group ID와 bucket, table row 존재만 계산할 수 있지만 target NULL 여부,
fit-success와 completion rate는 어느 split에서도 SELECT하지 않는다. train completion은
train unlock 뒤, development completion은 frozen model이 준비된 뒤, confirmation
completion은 development gate 통과 뒤 각각 자기 unlock 시점에만 읽는다. confirmation의
fit/outcome column은 development gate가 통과하기 전까지 SELECT하지 않는다. split을
다시 뽑거나 salt를 바꾸지 않는다. 어느 E/I 층화든 train independent slice가 160개
미만이면 16D covariance/geometry 주장은 STOP하고 event prediction 진단만 허용한다.

## 6. train-only sieve와 response operator

flattened masked history를 train median/MAD로 표준화한 뒤 train PCA를 적합한다. 후보
차원은

$$
d\in\{2,4,8,16,32\},
\qquad
d\le\min(p_{\rm nonconstant},n_{\rm train\ slice}-1)
$$

로 고정한다. $d>16$도 허용하지만 관측가능 몫 rank는 16을 넘을 수 없다. PCA basis와
모든 정규화는 train 내부 group-CV fold마다 다시 적합하고 dev/confirmation에서는
절대 재적합하지 않는다.

whitened coefficient $a\in\mathbb R^d$에서 기준 tensor는 $g_{\rm ref}=I_d$이고,
response operator는 deterministic multi-output RBF kernel ridge로 고정한다.

$$
k_\ell(a,b)=
\exp\left[-\frac{(a-b)^Tg_{\rm ref}(a-b)}{2\ell^2}\right].
$$

train 내부 5-fold slice-group CV의 고정 grid는

```text
d       = 2, 4, 8, 16, 32
ell     = 0.5, 1, 2, 4
ridge   = 1e-6, 1e-4, 1e-2, 1
gamma_R = 0.25, 0.5, 0.75, 1
rho     = 0.5, 1, 2, 4
```

이다. 1,280개 조합을 사후 탐색하지 않고 다음 순차 절차를 쓴다.

1. 각 fold마다 input scaler, target 16 channel median/MAD, PCA/history basis를 그
   fold의 4/5 group에만 다시 적합한다. 16개의 fold-standardized squared error에 각각
   $1/16$ weight를 준 5-fold slice-group OOF MSE로 $(d,\ell,\mathrm{ridge})$를
   선택한다. 이 값은 model-selection loss이지 unbiased 성능 추정치가 아니다.
2. 각 outer held-out fold $f$마다 preprocessing과 $M_{-f}$를 오직 $-f$에서 적합한다.
   $R_{-f,\gamma}$는 다시 $-f$ 내부의 four-fold group cross-fit residual만으로 적합하고,
   그 뒤 처음으로 $f$의 likelihood를 채점하여 `gamma_R`를 선택한다. 다른 fold의
   residual을 만든 모델이 $f$를 학습한 경우도 금지한다.
3. 선택된 operator와 covariance를 고정한 뒤 같은 outer folds에서 `rho`를 선택한다.
   query $f$의 geometry neighbor mean은 $-f$ sequence만 사용하며 §7의
   $w_b(a)=\exp[-D_G^2(a,b)/(2\rho^2)]$ ELPD가 목적함수다. `rho`는 $M$, $R$ 또는
   $G$를 다시 적합하지 않고 neighbor weight만 바꾼다. held-out fold의 residual이나
   target은 그 fold의 $R_\gamma$, neighbor mean 또는 bandwidth fit에 쓰지 않는다.

support가 5-fold를 허용하지 않으면 3-fold로 내리는 대신 STOP한다. 동률은 더 작은
$d$, 더 큰 ridge, 더 큰 `gamma_R`, 더 큰 bandwidth 순으로 고른다. 선택된
$(d,\ell,\mathrm{ridge})$의 full-train OOF residual로 최종 $R_\gamma$를 한 번 적합하고,
같은 hyperparameter tuple로 PCA, normalization과 $M_d$를 전체 train split에 정확히
한 번 재적합한 뒤 frozen object를 development에 전달한다. development 결과를 본 뒤
train+development로 다시 적합하지 않으며, gate가 통과하면 같은 frozen train-only
object를 confirmation에 그대로 적용한다. 최종 $R_\gamma$는 full-refit residual
covariance가 아니라 `cross-fitted predictive covariance estimator`이며, development,
confirmation과 $G$ 계산에서 정확히 같은 frozen matrix를 쓴다.

위 절차의 train OOF residual covariance $S$에서

$$
R_\gamma=(1-\gamma)S+\gamma\,\operatorname{diag}(S)
$$

를 쓰며, 역행렬 계산에만
$10^{-8}\operatorname{median}(\operatorname{diag}R_\gamma)I$를 더한다. 이 수치 floor는
관측 rank를 만들었다는 증거가 아니다. $R_\gamma$의 eigen-spectrum과 condition number를
보고한다.

## 7. 국소 pullback과 predictive test

동결된 $M_d:\mathbb R^d\to\mathbb R^{16}$의 analytic Jacobian으로

$$
J(a)=DM_d(a),
\qquad
G(a)=J(a)^TR_\gamma^{-1}J(a)
$$

를 계산한다. 따라서

$$
\operatorname{rank}G(a)\le\min(16,d).
$$

$G(a)$는 pointwise PSD pullback이다. 다음 평균은 전역 요약일 뿐 local Riemannian
metric으로 부르지 않는다.

$$
\bar G=\frac1n\sum_iG(a_i).
$$

geometry-specific predictor는 train sample 사이의 고정 local-secant diagnostic

$$
D_G^2(a,b)=\frac12(a-b)^T\{G(a)+G(b)\}(a-b)
$$

과 weight $w_b(a)=\exp[-D_G^2(a,b)/(2\rho^2)]$로 미래 $Y$의 weighted Gaussian
mean을 만든다. effective neighbor count
$(\sum w)^2/\sum w^2<10$이면 그 query는 abstain한다. 이것은 exact geodesic distance라는
주장이 아니다.

고정 controls는 다음과 같다.

1. constant multivariate Gaussian;
2. linear ridge causal-history predictor;
3. direct RBF $M_d$ predictor;
4. Euclidean $I_d$ neighbor kernel;
5. constant-full $\bar G$ neighbor kernel;
6. pointwise diagonal $\operatorname{diag}G(a)$ kernel;
7. pulse order를 pair/protocol 안에서 shuffle하되 channel marginal과 missing mask를
   보존한 adverse control;
8. past-response, stimulus-timing, membrane/static channel을 하나씩 뺀 동일-$d$ ablation.

score는 slice-cluster held-out Gaussian log predictive density다. 먼저 direct nonlinear
$M_d$가 constant와 linear control 각각보다

$$
\Delta\mathrm{ELPD}>2\,\mathrm{SE}_{\rm slice}
$$

여야 한다. 그 다음 geometry predictor가 나머지 모든 개별 control보다 같은 기준으로
우수해야 한다. 일부 control만 골라 비교하지 않는다. development에서 두 gate를 모두
통과할 때만 confirmation을 정확히 한 번 연다.

## 8. rank, missingness와 gauge falsifier

rank는 $\sigma_j(J)/\sigma_1(J)\ge10^{-4}$로 정의한다. train slice bootstrap 1,000회와
고정 hash 순서의 evaluation anchor에서 2.5% lower rank가 $r_\star\ge1$이고 development
support에서 동일 rank가 유지될 때만 그 support의 constant-rank quotient라고 한다.
특히 $r_\star\ge5$일 때만 “4차원을 넘는 관측가능 부분공간” 증거라고 쓴다. 그렇지
않으면 추정 rank를 그대로 보고한다. 이 표현은 해당 fit과 local support의 numerical
observable quotient rank에만 적용하며 생물학적 고유 차원이라는 뜻이 아니다. smooth
closed kernel subbundle을 확인하지 못하면 global manifold가 아니라 local quotient라고만
쓴다.

candidate cohort, 16-target complete cohort와 split별 completion rate는 각 split의
정식 unlock 이후에만 공개한다. train과 development의 completion rate 차이가 5
percentage point를 넘거나,
complete/incomplete 사이 어느 input-only covariate의 standardized difference가 0.1을
넘으면 geometry claim은 `SELECTION_SENSITIVE / STOP`이다. target 값에 맞춘 exclusion,
epsilon, censor threshold 또는 protocol subset 변경은 금지한다.

affine rechart $a'=Aa+b$에서는

$$
G'=A^{-T}GA^{-1},
\qquad
g_{\rm ref}'=A^{-T}g_{\rm ref}A^{-1}
$$

와 함께 PCA basis, penalty와 kernel distance를 transport한다. 고정 시험은 reverse
permutation, $A=\operatorname{diag}(2^{-1},\ldots,2)$, $A=I+0.2e_1e_2^T$와
$b_j=0.1(-1)^j$다. line element, prediction과 generalized spectrum의 상대 오차가
$10^{-6}$을 넘거나 transformed chart에서 isotropic kernel을 새로 적합해야만 일치하면
gauge gate는 실패한다. $D_G^2$는 transport 뒤 불변이어야 하며 dimensionless scalar
$\rho$는 재적합하거나 좌표별로 바꾸지 않는다.

## 9. acquisition/schema unlock과 claim ceiling

순서는 다음과 같고 건너뛸 수 없다.

1. expected 11,125,997,568 bytes와 일치하는 다운로드 완료;
2. local SHA-256 기록, SQLite `quick_check`와 full `integrity_check` 통과;
3. table/column/foreign-key와 row count만 조회하는 schema-only audit;
4. outcome value 없이 위 join cardinality, pulse ordering metadata와 four-fit-field가
   `pulse_response_fit -> pulse_response -> stim_pulse` 관계에 속한다는 schema provenance만
   확인;
5. 1--4를 독립 audit한 뒤 train outcome만 unlock하고, 이 시점에만 train fit 값의
   pulse-locality, finite/NULL pattern과 QC 값을 확인;
6. frozen train selection 후 development 한 번;
7. 모든 gate 통과 때만 confirmation 한 번.

성공해도 주장 상한은 “Allen r2.1 medium의 complete fitted event 관측에서, 미래
4-pulse response에 대해 안정적으로 식별되는 finite local observable quotient”다.
medium에는 raw waveform이 없으므로 full causal-history Hilbert metric, 뇌 시공간,
conductance, release probability, receptor, STDP, homeostasis, 장기 기억 또는 AGI
mechanism을 주장하지 않는다. 이 revision의 $\Delta F_{\rm CE}=0$이며 먼저 측정 가능한
생물 baseline을 세우는 단계다.

## 10. dimensionless gate

| core argument | raw dimension | normalization | status |
|---|---|---|---|
| interval, latency, rise, decay | $T$ | $/T_0$ | dimensionless |
| IC response/baseline potential | voltage | $/V_0$ | dimensionless |
| stimulus/baseline current | current | $/I_0$ | dimensionless |
| resistance, capacitance, distance, temperature | respective unit | $/R_0,/C_0,/L_0,/\Theta_0$ | dimensionless |
| $f$, count, missing mask | $T^{-1}$ or pure number | $fT_0$ or unchanged | dimensionless |
| RBF exponent | pure-number quadratic ratio | $D^2/(2\ell^2)$ | dimensionless |
| Gaussian likelihood quadratic | target ratio | residual whitening by $R_\gamma$ | dimensionless |

차원 상태는 `무차원`이다. 이는 차원 정합만 뜻하며 생물학적 정당성이나 metric
식별을 증명하지 않는다. focused checker는 다음 명령으로 고정했다.

```text
.codex\hooks\python.cmd pytest tests\test_dimensionless.py::test_ba_srm2_event_history_target_and_kernel_are_dimensionless -q
1 passed
```

`ce-dimensionless`가 지시한 `docs/참조/무차원_감사_수학.md`는 현재 작업트리에
존재하지 않아 skill 본문의 단일 판정과 기존 `dimensionless.py` checker를 사용했다.
