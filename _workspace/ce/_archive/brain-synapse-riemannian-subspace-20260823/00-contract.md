# BA-SRM1 계약 — 실제 시냅스 인자 기반 Riemannian 부분공간

Status: COMPLETE

CE_RUN: _workspace/ce/brain-synapse-riemannian-subspace-20260823
Candidate: BA-SRM1
Priority: HIGHEST_PRIORITY_FOR_BRAIN_AGI_RESEARCH
Date frozen: 2026-08-23 (Asia/Seoul)
Mode: full research
Revision: 1 — source/math lane의 pre-outcome P0 교정만 반영; response outcome 미접촉

## 1. 질문과 주장 상한

시냅스의 “연결 세기”를 단일 $W_{ij}$로 두지 않고, paired patch-clamp가
관측하는 휴지막 반응 세기·거리·막 통합·latency·kinetics·short-term
plasticity(STP), pre/post cell type로 분해했을 때, 실제로 식별되는
부분공간에 Riemannian metric을 정의할 수 있는가? 그 metric의 geodesic
neighborhood가 동일 자유도의 raw-factor/Euclidean model보다 held-out
synaptic dynamics를 더 잘 예측하는가?

`CLAIM_CEILING`: Allen-Synphys의 paired electrophysiology measurement에서
식별된 synapse-factor 부분공간에 대한 L1/L2 관측·동일 데이터셋 내부 예측 비교다.
독립 site/recording 전이가 없으므로 L3 증거로 부르지 않는다. 실제 뇌가
CE 기하를 사용한다는 주장, 기억의 실체, 전체 시냅스 상태의 정준 계량,
개별 $Npq$ 분해, 해부학적 연결의 인과성, 물리 시공간, 인간 일반화와 AGI
주장은 금지한다.

| Claim ID | 형식 지위 | 활성 주장 |
|---|---|---|
| `BA-SRM1-C1` | [정의/관측 입력] | synapse state를 단일 $W$가 아니라 typed factor registry로 분해한다. |
| `BA-SRM1-C2` | [정의/측정모형] | source-locked strict 4D chart와 scalar-summary 4D target을 사용한다. |
| `BA-SRM1-C3` | [조건부 정리] | $R\succ0$, $\operatorname{rank}J=4$이면 $J^TR^{-1}J$는 strict chart의 Riemannian metric이다. |
| `BA-SRM1-C4` | [예측] | variable response metric이 best matched control보다 development에서 $2SE$를 넘는지 검사한다. |
| `BA-SRM1-C5` | [미완성/금지 경계] | conductance, $Npq$, directed-delay distance, 장기 가소성, 기억·AGI 승격은 이 판본으로 식별되지 않는다. |

## 2. 선행 증거와 봉인

| Predecessor | 판정 | 보존 | 재시도 금지 |
|---|---|---|---|
| `brain-circuit-manifold-equations-20260821` + `brain-circuit-manifold-property-loop-20260821` | `MATH_PROPERTY_PASS / EMPIRICAL_UNTESTED` | full-rank $g_{\rm pass}=J^TGJ$와 delayed augmented tangent; 각 run의 `12-routes.md`, `31-validation.md`, `40-final-report.md` | synthetic PASS의 생물 승격; rank loss를 ridge로 숨김 |
| `neural-riemannian-metric-validation-20260818` | estimator 실행, biological winner 없음 | metric은 typed input·held-out target·animal split이 필요 | raw $W$가 유일한 $g$를 정한다는 주장 |
| `neural-riemannian-metric-independent-tests-20260818` | eligible new empirical input 부재로 구현 SKIPPED | 독립 입력 없이는 반복 실행하지 않음 | 기존 입력을 새 독립 검증으로 재표기 |
| `neural-riemannian-metric-multiroute-execution-20260818` | synthetic/grid/descriptive routes | 출력기하와 생물 연결기하를 분리 | synthetic geometry를 생물 증거로 승격 |
| `neural-riemannian-metric-3d-pfc-20260819` | kernel PASS, PFC biology `ACCESS_BLOCKED / UNTESTABLE` | joint continuous coordinates와 구조 bridge 필요 | metadata-only PFC를 물리 피질 계량으로 승격 |
| `brain-metric-routing-equation-rebuild-20260820` | 수식 감사 PASS, empirical unopened | Fisher geometry와 directed routing을 분리 | $C^{-1}$을 정준 metric으로 사용; metric→routing 승격 |
| BA-TR11/12/13 | curvature-memory 기각; selector STOP | curvature는 derived diagnostic일 수 있음 | curvature=memory, curvature route selector 재튜닝 |
| BA-CG1 (`_workspace/ce/ba-cg1-falsifier-prereg.md`) | directed delay quasi-metric DRAFT | delay/order는 raw-$L$ baseline과 order/sample null을 갖는 방향성 진단 | directed delay를 Riemannian distance로 부름 |
| BA-DV1 | source/schema 후 사용자 전환 | ASI morphology receipt | ASI를 conductance, release probability 또는 paired strength로 동일시 |

## 3. 시냅스 상태의 층별 분해

하나의 edge $(j\to i)$는 다음 typed state로 분리한다.

| 층 | 기호 | 의미 | BA-SRM1 관측 지위 |
|---|---|---|---|
| topology | $A_{ij}\in\{0,1\}$ | 연결 존재 | pair connectivity로 관측 가능; 이산 stratum |
| transmitter/type | $\chi_{ij}$ | E/I transmitter, pre/post cell class, species, clamp mode | categorical stratum |
| baseline response | $r_{1,ij}$ | 첫 유발 PSP/PSC amplitude | 직접 측정 summary; conductance와 동일시 금지 |
| delay | $d_{ij}$ | stimulus/spike에서 postsynaptic onset까지 latency | 직접 summary; axonal $L/v$와 synaptic delay의 합 |
| kinetics | $\tau_{r,ij},\tau_{d,ij}$ | rise/decay time | 관측 가능할 때 연속 좌표 |
| STP | $U_{ij},x_{ij},u_{ij},\tau_D,\tau_F$ | utilization, resource, depression/facilitation | fitted model 또는 pulse-train summary로만 |
| membrane | $C_i,R_i,V_{rest,i},\tau_{m,i}$ | postsynaptic integration | cell/recording covariate |
| morphology | $N_{ij}$, contacts, PSD/ASI, distance/location | 구조 multiplicity·geometry | Allen-Synphys가 주는 항목만; MICrONS/de Vivo는 후속 complement |
| release/quantal | $p_{ij},q_{ij},N_{release}$ | release probability, quantal size, release sites | 첫 판본에서 latent/unidentified |
| receptor | $\bar g^{AMPA/NMDA/GABA}$, $E_r$, gating | receptor-specific conductance | 첫 판본에서 latent/unidentified |
| plasticity | $e_{ij},M_i,h_i$ | eligibility, neuromodulator, homeostasis | 첫 판본에서 latent/unidentified |
| structural dynamics | birth/death, spine survival | edge existence 변화 | longitudinal data 없으므로 미적합 |
| extracellular | astrocyte, ions, uptake | measured modulation | compact universal state 없음; 제외 |

`A_{ij}`와 $\chi_{ij}$는 smooth coordinate가 아니다. topology와 sign/type이
다른 표본은 별도 stratum으로 비교하며, 사전 정의한 embedding 없이 서로의
geodesic distance를 만들지 않는다.

## 4. BIO_STARTING_MECHANISM

### 4.1 membrane–conductance baseline

확립된 출발 구조는

$$
C_i\frac{dV_i}{dt}
=-g_{L,i}(V_i-E_{L,i})
-\sum_{j,r}A_{ij}\bar g_{ij}^{(r)}s_{ij}^{(r)}(t)
(V_i-E_r)+I_i^{ext}(t)+\eta_i(t)
$$

이다. conductance $\bar g$는 siemens, voltage는 volt, current는 ampere다.
Allen의 PSP/PSC amplitude는 이 식의 관측 결과이지 $\bar g$ 자체가 아니다.
conductance 환산에는 clamp mode, holding potential, reversal potential과 access
resistance model이 필요하므로 첫 판본에서는 수행하지 않는다.

### 4.2 receptor event와 directed delay

arrival time $t_j^k+d_{ij}$ 사이에서

$$
\tau_r\dot s_{ij}^{(r)}=-s_{ij}^{(r)},
$$

arrival event에서

$$
s_{ij}^{(r)+}=s_{ij}^{(r)-}
+\kappa_{ij}^{(r)}u_{ij}^{+}x_{ij}^{-}
$$

로 둔다. $d_{ij}$는 event order를 정하지만 일반적으로
$d_{ij}\ne d_{ji}$이므로 Riemannian distance가 아니다.

### 4.3 Tsodyks–Markram STP baseline

spike 사이에서

$$
\tau_F\dot u=U-u,\qquad
\tau_D\dot x=1-x,
$$

arrival 직후

$$
u^+=u^-+U(1-u^-),\qquad
a_k=u^+x^-,\qquad
x^+=x^-(1-u^+)
$$

로 고정한다. protocol times와 response $a_k$가 있는 표본에만 적용한다.
DB가 model-fit parameter가 아닌 요약 통계만 제공하면 이 식은 기전
reference이고 파라미터를 역추정하지 않는다.

### 4.4 느린 plasticity와 구조 변화의 경계

후속 전체식의 typed slot은

$$
\dot e_{ij}=-e_{ij}/\tau_e+\mathcal K(pre,post),
\qquad
\frac{d\log\bar g_{ij}}{dt}=\eta M_i e_{ij}-H_i,
$$

$$
A_{ij}:0\to1\text{ with hazard }b_{ij},\qquad
A_{ij}:1\to0\text{ with hazard }d_{ij}^{death}
$$

이다. Allen-Synphys는 동일 synapse의 long-term $e,M,H,A(t)$를 함께 주지
않으므로 이 항들은 첫 판본의 fitting state가 아니다.

## 5. 무차원 factor chart와 식별 부분공간

전체 생물 상태의 후보 slot과 실제 판본의 좌표를 분리한다. 이번 `small` DB
판본의 유일한 엄격 metric chart는 서로 다른 측정 단계에서 얻은 네 좌표

$$
z_{\rm strict}=\left(
\log\frac{|r_{1}|}{r_{\rm ref,\chi}},
\log\frac{L_{\rm soma}}{L_{\rm ref}},
\log\frac{R_{\rm in,post}}{R_{\rm ref}},
\log\frac{\tau_{m,\rm post}}{t_{\rm ref}}
\right).
$$

로 고정한다. 열은 각각 `synapse.psp_amplitude`, `pair.distance`, postsynaptic
cell의 `intrinsic.input_resistance`, `intrinsic.tau`다.
$L_{\rm ref}=1\,\mathrm m$, $R_{\rm ref}=1\,\Omega$, $t_{\rm ref}=1\,\mathrm s$이고,
$r_{\rm ref,\chi}$는 train split의 synapse-type stratum별 $|r_1|$ 중앙값이다.
이 네 좌표는 모두 무차원이다. `pair.distance`는 soma 간 거리이며 axonal path
length나 directed delay가 아니다. PSP와 PSC, mouse와 human, excitatory와
inhibitory를 같은 chart에 섞지 않는다.

latency·rise·decay는 중요한 관측량이지만 `synapse` pipeline이 train 위치와
무관하게 최대 50 Hz의 여러 pulse response를 평균해 산출하므로 late-pulse target과
원자료를 공유한다. 따라서

$$
z_{\rm shared}=\left(
\log\frac{|r_1|}{r_{\rm ref,\chi}},
\log\frac{d}{t_{\rm ref}},
\log\frac{\tau_r}{t_{\rm ref}},
\log\frac{\tau_d}{t_{\rm ref}}
\right)
$$

는 별도 `SHARED_SUMMARY_DIAGNOSTIC`에만 쓰고 passive-pullback 또는 독립 예측
증거로 승격하지 않는다. directed latency $d_{ij}$ 자체도 Riemannian node distance가
아니며 BA-CG1의 quasi-metric 경로와 계속 분리한다.

기본 반응과 예측 target의 pulse 위치를 분리한다. target은 dynamics QC를 통과한
50 Hz current-clamp summary 중

$$
y=\left(
\frac{s_\chi a_2}{r_{\rm ref,\chi}},
\frac{s_\chi a_{6:8}}{r_{\rm ref,\chi}},
\frac{s_\chi a_{9:12}^{250\rm ms}}{r_{\rm ref,\chi}},
v_{5:8}
\right)
$$

이며, 열은 `dynamics.pulse_amp_stp_initial_50hz`,
`pulse_amp_stp_induction_50hz`, `pulse_amp_stp_recovery_250ms`,
`variability_stp_induced_state_50hz`다. $s_{\rm ex}=+1$, $s_{\rm in}=-1$로 PSP의
생리적 부호를 양의 세기 방향으로 맞춘다. 첫 세 target은 무차원 진폭,
$v_{5:8}$는 source algorithm이 이미 log-normalized한 무차원 variability다.
$a_2$, $a_{6:8}$, $a_{9:12}^{250\rm ms}$, $v_{5:8}$는 각각 pulse 2,
pulse 6--8 중앙값, pulse 9--12 중앙값, pulse 5--8 variability를 뜻하는
**scalar summary 한 개씩**이다. 따라서 $y\in\mathbb R^4$이고
$J=\partial_z\mathcal H\in\mathbb R^{4\times4}$다.
`stp_initial_50hz`, `stp_induction_50hz`, `stp_recovery_250ms`와 PPR은 pulse 1을
대수적으로 재사용하므로 primary target에서 제외한다.

frozen source code에서 `synapse.psp_amplitude`는 이전 자극 뒤 8초보다 긴 휴지기를
가진 current-clamp response만 평균하며, target은 pulse 2, 6--8, 250 ms 회복 뒤
9--12를 쓴다. small DB는 `resting_state_fit.ic_pulse_ids` blob을 보존하지만
`pulse_response`와 `stim_pulse` 행은 0개라 row-level 교집합을 재검산할 수 없다.
이를 `PIPELINE_SEPARATED / ROW_LEVEL_UNVERIFIED`로 기록한다. 반대로 latency·kinetics는
shared-pulse임이 source code로 확인되어 엄격 chart에서 제외한다.

허용 domain은 $|r_1|>0$, $L_{\rm soma}>0$, $R_{\rm in,post}>0$,
$\tau_{m,\rm post}>0$이고 모든 target이 finite인 행이다. 0, 음수, unresolved fit에는
임의 epsilon을 더하지 않는다. 결측 행과 탈락 이유를 열별·stratum별로 기록한다.
$U=0,1$의 logit, $N,p,q$, receptor conductance, morphology, eligibility와
homeostasis는 이번 4차원 chart에 넣지 않는다. 네 좌표 complete case가 부족하면
차원을 결과에 맞춰 줄이지 않고 `INSUFFICIENT_IDENTIFIED_SUBSPACE`로 중지한다.

이하 수식의 $z$는 $z_{\rm strict}$만 뜻한다.

## 6. MEASUREMENT_MODEL과 Riemannian hypothesis

각 pair의 source-locked baseline $z$와 late-pulse target $y$, stratum $\chi$에 대해

$$
y_{ij}=\mathcal H(z_{ij},\chi_{ij};\psi)+\epsilon_{ij},
\qquad
\epsilon\sim\mathcal N(0,R_{\chi})
$$

로 둔다. experiment·slice·cell·pair nesting을 보존한다. $\mathcal H$는 train에서
고정하는 2차 다항 ridge map이다. 절편·선형·대칭 2차항을 포함한 15개 basis와
$\alpha\in\{10^{-6},10^{-5},\ldots,10^2\}$를 사용하고, slice-grouped 5-fold
inner CV로 하나를 고른다. $R_\chi$는 train residual의 diagonal covariance이며
training-standardized target 좌표에서 각 분산 하한을 $10^{-6}$으로 둔다.

source-locked response map의 Jacobian을

$$
J(z)=\frac{\partial\mathcal H(z,\chi)}{\partial z}
$$

라 하면 선택 metric은 Gaussian-location response pullback

$$
\boxed{
g_{\rm resp}(z)=J(z)^TR_\chi^{-1}J(z)
}
$$

이다. 이는 $R$이 $z$에 의존하지 않는 조건부 Gaussian-location model에서만
Fisher와 일치한다. 일반 Fisher metric이라고 넓혀 부르지 않는다. 항상 PSD이고,
$R_\chi\succ0$이며 $J$가 4차원 full rank인 support에서만 Riemannian metric이라고
부른다. 그 밖에는 PSD pseudometric이며 ridge를 생물 정보나 rank 회복으로
해석하지 않는다.

### 기준 계량과 비교량

이번 판본은 새 biological drift를 만들지 않는다:

$$
\Delta F_{CE}=0.
$$

train에서만 $z$의 평균 $m$과 covariance $\Sigma$를 고정하고
$g_{\rm ref}=\Sigma^{-1}$로 둔다. 수치적으로는 $\Sigma$에
$10^{-6}\operatorname{tr}(\Sigma)/4$의 고정 shrinkage를 적용하며, 이후 rechart에서는
원 chart에서 이미 shrink한 $g_{\rm ref}$ 자체를 covariant tensor로 운반한다.
rechart 뒤 covariance나 shrinkage를 재계산하지 않고, $I$를 모든 chart에서 새로
삽입하지 않는다. descriptive response-metric 차이는

$$
\Delta g_{\rm resp}(z)=g_{\rm resp}(z)-g_{\rm ref}
$$

이다. 이 차이는 새 CE 생물 동역학이 아니라 descriptive model comparison이다.
좌표불변 평가는 generalized eigenvalues of $(g_{\rm resp},g_{\rm ref})$,
$\operatorname{tr}(g_{\rm ref}^{-1}g_{\rm resp})$, determinant ratio와 line-element
ratio로만 한다. 기억·물리력·곡률 저장 주장이 아니다.

M0-ref는 $g_{\rm ref}$, M0-diag는 $\operatorname{diag}(\bar g_{\rm resp})$,
M0-const는 train 평균 $\bar g_{\rm resp}$를 쓰며 M1만 state-dependent
$g_{\rm resp}(z)$를 쓴다. M1 kernel은

$$
K_g(z,z')=\exp\left[-d_{g_{\rm resp}}(z,z')^2/(2\ell_g^2)\right]
$$

를 쓴다. 거리는 원 frozen chart의 $g_{\rm ref}$ distance로 만든 train-only
symmetric-union $k$NN adjacency에서 trapezoid line element와 Dijkstra shortest
path로 근사한다. train graph가 연결되지 않는 $k$는 무효다. held-out query는
$g_{\rm ref}$ 기준 가장 가까운 train node $k$개에만 붙이고, train-only $\mathcal H$로
query metric을 평가해 같은 trapezoid edge weight를 쓴다. query-query edge와
test point를 이용한 adjacency 재구성은 금지한다. finite train path가 없는 query는
예측하지 않고 실패/abstention으로 센다.

$k\in\{8,16,32\}$, bandwidth multiplier
$m\in\{0.25,0.5,1,2,4\}$를 grouped inner CV에서만 고른다. 각 foldㆍmetric의
실제 bandwidth는 $\ell_g=m\,\operatorname{median}_{e\in E_{\rm train}}\ell_e$로
고정하며, 모든 M0/M1에 같은 multiplier grid와 같은 train-only 계산 규칙을 준다.
예측분산은 선택이 끝난 train inner-OOF 잔차의 target별 대각 분산으로 한 번
고정하고 developmentㆍconfirmation에서 재추정하지 않는다. $R_\chi$의 diagonal 선택은 residual target independence를
가정하는 고정 Gaussian model choice이며 source fact가 아니다. direct quadratic
$\mathcal H$, raw 4-factor ridge, cell-type-only,
missingness-only도 예측 control이다. constant metric이면 선형 rechart와 동치이므로
M1이 M0-const를 이겨야 state-dependent geometry라는 표현을 허용한다.

## 7. DATA_PROVENANCE와 접촉 규칙

- official registry: `https://registry.opendata.aws/allen-synphys/`.
- official bucket: `s3://allen-synphys/`, `us-west-2`.
- official access code: `https://github.com/AllenInstitute/aisynphys`.
- primary paper: Campagnola et al., *Science* 375 (2022), DOI
  `10.1126/science.abj5861`.
- dataset terms: Allen Institute Terms of Use; 재배포 조건을 별도 receipt에 기록.
- official code commit `545a990ee171e6c0d23dd4bba413e1ccbf2f0853`의
  `SynphysDatabase.list_current_versions()`와 official download manifest는 schema 22의
  `synphys_r2.1_small.sqlite`를 current로 가리킨다. manifest와 HTTP metadata를
  receipt에 함께 보존한 뒤 이 파일만 내려받는다.
- raw path: `data/external/allen-synphys/` (gitignored, 원본 덮어쓰기 금지).
- API commit, DB filename/release/schema version, URL, bytes, SHA-256, retrieval
  time을 `artifacts/realdata/download-receipt.json`에 기록한다.
- small DB는 relational summary용이다. waveform/conductance 확인은 별도
  full/NWB confirmation contract가 필요하다.

## 8. DATA_SPLIT

- 공개 schema에는 안정적인 donor ID가 없으므로 독립 분할 단위는 `slice.ext_id`로
  고정한다. 같은 donor의 여러 slice일 가능성을 제거하지 못하므로 population
  generalization을 주장하지 않는다.
- 식별자를 UTF-8 문자열로 만든 뒤 SHA-256 첫 byte의 값 $b$로 고정 분할한다:
  $b\bmod10\in\{0,1,2,3,4,5\}$ train,
  $\{6,7\}$ development, $\{8,9\}$ confirmation.
- 같은 slice의 experiment·cell·pair는 한 split에만 둔다. 한 pair의 반복 protocol과
  summary도 같은 split에만 둔다.
- pair/response row 단위 random split은 금지한다.
- mouse V1과 human MTG, E/I sign, clamp mode는 섞지 않고 strata로 보고한다.
  primary는 표본이 허용할 경우 mouse V1의 source-standardized response mode다.
- primary stratum은 complete pair 80개, slice group 20개 이상이어야 하고 train,
  development, confirmation에 각각 최소 10/5/5 slice가 있어야 한다. 미달이면
  prediction은 `DIAGNOSTIC_ONLY`이며 confirmation을 열지 않는다.
- schema audit와 missingness count는 outcome-free 접촉이다. confirmation outcome은
  contract·source/math/audit PASS 전 열지 않는다.

## 9. OBSERVABLES와 RESIDUAL_RULE

### O1 — baseline reproduction

source paper/database가 제시하는 connection, amplitude, latency, kinetics와 STP
summary의 표본 수·단위·stratum 정의를 재현한다. 이는 pipeline gate이지 새
증거가 아니다.

SQL extractor와 ORM/schema extractor의 integer count는 정확히 같아야 하고,
SI↔표시단위 round-trip 상대오차는 $10^{-12}$ 이하여야 한다. 공식 문서에 없는
표본 수를 논문 수치에 억지로 맞추지 않는다.

### O2 — metric admissibility

- slice-cluster bootstrap 1,000회에서 선택 subspace rank가 동일하고 $R\succ0$이어야 한다.
  재표집 seed는 excitatory `83201`, inhibitory `83202`로 고정한다.
- 수학적 SPD 판정은 모든 평가 support point에서
  $\sigma_{\min}(R_\chi^{-1/2}Jg_{\rm ref}^{-1/2})>0$인지와 분리한다. 아래
  $10^{-4}$는 exact rank 정의가 아니라 practical stability gate다.
- $g_{\rm ref}^{-1}g_{\rm resp}$의 상대 최소 generalized eigenvalue
  $\lambda_{min}/\lambda_{max}$ 2.5% bootstrap quantile이 $10^{-4}$ 이하이거나
  $\kappa(R)>10^6$이면 차원을 제거하지 않고 `RANK_UNIDENTIFIED`로 판정한다.
- gauge 시험은 seed `83101..83116`의 QR orthogonal 16개, 각 좌표 scale이
  $0.5$ 또는 $2$인 diagonal 16개, $I\pm0.25e_ie_j^T$ ($i\ne j$) shear 24개와
  각 좌표 $\pm1$ translation 8개로 고정하며 $\kappa(A)\le4$만 허용한다.
  node labels와 원 adjacency를 transport하고 rechart에서 kNN을 재구성하지 않는다.
  line element와 generalized spectrum은 상대오차 $10^{-8}$, 수치
  geodesic-neighbor prediction은 상대오차 $10^{-4}$ 안에서 같아야 한다.

### O3 — held-out prediction

slice별

$$
\Delta\mathrm{ELPD}
=\mathrm{ELPD}(M1)-\mathrm{ELPD}(M0)
$$

를 slice별 log predictive density 합으로 계산한다. SE는 slice-level paired
difference의 표준오차다. M1 생존 조건은 development에서 모든 control 중 최선보다
$\Delta\mathrm{ELPD}>2\,SE$이고, nonzero difference를 가진 development slice의
75% 이상에서 방향이 일치하는 것이다.
confirmation은 같은 frozen 식을 한 번 적용한다. 실패는 metric의 존재가 아니라
“geometry가 raw factors보다 추가 예측을 준다”는 부모 가설을 기각한다.

잔차는 `D`(data/provenance), `I`(identifiability/rank), `P`(prediction),
`C`(causal alternative), `B`(biological mismatch), `T`(theory structure)로
분류한다.

## 10. FALSIFIER와 MATCHED_CONTROLS

- F1: current version/DB checksum/schema/license receipt 불완전 → outcome 분석 중지.
- F2: resting amplitude·soma distance·postsynaptic intrinsic·late-pulse column 의미,
  pulse 위치 또는 pipeline separation receipt가 source-locked 되지 않음 → passive metric 금지.
- F3: $J$ rank loss, $R$ singular, bootstrap subspace 불안정 → Riemannian claim 금지.
- F4: M1이 direct $\mathcal H$, raw/Euclidean, diagonal 및 constant-full-SPD M0 중
  최선보다 $2SE$ 이상 개선하지 못함 → predictive geometry STOP.
- F5: slice split leakage 또는 confirmation으로 bandwidth/dimension 선택 → run 무효.
- F6: coordinate rechart에서 invariant prediction/line element가 깨짐 → formula/implementation STOP.
- F7: strength/distance/membrane shuffle이 같은 결과, 또는 direct raw response model이
  동률·우세 → geometry는 diagnostic으로 강등.
- F8: conductance, $Npq$, eligibility, homeostasis, morphology를 직접 측정한 것처럼
  해석 → claim block.

Matched controls는 direct quadratic $\mathcal H$, source-provided STP summary model,
raw factor ridge, Euclidean reference, diagonal response metric, constant full response
metric, strength-only, distance-only, membrane-only, `SHARED_SUMMARY_DIAGNOSTIC`,
protocol-order shuffle, cell-type-only, missingness-only, disconnected-pair null이다.
자유도·inner-selection budget을 M0/M1에서 맞춘다.

## 11. MODEL_SELECTION과 REVISION_TRIGGER

metric chart는 위의 4차원 $z$ 하나이며 결과를 본 차원 축소나 membrane/STP
parameter 추가를 허용하지 않는다. 선택 metric은 Gaussian-location response
pullback 하나다. graph Laplacian, diffusion inverse, curvature,
Finsler/quasi-metric은 이번 판본에서 열지 않는다. 결과를 본 뒤 metric family,
split, endpoint, bandwidth grid, rank threshold 또는 stratum을 바꾸지 않는다.

source/schema mismatch는 outcome 전에 physics-sourcer revision 한 번으로 고칠 수
있다. P/I/B/T 잔차가 나오면 한 판본 한 구조 변경으로 BA-SRM2를 열고,
BA-SRM1 결과·반례·confirmation 접촉 여부를 보존한다. full/NWB waveform,
MICrONS morphology, de Vivo sleep scaling, eligibility/homeostasis를 추가하는 것은
각각 독립 measurement model과 새 contract가 필요한 후속 단계다.
