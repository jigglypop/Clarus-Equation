# CE 우주론 구조적 대체 경로

Status: COMPLETE

기준일: 2026-08-16  
입력: 00-contract.md, 11-math.md, current-run artifacts와 두 PREDECESSOR run

## 0. 경로 선택 규칙

각 표는 목표를 고정한 뒤 기존 반례와 다른 작용, current, readout 또는
likelihood를 비교한다. 후보 하나가 새로 채택하는 구조 공리는 최대 하나다.
나머지 미지량은 이미 유도됐다고 가정하지 않고 missing lemma 또는 자유도로
노출한다.

- target-aware: 이미 목표 숫자나 자료를 본 뒤 후보·계수·branch를 고른 경우.
- dof: 연속 parameter와 초기자료의 보수적 하한.
- look-elsewhere: 최소 discrete 선택 수. 연속 scan은 별도다.
- kill test: 경로를 살리는 조건이 아니라 실패시키는 사전 조건.
- 단계 지위: 정의 → 공리 → 정리/산출 → 미완성 순서.

어느 표도 관측 예측이나 이론 승격을 선언하지 않는다.

## 1. U1 — 정본·원장 통합 경로

U1의 수학적 inventory는 끝났지만 migration은 미완성이다.

| ID | 구조적으로 다른 경로 | 새 구조 공리 | 단계 지위 | dof / look-elsewhere | target-aware | 교차검증 | kill test |
|---|---|---|---|---|---|---|---|
| U1-A | typed Python registry + 별도 observation manifest | 모든 cosmology quantity는 role/model/source를 가진 typed entry다 | schema [공리]; exact 계산 [산출]; migration [미완성] | 연속 0; config ID 설계 선택 | 아니오 | old alias parity, exact residual, manifest hash | 이름 없는 default나 source 없는 observation이 활성 출력에 남음 |
| U1-B | declarative ledger에서 code/docs를 생성 | 선언 원장이 유일 source다 | schema [공리]; generated parity [산출]; generator [미완성] | 연속 0; serialization family 최소 2 | 아니오 | generated module과 docs table parity | generator 밖에서 고친 literal이 CI를 통과 |
| U1-C | legacy module은 compatibility facade, scientific API는 explicit config | 새 scientific API에는 implicit default가 없다 | boundary [공리]; wrappers [산출]; migration [미완성] | 연속 0; facade/registry 2 route | 아니오 | legacy snapshot, warning, explicit-config failure | exact와 legacy가 같은 key를 공유하거나 alias 제거로 consumer가 깨짐 |

우선순위는 U1-A다. U1-B는 문서 생성기로, U1-C는 비파괴 호환층으로 함께
사용할 수 있다.

## 2. U2 — 확률에서 바리온 abundance로

목표는 관측 $\Omega_b$를 coupling이나 경계조건에 넣지 않고 conserved visible
current와 absolute yield를 계산하는 것이다.

| ID | 경로 | 새 구조 공리 1개 | 단계 지위 | 연속 dof 하한 | look-elsewhere | target-aware | 현재 수치 | cross-output / kill test |
|---|---|---|---|---:|---:|---|---|---|
| U2-A | reacting two-current + $v_D$ free-energy relaxation | label-blind precursor affinity가 $\mu_*v_D'(y)$다 | current/EFT [공리]; $y\to q$, entropy [산출]; microscopic origin/yield [미완성] | $m_X,\mu_*,\kappa(T),Y_X,m_\chi,T_R$ 등 6+ | decay/charge assignment 최소 2 | 예 | $q=0.0486467$, $v_D''(q)=17.3786$ | entropy/heat, BBN, isocurvature; detailed balance가 $v_D$를 못 만들거나 $Y_X$를 관측에서 역산하면 kill |
| U2-E | extinction-conditioned cascade의 모든 node를 relic로 보존 | branching event마다 stable relic 하나가 남는다 | PGF [정리]; aggregate $Dq$ [산출]; species/equal-energy/current [미완성] | tree density, rate, reservoir, mass distribution, freeze clock 5+ | detector 4 × role 2 이상 | 예 | $Dq=0.1545875$; alternatives $0.14323,0.13245,0.07729$ | multiplicity cumulant, reservoir entropy, relic spectrum; historical node가 current로 남지 않거나 energy weight가 generation-dependent면 kill |
| U2-B | conserved $U(1)_Q$ + asymmetric visible/dark freeze-out | visible/dark transfer가 하나의 conserved charge를 공유한다 | action/current [공리]; chemical sharing [산출]; $q$ eigenvector와 $Y_Q$ [미완성] | source, portal, $m_\chi$, annihilation, reheating 5+ | charge matrix/stoichiometry 다수 | 예 | $\Omega_c/\Omega_b=(m_\chi/m_B)(1-q)/q$ | dark mass, symmetric remnant, collider/decay, isocurvature; charge matrix가 $q$ share를 못 만들거나 symmetric relic가 크면 kill |
| U2-C | rolling topological field + anomaly source | compact rolling field가 charge-violating plasma에 결합한다 | action/anomaly [공리]; $Y_Q\propto\dot\theta/T$ [산출]; rolling attractor [미완성] | $f_a,U,c_Q,c_A,\theta_i,\dot\theta_i,T_R$ 7+ | potential/anomaly family 다수 | 예 | static $\theta=q$이면 정확히 $Y_Q=0$ | parity-odd GW/gauge signal, EDM, isocurvature; $\dot\theta=0$, washout 또는 초기조건 민감성이 남으면 kill |

### 2.1 conditioned route의 detector 선택

같은 theorem에서

$$
m=Dq,\qquad 1-e^{-m},\qquad me^{-m},\qquad m/2
$$

가 모두 자연스럽다. U2-E가 가장 target에 가깝다는 이유만으로 aggregate
detector를 채택할 수 없다. covariant measurement/species action이 aggregate
node energy를 실제 observable로 고정해야 한다.

### 2.2 권장 결합과 재개 조건

가장 짧은 연구 결합은 U2-A의 entropy-consistent relaxation, U2-B의 conserved
charge/annihilation, U2-C의 total source 가운데 두 층을 사용하는 것이다.
U2-E는 independent cascade가 생기면 더 낮은 readout dof 후보가 될 수 있다.
다음 네 문장이 동시에 증명되기 전 absolute abundance는 미완성이다.

1. species label과 current가 local action에서 나온다.
2. equal conditional energy 또는 계산된 unequal weight가 나온다.
3. reaction/freeze-out surface가 field/temperature로 유일하게 정해진다.
4. total yield와 entropy dilution이 관측 density 없이 계산된다.

## 3. U3 — 암흑물질·암흑에너지 분할

목표는 한 면의 숫자 분할만이 아니라 stress, EOM과 perturbation을 함께 갖는
dark sector다.

| ID | 경로 | 새 구조 공리 1개 | 단계 지위 | dof 하한 / look-elsewhere | target-aware | 고정되는 것 | cross-output / kill test |
|---|---|---|---|---|---|---|---|
| U3-D1 | effective interacting vacuum $Q^\nu=\xi\Theta\rho_{\rm de}u^\nu$ | 이 transfer current를 dark EFT로 채택 | background [산출]; perturbation [미완성] | $\xi$, switch/initial ratio 2+; $Q^\mu$ frame 최소 2 | 예 | $r_*=\xi/(1-\xi)$ | $w_{\rm eff}$, $H(z)$, growth; $\delta Q^\mu$ instability 또는 $a^{-0.8176}$ scaling이 preregistered likelihood를 실패하면 kill |
| U3-D2 | conformally coupled exponential scalar | CDM metric이 $e^{2\beta\phi/M_{\rm Pl}}g_{\mu\nu}$다 | action/EOM [공리]/[산출]; UV map/full perturbation [미완성] | $\lambda,\beta,V_0,m_{c0},\phi_i,\dot\phi_i$ 6+ | 예 | dark-only scaling ratio와 $w_{\rm eff}$ | fifth force, CDM mass drift, CMB/growth; ghost/gradient/large fifth force 또는 UV mismatch면 kill |
| U3-D3 | decoupled stable relic DM + four-form vacuum | vacuum sector가 quantized four-form flux다 | stress/flux [산출]; relic yield와 branch selection [미완성] | relic mass/yield, bare vacuum, flux unit/branch 4+; 많은 branches | 예 | 두 absolute density를 별도 microphysics로 정할 가능성 | membrane transition, metastability, relic signal; flux spacing 과대 또는 target branch를 사후 선택하면 kill |
| U3-D4 | finite spinodal/phase-transition surface에서 interaction 종료 | $v_D''=0$이 covariant phase-switch 조건이다 | $x_{\rm sp}=1/D$ [산출]; stress/clock/$\Omega_m$ map [미완성] | transition scale, nucleation/friction, trajectory 3+ | 예 | transient composition과 종료면 후보 | latent heat, GW, $\Omega_i(z)$; $x$를 $\Omega_m$로 정의만 하거나 surface가 여러 번 교차하면 kill |

### 3.1 spinodal crossing의 가장 강한 조건부 형태

matter+DE subsystem composition을

$$
y(a)=\frac{\rho_m}{\rho_m+\rho_{\rm DE}}
$$

로 두고 conserved dust와 constant vacuum을 **먼저** 채택하면

$$
y'=-3y(1-y)<0
$$

이므로 $y:1\to0$ 사이에서 $y=1/D$를 정확히 한 번 지난다. 그 면에서
radiation을 full closure에 남기면

$$
\Omega_m=\frac{1-\Omega_r}{D},\qquad
\Omega_b=q(1-\Omega_r),
$$

$$
\Omega_c=\frac{(1-Dq)(1-\Omega_r)}{D},\qquad
\Omega_{\rm DE}=\left(1-\frac1D\right)(1-\Omega_r).
$$

따라서 $\Omega_b=q$는 exact가 아니라 radiation correction
$q(1-\Omega_r)$를 받는다. 이 대수는 two-root 문제를 피하고 crossing을
유일하게 하지만 아직 물리 action이 아니다.

반대로 full-density spectator

$$
\mathcal F=(D-1)\rho_m-\rho_{\rm DE}-\rho_r=0
$$

는 radiation을 포함해 $\Omega_m=1/D$와 $\Omega_b=q$를 정확히 주지만,
표준 $\rho_m\propto a^{-3}$, $\rho_r\propto a^{-4}$, constant-vacuum
history에서는 $\mathcal F$가 early/late에 모두 음수여서 root가 보통 0개
또는 2개다. late root를 고르는 oriented clock, hysteresis 또는 memory field가
없으면 transition surface는 유일하지 않다.

matter dust와 vacuum은 EOS가 다르므로 하나의 thermodynamic composition
coordinate $y$에 넣은 $v_D(y)$가 metric variation에서 위 두 stress를 내는지
증명해야 한다. 또 $v_D''(1/D)=0$에서는 relaxation eigenvalue가 0이어서
critical slowing이 생긴다. spinodal을 freeze-out surface라고 부르려면
nucleation, finite-rate crossing, entropy/latent heat를 포함한 action이 필요하다.
이것이 없으면 $y=1/D$는 background가 언젠가 지나는 kinematic label일 뿐이다.
두 spectator는 모두 target density 식을 coupling에 재배열한 존재구성이며
$H_\Lambda$, crossing scale $H_*$와 현재 $H_0$를 같게 만들지 않는다.
absolute density normalization과 observer epoch는 여전히 미완성이다.

### 3.2 영구 attractor 경계

U3-D1/D2의 영구 attractor에는 conserved baryon fraction이 0으로 간다.
full route는 action-defined $\Sigma_*$에서 dark interaction을 종료하고 이후
baryon/CDM/DE를 전방 적분해야 한다. “오늘”을 switch condition에 넣으면 kill다.

historical 3-layer $R_3$와 conditioned composition을 결합한

$$
(\Omega_b,\Omega_c,\Omega_{\rm de})
=(0.0479920,0.2624602,0.6895477)
$$

는 route-discovery diagnostic이다. detector, phenomenological $R_3$, transient
surface의 세 선택을 이미 본 target 뒤 결합했으므로 blind output이 아니다.

## 4. U4 — 배경팽창·성장 구현 경로

U4의 교정 수학은 닫혔고 product integration이 남았다.

| ID | 구현 경로 | 새 구조 공리 | 단계 지위 | dof / target-aware | 수치 교차검증 | kill test |
|---|---|---|---|---|---|---|
| U4-N1 | interval-local Simpson + local-step RK4 + Heath check | 공용 FlatFLRW kernel만 background를 계산한다 | 식 [정리]; scratch [산출]; product [미완성] | solver tolerance만; 아니오 | cubic $3\times10^{-15}$, EdS 4th-order, Heath 일치 | grid doubling 실패, trace/kinematic Ricci 불일치, 마지막 interval 누락 |
| U4-N2 | adaptive Gauss--Kronrod + adaptive ODE/collocation | declared error estimator를 결과에 포함한다 | algorithm [공리]; adapter [미완성] | tolerances; 아니오 | N1과 held-out background grid | tolerance를 줄여도 수렴하지 않거나 event/singularity가 숨겨짐 |
| U4-N3 | CLASS/CAMB background/transfer를 reference backend로 사용 | full Einstein--Boltzmann backend를 기준으로 고정 | solver [공리]; CE adapter [미완성] | precision/nuisance; 아니오 | CLASS↔CAMB↔N1 late limits | 같은 physical input의 two-solver discrepancy가 budget 초과 |

legacy nonuniform Simpson, average-step growth, radiation Ricci는 각각 완전 반례가
있으므로 compatibility snapshot 외 과학 route에는 재사용하지 않는다.

## 5. U5 — $H_0$ readout 세 경로

| ID | 경로 | 새 구조 공리 1개 | 단계 지위 | dof 하한 | target-aware / look-elsewhere | 교차량 | kill test |
|---|---|---|---|---:|---|---|---|
| U5-A | full CLASS/CAMB CMB spectra + 단일 공식 likelihood | full spectra likelihood를 primary readout으로 택한다 | forward/likelihood [미완성] | $h,\omega_b,\omega_c,A_s,n_s,\tau,\sum m_\nu,N_{\rm eff},Y_p,w_0,w_a$ + nuisance | 사전 manifest면 아니오; solver 2종 | TT/TE/EE, lensing, peak heights, damping, BAO/BBN | CLASS/CAMB held-out mismatch, full likelihood failure, CE-fixed parameter를 풀면 배제 |
| U5-B | controlled $r_s+\theta_*$ compressed inference | validated compression domain을 고정한다 | synthetic solver [산출]; physical calibration [미완성] | 최소 $h$, 미고정 early/late nuisance 전부 | fit/backend 최소 2; 관측 적용 전 freeze 필요 | $z_*,z_d,r_s,r_d,D_M$, full spectra grid | active input 불변, no/multiple root, grid/backend error 초과, peak-height/damping 실패 |
| U5-C | SN+anisotropic BAO inverse distance ladder | independent absolute $r_d$ calibration을 별도 likelihood로 공급 | identifiability [정리]; data adapter [미완성] | $H_0r_d,\Omega_m,w_0,w_a,\mathcal M_{\rm SN}$ + nuisance | survey/calibration choices; data를 본 뒤 고르면 예 | survey/redshift split, chronometer/time delay, U5-A/B $r_d$ | $r_d$ calibration null에서 $H_0$가 식별되거나 covariance 제거로 비정상 정밀도 증가 |

우선 승격 후보는 U5-A, 제한적 교차검증은 U5-B, early/late 독립성 검사는
U5-C다. 현재 세 경로 어디에서도 physical $H_0$ posterior를 계산하지 않았다.

## 6. U6-A — 원시 스펙트럼 후보

목표는 같은 perturbation action에서 $A_s,n_s,r$, running과 reheating dependence를
계산하는 것이다.

| ID | action/readout | 새 구조 공리 1개 | 단계 지위 | dof 하한 / look-elsewhere | target-aware | 현재 산출 | cross-output / kill test |
|---|---|---|---|---|---|---|---|
| U6-S1 | $R+R^2/(6M^2)$ + Mukhanov--Sasaki | Starobinsky gravity action을 inflation sector로 택한다 | slow-roll [산출]; exact modes/scale origin [미완성] | $M$, reheating history 2+; inflation-family 선택 | 예 | $N=55$: $n_s=0.9649772$, $r=0.0034983$, running $-6.21\times10^{-4}$ | exact mode spectrum, reheating consistency, non-Gaussianity; joint $n_s,r$ 또는 EFT/scale generator 실패면 kill |
| U6-S2 | predecessor의 canonical CE scalar $M^4v_D(y(\phi))$를 inflaton으로 사용 | 기존 order parameter를 inflaton으로 식별한다 | action 존재 [산출]; slow-roll basin/spectrum [미완성] | field normalization $F$, $M$, initial field, reheating 4+; field map 다수 | 예 | 정지점/Hessian만 알려짐 | 50--60 e-fold, $A_s,n_s,r$, running; plateau가 없거나 field-domain/EFT가 깨지면 kill |
| U6-S3 | conditioned branching fluctuation을 spectator/curvaton sector로 전환 | branching relic에 하나의 light spectator current를 부여한다 | stochastic theorem [정리]; curvature transfer [미완성] | spectator mass/decay, energy fraction, initial state 4+; detector/transfer 다수 | 예 | conditioned cumulants만 계산 가능 | isocurvature, local $f_{\rm NL}$, scale dependence; amplitude target fit 또는 isocurvature/non-Gaussianity 실패면 kill |

U6-S1이 가장 낮은 기능적 자유도를 갖지만 $A_s$가 $M$을 calibration한다.
U6-S2/S3은 CE core와 더 직접 연결되지만 missing lemma가 더 많다.

## 7. U6-B — late vacuum 절대척도 후보

| ID | 경로 | 새 구조 공리 1개 | 단계 지위 | dof 하한 / look-elsewhere | target-aware | 교차량 | kill test |
|---|---|---|---|---|---|---|---|
| U6-L1 | phase-flow + current apparent horizon 또는 true de Sitter horizon | phase transfer operator가 $\kappa=\pi^2/2$, $s_0=-\pi\delta(1-q)$를 만든다 | variational existence [산출]; micro transfer [미완성] | $K$와 boundary/epoch convention; horizon 2종, sign/integer 다수 | 예 | apparent route $67.248$; true dS route는 $\Omega_\Lambda=0.6891$일 때 $H_0=81.010$ | transfer spectrum, horizon/epoch dependence; 결과 뒤 horizon·integer·sign을 바꾸거나 micro operator가 계수를 못 만들면 kill |
| U6-L2 | RG dimensional transmutation | CE topology가 renormalizable field content를 하나 고른다 | RG scale formula [정리]; field map [미완성] | UV coupling, thresholds; many representations | exponent를 역산하면 예 | running, thresholds, relics | independently fixed $b,g$가 필요한 scale/sign/stress를 못 만들면 kill |
| U6-L3 | four-form flux + membrane/sequestering completion | CE sector가 flux quantum과 global constraint를 고른다 | flux stress [산출]; branch/radiative stability [미완성] | bare vacuum, flux unit, integer/history; many branches | 가까운 branch 선택 시 예 | membrane transitions, spacing, gravitational response | spacing 과대, empty-universe/metastability 실패, loop vacuum이 다시 민감하면 kill |
| U6-L4 | causal-volume stochastic vacuum | branching cells가 diffeomorphism-invariant $V_4$ measure다 | order law [산출]; coefficient/sign/noise [미완성] | cell volume, noise coefficient, sign/correlation, initial state 4+ | coefficient fit 시 예 | $w(z)$ noise, sign statistics, large-angle correlations | CMB/expansion noise bound 초과 또는 안정한 positive branch가 없으면 kill |

가장 생산적인 조합은 inflation scale에는 U6-S1+U6-L2, late vacuum에는
U6-L1/L3/L4 가운데 하나를 독립적으로 시험하는 것이다. 같은 $H$ identity의
$H_0$와 $\rho_\Lambda^{1/4}$는 cross-output 두 개로 세지 않는다.

## 8. U7 — provenance와 blind 검증

| ID | 경로 | 새 구조 공리 1개 | 단계 지위 | dof / look-elsewhere | target-aware | 교차량 | kill test |
|---|---|---|---|---|---|---|---|
| U7-V1 | 아직 공개되지 않은 future release를 immutable manifest로 사전등록 | hash/release 전 후보·prior·kill rule을 동결한다 | protocol [공리]; evaluation [미완성] | 고정 후보 수를 manifest에 기록 | 아니오가 되도록 설계 | primary likelihood + predeclared secondary observable | freeze 전 자료 노출, unblinding 뒤 parameter/threshold 변경, covariance/hash 불일치 |
| U7-V2 | 이미 본 expansion data와 독립인 cross-experiment observable을 보류 | primary fit에 쓰지 않은 observable 하나를 holdout으로 고정 | split [공리]; current availability [미완성] | observable/survey 선택 수 명시 | 선택을 지금 하면 exploratory | lensing/growth/time-delay/BBN 중 사전 하나 | shared likelihood/object/covariance가 있거나 fit에 간접 사용됐으면 holdout 자격 kill |
| U7-V3 | immutable synthetic challenge + pipeline injection | simulation truth와 systematics budget을 blind한다 | software validation [산출]; natural confirmation 아님 | mock families/systematics | 아니오 | coverage, bias, covariance, null recovery | truth leakage, undercoverage, wrong covariance; 통과해도 물리 이론 확인으로 부르면 kill |
| U7-V4 | 단일 공식 current likelihood의 완전 재분석 | hybrid summary 대신 한 release/model/chain만 쓴다 | exploratory likelihood [경험식]; blind [미완성] | published model+nuisance | 예 | full posterior/covariance, solver crosscheck | posterior 성분 혼합, source/hash/model 누락, 같은 data를 holdout으로 재사용 |

현재 가능한 것은 U7-V3/V4의 pipeline·exploratory 검증이다. independent
confirmatory 판정은 U7-V1 또는 실제 독립성이 증명된 U7-V2가 필요하다.

## 9. 결합 경로와 의존성

낮은 현재 자유도와 강한 kill test를 기준으로 한 우선 작업 순서는 다음과 같다.

1. U1-A+C: exact/legacy/runtime/observation을 typed ledger와 alias로 분리한다.
2. U4-N1+N3: analytic kernel을 고정하고 CLASS/CAMB에 독립 대조한다.
3. U2-A/B와 U2-E를 경쟁시킨다. 같은 target을 공유해도 detector/current가
   다르므로 둘을 평균내지 않는다.
4. U3-D2 또는 U3-D4가 transition stress와 perturbation을 통과할 때만 density
   triplet을 forward kernel에 넣는다.
5. U5-A를 primary, U5-B/C를 독립 수치·late-route 교차검증으로 사용한다.
6. primordial은 U6-S1의 exact MS/reheating부터, vacuum은 U6-L2/L3/L4와
   horizon이 분리된 U6-L1을 경쟁시킨다.
7. 모든 model choice, priors와 kill thresholds를 U7-V1/V2 전에 동결한다.

의존성은

$$
\text{typed ledger}
\to
\{\text{current/yield},\text{dark action}\}
\to
\text{common FLRW/perturbation kernel}
\to
\{H_0,\text{growth},\text{CMB}\}
\to
\text{blind likelihood}
$$

순이다. 원시 scale과 late vacuum scale은 같은 parameter를 억지로 재사용하지
않고 별도 action에서 만나야 한다.

## 10. 목표별 현재 최선 후보와 재개 조건

| 목표 | 현재 최선 후보 | 지금 닫힌 것 | 다음 한 문장 lemma |
|---|---|---|---|
| U1 원장 | U1-A+C | exact/legacy 대응과 비파괴 순서 | 모든 consumer가 role/model/source 없는 값을 거부한다 |
| U2 abundance | U2-A+B, U2-E 경쟁 | conditional PGF, current EFT algebra | action이 species share와 total yield를 관측 density 없이 함께 고정한다 |
| U3 dark split | U3-D2 또는 transient U3-D4 | background point/Jacobian/no-go | UV coupling과 transition이 background+perturbation에서 유일하고 안정하다 |
| U4 background | U4-N1+N3 | exact FLRW/limits와 replacement numerics | product가 common kernel을 사용하고 all-grid regression을 통과한다 |
| U5 $H_0$ | U5-A, B/C crosscheck | identifiability와 synthetic solver | immutable full likelihood에서 모든 physical input이 활성이다 |
| U6 primordial | U6-S1 | conditional $n_s,r$, running | CE dynamics가 $M$과 reheating을 $A_s$ target 없이 정한다 |
| U6 vacuum | U6-L2/L3/L4, separated L1 | scale mechanisms와 phase horizon P0 | coefficient/branch/horizon이 action에서 정해지고 radiatively stable하다 |
| U7 blind | U7-V1/V2 | protocol과 prior provenance 반례 | 독립 release/object/covariance가 freeze 뒤 처음 공개된다 |

## 11. 종료 체크

- [x] 모든 [미완성] 목표에 구조적으로 다른 후보를 최소 3개 두었다.
- [x] 후보별 단계 지위, dof 하한, target-awareness, look-elsewhere와 kill test를 기록했다.
- [x] $Dq$ readout 네 종류와 species/current bridge를 분리했다.
- [x] reacting current, entropy, freeze-out, D1/D2 stress/EOM/Jacobian을 포함했다.
- [x] FLRW/H0/primordial/vacuum/provenance 경로를 서로 다른 실패조건으로 분리했다.
- [x] target claim은 보존하고 완전 반례가 맞은 route claim만 제외했다.

유망한 후보가 여럿이라는 사실은 그중 하나가 참이라는 증명이 아니다. 다음 gate는
각 missing lemma가 실제 action, code와 independent data에서 닫혔는지만 판정해야 한다.
