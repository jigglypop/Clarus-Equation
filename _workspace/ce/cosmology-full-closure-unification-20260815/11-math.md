# CE 우주론 전면 닫힘 수학 검증

Status: COMPLETE

검증 기준일: 2026-08-16  
역할: `ce-math-verifier` 독립 검산  
수치 증인: `artifacts/verify_full_cosmology_math.py`

PREDECESSOR: `_workspace/ce/cosmology-theory-repository-audit-20260815/11-math.md`  
PREDECESSOR: `_workspace/ce/cosmology-density-bridge-derivation-20260815/11-math.md`

## 0. 판정의 범위

이번 레인은 목표 주장과 특정 구현 경로를 분리했다. 따라서 아래의 `P0`는 그
반례가 직접 맞은 route claim만 제거하며, 바리온 abundance, 암흑부문 분할,
$H_0$, 원시 스펙트럼과 진공 절대척도라는 목표는 `[미완성]`으로 유지한다.

독립 검산으로 새로 닫힌 수학은 다음이다.

1. exact 원장과 legacy/표시/제품/관측 원장은 같은 숫자처럼 보여도 서로 다른
   typed quantity다.
2. 소멸 조건부 Poisson 분지과정의 offspring law는 정확히 평균 $Dq$의
   Poisson이고, equal-node-energy aggregate readout에서는 matter 내부 조성이
   $Dq$다.
3. reacting-current와 interacting-dark 후보의 배경 EOM, entropy 부호와
   dark-only Jacobian은 조건부로 맞다.
4. radiation 포함 평탄 FLRW, Ricci trace, 거리, 비균일-grid ODE의 교정식은
   맞다. 기존 `cosmology.py`와 `hubble_tension.py`의 특정 경로에는 재현 가능한
   P0 반례가 있다.
5. Starobinsky slow-roll 수치는 맞지만 $A_s$가 scalaron scale을 보정한다.
   phase-area 수치를 진짜 de Sitter entropy와 현재 Hubble entropy 중 어느
   것으로 읽는지에 따라 $H_0$가 달라진다.

이번 레인에서 활성 `[예측]`으로 판정한 항목은 **0개**다. 수치 PASS는 아래
missing lemma를 대신하지 않는다.

## 1. 공통 정의역, 차원과 지위 규칙

자연단위 $c=\hbar=k_B=1$, metric 부호 $(-+++)$를 쓴다. $N=\log a$이고
평탄 FLRW의 Ricci 부호는 $R=+6(\dot H+2H^2)$다. 모든 log, exp, 확률,
고정점과 fraction의 인자는 무차원이다.

| 양 | 질량차원 | 검산 경계 |
|---|---:|---|
| $q,D,\delta,y,\xi,\lambda,\beta,N,s=\log S$ | 0 | exp/log/fixed-point core |
| $H,T,m,\phi,M,M_{\rm Pl},\kappa$ | 1 | 기준 scale 없이 log에 넣지 않음 |
| $n_i,s_{\rm ent},J_i^\mu,S^\mu$ | 3 | number/entropy current |
| $\nabla_\mu J_i^\mu$ | 4 | 반응률 density |
| $\rho,p,V,T^{\mu\nu}$ | 4 | action density와 stress |
| $Q$ in $\dot\rho+3H(\rho+p)=Q$ | 5 | energy transfer |
| $V_4$ | $-4$ | $V_4^{-1/2}$는 cosmological-constant 차원 2 |

`P0`는 완전 반례 또는 결론을 바꾸는 식 오류, `P1`은 missing lemma·숨은
자유도·정밀도 공백, `P2`는 의미를 흐리는 표기/원장 문제다. 표의 `PASS`는
그 좁은 명제에 P0--P2 결함을 찾지 못했다는 뜻이지 이론 승격이 아니다.

## 2. U1--U7 최소 주장 원장

### 2.1 U1 — 정본과 수치 원장

| Claim ID | 최소 주장 | 실제 지위 | P 판정 | 범위 |
|---|---|---|---|---|
| `T-U1-CANON` | 모든 판본을 하나의 typed registry와 명시적 named configuration으로 연결한다 | `[미완성]` | P1 | inventory는 끝났으나 migration은 gate 뒤 작업 |
| `R-U1-Q-DEF` | $q_{\rm ext}$는 소멸확률, $s_{\rm branch}=1-q_{\rm ext}$는 생존확률이다 | `[정의]` | P2 | legacy docstring의 survival 혼동만 교정 필요 |
| `R-U1-Q-THM` | $D>1$에서 작은 고정점은 $(0,1/D)$에 유일하고 $Dq<1$이다 | `[정리]` | PASS | PREDECESSOR의 닫힌 정리 유지 |
| `R-U1-CORE` | $\alpha_s\to s_W^2\to\delta\to D\to q$가 CE core를 고른다 | `[경험식]` | P1 | $\alpha_s$의 scale/scheme과 $s_W^2=4\alpha_s^{4/3}$은 외부/모형 입력 |
| `R-U1-EXACT` | full-precision 사슬의 수치가 원장값이다 | `[산출]` | PASS | Decimal 70자리 독립 검산 |
| `R-U1-LEGACY` | $D=3.17776$의 $q$와 `0.0487/0.2623/0.6891`은 호환값이다 | `[공리]` | P2 | exact theorem output이나 관측 snapshot으로 부르면 안 됨 |
| `R-U1-OBS` | 관측 posterior와 theory/runtime 값을 같은 상수 원장에 둔다 | `[미완성]` (동일시 route 삭제) | P0 | provenance·covariance가 다른 quantity를 동일시하는 경로만 제외 |

### 2.2 U2 — 확률에서 바리온 abundance로

| Claim ID | 최소 주장 | 실제 지위 | P 판정 | 범위 |
|---|---|---|---|---|
| `T-U2-ABS` | 관측 밀도를 넣지 않고 $\Omega_bh^2$ 또는 $\Omega_b(\Sigma_*)$를 계산한다 | `[미완성]` | P1 | total yield, mass, entropy와 $H_*$가 필요 |
| `R-U2-DIRECT` | $q_{\rm ext}$ 자체가 오늘의 $\Omega_b$다 | `[공리]`인 legacy route | P0 | 정리·예측 지위만 제거; `LEGACY_DIRECT_READOUT_V1` 경계모형은 보존 |
| `R-U2-GW` | 소멸 조건부 offspring law는 ${\rm Poisson}(Dq)$다 | `[정리]` | PASS | Bayes/PGF에서 직접 증명 |
| `R-U2-COMP` | equal-energy conserved node의 aggregate descendant fraction은 $Dq$다 | `[산출]` | P1 | node-to-species/current 측정연산자가 추가 공리 |
| `R-U2-REACT` | two-current free-energy EFT가 $y\to q$를 안정화하고 entropy를 증가시킨다 | `[산출]` | P1 | effective Onsager closure는 맞지만 microscopic action은 없음 |
| `R-U2-FREEZE` | 동결 뒤 yield 식으로 $\Omega_bh^2$를 계산할 수 있다 | `[산출]` | P1 | $Y_X$, 질량, stoichiometry, dilution은 아직 자유 |
| `R-U2-SPIN` | $x_{\rm sp}=1/D$와 $m=Dq$에서 $mx_{\rm sp}=q$다 | `[산출]` | P1 | 대수는 exact; $x_{\rm sp}\leftrightarrow\Omega_m$ 물리 사상은 미유도 |

### 2.3 U3 — 암흑물질·암흑에너지 분할

| Claim ID | 최소 주장 | 실제 지위 | P 판정 | 범위 |
|---|---|---|---|---|
| `T-U3-SPLIT` | 하나의 공변 이론이 background와 perturbation에서 $\rho_c/\rho_{\rm de}$를 고정한다 | `[미완성]` | P1 | D1은 perturbation prescription, D2는 UV map과 full eigenmodes가 부족 |
| `R-U3-RLO` | $R_D=\alpha_sD$ 또는 3-layer $R$이 자연의 dark ratio다 | `[경험식]` | P1 | action에서 coupling을 유도하지 못함 |
| `R-U3-D1` | $Q^\nu=\xi\Theta\rho_{\rm de}u^\nu$는 안정한 dark-ratio fixed point를 갖는다 | `[산출]` | P1 | background 정리는 맞지만 $u^\nu,\delta Q^\nu$가 perturbation에서 미고정 |
| `R-U3-D2` | conformal scalar action의 dark-only scaling point와 2D Jacobian이 안정하다 | `[산출]` | P1 | $\lambda,\beta$가 target에서 역산됐고 fifth-force/full perturbation 미검증 |
| `R-U3-ALLFIX` | conserved baryon을 포함한 세 fraction이 가속기의 nonzero 상수 fixed point다 | `[미완성]` (영구 route 삭제) | P0 | no-go는 영구 fixed point만 죽이며 transient 목표는 보존 |
| `R-U3-TRANSIENT` | action-defined $\Sigma_*$에서 비율을 읽고 이후 보존 fluid를 전방 적분한다 | `[미완성]` | P1 | transition clock/action과 유일성이 필요 |

### 2.4 U4 — 배경팽창과 성장

| Claim ID | 최소 주장 | 실제 지위 | P 판정 | 범위 |
|---|---|---|---|---|
| `T-U4-KERNEL` | radiation 포함 flat FLRW/거리/성장을 하나의 kernel로 구현한다 | `[미완성]` | P1 | 수학은 검증됐고 product integration이 남음 |
| `R-U4-FLRW` | radiation 포함 $E(a)$, $R/H^2$와 세 극한 | `[정리]` | PASS | kinematic/trace 독립 일치 |
| `R-U4-RICCI-OLD` | legacy $12-9\Omega_m$이 radiation/running 배경에도 정확하다 | `[미완성]` (기존 산출 삭제) | P0 | radiation 및 $\epsilon'$ 누락 경로만 제외 |
| `R-U4-QUAD-OLD` | 현재 `simpson(y,x)`가 비균일 grid와 짝수 점을 처리한다 | `[미완성]` (기존 산출 삭제) | P0 | `compute_s_of_a` 의존 branch 범위 |
| `R-U4-GROWTH-OLD` | 현재 `solve_growth`가 arbitrary grid API를 만족한다 | `[미완성]` (기존 산출 삭제) | P0 | uniform-$N$ CLI snapshot은 별도 compatibility 결과 |
| `R-U4-REPLACE` | interval-local quadrature와 local-step RK4가 analytic/Heath limit에 수렴한다 | `[산출]` | PASS | grid refinement 기록 있음 |

### 2.5 U5 — $H_0$ readout

| Claim ID | 최소 주장 | 실제 지위 | P 판정 | 범위 |
|---|---|---|---|---|
| `T-U5-H0` | physical densities, neutrino, recombination과 likelihood에서 $H_0$를 식별한다 | `[미완성]` | P1 | 현재 관측 inference 없음 |
| `R-U5-TOY` | legacy toy가 baryon-aware CMB $\theta_*$에서 $H_0$를 추출한다 | `[경험식]` (baryon-aware 산출 삭제) | P0 | `om_b_h2`가 bit-identical하게 미사용 |
| `R-U5-FULL` | CLASS/CAMB full spectra likelihood route | `[미완성]` | P1 | 가장 강한 후보이나 adapter/likelihood 실행이 아직 없음 |
| `R-U5-COMP` | supplied $z_*$의 compressed solver가 synthetic $h$를 복원한다 | `[산출]` | P1 | same-grid 복원은 tautological; 독립 grid convergence와 full-solver calibration 필요 |
| `R-U5-IDL` | calibration 없는 BAO+SN은 $H_0r_d$만 식별한다 | `[정리]` | PASS | finite $H_0$ posterior가 나오면 hidden prior/bug |

### 2.6 U6 — 원시 스펙트럼과 우주상수 절대척도

| Claim ID | 최소 주장 | 실제 지위 | P 판정 | 범위 |
|---|---|---|---|---|
| `T-U6-PRIM` | perturbation action, MS evolution, vacuum, reheating에서 $A_s,n_s,r$을 함께 산출한다 | `[미완성]` | P1 | full mode integration과 CE scale generator 없음 |
| `R-U6-PROJECT` | 이미 본 $A_s$에 가까운 projector가 사전 예측이다 | `[경험식]` | P1 | target-aware 후보 선택; prediction 아님 |
| `R-U6-R2` | Starobinsky action의 first-order slow-roll $n_s,r$과 $A_s$ normalization | `[산출]` | P1 | $A_s$가 $M$을 고정하고 reheating이 $N_*$를 고정 |
| `T-U6-LAMBDA` | 관측 $H_0,\Omega_\Lambda$ 없이 vacuum absolute scale을 고른다 | `[미완성]` | P1 | RG/flux/stochastic/phase 모두 selection lemma가 없음 |
| `R-U6-IDENTITY` | 같은 horizon convention의 Friedmann--entropy 관계 | `[정리]` | PASS | one-scale identity이지 독립 예측 둘이 아님 |
| `R-U6-PHASE-VAR` | 선택한 quadratic $s$-flow action이 phase law를 정지해로 갖는다 | `[산출]` | P1 | 존재구성은 맞지만 $\kappa,s_0,K$의 미시 기원 없음 |
| `R-U6-PHASE-H0` | 진짜 de Sitter entropy가 곧 $H_0=67.248\ldots$를 준다 | `[경험식]` (무표시 H0 route 삭제) | P0 | phase $H$가 $H_\Lambda$라면 $H_0=81.010\ldots$; apparent-horizon 해석은 별도 경계공리 |
| `R-U6-RG/4F/STOCH` | RG, four-form, stochastic volume이 CE scale을 유일하게 고른다 | `[미완성]` | P1 | 각각 field content, branch, coefficient/sign이 자유 |

### 2.7 U7 — provenance와 blind 판정

| Claim ID | 최소 주장 | 실제 지위 | P 판정 | 범위 |
|---|---|---|---|---|
| `T-U7-PROV` | 단일 공식 release/likelihood, covariance, hash, snapshot을 고정한다 | `[미완성]` | P1 | inventory schema는 있으나 machine manifest migration 전 |
| `R-U7-HYBRID` | 다른 posterior 성분을 섞은 tuple이 하나의 공개 posterior다 | `[미완성]` (hybrid route 삭제) | P0 | `Planck_ACT_SPT_combined`와 식별 불가 row만 제외 |
| `R-U7-DESI` | 내장 DESI vector/SPD covariance의 이차형식 | `[산출]` | P1 | exploratory이며 parser 전반의 SPD gate와 immutable asset hash 필요 |
| `R-U7-HOLDOUT` | 현재 future manifest가 독립 confirmatory holdout이다 | `[미완성]` | P1 | `unassigned/NOT_READY`; 이미 본 DR2는 holdout 아님 |
| `R-U7-PRED` | 현재 density/$H_0$/primordial 근접도가 blind prediction이다 | `[경험식]` (blind route 삭제) | P0 | target-aware 또는 supplied-boundary 비교만 제외; 목표는 유지 |

공식 source lane의 Evidence ID를 대조하면 U7 판정은 다음처럼 고정된다.

- SRC-FULL-P0-01: 공식 DESI DR2 flat $\Lambda$CDM은
  $\Omega_m=0.3027\pm0.0036$, 따라서
  $\Omega_\Lambda=0.6973\pm0.0036$이며 같은 표의
  $\Omega_m$--$H_0$ 상관계수는 $-0.975$다. CE runtime
  $0.6891$의 단변량 잔차는 저장소의 $-0.78\sigma$가 아니라
  $-2.28\sigma$지만 full joint 판정은 공개 chain covariance를 써야 한다.
- SRC-FULL-P0-02: Planck_ACT_SPT_combined는 하나의 공식
  likelihood/chain이 아니므로 covariance가 존재하지 않는다.
- SRC-FULL-HOLD-01, SRC-OBS-HOLD-02: manifest는 unassigned이고
  qualifying future release를 찾지 못했다. 독립 holdout 수는 0이다.
  공개처 밖의 전역 부재까지 증명한 것은 아니다.
- SRC-U6-HOR-01은 current flat-FLRW apparent horizon과 asymptotic
  de Sitter event horizon의 구분을 확인한다. 이는 아래 phase P0의
  외부 provenance이며 그 자체가 CE phase law를 증명하지 않는다.

## 3. U1 독립 수치 원장

고정점은

$$
q=e^{-D(1-q)},\qquad g(q)=\log q+D(1-q)=0
$$

이다. $g''(q)=-q^{-2}<0$, $g(0^+)=-\infty$, $g(1)=0$이고

$$
g(1/D)=D-1-\log D>0
$$

이므로 $(0,1/D)$에 작은 근이 정확히 하나 있다. map derivative는 작은
근에서 $f'(q)=Dq<1$, $q=1$에서 $f'(1)=D>1$이다. 따라서 `q=소멸`,
`1-q=생존`의 의미가 고정된다.

70자리 Decimal 검산의 핵심 값은 다음과 같다.

| role | 값 |
|---|---:|
| exact $s_W^2=4\alpha_s^{4/3}$ | $0.231222068260755116590565382964\ldots$ |
| exact $\delta=s_W^2(1-s_W^2)$ | $0.177758423409973817923268877860\ldots$ |
| exact $D=3+\delta$ | $3.177758423409973817923268877860\ldots$ |
| exact $q$ | $0.048646719644028206426163828128\ldots$ |
| exact $1-q$ | $0.951353280355971793573836171872\ldots$ |
| exact $Dq$ | $0.154587523120074076002229959713\ldots$ |
| legacy $D=3.17776$의 $q$ | $0.048646633337214076305948590183\ldots$ |
| exact--legacy $q$ 차이 | $8.6306814130\times10^{-8}$ |

runtime triplet의 raw sum은 정확히 $1.0001$이고 이를 flat background용으로
정규화하면

$$
(\widehat\Omega_m,\widehat\Omega_\Lambda)
=(0.3109689031096890311,0.6890310968903109689)
$$

다. raw $0.6891$과 normalized $0.689031\ldots$는 같은 quantity가 아니다.
`artifacts/canonical-version-map.md`의 `CE_CORE_EXACT_V1`,
`LEGACY_DELTA_5DP_V1`, `LEGACY_ROUNDED_RUNTIME_V1`, 관측 manifest를 분리하는
설계가 수학적으로 필요하다.

## 4. U2: 조건부 분지과정, current와 abundance

### 4.1 소멸 조건부 정리

offspring PGF를 $f(s)=e^{D(s-1)}$, 전체 소멸 사건을 $E$라 하자. $K=k$이면
모든 subtree가 소멸할 확률이 $q^k$이므로

$$
P(K=k\mid E)=\frac{P(K=k)q^k}{q},
$$

$$
E[s^K\mid E]=\frac{f(qs)}q=e^{Dq(s-1)}.
$$

따라서 $K\mid E\sim{\rm Poisson}(m)$, $m=Dq<1$이다. subcritical tree의
root 포함 total progeny $T$에는

$$
E[T]=\frac1{1-m},\qquad E[T-1]=\frac m{1-m}.
$$

독립 tree를 많이 합치고 모든 node의 보존 energy가 같으면 강한 대수법칙으로

$$
\frac{\sum_j(T_j-1)}{\sum_jT_j}\longrightarrow
\frac{E[T-1]}{E[T]}=m=Dq.
$$

이 마지막 줄은 `descendant=baryon`, `root=non-baryon`인 **aggregate node
detector** 아래의 matter-composition 산출이다. conditioned PGF 자체가 그
detector를 고르지는 않는다.

| 같은 conditioned law의 readout | 값 |
|---|---:|
| aggregate descendants / all nodes | $m=0.1545875231200741$ |
| $P(K\ge1\mid E)$ | $0.1432314981920089$ |
| $P(K=1\mid E)$ | $0.1324457205817941$ |
| $E[(T-1)/T]$ | $m/2=0.0772937615600370$ |

legacy runtime triplet의 matter 내부 baryon target은
$0.0487/(0.0487+0.2623)=0.1565916398713826$다. 네 readout 중 $m$이 가장
가깝지만 차이는 $-0.0020041167513086$이고, target을 본 뒤 고른 최소
4-way 선택이다. 따라서 확률 정리는 PASS지만 species readout은 target-aware
`[미완성]`이다.

### 4.2 reacting-current 후보

전구체 current를 $N_i^\mu=n_i u^\mu$, $y=n_b/(n_b+n_c)$라 두고

$$
\rho=m_Xn+\rho_{\rm th}(n,s_{\rm ent})+n\mu_*v_D(y),
$$

$$
v_D'(y)=\log y+D(1-y),\qquad v_D''(y)=y^{-1}-D
$$

로 둔다. Brown-type convective constraint가 있는 perfect-fluid action에
명시적 Onsager closure

$$
\nabla_\mu N_b^\mu=\Gamma,\qquad
\nabla_\mu N_c^\mu=-\Gamma,
$$

$$
\mathcal A=\mu_b-\mu_c=\mu_*v_D'(y),\qquad
\Gamma=-\frac{\kappa n}{\mu_*}\mathcal A
$$

를 붙이면 FLRW에서

$$
\dot y=-\kappa v_D'(y),\qquad
\delta\dot y=-\kappa v_D''(q)\delta y.
$$

$v_D''(q)=17.378612228\ldots>0$이고

$$
T\nabla_\mu S^\mu=-\mathcal A\Gamma
=\kappa n\mu_*[v_D'(y)]^2\ge0
$$

이므로 국소 relaxation과 entropy 부호는 맞다. 다만 ordinary single-copy
ideal-fluid action만으로 비가역 Onsager law가 나오지는 않는다. Schwinger--Keldysh
또는 microscopic collision theory에서 같은 $v_D$, mobility와 heat stress를
유도하는 것이 `L-U2.1`이다.

동결면을 $\kappa(T_f)v_D''(q)\simeq H(T_f)$로 정의하고 total yield
$Y_X=n/s_{\rm ent}$를 별도 Boltzmann 식에서 얻으면

$$
\Omega_bh^2=
\frac{m_B\nu_b qY_Xs_0}{\rho_{c,0}/h^2},\qquad
\Omega_ch^2=
\frac{m_\chi\nu_\chi(1-q)Y_Xs_0}{\rho_{c,0}/h^2}.
$$

이는 올바른 abundance 식이지만 $Y_X,m_\chi,\nu_i$, entropy dilution을
고정하지 않는다. $q$를 critical-density fraction으로 직접 대입하지 않았다는
점과 absolute prediction이 아직 없다는 점을 동시에 유지해야 한다.

### 4.3 spinodal/transient 후보

$v_D''(x)=0$은

$$
x_{\rm sp}=\frac1D=0.3146872312990126
$$

를 주고 conditioned composition $m=Dq$와 곱하면

$$
mx_{\rm sp}=(Dq)\frac1D=q
$$

가 exact다. 그러나 이 항등식은 `order-parameter inflection = total matter
critical fraction`을 말하지 않는다. 공변 phase-transition action이
$x=x_{\rm sp}$에서 유일한 $\Sigma_*$를 정하는 것과, 그 면에서 독립 stress
방정식이 $\Omega_m=x_{\rm sp}$를 주는 것은 별도 lemma다. $x:=\Omega_m$로
정의해 항등식을 닫는 route는 기존 direct-density 반례를 이름만 바꿔
재도입하므로 P0다.

보조 artifact의 더 강한 multi-current 존재구성은 tilted free energy

$$
f_D(y,z)=v_D(y)-h(z)y
$$

에서 단순 $f_{yy}=0$가 아니라 stationary spinodal
$f_y=f_{yy}=0$을 요구한다. 유일한 내부 해는

$$
y_*=\frac1D,\qquad
h_* = D-1-\log D,\qquad
f_{yyy}(y_*)=-D^2.
$$

master energy가

$$
\rho=n\left[m_*+\mu_*f_D(y,z)\right]
$$

이고 cold, comoving, no-entrainment 두 current가 total stress를 포화하면
$e=\rho/n$와 stationary $e_y=0$에서

$$
\mu_M=e+(1-y)e_y=e,\qquad
\mu_X=e-ye_y=e.
$$

따라서 partial chemical energy가 같아져

$$
\frac{\rho_M}{\rho_M+\rho_X}
=\frac{n_M\mu_M}{n_M\mu_M+n_X\mu_X}=y_*=\frac1D
$$

가 조건부로 나온다. 이는 $y:=\Omega_m$라는 정의보다 강하다. 그러나 별도
radiation, heat, wall, vacuum 또는 gradient energy가 분모에 남으면
$\Omega_m\ne y$다. complement current를 실제 dark-energy stress로 바꾸는
decay/action도 아직 없다.

relaxation rate는

$$
\tau_{\rm rel}^{-1}=\kappa f_{yy}
$$

이므로 spinodal에서 0으로 간다. finite cooling은 critical slowing 때문에
exact $y_f=1/D$를 자동 보장하지 않는다. quasistatic singular limit,
constraint/freeze field 또는 계산된 Kibble--Zurek offset 중 하나가 필요하다.

두 density-spectator 대안에는 exactness--uniqueness tradeoff가 있다.

1. $y_L=\rho_m/(\rho_m+\rho_{\rm DE})$는 dust+constant-vacuum에서
   $y_L'=-3y_L(1-y_L)<0$라 $1/D$를 유일하게 지난다. full flat fraction은

   $$
   \Omega_m=\frac{1-\Omega_r}{D},\qquad
   \Omega_b=q(1-\Omega_r)
   $$

   이므로 radiation correction이 남는다.
2. full-density trigger

   $$
   \mathcal F=(D-1)\rho_m-\rho_{\rm DE}-\rho_r=0
   $$

   은 radiation을 포함해 $\Omega_m=1/D$, $\Omega_b=q$를 exact하게 주지만
   $\rho_m=Aa^{-3}$, $\rho_r=Ba^{-4}$, $\rho_{\rm DE}=C$이면
   $\mathcal F$가 early/late에 모두 음수이고 중간에 maximum 하나를 가져
   일반적으로 root가 0개 또는 2개다. late root를 고르려면 oriented
   clock, hysteresis 또는 memory field가 필요하다.

두 spectator 모두 목표 density 식을 coupling에 재배열한 target-engineered
존재구성이다. 또한

$$
H_\Lambda^2=\frac{8\pi G}{3}\rho_{\rm DE},\qquad
H_*^2=\frac{8\pi G}{3}
\left(\rho_{m,*}+\rho_{{\rm DE},*}+\rho_{r,*}\right),\qquad
H_0=H(a=1)
$$

는 서로 다른 정의다. crossing을 현재로 고정하는 별도 clock과 absolute
density normalization이 없으므로 일반적으로 $H_\Lambda\ne H_*\ne H_0$이며,
두 spectator 모두 absolute scale와 observer epoch를 예측하지 않는다.

## 5. U3: dark EOM, stress, Jacobian과 transient no-go

### 5.1 D1 interacting vacuum

$$
T_c^{\mu\nu}=\rho_cu_c^\mu u_c^\nu,\qquad
T_{\rm de}^{\mu\nu}=-\rho_{\rm de}g^{\mu\nu},
$$

$$
\nabla_\mu T_c^{\mu\nu}=Q^\nu,\qquad
\nabla_\mu T_{\rm de}^{\mu\nu}=-Q^\nu,\qquad
Q^\nu=\xi\Theta\rho_{\rm de}u^\nu
$$

이면 total stress는 보존된다. FLRW에서

$$
\rho_c'+3\rho_c=3\xi\rho_{\rm de},\qquad
\rho_{\rm de}'=-3\xi\rho_{\rm de},
$$

$$
r'=3[\xi-(1-\xi)r],\qquad
r_*=\frac\xi{1-\xi},\qquad
\lambda_r=-3(1-\xi).
$$

$R_D=\alpha_sD=0.3746259405358018$를 먼저 택하고
$\xi=R_D/(1+R_D)$로 정의하면

$$
\xi=0.2725293692550135,\quad
w_{\rm eff,D}=-(1-\xi)=-0.7274706307449865,
$$

$$
\rho_D\propto a^{-3\xi}=a^{-0.8175881078\ldots}.
$$

fixed point와 음의 eigenvalue는 exact 산출이다. 하지만 $\xi(D)$는 아직
model axiom이고, perturbation에서는 어느 $u^\mu$가 momentum transfer를
정하는지와 $\delta Q^\mu$를 고정해야 한다.

### 5.2 D2 conformal scalar

$$
S=\int\sqrt{-g}\left[\frac{M_{\rm Pl}^2}2R-
\frac12(\nabla\phi)^2-V_0e^{-\lambda\phi/M_{\rm Pl}}\right]
+S_c[e^{2\beta\phi/M_{\rm Pl}}g,\psi_c]+S_b[g,\psi_b]
$$

에서

$$
\dot\rho_c+3H\rho_c=\frac{\beta\dot\phi}{M_{\rm Pl}}\rho_c,
\qquad
\ddot\phi+3H\dot\phi+V_{,\phi}=-\frac\beta{M_{\rm Pl}}\rho_c
$$

이고 total stress는 보존된다. dark-only 변수
$x=\dot\phi/(\sqrt6HM_{\rm Pl})$, $z=\sqrt V/(\sqrt3HM_{\rm Pl})$의
scaling point는 $L=\lambda+\beta$에 대해

$$
x_*=\frac{\sqrt{3/2}}L,\quad
z_*^2=\frac\beta L+\frac{3}{2L^2},\quad
\Omega_{\phi,*}=\frac\beta L+\frac3{L^2},\quad
w_{\rm eff,*}=-\frac\beta L.
$$

seen values $R_D$와 $-w_{\rm eff}=0.69$를 두 조건으로 역산하면

$$
\lambda=2.7738107015730,\qquad \beta=6.1739657551142.
$$

독립 finite-difference Jacobian은

$$
J=\begin{pmatrix}
-0.4087940539&18.1058516048\\
-2.5142942546&-2.1262059462
\end{pmatrix},
$$

$$
\operatorname{eig}(J)=-1.2675000001\pm6.6922389947i
$$

를 준다. dark-only 국소 안정성은 맞지만 두 연속 coupling을 두 target에서
정한 target-aware 존재점이다. $V_0$, 초기 trajectory, fifth force, CDM mass
drift와 full perturbation을 독립적으로 고정하지 않으면 예측이 아니다.

### 5.3 conserved-baryon no-go

fixed-mass conserved baryon dust에는 $\rho_b'=-3\rho_b$이고
$H'/H=-3(1+w_{\rm tot})/2$이므로

$$
\frac{d\log\Omega_b}{d\log a}
=-3-2\frac{H'}H=3w_{\rm tot}.
$$

따라서 열린 가속 구간 $w_{\rm tot}<-1/3$에서 nonzero $\Omega_b$는 상수가
될 수 없다. D1 fixed point에서 baryon transverse eigenvalue는
$3w_{\rm eff}=-2.1824118922<0$이라 baryon fraction은 0으로 간다. 이 반례는
세 성분이 모두 nonzero 상수인 **영구** 가속 attractor만 죽인다. action-defined
transition에서 interaction이 꺼지고 이후 보존 fluid를 전방 적분하는 transient
route는 살아 있다.

### 5.4 contraction readout과 3-layer 결합

historical 3-layer 경험식은

$$
c_\Sigma=1.014734427109503,\qquad
R_3=0.3806266173406572
$$

를 준다. $m=Dq$를 matter 내부 baryon fraction, $R_3=\Omega_c/\Omega_{\rm de}$를
동일 면의 dark ratio로 택하고 flatness를 쓰면

$$
\Omega_m=\frac{R_3}{R_3+1-m},\quad
\Omega_b=m\Omega_m,\quad
\Omega_c=(1-m)\Omega_m
$$

이고

$$
(\Omega_b,\Omega_c,\Omega_{\rm de})
=(0.0479920471424,0.2624602207623,0.6895477320953).
$$

대수 closure는 $10^{-15}$보다 정확하지만 `aggregate detector`, historical
$R_3$, flat transient 면의 세 선택을 결합한 target-aware `[경험식]`이다.
action/EOM/current 없이 수치 근접만으로 승격할 수 없다.

## 6. U4: exact FLRW와 기존 코드 반례

physical density를 고정하고 후보 $h$를 바꿀 때

$$
E^2(a;h)=\frac{\omega_ra^{-4}+\omega_ma^{-3}}{h^2}
+\left(1-\frac{\omega_r+\omega_m}{h^2}\right)F_{\rm de}(a),
$$

$$
F_{\rm de}(a)=a^{-3(1+w_0+w_a)}e^{3w_a(a-1)}.
$$

massless-radiation $\Lambda$CDM에서는

$$
\frac RH^2=6\left(2+\frac{d\log H}{d\log a}\right)
=3\Omega_m+12\Omega_\Lambda
=12-9\Omega_m-12\Omega_r.
$$

radiation, matter, de Sitter 극한은 각각 $0,3,12$다. running
$M=(1-\epsilon)/2$, $L=(1+\epsilon)/2$ 배경에는 추가로

$$
\frac RH^2=12-9\frac{Ma^{-3}}{E^2}-12\frac{ra^{-4}}{E^2}
+\frac{3\epsilon'}{2E^2}(1-a^{-3})
$$

가 필요하다. 또한 $M'a^{-3}+L'=\epsilon'(1-a^{-3})/2$를 보상할
flow stress/current가 없으면 total continuity가 깨진다.

성장식은

$$
D''+\left(2+\frac{d\log H}{dN}\right)D'
-\frac32\mu(a,k)\Omega_m(a)D=0
$$

이고 arbitrary grid에서는 각 interval의 $\Delta N_i$를 써야 한다.

| 기존 route 반례 | 독립 값 | 무너지는 범위 |
|---|---:|---|
| radiation example의 old $R/H^2$ | $11.999999999991$ | \`hubble_tension.py\` Ricci/readout |
| 같은 점의 exact $R/H^2$ | $2.9985\times10^{-12}$ | `hubble_tension.py` Ricci/readout |
| log-grid $n=2001$, $a=0.0998849$ cumulative $S$ 상대오차 | $+0.3363065707$ | `compute_s_of_a`와 그 의존 branch |
| even grid에서 $\int_0^3x^2dx$ old 결과 | $2.6666666667$ (exact $9$) | 마지막 interval drop 동작 |
| warped nonuniform EdS growth 최대 상대오차 | $2.7897274907$ | arbitrary-grid public API |

replacement scratch는 nonuniform cubic quadrature 오차 $3\times10^{-15}$,
EdS grid refinement 오차

$$
7.554\times10^{-4},\quad5.578\times10^{-5},\quad3.791\times10^{-6}
$$

와 matter+$\Lambda$ Heath quadrature 교차검증을 통과했다. 이는 교정 수학의
증인이지 product 코드가 이미 교체됐다는 뜻은 아니다.

## 7. U5: acoustic readout과 식별성

$$
R_b(a)=\frac{3\omega_b}{4\omega_\gamma}a,\qquad
c_s(a)=\frac{c}{\sqrt{3(1+R_b(a))}},
$$

$$
r_s(z)=\int_0^{a(z)}\frac{c_s(a)}{a^2H(a)}da,\qquad
\theta_*=\frac{r_s(z_*)}{D_M(z_*)},\qquad r_d=r_s(z_d).
$$

$z_*$와 $z_d$는 서로 다른 visibility/drag 조건이다. 실제 readout에는
$\omega_b,\omega_c,T_{\rm CMB},N_{\rm eff},\sum m_\nu,Y_p$, recombination,
primordial spectrum과 nuisance가 모두 활성이어야 한다.

기존 `lcdm_theta_star_for_h`는 `om_b_h2=0.001`과 $0.1$에서 모두

$$
\theta_*=0.011923371862127247
$$

를 bit-identical하게 반환한다. 따라서 그 함수의 baryon-aware claim과 이를
사용한 $H_0$ 수치는 P0다.

새 compressed scratch는 same-grid injection에서 $h=0.68$을 복원한다. 그러나
2600-interval target을 서로 다른 grid로 역산하면

| intervals | recovered $h$ | $h-0.68$ |
|---:|---:|---:|
| 200 | $0.6799375033$ | $-6.250\times10^{-5}$ |
| 400 | $0.6799897376$ | $-1.026\times10^{-5}$ |
| 650 | $0.6799983103$ | $-1.690\times10^{-6}$ |
| 1300 | $0.6799998962$ | $-1.038\times10^{-7}$ |

로 수렴한다. same-grid $2\times10^{-8}$ 복원은 solver consistency이고 독립
accuracy가 아니다. 실제 $z_*$와 full solver calibration 전에는 conditional
compressed inference다.

late inverse ladder에서는

$$
\frac{D_M}{r_d}=\frac{c}{H_0r_d}\int\frac{dz}{E(z)},\qquad
\frac{D_H}{r_d}=\frac{c}{H_0r_dE(z)}.
$$

따라서 independent $r_d$ 또는 absolute calibration이 없으면 $H_0r_d$만
식별된다. 이 null test는 Route C의 정확한 kill condition이다.

## 8. U6: perturbation action, amplitude와 horizon 반례

### 8.1 필요한 Mukhanov--Sasaki 사슬

canonical single-field의 scalar quadratic action은

$$
S_2=\frac12\int d\tau d^3x\,z^2
\left[({\cal R}')^2-(\nabla{\cal R})^2\right],
\qquad z=\frac{a\dot\phi}{H},
$$

$$
v_k''+\left(k^2-\frac{z''}{z}\right)v_k=0,\qquad
{\cal P}_{\cal R}=\frac{k^3}{2\pi^2}\left|\frac{v_k}{z}\right|^2.
$$

Bunch--Davies 초기상태, horizon crossing, tensor mode와 reheating map을 함께
고정해야 $A_s,n_s,r$이 같은 action의 산출이 된다. 현재 artifact는 이 exact
mode integration을 구현하지 않고 first-order potential slow roll을 검산했다.

Starobinsky action

$$
S=\frac{M_{\rm Pl}^2}{2}\int\sqrt{-g}
\left(R+\frac{R^2}{6M^2}\right)
$$

아래 $N_*=55$에서

$$
n_s=0.964977222226,\quad r=0.00349829929427,\quad
\alpha_s^{\rm run}\simeq-0.000620899527,\quad
M/M_{\rm Pl}=1.21934234122\times10^{-5}.
$$

마지막 scale은 $A_s=2.099\times10^{-9}$를 외부 normalization으로 넣은 값이다.
$N_*$는 reheating history에 의존한다. 따라서 $n_s,r$, running은 conditional
cross-output이고 $A_s$ 및 $M$은 아직 CE 무입력 예측이 아니다.

### 8.2 phase-flow 존재구성과 non-uniqueness

$$
I_s=\int_0^{N_e}dN\,\frac K2(s'-\kappa)^2
$$

에 $s(0)=s_0$와 natural endpoint를 주면 $s'=\kappa$다. 그러므로
$\kappa=\pi^2/2$, $s_0=-\pi\delta(1-q)$를 **선택하면** 기존 law를 재현한다.
이는 positive quadratic action의 존재구성이지만 원하는 $s(N)$마다
$(s'-F)^2$류 action을 만들 수 있어 미시 유일성은 없다.

boundary correction을 넣은 값은

$$
\log S=281.737688630,\qquad H_{\rm phase}=67.2483459200\ {\mathrm{km\,s^{-1}\,Mpc^{-1}}}
$$

이고 correction은 $H$를 factor $1.3042641743$만큼 올린다. 한 cell 변화는
$H$를 factor $e^{-\pi^2/4}=0.0848049725$만큼 바꾸므로 integer/boundary 선택이
수치를 지배한다.

### 8.3 de Sitter entropy와 현재 Hubble entropy의 P0 경계

현재 apparent-Hubble-radius quantity를

$$
S_A(t_0)=\pi\frac{M_P^2}{H_0^2}
$$

로 **정의**하면 phase 값에서 $H_0=67.248\ldots$가 조건부로 나온다. 그러나
왜 $t_0$의 apparent horizon을 고르는지는 새 경계공리이고
$\rho_\Lambda=\Omega_\Lambda\rho_{\rm crit}$에는 외부 $\Omega_\Lambda$가 다시
들어간다.

반대로 진짜 asymptotic de Sitter entropy라면

$$
S_{\rm dS}=\pi\frac{M_P^2}{H_\Lambda^2},\qquad
H_\Lambda^2=\Omega_\Lambda H_0^2.
$$

따라서 같은 phase 값이 고정하는 것은 $H_\Lambda$이고

$$
H_0=\frac{H_{\rm phase}}{\sqrt{\Omega_\Lambda}}.
$$

$\Omega_\Lambda=0.6891$이면

$$
H_0=81.0103449949\ {\mathrm{km\,s^{-1}\,Mpc^{-1}}},
$$

즉 $67.248\ldots$와 $13.7620$ 차이다. 그러므로 `de Sitter entropy`와
`현재 H0 readout`을 무표시로 동일시하는 route claim은 P0다. 목표인
independent vacuum scale은 삭제하지 않고 apparent-horizon, true-de-Sitter,
four-form/RG/stochastic 후보로 분기한다.

### 8.4 다른 absolute-scale 후보의 정확한 공백

- RG transmutation의
  $\Lambda_{\rm RG}=\mu\exp[-8\pi^2/(bg^2(\mu))]$는 차원 scale을 만들지만
  CE field content, $b$, thresholds가 없다.
- four-form action은 $F_4=c\epsilon_4$와
  $\rho_\Lambda=\Lambda_{\rm bare}+c^2/2$를 만들지만 flux quantum과 branch를
  고르지 않는다. 현재 적힌 action에는 radiative sequestering global constraint도
  아직 없다.
- stochastic $\Delta\Lambda\sim V_4^{-1/2}\sim H^2$는 order relation이며
  coefficient, sign, correlation time과 covariant measure가 없다.

세 route 모두 구조적으로 유효한 연구 후보지만 현재 `[미완성]`이다.

## 9. 자유도, target-awareness와 look-elsewhere

| 경로 | 최소 연속 자유도/초기자료 | discrete/look-elsewhere | target-aware | 독립 kill output |
|---|---|---|---|---|
| GW aggregate composition | tree density, relic mass spectrum, reservoir, freeze clock | detector 최소 4종, root/descendant 역할 2종 | 예 | multiplicity cumulant, isocurvature, relic/entropy |
| reacting two-current | $m_X,\mu_*,\kappa(T),Y_X,m_\chi,T_R$ 등 6개 이상 | decay/charge assignment | 예 | BBN, entropy, mass ratio, isocurvature |
| asymmetric transfer | source yield, portal, charge, $m_\chi$, reheating 등 | charge matrix/stoichiometry | 예 | symmetric remnant, collider/decay, isocurvature |
| D1 interacting vacuum | $\xi$와 interaction switch 최소 2개 | $Q^\mu$ frame 선택 | 예 | $H(z),f\sigma_8$, early DE |
| D2 conformal scalar | $\lambda,\beta,V_0,m_{c0}$와 초기장 최소 6개 | potential/coupling family | 예; 2 target으로 $\lambda,\beta$ 역산 | fifth force, mass drift, full perturbation |
| 3-layer+$Dq$ triplet | detector, historical $R$, $\Sigma_*$ | 적어도 3개의 구조 선택 | 예 | redshift evolution, growth, species map |
| full CMB | $h,\omega_b,\omega_c,A_s,n_s,\tau,\sum m_\nu,N_{\rm eff},Y_p,w_0,w_a$ + nuisance | solver/likelihood release | 입력값을 사전 동결하면 아니오 | TT/TE/EE, lensing, BAO, BBN |
| compressed $\theta_*$ | $h$ + 미고정 early/late nuisance | fit/backend 선택 | 현재 합성은 아니오, 관측 적용은 관리 필요 | full-solver held-out grid |
| inverse ladder | $H_0r_d,\Omega_m,w_0,w_a,{\cal M}_{SN}$ + nuisance | survey/split | data를 본 뒤 선택하면 예 | calibration-null, survey split |
| Starobinsky | $M$, reheating history | inflation family 선택 | 예 | $n_s,r$, running, reheating consistency |
| phase flow | $K,\kappa,s_0$의 구조 선택과 integer convention | boundary sign/cell count/horizon 2종 이상 | 예 | horizon convention, micro transfer spectrum |
| RG | UV coupling, thresholds | field content/$b$ | 필요한 exponent 역산 시 예 | running, thresholds, relics |
| four-form | bare vacuum, flux unit, integer branch/history | 많은 branches | 가까운 branch 선택 시 예 | spacing, transitions, metastability |

숫자와 가까운 후보 하나를 선택해 다른 후보를 숨기면 유효 dof가 0이 되지 않는다.
위 선택 수는 보수적 하한이다.

## 10. P0/P1/P2 원장

### P0 — 정확한 route 범위

1. `q`를 정리만으로 오늘의 $\Omega_b$라고 부르는 direct route. 같은 $q$에서
   stress/weight/normalization을 바꿀 수 있다. 목표 abundance는 유지한다.
2. conserved baryon이 nonzero 상수인 가속 three-component fixed point.
   $d\log\Omega_b/d\log a=3w_{\rm tot}$가 완전 반례다. transient route는 유지한다.
3. legacy Ricci route. radiation 예에서 오차가 거의 12이고 running $\epsilon'$도
   누락됐다.
4. legacy nonuniform Simpson과 arbitrary-grid growth route. 각각 $+33.63\%$와
   최대 $278.97\%$ 반례가 있다.
5. legacy baryon-aware $H_0$ route. $\omega_b$를 $100$배 바꿔도 output이
   bit-identical하다.
6. phase entropy를 true de Sitter라고 부르면서 그 $H$를 현재 $H_0$로 읽는
   route. $\Omega_\Lambda=0.6891$에서 $67.248\to81.010$이다.
7. 같은 $H$--entropy--Friedmann identity에서 $H_0$와
   $\rho_\Lambda^{1/4}$를 독립 예측 두 개로 세는 route.
8. 식별 불가 hybrid posterior와 이미 본 data를 blind holdout으로 부르는 route.

### P1 — 최소 missing lemma

1. local branching event를 equal-energy conserved relic/current로 바꾸는
   covariant cascade action과 total-yield normalization.
2. $v_D$ affinity/mobility/heat를 microscopic collision 또는 SK action에서
   유도하고 유일한 freeze-out surface를 만드는 정리.
3. $R_D(D)$ 또는 $\xi(D),\lambda(D),\beta(D)$를 UV field content에서 정하고
   background+perturbation을 동시에 안정화하는 정리.
4. spinodal order parameter와 $\Omega_m$을 정의가 아니라 metric variation과
   transition stress로 연결하는 정리.
5. common radiation/neutrino background의 product 통합, actual recombination,
   CLASS/CAMB와 likelihood adapter.
6. full Mukhanov--Sasaki mode evolution, reheating, $A_s$ scale generator.
7. phase/RG/four-form/stochastic 후보의 microscopic coefficient/branch와
   radiative stability.
8. immutable official observation manifest, SPD/covariance와 독립 holdout.

### P2 — 원장/표기

1. $q$를 survival이라고 부르는 legacy 문구.
2. exact $q$, rounded $q$, runtime `0.0487`과 관측 snapshot의 무표시 혼용.
3. raw triplet과 flat-normalized background의 무표시 혼용.
4. $S_A(t_0)$와 $S_{\rm dS}$, $H_0$와 $H_\Lambda$의 horizon label 혼용.

## 11. 재현

```powershell
python _workspace/ce/cosmology-full-closure-unification-20260815/artifacts/verify_full_cosmology_math.py

python _workspace/ce/cosmology-full-closure-unification-20260815/artifacts/verify_density_dark_routes.py

python _workspace/ce/cosmology-full-closure-unification-20260815/artifacts/verify_background_h0_forward.py

python _workspace/ce/cosmology-full-closure-unification-20260815/artifacts/verify_primordial_entropy_routes.py

python _workspace/ce/cosmology-full-closure-unification-20260815/artifacts/verify_transient_transition_action.py
```

통합 verifier는 exact fixed-point residual $10^{-60}$ 이하, closed algebra
$10^{-12}$ 이하, ODE/quadrature grid refinement를 강제한다. 출력 마지막 줄의
`PASSED`는 algebra와 반례의 재현 성공이며 physical bridge의 PASS가 아니다.

## 12. 종료 판정

- U1 exact/legacy/rounded/observation 구분: 수학·inventory 완료, migration 미완성.
- U2 conditioned theorem과 reacting EFT algebra: 완료, species/current/yield bridge 미완성.
- U3 D1/D2 background와 local Jacobian: 완료, UV coupling·perturbation·transition 미완성.
- U4 교정식과 반례: 완료, product 통합 미완성.
- U5 identifiability와 세 forward route의 실패조건: 완료, physical inference 미완성.
- U6 slow-roll/phase/scale 후보와 horizon P0 경계: 완료, absolute scale과 exact MS 미완성.
- U7 provenance 수학과 prior no-go: 완료, immutable manifest와 blind holdout 미완성.

따라서 이 파일의 `Status: COMPLETE`는 **요청된 수학 검산 레인이 모든 U1--U7
주장에 지위와 반례 범위를 부여했다**는 뜻이다. “모든 물리 목표가 달성됐다”는
승격 선언이 아니다.
