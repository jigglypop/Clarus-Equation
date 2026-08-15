Status: COMPLETE

# 10 — 우주론 전면 닫힘 공식 출처·provenance 레인

- Lane: `ce-physics-sourcer`
- 웹 접근일 / snapshot: **2026-08-16 (Asia/Seoul)**
- 제품 소스 수정: **없음**
- 선행 관측 감사: [`cosmology-theory-repository-audit-20260815/10-sources.md`](../cosmology-theory-repository-audit-20260815/10-sources.md)
- 판정 경계: 여기서 `VERIFIED`는 논문·공식 배포처의 존재, 인용 내용, 값과 provenance가 확인됐다는 뜻이다. 그것만으로 CE의 수학적 bridge, action 또는 예측이 증명됐다는 뜻은 아니다.
- 불확도 표기: 관측 수치에는 68% 주변화 오차를 기본으로 적었다. 수치해석 정확도·정리·작용에는 관측 불확도와 공분산이 적용되지 않으므로 `N/A`로 적었다. 공개 chain을 이 레인에서 실제 적재하지 않은 경우 `full covariance 미적재`라고 명시했다.

## 1. 먼저 보존해야 하는 P0와 blind 경계

| Evidence ID | 선행 Evidence 연결 | 현재 확인 | 불확도·공분산 | 수치 영향 | 상태 |
|---|---|---|---|---|---|
| **SRC-FULL-P0-01** | 선행 `SRC-P0-01`, `SRC-DESI-II` | 저장소의 `DESI DR2+CMB` $\Omega_\Lambda=0.693\pm0.005$는 공식 DR2 flat $\Lambda$CDM posterior로 식별되지 않는다. 공식 값은 $\Omega_m=0.3027\pm0.0036$, 따라서 평탄성 아래 $\Omega_\Lambda=0.6973\pm0.0036$이다. | DESI 공개 chain; 같은 표의 $\Omega_m$–$H_0$ 상관계수는 $-0.975$. | CE $0.6891$ 잔차는 저장소 표기의 $-0.78\sigma$가 아니라 $-2.28\sigma$. | **VERIFIED P0, 그대로 보존** |
| **SRC-FULL-P0-02** | 선행 `SRC-P0-02`, `SRC-ACT`, `SRC-SPT`, `SRC-SPT-LAMBDA` | `Planck_ACT_SPT_combined=(0.02228,0.1195,68.43)`은 하나의 공식 likelihood/chain이 아니라 서로 다른 조합의 성분을 섞은 tuple이다. | 단일 covariance가 존재하지 않는다. ACT·SPT 각각의 공개 chain만 존재한다. | 선행 감사의 현재 corrected CMB-SPA 대비 $H_0$ 차이 $3.26\sigma$를 보존한다. | **VERIFIED P0; hybrid 사용 금지** |
| **SRC-FULL-HOLD-01** | 선행 `SRC-DESI-IV`; `experiments/preregistration/cosmology_future_holdout_v2.json` | DESI Results IV의 v1은 2026-07-29, 공식 공개는 2026-07-30으로 freeze 2026-08-02 이전이다. v3가 2026-08-04에 올라왔어도 같은 DR2 자료의 수정판이므로 새 blind holdout이 아니다. manifest도 `assignment_status=unassigned`이다. | Results IV의 joint likelihood/covariance는 기존 DR2 13점 BAO covariance의 drop-in 교체가 아니다. | 독립 holdout 수는 계속 **0**. | **VERIFIED manifest/날짜; qualifying release 미발견** |

위 세 항목은 새 이론 경로가 성공하더라도 완화할 수 없다. P0 수치를 다른 이름으로 재도입하거나, 이미 본 자료의 새 버전을 future holdout으로 세면 U7은 닫히지 않는다.

## 2. U2 — 확률에서 reacting current와 absolute abundance까지

### 2.1 conditioned Galton–Watson가 실제로 주는 것

| Evidence ID | 1차 출처가 확립하는 범위 | CE에 아직 주지 않는 것 | 불확도·공분산 / 수치 영향 | 판정 |
|---|---|---|---|---|
| **SRC-U2-BR-01** | Jagers–Lagerås는 supercritical branching process를 extinction에 조건부로 두면 다시 branching process이며 subcritical이 됨을 보인다. one-type Galton–Watson의 extinction 확률을 $q$라 하면 조건부 offspring PGF는 $f_q(s)=f(qs)/q$로 읽을 수 있다. [DOI 10.1214/ECP.v13-1419](https://doi.org/10.1214/ECP.v13-1419) | offspring label을 baryon species로 식별하는 법, equal-energy weighting, 공변 current, freeze-out hypersurface, entropy-normalized total yield, $\Omega_m$ 또는 critical-density normalization은 정리가 공급하지 않는다. | 확률 정리라 관측 오차·공분산 `N/A`. 기존 scalar 값을 composition probability로 보존하는 데는 쓸 수 있으나 $q=\Omega_b$ 직접 동일시는 **0개 bridge**를 추가한다. | **정리 provenance VERIFIED; CE density bridge UNVERIFIED** |

### 2.2 perfect fluid와 reacting/open mixture의 경계

| Evidence ID | 1차 출처 | 확인된 내용 | CE 구현에 필요한 추가 입력 | 판정 |
|---|---|---|---|---|
| **SRC-U2-FL-01** | Taub [DOI 10.1103/PhysRev.94.1468](https://doi.org/10.1103/PhysRev.94.1468); Schutz [DOI 10.1103/PhysRevD.2.2762](https://doi.org/10.1103/PhysRevD.2.2762); Brown [arXiv:gr-qc/9304026](https://arxiv.org/abs/gr-qc/9304026), [DOI 10.1088/0264-9381/10/8/017](https://doi.org/10.1088/0264-9381/10/8/017) | 고전 perfect-fluid variational principle, velocity potentials, stress tensor와 보존 current를 제공한다. Taub의 변분은 질량 보존을 명시적으로 유지하고 Brown action도 particle-number conservation을 내장한다. | $\nabla_\mu J_i^\mu\ne0$인 species conversion을 원하면 reaction/source 변수를 action 또는 명시적 effective-fluid closure에 새로 넣어야 한다. 보존 perfect-fluid action을 그대로 두고 branching 숫자만 대입해서는 생성 current가 되지 않는다. | **출처 VERIFIED; reacting CE action 미제시** |
| **SRC-U2-FL-02** | Andersson–Comer [arXiv:1306.3345](https://arxiv.org/abs/1306.3345), [DOI 10.1088/0264-9381/32/7/075008](https://doi.org/10.1088/0264-9381/32/7/075008); Kikuchi–Tsumura–Kunihiro [DOI 10.1103/PhysRevC.92.064909](https://doi.org/10.1103/PhysRevC.92.064909); Gavassino–Antonelli–Haskell [arXiv:2003.04609](https://arxiv.org/abs/2003.04609), [DOI 10.1088/1361-6382/abe588](https://doi.org/10.1088/1361-6382/abe588) | dissipative multifluid action, relativistic Boltzmann에서 유도한 reactive multicomponent hydrodynamics, chemical reaction-coordinate/bulk-viscous closure라는 서로 독립적인 reacting 경로가 존재한다. entropy production과 Onsager 조건도 검사 대상이 된다. | species별 반응률, stoichiometry, 보존 charge, collision kernel, transport/relaxation 계수와 초기분포를 지정해야 한다. 이 입력 없이 `branch probability -> abundance`는 닫히지 않는다. | **대체 경로 provenance VERIFIED; CE microphysics UNVERIFIED** |
| **SRC-U2-FL-03** | Kim, [arXiv:2209.11374v3](https://arxiv.org/abs/2209.11374) | pull-back variational fluid에서 particle creation을 clock 또는 interacting matter-space로 다루는 최신 후보이며, homogeneous FLRW에서 같은 생성사를 주는 clock들이 선형 섭동에서는 달라질 수 있음을 명시한다. | journal peer review가 확인되지 않은 preprint다. 단일-flow creation clock을 채택하려면 scalar mode, gauge-invariant perturbation과 sound-speed closure를 별도로 검증해야 한다. | **출처 존재 VERIFIED; 정본 후보 지위 UNVERIFIED** |

위 항목들은 관측 수치가 아니므로 uncertainty/covariance는 `N/A`다. 수치적 영향은 “반응률과 entropy normalization 없이 absolute $\Omega_b$를 계산할 수 있는 독립 식의 수 = 0”이라는 closure 영향이다.

### 2.3 Sakharov / freeze-out / Boltzmann abundance

| Evidence ID | 공식·1차 출처 요구사항 | CE에 대한 provenance 판정 | 불확도·공분산 / 수치 영향 | 상태 |
|---|---|---|---|---|
| **SRC-U2-AB-01** | Sakharov 원 논문/영문 재수록 [DOI 10.1070/PU1991v034n05ABEH002497](https://doi.org/10.1070/PU1991v034n05ABEH002497)은 baryon-number violation, C/CP 비대칭과 비평형 우주 진화를 요구하는 출발점이다. | composition 확률만으로 net baryon number가 생긴다는 근거가 아니다. baryogenesis를 주장하면 최소한 그 세 mechanism의 action/rate가 필요하다. | 관측 오차·공분산 `N/A`; 누락 mechanism 수가 그대로 남는다. | **VERIFIED source boundary** |
| **SRC-U2-AB-02** | Gondolo–Gelmini [DOI 10.1016/0550-3213(91)90438-4](https://doi.org/10.1016/0550-3213(91)90438-4)와 공식 [PDG 2025 Dark Matter review](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-dark-matter.pdf)는 relic abundance가 Boltzmann Liouville/collision operator, thermal reaction rate, $H(T)$, equilibrium density, entropy density와 $g_*(T)$ 및 freeze-out 뒤 entropy history에 의존함을 고정한다. | CE가 산출해야 할 최소 묶음은 species masses/energy weights, collision/source term, CP/B violation 여부, background $H(T)$, freeze-out surface, $Y_i=n_i/s$ 초기·최종값과 오늘의 critical-density 변환이다. | 이 입력의 공개 covariance는 해당 없음. 입력이 없으면 branching scalar가 맞아도 absolute abundance와 $\Omega_b$의 수치 영향은 **UNDETERMINED**. | **요건 VERIFIED; absolute yield UNVERIFIED** |

## 3. U3 — interacting dark sector의 action·scaling·perturbation

| Evidence ID | 1차 출처가 제공하는 대체 경로 | 반드시 고정할 항목 | 수치/공분산 영향 | 판정 |
|---|---|---|---|---|
| **SRC-U3-ACT-01** | Amendola의 coupled quintessence [arXiv:astro-ph/9908023](https://arxiv.org/abs/astro-ph/9908023), [DOI 10.1103/PhysRevD.62.043511](https://doi.org/10.1103/PhysRevD.62.043511); Pourtsidou–Skordis–Copeland의 pull-back fluid action [arXiv:1307.0458](https://arxiv.org/abs/1307.0458), [DOI 10.1103/PhysRevD.88.083505](https://doi.org/10.1103/PhysRevD.88.083505) | covariant action, field content, potential/coupling function을 택하면 background와 linear perturbation을 같은 모형에서 계산할 수 있다. | 정적 $\Omega_c/\Omega_\Lambda$ 또는 background $Q=3H\xi\rho_{DE}$만으로는 action이 유일하게 정해지지 않는다. $Q^\mu$, energy-flow 부호, momentum-transfer frame, scalar sound speed와 initial conditions를 고정해야 한다. | **action 경로 VERIFIED; CE action 선택 UNVERIFIED** |
| **SRC-U3-PERT-01** | Valiviita–Majerotto–Maartens [arXiv:0804.0232](https://arxiv.org/abs/0804.0232), [DOI 10.1088/1475-7516/2008/07/020](https://doi.org/10.1088/1475-7516/2008/07/020); Clemson et al. [arXiv:1109.6234](https://arxiv.org/abs/1109.6234), [DOI 10.1103/PhysRevD.85.043007](https://doi.org/10.1103/PhysRevD.85.043007) | 단순 constant-$w$ interaction은 early super-Hubble instability가 날 수 있고, 같은 background energy transfer도 DM-frame/DE-frame momentum transfer에 따라 growth가 달라진다. background보다 perturbation/growth가 coupling에 더 민감할 수 있다. | 공개 관측 covariance를 인용하는 행이 아니라 이론 안정성 경계이므로 `N/A`. background fixed ratio 하나를 통과해도 perturbation gate 통과 수는 **0**이다. | **perturbation 필요성 VERIFIED** |

따라서 U3에서 허용되는 source-level 승격은 “후보 action/effective system이 존재한다”까지다. CE 비율을 attractor로 읽으려면 같은 sign convention에서 fixed point의 존재·안정성·양의 density, radiation/matter epoch와 $C_\ell/P(k)$를 함께 산출해야 한다.

예를 들어 $\dot\rho_c+3H\rho_c=+Q$, $\dot\rho_{DE}+3H(1+w)\rho_{DE}=-Q$, $Q=3H\xi\rho_{DE}$ convention을 먼저 고정하면 $r=\rho_c/\rho_{DE}$는

$$
\frac{dr}{d\ln a}=3\{wr+\xi(1+r)\},
\qquad
r_*=-\frac{\xi}{w+\xi}
$$

를 만족한다. 이 식은 위 continuity convention의 대수적 귀결일 뿐 action의 유일성이나 perturbation 안정성을 보장하지 않는다. $Q$의 부호 convention을 바꾸면 식도 함께 바뀌므로 정본에는 두 continuity equations와 $Q^\mu$를 먼저 적어야 한다.

## 4. U4 — background/growth의 독립 검산 anchor

| Evidence ID | 1차 출처 | 확인된 적용 범위 | 구현 영향 | 판정 |
|---|---|---|---|---|
| **SRC-U4-GROW-01** | Ma–Bertschinger [arXiv:astro-ph/9506072](https://arxiv.org/abs/astro-ph/9506072), [DOI 10.1086/176550](https://doi.org/10.1086/176550) | baryon, CDM, photon, massless/massive neutrino를 포함한 coupled Einstein–Boltzmann–fluid perturbation equations와 gauge/initial-condition 기준을 제공한다. radiation-era부터 growth를 검산할 수 있는 multi-species 기준이다. | 단일 pressureless growth 식으로 radiation·neutrino epoch까지 검증했다고 부를 수 없다. U4의 independent solver는 U5의 CLASS/CAMB cross-check와 연결해야 한다. | **VERIFIED reference system** |
| **SRC-U4-GROW-02** | Heath [DOI 10.1093/mnras/179.3.351](https://doi.org/10.1093/mnras/179.3.351) | zero-pressure Friedmann–Lemaître universe의 density-growth 분석은 matter/late-time quadrature 검산 anchor가 된다. | 제목과 가정대로 zero-pressure 범위다. radiation이 유의한 $a\to0$ 극한의 독립 검증으로 외삽하지 않는다. | **VERIFIED limited-domain cross-check** |

두 행은 이론 방정식 출처라 관측 uncertainty/covariance는 `N/A`다. U1의 symbol/constant ledger와 U8의 alias·CLI·fail-closed 통합은 저장소 내부 provenance 문제이므로 별도의 외부 수치를 요구하지 않는다.

## 5. U5 — recombination, sound horizon, exact likelihood와 inverse ladder

### 5.1 Boltzmann·recombination 구현 기준

| Evidence ID | 공식 코드/1차 출처 | 확인된 요건 | 정확도·공분산 / 수치 영향 | 상태 |
|---|---|---|---|---|
| **SRC-U5-BOLTZ-01** | CLASS I [arXiv:1104.2932](https://arxiv.org/abs/1104.2932), CLASS II [arXiv:1104.2933](https://arxiv.org/abs/1104.2933), 공식 [CLASS repository](https://github.com/lesgourg/class_public); CAMB [DOI 10.1086/309179](https://doi.org/10.1086/309179), 공식 [CAMB repository](https://github.com/cmbant/CAMB) | $\omega_b,\omega_c$, radiation/neutrino, primordial spectrum, recombination과 metric/matter perturbation을 실제로 진화시켜 $C_\ell$와 matter spectra를 만드는 forward solver가 필요하다. | 코드 정확도는 관측 covariance와 별개다. compressed 식 하나로 $H_0$를 읽는 경로는 full solver 대비 residual과 유효범위를 따로 고정해야 한다. | **solver provenance VERIFIED** |
| **SRC-U5-CAMB2-01** | Lewis, [CAMB v2 arXiv:2607.14854](https://arxiv.org/abs/2607.14854), 공식 repository 위와 같음 | 2026-07-16 공개된 현 CAMB 논문은 integrator, fast recombination calibration, PPF dark energy 안정화와 lensing을 갱신했고 기본값의 주요 lensed-CMB/quasi-linear 범위 pointwise convergence target을 $10^{-3}$로 보고한다. | $10^{-3}$은 수치 convergence 목표이며 관측 $1\sigma$나 covariance가 아니다. 구 CAMB 결과를 정본으로 둘 경우 버전 pin과 v2 교차가 새 요구사항이다. | **VERIFIED current implementation source** |
| **SRC-U5-REC-01** | RECFAST: Seager–Sasselov–Scott [arXiv:astro-ph/9909275](https://arxiv.org/abs/astro-ph/9909275), [DOI 10.1086/312250](https://doi.org/10.1086/312250), 공식 [RECFAST page](https://www.astro.ubc.ca/people/scott/recfast.html); HyRec [arXiv:1011.3758](https://arxiv.org/abs/1011.3758), [DOI 10.1103/PhysRevD.83.043513](https://doi.org/10.1103/PhysRevD.83.043513); HYREC-2 [arXiv:2007.14114](https://arxiv.org/abs/2007.14114) | RECFAST는 modified few-level approximation이고 HyRec 계열은 multilevel/radiative-transfer 물리를 더 직접 다룬다. HYREC-2는 Planck 허용영역 전체에서 original HyRec 정확도를 유지하고 ideal cosmic-variance-limited $\ell\le5000$에서도 눈에 띄는 parameter bias가 없다고 검증한다. | `99.7% Planck region`, `$\ell\le5000$`은 검증 범위이지 새 관측값이 아니다; covariance `N/A`. CE forward model은 선택한 recombination backend와 cross-code residual을 pin해야 한다. | **VERIFIED; backend 미지정 CE readout UNVERIFIED** |

### 5.2 exact CMB/H0 likelihood의 최소 입력

| Evidence ID | 1차·공식 출처 | source-level 결론 | 공분산 / 영향 | 상태 |
|---|---|---|---|---|
| **SRC-U5-LIKE-01** | Planck likelihood paper [arXiv:1907.12875](https://arxiv.org/abs/1907.12875), [A&A 641 A5](https://doi.org/10.1051/0004-6361/201936386); 공식 [PLA likelihood code](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_spectrum_%26_Likelihood_Code) | exact 비교는 TT/TE/EE theory spectra, low/high-$\ell$ likelihood, beam/calibration, foreground·nuisance parameters, lensing 및 likelihood covariance를 함께 사용한다. $\theta_*=r_*/D_M(z_*)$ 하나는 joint posterior를 대체하지 않는다. | Planck은 released likelihood/covariance를 제공한다. CE가 $H_0$만 출력하고 $C_\ell$, nuisance와 covariance를 생략하면 full-likelihood 수치 영향은 **계산 불가**다. | **요건 VERIFIED; compressed-only closure 불가** |

### 5.3 inverse distance ladder는 독립성 종류를 구분해야 한다

| Evidence ID | 결과와 원 출처 | uncertainty / covariance | CE 수치 영향과 독립성 | 판정 |
|---|---|---|---|---|
| **SRC-U5-IDL-01** | DES-SN5YR + DESI BAO, 4차 cosmography: $H_0=67.19^{+0.66}_{-0.64}\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$. Camilleri et al. [arXiv:2406.05049](https://arxiv.org/abs/2406.05049), [DOI 10.1093/mnras/staf122](https://doi.org/10.1093/mnras/staf122) | 논문은 SN·BAO covariance를 사용한다. 이 레인에는 joint chain을 적재하지 않았으므로 표의 값은 asymmetric marginal이며 cross-probe posterior covariance는 미적재다. | CE/저장소 기준 $73.04$와 중심 차이는 $5.85\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$; CE 예측 오차가 없어 combined $\sigma$는 계산하지 않았다. BAO ruler가 CMB-calibrated $r_d$에 의존하므로 CMB와 완전히 독립인 절대척도 검증은 아니다. | **값 VERIFIED; independent absolute-scale holdout 아님** |
| **SRC-U5-IDL-02** | uncalibrated DESI DR1 BAO와 CMB acoustic angle을 쓰는 sound-horizon-size-independent 경로는 외부 자료 조합별 $H_0=69.2^{+1.3}_{-1.4}$, $70.3^{+1.4}_{-1.2}$, $69.6^{+1.3}_{-1.8}\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$. Zaborowski et al. [arXiv:2510.19149](https://arxiv.org/abs/2510.19149), [DOI 10.1088/1475-7516/2026/04/004](https://doi.org/10.1088/1475-7516/2026/04/004) | 각 값은 주변화 오차다; 세 조합 사이 covariance는 이 레인에 미적재. | $r_s$ 크기에는 독립적이지만 $\theta_*$ 및 Planck/ACT lensing, DES 자료를 사용한다. CE $73.04$와 중심 차이는 각각 $3.84$, $2.74$, $3.44\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$이며, 서로를 세 개 독립 holdout으로 셀 수 없다. | **값 VERIFIED; input-overlap 명시 필요** |

## 6. U6 — primordial spectrum, RG scale, vacuum energy와 horizon

### 6.1 $R+R^2$, Mukhanov–Sasaki, vacuum과 reheating

| Evidence ID | 1차 출처 | 출처가 허용하는 산출 | 출처가 대신 정하지 않는 것 | 판정 |
|---|---|---|---|---|
| **SRC-U6-INF-01** | Starobinsky [DOI 10.1016/0370-2693(80)90670-X](https://doi.org/10.1016/0370-2693(80)90670-X); Sasaki [DOI 10.1143/PTP.76.1036](https://doi.org/10.1143/PTP.76.1036); Mukhanov et al. review [DOI 10.1016/0370-1573(92)90044-Z](https://doi.org/10.1016/0370-1573(92)90044-Z) | 명시된 $R+R^2$ action을 scalar degree of freedom으로 옮기고, quadratic perturbation action과 Mukhanov–Sasaki mode equation, vacuum/normalization을 고정해 $A_s,n_s,r$를 한 경로에서 산출하는 표준 경로가 있다. | projector가 $n_s,r$ 숫자를 고르는 것은 이 유도가 아니다. $A_s$는 action의 scale을 정하고, pivot의 e-fold 수에는 post-inflation history가 들어간다. | **유도 경로 provenance VERIFIED; CE action-to-spectrum chain UNVERIFIED** |
| **SRC-U6-REH-01** | Kofman–Linde–Starobinsky [arXiv:hep-th/9405187](https://arxiv.org/abs/hep-th/9405187), [DOI 10.1103/PhysRevLett.73.3195](https://doi.org/10.1103/PhysRevLett.73.3195); Dai–Kamionkowski–Wang [DOI 10.1103/PhysRevLett.113.041302](https://doi.org/10.1103/PhysRevLett.113.041302) | inflation에서 hot universe로 가는 전이는 coupling과 reheating dynamics에 의존하며, $N_*$는 reheating equation of state/temperature와 연결된다. | CE entropy 식 하나가 reheating action, decay channels와 thermalization을 대신하지 않는다. | **reheating 의존성 VERIFIED; CE reheating closure UNVERIFIED** |

이 두 행은 이론 출처라 관측 covariance는 `N/A`다. 수치 영향은 $A_s,n_s,r$ 중 일부만 맞추어도 나머지와 reheating-dependent cross observable이 자동으로 닫히지 않는다는 것이다.

### 6.2 dimensional transmutation은 절대 $\Lambda$ 선택 원리가 아니다

| Evidence ID | 1차 출처 | source boundary | 수치/공분산 영향 | 판정 |
|---|---|---|---|---|
| **SRC-U6-RG-01** | Coleman–Weinberg [DOI 10.1103/PhysRevD.7.1888](https://doi.org/10.1103/PhysRevD.7.1888); Coleman–Weinberg program의 dimensional-transmutation/gauge-independence 분석 [DOI 10.1016/0550-3213(76)90061-4](https://doi.org/10.1016/0550-3213(76)90061-4); Weinberg CC review [DOI 10.1103/RevModPhys.61.1](https://doi.org/10.1103/RevModPhys.61.1) | RG dimensional transmutation은 dimensionless coupling과 renormalization boundary condition을 dimensional scale로 교환할 수 있다. 그러나 field content, beta functions, matching condition과 vacuum-energy counterterm 없이 관측 $\Lambda$를 고르거나 radiative stability를 보장하지 않는다. | 관측 covariance `N/A`. CE scale을 얻으려면 적어도 하나의 boundary/matching datum 또는 별도 dynamical selection law가 남으므로 “무입력 절대척도” 수는 **0**이다. | **mechanism VERIFIED; CE absolute scale UNVERIFIED** |

### 6.3 four-form, sequestering, causal-set stochastic $\Lambda$

| Evidence ID | 1차 출처가 실제로 주는 것 | CE absolute-scale 주장에 남는 것 | 불확도·공분산 / 수치 영향 | 판정 |
|---|---|---|---|---|
| **SRC-U6-4F-01** | Brown–Teitelboim [DOI 10.1016/0370-2693(87)91190-7](https://doi.org/10.1016/0370-2693(87)91190-7)은 membrane nucleation에 의한 $\Lambda$ neutralization을, Bousso–Polchinski [arXiv:hep-th/0004134](https://arxiv.org/abs/hep-th/0004134)은 quantized four-form flux와 multi-flux discretuum을 제시한다. | flux 수, charges, compactification, bare vacuum term, membrane dynamics와 vacuum-selection measure를 지정해야 한다. four-form의 존재만으로 CE가 원하는 유일한 현재값은 정해지지 않는다. | 이론 parameter space라 관측 covariance `N/A`; single fixed CE 값의 사전확률/오차는 **UNDEFINED**. | **대체 mechanism VERIFIED; unique CE scale UNVERIFIED** |
| **SRC-U6-SEQ-01** | Kaloper–Padilla [arXiv:1309.6562](https://arxiv.org/abs/1309.6562), framework [arXiv:1406.0711](https://arxiv.org/abs/1406.0711), [DOI 10.1103/PhysRevD.90.084023](https://doi.org/10.1103/PhysRevD.90.084023); local formulation [arXiv:1505.01492](https://arxiv.org/abs/1505.01492), [DOI 10.1103/PhysRevLett.116.051302](https://doi.org/10.1103/PhysRevLett.116.051302) | vacuum-energy loop contributions을 curvature에서 sequester해 residual을 radiatively stable하게 만드는 action 경로가 있다. local 논문은 남는 finite curvature contribution이 a priori arbitrary하며 측정으로 정해야 한다고 명시한다. | 따라서 sequestering은 radiative-stability 후보이지 현재 $\Lambda$의 absolute numerical prediction이 아니다. | covariance `N/A`; residual scale을 결정하는 CE 독립식은 여전히 **0**. | **radiative-stability route VERIFIED; absolute prediction UNVERIFIED** |
| **SRC-U6-CS-01** | Ahmed–Dodelson–Greene–Sorkin [arXiv:astro-ph/0209274](https://arxiv.org/abs/astro-ph/0209274), [DOI 10.1103/PhysRevD.69.103523](https://doi.org/10.1103/PhysRevD.69.103523)은 causal-set/unimodular 동기의 stochastic everpresent $\Lambda$를 제시한다. Barrow [arXiv:gr-qc/0612128](https://arxiv.org/abs/gr-qc/0612128), [DOI 10.1103/PhysRevD.75.067301](https://doi.org/10.1103/PhysRevD.75.067301)과 Das–Nasiri–Yazdi [arXiv:2307.13743](https://arxiv.org/abs/2307.13743), [DOI 10.1088/1475-7516/2024/10/076](https://doi.org/10.1088/1475-7516/2024/10/076)은 CMB/late-time 적합의 제약을 보인다. | stochastic process, seed/measure, temporal/spatial correlation, perturbations과 CMB likelihood를 고정해야 한다. 현재 Model 1의 좋은 seed는 전형적이지 않고 CMB 적합도도 $\Lambda$CDM보다 나쁘다는 검증이 있다. | 단일 deterministic CE $\Lambda$의 오차·covariance로 바꿀 수 없다. stochastic ensemble likelihood가 필요하다. | **mechanism/test provenance VERIFIED; precise CE $\Lambda$ UNVERIFIED** |

### 6.4 de Sitter event horizon과 FLRW apparent horizon을 분리해야 한다

| Evidence ID | 1차 출처 | 정확한 provenance 경계 | 수치 영향 | 판정 |
|---|---|---|---|---|
| **SRC-U6-HOR-01** | Gibbons–Hawking [DOI 10.1103/PhysRevD.15.2738](https://doi.org/10.1103/PhysRevD.15.2738); Cai–Kim [arXiv:hep-th/0501055](https://arxiv.org/abs/hep-th/0501055), [DOI 10.1088/1126-6708/2005/02/050](https://doi.org/10.1088/1126-6708/2005/02/050); 동적 trapping-horizon 기반 Hayward [arXiv:gr-qc/9710089](https://arxiv.org/abs/gr-qc/9710089), [DOI 10.1088/0264-9381/15/10/017](https://doi.org/10.1088/0264-9381/15/10/017) | exact de Sitter의 observer event horizon은 $r_{dS}=H_{dS}^{-1}$이고 $S=A/(4G)$. 따라서 **unreduced** $M_P^2=1/G$ convention이면 $S_{dS}=\pi M_P^2/H_{dS}^2$이고, reduced $\bar M_P^2=1/(8\pi G)$이면 $S_{dS}=8\pi^2\bar M_P^2/H_{dS}^2$다. 일반 FLRW의 apparent-horizon radius는 $r_A=(H^2+k/a^2)^{-1/2}$이고 Einstein gravity에서 $S_A=A_A/(4G)$다. | $\Lambda$+matter 혼합 현재 epoch에는 $H_\Lambda=\sqrt{\Lambda/3}=H_0\sqrt{\Omega_\Lambda}\ne H_0$. 평탄한 현재 apparent horizon은 $r_A=H_0^{-1}$이지만 asymptotic de Sitter event horizon은 $H_\Lambda^{-1}$이며, 일반 event-horizon radius도 순간 $H^{-1}$와 같지 않다. 같은 area law를 써도 $S_{dS}(H_\Lambda)/S_A(H_0)=1/\Omega_\Lambda$다. | **horizon identities VERIFIED; CE phase-area의 horizon 선택 UNVERIFIED** |

따라서 phase-area 식이 $H_0$를 넣으면 “현재 flat-FLRW apparent-horizon readout”이고, $H_\Lambda$를 넣으면 “같은 $\Lambda$가 지배하는 asymptotic de Sitter event-horizon readout”이다. 둘을 출처 없이 같은 entropy라고 쓰면 absolute-scale 예측이 아니라 서로 다른 horizon 정의를 섞은 것이다.

## 7. U7 — 2026-08-16 공식 관측 snapshot과 covariance

선행 run이 2026-08-15에 공식 값과 잔차를 이미 재현했으므로, 아래는 변경 여부와 새 공식 배포만 확인했다. “변경 없음”의 수치 영향은 선행 판정 대비 0이다.

| Evidence ID | 공식 snapshot / 최신 확인 | uncertainty·covariance | 선행 대비 영향 | 상태 |
|---|---|---|---|---|
| **SRC-OBS-PLANCK-01** | ESA [Planck publications](https://www.cosmos.esa.int/web/planck/publications)와 [PLA](https://www.cosmos.esa.int/web/planck/pla)는 full-mission final cosmology release를 Planck 2018/A&A 2020으로 유지한다. 수치는 선행 `SRC-PLANCK` 그대로다. | PLA likelihood, chain와 covariance 공개. | 0; 새 Planck cosmology posterior 없음. | **VERIFIED unchanged** |
| **SRC-OBS-DESI-01** | 공식 [DESI DR2 papers](https://data.desi.lbl.gov/doc/papers/dr2/)의 최신 key paper는 Results IV [arXiv:2607.27410v3](https://arxiv.org/abs/2607.27410)이다. Results II 값·13점 covariance와 Results IV의 $z_{eff}=2.33$ joint AP+BAO $D_H/r_d=8.600\pm0.066$, $D_M/r_d=39.32\pm0.33$은 선행 `SRC-DESI-II`, `SRC-P1-02`, `SRC-DESI-IV`와 일치한다. | Results II 13x13 covariance 공개. Results IV 두 양은 상관되어 있으므로 주변오차만으로 기존 마지막 두 row를 교체할 수 없다. | P0 $-2.28\sigma$ 유지; holdout 수 0 유지. | **VERIFIED current; not independent holdout** |
| **SRC-OBS-ACT-01** | NASA LAMBDA [ACT DR6.02](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/index.html) 및 [chain readme](https://lambda.gsfc.nasa.gov/product/act/act_dr6.02/act_dr6.02_chains_info.html). 선행 P-ACT+lensing+DESI DR1 값 $(\omega_b,\omega_c,H_0)=(0.0226\pm0.0001,0.118\pm0.001,68.22\pm0.36)$ 유지. | spectra covariance와 MCMC chain 공개. chain tag에 Planck low-$\ell$, Planck/ACT lensing, DESI/SDSS가 명시되므로 조합 간 독립성은 tag별로 판정해야 한다. | 0; hybrid tuple 복권 불가. | **VERIFIED unchanged** |
| **SRC-OBS-SPT-01** | NASA LAMBDA [SPT-3G D1 official release](https://lambda.gsfc.nasa.gov/product/spt/spt_3gd1/)와 [corrected chain download](https://lambda.gsfc.nasa.gov/product/spt/spt3g_d1_bandp_liklyhood_get.html)는 2026-04-01 CMB-SPA bug-fixed chain을 유지한다. 선행 값 $(0.022398\pm0.000095,0.12028\pm0.00094,67.19\pm0.38)$ 유지. 2026 high-$\ell$ secondary TT product [arXiv:2601.20551v2](https://arxiv.org/abs/2601.20551)은 별도 foreground/reionization likelihood다. | corrected chain과 likelihood 공개; ACT/Planck와의 cross-experiment covariance는 단순 tuple에 없음. high-$\ell$ product도 CMB-SPA posterior를 교체하지 않는다. | 0; P0 hybrid 판정 유지. | **VERIFIED current** |
| **SRC-OBS-SH0ES-01** | Breuval et al. 4-anchor SH0ES $H_0=73.17\pm0.86\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$: [arXiv:2404.08038](https://arxiv.org/abs/2404.08038), [DOI 10.3847/1538-4357/ad630e](https://doi.org/10.3847/1538-4357/ad630e). 선행 `SRC-SH0ES24`와 동일. | 오차에 ladder systematics 포함; 이 레인에는 full calibration covariance 미적재. | 0. | **VERIFIED unchanged** |
| **SRC-OBS-TDCOSMO-01** | published canonical TDCOSMO-2025는 8 time-delay lenses + Pantheon+ flat $\Lambda$CDM에서 $H_0=71.6^{+3.9}_{-3.3}\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$: [arXiv:2506.03023v4](https://arxiv.org/abs/2506.03023), [A&A DOI 10.1051/0004-6361/202555801](https://doi.org/10.1051/0004-6361/202555801). | asymmetric marginal; mass-sheet/kinematic systematics를 넓게 포함하지만 이 레인에 full posterior covariance 미적재. | 선행 `SRC-TDCOSMO25`와 0. CE/저장소 $73.04$와 중심 차이 $-1.44$이나 CE uncertainty가 없어 combined $\sigma$ 계산 안 함. | **VERIFIED journal canonical; holdout 아님** |
| **SRC-OBS-HOLD-02** | freeze 뒤 2026-08-16까지 ESA Planck, DESI papers/data, NASA LAMBDA ACT/SPT 공식 공개처를 확인했으나 preregistration 조건을 만족하는 “freeze 뒤 최초 공개, 사전 미열람, full-covariance BAO” 자료를 찾지 못했다. | 부재 증명은 공개처 범위 밖까지 확장할 수 없다. manifest의 `unassigned`가 authoritative local state다. | 독립 blind pass/fail 계산 불가. | **QUALIFYING RELEASE NOT FOUND; global absence UNVERIFIED** |

## 8. source-level 통합 결론

| U 항목 | 출처가 허용하는 활성 대체 경로 | 출처만으로는 닫히지 않는 핵심 |
|---|---|---|
| U2 | conditioned extinction + reacting multifluid/Boltzmann current | species map, reaction kernel, freeze-out, entropy/energy normalization과 absolute $\Omega_m$ |
| U3 | coupled scalar 또는 pull-back dark-fluid action | CE coupling/action, $Q^\mu$, perturbation frame·sound speed·initial conditions와 stability |
| U4 | multi-species Einstein–Boltzmann growth + 제한영역 pressureless quadrature | radiation/matter/de Sitter 극한, 동일 initial condition의 독립 solver와 grid convergence |
| U5 | CLASS/CAMB + HyRec/HYREC-2 + released likelihood/covariance; inverse ladder 교차 | exact backend/version, full spectra/nuisance likelihood, overlap-aware covariance와 독립 holdout |
| U6 | $R+R^2$ + Mukhanov–Sasaki + reheating; RG/four-form/sequester/causal-set 후보 | amplitude scale와 reheating microphysics; unique/radiatively stable $\Lambda$ selection; horizon identity |
| U7 | 공식 single-release posterior/chain | P0 두 건의 수정, 아직 없는 future holdout |

이 레인이 수학 레인에 넘기는 가장 중요한 금지선은 다음과 같다.

1. conditioned Galton–Watson 정리를 baryon density 정리로 승격하지 않는다.
2. Brown/Schutz/Taub의 보존 perfect-fluid action을 particle-creation action으로 부르지 않는다.
3. background dark-sector ratio 또는 $Q$ 하나를 covariant action·linear perturbation 완결로 부르지 않는다.
4. $\theta_*$, toy curvature 또는 inverse ladder의 중심값 하나를 full CMB likelihood나 완전 독립 absolute-scale 측정으로 부르지 않는다.
5. dimensional transmutation, sequestering, four-form 또는 stochastic $\Lambda$의 문헌 존재를 CE의 고유한 현재 $\Lambda$ 예측으로 부르지 않는다.
6. 현재 apparent-horizon $H_0^{-1}$과 asymptotic de Sitter event-horizon $H_\Lambda^{-1}$을 같은 phase-area scale로 쓰지 않는다.
7. `SRC-FULL-P0-01`, `SRC-FULL-P0-02`와 독립 holdout 0건을 후속 통합에서 보존한다.

## 9. 종료 체크리스트

- [x] 계약 U2, U3, U5, U6의 신규 경로마다 논문/공식 문서 Evidence ID가 있다.
- [x] U4 성장 검산의 multi-species 기준과 pressureless quadrature 적용 한계를 분리했다.
- [x] Galton–Watson, relativistic reacting fluid, Sakharov/freeze-out, interacting DE, Boltzmann/recombination, inverse ladder, inflation/reheating, RG, four-form/sequestering/causal-set를 1차·공식 출처로 분리했다.
- [x] de Sitter event horizon과 general-FLRW apparent horizon, unreduced/reduced Planck-mass convention을 분리했다.
- [x] 외부 관측 수치마다 uncertainty, covariance 상태, 수치 영향과 VERIFIED/UNVERIFIED 경계를 적었다.
- [x] Planck/DESI/ACT/SPT/SH0ES/TDCOSMO 최신 공식 snapshot을 2026-08-16에 확인했다.
- [x] 선행 P0 두 건과 future holdout 부재를 보존 연결했다.
- [x] 출처 인용을 CE 수학적 닫힘으로 승격하지 않았다.
