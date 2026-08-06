# 05n. CE/PreEq 코드-현대 우주론 비교 감사

## 0. 위치와 목표

이 문서는 현재 코드베이스의 CE/PreEq 수학 도구와 우주론 수치 사용을 2025-2026년 관측 우주론 및 암흑물질 연구 결과와 대조한다. 목표는 네 가지다.

1. 코드와 테스트가 실제로 검증하는 수학적 표면을 분리한다.
2. 문서군의 우주론 수치 주장, 특히 \(\Omega_b\approx\varepsilon^2\), \(D_{\mathrm{eff}}\), \(N_{\mathrm{eff}}\) 조건을 자체 감사 조건과 함께 정리한다.
3. 최신 CMB/BAO/SNe/Euclid/암흑물질 탐색 결과와 항목별로 비교한다.
4. 검증된 것, 숫자 일치, 열린 가정, 반증 가능 조건을 판정한다.

핵심 결론:

> 현재 checkout은 finite PreEq/Gibbs·fraction-layer 감사에 더해 05o의
> Friedmann background, linear-growth ODE와 DESI DR2 compressed-BAO
> covariance gate까지 구현한다. canonical density를 명시 주입한 두 고정
> background 패키지는 모두 `REJECT`다. 아직 없는 것은 precision
> recombination/Boltzmann perturbation, CMB+BAO+SNe joint likelihood,
> dark-matter particle·structure likelihood다. 따라서 내부 수학과
> forward-model 계산은 재현되지만 경쟁 우주론의 물리 검증은 닫히지 않았다.

> **2026-08-06 이중 snapshot 계약:** 이 문서의
> `constants.py` 표는 checkout runtime에 남아 있는 legacy fixture
> \(0.0487,0.2623,0.6891\)를 정확히 감사한다. 원고 canonical manifest는
> \((0.0486382585,0.2610881744,0.6902735671)\)이며, 이를 구 runtime API에
> 명시 주입한 DESI DR2 결과는 외부 \(r_d=147.09\,{\rm Mpc}\)에서
> \(\chi^2=40.20145,\ p=1.2828\times10^{-4}\), EH-hybrid
> \(r_d=151.50523\,{\rm Mpc}\)에서
> \(\chi^2=41.19455,\ p=8.8602\times10^{-5}\)로 모두 `REJECT`다.
> legacy 상수나 무인자 runner 출력을 현행 원고 수치로 재사용하지 않는다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| finite Gibbs/PreEq core | `Exact/Tooling` | `pre_eq.py`와 테스트가 직접 검증 |
| hard constraint, layer-cake, mean-field bounds | `Exact under finite model` | 유한공간 함수와 Gamma Monte Carlo 회귀 존재 |
| \(\varepsilon^2=e^{-(1-\varepsilon^2)D_{\mathrm{eff}}}\) 수치해 | `Tooling/Exact` | bootstrap 방정식 자체를 받아들이면 수치해는 닫힘 |
| \(\Omega_b\approx\varepsilon^2\) 물리 식별 | `Phenomenology/Bridge` | 관측 likelihood에서 유도되지 않고 평균장 식별에 의존 |
| \(N_{\mathrm{eff}}\ge445\), \(\bar\rho\le2.2\times10^{-3}\) | `Audit/Exact under assumptions` | \(\Omega_b\) 오차 예산과 Gaussian-mode benchmark의 필요조건 |
| 암흑물질 직접탐색 설명력 | `Not implemented` | WIMP/axion/sterile-neutrino 질량, 산란단면, detector likelihood 부재 |
| 고정-background·growth·DESI compressed BAO | `Tooling/Phenomenology/REJECT` | 05o의 \(H(z)\), distance, linear \(D(a)\), \(S_8\), full-cov compressed BAO 구현; 두 canonical package 모두 기각 |
| 현대 우주론과 경쟁하는 예측 모델 | `Open` | CMB·SN·full-shape·lensing을 같은 parameter vector로 공동 fit/predict하는 likelihood 없음 |

## 1. 코드가 실제로 닫은 표면

### 1.1 finite PreEq 확률 도구

`reality_stone/python/reality_stone/clarus/pre_eq.py`는 유한 후보공간 위의 확률 도구다. 모듈 주석 자체가 "finite-dimensional"이며 "no CE path space, no physical readout"이라고 경계를 둔다. 실제 API는 다음을 닫는다.

| 기능 | 구현 | 검증 |
|---|---|---|
| prior 정규화 | `normalize_weights` | 음수, 0 질량, 비유한값 거부 |
| Gibbs 재가중 | `gibbs_reweight` | \(\beta\to\infty\) 근방에서 최저 energy 후보로 농축 |
| manifest index | `manifest_indices` | prior support와 finite energy 위의 argmin |
| 비선택 잔류 | `nonselected_residual` | raw mass와 conditional shape 분리 |
| kernel composition/pushforward | `compose_weighted_kernels`, `kernel_pushforward` | 두 단계와 합성 kernel 일치 |
| tropical/min-plus | `tropical_compose`, `gibbs_kernel`, `tropicalize` | log-sum-exp error \(\le\log n/\beta\) |
| joint Bayes readout | `joint_gibbs`, `marginals`, `conditional_values`, `free_energy_for_conditions` | row-wise Gibbs와 free-energy marginal 재구성 |
| Born prior finite shadow | `born_prior`, `refined_branch_prior` | phase invariant, branch refinement 일치 |
| hard constraint fraction | `survival_fraction`, `conditioned_prior` | threshold monotonicity, zero-mass 조건 거부 |
| tilt/layer-cake/mean-field | `tilt_survival`, `layer_cake_survival`, `mean_field_bounds` | layer-cake 정확 일치, Jensen bracket |

따라서 `pre_eq.py`의 과학적 지위는 "물리학 엔진"이 아니라 "후보공간, energy/defect, prior, \(\beta\)가 주어졌을 때의 수학적 reweighting 엔진"이다.

### 1.2 fraction-layer 수치 감사

`examples/pre_eq/fraction_threshold.py`와 `tests/test_pre_eq_fraction.py`는 05k-05m의 유한/Gamma 모델을 회귀한다.

| 검증 항목 | 수학 내용 | 물리 해석의 한계 |
|---|---|---|
| threshold scaling | \(u_{\mathrm{th}}=N/2+z_q\sqrt{N/2}\)이면 fixed fraction | \(z_q\) 값 자체는 선택값 |
| path-energy fraction gap | central regime에서 \(O(N^{-1/2})\) | 실제 CE path integral 아님 |
| mode action moments | \(u_k=z^2/2\)의 평균/분산 \((1/2,1/2)\) | 자유 Gaussian mode benchmark |
| mean-field ratio | \(\langle e^{-\Phi}\rangle/e^{-\langle\Phi\rangle}=1+O(1/N_{\mathrm{eff}})\) | \(\Phi\)가 intensive independent mode sum이라는 가정 필요 |
| R2 배제 | \(N_{\mathrm{eff}}=D_{\mathrm{eff}}\)이면 \(\Omega_b\)가 크게 밀림 | 배제는 관측값을 넣은 감사 |
| \(N_{\mathrm{eff}}\) 하한 | Planck \(\Omega_b\) 예산 아래 \(N_{\mathrm{eff}}\ge445\) | 실제 \(N_{\mathrm{eff}}\) 물리값은 미도출 |

중요한 점은 이 테스트들이 "CE가 \(\Omega_b\)를 물리적으로 예측했다"를 검증하지 않고, "\(\Omega_b\) 일치를 주장하려면 어떤 통계 조건이 필요한가"를 검증한다는 것이다.

### 1.3 우주론 비율 상수의 코드 사용

`reality_stone/python/reality_stone/clarus/constants.py`는 다음 비율을 single source of truth로 둔다.

| 상수 | 값 | 코드상 역할 | 우주론 대응 서사 |
|---|---:|---|---|
| `ACTIVE_RATIO` | 0.0487 | active/task-active fraction | \(\varepsilon^2\), \(\Omega_b\) 대응 |
| `STRUCT_RATIO` | 0.2623 | structural/plastic fraction | \(\Omega_{\mathrm{DM}}\) 대응 |
| `BACKGROUND_RATIO` | 0.6891 | frozen/background fraction | \(\Omega_\Lambda\) 대응 |
| `BOOTSTRAP_CONTRACTION` | 0.155 | \(\rho=D_{\mathrm{eff}}\varepsilon^2\) | bootstrap contraction |

`engine.py`는 이 비율을 active/struct/background partition과 decoder candidate ratio에 사용한다. `runtime.py`는 WAKE/NREM/REM occupancy를 \((\Omega_\Lambda,\Omega_{\mathrm{DM}},\Omega_b)\) simplex에 대응시켜 KL을 계산한다. `ce_euler.py`는

$$
D_{\mathrm{eff}}=3+s_A^2(1-s_A^2)
$$

형태의 effective dimension을 rotary base 등에 사용한다. 여기서 현행
\(s_A^2:=4\alpha_s^{4/3}\)는 CE 등록 출력이며 물리적 약혼합각의 특정
scheme과 동일시하지 않는다.

이것은 코드 아키텍처에서 우주론 비율을 은유적/구조적 hyperparameter로 쓰는 것이며, CMB 또는 대규모구조 관측을 forward-modeling하는 것이 아니다.

## 2. 문서의 우주론 주장과 자체 감사 조건

### 2.1 \(\Omega_b\) 생존분율 주장의 위치

05j-05m 문서군에서 우주론 연결은 주로 다음 식에 모인다.

$$
P_{\mathrm{survive}}=\langle e^{-\Phi}\rangle,
\qquad
\langle\Phi\rangle=\sigma D_{\mathrm{eff}},
\qquad
\varepsilon^2=e^{-(1-\varepsilon^2)D_{\mathrm{eff}}}.
$$

문서의 중요한 자기 제한은 다음이다.

| 항목 | 문서 판정 | 감사 의미 |
|---|---|---|
| \(\langle e^{-\Phi}\rangle\) | `supp:stat`, bounded tilt | manifest 선택이 아니라 fixed-\(\beta\) 분율 |
| \(S_E<S_{\mathrm{th}}\) | `supp:hard` | finite-\(N\) 조건화에서만 의미 있음 |
| \(\Phi\) mean-field | Jensen 하한 | \(e^{-\langle\Phi\rangle}\)은 참 survival의 하한 |
| \(z_q\approx-1.66\) | `Selection/Open` | \(\varepsilon^2\)에서 역산되는 분위수 |
| \(D_{\mathrm{eff}}\) 분해 | 평균의 분해 | 요동 mode 수가 아님 |
| \(N_{\mathrm{eff}}\) | `Open/Experiment` | 관측이 하한을 강제하지만 값은 미도출 |

즉 문서가 이미 경고하듯, \(\Omega_b\) 숫자 일치는 "manifest 극한에서의 경로 선택"이 아니라 "finite-\(\beta\)/finite-\(N\) fraction layer"의 주장이다.

### 2.2 \(N_{\mathrm{eff}}\) 감사의 핵심

05m의 중심 결론은 다음이다.

$$
m=\langle\Phi\rangle=(1-\varepsilon^2)D_{\mathrm{eff}}\approx3.0232.
$$

Gaussian-mode benchmark에서

$$
\frac{\langle e^{-\Phi}\rangle}{e^{-\langle\Phi\rangle}}
=
\exp\Big(m-k\ln(1+m/k)\Big),
\qquad
k=N_{\mathrm{eff}}/2.
$$

Planck \(\Omega_b\) 상대 예산 \(\delta_{\mathrm{rel}}\approx0.0206\)을 요구하면

$$
N_{\mathrm{eff}}\gtrsim445,
\qquad
\bar\rho\lesssim2.2\times10^{-3}.
$$

따라서 "CE가 \(\Omega_b\)와 맞는다"는 문장은 단독으로는 불충분하다. 올바른 완성형은 다음이다.

> CE 평균장 식별이 \(\Omega_b\) 정밀도 안에서 맞으려면 \(\Phi\)가 수백 개 이상의 유효 독립 미시 모드로 분산되고, 평균 쌍별 상관이 \(0.2\%\) 수준 이하로 억제되어야 한다.

이 조건은 현대 관측과 비교할 때 CE의 장점이자 부담이다. 장점은 숫자 주장을 반증 가능한 분산/상관 조건으로 바꾼 점이고, 부담은 그 mode 수와 상관 구조를 아직 물리적으로 계산하지 않았다는 점이다.

## 3. 최신 우주론 결과와의 비교

### 3.1 CMB baseline: Planck에서 ACT/SPT 결합으로

Planck 2018 base-\(\Lambda\)CDM은 대략

$$
\Omega_bh^2\approx0.0224,\qquad
\Omega_ch^2\approx0.120,\qquad
H_0\approx67.4\ \mathrm{km\,s^{-1}\,Mpc^{-1}},
\qquad
\sigma_8\approx0.811
$$

을 준다. 2025년 ACT DR6, SPT-3G, Planck 결합 분석들은 같은 \(\Lambda\)CDM 여섯 파라미터 틀에서 대체로 일관된 baryon/cold-dark-matter density를 준다. 예컨대 Planck+ACT+SPT 결합은 \(\Omega_bh^2\approx0.02228\), \(\Omega_ch^2\approx0.1195\) 수준의 결과를 보고한다.

CE 코드와의 비교:

| 관측 항목 | 현대 연구 지위 | 현재 코드의 대응 |
|---|---|---|
| \(\Omega_bh^2\) | CMB acoustic peaks와 BBN 일관성으로 정밀 측정 | canonical \(\Omega_b\)와 외부 \(H_0\)를 background에 주입 가능; recombination/acoustic likelihood는 없음 |
| \(\Omega_ch^2\) | CMB lensing/acoustic structure로 정밀 측정 | canonical \(\Omega_{\mathrm{DM}}\)를 background·growth에 주입 가능; Boltzmann perturbation은 없음 |
| \(H_0\) | CMB+BAO와 local distance ladder tension 지속 | 외부 \(H_0\) 입력과 사후 readout diagnostic 구현; 독립 유도·joint likelihood는 없음 |
| \(\sigma_8/S_8\) | CMB와 weak lensing/cluster 사이 probe-dependent tension | GR/CPL linear growth 구현, \(\sigma_{8,0}\)는 외부 입력; shear likelihood는 없음 |
| \(N_{\mathrm{eff}}\) | CMB의 relativistic species \(N_{\mathrm{eff}}\)와 별개 | CE의 \(N_{\mathrm{eff}}\)는 \(\Phi\) mode count이므로 표준 cosmological \(N_{\mathrm{eff}}\)와 혼동 금지 |

따라서 CE의 \(\Omega_b\) 숫자 대응은 CMB likelihood의 압축 결과와 비교될 수는 있지만, 그 likelihood를 재현하거나 대체하지 않는다.

### 3.2 DESI DR2와 동적 암흑에너지

DESI DR2는 3년치 1,400만 개 이상의 galaxies/quasars BAO 측정을 통해 flat \(\Lambda\)CDM이 여전히 좋은 fit을 주지만, BAO 선호 파라미터가 CMB와 약 \(2.3\sigma\) 긴장을 보이고, \(w_0w_a\) 동적 암흑에너지 모델이 DESI BAO+CMB에서 약 \(3.1\sigma\), SNe 포함 조합에 따라 \(2.8-4.2\sigma\) 선호될 수 있음을 보고했다.

CE 코드와의 비교:

| DESI 쟁점 | 필요한 모델 요소 | 현재 코드 상태 |
|---|---|---|
| BAO distance-redshift relation | \(H(z)\), \(D_A(z)\), sound horizon \(r_d\) | 구현; 외부 \(r_d\)와 EH-hybrid를 provenance로 분리 |
| \(w_0,w_a\) dark-energy equation of state | \(w(a)=w_0+w_a(1-a)\), Friedmann integration | 조건부 CPL forward map 구현; \(w_0,w_a\)의 CE 유도는 없음 |
| CMB+BAO joint likelihood | covariance, nuisance, parameter posterior | 없음 |
| \(\Omega_\Lambda\) 값 | canonical \(0.6902735671\) | boundary output; 절대 scale과 동적 stress는 별도 |

결론: CE는 이제 \(H(z)\), \(w(z)\), BAO observable을 산출해 고정 패키지를
직접 기각할 수 있다. 그러나 같은 DR2로 맞춘 scale ablation은 예측이 아니며,
CMB·SN와 nuisance를 포함한 joint explanation도 아니다.

### 3.3 \(S_8\), weak lensing, structure growth

2026년 기준 \(S_8\) 논의는 더 복잡하다. CMB 결합 baseline은 \(S_8\approx0.836\) 수준으로 높은 값을 주는 반면, DES Y6는 \(S_8\approx0.794\) 수준의 낮은 값을 보고해 약 \(2.4-2.7\sigma\) 긴장을 보인다. 반면 KiDS Legacy 등 일부 late-universe probe는 CMB와 더 잘 맞는다는 보고도 있어, survey systematic과 new physics 가능성이 함께 논의된다.

CE 코드와의 비교:

| \(S_8\) 구성요소 | 현재 코드 대응 |
|---|---|
| matter power spectrum \(P(k,z)\) | 없음 |
| growth factor \(D(z)\), \(\sigma_8\) | GR/CPL 조건부 \(D(z)\) 구현; \(\sigma_{8,0}=0.811\)은 외부 입력 |
| lensing kernel, shear two-point function | 없음 |
| baryonic feedback/systematics | 없음 |
| \(\Omega_m=\Omega_b+\Omega_c\) | canonical boundary로 background·linear-growth ODE 실행 |

현재 조건부 출력은 \(S_8(0)=0.824042\)지만 \(\sigma_{8,0}\)가 외부
normalization이고 lensing kernel·baryonic systematics가 없으므로
\(S_8\) tension 해결로 세지 않는다. 05m의 \(\Phi\) mode 상관 상한도
\(S_8\)의 matter clustering amplitude와는 다른 양이다.

### 3.4 Euclid 상태

Euclid Q1은 2025년 3월 약 63.1 deg², 약 2,600만 detections 규모의 quick data release로 공개되었지만, 공식 설명상 cosmology release가 아니라 pipeline과 astrophysics 연구를 위한 초기 공개다. 첫 핵심 cosmology data release는 2026년 10월 예정으로, weak lensing, galaxy clustering, dark matter/dark energy mapping이 본격 비교 대상이 된다.

CE 코드와의 비교:

| Euclid 산출물 | 현재 CE 대응 |
|---|---|
| 3D weak-lensing dark matter map | 없음 |
| galaxy clustering/BAO | 없음 |
| strong lens candidate catalog | 없음 |
| dark energy constraints | 없음 |

따라서 현재 시점에서 Euclid는 CE의 검증 데이터라기보다 향후 필요한 관측 benchmark다.

## 4. 최신 암흑물질 연구와의 비교

### 4.1 직접탐색: xenon TPC의 WIMP 미검출과 neutrino fog

2025-2026년 직접탐색의 중심은 LZ, XENONnT, PandaX-4T의 multi-ton liquid xenon TPC다.

| 실험 | 최신 핵심 결과 | CE 코드 대응 |
|---|---|---|
| XENONnT | 3.1 tonne-year WIMP search, no significant excess, SI limit \(\sim1.7\times10^{-47}\mathrm{cm^2}\) at 30 GeV | 없음 |
| PandaX-4T | 1.54 tonne-year WIMP search, no significant excess, SI limit \(\sim1.6\times10^{-47}\mathrm{cm^2}\) at 40 GeV; \(^{8}\mathrm B\) CEvNS indication \(2.64\sigma\) | 없음 |
| LZ | 5.7 tonne-year low-mass search, no DM excess; \(^{8}\mathrm B\) CEvNS evidence \(4.5\sigma\) | 없음 |

이 결과들의 물리적 의미는 두 가지다.

1. 표준 GeV-TeV WIMP의 spin-independent cross-section parameter space가 계속 줄어든다.
2. low-mass WIMP 영역에서 solar \(^{8}\mathrm B\) neutrino CEvNS가 dark matter recoil과 유사한 irreducible background, 즉 neutrino fog로 들어온다.

CE/PreEq 코드에는 다음이 없다.

| 직접탐색에 필요한 것 | 현재 상태 |
|---|---|
| dark matter mass \(m_\chi\) | 없음 |
| nucleon/electron scattering cross-section | 없음 |
| recoil spectrum \(dR/dE\) | 없음 |
| detector response, threshold, background likelihood | 없음 |
| neutrino CEvNS background | 없음 |

따라서 `STRUCT_RATIO=0.2623`이 \(\Omega_{\mathrm{DM}}\)에 대응한다는 서사만으로는 암흑물질 직접탐색과 비교할 수 없다. 현재 코드는 "암흑물질의 우주 평균 밀도 비율에 해당하는 상수"를 갖고 있을 뿐, "암흑물질이 어떤 입자이고 어떻게 검출기에 산란하는가"를 말하지 않는다.

### 4.2 axion, light dark matter, sterile neutrino

최신 암흑물질 연구는 WIMP 외에도 axion/ALP, dark photon, sub-GeV light dark matter, sterile neutrino, primordial black hole 등을 넓게 다룬다. Xenon TPC는 electron recoil 또는 absorption 채널로 일부 비-WIMP 후보를 탐색하고, 별도의 axion haloscope/helioscope 및 X-ray decay search도 병행된다.

CE 코드와의 비교:

| 후보군 | 필요한 예측 | 현재 CE 대응 |
|---|---|---|
| axion/ALP | mass, coupling \(g_{a\gamma}\), \(g_{ae}\), absorption/oscillation rate | 없음 |
| dark photon | kinetic mixing, absorption spectrum | 없음 |
| sterile neutrino | decay line, structure suppression | 없음 |
| light DM | electron/nucleus scattering form factor | 없음 |
| PBH | mass function, lensing/accretion constraints | 없음 |

따라서 CE의 dark matter 관련 현재 지위는 particle model이 아니라 density-ratio naming에 가깝다.

### 4.3 small-scale structure: JWST, Lyman-\(\alpha\), lensing

2025-2026년 small-scale dark matter 연구는 JWST high-redshift galaxy counts, Lyman-\(\alpha\) forest, strong lensing, dwarf satellite counts를 결합해 warm/fuzzy/self-interacting dark matter를 제한한다. 예컨대 일부 JWST star-formation 분석은 fuzzy dark matter mass \(m_{\mathrm{FDM}}>5.6\times10^{-22}\mathrm{eV}\), warm dark matter mass \(m_{\mathrm{WDM}}>1.5\mathrm{keV}\) 수준의 95% 제한을 보고한다. Lyman-\(\alpha\) forest emulator 연구들은 \(10^{-22}\mathrm{eV}\)급 pure fuzzy dark matter를 점점 불리하게 본다.

CE 코드와의 비교:

| 구조형성 관측 | 필요한 계산 | 현재 CE 대응 |
|---|---|---|
| halo mass function | Boltzmann/transfer function, N-body/hydro simulation | 없음 |
| Lyman-\(\alpha\) flux power | IGM thermal history, emulator/likelihood | 없음 |
| JWST UV luminosity function | star formation, feedback, reionization | 없음 |
| strong lensing substructure | halo substructure mass function | 없음 |

따라서 CE가 암흑물질 "양"을 상수로 둔다고 해도, 암흑물질 "성질"에 관한 최신 연구와 비교 가능한 예측은 아직 없다.

## 5. 숫자 일치의 의미와 한계

### 5.1 \(\Omega_b\) 근사 일치

CE bootstrap 계열은 \(\varepsilon^2\approx0.04865\)를 산출하고, 이는 Planck 계열의 \(\Omega_b\approx0.0486\)와 매우 가깝다. 이 숫자 일치는 코드베이스 안에서 중요한 동기다.

하지만 현대 우주론에서 \(\Omega_b\)는 독립 단일 숫자가 아니라 다음과 함께 posterior로 결정된다.

| 함께 결정되는 것 | 이유 |
|---|---|
| \(h\), \(\Omega_bh^2\) | CMB acoustic scale과 physical density |
| \(\Omega_ch^2\) | matter-radiation equality, peak heights |
| \(n_s\), \(A_s\), \(\tau\) | primordial spectrum과 reionization |
| \(Y_p\), BBN abundance | baryon density의 독립 검증 |
| BAO/SNe geometry | late-time expansion과 결합 |

CE의 \(\varepsilon^2\)는 \(\Omega_b\) 값 하나와 비교된다. 아직 \(\Omega_bh^2\), \(r_s\), acoustic peak ratios, BBN abundance를 동시에 산출하지 않는다. 그러므로 숫자 일치는 흥미로운 phenomenological coincidence 또는 target matching이지, CMB cosmology likelihood를 통과한 예측은 아니다.

### 5.2 \(\Omega_{\mathrm{DM}}\), \(\Omega_\Lambda\) 비율

`STRUCT_RATIO=0.2623`, `BACKGROUND_RATIO=0.6891`은 현대 flat \(\Lambda\)CDM의 \(\Omega_{\mathrm{DM}}\), \(\Omega_\Lambda\) 수준과 비슷하다. 그러나 최신 연구의 쟁점은 단순 현재값보다 다음이다.

| 항목 | 현대 쟁점 | 현재 CE 상태 |
|---|---|---|
| dark energy | \(w=-1\)인지, \(w_0w_a\) 동적인지 | CPL forward map은 구현; \(w_0,w_a\)의 CE 작용 유도 없음 |
| dark matter | cold/warm/fuzzy/SIDM/particle model | density ratio만 있음 |
| growth | \(\sigma_8\), \(S_8\), lensing tension | linear growth와 \(S_8\) 계산; \(\sigma_{8,0}\) 외부 입력, lensing likelihood 없음 |
| expansion | \(H_0\) tension, BAO distance ladder | distance/BAO gate 구현; canonical fixed package `REJECT`, joint ladder 없음 |

따라서 세 비율의 합이 현대 우주론 파라미터와 유사하다는 사실은 출발점이지 완결된 비교가 아니다.

## 6. 반증 가능 조건

현재 문서군이 현대 연구와 접점을 만들 수 있는 가장 좋은 지점은 이미 05m에 들어 있는 감사 조건이다.

| 조건 | 의미 | 어떻게 반증/검증할 수 있는가 |
|---|---|---|
| \(\operatorname{Var}(\Phi)\le0.041\) | mean-field \(\Omega_b\) 식별의 1σ 예산 | CE \(\Phi\) 분포를 실제 모델에서 샘플링 |
| \(N_{\mathrm{eff}}\gtrsim445\) | Gaussian-mode benchmark의 유효 독립 mode 수 하한 | cutoff/volume/field content에서 \(N_{\mathrm{eff}}\) 산출 |
| \(\bar\rho\lesssim2.2\times10^{-3}\) | mode 평균 상관 상한 | 상호작용 보정이 쌍별 상관으로 얼마나 들어가는지 계산 |
| \(z_q\approx-1.66\)의 물리 | threshold 위치의 독립 원리 | \(\varepsilon^2\)에서 역산하지 않고 threshold 원리 제시 |
| \(\Phi\ne S_E\) 구분 | intensive tilt와 extensive action 혼동 방지 | 코드/문서 기호와 scaling 검사 |

이 조건들은 "현대 최신 결과와 비교"에서 가장 생산적인 형태다. 즉 지금 해야 할 일은 CE를 DESI/LZ와 바로 맞붙이는 것이 아니라, CE 내부의 \(\Phi\) 분산, mode 수, 상관 구조를 계산 가능한 모델로 내려서 관측 정밀도 예산을 통과시키는 것이다.

## 7. 현대 연구 대비 결론표

| 분야 | 최신 연구 요지 | CE/PreEq 현재 지위 | 판정 |
|---|---|---|---|
| Planck/ACT/SPT CMB | \(\Omega_bh^2,\Omega_ch^2\) 정밀, \(\Lambda\)CDM 대체로 견고 | \(\Omega_b,\Omega_{\mathrm{DM}},\Omega_\Lambda\) 유사 비율 상수 | 숫자 비교만 가능 |
| DESI DR2 | 동적 dark energy \(w_0w_a\) 선호 가능성 \(3.1\sigma\) 내외 | background/CPL 및 DR2 compressed-BAO covariance 구현; canonical fixed packages는 `REJECT` | 고정 background 반증 완료, joint 설명은 미완 |
| \(H_0\) tension | CMB/BAO 낮은 \(H_0\), local 높은 \(H_0\) | 외부 \(H_0\) 입력과 post-hoc source-role readout | 기술 진단만 가능, 독립 예측 아님 |
| \(S_8\) tension | DES Y6 등 일부 low \(S_8\), probe-dependent systematic | conditional growth와 외부 \(\sigma_{8,0}\); lensing 없음 | 조건부 계산만 가능 |
| Euclid Q1/DR1 | Q1은 비-cosmology quick release, DR1이 향후 핵심 | lensing/clustering 없음 | 향후 benchmark |
| WIMP direct detection | LZ/XENONnT/PandaX 미검출, \(10^{-47}\mathrm{cm^2}\)대 제한 | mass/cross-section/recoil 없음 | 비교 불가 |
| neutrino fog | \(^{8}\mathrm B\) CEvNS가 low-mass DM 배경으로 부상 | CEvNS 없음 | 비교 불가 |
| axion/light DM | 후보별 coupling/mass 제한 강화 | particle model 없음 | 비교 불가 |
| WDM/FDM/SIDM structure | JWST/Lyman-\(\alpha\)/lensing으로 small-scale 제한 | transfer function/halo model 없음 | 비교 불가 |

## 8. 권장 다음 작업

현대 연구와 실제 비교 가능한 단계로 가려면 다음 순서가 필요하다.

1. **용어 분리:** `ACTIVE_RATIO`, `STRUCT_RATIO`, `BACKGROUND_RATIO`를 코드 주석에서 "cosmology-inspired fixed ratios"와 "observational cosmology prediction"으로 명확히 구분한다.
2. **\(\Phi\) sampler:** 05m의 \(\Phi\) mode decomposition을 실제 finite model로 구현해 \(\operatorname{Var}(\Phi)\), \(N_{\mathrm{eff}}\), \(\bar\rho\)를 산출한다.
3. **background 정밀화:** 구현된 \((\Omega_b,\Omega_c,\Omega_\Lambda,H_0,w_0,w_a)\) forward model을 radiation, neutrino, recombination/Boltzmann 계층으로 확장한다.
4. **likelihood boundary 확장:** 구현된 DESI compressed-BAO adapter의 provenance를 유지하면서 CMB/SN/DES Y6 nuisance와 cross-covariance를 포함한 joint adapter를 추가한다.
5. **dark matter split:** \(\Omega_{\mathrm{DM}}\) ratio와 particle dark matter model을 분리한다. WIMP/axion/FDM/WDM 중 하나를 선택하지 않으면 직접탐색/구조형성 비교는 불가능하다.
6. **README 판정 강화:** 우주론 비율은 현재 `Phenomenology/Bridge`이며, dark matter particle physics는 `Open/Not implemented`라고 명시한다.

## 9. 참고한 최신 결과

이 절의 외부 결과는 코드 판정의 기준점으로만 사용한다. CE/PreEq 문서군의 정리 지위와 섞지 않는다.

| 축 | 대표 결과 | 감사에서 사용한 요점 |
|---|---|---|
| Planck 2018 | [Planck 2018 results. VI. Cosmological parameters](https://www.aanda.org/articles/aa/abs/2020/09/aa33910-18/aa33910-18.html) | \(\Omega_bh^2\), \(\Omega_ch^2\), \(H_0\), \(\sigma_8\) baseline |
| CMB 결합 | [Combining cosmic microwave background datasets with consistent foreground modelling](https://arxiv.org/html/2511.04733), [SPT-3G D1](https://arxiv.org/pdf/2506.20707) | Planck/ACT/SPT 결합이 \(\Lambda\)CDM 파라미터를 더 조임 |
| DESI DR2 | [DESI DR2 Results II](https://arxiv.org/html/2503.14738), [On DESI's DR2 exclusion of ΛCDM](https://arxiv.org/html/2504.15336) | BAO+CMB에서 \(w_0w_a\) 동적 암흑에너지 선호, SNe 조합 의존성 |
| \(S_8\) tension | [Status of the \(S_8\) Tension: A 2026 Review](https://arxiv.org/html/2602.12238v2) | CMB baseline과 late-universe probes 사이 probe-dependent tension |
| Euclid | [Euclid Q1 data release](https://www.euclid-ec.org/science/q1/), [Euclid Q1 press release](https://www.euclid-ec.org/public/press-releases/euclid-quick-data-release-1/) | Q1은 cosmology release가 아니며 DR1이 향후 핵심 |
| XENONnT | [WIMP Dark Matter Search using a 3.1 Tonne-Year Exposure](https://arxiv.org/html/2502.18005) | WIMP 미검출, SI limit \(\sim1.7\times10^{-47}\mathrm{cm^2}\) |
| PandaX-4T | [Dark Matter Search Results from 1.54 Tonne-Year Exposure](https://pubmed.ncbi.nlm.nih.gov/39913753/), [Solar \(^{8}\mathrm B\) CEvNS indication](https://arxiv.org/html/2407.10892) | WIMP 미검출과 \(^{8}\mathrm B\) neutrino fog indication |
| LZ | [LZ light-DM and \(^{8}\mathrm B\) CEvNS result](https://arxiv.org/html/2512.08065) | low-mass DM 미검출, CEvNS evidence \(4.5\sigma\) |
| small-scale DM | [JWST star formation and dark matter models](https://www.aanda.org/articles/aa/pdf/2025/10/aa55390-25.pdf), [Lyman-\(\alpha\) constraints on fuzzy dark matter](https://arxiv.org/html/2606.06969) | FDM/WDM 제한과 baryonic feedback degeneracy |

## 10. 결론

$$
\boxed{
\text{현재 코드가 닫은 것}
=
\text{finite Gibbs/PreEq 수학}
\;+\;
\text{fraction-layer 감사}
\;+\;
\text{conditional background/growth와 DESI compressed-BAO gate}
}
$$

$$
\boxed{
\text{현재 코드가 닫지 않은 것}
=
\text{precision CMB와 CMB/BAO/SNe joint likelihood}
\;+\;
\text{nonlinear structure와 lensing likelihood}
\;+\;
\text{dark matter particle/detector physics}
}
$$

따라서 CE/PreEq의 현재 우주론 지위는 다음 한 문장으로 정리된다.

> canonical \(\Omega_b,\Omega_{\mathrm{DM}},\Omega_\Lambda\)를 조건부
> Friedmann/growth/BAO forward model로 내렸고 DESI DR2 compressed gate에서
> 두 고정 background를 기각했다. 그러나 precision CMB·SN·lensing과 joint
> nuisance likelihood가 없으므로 현대 관측 우주론을 설명하는 완결 모형으로
> 승격되지는 않았다.

다음 병목은 더 많은 상수를 맞추는 것이 아니라, \(\Phi\)의 분산/상관을
실제 계산하고 이미 구현된 expansion/growth observable을
recombination·lensing·joint likelihood까지 확장하는 것이다.
