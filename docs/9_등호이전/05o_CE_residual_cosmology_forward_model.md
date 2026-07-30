# 05o. CE Residual Cosmology Forward Model

## 0. 목표

[05n_CE_cosmology_modern_audit.md](05n_CE_cosmology_modern_audit.md)는 CE 성분비가 최신 CMB 계열 \(\Omega\) 값과 가깝지만, 현대 우주론과 맞붙으려면 \(H(z)\), distance, growth observable로 내려가야 한다고 결론냈다. 이 문서는 그 첫 번째 실행 가능한 forward model을 고정한다.

핵심 결론:

> CE 성분비를 현재 우주의 boundary condition으로 쓰면, 보수적인 \(\Lambda\)CDM 한계에서 \(H(z)\), luminosity distance, linear growth, \(S_8\), \(f\sigma_8\)를 계산할 수 있다. 이 단계는 background/growth 검산까지 닫지만, particle dark matter와 detector likelihood는 여전히 `Open`이다.

현재 판정:

| 항목 | 판정 | 코드 |
|---|---|---|
| CE 성분비 \(\to H(z)\) | `Tooling/Exact` | `examples/physics/ce_residual_forward_model.py` |
| BAO 압축 관측량 \(D_M/r_d,D_H/r_d,D_V/r_d\) | `Tooling/Exact` | `bao_observable` |
| diagonal/full BAO \(\chi^2\) | `Tooling/Exact`, DESI DR2 compressed mean/cov embedded | `bao_chi2`, `bao_chi2_with_covariance`, `named_bao_dataset` |
| BAO p-value·판정·잔차 기여도 | `Tooling/Exact` | `assess_bao_fit` |
| \(H_0r_d\) scale ablation | `Diagnostic fit`, CE 예측 아님 | `BAOScaleFitDiagnostic` |
| 외부/초기우주 근사 \(r_d\) 선택 | `External input` / `Selection/Approximation` | `sound_horizon_selection`, `early_universe_sound_horizon` |
| CPL \(w_0w_a\) 확장 | `Tooling/Exact` | `dark_energy_scale` |
| linear growth \(D(a)\), \(f\sigma_8\) | `Tooling/Exact under model` | `solve_growth` |
| \(S_8\) 현재값 검산 | `Tooling/Exact` | `s8_today` |
| WIMP/axion/FDM detector physics | `Not implemented` | coverage flag |

## 1. 현재 경계조건

코드는 CE 상수 파일에서 다음 값을 읽는다.

$$
\Omega_b=0.0487,\qquad
\Omega_{\mathrm{DM}}=0.2623,\qquad
\Omega_\Lambda=0.6891.
$$

반올림으로 합이 \(1.0001\)이므로 background 계산에는 내부 정규화 값을 쓴다.

$$
\widehat\Omega_m
=
\frac{\Omega_b+\Omega_{\mathrm{DM}}}
{\Omega_b+\Omega_{\mathrm{DM}}+\Omega_\Lambda},
\qquad
\widehat\Omega_\Lambda
=
\frac{\Omega_\Lambda}
{\Omega_b+\Omega_{\mathrm{DM}}+\Omega_\Lambda}.
$$

기본값은

$$
h_0=0.674,\qquad
\sigma_{8,0}=0.811,\qquad
w_0=-1,\qquad
w_a=0.
$$

여기서 \(\Omega_b,\Omega_{\rm DM},\Omega_\Lambda\)는 CE boundary output이고,
\(H_0,r_d,\sigma_{8,0}\)는 외부 관측 입력이다. \(w_0,w_a,\mu(a)\)는 모형
가정이다. 즉 첫 검산은 CE 성분비와 외부 거리·성장 calibration을 결합한 flat
\(\Lambda\)CDM 한계다.

## 2. Background 수식

동적 dark energy 확장을 위해 CPL parameterization을 쓴다.

$$
w(a)=w_0+w_a(1-a),
$$

$$
F_{\mathrm{DE}}(a)
:=
\frac{\rho_{\mathrm{DE}}(a)}{\rho_{\mathrm{DE}}(1)}
=
a^{-3(1+w_0+w_a)}\exp\big(3w_a(a-1)\big).
$$

따라서

$$
E^2(a)
:=
\frac{H^2(a)}{H_0^2}
=
\widehat\Omega_m a^{-3}
+
\widehat\Omega_\Lambda F_{\mathrm{DE}}(a).
$$

\(\Lambda\)CDM 한계 \(w_0=-1,w_a=0\)에서는

$$
F_{\mathrm{DE}}(a)=1,
\qquad
E^2(a)=\widehat\Omega_m a^{-3}+\widehat\Omega_\Lambda.
$$

Luminosity distance는

$$
D_L(z)
=
\frac{c}{H_0}(1+z)\int_0^z\frac{dz'}{E(z')}.
$$

BAO 비교에는 flat geometry에서

$$
D_M(z)=\frac{D_L(z)}{1+z},
\qquad
D_H(z)=\frac{c}{H(z)}
$$

를 쓰고, isotropic BAO 요약량은

$$
D_V(z)=\big[zD_M(z)^2D_H(z)\big]^{1/3}
$$

로 둔다. 기존 결과와의 호환을 위한 `--rd-mode external`의 기본 sound
horizon scale은

$$
r_d=147.09\ \mathrm{Mpc}
$$

이다. `--rd-mode early-universe`는 CE 밀도 boundary와 외부
\(H_0,T_{\rm CMB}\), 표준모형 가정 \(N_{\rm eff}=3.044\),
Eisenstein--Hu drag fit에서 \(r_d\)를 선계산한다. 두 모드는 provenance에서
분리한다. 출력은 \(D_M/r_d\), \(D_H/r_d\), \(D_V/r_d\)다. 이는 DESI
likelihood 전체가 아니라 compressed observable layer다.

대각 오차만 있는 첫 likelihood는

$$
\chi^2_{\mathrm{BAO,diag}}
=
\sum_i
\left(
\frac{O_i^{\mathrm{model}}-O_i^{\mathrm{obs}}}{\sigma_i}
\right)^2,
\qquad
O_i\in\{D_M/r_d,\ D_H/r_d,\ D_V/r_d\}.
$$

full covariance가 주어지면

$$
\chi^2_{\mathrm{BAO}}
=
\Delta O^\top C^{-1}\Delta O,
\qquad
\Delta O_i=O_i^{\mathrm{model}}-O_i^{\mathrm{obs}}
$$

를 쓴다. 공식 DESI DR2 compressed Gaussian BAO mean/covariance는
`CobayaSampler/bao_data`의 `desi_bao_dr2` ASCII 파일을 source로 고정했다.
현재 내장 dataset은 다음 두 개다.

| 이름 | 내용 | source file |
|---|---|---|
| `desi-dr2-bgs` | BGS 단일 \(D_V/r_d\) point, \(z=0.295\) | `desi_gaussian_bao_BGS_BRIGHT-21.35_GCcomb_mean/cov.txt` |
| `desi-dr2-all` | DESI DR2 BGS/LRG/ELG/QSO/Ly\(\alpha\) 13-point compressed vector와 13x13 covariance | `desi_gaussian_bao_ALL_GCcomb_mean/cov.txt` |

CLI data format은

```text
z:kind:value:sigma
```

이며 `kind`는 `dm`, `dh`, `dv` 중 하나다. 여러 점은 comma로 연결한다.

covariance CLI format은 row를 semicolon으로, column을 comma 또는 whitespace로 구분한다.

```text
0.04,0.01;0.01,0.09
```

공식 내장 dataset은 `--bao-dataset`으로 실행한다.

```powershell
python examples\physics\ce_residual_forward_model.py --bao-dataset desi-dr2-all
python examples\physics\ce_residual_forward_model.py --rd-mode early-universe --bao-dataset desi-dr2-all
```

현재 CE 기본 \(\Lambda\)CDM boundary에서 `desi-dr2-all` full covariance 결과는
\(\chi^2=37.100260857\), \(N=13\)이다. 이것은 compressed BAO layer의 직접 검산값이며,
아직 CMB/SN joint fit이나 nuisance marginalization을 포함하지 않는다.

### 2.1 초기우주 \(r_d\) 근사

물리 밀도를

$$
\omega_b=\Omega_bh^2,\qquad
\omega_m=\Omega_mh^2
$$

로 쓰고,

$$
\omega_\gamma
=
2.469\times10^{-5}
\left(\frac{T_{\rm CMB}}{2.7255\,{\rm K}}\right)^4,
\qquad
\omega_r
=
\omega_\gamma\left(1+0.22710731766N_{\rm eff}\right)
$$

로 둔다. Eisenstein--Hu drag fit은

$$
z_d
=
\frac{1291\omega_m^{0.251}}
{1+0.659\omega_m^{0.828}}
\left(1+b_1\omega_b^{b_2}\right),
$$

$$
b_1
=
0.313\omega_m^{-0.419}
\left(1+0.607\omega_m^{0.674}\right),
\qquad
b_2=0.238\omega_m^{0.223}
$$

이다. \(a_d=(1+z_d)^{-1}\),

$$
R_b(a)=\frac{3\omega_ba}{4\omega_\gamma},
\qquad
c_s(a)=\frac{c}{\sqrt{3(1+R_b(a))}}
$$

에서

$$
r_d
=
\int_0^{a_d}
\frac{c_s(a)}{a^2H(a)}\,da
$$

를 10,001점 composite Simpson 적분으로 계산한다.

현재 고정 입력은

$$
z_d^{\rm EH}=1020.020419907,
\qquad
r_d^{\rm EH\ hybrid}=151.318753028\ {\rm Mpc}
$$

를 준다. DESI DR2 13점 full covariance 결과는

$$
\chi^2=40.468225544,\qquad
\nu=13,\qquad
p=1.16176098\times10^{-4},
$$

즉 `REJECT`다. 외부 \(r_d=147.09\,{\rm Mpc}\)의
\(\chi^2=37.100260857\)보다 오히려 나빠졌다.

이 결과는 “초기우주 수식을 넣었으니 CE가 \(r_d\)를 유도했다”는 뜻이
아니다. 이 계산은 외부 \(H_0,T_{\rm CMB}\), 표준 \(N_{\rm eff}\)와 경험적
drag fit을 섞은 근사다. Eisenstein--Hu \(z_d\)는 precision recombination
또는 Boltzmann solver가 아니므로, 현재 차이는 새 자유도를 더하기보다 drag
optical depth와 열이력을 실제로 닫아야 한다는 실패 진단이다. DESI 값은
함수의 runtime 입력이 아니지만 이 모드는 DR2 잔차를 본 뒤 추가됐으므로 DR2를
untouched holdout으로 세지 않는다.

### 2.2 drag optical-depth history adapter

precision 재결합 코드를 내부에서 흉내 내지 않고, 외부 solver의
\(x_e(z)=n_e/n_H\) export를 받는 adapter를 둔다.

$$
\tau_{\rm drag}(z)
=
\int_0^z
\frac{c\sigma_Tn_{H,0}x_e(z')(1+z')^2}
{H(z')R(z')}\,dz',
\qquad
R(z)=\frac{3\omega_b}{4\omega_\gamma(1+z)}.
$$

\(\tau_{\rm drag}(z_d)=1\)을 piecewise-linear rate 적분과 bracket 내부
bisection으로 찾고, \(r_d=r_s(z_d)\)를 기존 Simpson sound-horizon 적분으로
계산한다. 입력은 raw-byte SHA-256과 함께 solver/version/backend, \(Y_p\),
cosmology, 단위, column/delimiter/grid provenance를 가져야 한다.

합성 analytic history 검산은

$$
z_d=1059.25,\qquad
r_d=147.649757605\ {\rm Mpc}
$$

를 회수한다. 이것은 실제 CLASS/CAMB/HyRec 출력이나 CE 예측이 아니다. 실제
solver가 없던 환경에서 adapter의 식·단위·root를 검산한 manufactured
fixture이며, precision 교차검증은 solver export를 등록한 뒤에만 성립한다.

### 2.3 고정 모델 판정

현재 코드는

$$
p_{\rm surv}
=
Q\!\left(\frac{\nu}{2},\frac{\chi^2}{2}\right)
$$

를 계산하고, \(p\ge0.05\)를 `PASS`, \(0.0027\le p<0.05\)를 `TENSION`,
\(p<0.0027\)을 `REJECT`로 고정한다. 기본 모델은 BAO 데이터로 fit한 파라미터가
없으므로 \(\nu=13\)이고,

$$
\frac{\chi^2}{\nu}=2.853866220,
\qquad
p=3.9957326\times10^{-4}
$$

라서 `REJECT`다. 이것은 “CE core 전체 기각”이 아니라

$$
\text{CE density boundary}
+
\text{외부 }H_0,r_d
+
\Lambda\text{CDM closure}
$$

패키지의 고정 BAO goodness-of-fit 실패다.

관측별 covariance 기여는

$$
c_i=r_i(C^{-1}r)_i,
\qquad
\sum_i c_i=\chi^2
$$

로 출력한다. 가장 큰 기여는 \(z=0.934\)의 \(D_M/r_d\), \(z=0.706\)의
\(D_H/r_d,D_M/r_d\), \(z=0.510\)의 \(D_H/r_d\)에서 나온다.

### 2.4 왜 안 맞는가: \(H_0r_d\) scale ablation

BAO ratio는 고정된 shape에서

$$
O_i
=
\frac{c}{H_0r_d}
F_i(z;\Omega_m,w_0,w_a)
$$

처럼 공통 scale을 갖는다. 예측 \(y\), 데이터 \(d\)에 대해 \(y(q)=qy\)만
허용하면 최적 scale은

$$
q_*
=
\frac{y^\top C^{-1}d}
{y^\top C^{-1}y}
=
0.986476933470
$$

이다. 진단 결과는

$$
\chi^2_{\rm scale}=12.608346862,
\qquad
\nu=12,
\qquad
p=0.398138
$$

로 `PASS`다. \(k\)를 BAO 데이터로 fit한 파라미터 수로 두고

$$
{\rm AIC}=\chi^2+2k,
\qquad
{\rm BIC}=\chi^2+k\ln N
$$

를 적용하면 고정모델은 \({\rm AIC}={\rm BIC}=37.1003\), scale fit은
\({\rm AIC}=14.6083,\ {\rm BIC}=15.1733\)이다. 즉 1개 자유도 패널티로는
개선이 사라지지 않는다. 이는 \(H_0=67.4\)를 고정하면
\(r_d=149.106375\,{\rm Mpc}\), \(r_d=147.09\,{\rm Mpc}\)를 고정하면
\(H_0=68.323949\,{\rm km\,s^{-1}Mpc^{-1}}\)에 해당한다.

따라서 현재 불일치의 1차 원인은 density shape보다 외부 \(H_0r_d\)
normalization이다. 그러나 같은 DESI 데이터로 맞춘 1-parameter 진단이므로 이
값은 CE 예측이나 closure가 아니다. 독립 CMB/sound-horizon calibration 전에
기본값으로 채택하면 사후보정이다.

## 3. Growth 수식

선형 성장률은 \(x=\ln a\)에 대해

$$
\frac{d^2D}{dx^2}
+
\left(2+\frac{d\ln H}{d\ln a}\right)
\frac{dD}{dx}
-
\frac32\mu(a)\Omega_m(a)D
=0
$$

를 푼다. 기본 검산은 GR 한계이므로

$$
\mu(a)=1.
$$

잔류 sector가 성장에 주는 효과를 탐색할 수 있도록 phenomenological coupling도 열어 둔다.

$$
\mu(a)
=
1-\epsilon_{\mathrm{grav}}
\frac{\Omega_{\mathrm{DE}}(a)}{\Omega_{\mathrm{DE}}(1)}.
$$

기본값은 \(\epsilon_{\mathrm{grav}}=0\)이다. 즉 이 문서의 닫힌 판정은 "성장 방정식 도구를 만들었다"이지, "수정중력 신호를 예측했다"가 아니다.

관측량은

$$
\sigma_8(z)=\sigma_{8,0}D(z),
\qquad
f\sigma_8(z)=\frac{d\ln D}{d\ln a}\sigma_8(z),
$$

그리고

$$
S_8(0)=\sigma_{8,0}\sqrt{\frac{\widehat\Omega_m}{0.3}}
$$

으로 계산한다.

## 4. 검산 출력

실행:

```powershell
python examples\physics\ce_residual_forward_model.py
```

대표 출력:

| 양 | 값 |
|---|---:|
| \(\Omega_b\) | 0.048700 |
| \(\Omega_{\mathrm{DM}}\) | 0.262300 |
| \(\Omega_m\) | 0.311000 |
| \(\Omega_\Lambda\) | 0.689100 |
| \(S_8(0)\) | 0.825734 |
| \(E(1)\) at \(z=1\) | 1.782353 |
| \(D_L(z=1)\) Mpc | 6818.454 |
| \(D_M(z=1)/r_d\) | 23.178 |
| \(D_H(z=1)/r_d\) | 16.966 |
| \(D_V(z=1)/r_d\) | 20.889 |
| \(f\sigma_8(z=0.5)\) | 0.473314 |

해석:

- \(S_8(0)\approx0.826\)은 2026 combined CMB baseline \(S_8\approx0.836\)과 가까운 쪽이다.
- DES Y6 같은 낮은 \(S_8\approx0.794\) probe와의 긴장은 이 모델이 아직 해소하지 않는다.
- \(w_0w_a\)를 바꾸면 \(E(z)\), \(D_L(z)\)가 변하므로 DESI DR2류 background tension을 실험할 최소 도구는 생겼다.

## 5. 테스트

회귀검사:

```powershell
uv run --extra dev python -m pytest tests\test_ce_residual_forward_model.py tests\test_recombination_drag_adapter.py tests\test_cosmology_ratio_audit.py -q
```

현재 결과:

```text
33 passed
```

검증 항목:

| 테스트 | 내용 |
|---|---|
| `test_lcdm_limit_has_constant_dark_energy_density` | \(w_0=-1,w_a=0\)에서 \(F_{\mathrm{DE}}=1\) |
| `test_ce_forward_model_uses_ce_density_ratios_for_background` | CE 성분비가 flat background를 만들고 \(E(0)=1\) |
| `test_ce_s8_today_is_close_to_combined_cmb_s8_baseline` | \(S_8(0)\)가 combined CMB baseline 근방 |
| `test_growth_solution_is_normalized_and_monotone` | \(D(1)=1\), 성장 단조성, \(f\sigma_8\) 양수 |
| `test_dynamic_dark_energy_changes_background_observables` | \(w_0w_a\) 변경이 \(E(z)\), \(D_L(z)\)를 바꿈 |
| `test_bao_observable_uses_consistent_distance_definitions` | \(D_M/r_d,D_H/r_d,D_V/r_d\) 정의 일치 |
| `test_early_universe_radiation_and_sound_speed_relations` | \(\omega_\gamma,\omega_r,c_s\) 관계와 단위 검산 |
| `test_early_universe_sound_horizon_matches_analytic_sanity_gate` | Simpson 수렴과 물질--복사 닫힌형 교차검산 |
| `test_sound_horizon_modes_and_early_input_provenance_are_separate` | 외부 \(r_d\)와 EH hybrid 입력 역할 분리 |
| `test_bao_observable_is_sensitive_to_dynamic_dark_energy` | \(w_0w_a\) 변경이 BAO 압축량을 바꿈 |
| `test_bao_diagonal_chi2_is_zero_for_fiducial_generated_data` | 자기 생성 fiducial data에서 \(\chi^2=0\) |
| `test_bao_diagonal_chi2_detects_dynamic_de_shift` | fiducial BAO data가 동적 DE shift를 검출 |
| `test_bao_full_covariance_matches_diagonal_chi2_for_diagonal_covariance` | 대각 covariance가 diagonal \(\chi^2\)와 일치 |
| `test_bao_full_covariance_responds_to_correlations` | correlation term이 \(\chi^2\)를 바꿈 |
| `test_chi_square_survival_and_preregistered_verdict_boundaries` | p-value와 판정 경계 고정 |
| `test_bao_assessment_decomposes_diagonal_chi2_into_raw_pull_squares` | 대각 covariance 기여도 검증 |
| `test_desi_dr2_full_covariance_fixed_model_is_rejected_and_decomposed` | DESI 고정모델 `REJECT`, scale 진단 `PASS`, 기여도 합 검증 |
| `test_desi_dr2_assesses_precomputed_early_rd_without_runtime_data_input` | EH hybrid도 DESI DR2에서 `REJECT`이고 외부 모드보다 악화됨을 고정 |
| `test_covariance_parser_and_inverter_validate_matrix` | covariance parser와 inverse 검증 |
| `test_named_desi_dr2_bao_datasets_are_available` | DESI DR2 BGS/ALL named dataset registry 검증 |
| `test_named_desi_dr2_bao_dataset_chi2_runs_against_ce_model` | 공식 BGS covariance로 CE model \(\chi^2\) 계산 가능 |
| `test_unknown_named_bao_dataset_raises` | unknown dataset 방어 |
| `test_parse_bao_data_validates_kind_and_sigma` | BAO data parser 검증 |
| `test_forward_coverage_keeps_dark_matter_particle_physics_open` | particle/detector DM은 여전히 Open |
| `test_parameter_provenance_separates_predictions_inputs_and_assumptions` | CE boundary·외부 입력·모형 가정 분리 |
| `tests/test_recombination_drag_adapter.py` | external history SHA/provenance·단위·grid·cosmology 일치와 \(\tau_{\rm drag}=1\) bracket, 합성 \(z_d,r_d\) 회귀 |

## 6. 남은 것

이 forward model이 새로 닫은 것은 background/growth 계산이다. 아직 닫지 않은 것은 다음이다.

| 병목 | 필요한 다음 수식 |
|---|---|
| DESI likelihood | official source 자동 동기화, nuisance/systematic 처리, 다른 likelihood convention cross-check |
| \(H_0r_d\) calibration | EH hybrid와 외부 \(x_e(z)\) history adapter는 구현됨; 다음은 실제 CLASS/HyRec·CAMB export, solver 자체 \(z_d,r_d\) 교차검증, radiation·neutrino·atomic physics, \(\theta_*\) |
| 낮은 \(S_8\) probes | baryonic feedback 또는 \(\mu(a)\)의 원리적 유도 |
| 암흑물질 직접탐색 | \(m_\chi\), \(\sigma_{\chi N}\), recoil spectrum |
| FDM/WDM/SIDM | transfer function \(T(k)\), halo mass function |

## 7. 결론

$$
\boxed{
\text{CE ratios}
\longrightarrow
H(z),\ D_L(z),\ D(a),\ f\sigma_8(z),\ S_8(0)
}
$$

이 단계에서 우리가 가리키는 우주는 다음처럼 더 선명해진다.

> CE 성분비는 background/growth boundary로 실행 가능하지만, 외부
> \(r_d=147.09\,{\rm Mpc}\) 패키지와 EH hybrid
> \(r_d=151.318753\,{\rm Mpc}\) 패키지는 DESI DR2 compressed BAO에서 모두
> `REJECT`다. 단일 \(H_0r_d\) scale 진단은
> \(r_d=149.106375\,{\rm Mpc}\)에서 적합도를 회복하지만 같은 데이터로 맞춘
> 값이다. 다음 병목은 새 density 식이 아니라 precision recombination과 독립
> sound-horizon/distance calibration이다.
