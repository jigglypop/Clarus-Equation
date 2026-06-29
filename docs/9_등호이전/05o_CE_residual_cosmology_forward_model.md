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

즉 첫 검산은 CE 성분비를 boundary로 한 flat \(\Lambda\)CDM 한계다.

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

로 둔다. 코드의 기본 sound horizon scale은

$$
r_d=147.09\ \mathrm{Mpc}
$$

이고 출력은 \(D_M/r_d\), \(D_H/r_d\), \(D_V/r_d\)다. 이는 DESI likelihood 전체가 아니라, covariance를 붙이기 전의 compressed observable layer다.

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
```

현재 CE 기본 \(\Lambda\)CDM boundary에서 `desi-dr2-all` full covariance 결과는
\(\chi^2=37.100260857\), \(N=13\)이다. 이것은 compressed BAO layer의 직접 검산값이며,
아직 CMB/SN joint fit이나 nuisance marginalization을 포함하지 않는다.

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
uv run --extra dev python -m pytest tests\test_ce_residual_forward_model.py tests\test_cosmology_ratio_audit.py -q
```

현재 결과:

```text
20 passed
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
| `test_bao_observable_is_sensitive_to_dynamic_dark_energy` | \(w_0w_a\) 변경이 BAO 압축량을 바꿈 |
| `test_bao_diagonal_chi2_is_zero_for_fiducial_generated_data` | 자기 생성 fiducial data에서 \(\chi^2=0\) |
| `test_bao_diagonal_chi2_detects_dynamic_de_shift` | fiducial BAO data가 동적 DE shift를 검출 |
| `test_bao_full_covariance_matches_diagonal_chi2_for_diagonal_covariance` | 대각 covariance가 diagonal \(\chi^2\)와 일치 |
| `test_bao_full_covariance_responds_to_correlations` | correlation term이 \(\chi^2\)를 바꿈 |
| `test_covariance_parser_and_inverter_validate_matrix` | covariance parser와 inverse 검증 |
| `test_named_desi_dr2_bao_datasets_are_available` | DESI DR2 BGS/ALL named dataset registry 검증 |
| `test_named_desi_dr2_bao_dataset_chi2_runs_against_ce_model` | 공식 BGS covariance로 CE model \(\chi^2\) 계산 가능 |
| `test_unknown_named_bao_dataset_raises` | unknown dataset 방어 |
| `test_parse_bao_data_validates_kind_and_sigma` | BAO data parser 검증 |
| `test_forward_coverage_keeps_dark_matter_particle_physics_open` | particle/detector DM은 여전히 Open |

## 6. 남은 것

이 forward model이 새로 닫은 것은 background/growth 계산이다. 아직 닫지 않은 것은 다음이다.

| 병목 | 필요한 다음 수식 |
|---|---|
| DESI likelihood | official source 자동 동기화, nuisance/systematic 처리, 다른 likelihood convention cross-check |
| \(H_0\) tension | \(r_d\), \(\theta_*\), local distance ladder readout |
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

> CE 성분비는 최신 CMB 계열의 현재 밀도비와 잘 맞고, 그 값을 background/growth forward model의 boundary condition으로 쓸 수 있다. 그러나 DESI/Euclid/LZ와 직접 경쟁하려면 이제 각 관측량의 likelihood와 dark matter microphysics를 추가해야 한다.
