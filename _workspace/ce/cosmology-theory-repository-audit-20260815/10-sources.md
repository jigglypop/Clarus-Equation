Status: COMPLETE

# 10 — 우주론 관측·출처·likelihood 감사

- Lane: `physics-sourcer`
- 감사 기준일 / 웹 접근일: **2026-08-15 (Asia/Seoul)**
- 코드 수정: **없음**
- 판정 원칙: 같은 모형·같은 데이터 조합·같은 pivot에서만 숫자를 직접 비교했다. 공동 posterior가 필요한 파생량은 공개 chain/covariance가 없으면 `UNVERIFIED`로 남겼다.
- 요약: DESI DR2의 13점 mean/covariance 자체는 맞지만, 저장소의 별도 `DESI DR2+CMB` 요약값 하나와 ACT/SPT 혼합 기준선 하나는 결론에 영향을 주는 P0이다. 현재 preregistration에는 독립 future holdout이 없다.

## 1. 판정표

| Evidence ID | 등급 | 저장소 주장 / 위치 | 1차·공식 출처 대조 | 수치 영향 | 판정 |
|---|---:|---|---|---|---|
| SRC-P0-01 | **P0** | `tests/scorecard.py:195-205`: `DESI DR2+CMB Omega_Lambda=0.693+/-0.005`, CE offset `-0.78 sigma` | DESI DR2 Results II의 flat ΛCDM `DESI+CMB`는 `Omega_m=0.3027+/-0.0036`, `H0=68.17+/-0.28`, corr `-0.975`이다. 따라서 같은 표기 정밀도에서 `Omega_Lambda=0.6973+/-0.0036`이다. [SRC-DESI-II] | `(0.6891-0.693)/0.005=-0.78`에서 `(0.6891-0.6973)/0.0036=-2.28 sigma`로 바뀐다. | **잘못된 DR2 인용이 결론을 바꿈.** `0.693+/-0.005`의 DR2 provenance는 확인되지 않았다. |
| SRC-P0-02 | **P0** | `examples/physics/cosmology_ratio_audit.py:118-122`: `Planck_ACT_SPT_combined=(0.02228,0.1195,68.43)` | 단일 논문/likelihood/posterior로 식별되지 않는다. `H0=68.43+/-0.27`은 P-ACT+DESI **DR2**에서 확인되지만, `omega_b=0.0226+/-0.0001`, `omega_c=0.118+/-0.001`, `H0=68.22+/-0.36`은 P-ACT+lensing+DESI **DR1** 조합이다. 현재 수정된 CMB-SPA는 `(0.022398+/-0.000095,0.12028+/-0.00094,67.19+/-0.38)`. [SRC-ACT] [SRC-SPT] [SRC-SPT-LAMBDA] | 혼합 행의 H0는 현재 CMB-SPA보다 `3.26 sigma`; `omega_b`, `omega_c`는 각각 `-1.24`, `-0.83 sigma`. | **출처가 성립하지 않는 hybrid baseline.** 4% 상대오차 boolean은 우연히 유지되더라도 관측 기준선으로 사용할 수 없다. |
| SRC-P1-01 | **P1** | `tests/run_validation.py:37,72`: `OMEGA_B_OBS=0.04865`, 고정 `sigma=0.00005` | Planck가 직접 보고한 양은 `omega_b=Omega_b h^2=0.02237+/-0.00015`와 `H0=67.36+/-0.54`이다. `Omega_b` 오차에는 둘의 chain covariance가 필요하다. 주변화 중심값만 변환하면 `Omega_b≈0.04930`; 상관을 무시한 설명용 오차는 `≈0.00086`으로 저장소 sigma보다 약 17배 크다. [SRC-PLANCK] | bootstrap 해 `q=0.0486466333`의 offset은 저장소에서 `-0.067 sigma`; 상관 무시 근사에서는 `-0.76 sigma`. 정확한 값은 Planck chain 없이는 확정 불가. | `0.04865+/-0.00005`를 Planck 관측으로 식별할 수 없음. 수학적 고정점 검증과 관측 일치 판정을 분리해야 한다. |
| SRC-P1-02 | **P1** | `ce_residual_forward_model.py:1077-1126`: DESI DR2 13점과 full covariance; source 문자열만 있음 | 13점 순서와 값, 13x13 covariance는 DESI가 안내하는 public Cobaya DR2 Gaussian BAO likelihood와 일치한다. 별도 engineering script는 commit과 SHA-256까지 pin한다. [SRC-DESI-DATA] [SRC-COBAYA-DR2] [SRC-DESI-II] | 현 기본값의 full-cov 결과는 `chi2=37.1003/13`, `p=3.996e-4`; 외부 scale 하나를 맞추면 `q=0.986477`, `chi2=12.6083/12`, `p=0.3981`. | **숫자 VERIFIED, provenance 경로 P1.** 실행 주 경로도 commit/hash/paper DOI를 노출해야 한다. |
| SRC-P1-03 | **P1** | `cosmology_discrimination_gates.py:193-198`: `Omega_DM/Omega_Lambda=0.3854+/-0.004`(Planck), `0.373+/-0.010`(DESI DR2+CMB) | 어느 값도 논문 표나 versioned chain ID로 식별되지 않았다. 이 비율의 오차는 `Omega_c`, `h`, `Omega_Lambda`의 joint posterior covariance가 필요하다. DESI DR2 flat central values만으로 정확한 DM/DE 비율과 sigma를 재구성할 수 없다. | gate separation을 재계산할 수 없음. 단순 주변오차 조합은 허용하지 않았다. | **UNVERIFIED.** chain·모형·중성미자 convention·derived-parameter 코드를 pin하기 전 관측 gate가 아니다. |
| SRC-P1-04 | **P1** | `cosmology_ratio_audit.py:109-137`: CMB 압축 tuple 네 개에 오차·공분산·likelihood ID 없음 | Planck 행은 반올림 snapshot, ACT 행은 DR1 조합, SPT 행은 수정된 2026 v2 central values로 식별된다. hybrid 행만 식별 불가하다. SPT 공식 배포처는 과거 CMB-SPA가 Planck nuisance prior를 다중 계산해 0.2–0.3 sigma 이동했고 수정 chain을 재배포했다고 명시한다. [SRC-PLANCK] [SRC-ACT] [SRC-SPT-LAMBDA] | 상대오차 4% 판정은 posterior 유의도가 아니며 dataset 간 상관도 반영하지 않는다. | identifiable한 세 행도 release/likelihood/error/covariance를 함께 pin해야 한다. |
| SRC-P1-05 | **P1** | `hubble_tension.py:347-370`: SH0ES target `73.04`, observed gap `~5.6`; 관측 오차 미전파 | `73.04+/-1.04`는 유효한 SH0ES 2022 snapshot이다. 현재 4-anchor SH0ES update는 `73.17+/-0.86`; 현재 corrected CMB-SPA는 `67.19+/-0.38`. [SRC-SH0ES22] [SRC-SH0ES24] [SRC-SPT] | 현재 gap은 `5.98+/-0.94`, 즉 `6.36 sigma`. 코드의 model gap `5.5595`는 현재 중심 gap보다 `0.4205` 낮아 combined observational error의 `0.45 sigma`. | 값 자체는 역사적 snapshot으로 VERIFIED. “observed ~5.6”과 sigma 없는 closure 표기는 stale/incomplete. |
| SRC-P1-06 | **P1** | `docs/3_상수/10_*`, `11_TDCOSMO_*`, `12_*`: source manifest, posterior/covariance adapter가 없음 | 현재 TDCOSMO milestone은 8 time-delay quasars+Pantheon+ flat ΛCDM에서 `H0=71.6^(+3.9)_(-3.3)`; SLACS+SL2S를 더하면 4.6% precision이다. 2026 단일 렌즈 결과는 DESI DR2 `Omega_m` prior 아래 `73.2^(+4.8)_(-4.7)`이며 다음 hierarchical analysis용이다. [SRC-TDCOSMO25] [SRC-TDCOSMO26] | 저장소에는 ingest할 posterior/chain/checksum이 없으므로 source-role 분해나 covariance gate를 재현할 수 없다. | 문서가 `[미완성]`으로 표시한 것은 정직함. 관측 closure 승격은 불가. |
| SRC-P1-07 | **P1** | `cosmological_constant_holographic_gate.py:16,29,85`: “observed 2.24 meV”, CE `2.2412 meV`, `<0.2%` 일치 | `rho_Lambda^(1/4)`은 독립 직접 관측값이 아니라 `H0`와 `Omega_Lambda`에서 유도된다. source/error/covariance가 없다. Planck 주변오차를 상관 없이 1차 전파해도 상대오차가 약 `0.48%`이며, 선형 합은 `0.67%`; 정확값은 chain이 필요하다. [SRC-PLANCK] | 주장한 0.054% 중심값 일치는 관측 오차보다 훨씬 작고 독립 검증이 아니다. | source-free “observed” 상수와 `<0.2%` 정밀도 주장을 사용하지 말아야 한다. |
| SRC-P1-08 | **P1** | `ce_residual_forward_model.py:55,1526`: `sigma8=0.811` 외부 입력으로 `S8=0.825734` | Planck `sigma8=0.8111+/-0.0060`은 유효한 외부 snapshot. 그러나 CE S8은 그 입력을 재사용한 파생량이다. 최신 KiDS-Legacy cosmic shear는 `S8=0.815^(+0.016)_(-0.021)`. [SRC-PLANCK] [SRC-KIDS] | 중심 차이는 KiDS 상방 오차로 약 `0.67 sigma`, 하지만 full likelihood 없이 근사일 뿐이다. | 저장소에 독립 LSS likelihood/covariance가 없으므로 S8은 CE 고유 예측 검증이 아니다. |
| SRC-P1-09 | **P1** | `scorecard.py:261-265`: `w0=-0.776+/-0.034`를 vague “Planck-family” snapshot으로 사용 | 정확한 paper/likelihood/chain을 식별하지 못했다. 현재 DESI DR2+CMB+Pantheon+ w0waCDM은 `w0=-0.838+/-0.055`, `wa=-0.62^(+0.22)_(-0.19)`. [SRC-DESI-II] | CE `w0=-0.769`은 w0 주변오차만 쓰면 `+1.25 sigma`, 그러나 wa와의 covariance를 버린 1차원 비교는 model fit이 아니다. | 구 snapshot도 exact provenance가 필요하고, current 비교는 2D posterior로 해야 한다. |
| SRC-P2-01 | **P2** | `scorecard.py:173-205`: “single dataset” 설명과 달리 `omega_b=0.02237`/`Omega_Lambda=0.6847`은 Planck-only, `omega_c=0.11933`은 Planck+BAO | 두 숫자군 모두 Planck 논문에서 유효하다. Planck+BAO에는 `omega_b=0.02242+/-0.00014`, `omega_c=0.11933+/-0.00091`, `Omega_Lambda=0.6889+/-0.0056`; Planck TTTEEE+lowE+lensing에는 `0.02237`, `0.1200`, `0.6847`이다. [SRC-PLANCK] | 현재 개별 score는 크게 바뀌지 않지만 joint dataset이라고 말할 수 없다. | 각 행에 정확한 likelihood 조합을 표기하거나 한 조합으로 통일할 것. |
| SRC-P2-02 | **P2** | `primordial_spectrum_readout_gate.py:28-32`: `ln(10^10 As)=3.044+/-0.014`, `As*1e9=2.099+/-0.029`; inflation 문서 `ns=0.96772`, `r=0.00397`, BK18 `r<0.036` | Planck 변환은 `exp(3.044)/10=2.098903`, 선형 sigma `0.02938`로 정확하다. BK18 limit도 정확하다. 최신 CMB 조합은 `ns=0.9682+/-0.0032`, `r<0.034`; CMB+DESI는 `ns=0.9728+/-0.0029`. [SRC-PLANCK] [SRC-BK18] [SRC-INFLATION26] | CE ns는 Planck에서 `+0.67 sigma`, 최신 CMB에서 `-0.15 sigma`, CMB+DESI에서 `-1.75 sigma`; r은 두 limit 아래. | 역사적 snapshot은 VERIFIED. “현재 제약”을 말할 때만 update 필요. |
| SRC-P2-03 | **P2** | `ce_residual_forward_model.py:30,33`: `Tcmb=2.7255 K`, `Neff=3.044` | `Tcmb=2.72548+/-0.00057 K`, SM benchmark `Neff=3.0440+/-0.0002`로 중앙값이 맞다. [SRC-TCMB] [SRC-NEFF] | 수치 영향 없음. | source ID와 `Neff`가 관측값이 아니라 SM 계산 가정임을 명시할 것. |

## 2. 저장소 기준선 inventory

### 2.1 Planck / CMB / BAO / late-time

| 저장소 입력·관측량 | 값 | authoritative identity | 상태 |
|---|---:|---|---|
| `H0` | `67.4` | Planck 2018 TTTEEE+lowE+lensing `67.36+/-0.54` | VERIFIED, 반올림 |
| `rd` | `147.09 Mpc` | 같은 Planck 조합 `147.09+/-0.26 Mpc` | VERIFIED external calibration; CE 유도 아님 |
| `sigma8` | `0.811` | 같은 Planck 조합 `0.8111+/-0.0060` | VERIFIED external input |
| `omega_b h2` | `0.02237+/-0.00015` | Planck TTTEEE+lowE+lensing | VERIFIED |
| `omega_c h2` | `0.11933+/-0.00091` | Planck **+BAO** | VERIFIED 값, dataset label 혼합 |
| `Omega_Lambda` | `0.6847+/-0.0073` | Planck TTTEEE+lowE+lensing | VERIFIED |
| `Omega_Lambda` 대안 | `0.6889+/-0.0056` | Planck+BAO | VERIFIED |
| `S8` | Planck `0.832+/-0.013` | Planck TTTEEE+lowE+lensing | VERIFIED reference; CE forward value는 외부 sigma8 재사용 |
| `ACT_DR6_DESI_reported` | `(0.0226,0.118,68.22)` | P-ACT+ACT/Planck lensing+DESI **DR1** | central values VERIFIED; label/오차 누락 |
| `SPT3G_CMBSPA` | `(0.022398,0.12028,67.19)` | corrected CMB-SPA v2, errors `(0.000095,0.00094,0.38)` | VERIFIED current central values; fixed-chain provenance 누락 |
| `Planck_ACT_SPT_combined` | `(0.02228,0.1195,68.43)` | single posterior로 식별 불가 | **INVALID HYBRID** |
| SH0ES | `73.04` | SH0ES 2022 `73.04+/-1.04` | valid old snapshot; current 4-anchor `73.17+/-0.86` |
| TDCOSMO | 값 없음 | TDCOSMO 2025 `71.6^(+3.9)_(-3.3)` | 저장소 ingest/chain 없음, `[미완성]` |
| KiDS / S8 | likelihood 없음 | KiDS-Legacy `0.815^(+0.016)_(-0.021)` | 저장소 독립 gate 없음 |

### 2.2 primordial spectrum / inflation

| 항목 | 저장소 | 확인된 source snapshot | 상태 |
|---|---:|---:|---|
| `ln(10^10 As)` | `3.044+/-0.014` | Planck 2018 TTTEEE+lowE+lensing | VERIFIED |
| `As*1e9` | `2.099+/-0.029` | 위 로그값의 올바른 변환 | VERIFIED; 독립 관측 아님 |
| `ns` reference | `0.9649+/-0.0042` | Planck 2018 | VERIFIED |
| CE inflation `ns` | `0.96772` | model output | Planck `0.67 sigma`; current CMB+DESI `-1.75 sigma` |
| CE inflation `r` | `0.00397` | model output | BK18 `<0.036`, current combo `<0.034` 아래 |

### 2.3 기타 외부 상수와 derived observables

| 항목 | 저장소 | 출처/역할 | 상태 |
|---|---:|---|---|
| `Tcmb` | `2.7255 K` | FIRAS/문헌 결합 `2.72548+/-0.00057 K` | VERIFIED |
| `Neff` | `3.044` | SM neutrino-decoupling 계산 `3.0440+/-0.0002` | VERIFIED **model assumption**, 관측 측정 아님 |
| `rho_Lambda^(1/4)` | `observed 2.24 meV` | H0·OmegaLambda에서 유도되는 종속량 | source/error 없음; 독립 검증 아님 |
| `Omega_DM/Omega_Lambda` | `0.3854+/-0.004`, `0.373+/-0.010` | chain-derived quantity여야 함 | **UNVERIFIED** |

## 3. DESI DR2 13점 provenance와 수치 영향

### 3.1 mean vector / covariance

저장소 row order는 아래와 같다. 단위는 모두 sound horizon으로 나눈 무차원 거리다.

| z | observable | value |
|---:|---|---:|
| 0.295 | `DV/rd` | 7.94167639 |
| 0.510 | `DM/rd`, `DH/rd` | 13.58758434, 21.86294686 |
| 0.706 | `DM/rd`, `DH/rd` | 17.35069094, 19.45534918 |
| 0.934 | `DM/rd`, `DH/rd` | 21.57563956, 17.64149464 |
| 1.321 | `DM/rd`, `DH/rd` | 27.60085612, 14.17602155 |
| 1.484 | `DM/rd`, `DH/rd` | 30.51190063, 12.81699964 |
| 2.330 | `DH/rd`, `DM/rd` | 8.631545674846294, 38.988973961958784 |

이 vector와 `ce_residual_forward_model.py:1077-1090`의 13x13 covariance는 public Cobaya `desi_bao_dr2` Gaussian likelihood와 일치한다. 저장소 내부의 더 강한 pin은 다음과 같다.

| provenance field | pinned value |
|---|---|
| upstream repository | `CobayaSampler/bao_data` |
| commit | `bb0c1c9009dc76d1391300e169e8df38fd1096db` |
| mean file | `desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt` |
| mean bytes / SHA-256 | `472` / `9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585` |
| covariance file | `desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt` |
| covariance bytes / SHA-256 | `2547` / `252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509` |
| embedded dataset digest | `e8f24d5ef0ce808f9c1f67d52fead2eede5cf9265e31aed41db94ecb30c324f5` |
| paper / supplement | DESI DR2 Results II; Zenodo DOI `10.5281/zenodo.16644576` |

단, 이 pin은 `docs/2_경로적분과_응용/improvement_loop_engineering.py:39-60`에 있고 실제 BAO dataset 객체는 generic source 문자열만 출력한다. `docs/9_등호이전/05o_*:190-200`은 오히려 commit/acquisition 고정을 미완성으로 적는다. 수치 fixture는 강하지만 provenance가 한 경로로 합쳐져 있지 않다.

### 3.2 저장소 실행 결과

읽기 전용 실행:

```text
python examples/physics/ce_residual_forward_model.py --bao-dataset desi-dr2-all
```

| 모드 | chi2 / dof | p | scale 해석 |
|---|---:|---:|---|
| fixed `H0=67.4`, `rd=147.09` | 37.100260857 / 13 | 0.0003995733 | `REJECT` |
| scale 1개 diagnostic fit | 12.608346862 / 12 | 0.3981382 | `q=0.986476933`; fixed H0에서 `rd=149.1064 Mpc`, fixed rd에서 `H0=68.32395` |

두 번째 행은 외부 `H0*rd` scale을 데이터에 맞춘 진단이며 CE의 사전 예측이 아니다.

### 3.3 2026 DESI Results IV는 drop-in 교체도 future holdout도 아님

DESI Results IV는 같은 DR2 Ly-alpha 자료에 AP/full-shape 정보를 더해 z=2.33에서 `DH/rd=8.600+/-0.066`, `DM/rd=39.32+/-0.33`을 준다. 기존 pure-BAO row와 component-wise 비교하면 각각 `-0.48 sigma`, `+1.00 sigma` 이동이다. 그러나 likelihood와 covariance가 달라 기존 13점 vector의 마지막 두 원소만 바꾸면 안 된다. [SRC-DESI-IV]

공식 DESI 페이지는 이 결과가 **2026-07-30** 공개됐다고 명시하고 arXiv v1도 **2026-07-29**이다. preregistration freeze `2026-08-02`보다 이르며, 기존에 본 DR2와 같은 관측 자료를 확장한 결과다. 따라서 `experiments/preregistration/cosmology_future_holdout_v2.json:207-227`의 “freeze 이후 최초 qualifying release” 조건을 충족하지 않는다. 현재 manifest가 `future_holdout.assignment_status=unassigned`인 것은 올바르며, **현재 checkout에는 독립 평가 가능한 future cosmology holdout이 없다.**

## 4. 공분산·독립성 경계

| 비교 | 필요한 정보 | 저장소 상태 | 허용 판정 |
|---|---|---|---|
| `Omega_b`와 bootstrap q | Planck `omega_b-h` joint chain | 없음 | 중심값 설명만 가능; repo sigma 금지 |
| `Omega_DM/Omega_Lambda` | `omega_c,h,Omega_Lambda` joint chain와 neutrino convention | 없음 | `UNVERIFIED` |
| DESI `w0,wa`와 CE | 2D posterior/covariance, 동일 CPL branch | 없음 | w0 marginal 참고만 가능 |
| S8 | 독립 LSS likelihood와 nuisance marginalization | 없음 | 외부 sigma8 재사용 readout일 뿐 |
| `rho_Lambda^(1/4)` | H0-OmegaLambda covariance | 없음 | 독립 observation으로 세지 않음 |
| TDCOSMO source-role | lens posterior/chain, sample labels, checksums | 없음 | 문서의 `[미완성]` 유지 |
| DESI 13점 BAO | full 13x13 covariance | 있음 | 현재 fixed-input chi2는 재현 가능 |

## 5. source ledger

모든 링크의 접근일은 2026-08-15이다.

- **[SRC-PLANCK]** Planck Collaboration, [Planck 2018 results VI: Cosmological parameters](https://arxiv.org/abs/1807.06209), v4 / A&A 641 A6. Table/chain 조합을 구분해 사용했다.
- **[SRC-DESI-II]** DESI Collaboration, [DESI DR2 Results II](https://arxiv.org/html/2503.14738), v3 / PRD 112 083515. Table 4 mean vector, Table 5 cosmological parameters, public-data section.
- **[SRC-DESI-DATA]** DESI 공식 데이터 문서, [DR2 Publications and cosmology chains](https://data.desi.lbl.gov/doc/papers/dr2/).
- **[SRC-COBAYA-DR2]** DESI가 연결하는 public likelihood, [CobayaSampler/bao_data `desi_bao_dr2`](https://github.com/CobayaSampler/bao_data/tree/bb0c1c9009dc76d1391300e169e8df38fd1096db/desi_bao_dr2). 저장소 pin commit을 링크했다.
- **[SRC-DESI-SUPP]** DESI DR2 Results II supplementary material, [Zenodo 10.5281/zenodo.16644576](https://doi.org/10.5281/zenodo.16644576).
- **[SRC-DESI-IV]** DESI Collaboration, [DESI DR2 Results IV: Ly-alpha AP](https://arxiv.org/abs/2607.27410), v3, 최초 제출 2026-07-29.
- **[SRC-ACT]** ACT Collaboration, [ACT DR6 Power Spectra, Likelihoods and ΛCDM Parameters](https://arxiv.org/abs/2503.14452), v2 / JCAP accepted.
- **[SRC-SPT]** SPT-3G Collaboration, [SPT-3G D1 spectra and cosmology](https://arxiv.org/abs/2506.20707v2), corrected v2 2026-04-01 / PRD.
- **[SRC-SPT-LAMBDA]** NASA LAMBDA, [SPT-3G D1 official likelihood and corrected chains](https://lambda.gsfc.nasa.gov/product/spt/spt3g_d1_bandp_liklyhood_info.html), bug-fix notice 2026-04-01.
- **[SRC-SH0ES22]** Riess et al., [SH0ES comprehensive HST distance ladder](https://arxiv.org/abs/2112.04510), `73.04+/-1.04`.
- **[SRC-SH0ES24]** Breuval et al., [SMC fourth anchor for SH0ES](https://arxiv.org/abs/2404.08038), `73.17+/-0.86`; journal DOI [10.3847/1538-4357/ad630e](https://doi.org/10.3847/1538-4357/ad630e).
- **[SRC-SH0ES-JWST]** Riess et al., [JWST validation of HST distance measurements](https://arxiv.org/abs/2408.11770), full HST set `73.2+/-0.9`, JWST subset `72.6+/-2.0`.
- **[SRC-TDCOSMO25]** TDCOSMO Collaboration, [TDCOSMO 2025 milestone](https://arxiv.org/abs/2506.03023), v4 / A&A 704 A63.
- **[SRC-TDCOSMO26]** Sheu et al., [TDCOSMO XXV single-lens analysis](https://arxiv.org/abs/2604.14145), v2.
- **[SRC-KIDS]** KiDS Collaboration, [KiDS-Legacy cosmic shear](https://arxiv.org/abs/2503.19441), v2 / A&A 703 A158.
- **[SRC-BK18]** BICEP/Keck Collaboration, [BK18 primordial gravitational-wave constraint](https://arxiv.org/abs/2110.00483), PRL 127 151301.
- **[SRC-INFLATION26]** Balkenhol et al., [Inflation at the End of 2025](https://arxiv.org/abs/2512.10613), v2 / OJA 9 (2026).
- **[SRC-TCMB]** Fixsen, [The Temperature of the Cosmic Microwave Background](https://arxiv.org/abs/0911.1955), `2.72548+/-0.00057 K`.
- **[SRC-NEFF]** Bennett et al., [Precision Standard-Model Neff](https://arxiv.org/abs/2012.02726), `3.0440+/-0.0002`.

## 6. 상위 handoff

1. **P0:** scorecard의 `DESI DR2+CMB Omega_Lambda=0.693+/-0.005`를 활성 관측 주장으로 사용하지 말 것. 공식 DR2 flat ΛCDM은 `0.6973+/-0.0036`이며 CE offset이 `-0.78`에서 `-2.28 sigma`로 변한다.
2. **P0:** `Planck_ACT_SPT_combined` 행을 삭제하거나 실제 하나의 versioned chain에서 다시 추출할 것. 현재 tuple은 식별 불가능한 hybrid다.
3. **P1:** bootstrap `OMEGA_B_OBS`의 `0.00005` sigma, DM/DE ratio 두 개, rho-Lambda “observed” 값, TDCOSMO source-role은 provenance/covariance가 없어 관측 판정으로 승격할 수 없다.
4. **P1:** DESI 13점 수치와 covariance는 보존 가능하다. 다만 실행 dataset 객체에 이미 존재하는 commit/hash/paper/supplement provenance를 연결해야 한다.
5. **Holdout:** DESI Results IV는 freeze 전 공개됐고 같은 DR2 자료의 확장 likelihood다. manifest를 억지로 assign하지 말 것. 독립 future holdout은 아직 없다.
