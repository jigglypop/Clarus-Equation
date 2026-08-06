# CE 관측 기준선과 입력 Provenance — 2026-08-06

## 1. 목적

같은 CE 수치를 서로 다른 데이터 조합·오차·parameter definition과 비교해
서로 다른 $\sigma$를 만드는 일을 막는다. 이 파일은 관측값을 이론의
공리로 만들지 않고, 비교에 쓰는 snapshot과 역할을 고정한다.

## 2. 결합상수 기준선

| 양 | canonical definition | snapshot | 역할 |
|---|---|---|---|
| $\alpha_s(M_Z)$ | $\overline{\rm MS}$, $n_f=5$, $\mu=M_Z$ | PDG 2026 QCD review가 채택한 PDG 2025 world average $0.1180\pm0.0009$ | Track A 입력 또는 Track B holdout 중 하나만 선택 |
| $\sin^2\theta_W$ | 반드시 scheme을 suffix로 표기 | effective leptonic angle $\bar s_\ell^2=0.23154\pm0.00006$; $\overline{\rm MS}$ 값 $\hat s_Z^2=0.23122\pm0.00006$ | 두 scheme의 수치 혼용 금지 |
| $\alpha_{em}(0)$ | Thomson limit | CODATA/PDG reference | CE 기하식 비교 대상 |
| $\alpha_{em}(M_Z)$ | vacuum polarization을 포함한 running coupling | PDG electroweak fit provenance | Track B 분포 입력; manifest의 $1/127.95$는 두 근 산술만 확인하는 central-value 진단 |

주요 출처:

- [PDG 2026 QCD review, Eq. (9.25)](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-qcd.pdf)
- [PDG 2026 Standard Model/electroweak review, Table 10.5](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-standard-model.pdf)

$\alpha_{em}(M_Z)$의 uncertainty에는 hadronic vacuum polarization이
포함되며 $\alpha_s$와의 상관도 생길 수 있다. Track B에서는 central
value뿐 아니라 이 covariance를 전달한다. 따라서 현 manifest의
\(1/127.95\)로 얻은 두 근은 branch 존재 검산이며 정밀 물리 추론이 아니다.

## 3. 우주론 기준선

Planck 2018 base-$\Lambda$CDM TT,TE,EE+lowE+lensing 계열의 대표 물리
밀도 snapshot을 기준 단위로 둔다.

| 양 | 대표 snapshot | 주의 |
|---|---:|---|
| $\omega_b=\Omega_bh^2$ | $0.0224\pm0.0001$ 수준 | $\Omega_b$와 직접 비교 금지 |
| $\omega_c=\Omega_ch^2$ | $0.120\pm0.001$ 수준 | CE의 `DM` 정의와 동일한지 확인 |
| $H_0$ | $67.4\pm0.5\ {\rm km\,s^{-1}Mpc^{-1}}$ | $h=H_0/100$ |
| $\Omega_m$ | $0.315\pm0.007$ | 같은 chain의 파생량 사용 |
| $n_s$ | $0.965\pm0.004$ 수준 | pivot scale 표기 |

출처: [Planck 2018 VI](https://doi.org/10.1051/0004-6361/201833910).

$\Omega_b$, $\Omega_c$, $\Omega_\Lambda$를 서로 다른 데이터 조합에서
하나씩 골라 합하지 않는다. 한 chain 또는 공개 covariance sample에서
동시에 계산한다.

DESI DR2 BAO 비교에는
[DESI DR2 cosmology paper](https://arxiv.org/abs/2503.14738)와 공개
covariance를 사용한다. 단일 marginal $w_0$ 차이와 다변량 BAO
$\chi^2$를 서로 대체하지 않는다. 같은 DR2로 scale을 fit한 결과는 DR2
holdout 예측으로 기록하지 않는다.

## 4. Higgs·portal 기준선

| 양 | 최신 snapshot | 문서 계약 |
|---|---:|---|
| Higgs mass | \(125.11\pm0.09_{\rm stat}\pm0.06_{\rm syst}\,\mathrm{GeV}\) | PDG 2026 review가 인용한 가장 정밀한 ATLAS Run-1+Run-2 조합 |
| SM Higgs width at 125 GeV | \(4.10\,\mathrm{MeV}\) | portal tree-width의 SM denominator |
| direct invisible branching limit | ATLAS observed \(\mathrm{BR}_{\rm inv}<0.107\) at 95% CL | PDG 2026 review가 열거한 현재 직접 Run-2 결과 중 가장 강한 한계; CMS \(<0.15\)와 혼합하지 않음 |

출처:
[PDG 2026 Higgs review, §§11.3.7, 11.3.8, 11.4.3](https://pdg.lbl.gov/2026/reviews/rpp2026-rev-higgs-boson.pdf).
이 한계는 SM Higgs production 가정을 포함한 direct-search constraint다.
다른 coupling fit이나 undetected-width 한계와 같은 값으로 취급하지 않는다.

## 5. 저에너지 정밀 기준선

| 양 | 최신 snapshot | 문서 계약 |
|---|---:|---|
| \(\Delta a_\mu^{\rm exp-SM}\) | \(38(63)\times10^{-11}\) | WP2025 lattice-based consensus와 Fermilab final average를 사용; WP2020 \(249(48)\times10^{-11}\)은 역사 비교만 |
| proton rms charge radius | CODATA 2022 \(r_p=0.84075(64)\,\mathrm{fm}\) | 옛 CODATA 2010--CREMA 차 \(0.0587\,\mathrm{fm}^2\)를 active fit target으로 쓰지 않음 |
| 2026 atomic-H 2S--6P | \(r_p=0.8406(15)\,\mathrm{fm}\) | 독립 최신 측정; CODATA 권고 평균을 대체하지 않고 보조 validation으로 사용 |

출처:

- [Muon \(g-2\) Theory Initiative WP2025](https://muon-gm2-theory.illinois.edu/white-paper-25/)
- [Fermilab final \(g-2\) result](https://lss.fnal.gov/archive/2025/pub/fermilab-pub-25-0364-ppd.pdf)
- [CODATA 2022 recommended values](https://physics.nist.gov/cuu/pdf/JPCRD2022CODATA.pdf)
- [2026 atomic-hydrogen 2S--6P result](https://www.nature.com/articles/s41586-026-10124-3)

## 6. 역할 태그

모든 관측 행은 다음 중 하나를 가져야 한다.

| 태그 | 의미 |
|---|---|
| `REFERENCE` | 단위·정의 확인; score에 넣지 않음 |
| `INPUT` | 모형 계산 전에 고정 |
| `CALIBRATION` | 계수 또는 branch 선택에 사용 |
| `VALIDATION` | calibration 후 보지 않은 비교값 |
| `HOLDOUT` | 사전등록 뒤 공개·배정된 데이터 |

한 데이터셋은 같은 revision에서 두 역할을 가질 수 없다.

## 7. $\sigma$와 likelihood 규칙

단일 독립 Gaussian 양에만

$$
z=\frac{y_{\rm CE}-y_{\rm obs}}
{\sqrt{\sigma_{\rm CE}^2+\sigma_{\rm obs}^2}}
$$

를 쓴다. 상관된 벡터에는

$$
\chi^2=(\boldsymbol y_{\rm CE}-\boldsymbol y_{\rm obs})^T
C^{-1}(\boldsymbol y_{\rm CE}-\boldsymbol y_{\rm obs})
$$

를 사용한다. singular/estimated covariance에는 공급 논문의 likelihood
prescription을 따른다. parameter fitting을 했다면 dof와 유효 parameter
수를 차감한다.

## 8. 버전 규칙

관측 snapshot을 바꾸면 다음을 한 commit/revision에서 함께 바꾼다.

1. 이 파일의 값·출처·접근일
2. `상수.md`의 비교표
3. scorecard manifest
4. covariance/hash와 실행 결과
5. `PROOF_STATUS_MATRIX.md`의 판정

중심값만 새 release로 바꾸고 과거 오차나 covariance를 유지하지 않는다.
