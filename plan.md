# CE 우주론 다음 연구 플랜

## 0. 현재 결론

지금 우주론 쪽 핵심 개선은 숫자를 억지로 맞춘 것이 아니라, 관측량의 읽기 방식을 분류하기 시작했다는 점이다.

CE 우주론 관측량은 적어도 다음 다섯 종류로 나눠야 한다.

| 분류 | 읽는 것 | 대표 관측량 | 현재 상태 |
|---|---|---|---|
| 고정점 상태량 | 자기재귀 고정점의 종점 값 | \(\Omega_b\), \(\Omega_\Lambda\), \(\Omega_{DM}\), \(H_0t_0\), \(n_s\) | scorecard 채점 |
| 내부 총응답 | 고정점이 새 조건에 재조정되는 전체 민감도 | 수축률, 안정성 | 수학 구조 |
| 투영 잔차 구동력 | 전이 중 외부 관측 채널에 남는 잔차 원천 | \(A_s\) 후보 | `Open` |
| freeze-out 잔차 | 동결 시점에서 멈춘 전이값 | \(\eta\), \(T_{\rm CMB}\) | 현상론적 채점 |
| late-time running | 후기 우주에서 관측 추출값을 바꾸는 잔차 흐름 | \(H_0\), \(S_8\), 성장률 | 부분 해결 |

가장 중요한 새 후보는 `A3c`다.

> 전이 스칼라 관측량은 고정점의 전체 응답이 아니라, 관측 채널로 투영된 잔차 구동력을 읽는다.

이 원리 후보는 \(A_s\) raw 실패를 크게 줄인다. 정확 총응답으로 읽으면 \(A_s\)는 관측보다 크게 실패하지만, 유효 기하 정규화된 잔차 구동력으로 읽으면 \(2.104\times10^{-9}\), 관측 대비 약 \(+0.18\%\)까지 내려간다.

단, 아직 `Open`이다. 숫자는 강하지만 선택 원리가 독립적으로 닫힌 것은 아니다.

---

## 1. 지금 채택한 원칙

1. 숫자가 맞는 것과 원리가 닫힌 것은 분리한다.
2. `Open`은 실패가 아니라 비채점 재개방 항목이다.
3. `Open test`는 아직 관측 비교를 scorecard에 넣지 않는 미래 검증 손잡이다.
4. 어떤 값이 틀릴 때마다 새 필터를 붙이지 않는다.
5. 같은 읽기 규칙이 여러 관측량에 반복되어야 승격한다.

---

## 2. 이미 생긴 산출물

| 산출물 | 위치 | 역할 |
|---|---|---|
| 자기재귀성 대칭 장 | `docs/2_경로적분과_응용/14_자기재귀성_대칭.md` | \(d=0\) 항등 경계상, \(d=3\) 수축 고정상, A3c 후보 |
| 우주론 재귀 예측 카드 | `examples/physics/recursive_cosmology_predictions.py` | \(\Omega\), \(A_s\), \(\eta\), \(T_{\rm CMB}\), A3c 후보 계산 |
| 우주론 장 업데이트 | `docs/3_상수/7_우주론.md` | \(A_s\) 재유도, running/tensor 다음 검증 |
| scorecard 후보 출력 | `examples/physics/scorecard.py` | `Open` 비채점, `candidate_predictions` 출력 |
| 전이구간 업데이트 | `docs/2_경로적분과_응용/12_전이구간.md` | 예전 \(A_s\) 보정 폐기, A3c 후보로 재분류 |

---

## 3. 즉시 해야 할 정리

### 3.1 등급 체계 통일

`docs/상수.md`의 등급 설명에 다음을 명시한다.

| 등급 | 의미 |
|---|---|
| `Exact` | 정의 또는 순수 수학으로 닫힘 |
| `Selection` | 안정 분기, 유일성, 정규화 선택으로 닫힘 |
| `Bridge` | 코어 변수를 표준모형/우주론 관측량으로 옮김 |
| `Phenomenology` | 동결 시점, NLO 보정, 응용 닫힘이 필요 |
| `Open` | scorecard \(\chi^2\)에서 제외된 재개방 항목 |
| `Open test` | 아직 점수화하지 않는 미래 검증 예측 |

목표는 `Open = broken`이 아니라 `Open = non-scored candidate under test`로 읽히게 만드는 것이다.

### 3.2 scorecard 표시 정리

현재 `scorecard.py`는 `Open` 항목을 chi-square에서 제외하고 `o`로 표시한다. 이 상태를 문서에 명시한다.

---

## 4. 다음 수치 게이트

### Gate A: primitive spectrum card

원시 스펙트럼 네 항목을 한 카드로 묶는다.

| 관측량 | 상태 | 목적 |
|---|---|---|
| \(n_s\) | 채점됨 | \(N_e\) 계열 기준점 |
| \(A_s\) | `Open` | A3c 직접 검증 |
| \(\alpha_{\rm spec}\) | `Open test` | running 검증 |
| \(r_{\rm tensor}\) | `Open test` | B-mode 검증 |

현재 예측:

- \(n_s=0.96503475\)
- \(A_s^{A3c}=2.104\times10^{-9}\), 후보
- \(\alpha_{\rm spec}=-6.11\times10^{-4}\)
- \(r_{\rm tensor}=0.00367\)

성공 조건:

- \(A_s\) 후보가 다른 스펙트럼 항목과 같은 \(N_e\) 구조를 유지한다.
- running과 tensor ratio가 향후 관측 상한 또는 중심값과 충돌하지 않는다.

실패 조건:

- running 또는 \(r\)가 CE 예측과 구조적으로 맞지 않는다.
- \(A_s\)만 따로 맞고 다른 원시 스펙트럼 항목과 연결되지 않는다.

### Gate B: 암흑물질 정체

현재 CE는 \(\Omega_{DM}\) 양은 잘 준다. 하지만 정체는 아직 닫지 못했다.

필요한 예측:

- 국소 자유도 존재 여부
- 질량 또는 유효 스케일
- 표준모형과의 결합 세기
- 직접검출 단면적 또는 비검출 조건
- 은하 규모에서 core/cusp, missing satellite에 주는 효과

### Gate C: 구조 형성과 \(S_8\)

현재 CE는 \(S_8\) 텐션을 완화하지만 완전히 닫지는 않는다.

필요한 예측:

- \(\sigma_8(z)\)
- growth factor
- halo mass function
- JWST high-z 은하 수
- 초기 블랙홀 seed 조건

### Gate D: 초기조건과 엔트로피

\(d=0\) 자기재귀 항등 경계상을 초기 우주 조건으로 승격할 수 있는지 본다.

필요한 예측:

- 초기 엔트로피 스케일
- 곡률 초기값
- 요동 초기조건
- singularity 회피 조건
- e-fold 수와의 연결

### Gate E: CMB 대각 이상현상

잔차 투영 원리가 방향성 또는 low-\(\ell\) 이상현상을 설명할 수 있는지 본다.

필요한 예측:

- quadrupole suppression 크기
- hemispherical asymmetry amplitude
- preferred axis 통계
- cosmic variance 대비 유의성

---

## 5. 작업 순서

1. `docs/상수.md` 등급 체계 정리
2. `primitive_spectrum_prediction_card.py` 또는 기존 `recursive_cosmology_predictions.py` 내 독립 섹션 강화
3. `docs/3_상수/7_우주론.md`에 원시 스펙트럼 카드 표 추가
4. \(A_s\)를 scorecard 채점에 넣지 않고 `Open`으로 유지
5. running과 \(r\)를 `Open test`로 유지
6. 이후 암흑물질 게이트로 이동

---

## 6. 현재 판단

지금 가장 강한 실마리는 A3c다.

하지만 A3c는 아직 원리가 아니라 후보 원리다. 다음 단계에서 해야 할 일은 A3c가 \(A_s\) 하나만 살리는 장치인지, 원시 스펙트럼 전체를 묶는 실제 관측 원리인지 가르는 것이다.

따라서 다음 작업의 초점은 철학이 아니라 숫자 게이트다.

> 맞는 값은 채점하고, 후보는 비채점으로 남기고, 틀릴 수 있는 예측을 계속 추가한다.

---

## 7. 진행 로그

2026-05-06:

- 1번 완료: `docs/상수.md`를 6등급 체계로 정리했다.
- 2번 완료: `examples/physics/primitive_spectrum_prediction_card.py`를 추가했다.
- 3번 완료: `docs/3_상수/7_우주론.md`에 원시 스펙트럼 카드 표를 추가했다.
- 4번 유지: \(A_s\)는 raw 실패와 A3c 후보를 분리하고 `Open`으로 남겼다.
- 5번 유지: \(\alpha_{\rm spec}\), \(r_{\rm tensor}\)는 `Open test`로 남겼다.
- 다음 단계: 암흑물질 게이트로 이동해 \(\Omega_{DM}\) 값이 아니라 정체성, 질량/상호작용, 검출 가능성을 계산한다.

2026-05-06, Gate B 시작:

- `examples/physics/dark_matter_identity_gate.py`를 추가했다.
- `docs/3_상수/7_우주론.md`에 암흑물질 정체 게이트를 추가했다.
- 닫힌 결과: \(\Omega_{DM}=0.26227972\), \(\Omega_{DM}/\Omega_\Lambda=0.38062660\), \(\Omega_{DM}/\Omega_b=5.39151919\).
- 정체성 판정: CE 암흑물질은 우선 국소 WIMP가 아니라 클라루스장 진공 위의 집합적 요동 성분이다.
- 비검출 조건: 정확 \(Z_2\) 집합 가지에서는 \(\theta_{\rm mix}=0\), \(\sigma_{\rm recoil}^{\rm tree}=0\).
- 열린 브리지: \(m_\phi=m_p\delta^2=29.64757\,{\rm MeV}\)는 암흑물질 입자 질량 확정이 아니라 국소 스칼라 표현 후보이다.
- 음성 결과: \(\lambda_C=6.65575\,{\rm fm}\)라서 이 브리지 스칼라 하나로 은하 core/cusp를 직접 해결할 수 없다. 다음은 Gate C, 성장률과 구조 형성이다.

2026-05-06, Gate C 시작:

- `examples/physics/structure_growth_gate.py`를 추가했다.
- `docs/3_상수/7_우주론.md`에 구조 형성과 \(S_8\) 게이트를 추가했다.
- 성장식: \(D''+[2+d\ln H/d\ln a]D'-\frac32\Omega_m(a)D=0\).
- 관측량: \(S_8=\sigma_8\sqrt{\Omega_m/0.3}\), \(f\sigma_8=fD\sigma_8(0)\).
- 정적 밀도분할 가지: \(\Omega_m=0.31092644\), \(\sigma_8=0.78533433\), \(S_8=0.79950795\), \(\chi^2(f\sigma_8)=11.687/18\). LSS는 좋아지지만 Planck \(S_8\)와 \(-2.50\sigma\)라서 동시 폐쇄는 아님.
- 후기 \(H_0\) 가지: \(\Omega_m=0.343\), \(\sigma_8=0.77074116\), \(S_8=0.82412860\), \(\chi^2(f\sigma_8)=13.179/18\). LCDM 대비 \(\Delta\chi^2=-2.907\), KiDS 잔차 \(3.25\sigma\to2.91\sigma\).
- 결론: \(S_8\)는 부분 완화다. 다음은 halo mass function, JWST high-\(z\) 은하 수, 스케일 의존 집합 응축 성장식.

2026-05-06, Gate D 시작:

- `examples/physics/initial_conditions_entropy_gate.py`를 추가했다.
- `docs/3_상수/7_우주론.md`에 초기조건과 재귀 엔트로피 게이트를 추가했다.
- 재귀 엔트로피: \(\mathcal S_{\mathcal R}=-\log x=D(1-x)\).
- \(d=0\) 경계상: \(x_0=1\), \(\sigma_0=0\), \(\mathcal S_{\mathcal R,0}=0\).
- 물리 가지: \(x=0.04864672\), \(\sigma=0.95135328\), \(\mathcal S_{\mathcal R}=3.02317090\).
- 안정성: \(k=D_{\rm eff}x=0.15458752\), \(k^{20}=6.07\times10^{-17}\), \(10^{-60}\) 억압 반복수 \(n=74.00\).
- e-fold 연결: \(N_e=57.19965162\), \(\mathcal S_{\mathcal R}/N_e=0.05285296\).
- 결론: 무차원 초기 경계조건은 닫히지만, 절대 열역학 엔트로피, FLRW 곡률 초기값, 특이점 회피 정리는 `Open`.

2026-05-06, Gate E 시작:

- `examples/physics/cmb_anomaly_gate.py`를 추가했다.
- `docs/3_상수/7_우주론.md`에 CMB 대각 이상현상 게이트를 추가했다.
- A3c 잔차: \(Q_{\rm A3c}=(2/\pi)\sigma^{D_{\rm eff}/(D_{\rm eff}+1)}x(1-x)=0.02836622\).
- quadrupole power suppression 후보: \((2/\pi)^2=0.40528473\), 기하 정규화 포함 \(0.37567493\).
- quadrupole cosmic variance: \(\sqrt{2/5}=0.63245553\). 단독 pull은 \(-0.94\) to \(-0.99\sigma\)라 결정적이지 않다.
- hemispherical contrast 후보: \(2Q_{\rm A3c}/\sigma=0.05963341\), 보수적 \(2Q_{\rm A3c}=0.05673244\). 즉 \(5.7{-}6.0\%\).
- 결론: large-angle anomaly의 크기 스케일은 나온다. preferred axis, multipole phase correlation, full CMB likelihood는 `Open`.

2026-05-06, A-E 종합:

- `examples/physics/cosmology_gate_summary.py`를 추가했다.
- `docs/3_상수/7_우주론.md`에 우주론 게이트 종합 판정을 추가했다.
- 닫힌/채점 축: \(\Omega_b,\Omega_\Lambda,\Omega_{DM},n_s,H_0t_0,T_{\rm CMB},\eta\) scorecard 패키지.
- 강한 후보: \(A_s^{\rm A3c}=2.1038\times10^{-9}\), d=0 재귀 엔트로피 경계, 암흑물질 집합 요동 정체.
- 열린 테스트: \(\alpha_{\rm spec}\), \(r_{\rm tensor}\), \(S_8\) 잔차, halo/JWST 성장, CMB large-angle axis/likelihood.
- 다음 라운드 우선순위: full CMB likelihood보다 먼저 외부 데이터 없이 계산 가능한 halo mass function/JWST high-\(z\) 게이트 또는 running/tensor 관측상한 게이트를 세운다.

2026-05-06, 수식 의미 분석과 다음 게이트:

- `docs/3_상수/9_우주론_수식_의미와_후보.md`를 추가했다.
- 핵심 개선: 수식 자체를 고친 것이 아니라, 관측량 읽기 층을 고정점 상태량 / 총 민감도 / 투영 잔차 구동력 / 동결 잔차 / 후기 추출 잔차로 분리했다.
- `examples/physics/inflation_next_tests_gate.py`를 추가했다.
- running/tensor 결과: \(\alpha_{\rm spec}=-6.11284281\times10^{-4}\), \(r_{\rm tensor}=0.00366771\), \(r/(-\alpha_{\rm spec})=6\).
- Planck 2018 running \(-0.0045\pm0.0067\) 대비 \(+0.58\sigma\). BK18+Planck+BAO \(r<0.032\) 대비 CE \(r\)는 상한의 11.5%.
- 판정: running/tensor는 현재 관측상한과 충돌하지 않지만 검출이 아니므로 `Open test`.

2026-05-06, halo/JWST 구조 프록시:

- `examples/physics/high_z_structure_proxy_gate.py`를 추가했다.
- 개선/수정 판정: 배경 성장과 \(\sigma_8\) 보정만으로 CE가 high-\(z\) rare halo를 자동 증가시킨다는 주장은 성립하지 않는다.
- \(z=8,10,12,15\)에서 선형 진폭비는 CE static/LCDM \(\simeq0.97126\), CE \(H_0\) branch/LCDM \(\simeq0.93206\).
- Press-Schechter 꼬리 프록시 \(F\propto\operatorname{erfc}(\nu/\sqrt2)\) 기준, \(z=10,\nu=3\)에서 CE static/LCDM \(=0.744414\), CE \(H_0\) branch/LCDM \(=0.477133\).
- 결론: JWST/halo 항목은 `Open` 유지. 닫으려면 scale-dependent transfer, 집합 응축 성장항, 또는 small-scale primordial power 보정이 필요하다.

2026-05-06, small-scale boost 요구량:

- `examples/physics/small_scale_boost_requirement_gate.py`를 추가했다.
- \(\nu=3\) rare tail에서 LCDM과 같아지려면 CE static은 +2.96%, CE \(H_0\) branch는 +7.29% small-scale amplitude boost가 필요하다.
- 같은 \(\nu=3\)에서 2x/5x/10x 꼬리비를 만들려면 CE static은 +11.02%/+25.04%/+39.67%, CE \(H_0\) branch는 +15.69%/+30.29%/+45.54%가 필요하다.
- 후보 크기 검산: \(1+Q_{\rm A3c}=1.02837\)은 static 가지를 거의 LCDM 수준으로 복구하지만 강한 enhancement는 부족하다. \(1+\delta=1.17776\)은 \(\nu=3\)에서 static 꼬리비 3.23, \(H_0\) 가지 꼬리비 2.33까지 가능하다.
- 다음 후보식: \(T_{\rm CE}(k)=T_{\Lambda{\rm CDM}}(k)[1+\delta W_{\rm cond}(k)]\). 단, \(W_{\rm cond}(k)\)는 큰 스케일에서 꺼져야 하므로 아직 `Open`.

2026-05-06, 자기재귀참조 수학/물리 근거 보강:

2026-05-06, A3c 중력 환경 readout 증명 후보:

- `docs/2_경로적분과_응용/14_자기재귀성_대칭.md`에 중력 환경 읽기 원리를 추가했다.
- 핵심 해석: \(D/(D+1)\)의 \(+1\)은 임의 bulk/boundary 보정이 아니라, 내부 게이지-물질 재귀 \(D\)를 관측 가능하게 만드는 중력/시공간 환경이다.
- A3c를 세 성분으로 분해했다.
  - \(x(1-x)\): 자기재귀 잔차 \(r_{\mathcal R}\)의 국소 구동력.
  - \(2/\pi\): \(d=0\to d=3\) 반주기 위상 투영.
  - \(\sigma^{D_{\rm eff}/(D_{\rm eff}+1)}\): \(+1\) 중력 환경에서 읽는 결핍 측도.
- `examples/physics/gravitational_environment_readout_gate.py`를 추가했다.
- `docs/3_상수/9_우주론_수식_의미와_후보.md`에 A3c 중력 환경 읽기 원리 섹션을 추가했다.
- 현재 판정: raw susceptibility readout은 계속 기각. GER/A3c readout은 \(A_s=2.1038\times10^{-9}\)로 broad gate를 통과하지만, 아직 `Selection candidate`.
- 다음 승격 조건: 같은 \(+1\) 중력 환경 readout을 horizon entropy, FLRW scale map, CMB large-angle anomaly 중 하나 이상에서 반복 검산한다.

2026-05-06, A3c 반복 readout 검증:

- `examples/physics/a3c_readout_closure_gate.py`를 추가했다.
- 공통 투영 계수:
  \[
  P_{\rm GER}=(2/\pi)\sigma^{D_{\rm eff}/(D_{\rm eff}+1)}=0.61292326.
  \]
- 같은 \(P_{\rm GER}\)로 세 값을 반복 계산했다.
  - \(Q_{\rm A3c}=P_{\rm GER}x(1-x)=0.02836622\).
  - \(A_s=2.10380875\times10^{-9}\), pull \(+0.13\sigma\).
  - quadrupole power handle \(P_{\rm GER}^2=0.37567493\), quadrupole scale \(-0.99\sigma\) cosmic variance.
  - hemispherical handle \(2Q_{\rm A3c}/\sigma=2P_{\rm GER}x=0.05963341\).
- `docs/3_상수/9_우주론_수식_의미와_후보.md`에 반복 검산 섹션을 추가했다.
- `examples/physics/cosmology_gate_summary.py`에 A3c common projection 행을 추가했다.
- 판정: A3c/GER은 \(A_s\) 하나의 보정식에서 벗어나 CMB large-angle amplitude handle까지 재사용됐다. 하지만 preferred axis, phase correlation, full CMB likelihood가 없으므로 아직 `Selection candidate/Open test`.
- horizon entropy는 같은 \(+1\) 철학이지만 별도 phase-area bridge로 분리했다:
  \[
  \log S_{\rm dS}\simeq(\pi^2/2)N_e-\pi\delta\sigma.
  \]

2026-05-06, A3c preferred-axis no-go:

- `examples/physics/a3c_preferred_axis_no_go_gate.py`를 추가했다.
- `docs/3_상수/9_우주론_수식_의미와_후보.md`에 preferred-axis no-go 섹션을 추가했다.
- 판정: 현재 GER 입력 \(\alpha_s,\delta,D_{\rm eff},x,\sigma,P_{\rm GER},Q_{\rm A3c}\)는 모두 SO(3) 스칼라이므로 preferred unit vector \(n_i\), traceless tensor \(T_{ij}\), phase map \(\phi_{\ell m}\)을 만들 수 없다.
- 따라서 스칼라 A3c/GER은 amplitude handle을 닫는 후보이지, CMB preferred-axis/phase/likelihood closure가 아니다.
- 다음 bridge 조건:
  1. early-late horizon gradient에서 \(n_i\)를 유도,
  2. anisotropic recursive defect flow에서 \(T_{ij}\)를 유도,
  3. CMB map/covariance data-facing bridge에서 축을 추정하고 CE는 \(P_{\rm GER}\) 진폭만 사전등록.
- `examples/physics/cosmology_gate_summary.py`에 preferred-axis no-go 행을 추가했다.

2026-05-06, A3c 조건부 axis bridge:

- `examples/physics/a3c_conditional_axis_bridge_gate.py`를 추가했다.
- 스칼라 GER이 축 \(n_i\)를 만들지는 못하지만, 축이 별도 브리지에서 주어지면 CE가 닫는 양을 수식으로 고정했다.
- 고정 진폭:
  \[
  P_{\rm GER}=0.61292326,\quad Q_{\rm A3c}=0.02836622.
  \]
- quadrupole power handle:
  \[
  S_Q=P_{\rm GER}^2=0.37567493.
  \]
- hemispherical modulation:
  \[
  A_H=2Q_{\rm A3c}/\sigma=2P_{\rm GER}x=0.05963341.
  \]
- 조건부 tensor:
  \[
  T_{ij}=S_Q(n_in_j-\delta_{ij}/3),\quad {\rm Tr}\,T=0,\quad
  \|T\|_F=S_Q\sqrt{2/3}=0.30673729.
  \]
- 기본 \(z\)-축과 \((1,1,1)\) 축에서 모두 \(|m|=0.05963341\), \(\|T\|_F=0.30673729\) 검증.
- 판정: 남은 완성 문제는 더 이상 amplitude가 아니라 \(n_i\)를 유도하거나 CMB data-facing bridge에서 ingest하는 것이다.

2026-05-06, A3c CMB data-facing axis ingest:

- `examples/physics/a3c_cmb_axis_ingest_gate.py`를 추가했다.
- 대표 관측축/진폭을 ingest하되 CE는 \(A_H=0.05963341\), \(S_Q=0.37567493\), \(\|T\|_F=0.30673729\)만 사전등록값으로 검산한다.
- Planck/PR3 temperature HPA representative:
  - 축 \((l,b)=(205^\circ,-20^\circ)\), 관측 진폭 \(0.070\pm0.021\).
  - CE \(A_H=0.05963341\), pull \(-0.49\sigma\).
- Planck PR4 Sevem E-mode local variance:
  - 축 \((l,b)=(234^\circ,-14^\circ)\), broad modulation range \(0.060{-}0.130\).
  - CE \(A_H=0.05963341\), lower edge와 \(-0.0004\) 차이로 edge-compatible.
- temperature/polarization 대표축 각거리: \(28.33^\circ\).
- 판정: CE amplitude는 representative HPA amplitude와 양립한다. 축은 여전히 유도된 것이 아니라 관측 ingest이며, full CMB map/covariance likelihood가 다음 병목이다.

2026-05-06, A3c CMB amplitude likelihood proxy:

- `examples/physics/a3c_cmb_likelihood_proxy_gate.py`를 추가했다.
- representative HPA amplitude rows를 단순 가우시안 proxy로 묶어 \(A=0\), CE fixed \(A=A_H\), one-parameter best-fit \(A\)를 비교했다.
- 입력:
  - temperature representative \(A=0.070\pm0.021\)
  - PR4 Sevem E-mode proxy \(A=0.090\pm0.035\)
- weighted best-fit:
  \[
  A_{\rm fit}=0.07529412\pm0.01800735.
  \]
- CE fixed:
  \[
  A_{\rm CE}=0.05963341,\quad (A_{\rm CE}-A_{\rm fit})/\sigma_{\rm fit}=-0.87.
  \]
- proxy likelihood:
  \[
  \chi^2(A=0)=17.7234,\quad
  \chi^2(A_{\rm CE})=0.9964,\quad
  \chi^2(A_{\rm fit})=0.2401.
  \]
- 판정: CE fixed amplitude는 null 대비 \(\Delta\chi^2=-16.7269\) 개선이고 best-fit 대비 \(+0.7564\)만 나쁘다. 단, 이건 full CMB likelihood가 아니라 pre-likelihood amplitude compatibility.

2026-05-06, A3c/GER closure package 마무리:

- `examples/physics/a3c_closure_package_gate.py`를 추가했다.
- 최종 판정:
  - raw scalar readout \(A_s[|dx/dD|]\)는 \(+191.18\sigma\)로 기각.
  - GER scalar readout \(A_s[Q_{\rm A3c}]\)는 \(+0.13\sigma\)로 broad gate 통과.
  - \(P_{\rm GER}=0.61292326\), \(Q_{\rm A3c}=0.02836622\), \(A_H=0.05963341\), \(S_Q=0.37567493\).
  - 조건부 tensor norm \(\|T\|_F=0.30673729\).
  - representative HPA proxy에서 \(\Delta\chi^2_{\rm CE-null}=-16.7269\), \(\Delta\chi^2_{\rm CE-fit}=+0.7564\).
- 승격 문장: A3c/GER은 `Selection candidate scalar readout principle` 및 `pre-likelihood CMB large-angle amplitude bridge`.
- 승격하지 않는 문장: `Exact theorem` 또는 `full CMB anomaly closure`.
- 남은 blocker:
  \[
  \text{derive }n_i\quad\text{or}\quad
  \text{run map/covariance likelihood with }A=A_H\text{ fixed}.
  \]
- 반증 조건을 gate에 명시했다: \(A_s/running/tensor\) family rejection, robust HPA amplitude mismatch, full likelihood failure, \(n_i\) phase test failure.

- `docs/2_경로적분과_응용/14_자기재귀성_대칭.md`에 "풀 경로 보존에서 자기재귀가 나오는 이유" 절을 추가했다.
- 핵심 정리: 경로 측도공간 \(\mathcal M(\Gamma)\)에서 닫힌 연산자 \(\mathcal T_D:\mathcal M(\Gamma)\to\mathcal M(\Gamma)\), 외부 선택자 금지, 차원 깊이의 곱적 감쇠를 함께 두면 \(x=\exp[-D(1-x)]\)가 나온다.
- 물리 해석: CE는 stationary phase처럼 경로 하나를 고르거나 Wilsonian coarse graining처럼 자유도를 버리는 것이 아니라, 풀 경로공간이 자기 억압률을 만들고 그 잔차를 관측 채널로 읽는 구조다.
- 지위 보정: \(K[x]=1-x\)는 최소 자기재귀 kernel이다. 더 일반적인 \(x=\exp[-DK[x]]\) kernel의 유일성은 아직 `Open`.
- `docs/3_상수/9_우주론_수식_의미와_후보.md`에도 같은 근거와 물리 해석을 반영했다.

2026-05-06, \(d=0\) 항등성의 성질과 접근 가능성:

- `docs/2_경로적분과_응용/14_자기재귀성_대칭.md`에 14.3a 절을 추가했다.
- \(d=0\) 항등성의 세 성질을 완전 생존성 \(x_0=1\), 무결핍성 \(\sigma_0=0\), 무재귀엔트로피 \(\mathcal S_{\mathcal R,0}=0\)로 정리했다.
- 3차원에 남는 영향은 결핍량 \(\sigma_D=1-x_D\), 수축률 \(k_D=D_{\rm eff}x_D\), 차원 결핍지수 \(\delta=D_{\rm eff}-3\)로 분해했다.
- 현재 값: \(\sigma_D=0.95135328\), \(\mathcal S_{\mathcal R}=3.02317090\), \(k_D=0.15458752\).
- 판정: \(d=3\) 안에서 잔차를 줄이는 내부 접근은 가능하지만, \(d=0\) 항등상 자체는 Lambert \(W_{-1}\) 경계 가지라 유한한 수축 동역학으로 도달한다고 말할 수 없다. 즉 \(d=0\)은 목적지가 아니라 \(d=3\) 잔차 흐름의 거울 경계조건이다.

2026-05-06, \(d=0\) 존재성 검증 시도:

- `examples/physics/d0_boundary_verification_gate.py`를 추가했다.
- 직접 존재 판정: \(d=0\)은 \(d=3\) 사건/장소가 아니라 경계 가지이므로 직접 관측으로 확정할 수 없다.
- 수학적 경계: \(D=0\Rightarrow x_0=1,\sigma_0=0,\mathcal S_{\mathcal R,0}=0\), \(r_R(x_0;0)=0\).
- \(d=3\) 흔적 패키지: \(D_{\rm eff}=3.17775842\), \(x=0.04864672\), \(\sigma=0.95135328\), \(\mathcal S_{\mathcal R}=3.02317090\), \(k=0.15458752\), \(N_e=57.19965162\).
- 관측 흔적: \(\Omega_b\) pull \(-1.63\sigma\), \(n_s\) pull \(+0.03\sigma\), \(A_s\) raw는 실패, \(A_s\) projected residual은 \(+0.13\sigma\).
- 판정: \(d=0\) 직접 존재는 증명되지 않았다. 하지만 \(d=3\) 물리의 zero-residual boundary condition으로는 살아남는다. 향후 \(n_s\), running, tensor, \(A_s\) 잔차 읽기 원리가 깨지면 형식적 이상점으로 강등한다.

2026-05-06, \(d=0\) 강건성/반례 게이트:

- `examples/physics/d0_boundary_robustness_gate.py`를 추가했다.
- \(\delta=0\) plain \(d=3\) null은 \(x=0.05952021\), \(\Omega_b\) pull \(+25.55\sigma\), \(\chi^2(\Omega_b,n_s)=653.042\)로 깨진다.
- CE 최소 kernel은 \(x=0.04864672\), \(\Omega_b\) pull \(-1.63\sigma\), \(n_s\) pull \(+0.03\sigma\), \(\chi^2=2.668\)로 살아남는다.
- \(K[x]=c(1-x)\)에서 \(\Omega_b\)만 맞추는 \(c=0.99627164\) 조정은 가능하다. 따라서 \(K[x]=1-x\)의 유일성은 아직 증명되지 않았다.
- kernel scan: \(c=0.98\)은 \(\Omega_b\) \(+7.41\sigma\), \(c=1.02\)는 \(-10.00\sigma\)라서 \(c=1\) 근방이 강하게 압박된다.
- \(A_s\) total sensitivity는 \(+191.18\sigma\)로 기각, projected residual은 \(+0.13\sigma\)로 생존.
- 다음 hard test: running/tensor가 같은 \(N_e\) family를 따라야 하고, \(\mathcal S_{\mathcal R}\)를 FLRW curvature/reheating/horizon entropy로 옮기는 scale map이 필요하다.

2026-05-06, FLRW scale map 1차:

- `examples/physics/flrw_scale_map_gate.py`를 추가했다.
- 닫힌 무차원 map: \(e^{-2N_e}=2.07498\times10^{-50}\), \(k^{N_e}=4.17888\times10^{-47}\).
- 평탄성 억압 반복수: \(k^n<10^{-5}\)에는 \(n=6.17\), \(k^n<10^{-60}\)에는 \(n=74.00\).
- scale-lift 후보: \(r=0.00366771\), \(A_s=2.10\times10^{-9}\)를 쓰면 \(H_{\rm inf}/M_{\rm Pl}=6.16512\times10^{-6}\), \(H_{\rm inf}=1.50121\times10^{13}\,{\rm GeV}\), \(V^{1/4}=7.95701\times10^{15}\,{\rm GeV}\). A3c \(A_s\)를 쓰면 \(H_{\rm inf}=1.50257\times10^{13}\,{\rm GeV}\).
- 열린 부분: horizon entropy는 \(S_{\rm dS}\simeq2.26544\times10^{122}\), \(\log S_{\rm dS}=281.73315\)라서 \(\mathcal S_{\mathcal R}=3.02317\)만으로는 닫히지 않는다. \(\log S_{\rm dS}/\mathcal S_{\mathcal R}=93.19127\)의 추가 lift factor가 필요하다.
- 판정: \(d=0\)은 FLRW 평탄화 비율과 안정성에는 연결되지만, 절대 horizon/reheating entropy에는 아직 연결되지 않는다. 완전한 FLRW 초기상태 정리는 아님.

2026-05-06, horizon entropy lift 후보:

- `examples/physics/horizon_entropy_lift_gate.py`를 추가했다.
- 기준 \(H_0=67.4\,{\rm km\,s^{-1}Mpc^{-1}}\)에서 \(\log S_{\rm dS}=281.73315068\), \(\log S_{\rm dS}/N_e=4.92543473\).
- leading 후보 \((\pi^2/2)N_e=282.26896669\)는 \(H_0=51.5595\)를 예측해 \(-23.50\%\)로 실패한다.
- 결핍-암흑 보정 후보:
  \[
  \log S_{\rm dS}^{\rm cand}=(\pi^2/2)N_e-\pi\delta\sigma.
  \]
- 이 후보는 \(\log S_{\rm dS}^{\rm cand}=281.73768863\), \(H_0^{\rm cand}=67.247245\,{\rm km\,s^{-1}Mpc^{-1}}\), 기준 대비 \(-0.227\%\).
- 판정: 강한 수치 패턴이지만 아직 `Candidate`. 승격하려면 \(\pi^2/2\) phase-area lift와 \(\pi\delta\sigma\) boundary correction을 독립적으로 유도해야 한다.

2026-05-06, horizon entropy lift 조건부 유도:

- `examples/physics/horizon_entropy_lift_derivation_gate.py`를 추가했다.
- \(\pi^2/2\)는 반주기 위상면적 \(A_{\rm phase}=\int_0^\pi\theta\,d\theta\)로 쓸 수 있다.
- \(\pi\delta\sigma\)는 같은 반주기 위의 균일 결핍 경계밀도 \(B_{\rm defect}=\int_0^\pi\delta\sigma\,d\theta\)로 쓸 수 있다.
- 따라서 조건부로 \(\log S_{\rm dS}\simeq N_eA_{\rm phase}-B_{\rm defect}=(\pi^2/2)N_e-\pi\delta\sigma\).
- 수치: \(B_{\rm defect}=0.53127806\), \(\log S_{\rm pred}=281.73768863\), \(\Delta\log S=+0.00453795\), \(H_0=67.247245\).
- 역산 경계밀도: \(\rho_{\rm req}=0.17055553\), 후보 \(\delta\sigma=0.16911106\), 오차 \(-0.847\%\).
- 지위: `Raw numerology`에서 `Conditional/Bridge`로 승격. 남은 핵심 가정은 horizon log-entropy가 왜 e-fold당 반주기 위상면적을 세는지다.

2026-05-06, phase-area 계수 내부 고정:

- `examples/physics/phase_area_principle_gate.py`를 추가했다.
- 반주기 순서 위상면적: \(\int_0^\pi\theta d\theta=\pi^2/2=4.93480220\).
- adjoint-normalized full phase square: \((2\pi)^2/(d^2-1)\).
- 두 값이 같으려면 \((2\pi)^2/(d^2-1)=\pi^2/2\Rightarrow d^2-1=8\Rightarrow d=3\).
- 수치: \(d=2\)는 \(+166.667\%\), \(d=3\)은 \(0.000\%\), \(d=4\)는 \(-46.667\%\).
- 판정: \(\pi^2/2\)는 horizon lift 안에서 임의 계수가 아니라 \(d=3\) adjoint phase measure와 반주기 위상면적이 만나는 내부 고정 계수다. 남는 가정은 이 계수가 horizon log-entropy의 e-fold readout으로 들어간다는 점이다.

2026-05-06, horizon readout 원리 압박:

- `examples/physics/horizon_readout_principle_gate.py`를 추가했다.
- 같은 경계 보정 \(\pi\delta\sigma=0.53127806\) 아래에서 readout 계수를 비교했다.
- 실패: 계수 \(1\), \(2\), \(\pi/2\), \(2/\pi\), \(2\pi^2\)는 \(H_0\)가 극단적으로 빗나간다.
- 생존: \(\pi^2/2\)와 \((2\pi)^2/8\)만 \(H_0=67.247245\)로 살아남는다.
- 역산 필요 계수: \(c_{\rm req}=(\log S_{\rm dS}+\pi\delta\sigma)/N_e=4.93472287\), \(\pi^2/2=4.93480220\)와 \(+0.002\%\).
- 표준 slow-roll 분리: \(d\log S_{\rm hor}/dN=2\epsilon_H\)와는 다른 식이다. 즉 이건 인플레이션 중 국소 entropy growth가 아니라 primordial e-fold phase count가 late-time de Sitter entropy로 lift되는 경계 readout이다.
- 판정: 계수 선택은 거의 고정되었고, 남은 가정은 late-time horizon이 왜 primordial phase-area count를 읽는가이다.

2026-05-06, early-late boundary invariant:

- `examples/physics/early_late_boundary_invariant_gate.py`를 추가했다.
- 제안 불변량: \(I_H=\log S_{\rm dS}+\pi\delta\sigma\), \(I_{\rm phase}=(\pi^2/2)N_e\).
- 경계 readout 원리: \(I_H\simeq I_{\rm phase}\).
- 앞방향: \(\log S_{\rm pred}=281.73768863\), \(H_0^{\rm pred}=67.247245\,{\rm km\,s^{-1}Mpc^{-1}}\), 기준 \(67.4\) 대비 \(-0.227\%\).
- 역방향: 기준 \(H_0=67.4\)에서 \(N_e^{(H_0)}=57.19873204\), CE \(N_e=57.19965162\), \(\Delta N_e=-0.00091958\).
- gauge count 환산: \(N_{\rm gauge}^{(H_0)}=11.99980708\), 목표 12와 \(-0.00019292\) 차이.
- 민감도: \(H_0\propto\exp[-(\pi^2/4)N_e]\), 한 e-fold 변화는 \(H_0\)를 \(0.08480497\)배로 바꾼다.
- 판정: late-time horizon이 primordial phase-area count를 읽는 이유를 같은 경계 불변량의 양끝으로 해석할 수 있다. 지위는 `Bridge`; 독립 \(H_0\) 데이터와 BAO/CMB 조합 반례 검사가 필요하다.

2026-05-06, H0 branch falsification:

- `examples/physics/h0_readout/h0_dataset_falsification_gate.py`를 추가했다.
- boundary invariant
  \[
  \log S_{\rm dS}(H_0)+\pi\delta\sigma\simeq(\pi^2/2)N_e
  \]
  를 \(H_0\)에 대해 풀면
  \[
  H_0^{\rm CE}=67.247245\,{\rm km\,s^{-1}Mpc^{-1}}.
  \]
- Planck 2018 base LCDM \(67.4\pm0.5\): pull \(-0.31\sigma\), 통과.
- DESI-like BAO \(68.51\pm0.58\): pull \(-2.18\sigma\), 약한 긴장.
- CCHP 2025 JAGB/TRGB 계열: \(-0.20\sigma\)에서 \(-1.62\sigma\), 통과.
- SH0ES high Cepheid/SN 계열: \(73.04\pm1.04\)에서 \(-5.57\sigma\), JWST update \(73.17\pm0.86\)에서 \(-6.89\sigma\), 반증.
- 결론: 이 식은 Hubble tension을 양쪽 모두 설명하지 않는다. 낮은 \(H_0\) branch를 예측한다. high Cepheid/SN branch가 최종 승리하면 현재 horizon invariant는 틀렸거나 late-time branch correction이 필요하다.

2026-05-06, H0 late branch correction:

- `examples/physics/h0_readout/h0_late_branch_correction_gate.py`를 추가했다.
- low branch:
  \[
  \log S_{\rm low}=(\pi^2/2)N_e-\pi\delta\sigma,
  \qquad
  H_0=67.247245.
  \]
- 관측 \(H_0\)가 요구하는 추가량:
  \[
  \Delta\log S_{\rm req}=-2\log(H_0^{\rm obs}/H_0^{\rm low}).
  \]
- SH0ES HST \(73.04\): \(\Delta\log S=-0.16526238\), \((-\delta\sigma)\) 대비 0.9772.
- SH0ES JWST \(73.17\): \(\Delta\log S=-0.16881891\), \((-\delta\sigma)\) 대비 0.9983.
- CE 내부 결핍 밀도: \(\delta\sigma=0.16911106\).
- 따라서 local endpoint correction
  \[
  \log S_{\rm high}=(\pi^2/2)N_e-\pi\delta\sigma-\delta\sigma
  \]
  는 \(H_0=73.180689\)를 예측한다.
- 판정: high branch를 살리는 보정은 새 상수가 아니라 기존 결핍 밀도 \(\delta\sigma\)다. 그러나 이것을 허용하면 단일 H0 예측이 아니라 global horizon readout과 local endpoint readout의 이중 구조가 된다. 다음 문제는 어떤 관측 채널이 어떤 readout을 쓰는지 유도하는 것이다.

2026-05-06, H0 readout selector:

- `examples/physics/h0_readout/h0_readout_selector_gate.py`를 추가했다.
- branch를 하나의 식으로 묶었다.
  \[
  \log S(q)=(\pi^2/2)N_e-\pi\delta\sigma-q\delta\sigma.
  \]
- \(q=0\): global horizon readout, \(H_0=67.247245\).
- \(q=1\): local endpoint-defect readout, \(H_0=73.180689\).
- 역산 결과:
  - Planck/CMB: \(q=0.0268\).
  - DESI BAO+BBN: \(q=0.2200\).
  - CCHP JAGB: \(q=0.0968\).
  - CCHP JWST-only TRGB: \(q=0.2717\).
  - CCHP HST+JWST TRGB: \(q=0.5402\).
  - SH0ES HST Cepheids/SNe: \(q=0.9772\).
  - SH0ES JWST update: \(q=0.9983\).
- 기대 locality ordering에 대한 \(\chi^2/{\rm dof}=0.220/6\).
- 판정: 데이터는 CMB \(\to q\simeq0\), BAO/JAGB/TRGB \(\to\) 중간, Cepheid/SN \(\to q\simeq1\) 순서를 따른다. 하지만 아직 \(q\)를 광전파/거리사다리 경로적분에서 직접 유도하지 않았으므로 `Bridge/Open test`.

2026-05-06, H0 channel topology:

- `examples/physics/h0_readout/h0_channel_topology_gate.py`를 추가했다.
- selector를 topology 값으로 압축했다.
  \[
  q_{\rm topo}=\frac{L}{L+G}.
  \]
- \(L\): local endpoint closure weight.
- \(G\): global ruler/horizon closure weight.
- 비교:
  - Planck/CMB: \(L:G=0:1\), \(q_{\rm topo}=0.0000\), \(q_{\rm req}=0.0268\).
  - DESI BAO+BBN: \(1:3\), \(q_{\rm topo}=0.2500\), \(q_{\rm req}=0.2200\).
  - CCHP JAGB: \(1:9\), \(q_{\rm topo}=0.1000\), \(q_{\rm req}=0.0968\).
  - CCHP JWST-only TRGB: \(1:3\), \(q_{\rm topo}=0.2500\), \(q_{\rm req}=0.2717\).
  - CCHP HST+JWST TRGB: \(1:1\), \(q_{\rm topo}=0.5000\), \(q_{\rm req}=0.5402\).
  - SH0ES Cepheid/SN: \(1:0\), \(q_{\rm topo}=1.0000\), \(q_{\rm req}=0.9772{-}0.9983\).
- topology \(\chi^2/{\rm dof}=0.220/6\).
- 판정: \(q\)는 자유 fitting 상수가 아니라 관측 채널의 endpoint-locality topology 후보가 되었다. 하지만 \(L,G\)를 likelihood/covariance에서 직접 계산하지 않았으므로 아직 `Bridge/Open test`.

2026-05-06, H0 prospective external channels:

- `examples/physics/h0_readout/h0_prospective_channel_gate.py`를 추가했다.
- 기존 selector table에 쓰지 않은 외부 채널에 대해 \(H_0\) 값을 맞추기 전에 \(L:G\)를 먼저 선언했다.
- 예측:
  - H0LiCOW/TDCOSMO power-law lenses: \(L:G=1:0\), \(q=1\), \(H_0^{\rm pred}=73.181\), 관측 \(73.3\pm1.75\), pull \(-0.07\).
  - TDCOSMO+SLACS hierarchical lenses: \(1:3\), \(q=0.25\), \(H_0^{\rm pred}=68.684\), 관측 \(67.4\pm3.65\), pull \(+0.35\).
  - Megamaser Cosmology Project: \(1:0\), \(q=1\), \(H_0^{\rm pred}=73.181\), 관측 \(73.9\pm3.0\), pull \(-0.24\).
  - GW170817 bright siren: \(1:1\), \(q=0.5\), \(H_0^{\rm pred}=70.151\), 관측 \(70.3\pm5.15\), pull \(-0.03\).
  - O4a dark+bright sirens: \(1:1\), \(q=0.5\), \(H_0^{\rm pred}=70.151\), 관측 \(68.0\pm4.1\), pull \(+0.52\).
- 합산 \(\chi^2/{\rm dof}=0.462/5\).
- 판정: \(q_{\rm topo}\)는 외부채널에서도 high/local, low/hierarchical, middle/GW 구조를 재현한다. 다음 과제는 \(L,G\)를 손배정하지 않고 covariance graph에서 자동 계산하는 것.

2026-05-06, H0 covariance graph selector:

- `examples/physics/h0_readout/h0_covariance_graph_selector_gate.py`를 추가했다.
- 손배정 \(L,G\) 대신 관측 graph의 conductance로 selector를 계산한다.
  \[
  C_{\rm path}=\frac{\prod_{e\in p}r_e}{|p|},
  \qquad
  q_{\rm graph}=\frac{C_L}{C_L+C_G}.
  \]
- toy graph 결과:
  - Planck: \(C_L:C_G=0:1\), \(q=0\), \(H_0=67.247\).
  - DESI BAO+BBN: \(1:3\), \(q=0.25\), \(H_0=68.684\).
  - CCHP JAGB: \(1:9\), \(q=0.1\), \(H_0=67.818\).
  - CCHP TRGB JWST-only: \(1:3\), \(q=0.25\), \(H_0=68.684\).
  - CCHP TRGB HST+JWST: \(1:1\), \(q=0.5\), \(H_0=70.151\).
  - SH0ES/Cepheid, megamaser: \(1:0\), \(q=1\), \(H_0=73.181\).
  - GW representative: \(1:1\), \(q=0.5\), \(H_0=70.151\).
- 합산 \(\chi^2/{\rm dof}=0.402/10\).
- 판정: \(q\)는 이제 자유상수 \(\to\) topology ratio \(\to\) graph conductance selector로 승격. 아직 edge reliability \(r_e\)가 실제 covariance matrix에서 온 것은 아니므로 다음은 Fisher/covariance 기반 edge 자동화다.

2026-05-06, H0 Fisher-edge robustness:

- `examples/physics/h0_readout/h0_fisher_edge_robustness_gate.py`를 추가했다.
- 각 nonzero conductance를
  \[
  C\to C e^\eta,\qquad \eta\sim\mathcal N(0,0.25^2)
  \]
  로 20,000회 흔들었다.
- 결과:
  - Planck \(q=0\): \(H_0=67.247\) 완전 고정.
  - JAGB \(q=0.1\): 16-84% \(H_0=67.662{-}68.029\), stability 0.968.
  - BAO/TRGB/TDCOSMO hierarchical \(q=0.25\): \(H_0\simeq68.34{-}69.10\), stability \(\sim0.89\).
  - TRGB mixed/GW \(q=0.5\): \(H_0\simeq69.63{-}70.67\), stability \(\sim0.78\).
  - SH0ES/megamaser \(q=1\): \(H_0=73.181\) 완전 고정.
- 중심값 \(\chi^2/{\rm dof}=0.402/10\).
- 판정: graph selector는 synthetic Fisher-edge perturbation에 대해 branch 순서를 보존한다. 다음은 실제 covariance/Fisher matrix를 edge reliability로 넣는 것이다.

2026-05-06, H0 Fisher matrix selector:

- `examples/physics/h0_readout/h0_fisher_matrix_selector_gate.py`를 추가했다.
- normalized Fisher edge:
  \[
  r_{ij}=\frac{|F_{ij}|}{\sqrt{F_{ii}F_{jj}}}.
  \]
- selector:
  \[
  q_F=\frac{C_L(F)}{C_L(F)+C_G(F)}.
  \]
- schematic Fisher matrix 검산:
  - Planck \(q_F=0\), \(H_0=67.247\).
  - DESI \(q_F=0.25\), \(H_0=68.684\).
  - JAGB \(q_F=0.1\), \(H_0=67.818\).
  - TRGB mixed/GW \(q_F=0.5\), \(H_0=70.151\).
  - SH0ES \(q_F=1\), \(H_0=73.181\).
- 파라미터 단위 rescaling \(F\to DFD\)에 대해 \(\max|\Delta q_F|\sim10^{-16}\).
- 합산 \(\chi^2/{\rm dof}=0.199/6\).
- 판정: \(q\)는 Fisher endpoint conductance로 쓸 수 있는 형태까지 내려왔다. 남은 과제는 실제 공개 likelihood/covariance에서 \(F\)를 읽어 \(q_F\)를 계산하는 것.

2026-05-06, Fisher/covariance IO gate:

- `examples/physics/h0_readout/h0_fisher_matrix_io_gate.py`를 추가했다.
- JSON 입력으로 실제 Fisher 또는 covariance matrix를 받을 수 있다.
- `matrix_type="covariance"`이면 내부에서 \(F=C^{-1}\)로 변환한다.
- edge:
  \[
  r_{ij}=|F_{ij}|/\sqrt{F_{ii}F_{jj}}.
  \]
- smoke test:
  - \(C_L=0.2\), \(C_G=0.2\), \(q_F=0.5\).
  - \(H_0^{\rm pred}=70.151263\).
  - GW-like \(70.3\pm5.15\) 대비 pull \(-0.029\).
- 판정: 실제 likelihood/covariance를 넣을 data-facing interface가 준비되었다.

2026-05-06, Fisher/covariance IO regression:

- 예제 JSON 추가:
  - `examples/physics/h0_readout/h0_fisher_io_examples/gw_like_fisher.json`
  - `examples/physics/h0_readout/h0_fisher_io_examples/gw_like_covariance.json`
- 회귀 게이트 추가:
  - `examples/physics/h0_readout/h0_fisher_io_regression_gate.py`
- 결과:
  - Fisher 입력: \(C_L=0.2\), \(C_G=0.2\), \(q_F=0.5\), \(H_0=70.151263\).
  - Covariance 입력: \(C_L=0.2\), \(C_G=0.2\), \(q_F=0.5\), \(H_0=70.151263\).
  - \(\Delta q_F=0\), \(\Delta H_0=0\).
- 판정: Fisher/covariance 입력 경로가 동등하게 작동한다.

2026-05-06, Fisher/covariance IO batch:

- `examples/physics/h0_readout/h0_fisher_io_batch_gate.py`를 추가했다.
- JSON 파일 하나 또는 JSON 디렉터리를 입력받아 \(q_F\), \(H_0^{\rm pred}\), pull, \(\chi^2\)를 batch 출력한다.
- 예제 디렉터리 실행:
  \[
  \chi^2/{\rm dof}=0.001668/2.
  \]
- 판정: 실제 공개 covariance를 채널별 JSON으로 변환하면 전체 readout table을 자동 생성할 수 있다.

2026-05-06, Fisher/covariance IO validation:

- `examples/physics/h0_readout/h0_fisher_io_validate_gate.py`를 추가했다.
- 검사 항목: 필수 필드, unique nodes, node 참조 무결성, `matrix_type`, 정방행렬, finite number, 대칭성, 양의 대각, covariance 역행렬 가능성.
- 예제 디렉터리 결과:
  - `gw_like_covariance.json`: PASS.
  - `gw_like_fisher.json`: PASS.
- 실제 데이터 실행 순서:
  1. `python examples/physics/h0_readout/h0_fisher_io_validate_gate.py path/to/channels`
  2. `python examples/physics/h0_readout/h0_fisher_io_batch_gate.py path/to/channels`

2026-05-06, Fisher/covariance negative validation:

- `examples/physics/h0_readout/h0_fisher_io_negative_gate.py`를 추가했다.
- 네 가지 깨진 입력을 검사:
  - unknown local node
  - nonsymmetric matrix
  - nonpositive diagonal
  - singular covariance
- 결과: 4/4 negative cases PASS.
- 판정: 좋은 JSON뿐 아니라 나쁜 JSON을 거부하는 입력 방어선도 생겼다.

2026-05-06, CSV covariance/Fisher adapter:

- `examples/physics/h0_readout/h0_covariance_csv_to_json.py`를 추가했다.
- `examples/physics/h0_readout/h0_fisher_io_examples/gw_like_fisher.csv` 예제를 추가했다.
- CSV -> JSON 변환 결과:
  - `examples/physics/h0_readout/h0_fisher_io_examples/gw_like_fisher_from_csv.json`
- validate 결과: 3/3 PASS.
- batch 결과: fisher, covariance, csv->json 세 입력 모두 \(q_F=0.5\), \(H_0=70.151263\).
- 판정: 실제 공개 covariance/Fisher matrix가 CSV로 제공되면 변환 후 validate/batch 파이프라인에 바로 넣을 수 있다.

2026-05-06, labelled CSV adapter:

- `examples/physics/h0_readout/h0_covariance_labelled_csv_to_json.py`를 추가했다.
- `examples/physics/h0_readout/h0_fisher_io_examples/gw_like_fisher_labelled.csv` 예제를 추가했다.
- labelled CSV -> JSON 결과:
  - `examples/physics/h0_readout/h0_fisher_io_examples/gw_like_fisher_from_labelled_csv.json`
- manifest에 labelled CSV 변환 산출물을 등록했다.
- full suite 결과: PASS, batch \(\chi^2/{\rm dof}=0.003336/4\).
- 판정: 공개 covariance가 header/row label이 있는 table 형태여도 변환 가능하다.

2026-05-06, H0 readout law audit:

- `docs/3_상수/10_H0_readout_law_audit.md`를 추가했다.
- 핵심 수식:
  \[
  \log S(q)=\frac{\pi^2}{2}N_e-\pi\delta\sigma-q\delta\sigma,
  \qquad
  q_F=\frac{C_L(F)}{C_L(F)+C_G(F)}.
  \]
- 평가:
  - low/high branch 수식 구조는 충분히 압축됨.
  - \(q\)는 자유 parameter에서 Fisher endpoint conductance로 내려옴.
  - IO/validation/batch/full-suite 준비 완료.
  - 아직 실제 공개 covariance로 \(q_F\)를 계산하지 않았으므로 `Bridge/Open test`.
- 진행 판단: 계속 진행 가능. 단, 다음은 새 수식 추가가 아니라 실제 covariance 투입이어야 한다.
- 추천 첫 실제 채널: GW standard siren 또는 TDCOSMO 계열.

2026-05-06, first real source scout:

- `examples/physics/h0_readout/h0_real_covariance_targets.json`를 추가했다.
- `examples/physics/h0_readout/h0_real_source_scout_gate.py`를 추가했다.
- 첫 후보: TDCOSMO/hierarchy_analysis_2020_public.
- 고정 commit: `6c293af582c398a5c9de60a51cb0c44432a3c598`.
- `git ls-remote` 결과 remote HEAD 일치: PASS.
- 후보 HDF5 chain:
  - `JointAnalysis/tdcosmo_ifu_chain_slope_log_scatter.h5`
  - `JointAnalysis/tdcosmo_slacs_chain_slope_log_scatter.h5`
  - `JointAnalysis/tdcosmo_slacs_ifu_chainifu_separate_slope_log_scatter.h5`
  - `TDCOSMO_sample/tdcosmo_chain_alpha_free.h5`
  - `TDCOSMO_sample/tdcosmo_chain_alpha_free_om.h5`
  - `TDCOSMO_sample/tdcosmo_chain_alpha_fixed_om.h5`
- 환경 상태: `h5py` 없음.
- 판정: 실제 source는 reachable. 다음은 HDF5 reader 설치 또는 CSV/text likelihood product 우회 탐색이다.

2026-05-06, Fisher/covariance IO full suite:

- `examples/physics/h0_readout/h0_fisher_io_full_suite.py`를 추가했다.
- 포함:
  - manifest validation gate
  - validate gate
  - negative validation gate
  - Fisher/covariance regression gate
  - batch gate
- 실행 결과: full Fisher/covariance IO suite passed.
- 판정: 실제 covariance를 넣기 전후의 회귀 기준 명령으로 사용한다.

2026-05-06, source manifest:

- `examples/physics/h0_readout/h0_fisher_io_examples/manifest.json`을 추가했다.
- `examples/physics/h0_readout/h0_fisher_manifest_validate_gate.py`를 추가했다.
- manifest 필수 항목: dataset bundle, version, channels, channel file, source, matrix role, channel class, notes, non-synthetic source URL.
- `validate`와 `batch` runner는 `manifest.json`을 channel JSON에서 제외하도록 수정했다.
- full suite에 manifest validation을 포함했다.
- 판정: 실제 공개 covariance는 matrix뿐 아니라 출처/버전 manifest까지 함께 있어야 채택한다.

2026-05-06, source manifest negative validation:

- `examples/physics/h0_readout/h0_fisher_manifest_negative_gate.py`를 추가했다.
- 네 가지 깨진 manifest를 검사:
  - missing channel file
  - non-synthetic missing source URL
  - duplicate channel file
  - missing channel field
- 결과: 4/4 manifest negative cases PASS.
- full suite에 포함했고 통과했다.
- 판정: 실제 공개 covariance bundle은 matrix-level validation과 source-level validation을 모두 통과해야 채택한다.

2026-05-06, H0 readout file organization and TDCOSMO provenance closure:

- H0/TDCOSMO readout gate들을 `examples/physics/h0_readout/` 아래로 정리했다.
- zebrafish 관련 산출물은 `examples/physics/evolution/` 쪽으로 분리했다.
- 새 실행 기준:
  - `python examples/physics/h0_readout/h0_fisher_io_full_suite.py`
- `examples/physics/h0_readout/README.md`를 추가해 main entry point와 provenance chain을 문서화했다.
- `h0_real_data/`는 공개 HDF5 chain과 notebook cache이므로 git ignore에 추가했다.
- TDCOSMO converter의 기본 factor source를 `ast`로 바꾸었다.
- 이제 covariance JSON의 likelihood factor graph는 public notebook의 `MCMCSampler` 첫 번째 인자에서 직접 생성된다.
- 네 chain의 AST factor extraction:
  - TDCOSMO-only: local endpoint branch
  - TDCOSMO+IFU: local endpoint branch
  - TDCOSMO+SLACS: global closure branch
  - TDCOSMO+SLACS+IFU: global closure branch
- full suite 결과: PASS.
- batch 결과:
  - TDCOSMO-only: local branch, predicted high-side readout, observed chain과 within error.
  - TDCOSMO+IFU: local branch, predicted high-side readout, observed chain과 within error.
  - TDCOSMO+SLACS: global branch, predicted low-side readout, observed chain과 within error.
  - TDCOSMO+SLACS+IFU: global branch, predicted low-side readout, observed chain과 within error.
- role ablation 결과:
  - 모든 MST freedom을 local 또는 global로 고정하면 branch structure 설명력이 나빠진다.
  - source-aware role transition이 필요하다는 반증 게이트를 통과했다.
- `docs/3_상수/11_TDCOSMO_real_covariance_audit.md`에 논문용 significance 섹션을 추가했다.
- 판정: 현재 H0 readout law의 가장 강한 결과는 "수치를 사후 fitting했다"가 아니라,
  public source likelihood composition에서 readout branch transition이 재현된다는 점이다.

2026-05-06, external H0 channel roadmap:

- `examples/physics/h0_readout/h0_external_channel_roadmap_gate.py`를 추가했다.
- full suite에 external roadmap gate를 포함했다.
- 후보 채널:
  - BAO+SN inverse distance ladder: global standard-ruler closure, low-side branch 예상.
  - SH0ES-style local ladder: local calibrator endpoint closure, high-side branch 예상.
  - GW standard sirens: distance-redshift bridge, intermediate branch 예상.
  - CMB acoustic-scale inference: early global horizon closure, low-side branch 예상.
- gate 결과: 다음 반증 타깃은 BAO+SN inverse-distance-ladder covariance provenance.
- full suite 결과: PASS.
- 판정: 다음 단계는 최종 H0 숫자를 넣는 것이 아니라,
  BAO+SN covariance/compressed likelihood의 label을 observable/local/global role set으로 변환하는 adapter다.

2026-05-06, BAO+SN source scout and BAO role adapter:

- `examples/physics/h0_readout/h0_external_source_targets.json`를 추가했다.
- `examples/physics/h0_readout/h0_bao_sn_source_scout_gate.py`를 추가했다.
- source scout:
  - `CobayaSampler/bao_data` HEAD `bb0c1c9009dc...` 확인 PASS.
  - `PantheonPlusSH0ES/DataRelease` HEAD `c447f0fea703...` 확인 PASS.
  - Pantheon+는 Windows full checkout에서 긴 경로/대소문자 충돌 문제가 있으므로 sparse/direct fetch가 필요하다.
- `examples/physics/h0_readout/h0_bao_mean_cov_role_adapter_gate.py`를 추가했다.
- BAO adapter 입력:
  - `desi_2024_gaussian_bao_ALL_GCcomb_mean.txt`
  - `desi_2024_gaussian_bao_ALL_GCcomb_cov.txt`
- BAO adapter 결과:
  - measurements: 12
  - quantities: `DH_over_rs`, `DM_over_rs`, `DV_over_rs`
  - local nodes: 0
  - global nodes: 14
- 판정: DESI BAO mean/covariance labels는 global standard-ruler closure 후보로 매핑된다.
- 다음 단계: 이 role graph를 H0 readout JSON으로 투입 가능한 aggregate covariance channel로 만들지,
  아니면 BAO+SN joint likelihood adapter를 먼저 만들지 결정한다.

2026-05-06, BAO branch-only readout:

- `examples/physics/h0_readout/h0_bao_global_readout_gate.py`를 추가했다.
- full suite에 BAO/SN source scout, BAO role adapter, BAO global readout gate를 포함했다.
- BAO branch-only gate는 관측 H0 값을 붙이지 않는다.
- 목적: DESI BAO source role이 local branch가 아니라 global branch를 선택하는지 확인.
- 결과:
  - nodes: 14
  - local nodes: 0
  - global nodes: 13
  - local conductance: 0
  - global conductance: positive
  - selector: global endpoint
  - branch prediction: low-side H0 branch
- full suite 결과: PASS.
- 판정: TDCOSMO 밖의 독립 계열인 DESI BAO에서도 source role 기준 branch 방향은 이론 기대와 일치한다.
- 다음 단계: Pantheon+SH0ES sparse/direct fetch adapter로 local ladder contrast를 만들거나,
  BAO+SN joint source role graph를 구성한다.

2026-05-06, Pantheon+SH0ES local-ladder readout:

- `examples/physics/h0_readout/h0_pantheon_shoes_role_adapter_gate.py`를 추가했다.
- `examples/physics/h0_readout/h0_pantheon_shoes_local_readout_gate.py`를 추가했다.
- full suite에 두 gate를 포함했다.
- public source:
  - `Pantheon+SH0ES.dat`
  - `Pantheon+SH0ES_STAT+SYS.cov`
  - repo HEAD `c447f0fea703...`
- role adapter 결과:
  - rows: 1701
  - columns: 47
  - surveys: 20
  - Cepheid calibrator rows: 77
  - SH0ES Hubble-flow rows: 277
  - calibrator/Hubble-flow overlap: 0
  - local nodes: 3
  - global nodes: 0
- branch-only readout 결과:
  - local conductance: positive
  - global conductance: 0
  - selector: local endpoint
  - branch prediction: high-side H0 branch
- full suite 결과: PASS.
- 판정:
  - DESI BAO는 global/low-side branch.
  - Pantheon+SH0ES는 local/high-side branch.
  - Hubble tension의 대표 외부 두 계열이 source role 기준으로 이론 기대 방향을 고른다.
- 다음 단계:
  - BAO+Pantheon+SH0ES joint source-role contrast gate를 만든다.
  - 그 뒤 최종 H0 숫자 비교가 아니라 branch separation/ablation 비교로 논문용 핵심 그림을 만든다.

2026-05-06, cross-channel branch contrast:

- `examples/physics/h0_readout/h0_cross_channel_branch_contrast_gate.py`를 추가했다.
- full suite에 포함했다.
- 포함 채널:
  - TDCOSMO-only: local/high
  - TDCOSMO+IFU: local/high
  - TDCOSMO+SLACS: global/low
  - TDCOSMO+SLACS+IFU: global/low
  - DESI BAO: global/low
  - Pantheon+SH0ES: local/high
- 결과:
  - local family mean selector: 0.894119
  - global family mean selector: 0.001853
  - cross-channel separation: 0.892266
- full suite 결과: PASS.
- 판정: joint H0 refit 이전의 source role만으로도 독립 H0 채널들이 local/high와 global/low로 강하게 분리된다.
- 논문 핵심 그림 후보:
  - x축: source readout role 또는 selector
  - y축: H0 branch
  - 점: TDCOSMO-only, TDCOSMO+IFU, TDCOSMO+SLACS, TDCOSMO+SLACS+IFU, DESI BAO, Pantheon+SH0ES

2026-05-06, cross-channel role ablation:

- `examples/physics/h0_readout/h0_cross_channel_role_ablation_gate.py`를 추가했다.
- full suite에 포함했다.
- 결과:
  - declared source-aware roles: 6/6
  - all-local ablation: 3/6
  - all-global ablation: 3/6
  - flipped-role ablation: 0/6
- full suite 결과: PASS.
- 판정: cross-channel branch split은 모든 채널을 local 또는 global로 고정해서 얻는 trivial result가 아니다.
  source-aware role assignment가 필요하다.
- 논문 방어선:
  - "역할을 임의로 붙였으니 당연히 갈라진 것 아니냐"라는 비판에 대해,
    static/flipped role map이 실패한다는 ablation table로 답한다.

2026-05-06, cross-channel threshold robustness:

- `examples/physics/h0_readout/h0_cross_channel_threshold_robustness_gate.py`를 추가했다.
- full suite에 포함했다.
- threshold sweep:
  - 0.55: 6/6
  - 0.60: 6/6
  - 0.65: 6/6
  - 0.70: 6/6
  - 0.75: 6/6
  - 0.80: 6/6
- bridge count: 모든 threshold에서 0.
- min local selector: 0.830134
- max global selector: 0.003704
- full suite 결과: PASS.
- 판정: branch split은 특정 classification threshold 하나를 손으로 고른 결과가 아니다.
  local/high와 global/low가 selector space에서 충분히 멀리 분리되어 있다.

2026-05-06, paper figure table:

- `examples/physics/h0_readout/h0_paper_figure_table_gate.py`를 추가했다.
- full suite에 포함했다.
- paper-ready table:
  - DESI BAO: standard ruler, global, global/low
  - TDCOSMO+SLACS: time-delay lensing, global, global/low
  - TDCOSMO+SLACS+IFU: time-delay lensing, global, global/low
  - Pantheon+SH0ES: distance ladder, local, local/high
  - TDCOSMO+IFU: time-delay lensing, local, local/high
  - TDCOSMO-only: time-delay lensing, local, local/high
- local rows: 3
- global rows: 3
- full suite 결과: PASS.
- 판정: 논문 Figure 1 또는 Table 1 후보가 재현 가능한 gate로 고정됐다.

2026-05-06, paper claim audit:

- `examples/physics/h0_readout/h0_paper_claim_audit_gate.py`를 추가했다.
- full suite에 포함했다.
- claim map:
  - C1: TDCOSMO role metadata는 likelihood factor에서 재현된다.
  - C2: 공개 notebook sampler composition은 likelihood-factor graph와 일치한다.
  - C3: TDCOSMO는 SLACS population closure가 들어오면 local/high에서 global/low로 branch 전이한다.
  - C4: static TDCOSMO role map은 source-aware transition보다 나쁘다.
  - C5: DESI BAO는 H0 refit 이전에 global/low branch를 고른다.
  - C6: Pantheon+SH0ES는 H0 refit 이전에 local/high branch를 고른다.
  - C7: 6개 channel row가 source role만으로 local/high와 global/low family로 갈라진다.
  - C8: all-local, all-global, flipped role ablation은 cross-channel split을 만들지 못한다.
  - C9: threshold sweep에서도 split이 유지된다.
  - L1: full joint BAO/SN/TDCOSMO posterior refit은 아직 남은 한계다.
- 판정: 이제 논문에서 주장 가능한 문장과 아직 한계로 남겨야 할 문장이 gate 단위로 분리됐다.

2026-05-06, next expansion plan:

- 일반 설명에서 정리한 다음 확장축을 plan에 고정했다.
- 우선순위:
  - CMB acoustic-scale inference: early global horizon closure이므로 global/low branch 예상.
  - GW standard sirens: detector distance와 redshift anchor가 섞인 bridge/intermediate branch 예상.
  - quantum measurement/readout: 측정값이 대상의 절대값이 아니라 coupling/source role이 고른 branch일 가능성.
  - black-hole/holographic readout: local horizon observer와 global boundary observer의 readout 차이로 확장 가능.
- 즉시 진행 방향:
  - CMB를 branch-only gate로 먼저 추가한다.
  - 이후 public Planck covariance/likelihood ingestion으로 source-role-only 예측을 실제 data-facing test로 승격한다.

2026-05-06, CMB acoustic-scale global branch:

- `examples/physics/h0_readout/h0_cmb_acoustic_global_readout_gate.py`를 추가했다.
- CMB acoustic angle을 early global horizon closure로 놓는 branch-only payload를 만들었다.
- local nodes: 0.
- global nodes:
  - sound horizon
  - last scattering surface
  - early density closure
  - angular diameter distance to recombination
  - recombination history
- 예상:
  - q_F = 0
  - global/low branch
- full suite와 cross-channel table에 포함했다.
- 한계:
  - 아직 Planck likelihood/covariance를 직접 ingest한 것은 아니다.
  - 현재 단계는 CMB의 source role이 어떤 branch를 예측하는지 고정하는 예비 gate다.

2026-05-06, CMB source scout:

- `examples/physics/h0_readout/h0_cmb_source_scout_gate.py`를 추가했다.
- 목적:
  - CMB branch-only gate를 실제 Planck covariance/likelihood ingestion으로 올리기 전,
    어떤 공개 source를 기준으로 삼을지 고정한다.
- source targets:
  - Planck Legacy Archive cosmological parameter chains:
    `https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Cosmological_Parameters`
  - Planck 2018 likelihood paper:
    `https://arxiv.org/abs/1907.12875`
  - Planck 2018 cosmological parameters paper:
    `https://arxiv.org/abs/1807.06209`
- next ingest target:
  - `base_plikHM_TTTEEE_lowl_lowE_lensing` chain covariance.
- 판정:
  - CMB는 cross-channel table에 들어갔지만 아직 `claimable-with-scope`다.
  - 다음 단계에서 실제 Planck chain/covariance를 읽어야 CMB도 BAO/Pantheon처럼 data-facing gate가 된다.

2026-05-06, Planck PR3 CMB covariance adapter:

- IRSA Planck PR3 ancillary data에서 63 MB package를 내려받았다.
  - source: `https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/`
  - package: `COM_CosmoParams_base-plikHM-TTTEEE-lowl-lowE_R3.00.zip`
- 필요한 파일만 `h0_real_data/Planck_PR3/extract/` 아래 추출했다.
  - `base_plikHM_TTTEEE_lowl_lowE_lensing.paramnames`
  - `dist/base_plikHM_TTTEEE_lowl_lowE_lensing.covmat`
  - `dist/base_plikHM_TTTEEE_lowl_lowE_lensing.margestats`
- `examples/physics/h0_readout/h0_cmb_planck_covariance_adapter_gate.py`를 추가했다.
- 실제 Planck covariance adapter 결과:
  - parameters: 27
  - observable: `theta`
  - local nodes: 0
  - global nodes: 26
  - q_F = 0
  - H0 branch prediction: low/global
  - Planck marginal H0: 약 67.36 +/- 0.54
  - pull: 1 sigma 안쪽
- cross-channel row를 synthetic CMB acoustic row에서 Planck CMB covariance row로 교체했다.
- 판정:
  - CMB는 이제 `claimable-with-scope`에서 `claimable`에 가까워졌다.
  - 정확한 표현은 "Planck PR3 covariance selects the global/low branch under the acoustic-scale source-role map"이다.

2026-05-06, GW standard-siren bridge:

- `examples/physics/h0_readout/h0_gw_source_scout_gate.py`를 추가했다.
- `examples/physics/h0_readout/h0_gw_standard_siren_bridge_gate.py`를 추가했다.
- source targets:
  - GW170817 bright siren H0 posterior reference.
  - LIGO/Virgo GW170817 standard-siren provenance.
  - O4a standard-siren population extension.
- role interpretation:
  - GW amplitude/distance: local absolute-distance readout.
  - host/counterpart redshift and velocity correction: global/environment anchor.
  - expected readout: bridge/intermediate.
- result:
  - local nodes: 1
  - global nodes: 1
  - q_F = 0.5
  - H0 branch prediction: about 70.15
  - reference GW170817 H0: about 70.3 +/- 5.15
  - pull: near zero.
- 판정:
  - GW standard siren은 local/high 또는 global/low endpoint가 아니라 bridge branch를 고른다.
  - 이론이 양끝 분류뿐 아니라 중간 readout도 예측할 수 있음을 보여주는 첫 게이트다.

2026-05-06, three-family H0 readout table:

- `examples/physics/h0_readout/h0_three_family_readout_table_gate.py`를 추가했다.
- 목적:
  - endpoint-only paper table과 별도로, GW bridge까지 포함한 3-family table을 만든다.
- rows:
  - global/low:
    - DESI BAO
    - Planck CMB
    - TDCOSMO+SLACS
    - TDCOSMO+SLACS+IFU
  - bridge/intermediate:
    - GW170817 bright siren
  - local/high:
    - Pantheon+SH0ES
    - TDCOSMO+IFU
    - TDCOSMO-only
- 판정:
  - H0 readout law는 local/high와 global/low 양끝만 분류하는 법칙이 아니라,
    source coupling이 섞이면 intermediate readout도 낼 수 있다.

2026-05-06, paper package:

- `examples/physics/h0_readout/h0_paper_package_gate.py`를 추가했다.
- 목적:
  - 논문 그림과 한계 문장을 gate로 묶는다.
- Figure package:
  - Figure 1: endpoint source-role split.
  - Figure 2: three-family readout law.
- required limitations:
  - full joint BAO/SN/TDCOSMO posterior refit은 아직 남았다.
  - GW bridge는 event-level posterior samples가 아니라 source-role covariance abstraction이다.
  - CMB는 Planck PR3 parameter covariance를 읽은 것이고, Planck likelihood 전체를 새로 최적화한 것은 아니다.
- 판정:
  - 논문 초안의 최소 골격이 gate로 고정됐다.

2026-05-06, H0 paper draft spine:

- `docs/3_상수/12_H0_source_role_readout_paper_draft.md`를 추가했다.
- 포함 내용:
  - working title
  - abstract draft
  - core claim
  - Figure 1 endpoint split
  - Figure 2 three-family readout law
  - TDCOSMO, BAO/CMB, Pantheon+SH0ES, GW results narrative
  - ablations
  - required limitations
  - next tests
- `examples/physics/h0_readout/h0_paper_draft_gate.py`를 추가했다.
- 판정:
  - 이제 논문 초안의 주장/그림/한계/다음 테스트 spine이 gate로 고정됐다.

2026-05-06, H0 paper methods spine:

- `docs/3_상수/12_H0_source_role_readout_paper_draft.md`에 Methods 섹션을 추가했다.
- 정의한 것:
  - Fisher/covariance payload.
  - normalized edge reliability.
  - local endpoint conductance `C_L`.
  - global closure conductance `C_G`.
  - selector `q_F = C_L/(C_L+C_G)`.
  - branch readout `H0(q_F)`.
- `h0_paper_draft_gate.py`가 methods section과 핵심 기호를 검사하도록 강화했다.
- 판정:
  - 초안이 이제 단순 설명 문서가 아니라, 계산 가능한 method spine을 갖는다.

2026-05-06, H0 paper reviewer safeguards:

- `docs/3_상수/12_H0_source_role_readout_paper_draft.md`에 reviewer objections 섹션을 추가했다.
- 선제 대응:
  - H0 값을 보고 family를 나눈 것 아닌가?
  - local/global role assignment가 임의 아닌가?
  - threshold를 튜닝한 것 아닌가?
  - full cosmological inference가 아닌데 과장 아닌가?
  - GW bridge result가 아직 약하지 않은가?
- `h0_paper_draft_gate.py`가 reviewer-safeguard section과 핵심 방어 문구를 검사하도록 강화했다.
- 판정:
  - 초안은 이제 주장뿐 아니라 예상 반론과 방어선까지 포함한다.

2026-05-06, H0 paper provenance table:

- `examples/physics/h0_readout/h0_paper_provenance_table_gate.py`를 추가했다.
- `docs/3_상수/12_H0_source_role_readout_paper_draft.md`에 Data provenance 섹션을 추가했다.
- 목적:
  - 모든 H0 readout row가 public source, source role, primary gate, status와 연결되도록 한다.
- rows:
  - TDCOSMO-only
  - TDCOSMO+IFU
  - TDCOSMO+SLACS
  - TDCOSMO+SLACS+IFU
  - DESI BAO
  - Planck CMB
  - Pantheon+SH0ES
  - GW170817 bright siren
- 판정:
  - 논문 초안이 결과 표뿐 아니라 data provenance table도 갖는다.

2026-05-06, H0 paper numeric results table:

- `examples/physics/h0_readout/h0_paper_numeric_results_gate.py`를 추가했다.
- `docs/3_상수/12_H0_source_role_readout_paper_draft.md`에 Numeric results 섹션을 추가했다.
- table scope:
  - selector `q_F`
  - readout family
  - H0 readout
  - reference status
  - branch-only rows는 branch-only로 표시한다.
- 판정:
  - 논문 초안은 이제 provenance table과 numeric result table을 모두 가진다.

2026-05-06, H0 paper figure captions:

- Added `examples/physics/h0_readout/h0_paper_caption_gate.py`.
- Expanded the paper draft Figure 1 and Figure 2 captions.
- Figure 1 now states the endpoint split and the key ordering: source roles are assigned before H0 comparison.
- Figure 2 now states the bridge/intermediate interpretation and explicitly warns that the diagram is not a joint posterior fit.
- Added the caption gate to the full suite and claim audit as C19.

2026-05-06, H0 paper plain significance:

- Added `examples/physics/h0_readout/h0_paper_plain_significance_gate.py`.
- Added `## Plain-language significance` to the H0 paper draft.
- The new section explains the result without equations: CMB/BAO close through global rulers, distance ladders close through nearby anchors, and standard sirens sit between distance and redshift anchoring.
- Added the significance gate to the full suite and claim audit as C18.

2026-05-06, H0 paper prediction ledger:

- Added `examples/physics/h0_readout/h0_paper_prediction_ledger_gate.py`.
- The ledger turns the source-role readout into future-facing tests: GW event-level posteriors, TRGB/JAGB/CCHP ladders, BAO+SN inverse-distance-ladder covariance, CMB covariance variants, and alternative TDCOSMO notebooks/chains.
- Updated the paper draft with `## Predictions and falsification`, including explicit ways the proposal could weaken or fail after source roles are fixed first.
- Wired the prediction ledger into `h0_fisher_io_full_suite.py` and `h0_paper_claim_audit_gate.py`.
