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

- `examples/physics/h0_dataset_falsification_gate.py`를 추가했다.
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

- `examples/physics/h0_late_branch_correction_gate.py`를 추가했다.
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

- `examples/physics/h0_readout_selector_gate.py`를 추가했다.
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

- `examples/physics/h0_channel_topology_gate.py`를 추가했다.
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

- `examples/physics/h0_prospective_channel_gate.py`를 추가했다.
- 기존 selector table에 쓰지 않은 외부 채널에 대해 \(H_0\) 값을 맞추기 전에 \(L:G\)를 먼저 선언했다.
- 예측:
  - H0LiCOW/TDCOSMO power-law lenses: \(L:G=1:0\), \(q=1\), \(H_0^{\rm pred}=73.181\), 관측 \(73.3\pm1.75\), pull \(-0.07\).
  - TDCOSMO+SLACS hierarchical lenses: \(1:3\), \(q=0.25\), \(H_0^{\rm pred}=68.684\), 관측 \(67.4\pm3.65\), pull \(+0.35\).
  - Megamaser Cosmology Project: \(1:0\), \(q=1\), \(H_0^{\rm pred}=73.181\), 관측 \(73.9\pm3.0\), pull \(-0.24\).
  - GW170817 bright siren: \(1:1\), \(q=0.5\), \(H_0^{\rm pred}=70.151\), 관측 \(70.3\pm5.15\), pull \(-0.03\).
  - O4a dark+bright sirens: \(1:1\), \(q=0.5\), \(H_0^{\rm pred}=70.151\), 관측 \(68.0\pm4.1\), pull \(+0.52\).
- 합산 \(\chi^2/{\rm dof}=0.462/5\).
- 판정: \(q_{\rm topo}\)는 외부채널에서도 high/local, low/hierarchical, middle/GW 구조를 재현한다. 다음 과제는 \(L,G\)를 손배정하지 않고 covariance graph에서 자동 계산하는 것.
