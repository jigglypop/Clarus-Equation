# U4/U5 배경우주·성장·\(H_0\) forward route 독립 연구

Status: COMPLETE

이 문서는 examples/physics/hubble_tension.py, examples/physics/cosmology.py,
examples/physics/ce_residual_forward_model.py, 관련 테스트와
reality_stone/python/reality_stone/clarus/constants.py를 대상으로 한 독립 수학·코드
경로 감사다. 제품·정본 파일은 수정하지 않았다. 아래의 합성 \(\theta_*\) 역산은
solver 검증일 뿐 관측 적합이나 CE의 \(H_0\) 예측이 아니다.

## 1. 결론

현재 저장소에는 U4/U5를 끝까지 통과하는 단일 경로가 없다. 그러나 기존 작업을
폐기할 이유도 없다. 다음 세 층으로 합치면 된다.

1. ce_residual_forward_model.py의 baryon-loaded sound-horizon, 외부 재결합 history
   hash/provenance, BAO covariance 구현을 보존한다.
2. cosmology.py의 거리·성장 API는 compatibility facade로 남기되, radiation을 포함한
   하나의 flat-FLRW background와 비균일 격자 적분기로 내부 구현을 교체한다.
3. hubble_tension.py의 \(\epsilon\)-flow는 폐기하지 않고 legacy/toy route로 보존한다.
   다만 그 route가 background를 구동하려면 별도의 공변 stress/current가 필요하며,
   그것이 생기기 전에는 \(H_0\) 수치 readout을 내지 않는다.

정확한 배경식과 수치 적분기는 지금 닫을 수 있다. 실제 \(H_0\) readout은 (A) full
Boltzmann likelihood, (B) 검증 범위를 고정한 compressed acoustic-scale, (C) 독립
\(r_d\) calibration을 가진 late-time inverse ladder의 세 경로를 병렬로 유지하는 것이
최소 안전 구조다. 기본 승격 경로는 A이고, B는 A에 대한 교차검증을 통과한 제한적
근사, C는 독립 검산 경로다.

## 2. 현재 판본 지도

| 위치 | 현재 유효 자산 | U4/U5에서의 한계 | 통합 처리 |
|---|---|---|---|
| reality_stone/.../constants.py:24-29 | CE의 rounded composition snapshot | cosmology solver가 아니며 세 수의 합이 \(1.0001\) | 원시 snapshot과 background closure를 분리해 보존 |
| examples/physics/cosmology_ratio_audit.py | 상수 로딩과 ratio 비교 | 자체 coverage는 background/growth가 없다고 기록 | compatibility report로 유지 |
| examples/physics/cosmology.py | 거리, 성장 ODE, CLI, 보간 유틸 | radiation 없음; public Simpson과 growth가 비균일 격자를 잘못 처리 | facade로 유지하고 공통 kernel 호출 |
| examples/physics/ce_residual_forward_model.py | CE boundary, BAO, covariance, baryon loading, \(x_e(z)\) adapter | 후기 \(E(a)\)에는 radiation이 없고 초기 \(H(a)\)와 정규화가 다름; \(\theta_*\) solver 없음 | canonical forward-model shell 후보 |
| examples/physics/hubble_tension.py | \(\epsilon\)-flow 연구 아이디어와 parameter scans | Ricci, conservation, acoustic physics, 입력 활성성에 완전 반례 | 이름을 보존한 legacy route; 예측 API에서는 fail-closed |
| tests/test_ce_residual_forward_model.py | BAO/거리/성장/초기 \(r_d\) 회귀 | radiation/matter/de Sitter 극한과 비균일 grid 시험 없음 | 공통 kernel 극한 시험을 추가 |
| tests/test_recombination_drag_adapter.py | 합성 \(x_e\) table parser·hash·drag root 검증 | 실제 solver export와 \(z_*\)/visibility 검증 없음 | full backend fixture가 들어올 때 그대로 확장 |

reality_stone/python/reality_stone/clarus에는 위 rounded constants 외에 독립적인
FLRW/background/growth cosmology 구현이 없다. 따라서 “reality_stone cosmology”와
examples의 서로 다른 solver를 병합하는 문제가 아니라, constants ledger와 examples의
세 판본을 하나의 forward kernel에 연결하는 문제다.

## 3. 정확한 flat-FLRW 식

### 3.1 물리 밀도와 평탄 closure

\(h=H_0/(100\,{\rm km\,s^{-1}Mpc^{-1}})\),
\(\omega_i=\Omega_{i0}h^2\)로 둔다. massless-neutrino 근사와 CPL dark energy에서는

\[
E^2(a)=\frac{H^2(a)}{H_0^2}
=\Omega_{r0}a^{-4}+\Omega_{m0}a^{-3}
+\Omega_{{\rm de},0}F_{\rm de}(a),
\]

\[
F_{\rm de}(a)
=a^{-3(1+w_0+w_a)}e^{3w_a(a-1)},\qquad
\Omega_{r0}+\Omega_{m0}+\Omega_{{\rm de},0}=1.
\]

후보 \(h\)를 바꾸면서 초기우주 물리 밀도를 고정할 때는 다음 형태가 오류를 줄인다.

\[
E^2(a;h)=\frac{\omega_ra^{-4}+\omega_ma^{-3}}{h^2}
+\left(1-\frac{\omega_r+\omega_m}{h^2}\right)F_{\rm de}(a).
\]

즉 \(h\)를 찾는 동안 고정해야 할 것은 \(\Omega_r\)가 아니라
\(\omega_\gamma(T_{\rm CMB})\), \(N_{\rm eff}\), \(\omega_b\), \(\omega_c\)다.

massive neutrino를 허용하면 단순한 \(\Omega_\nu a^{-4}\) 또는
\(\Omega_\nu a^{-3}\)로는 충분하지 않다. backend는
\(X_\nu(a)=\rho_\nu(a)/\rho_{\rm crit,100}\)와
\(Y_\nu(a)=p_\nu(a)/\rho_{\rm crit,100}\)를 함께 제공해야 한다. 배경 적분은
\(X_\nu(a)\), Ricci trace는 \(X_\nu(a)-3Y_\nu(a)\)를 사용한다. full Boltzmann
route에서는 이 계산을 CLASS/CAMB에 맡긴다.

현재 rounded CE snapshot은

\[
0.0487+0.2623+0.6891=1.0001
\]

이다. 여기에 기본 \(h=0.674,T_{\rm CMB}=2.7255\,{\rm K},N_{\rm eff}=3.044\)의
\(\Omega_r=9.1923323\times10^{-5}\)까지 더하면 \(E^2(1)=1.0001919233\)이다.
이를 조용히 정규화하면 안 된다. 통합 ledger는 다음 둘을 별도 필드로 둬야 한다.

- composition_snapshot: 과거/rounded CE 세 수를 그대로 보존한다.
- background_closure: exact unrounded 값을 공급하거나, flat route라면
  \(\Omega_{{\rm de},0}=1-\Omega_m-\Omega_r\)로 산출한다. 다른 선택은 명시적인
  curvature route여야 한다.

### 3.2 \(d\ln H/d\ln a\)와 Ricci scalar

\(N=\ln a\)라 하면

\[
\frac{d\ln H}{dN}=\frac{1}{2E^2}\frac{dE^2}{dN},
\]

\[
\frac{dE^2}{dN}
=-4\Omega_{r0}a^{-4}-3\Omega_{m0}a^{-3}
-3[1+w(a)]\Omega_{{\rm de},0}F_{\rm de}(a).
\]

부호 규약 \(R=+6(\dot H+2H^2)\)에서

\[
\boxed{\frac{R}{H^2}=6\left(2+\frac{d\ln H}{dN}\right)}.
\]

Einstein trace로 독립 계산하면

\[
\frac{R}{H^2}
=3\Omega_m(a)+3[1-3w(a)]\Omega_{\rm de}(a)
+3\Omega_\nu(a)[1-3w_\nu(a)].
\]

photon과 massless neutrino의 trace는 0이다. \(\Lambda\)CDM massless-radiation
특수형은

\[
\boxed{\frac{R}{H^2}=3\Omega_m+12\Omega_\Lambda
=12-9\Omega_m-12\Omega_r}.
\]

따라서 radiation, matter, de Sitter 극한은 각각 \(0,3,12\)다.

### 3.3 기존 running-\(\epsilon\) 배경에 필요한 추가 항

hubble_tension.py는

\[
M(a)=\frac{1-\epsilon(a)}2,\qquad
L(a)=\frac{1+\epsilon(a)}2,
\]

\[
E^2=M(a)a^{-3}+L(a)+r a^{-4}
\]

를 사용한다. 이것을 하나의 대수적 \(H(a)\) 정의로 받아들이더라도 정확한 Ricci는

\[
\frac{R}{H^2}
=12-9\frac{Ma^{-3}}{E^2}-12\frac{ra^{-4}}{E^2}
+\frac{3\epsilon'}{2E^2}(1-a^{-3}),
\qquad \epsilon'=\frac{d\epsilon}{dN}.
\]

현재 식 \(12-9\Omega_m\)에는 radiation 항과 running-derivative 항이 모두 없다.
더 근본적으로, \(\rho_m\propto M(a)a^{-3}\), \(\rho_\Lambda\propto L(a)\)라 두면

\[
\frac{Q_m+Q_\Lambda}{H\rho_{\rm crit,0}}
=M'a^{-3}+L'=\frac{\epsilon'}2(1-a^{-3})\ne0
\]

가 일반적이다. 즉 별도의 flow-field stress나 상호작용 current가 이 항을 보상하지
않으면 total conservation과 Bianchi identity를 만족하지 않는다. 해결책은
\(\epsilon\)-flow를 내리는 것이 아니라, 다음 중 하나를 완성하는 것이다.

- 공변 action에서 \(T^{\mu\nu}_{\epsilon}\)와 교환 current \(Q_i^\nu\)를 함께 유도한다.
- 또는 \(\epsilon\)을 background coefficient가 아닌 observable/branch label로만 두고,
  보존되는 fluid background를 별도로 푼다.

그 전까지 이 route는 OPEN_DYNAMICAL_ROUTE이며 \(H_0\) 숫자를 반환하지 않는다.

### 3.4 거리

평탄 배경에서

\[
\chi(z)=D_M(z)=\frac{c}{H_0}\int_0^z\frac{dz'}{E(z')}
=\frac{c}{H_0}\int_{a(z)}^1\frac{da}{a^2E(a)},
\]

\[
D_A=\frac{D_M}{1+z},\qquad D_L=(1+z)D_M,\qquad
D_H=\frac{c}{H(z)}.
\]

comoving \(r_s\)를 쓰는 음향각의 분모는 \(D_A\)가 아니라 \(D_M\)다.
hubble_tension.py 본문 코드는 실제로 comoving distance를 써서 이 부분은 맞지만,
파일 상단 설명의 \(r_s/D_A\) 표기는 고쳐야 한다.

### 3.5 sound horizon과 재결합 입력

\[
R_b(a)=\frac{3\rho_b}{4\rho_\gamma}
=\frac{3\omega_b}{4\omega_\gamma}a,
\qquad
c_s(a)=\frac{c}{\sqrt{3(1+R_b(a))}},
\]

\[
r_s(z)=\int_0^{a(z)}\frac{c_s(a)}{a^2H(a)}\,da.
\]

CMB acoustic angle과 BAO standard ruler는 구분해야 한다.

\[
\theta_*=\frac{r_s(z_*)}{D_M(z_*)},\qquad
r_d=r_s(z_d).
\]

\(z_*\)는 photon visibility/decoupling, \(z_d\)는 baryon drag optical-depth
조건에서 나온다. 필요한 입력은 최소 \(\omega_b,\omega_c,T_{\rm CMB},N_{\rm eff}\),
neutrino masses/hierarchy, \(Y_p\), recombination model이다. 기존 hashed \(x_e(z)\)
adapter는 \(z_d\) 자산으로 보존할 수 있지만, \(z_*\), visibility와 CMB spectra를
제공하지는 않는다.

radiation-only, constant-\(c_s\) 극한에서는

\[
r_s(a_*)\frac{H_0}{c}=\frac{a_*}{\sqrt{3\Omega_r}},
\]

matter-only 극한에서는

\[
r_s(a_*)\frac{H_0}{c}=\frac{2\sqrt{a_*}}{\sqrt{3\Omega_m}}.
\]

이 두 식은 verifier에서 직접 대조했다.

### 3.6 \(\theta_*\)의 implicit \(H_0\) solve

관측 또는 합성 target \(\theta_*^{\rm target}\)에 대해

\[
F(h)=\log\frac{\theta_*^{\rm forward}
(h;\omega_b,\omega_c,T_{\rm CMB},N_{\rm eff},Y_p,\ldots)}
{\theta_*^{\rm target}}=0
\]

을 푼다. 후보 \(h\)마다 flat closure, massive-neutrino background,
recombination과 distance를 함께 갱신해야 한다. bracket 양 끝의 부호, 유한성,
단조성, root tolerance와 grid/backend convergence를 검사하고, bracket이 없으면
INCOMPLETE/NO_ROOT로 종료한다. density를 \(0.999\)로 clamp하거나 NaN을 성공
출력으로 바꾸면 안 된다.

## 4. 성장식과 비균일 격자

smooth dark energy, sub-horizon, scale-independent \(\mu\) 근사에서 pressureless
clustering matter의 성장식은

\[
D''+\left(2+\frac{d\ln H}{dN}\right)D'
-\frac32\mu(a,k)\Omega_m(a)D=0,
\qquad N=\ln a.
\]

일반 비균일 grid에서는 각 구간마다
\(\Delta N_i=N_{i+1}-N_i\)를 사용해야 한다. 현재 cosmology.solve_growth는 전체
평균 dln을 모든 구간에 사용하므로 함수 API에 비균일 a_grid를 넘기면 틀린다.
현재 CLI의 logspace는 우연히 uniform-\(N\)이라 이 결함을 가린다.

matter+\(\Lambda\) GR에서는 독립 quadrature가 있다.

\[
D_+(a)=\frac{5\Omega_{m0}}2E(a)
\int_0^a\frac{da'}{a'^3E^3(a')}.
\]

이를 정규화해 RK4와 교차검증할 수 있다. radiation era에는 \(D=a,D'=D\) 초기화가
정확하지 않다. 다음 정책이 필요하다.

- late-time smooth-DE solver: matter domination 안에서 초기화하거나 위 quadrature로
  growing mode를 공급한다.
- recombination/equality 이전 또는 massive-neutrino/scale-dependent 성장: full
  Einstein--Boltzmann transfer function을 사용한다.
- \(\mu(a,k)\)가 scale-dependent이면 하나의 \(D(a)\) 배열로 축약하지 않는다.

공용 simpson(y,x)에도 두 결함이 있다. 비균일 \(x\)인데 endpoint 평균 간격을
사용하며, 점 수가 짝수면 마지막 점과 마지막 구간을 조용히 버린다. 이는
compute_s_of_a가 logspace a_grid를 넘기는 경로에서 실제 오류다. 최소 교체는
각 구간 \([x_i,x_{i+1}]\)마다 midpoint를 평가하는 local Simpson이고, midpoint를
구할 수 없는 tabulated data는 명시적 trapezoid 또는 검증된 nonuniform rule을
선택해야 한다.

## 5. 현재 U4/U5 결함·미사용 입력 원장

| ID | 위치 | 완전 반례/영향 | 최소 교정 |
|---|---|---|---|
| BG-01 | hubble_tension.py:118-123 | radiation era에도 \(R/H^2\to12\); 정확한 극한은 0 | \(-12\Omega_r\)와 running derivative 포함 |
| BG-02 | hubble_tension.py:190-201 | 시간의존 \(M,L\)을 넣지만 stress/current 없이 total continuity 불성립 | 공변 flow stress/current 또는 label-only route |
| BG-03 | hubble_tension.py:104-111,197-198 | \(M+L=1\) 뒤 radiation을 더해 \(E(1)>1\) | 한 번만 적용되는 flat closure |
| BG-04 | ce_residual_forward_model.py:926-955 | 후기 background/growth가 radiation·neutrino를 완전히 생략 | 공용 background에서 동일 \(E,d\ln H\) 사용 |
| BG-05 | ce_residual_forward_model.py:416-427,546-548 | 초기 background는 raw CE 세 성분+radiation을 더해 \(H(0)\ne H_0\) | physical-density closure로 통일 |
| BG-06 | CEForwardParams.density_norm | matter+DE만 재정규화하여 early와 late가 서로 다른 우주 | 원시 composition과 closure 객체 분리 |
| NUM-01 | cosmology.py:31-54 | nonuniform \(x\)를 uniform으로 적분 | per-interval midpoint Simpson/adaptive quadrature |
| NUM-02 | 같은 함수 | 짝수 점에서 마지막 구간을 버림 | 전 구간 사용; 짝수 입력도 명시적으로 처리 |
| NUM-03 | cosmology.py:157-203 | 비균일 growth grid에 평균 step 사용 | local \(\Delta N_i\) RK4 또는 adaptive ODE |
| GR-01 | 두 growth solver | radiation을 넣으면 \(a_{\min}=10^{-3}\)의 \(D=a\) 초기조건은 정확하지 않음 | growing-mode initializer/Heath/full transfer |
| GR-02 | residual_mu_of_a, 단일 \(D(a)\) | massive neutrino나 scale-dependent \(\mu\)에는 \(D(a,k)\) 필요 | full Boltzmann 또는 \(k\)-grid ODE |
| AC-01 | lcdm_theta_star_for_h(..., om_b_h2, ...) | AST와 수치 모두 om_b_h2가 미사용임을 확인 | \(R_b,c_s,z_*\)에 실제 연결 |
| AC-02 | hubble_tension.sound_horizon | \(c_s=c/\sqrt3\) 고정 | baryon-loaded \(c_s(a)\) 사용 |
| AC-03 | 같은 파일 | \(z_*\) 고정, recombination·\(Y_p\)·\(N_{\rm eff}\) 비활성 | backend가 \(z_*,z_d\)를 산출 |
| AC-04 | 같은 파일 | CMB \(r_s(z_*)\)와 BAO \(r_d=r_s(z_d)\) 구분이 API에 없음 | 두 타입/필드 분리 |
| H0-01 | lcdm_theta_star_for_h | \(h\)를 바꾸어도 fixed OMEGA_R0; 고정 \(T_{CMB}\) 조건 위반 | fixed \(\omega_r\), 후보별 \(\Omega_r=\omega_r/h^2\) |
| H0-02 | 같은 함수 :285 | h2 계산 결과가 미사용; ruff F841 | 제거가 아니라 공용 background로 대체 |
| H0-03 | 같은 함수 :262 | \(\omega_b\) docstring과 실행이 불일치 | input-activity test를 gate로 고정 |
| H0-04 | extract_h0_cmb | h0_true로 \(\omega_m\)를 만든 뒤 \(H_0\)를 “추출”; 독립 readout이 아님 | 독립 physical-density input ledger |
| H0-05 | 같은 함수 | density clamp가 bracket 밖 모델을 다른 모델로 바꿈 | domain error/fail-closed |
| H0-06 | 같은 함수 | absolute \(\theta\) tolerance \(10^{-6}\)만 사용 | log-residual+root/grid/backend 오차 예산 |
| H0-07 | --optimize, --target-tension | 관측 gap을 직접 최소화하는 target-aware branch | 탐색 연구로 격리; prediction path에서 금지 |
| H0-08 | CLI 전반 | no-root/물리 미완성도 정상 exit 가능 | report와 scientific gate exit 분리 |
| CODE-01 | hubble_tension.py:78 | Background import 미사용 | facade 통합 시 실제 공용 Background 사용 |
| CODE-02 | U4/U5 legacy | 직접 pytest가 없어 위 반례가 재발 가능 | analytic/input-activity/compatibility 회귀 추가 |

ce_residual_forward_model.py의 baryon loading과 drag-history adapter 자체는 위 결함이
아니다. 문제는 그것들이 후기 background와 \(\theta_*\) readout에 아직 연결되지 않은
것이다.

## 6. 최소 통합 architecture

### 6.1 데이터 구조

1. CosmologyInputLedger
   - 값, 단위, 역할(derived, external, model_assumption), source, hash/version.
   - early physical inputs와 late fractions를 동시에 저장하되 중복 지정 시 일치 검사.
2. CompositionSnapshot
   - CE exact chain, rounded public snapshot, historical alternatives를 이름 있는 판본으로
     보존한다. 어느 것도 조용히 덮어쓰지 않는다.
3. FlatFLRWBackground
   - closure policy, neutrino provider, \(E^2,d\ln H/dN,R/H^2,\Omega_i(a)\), 거리.
   - kinematic Ricci와 stress-trace Ricci를 모두 계산하고 불일치하면 fail한다.
4. ThermalHistoryBackend
   - EisensteinHuCompressed, HashedXeHistory, ClassBackend, CambBackend.
   - 공통 결과는 \(z_*,z_d,r_s(z_*),r_d\), visibility/drag provenance와 오차다.
5. GrowthBackend
   - LateSmoothGrowth(nonuniform RK4 + Heath cross-check)와
     BoltzmannTransferGrowth를 구분한다.
6. H0Readout
   - FullCMBLikelihood, CompressedTheta, InverseDistanceLadder의 tagged result.
   - prediction, conditional_inference, diagnostic 지위를 결과 객체에 강제한다.

### 6.2 입력 활성성

| 입력 | 최소 사용처 |
|---|---|
| \(h\) | flat closure, late \(E(a)\), 거리, root |
| \(\omega_b\) | \(\Omega_b\), baryon loading, recombination |
| \(\omega_c\) | equality, \(H(a)\), transfer/growth |
| \(T_{\rm CMB}\) | \(\omega_\gamma\), recombination |
| \(N_{\rm eff},m_\nu\) | radiation/neutrino background와 perturbation |
| \(Y_p\) | \(n_H,x_e(z)\), visibility/drag |
| \(A_s,n_s,\tau\) | full CMB spectra/likelihood; background-only route에서는 명시적 inactive |
| \(w_0,w_a\) | flat closure 이후 late background/distance |

각 active input에 finite perturbation test를 두어 출력이 불변이면 실패시킨다. 반대로
inactive input은 결과 metadata에 inactive_for_this_route라고 기록한다.

### 6.3 과거판본 보존과 compatibility

- 기존 import/function 이름은 wrapper로 남긴다.
- cosmology.Background는 새 background의 radiation=0 compatibility constructor가
  되며 경고와 provenance를 반환한다.
- hubble_tension의 scan/flow 함수는 LegacyEpsilonFlowRoute 아래 유지한다.
  옛 숫자를 재현하는 reproduce_historical=True는 허용하되 결과 지위는
  HISTORICAL_TOY, prediction gate는 nonzero다.
- ce_residual_forward_model을 canonical shell로 선택하되, raw CE ratio와 flat
  closure의 차이를 출력한다.
- 이름이 같은 서로 다른 기본값은 삭제하지 않고 named configuration으로 바꾼다.

## 7. 구조적으로 다른 \(H_0\) readout 세 경로

### Route A — full Boltzmann + CMB likelihood (기본 승격 경로)

**계산.** CLASS 또는 CAMB에 physical densities, neutrino sector, helium,
recombination, primordial spectrum, reionization과 CE background/action을 전달한다.
단일 \(\theta_*\)가 아니라 TT/TE/EE, lensing과 선택한 likelihood를 함께 계산한다.

**자유도.** 기본 후보 벡터는
\((h,\omega_b,\omega_c,A_s,n_s,\tau,\Sigma m_\nu,N_{\rm eff},Y_p,w_0,w_a,
\text{nuisance})\)다. CE가 독립적으로 닫은 항만 고정하며, composition ratio만으로
physical density normalization이 생겼다고 세지 않는다.

**provenance.** solver family/version/commit, compiler와 precision settings, input
file hash, recombination backend, likelihood release/hash, masks, covariance,
foreground/calibration nuisance, chain seed와 convergence가 필요하다. CLASS의 공식
explanatory.ini와 CAMB의 set_cosmology는 각각 \(\omega_i\), neutrino,
\(Y_p\), \(H_0\)/\(\theta_*\) 입력을 명시적으로 지원한다.

**교차 관측량.** acoustic peak 위치뿐 아니라 peak-height ratio, damping tail,
lensing, matter transfer, BAO \(D/r_d\), BBN과 late growth를 동시에 본다.

**kill test.** (1) 동일 입력의 vanilla CLASS/CAMB background와 CE adapter가 합의하지
않음, (2) independent solver 두 개의 held-out grid가 사전 오차 예산 밖에서 불일치,
(3) 전체 spectra likelihood가 preregistered 기준을 실패, (4) CE가 고정했다고 한
parameter를 풀었을 때 posterior가 그 값을 배제하면 route를 죽인다. solver process
exit 0만으로 통과시키지 않는다.

### Route B — controlled compressed \(r_s+\theta_*\) model

**계산.** 공용 flat background, baryon-loaded \(c_s\), precision \(z_*\) adapter 또는
유효범위를 고정한 fit으로 \(r_s(z_*)\)와 \(D_M(z_*)\)를 계산하고 implicit \(h\) root를
푼다. BAO에는 별도로 \(z_d,r_d\)를 사용한다.

**자유도.** 최소 \(h\); \(\omega_b,\omega_c,N_{\rm eff},m_\nu,Y_p,w_0,w_a\) 중
외부/CE가 독립적으로 고정하지 않은 것은 모두 nuisance/free parameter다.

**provenance.** compressed \(\theta_*\)의 정의와 covariance, recombination fit 또는
hashed history, fit calibration domain, full-solver 비교 grid와 최대 오차가 필요하다.
Eisenstein--Hu \(z_d\)는 BAO drag 근사이며 \(z_*\) 대용이 아니다.

**교차 관측량.** full solver의 \(z_*,z_d,r_s,r_d,D_M,\theta_*\), 그리고 여러
\(\omega_b,\omega_c,N_{eff}\) held-out 점에서 비교한다. 결과 \(h\)는 Route A와도
비교한다.

**kill test.** (1) active-input perturbation 중 하나라도 불변, (2) grid doubling/root
tolerance 변화가 오차 예산 초과, (3) full solver held-out discrepancy가 선언된
compression error 초과, (4) root가 bracket되지 않거나 복수 root의 선택 규칙이 없음,
(5) \(\theta_*\)는 맞지만 peak heights/damping 교차검증을 실패하면 적용 범위를
축소하거나 route를 죽인다. 이 경로의 결과는 통과해도 conditional compressed
inference이지 zero-parameter prediction이 아니다.

### Route C — late-time inverse distance ladder

**계산.** SN relative distance와 anisotropic BAO
\((D_M/r_d,D_H/r_d)\)의 full covariance를 결합한다. 독립 early-time/BBN
calibration으로 \(r_d\)를 공급할 때만 \(H_0\)를 분리한다.

**식별성.** BAO는

\[
\frac{D_M}{r_d}=\frac{c}{H_0r_d}\int\frac{dz}{E(z)},\qquad
\frac{D_H}{r_d}=\frac{c}{H_0r_dE(z)}
\]

이므로 calibration이 없으면 \(H_0r_d\)만 잰다. SN의 absolute magnitude/intercept도
\(H_0\)와 퇴화한다. 따라서 \(r_d\) 또는 다른 absolute calibration 없이 유한한
\(H_0\) posterior가 나오면 숨은 prior/버그다.

**자유도.** \((H_0r_d,\Omega_m,w_0,w_a,\mathcal M_{SN})\), dataset nuisance와
systematics; 독립 \(r_d\) calibration을 넣을 때 \(H_0\)가 파생된다.

**provenance.** BAO mean/full covariance와 convention, SN light-curve/covariance와
calibration, BBN abundance/nuclear-rate likelihood 또는 full early solver,
중복 object와 cross-covariance가 필요하다.

**교차 관측량.** BAO redshift/species split, SN survey split, cosmic chronometers 또는
time-delay 같은 absolute late probes, Route A/B의 \(r_d,H_0\)와 비교한다.

**kill test.** \(r_d\) calibration을 제거한 null run에서 \(H_0\)가 식별되면 즉시
실패한다. dataset split instability, covariance 제거 시 비정상적 정밀도 증가,
SN+BAO가 공통 expansion history를 preregistered criterion에서 거부하는 경우도
route를 죽인다.

## 8. 독립 verifier

실행:

    python _workspace/ce/cosmology-full-closure-unification-20260815/artifacts/verify_background_h0_forward.py

기록:

    nonuniform_quadrature PASS
    flrw_limits PASS
    ricci_trace_identity PASS
    sound_horizon_limits PASS
    growth_nonuniform_grid_convergence PASS 7.554e-04 5.578e-05 3.791e-06
    growth_heath_crosscheck PASS
    synthetic_theta_h_recovery PASS h=0.6800000000
    controlled_omega_b_input_active PASS delta_theta=3.356928e-04
    legacy_omega_b_h2_unused True
    legacy_local_h2_assignment_unused True
    STATUS COMPLETE: numerical verifier only; no observational prediction

검증 범위:

- arbitrary nonuniform grid의 cubic 적분을 \(3\times10^{-15}\) 이내로 재현.
- flat normalization과 radiation/matter/de Sitter의 \(R/H^2=(0,3,12)\).
- kinematic Ricci와 stress-trace Ricci를 \(a=10^{-6}\)부터 1까지 교차검증.
- radiation/matter analytic sound-horizon와 matter/de Sitter analytic distance.
- warped nonuniform \(\ln a\) grid에서 EdS 성장의 grid refinement.
- matter+\(\Lambda\) RK4와 Heath quadrature의 일치.
- 모든 early input이 활성인 합성 \(\theta_*\)에서 주입한 \(h=0.68\) 복원.
- AST로 legacy \(\omega_bh^2\) 인자와 local h2 assignment의 미사용 확인.

verifier는 standard library만 사용한다. 실제 recombination accuracy나 관측
likelihood를 검증하지 않으며 그 결과를 예측으로 승격할 수 없다.

## 9. gate 이후 최소 구현 순서

1. 공용 FlatFLRWBackground와 physical-density closure를 추가하고 U4 analytic
   limits를 먼저 고정한다.
2. nonuniform quadrature/RK4를 추가한 뒤 기존 uniform-grid 결과와 compatibility
   snapshot을 비교한다.
3. CE raw ratios, rounded snapshot, flat remainder를 하나의 provenance ledger에
   병렬 보존한다.
4. 기존 early sound-horizon와 hashed \(x_e\) adapter를 공용 background에 연결한다.
5. \(z_*\)와 \(z_d\) 결과 타입을 분리하고 Route B의 fail-closed synthetic solver를
   먼저 구현한다.
6. CLASS/CAMB adapter와 immutable input/output hash를 추가해 Route A를 연다.
7. 기존 DESI covariance 자산과 SN/independent-\(r_d\) adapter로 Route C를 연다.
8. 세 route의 cross-observable/kill test가 닫힌 뒤에만 \(H_0\)를 observational
   result로 출력한다. 그 전에는 legacy 숫자를 삭제하지 않고 HISTORICAL_TOY로
   보존한다.

## 10. 1차·공식 참고 경로

- CLASS 공식 입력 규약 및 parameter definitions:
  <https://github.com/lesgourg/class_public/blob/master/explanatory.ini>
- CLASS 공식 저장소: <https://github.com/lesgourg/class_public>
- CAMB 공식 parameter API와 exact thetastar \(H_0\) solve:
  <https://camb.readthedocs.io/en/latest/model.html>
- Eisenstein & Hu, baryon drag/transfer fitting formula 원 논문:
  <https://arxiv.org/abs/astro-ph/9709112>
- Heath, matter+\(\Lambda\) growth quadrature, MNRAS 179 (1977) 351:
  <https://doi.org/10.1093/mnras/179.3.351>
- massive-neutrino background/perturbation review:
  <https://arxiv.org/abs/astro-ph/0603494>
- DES-SN + DESI inverse-distance-ladder 방법의 공개 논문:
  <https://arxiv.org/abs/2406.05049>

## 11. 최종 판정표

| 항목 | 현재 | 이 artifact에서 닫힌 것 | 승격 전 남은 것 |
|---|---|---|---|
| U4 flat background | 서로 다른 early/late 식 | radiation 포함 exact 식과 closure 규칙 | canonical code 통합 |
| U4 Ricci | legacy 완전 반례 | kinematic/trace 식과 3극한 검증 | running flow stress/current |
| U4 quadrature | nonuniform/짝수점 결함 | arbitrary-grid local Simpson 검증 | product 교체와 회귀 |
| U4 growth | uniform-\(N\)에만 우연히 유효 | local-step RK4, EdS/Heath 검증 | radiation-era/full transfer backend |
| U5 sound horizon | 유효 자산이 고립 | baryon loading·\(z_*/z_d\) 경계 확정 | 실제 recombination/full solver |
| U5 implicit \(H_0\) | \(\omega_b\) 미사용 toy | input-active synthetic root 검증 | 관측 likelihood/provenance |
| U5 대안 경로 | 한 toy route 중심 | A/B/C의 dof·교차관측·kill test | gate 승인 후 구현 |

따라서 이 레인의 판정은 MATH_AND_ARCHITECTURE_COMPLETE이며
PHYSICAL_H0_INFERENCE_INCOMPLETE다. 후자는 목표를 내린 판정이 아니라, 실제로
성공할 세 경로를 서로 다른 실패 조건과 함께 보존한 구현 경계다.
