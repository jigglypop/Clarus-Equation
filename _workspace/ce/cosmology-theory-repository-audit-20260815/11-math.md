# 우주론 수학 독립 검산

Status: COMPLETE

검산 기준일: 2026-08-15  
검산 범위: C1–C6, 살아 있는 우주론 문서·Python 경로·fixture·사전등록·테스트  
재현 계산: [verify_cosmology_math.py](artifacts/verify_cosmology_math.py)

## 0. 결론

현재 저장소에서 독립적으로 닫힌 우주론 결과는 다음의 좁은 범위다.

1. $D>1$일 때 고정점 방정식 $q=e^{-D(1-q)}$에는 $q=1$ 외에
   $(0,1/D)$의 유일한 최소 양의 가지가 있다. 선택한 $D$에 대한
   BootstrapSolver의 수치해도 맞는다.
2. 평탄 FLRW/CPL 배경, 거리, 표준 GR 성장 계산은 명시한
   $(H_0,r_d,\sigma_{8,0})$와 밀도 경계조건 아래 조건부 산출로서 대체로 맞다.
3. de Sitter entropy와 Friedmann 식을 같은 Planck-mass convention으로
   결합한 항등식은 맞다.
4. 내장 DESI 13차 covariance는 실제로 SPD이고, 고정 입력에 대한
   full-covariance 이차형식 구현은 조건부로 맞다.

그러나 현재 우주론 경로에는 결론을 바꾸는 네 종류의 P0가 있다.

- Hubble-tension toy가 radiation이 있는 배경에 matter+$\Lambda$ 식
  $R/H^2=12-9\Omega_m$을 사용하며, LCDM 음향각 함수의
  $\omega_b h^2$ 인자가 완전히 무시된다. 출력된
  $\Delta H_0=+5.5595$는 이 코드에서 유효한 readout이 아니다.
- [cosmology.py](../../../examples/physics/cosmology.py)의 선택적
  cumulative $S(a)$가 log-spaced grid에 uniform-grid Simpson 공식을
  사용한다. 이 가지의 $S(a)$ 및 그에 의존하는 성장 산출은 무효다.
- holographic gate의 “zero free parameters” 및 독립적 절대척도 재현
  해석은 거짓이다. $\rho_\Lambda$는 같은 $H_0$/$S_{\rm dS}$ 항등식의
  재표현이고, $\Omega_\Lambda$와 phase-area 법칙도 외부 선택이다.
- $q\mapsto\Omega_b$ 및 DM/DE 분할을 코드가 ce_prediction으로 표기하는
  것은 정본의 [공리]/[경험식] 지위와 충돌한다. 게다가 살아 있는 두
  배경은 $(\Omega_m,\Omega_\Lambda)=(0.307918,0.692082)$와
  $(0.310969,0.689031)$로 서로 다르다. branch tag 없이 “하나의 CE 예측”
  으로 합칠 수 없다.

따라서 이번 감사에서 [예측]으로 승격되는 우주론 주장은 없다.

## 1. C1–C6 판정표

| 항목 | 형식 지위 | 판정 | 우선순위 |
|---|---|---|---|
| C1a: $D>1$ 고정점 가지·유일성 | [정리] | 증명 성립 | PASS |
| C1b: 선택한 $D$의 수치근 | [산출] | solver는 일치, 원장 정밀도는 불일치 | P1 |
| C2a: $q_{\rm ext}\mapsto\Omega_b$ | [공리] | 고정점 정리에서 나오지 않음 | P0: 예측 표기 삭제 |
| C2b: 잔여분의 DM/DE 분할 | [경험식] | LO와 3-layer라는 서로 다른 ansatz | P0: 단일 배경 주장 삭제 |
| C3a: 평탄 FLRW/CPL/GR 가정 | [공리] | 범위가 명시되면 허용 | PASS |
| C3b: 배경·거리·기본 성장 | [산출] | residual forward model 기본 가지는 수치적으로 일관 | PASS |
| C3c: cumulative $S(a)$ 성장 | [산출] | 비균일-grid 적분 반례 | P0 |
| C4a: projected primordial $A_s$ | [경험식] | 다중 후보를 본 뒤 관측값 근처 projector 선택 | P1, [예측] 아님 |
| C4b: $H_0$ tension readout | [경험식] | 방정식 오류와 unused baryon input | P0 |
| C4c: Friedmann–entropy 관계 | [정리] | Planck convention별로 동치 | PASS |
| C4d: phase-area로 절대척도 예측 | [미완성] | 같은 $H_0$ 항등식, 외부 선택, 정본과 충돌 | P0 |
| C5a: 로컬 DESI 비교 | [경험식] | provenance가 완결되지 않은 exploratory 비교 | P1 |
| C5b: future holdout | [미완성] | frozen 문법은 valid, holdout은 unassigned/NOT_READY | P1 |
| C5c: covariance 통계 | [산출] | 내장 행렬은 SPD, parser는 SPD를 보장하지 않음 | P1 |
| C6: 테스트의 보장 범위 | [정리] | 구현 계약만 보장; 물리 사상은 보장하지 않음 | PASS |
| C6 실행 결과 | [산출] | 대상 테스트 91개 통과와 P0가 동시에 존재 | P0 주장에 대한 방어 아님 |

## 2. C1: 고정점 가지와 독립 증명

### 2.1 정의와 존재·유일성

다음 함수를 정의한다.

$$
h(q)=\log q+D(1-q),\qquad q>0.
$$

고정점 방정식은 $h(q)=0$과 같다. $D>1$이면

$$
h'(q)=\frac1q-D,\qquad h''(q)=-\frac1{q^2}<0.
$$

따라서 $h$는 $(0,1/D)$에서 엄격히 증가하고 $(1/D,\infty)$에서
엄격히 감소한다. 또한

$$
\lim_{q\to0^+}h(q)=-\infty,\quad
h(1/D)=D-1-\log D>0,\quad h(1)=0.
$$

$D-1-\log D>0$은 $D>1$에서 성립하므로 $(0,1/D)$에 정확히 한 근이
있고, 감소 구간의 다른 근은 $q=1$뿐이다. 그러므로 비자명근은 유일한
최소 양의 근이다. Lambert $W$로 쓰면

$$
q_{\rm ext}=-\frac1D W_0(-De^{-D}),\qquad
q_{\rm triv}=-\frac1D W_{-1}(-De^{-D})=1.
$$

비자명근에서 반복사상 $T(q)=e^{-D(1-q)}$의 multiplier는
$T'(q_{\rm ext})=Dq_{\rm ext}<1$이고, $q=1$에서는 $T'(1)=D>1$이다.
즉 작은 가지는 국소 안정, 자명가지는 불안정이다.

경계도 중요하다. $D=1$에서는 $q=1$이 접하는 유일한 양의 근이고,
$D<1$에서는 $(0,1)$ 비자명근이 없다. 따라서 “최소 비자명근”이라는
문장은 반드시 $D>1$ 조건을 포함해야 한다.

### 2.2 수치 원장

정의에서 다시 계산한 값은 다음과 같다.

| 양 | 독립 계산 |
|---|---:|
| $\alpha_s$ | 0.11789 |
| $\sin^2\theta_W=4\alpha_s^{4/3}$ | 0.23122206826075514 |
| $\delta=\sin^2\theta_W(1-\sin^2\theta_W)$ | 0.17775842340997383 |
| $D_{\rm eff}=3+\delta$ | 3.1777584234099736 |
| $q_{\rm ext}$ | 0.04864671964402817 |
| 방정식 잔차 | $2.78\times10^{-17}$ |
| $Dq_{\rm ext}$ | 0.154587523 |

[BootstrapSolver](../../../reality_stone/python/reality_stone/clarus/bootstrap_solver.py)은
반올림된 $D=3.17776$을 사용한다. 그 정의의 독립 근
0.04864663333721407과 solver 출력 0.048646633337339165의 차이는
$1.25\times10^{-13}$이므로 solver 자체는 합격이다.

다만 다음 세 숫자는 같은 정밀도의 “단일 원장”이 아니다.

- 정의에서 계산한 $D$: 3.1777584234099736
- BootstrapSolver의 $D$: 3.17776, 차이 $1.57659\times10^{-6}$
- [constants.py](../../../reality_stone/python/reality_stone/clarus/constants.py)의
  ACTIVE_RATIO: 0.0487

ACTIVE_RATIO를 정확한 $D$의 고정점이라고 대입하면 값 차이는
$5.32804\times10^{-5}$, 방정식 잔차는 $4.5043\times10^{-5}$다.
계약의 대수식 기본 허용오차 $10^{-12}$를 통과하지 않는다. 반올림
경계값이라고 명시해 쓰는 것은 가능하지만, exact fixed-point output으로
사용해서는 안 된다.

## 3. C2: 성분비 사상과 이중 배경

### 3.1 고정점에서 밀도분율은 나오지 않는다

$q$는 무차원 고정점이다. 이를 오늘의 baryon critical-density fraction
$\Omega_b$와 동일시하려면 시간, frame, renormalization scale,
critical-density 정의 및 관측 사상을 추가해야 한다. 따라서

$$
q_{\rm ext}\equiv\Omega_b(t_0)
$$

는 [정리]가 아니라 [공리]다. 남은 $1-q$를 DM과 DE로 나누는 비율은
고정점 방정식에 전혀 나타나지 않으므로 별도의 [경험식]이다.

이 판정은 정본과도 일치한다. [현대 우주론 감사](../../../docs/9_등호이전/05n_CE_cosmology_modern_audit.md)는
그 물리 사상을 [미완성]으로, [우주론 정본](../../../docs/3_상수/7_우주론.md)은
projected $A_s$를 [경험식], $H_0,S_8$ readout을 [미완성]으로 둔다.

반면 [parameter_provenance](../../../examples/physics/ce_residual_forward_model.py#L825)는
$\Omega_b,\Omega_{\rm DM},\Omega_\Lambda$ 세 값을 ce_prediction으로
기록하고 테스트도 그 문자열을 고정한다. 이는 C6가 경고하는 전형적인
상태 과장이다. 테스트는 문자열 계약을 보장했을 뿐 물리 사상을 증명하지
않았다.

### 3.2 서로 다른 살아 있는 배경

LO 분할은

$$
r_{\rm LO}=\alpha_sD,\quad
\Omega_\Lambda=\frac{1-q}{1+r_{\rm LO}},\quad
\Omega_{\rm DM}=\frac{(1-q)r_{\rm LO}}{1+r_{\rm LO}}.
$$

정의에서 계산하면

| branch | $\Omega_b$ | $\Omega_{\rm DM}$ | $\Omega_m$ | $\Omega_\Lambda$ |
|---|---:|---:|---:|---:|
| fixed-point + LO | 0.048646720 | 0.259271709 | 0.307918429 | 0.692081571 |
| constants raw | 0.0487 | 0.2623 | 0.3110 | 0.6891 |
| constants normalized | 0.048695130 | 0.262273773 | 0.310968903 | 0.689031097 |
| 3-layer discrimination | branch별 | branch별 | 0.310927213 | 0.689072787 |

constants의 raw 합은 1.0001이므로 residual model은 내부에서 정규화한다.
LO와 normalized-constants의 $\Omega_m$ 차이는 0.003050474, 즉 LO 대비
약 0.99%다. 3-layer와 normalized constants 차이는
$4.17\times10^{-5}$다.

[cosmology.py](../../../examples/physics/cosmology.py#L345)의 기본 bootstrap
경로는 LO 배경 $(0.307918429,0.692081571)$ 및
$H_0t_0=0.957087030$을 출력한다. 반면
[ce_residual_forward_model.py](../../../examples/physics/ce_residual_forward_model.py#L43)는
constants를 정규화한 $(0.310968903,0.689031097)$을 사용하며
$H_0t_0=0.954439551$이다.

두 값은 각각 자기 branch 안에서는 수학적으로 가능하다. 문제는 둘을
동일한 “CE 고유 배경” 또는 동일 ce_prediction으로 표기하는 것이다.
분할 공리와 branch selector를 명시하지 않는 단일 예측 주장은 삭제해야
한다. 살아남는 좁은 주장은 “선택한 분할 ansatz를 경계조건으로 넣으면
해당 FLRW 산출이 나온다”뿐이다.

## 4. C3: FLRW/CPL 거리·나이·성장

### 4.1 조건부 정의와 무차원성

residual forward model의 late-time 배경은 정규화된
$\Omega_{m0}+\Omega_{{\rm de},0}=1$ 아래

$$
E^2(a)=\Omega_{m0}a^{-3}
+\Omega_{{\rm de},0}
a^{-3(1+w_0+w_a)}e^{3w_a(a-1)}.
$$

CPL 인자는 $3w_a(a-1)$로 무차원이다. 광도거리는

$$
D_L(z)=(1+z)\frac{c}{H_0}\int_0^z\frac{dz'}{E(z')}
$$

이며 $c$를 km/s, $H_0$를 km/s/Mpc로 써 결과가 Mpc가 된다.
성장은 $x=\log a$에서

$$
D''+\left(2+\frac{d\log H}{d\log a}\right)D'
-\frac32\mu(a)\Omega_m(a)D=0
$$

을 푼다. 기본 residual 경로의 $\mu=1$은 GR [산출]이고,
$\mu\neq1$ 결합은 별도 [경험식]이다. $H_0,r_d,\sigma_{8,0}$는
모두 외부 [공리] 입력이다.

### 4.2 독립 수치 검산

기본 CEForwardParams에서 얻은 결과다.

| 검사 | 코드 | 독립 기준 | 오차 |
|---|---:|---:|---:|
| $E(1)$ | 1.0 | 1.0 | 0 |
| CPL scale, $a=0.61,w_0=-0.83,w_a=0.27$ | 정의값과 일치 | 연속방정식 직접 적분 | $2.22\times10^{-16}$ |
| $d\log H/d\log a$, $a=0.37$ | 코드 | 중앙차분 | $-1.87\times10^{-11}$ |
| $D_L(z=1)$ | 6818.454139268602 Mpc | adaptive Simpson 6818.454139268605 Mpc | 상대 $-4.44\times10^{-16}$ |
| $H_0t_0$ | 0.954439551364412 | analytic 0.954439551365607 | $-1.20\times10^{-12}$ |
| $D(a=0.5)$ | 0.6080843217 | exact integral 0.6080844112 | 상대 $-1.47\times10^{-7}$ |
| $f(a=0.5)$ | 코드 | exact-growth 유한차분 | $-1.10\times10^{-7}$ |

평탄 dust+$\Lambda$ 나이의 독립식은

$$
H_0t_0=
\frac{2}{3\sqrt{\Omega_\Lambda}}
\times
\operatorname{asinh}\!\left(\sqrt{\frac{\Omega_\Lambda}{\Omega_m}}\right).
$$

재현 스크립트도 같은 곱을 직접 사용한다.

### 4.3 cumulative $S(a)$의 완전 반례

[simpson](../../../examples/physics/cosmology.py#L31)은
$h=(x_{\rm last}-x_{\rm first})/(n-1)$ 하나만 쓰므로 균일 grid 전용이다.
그러나 [compute_s_of_a](../../../examples/physics/cosmology.py#L136)는
logspace $a$ grid를 그대로 넘긴다. 2001점 기본 grid에서

| $a$ 근방 | 코드 $S(a)$ | adaptive 기준 | 절대 오차 | 상대 오차 |
|---|---:|---:|---:|---:|
| 0.1 | 0.000274687 | 0.000205151 | $6.95\times10^{-5}$ | 약 +33.9% |
| 0.5 | 0.113742947 | 0.111657931 | 0.002085 | 약 +1.87% |
| 0.9 | 0.740628150 | 0.749593350 | -0.008965 | 약 -1.20% |

또한 점 개수가 짝수면 함수가 마지막 점을 조용히 버린다. 따라서
--sdef cumulative 산출과 그것을 입력으로 받는 성장 결과는 P0다.
기본 --sdef ratio 및 residual forward model의 uniform-grid 거리 적분은
이 반례의 대상이 아니므로 보존한다.

## 5. C4: $H_0$, 우주상수, 원시 스펙트럼

### 5.1 Hubble-tension readout: 두 개의 완전 반례

[hubble_tension.py](../../../examples/physics/hubble_tension.py#L120)는
radiation을 $H(a)$에 포함하면서

$$
\frac{R}{H^2}=12-9\Omega_m(a)
$$

를 쓴다. 평탄 matter+radiation+$\Lambda$에서는

$$
\frac{R}{H^2}
=3\Omega_m+12\Omega_\Lambda
=12-9\Omega_m-12\Omega_r
$$

가 정확하다. 누락 오차는 언제나 $12\Omega_r$다.

| $a$ | $\Omega_r(a)$ | 정확한 $R/H^2$ | 코드 | 코드-정확값 |
|---|---:|---:|---:|---:|
| 1 | $9.20\times10^{-5}$ | 현재값 | 현재값+$0.0011039$ | 0.0011039 |
| $10^{-3}$ | 0.210767 | 2.3676976 | 4.8969072 | 2.52921 |
| $10^{-6}$ | 0.996269 | 0.0111918 | 11.9664246 | 11.9552 |

두 번째 반례는 [lcdm_theta_star_for_h](../../../examples/physics/hubble_tension.py#L258)의
om_b_h2 인자다. $\omega_bh^2=0.001$과 0.1을 넣어도 두 결과가 정확히
0.012050273607668492로 같았다. 인자가 함수 본문에서 사용되지 않기
때문이다. 즉 “baryon physical density를 고정한 LCDM 음향각 matching”
이라는 설명은 구현되지 않았다.

추가로 $c_s=c/\sqrt3$, 고정 $z_\star$, 고정 $\Omega_{r0}$을 사용하고,
$\Omega_m+\Omega_\Lambda=1$에 radiation을 별도로 더해
$E(1)=\sqrt{1+\Omega_{r0}}=1.000045999$가 된다. $H_0$의 정의와도
미세하게 어긋난다.

실행 출력의 “$\Delta H_0=+5.5595$ (observed $\sim+5.6$)”는 위 오류와
SH0ES $H_{0,\rm true}=73.04$ 입력에 의존한다. 따라서 readout 부모
주장은 삭제한다. 살아남는 것은 수정·사전등록 전의 [경험식] toy ODE뿐이다.

### 5.2 holographic cosmological constant: 맞는 항등식과 틀린 독립성

gate는 non-reduced Planck mass $M_P$를 명시하고

$$
S_{\rm dS}=\pi\left(\frac{M_P}{H}\right)^2,\qquad
\rho_{\rm crit}=\frac{3}{8\pi}H^2M_P^2
$$

를 결합해

$$
\rho_\Lambda=
\Omega_\Lambda\frac38\frac{M_P^4}{S_{\rm dS}}
$$

를 얻는다. 이 대수는 맞다. reduced mass
$\bar M_P=M_P/\sqrt{8\pi}$를 쓰면

$$
S_{\rm dS}=\frac{8\pi^2\bar M_P^2}{H^2},\qquad
\rho_\Lambda=
\Omega_\Lambda\frac{24\pi^2\bar M_P^4}{S_{\rm dS}},
$$

이고 두 convention의 독립 계산 차이는 $5.55\times10^{-16}$ 이하다.
이 파일 내부에 Planck-mass 혼용 반례는 없다.

문제는 해석이다. phase-area 식

$$
\log S_{\rm dS}=\frac{\pi^2}{2}N_e-\pi\delta\sigma,\qquad
N_e=\frac d2D_{\rm eff}N_g
$$

은 고정점 정리에서 나오지 않는 추가 bridge다. $d=3$, $N_g=12$,
$\alpha_s$, $\Omega_\Lambda=0.6891$도 선택·입력이다. 따라서
“zero free parameters”는 “연속 변수를 이 실행에서 fit하지 않았다”는
뜻으로만 제한할 수 있고 “무입력”을 뜻하지 않는다.

독립 재계산값은

| 양 | 값 |
|---|---:|
| $\log S_{\rm dS}$ | 281.73768863 |
| $N_e$ | 57.19965162 |
| 같은 entropy에서 환산한 $H_0$ | 67.24834597 km/s/Mpc |
| $\rho_\Lambda^{1/4}$ | 2.24120277 meV |

$S_{\rm dS}$로 $H_0$를 정한 다음 같은 Friedmann–entropy 항등식으로
$\rho_\Lambda$를 계산했으므로 둘은 교차예측 두 개가 아니라 하나의
척도를 두 방식으로 쓴 것이다. 관측 $\rho_\Lambda$도 $H_0$ 및
$\Omega_\Lambda$로 환산되는 상관된 양이다.

더구나 active [gate header](../../../examples/physics/cosmological_constant_holographic_gate.py#L1)와
출력은 “dark energy density scale is not an independent open problem”,
“zero free parameters”, “absolute scale reproduced”를 말하지만,
[README](../../../README.md#L195)와
[Dark Energy Derivation](../../../docs/5_유도/04_Dark_Energy_Derivation.md#L194),
[우주론 정본](../../../docs/3_상수/7_우주론.md#L287)은 $H_0$ 절대단위와
$V_0$를 [미완성]/외부 입력으로 둔다. gate의 강한 부모 주장은 P0로
삭제하고, 위 항등식 및 “phase-area 공리를 가정할 때의 조건부 산출”만
보존한다.

hierarchy 출력에도 P1 산식 표기 문제가 있다. gate가 출력하는
$(\pi^2/2)N_e/\ln 10=122.58785465$는 실제
$-\log_{10}(\rho_\Lambda/M_P^4)=122.94481000$이 아니다.
$-\pi\delta\sigma$ 보정과 $3\Omega_\Lambda/8$ prefactor를 생략했기
때문이며 차이는 0.356955다.

### 5.3 primordial readout

[primordial_spectrum_readout_gate.py](../../../examples/physics/primordial_spectrum_readout_gate.py)는
다섯 readout을 한 관측 목표와 비교한다.

| readout | $10^9A_s$ | 관측과 차이 |
|---|---:|---:|
| total fixed-point response | 7.83532 | 197.80$\sigma$ |
| local residual drive | 5.60008 | 120.73$\sigma$ |
| phase projected drive | 2.269626 | 5.88$\sigma$ |
| integer geometry projected | 2.106042 | 0.243$\sigma$ |
| effective geometry projected | 2.1038087 | 0.166$\sigma$ |

모든 인자는 무차원이고 arithmetic은 재현된다. 그러나 관측 목표가 요구하는
projection exponent는 0.78358043이고 선택한 effective exponent
$D_{\rm eff}/(D_{\rm eff}+1)=0.76063719$다. 이미 본 목표에 대해
최소 다섯 후보, source 선택, geometry exponent와 $N_e$ 선택을 비교했다.
따라서 가장 가까운 값을 독립 [예측]으로 세는 것은 look-elsewhere
오류다.

canonical perturbation action에서 Mukhanov–Sasaki 정규화와
reheating/$N_\star$ bridge가 유도되지 않았으므로 현재 값은 [경험식]이다.
관측값을 보지 않은 action-level spectrum 계산과 후보군 전체 penalty가
필요하다.

## 6. C5: covariance, likelihood, dof와 holdout

### 6.1 covariance

내장 DESI DR2 13x13 covariance를 독립 Cholesky 분해하면 최소 pivot은
0.00578998687로 양수다. 이 fixture에 대해 full-covariance

$$
\chi^2=r^\mathsf{T}C^{-1}r
$$

계산은 조건부로 유효하다.

하지만 [parse_covariance_matrix](../../../examples/physics/ce_residual_forward_model.py#L1168)는
정방·대칭·양의 대각만 확인하고 SPD를 검사하지 않는다. 완전 반례

$$
C=\begin{pmatrix}1&2\\2&1\end{pmatrix}
$$

는 parser를 통과하지만 $\det C=-3$이고 Cholesky가 실패한다.
$r=(1,-1)$이면 구현이 허용한 이차형식은
$r^\mathsf{T}C^{-1}r=-2$로, chi-square가 될 수 없다. parser의
“covariance” 계약에는 P1 SPD gate가 필요하다.

### 6.2 자료 지위와 자유도

- 로컬 DESI vector/covariance는 raw-byte hash와 upstream commit이
  완결되지 않았다고 코드가 스스로 밝힌 exploratory fixture다.
- future holdout v2 manifest 문법은 VALID지만
  holdout=unassigned, evaluation=NOT_READY다. 관측 결과가 없으므로
  [예측] 성공이나 실패를 선언할 수 없다.
- default core model-selection gate는 후보 27개에 독립 selection
  observation이 1개뿐이고 selection_status=UNDERIDENTIFIED다.
  $\Omega_b$ 하나로 family를 고르는 것은 불가능하다.
- 사전등록에는 external-$r_d$와 EH-hybrid 두 후보가 있다. holdout에서
  fit하는 연속 자유도는 각 후보 0으로 고정할 수 있지만, 후보 선택
  자유도 2와 이미 본 자료에서 설계한 target awareness는 사라지지 않는다.
- $H_0,r_d,\sigma_{8,0},w_0,w_a$, nuisance, selection 및 covariance가
  동결되지 않은 비교는 관측 [예측]이 아니라 [경험식]이다.

## 7. S8, 부호·경계·근사·단위

### 7.1 S8 raw 대 normalized $\Omega_m$

residual 배경은 raw constants 합 1.0001을 정규화해
$\Omega_{m,\rm bg}=0.3109689031$을 쓴다. 하지만
[s8_today](../../../examples/physics/ce_residual_forward_model.py#L1525)는
raw $\Omega_m=0.311$을 쓴다.

| 정의 | 값 |
|---|---:|
| 코드 $S_8=0.811\sqrt{0.311/0.3}$ | 0.82573448315 |
| 배경 일관 $S_8=0.811\sqrt{0.3109689031/0.3}$ | 0.82569319952 |
| 차이 | $4.12836\times10^{-5}$ |

작아서 현재 결론을 바꾸지는 않지만 단일 background convention을
깨는 P2다. 기존 broad tolerance 테스트는 이를 검출하지 못한다.

### 7.2 무차원 감사

직접 확인한 결과:

- 고정점 exponent $D(1-q)$: 무차원.
- CPL exponent $3w_a(a-1)$: 무차원.
- $\log a$, $\log S$, phase-area exponent: 무차원.
- BAO $D_M/r_d,D_H/r_d,D_V/r_d$: 무차원.
- 성장 $\mu,\Omega_m,D,f,\sigma_8$: 무차원.
- primordial source와 $A_s$: 무차원.
- $H_0t_0$: 무차원.
- $D_L$: $c/H_0$로 Mpc 환산, 단위 일관.
- holographic gate: 자연단위 $c=\hbar=1$을 eV와 seconds/Mpc 사이에서
  변환할 때 non-reduced convention이 일관.

현재 [dimensionless checker](../../../reality_stone/python/reality_stone/clarus/dimensionless.py)는
이 전체 경로를 등록하지 않는다. CPL, Hubble readout, holographic,
primordial, covariance가 coverage 밖이고 exp/log를 문자열 수준으로
찾는 검사는 vector/matrix·Planck convention을 증명하지 못한다. 현재
PASS는 등록된 소수 식의 구현 검사일 뿐 전체 우주론 무차원 보증이 아니다.
이는 P1 검증 공백이다.

### 7.3 기타 경계와 표기

- BootstrapSolver docstring은 $q$를 “survival rate”로 부르지만
  branching-process 표준 해석에서는 작은 근이 extinction probability다.
  물리 사상과 섞지 않도록 P2 수정이 필요하다.
- OMEGA_B_OBS_SIGMA=0.00005 옆 주석은 “±0.0005”라 써 10배 다르다.
  실제 sigma 비교의 의미가 달라지는 P1 문서/통계 불일치다.
- legacy Simpson은 짝수 길이에서 마지막 interval을 버리는 경계 동작을
  명시하지 않는다.
- $\Omega_i\ge0$, $a>0$, $H_0>0$, $r_d>0$ 경계는 residual model의
  주요 경로에서 확인되지만 covariance SPD는 누락됐다.

## 8. P0/P1/P2 원장

### P0

1. Hubble-tension Ricci 식에서 $-12\Omega_r$ 누락.
   반례: $a=10^{-6}$에서 $R/H^2$ 오차 11.9552.
   삭제 범위: 현재 $\Delta H_0=+5.6$ closure/readout 결론.
2. Hubble LCDM 음향각 함수의 om_b_h2가 unused.
   반례: 0.001과 0.1 입력이 bit-identical.
   삭제 범위: baryon density를 고정한 CMB matching 주장.
3. cumulative $S(a)$에 비균일 grid용이 아닌 Simpson 적용.
   반례: $a\simeq0.1$에서 약 33.9% 오차.
   삭제 범위: --sdef cumulative 및 그 성장 산출.
4. holographic “zero free parameters/독립 절대척도 재현”.
   반례: 같은 $S_{\rm dS}$–$H_0$–$\rho$ 항등식이며
   $(d,N_g,\alpha_s,\Omega_\Lambda)$ 및 phase-area bridge가 필요.
   삭제 범위: 독립 우주상수 예측; 항등식은 보존.
5. 밀도분율 ce_prediction 및 단일 CE 배경 표기.
   반례: 정본은 사상을 미완성/공리로 두고, live LO와 constants
   background가 0.99% 다름.
   삭제 범위: 사상·분할이 유도된 예측이라는 부모 주장.

### P1

1. exact $D,q$와 반올림 BootstrapSolver/constants 원장의
   $10^{-12}$ 불일치.
2. covariance parser의 SPD 검증 부재.
3. projected primordial $A_s$의 target-aware 다중후보 및
   action-normalization bridge 부재.
4. dimensionless checker의 우주론 coverage 공백.
5. holographic hierarchy exponent에서 correction/prefactor 누락.
6. exploratory DESI provenance의 upstream commit/raw hash 공백 및
   holdout NOT_READY.
7. OMEGA_B_OBS_SIGMA 값과 주석의 10배 불일치.
8. Hubble 배경의 $E(1)\ne1$, constant sound speed, 고정 $z_\star$ 근사.

### P2

1. $S_8$가 normalized background 대신 raw $\Omega_m$ 사용.
2. 고정점 작은 근을 survival rate로 부르는 용어 혼동.
3. LO/3-layer/normalized-constants의 출력에 branch id가 일관되게
   붙지 않는 표기 문제.

## 9. C6와 테스트 해석

다음 대상 테스트 91개가 통과했다. 이는 등록된 수치 회귀와 schema가
현재 구현과 일치한다는 [산출]이다. 그러나 테스트가
parameter_provenance의 ce_prediction 문자열을 직접 assert하는 경우처럼,
잘못된 물리 지위를 더 단단히 고정할 수도 있다.

따라서 테스트 통과로 보장되는 것은 다음뿐이다.

- 선택한 식과 fixture의 구현 회귀
- frozen manifest의 문법
- solver의 선택한 반올림 $D$에 대한 수치 일치

보장되지 않는 것은 $q\mapsto\Omega_b$의 자연 사상, 분할 ansatz,
phase-area law, $H_0$ 절대척도, projected $A_s$, 관측 독립성이다.

## 10. 재현 명령

저장소 루트에서:

    python "_workspace/ce/cosmology-theory-repository-audit-20260815/artifacts/verify_cosmology_math.py"

    python -m pytest tests/test_bootstrap_solver.py tests/test_core_model_selection.py tests/test_cosmology_ratio_audit.py tests/test_ce_residual_forward_model.py tests/test_recombination_drag_adapter.py tests/test_primordial_spectrum_readout_gate.py tests/test_dimensionless.py tests/test_holdout_preregistration.py -q -p no:cacheprovider --basetemp "_workspace/ce/cosmology-theory-repository-audit-20260815/artifacts/pytest-math"

    python examples/physics/hubble_tension.py

    python examples/physics/cosmological_constant_holographic_gate.py

    python examples/physics/primordial_spectrum_readout_gate.py

    python examples/physics/core_model_selection_gate.py

    python experiments/preregistration/validate_holdout_manifest.py experiments/preregistration/cosmology_future_holdout_v2.json

관측 결과:

- 독립 검산 스크립트: exit 0, 위 표의 모든 값 출력.
- 대상 pytest: 91 passed in 8.24s.
- Hubble script: exit 0이지만 $\Delta H_0=+5.5595$는 위 P0 반례로 무효.
- holographic gate: exit 0이지만 “<0.2%”는 같은-scale 재표현 비교.
- core selection: algebraic PASS, selection UNDERIDENTIFIED.
- holdout validator: VALID, holdout=unassigned, evaluation=NOT_READY.

## 11. 완결성 게이트

완전 반례가 무너뜨리는 부모 주장은 활성 결론에서 제거해야 한다.

- 제거: “현재 Hubble toy가 baryon-aware CMB readout으로 tension을
  설명했다.”
- 제거: “phase-area gate가 우주상수 절대값을 zero-input 독립 예측했다.”
- 제거: “세 밀도분율은 고정점에서 유도된 ce_prediction이다.”
- 제거: “cumulative $S(a)$ 가지가 현재 구현에서 수치적으로 검증됐다.”

보존 가능한 좁은 결론:

- $D>1$ 고정점 정리와 선택한 $D$에서의 수치근.
- 명시한 밀도분율을 경계조건으로 둔 조건부 FLRW/CPL/GR 산출.
- Friedmann–de Sitter entropy 항등식과 명시적 Planck convention.
- SPD가 별도로 보장된 frozen covariance에 대한 이차형식 계산.
