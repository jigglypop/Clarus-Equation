# CE 귀납 강의: 관측 패턴에서 반증 가능한 모형까지

## 0. 귀납의 원칙

수치가 가까운 식을 찾는 일은 가설 생성이지 증명이 아니다. 이 강의는
관측 패턴을 발견한 뒤 다음 순서로 모형을 고정한다.

$$
\text{패턴 발견}
\to\text{작용·측도·branch 지정}
\to\text{입력/보정 분리}
\to\text{새 출력 계산}
\to\text{공동 likelihood와 holdout}.
$$

최신 관측 정의는
[`../0_검증과감사/OBSERVATIONAL_BASELINE_2026-08-06.md`](../0_검증과감사/OBSERVATIONAL_BASELINE_2026-08-06.md),
수치 사슬은
[`../0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`](../0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json)을
따른다.

## 1. 첫 패턴: 약 5%의 바리온 분율

평탄 $\Lambda$CDM chain에서 $\omega_b=\Omega_bh^2$와 $h$를 함께 추정하면
$\Omega_b$가 약 $0.049$라는 사실을 얻는다. 단순 지수

$$
e^{-3}=0.049787\ldots
$$

가 가까운 것은 가설의 단서다. 여기서 곧바로
$e^{-3}=\Omega_b$라고 쓰지 않고 다음 세 구조를 분리한다.

1. 독립 비용의 곱성에서 $S(D)=e^{-\lambda D}$를 얻는다.
2. optical-depth 단위로 $\lambda=1$을 선택한다.
3. 바리온은 path-count가 아니라 stress-energy 가중 projector로 읽는다.

## 2. 전자약 패턴을 invariant로 바꾸기

과거에는 질량행렬의 특정 비대각 원소를 “혼합량”으로 사용했지만, 그 값은
기저 회전으로 바뀐다. 최신 가설은 gauge projector와 spectral projector의
overlap을 쓴다.

$$
p_W=\operatorname{Tr}(P_WP_Z),
\qquad
p_B=\operatorname{Tr}(P_BP_Z),
$$

$$
\boxed{\delta_N=p_Wp_B=s_N^2(1-s_N^2)}.
$$

Track A 입력 $\alpha_s(M_Z)=0.1180$과 CE 관계
$s_A^2:=4\alpha_s^{4/3}$를 등록하고 projector boundary
$s_N^2=s_A^2$를 선택하면

$$
s_A^2=0.2315097758,
\qquad
\delta_N=0.1779129995,
\qquad
D_N=3.1779129995
$$

를 얻는다. \(s_A^2\)를 on-shell, \(\overline{\rm MS}\) 또는 effective
약혼합각으로 보내는 scheme map은 아직 Open이다. 관측 $s_W^2$를 먼저 넣어
$\alpha_s$를 역산하면 그것은 calibration이며 다시 예측으로 세지 않는다.

## 3. 패턴에서 고정점 정리로

단순 $e^{-D_N}$ 대신 남은 분율이 비용에 되먹임한다고 가정하면

$$
x=e^{-(1-x)D_N}
$$

이 된다. 이 식의 저분율 branch는

$$
x=-\frac{W_0(-D_Ne^{-D_N})}{D_N}
=0.0486382585
$$

이고 multiplier $D_Nx=0.1545681540<1$이다. $D_N>1$에는 경계해
$x=1$도 존재하므로 branch 조건을 숨기지 않는다.

이제 B2에서

$$
x=\frac{\langle E_b\rangle}{\langle E_{\rm tot}\rangle}
$$

로 표본공간을 에너지 가중 측도로 고정한다. 관측 초곡면의 평탄성 아래
$\Omega_b=x$가 된다. 이 구성은 숫자의 우연한 일치를 실제 계산 가능한
projector 모형으로 바꾼다.

## 4. 암흑성분 패턴

관측상 $\Omega_{DM}/\Omega_{DE}$는 약 $0.38$이다. 후보식을 여러 개
사후 비교한 뒤 가장 가까운 것을 택하면 look-elsewhere 효과가 생긴다.
최신 CE는 하나의 functional을 사전 고정한다.

$$
\boxed{R=\alpha_sD_N(1+x\delta_N)=0.3782386966}.
$$

평탄성에서

$$
(\Omega_b,\Omega_{DM},\Omega_{DE})
=(0.0486382585,0.2610881744,0.6902735671)
$$

가 유일하게 정해진다. 합이 1인 것은 대수 검산이지 데이터 적합의
증거가 아니다.

## 5. marginal 일치와 full likelihood의 차이

개별 밀도 중앙값만 보면 CE 벡터는 표준 우주론 값에 가깝다. 그러나 같은
parameter vector로 CMB·BAO·SN·성장 자료를 동시에 계산하면 상관과
sound horizon이 드러난다.

최신 explicit-canonical forward run의 예는 다음과 같다.

| 설정 | $\chi^2$ | dof | $p$ |
|---|---:|---:|---:|
| external $H_0=67.4$, $r_d=147.09$ Mpc | 40.2015 | 13 | $1.28\times10^{-4}$ |
| EH-hybrid $r_d=151.5052$ Mpc | 41.1946 | 13 | $8.86\times10^{-5}$ |

따라서 이 두 고정 background benchmark는 공동 gate에서 reject된다. 이
결과를 없애기 위해 오차막대를 넓히거나 일부 자료만 고르지 않는다.
살아남을 수 있는 확장은 공변 scalar-tensor stress와 perturbation을 같은
action에서 풀고, 새 파라미터 수의 penalty까지 포함해 다시 비교하는 것이다.

이것은 모형을 “강등”한 결과가 아니라, 숫자 중앙값의 인상비평을 실제
likelihood 판정으로 완성한 결과다.

## 6. 결합상수의 교차 가설

Track B에서는 독립 $\alpha_{em}(M_Z)=1/127.95$를 넣어

$$
a+\frac{\alpha_{em}}{4a^{4/3}}
+\alpha_{em}=\frac1{2\pi}
$$

를 푼다. 양의 근 두 개 가운데 SM-like hierarchy를 사전 선택하면

$$
\alpha_s=0.1173186647,
\quad s_W^2=0.2297291680,
\quad\alpha_w=0.0340207254
$$

를 얻는다. Track A와 차이는 truncation·경계조건의 검증량이다. 두 트랙의
입력을 섞어 더 좋은 숫자만 취하지 않는다.

## 7. flavour 패턴의 완성 방법

CKM 또는 PMNS의 원소마다 서로 다른 거듭제곱을 맞추는 방식은 unitary
행렬 하나를 보장하지 않는다. 최신 benchmark는 하나의 Yukawa matrix를
먼저 정의한다.

$$
Y_f=U_{fL}\operatorname{diag}(y_{f1},y_{f2},y_{f3})U_{fR}^\dagger.
$$

그 결과로

$$
V_{\rm CKM}=U_{uL}^\dagger U_{dL}
$$

를 얻는다. 최신 seed의 예는

$$
(s_{12},s_{23},s_{13})
=(0.22724210,0.04168209,0.00372494)
$$

이고 $\delta_q=1.2$에서
$J=3.1996\times10^{-5}$, unitary residual은 약
$3.7\times10^{-18}$이다. 이 수치는 하나의 공동 texture benchmark이며,
질량·혼합·CP 데이터를 같은 covariance에서 검사한다.

## 8. strong CP 패턴의 올바른 보강

$e^{-8\pi^2/\alpha_s}$ 같은 식은 instanton convention부터 틀리며,
sector 억압만으로 $\bar\theta$를 0으로 만들지도 못한다. 올바른 semiclassical
action은 $2\pi/\alpha_s$이고, 해결 branch는 pseudoscalar $a$와

$$
\left(\bar\theta+\frac a{f_a}\right)G\widetilde G
$$

coupling을 둔다. QCD potential의 최소가 effective theta를 0으로 만든다.
nEDM과 axion search가 독립 holdout이다.

## 9. 인플레이션 패턴의 실제 계산

$n_s\simeq1-2/N$, $r\simeq12/N^2$가 관측과 가깝다는 사실만으로 CE
인플레이션이 되지 않는다. finite-$\xi$ Jordan action을 Einstein frame으로
옮겨 정확히 풀어야 한다.

$\xi=\alpha_s^{1/3}=0.4904868132$,
$N_*=57.1999$, $A_s=2.10\times10^{-9}$인 quartic benchmark는

$$
n_s=0.96617114,
\qquad
r=0.00434561,
\qquad
\lambda_4=1.3434991\times10^{-10}
$$

을 준다. $A_s$는 $\lambda_4$ calibration이고, $n_s,r$가 검증 출력이다.

## 10. 바리온 비대칭의 교훈

$\Omega_b$와 $\eta_b$의 수치가 함께 맞는다는 것만으로 생성 메커니즘이
생기지 않는다. 현재 밀도 변환은

$$
\eta_b^{\rm density}
=\frac{\Omega_b\rho_c}{m_bn_\gamma}
=6.0412\times10^{-10}
$$

을 주지만 $H_0,T_{\rm CMB},m_b$ 입력을 포함한다. 실제 생성은 CP-odd
Wilson operator, bubble bounce, transport와 sphaleron washout을 풀어야
한다. density 변환과 transport output의 일치가 closure test다.

## 11. 귀납 추론의 통계 계약

후보식이 $N$개였다면 최종 하나만 보고한 residual은 유효하지 않다.
최신 절차는 다음을 기록한다.

| 단계 | 기록 |
|---|---|
| 후보 생성 | 모든 식과 계수 범위 |
| 선택 | 선택 자료와 objective |
| calibration | 사용한 관측량·covariance |
| validation | 겹치지 않는 자료 |
| holdout | 분석 전 잠근 결과 |
| 복잡도 | 파라미터 수와 prior volume |

관측 벡터 $\mathbf d$, 이론 벡터 $\mathbf m(\theta)$에 대해

$$
\chi^2=(\mathbf d-\mathbf m)^TC^{-1}(\mathbf d-\mathbf m)
$$

을 사용한다. 동일 자료의 여러 marginal 표를 독립 측정처럼 합하지 않는다.

## 12. 귀납에서 연역으로 넘어가는 기준

다음 조건을 만족할 때 패턴은 계산 모형이 된다.

1. 차원과 기호 타입이 완결돼 있다.
2. action 또는 normalized measure가 있다.
3. EOM·stress·branch가 재현된다.
4. 모든 입력과 calibration이 공개된다.
5. 선택에 쓰지 않은 holdout이 있다.
6. 실패한 full likelihood도 그대로 기록한다.

이 기준으로 CE의 수학 고정점과 finite-$\xi$ benchmark는 재현 가능한
모형이 되었고, 고정 cosmology benchmark는 현재 자료에서 reject된다는
결론까지 포함해 귀납 루프가 닫혀 있다.
