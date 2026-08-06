# CE 다섯 상수 강의: $e,\pi,i,1,0$의 정확한 역할

## 0. 생성 문법과 물리 입력

Euler 항등식

$$
e^{i\pi}+1=0
$$

은 질량·결합상수·우주밀도를 직접 산출하는 물리 방정식이 아니다. CE는
여기 등장하는 다섯 상수를 다음 연산 문법으로 사용한다.

| 상수 | 수학적 역할 | CE의 사용 |
|---|---|---|
| $e$ | 연속 character와 semigroup | 지수 생존·Gibbs 가중 |
| $\pi$ | compact phase의 주기·정규화 | $2\pi$ gauge 주기, loop measure |
| $i$ | unitary 위상·간섭 | Lorentzian 진폭, CP phase |
| $1$ | identity·정규화 | 총확률, density closure |
| $0$ | 영점·경계·constraint | EOM residual, branch selection |

물리 수치는 이 문법에 action, representation, scale와 관측 입력을 더한 뒤
나온다.

## 1. $e$: 가법 비용에서 곱적 가중으로

연속 양의 함수 $S:\mathbb R_{\ge0}\to\mathbb R_{>0}$가

$$
S(D_1+D_2)=S(D_1)S(D_2),
\qquad S(0)=1
$$

을 만족하면

$$
S(D)=e^{-\lambda D}
$$

이다. $D$를 optical depth 단위로 정의하면 $\lambda=1$이고

$$
\boxed{S(D)=e^{-D}}.
$$

여기서 정확한 것은 함수형 정리다. 어떤 물리적 결함이 가법이고 독립인지,
그리고 $D$를 어떤 operator로 계산하는지는 모형 선택이다.

### 1.1 세 다른 지수 연산자

$$
e^{-\chi\sqrt A}\quad\text{(Poisson)},
\qquad
e^{-sA}\quad\text{(heat)},
\qquad
e^{-iHt}\quad\text{(unitary time)}
$$

는 서로 바꾸어 쓸 수 없다. 첫 둘은 Euclidean semigroup이고 마지막은
Lorentzian 시간진화다.

## 2. $\pi$: compact phase와 측도

$U(1)$ 위상은 $\theta\sim\theta+2\pi$다. 따라서 Fourier mode,
loop measure와 topological charge의 정규화에 $2\pi$가 나타난다.

CE의

$$
\alpha_{\rm total}=\frac1{2\pi}
$$

는 “한 gauge 주기를 한 optical-depth 단위로 정규화한다”는 경계조건이다.
서로 다른 running coupling의 합이 일반적으로 scheme invariant인 것은
아니므로 같은 $\mu$와 scheme에서만 시험한다.

QCD instanton에서는

$$
S_{\rm inst}=\frac{8\pi^2}{g_s^2}
=\frac{2\pi}{\alpha_s}
$$

이다. $8\pi^2/\alpha_s$로 쓰면 coupling 정의를 네 배 잘못 적용한 것이다.

## 3. $i$: 위상과 CP

Lorentzian 경로적분은

$$
Z=\int\mathcal D\varphi\,e^{iS[\varphi]/\hbar}
$$

를 사용한다. $i$는 간섭을 가능하게 하지만 CP 위반의 크기나 최대 위상을
자동으로 고정하지 않는다. 물리적 CP violation은 재위상 뒤에도 남는
invariant, 예를 들어

$$
J=\operatorname{Im}(V_{us}V_{cb}V_{ub}^*V_{cs}^*)
$$

로 검사한다.

Euclidean continuation에서는 $t=-i\tau$와 contour 조건을 명시해
$e^{-S_E}$를 얻는다. Lorentzian 부호와 Euclidean 최대원리를 같은 증명에
섞지 않는다.

## 4. $1$: identity와 보존

정규화된 density matrix와 확률은

$$
\operatorname{Tr}\rho=1,
\qquad
\sum_i p_i=1
$$

을 만족한다. 평탄 cosmology를 모형 조건으로 채택하면

$$
\Omega_b+\Omega_{DM}+\Omega_{DE}=1.
$$

합이 1인 것은 closure 조건이며 각 성분의 미시적 정체를 증명하지 않는다.
각 성분은 공변 stress tensor와 perturbation response로 별도 정의해야 한다.

## 5. $0$: EOM과 branch

작용의 stationary condition은

$$
\frac{\delta S}{\delta\varphi}=0
$$

이다. 수치 계산에서는 residual 0과 boundary condition을 함께 검사한다.

Hodge-type 차원식

$$
\binom d2=d
\quad\Longleftrightarrow\quad
d(d-3)=0
$$

은 $d=0$과 $d=3$을 모두 가진다. $d\ge1$의 nonzero model class를
지정할 때 $d=3$이 유일하다. $0$을 삭제한 뒤 “유일해”라고 부르지 않는다.

## 6. 다섯 문법을 담는 공변 작용

최소 물리 모형은

$$
S=\int d^4x\sqrt{-g}\left[
\frac{M_P^2}{2}R+\mathcal L_{\rm SM}
-\frac12(\nabla\phi)^2-V(\phi)
-\frac\xi2R\phi^2
-\frac{\lambda_{H\phi}}2\phi^2H^\dagger H
\right]
$$

이다. 여기서

- $e$는 Euclidean weight와 bootstrap character에,
- $\pi$는 compact phase와 loop normalization에,
- $i$는 Lorentzian amplitude에,
- $1$은 state와 density normalization에,
- $0$은 EOM과 constraint에

나타난다. 이 대응은 문법의 역할이고, coupling 값은 입력·matching·RG로
고정한다.

## 7. 최신 Track A 수치 사슬

외부 입력은

$$
\boxed{\alpha_s^{\overline{\rm MS}}(M_Z)=0.1180}.
$$

neutral-projector benchmark는

$$
s_A^2:=4\alpha_s^{4/3}=0.2315097758,
$$

$$
\delta_N=s_A^2(1-s_A^2)=0.1779129995,
$$

$$
D_N=3+\delta_N=3.1779129995
$$

를 준다. 여기서 \(s_A^2\)는 CE 내부 registered output이고 물리적
\(s_W^2\)의 on-shell, \(\overline{\rm MS}\), effective convention과의
동일시는 아직 scheme-matching `Open`이다. bootstrap의 저분율 branch는

$$
\boxed{x=-\frac{W_0(-D_Ne^{-D_N})}{D_N}=0.0486382585}.
$$

분할 functional

$$
R=\alpha_sD_N(1+x\delta_N)=0.3782386966
$$

에서

$$
\boxed{(\Omega_b,\Omega_{DM},\Omega_{DE})
=(0.0486382585,0.2610881744,0.6902735671)}.
$$

모든 숫자는
[`../0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`](../0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json)에서
재계산된다.

## 8. B2가 $x$를 물리량으로 만드는 법

path-count 확률과 에너지분율은 일반적으로 다르다. Euclidean 준비 측도
$d\mu$와 총에너지 $E_{\rm tot}$로

$$
d\nu_E=\frac{E_{\rm tot}d\mu}{\int E_{\rm tot}d\mu}
$$

를 정의하고, baryon projector $\Pi_b$에 대해

$$
x=\frac{\operatorname{Tr}
(\rho H^{1/2}\Pi_bH^{1/2})}{\operatorname{Tr}(\rho H)}
$$

로 둔다. 평탄 관측 초곡면에서는 이 값이 $\Omega_b$다. $x$가 1에서
0.05로 “시간에 따라 바뀐다”는 주장은 별도 $D$-field와 전달방정식 없이
따르지 않는다.

## 9. $i$가 필요한 두 완성 모형

### 9.1 flavour

하나의 unitary construction에서 CKM/PMNS를 얻고 전체 행렬의 unitarity와
Jarlskog invariant를 검사한다. 개별 원소를 서로 다른 수치식으로 맞추지
않는다.

### 9.2 strong CP

shift-symmetric pseudoscalar의

$$
\left(\bar\theta+\frac a{f_a}\right)G\widetilde G
$$

coupling과 QCD potential이 vacuum minimum에서 effective theta를 0으로
만든다. CP-even scalar나 $i$ 자체는 이 일을 하지 않는다.

## 10. $e$가 필요한 finite-$\xi$ 인플레이션

비최소결합

$$
\xi=\alpha_s^{1/3}=0.4904868132
$$

인 quartic benchmark를 large-$\xi$ 근사 없이 풀면, 지정 입력
$N_*=57.1999$, $A_s=2.10\times10^{-9}$에서

$$
n_s=0.96617114,
\qquad r=0.00434561,
\qquad\lambda_4=1.3434991\times10^{-10}
$$

을 얻는다. $A_s$를 이용해 $\lambda_4$를 정규화했으므로 $A_s$를 다시
예측으로 세지 않는다.

## 11. 다섯 상수가 고정하지 않는 것

다음은 Euler 항등식만으로 나오지 않는다.

| 대상 | 추가로 필요한 것 |
|---|---|
| $\alpha_s(M_Z)$ | renormalization scheme·scale와 관측/RG 입력 |
| 질량 $m_f$ | Yukawa matrix와 Higgs scale |
| $H_0,T_{\rm CMB}$ | cosmological initial condition와 thermal history |
| $\eta_b$ | CP-odd source, bubble/transport, sphaleron |
| $\rho_{DE}$ 절대값 | 차원 scale와 renormalized gravity sector |
| 공학 효율 | 전 cycle의 입력·reset·손실 에너지 |

이 항목을 입력 장부에 적는 것은 이론을 약화시키는 일이 아니라, 무차원
문법을 차원 있는 자연과 연결하는 데 필요한 완성 조건이다.

## 12. 관측 판정

개별 중앙값 근접과 공동 적합을 구분한다. 최신 고정 density vector를
명시 주입한 13-dof forward benchmark는

$$
\chi^2=40.2015,
\qquad p=1.28\times10^{-4}
$$

로 reject된다. 따라서 현재의 고정 background 분할은 수학적으로 닫혀
있지만 full cosmology 자료의 최종 모형은 아니다. 다음 시도는 같은 공변
scalar-tensor action에서 background와 perturbation을 함께 풀고 추가
파라미터 penalty를 지불해야 한다.

## 13. 자동 검산

```powershell
powershell -ExecutionPolicy Bypass -File docs/0_검증과감사/run_full_consistency_gate.ps1
```

이 명령은 문서 구조뿐 아니라 다음 수치를 다시 계산한다.

- Track A와 Track B의 근
- $s_A^2,\delta_N,D_N,x,R,\boldsymbol\Omega$
- Gamma count/energy CDF의 차이
- B2 에너지 측도의 정규화
- finite-$\xi$의 $n_s,r,\lambda_4$
- $H_0,T_{\rm CMB}$ 입력을 포함한 $\eta_b$ 변환

다섯 상수는 이 계산의 언어를 제공하고, 작용·입력·likelihood가 그 언어를
검증 가능한 물리 모형으로 완성한다.
