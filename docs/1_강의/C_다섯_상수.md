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

이 장의 논리 지위 `Definition`, `Exact`, `Exact conditional`, `Convention`,
`Selection`, `Bridge`, `Phenomenology`와 validation 상태 `Open`, `Rejected`는
[A 강의의 지위 계약](A_연역적_유도.md#0-이-강의의-계약)과 같은 뜻이다.
`Calibration input`, `Calibration output`은 별도 provenance 역할이다.
다섯 상수의 수학적 역할과 그 역할을 특정 물리량에 대응하는 bridge를 같은
증거로 세지 않는다.

## 1. $e$: 가법 비용에서 곱적 가중으로

**지위: `Exact conditional` — S1--S4 아래의 함수형 정리.**

$S:[0,\infty)\to(0,1]$에 다음을 가정한다.

1. **S1, 정규화:** $S(0)=1$.
2. **S2, 연결 곱성:**

$$
S(D_1+D_2)=S(D_1)S(D_2),
$$

3. **S3, 정칙성:** $S$는 측정가능하다. 연속성을 가정해도 충분하다.
4. **S4, 비자명성:** 어떤 $D_0>0$에서 $S(D_0)<1$이다.

$f(D):=-\log S(D)$라 하면 S2에서 $f$는 가법함수다. S3에 의해
$f(D)=\kappa_{\rm surv}D$이고, $0<S\le1$에서
$\kappa_{\rm surv}\ge0$, S4에서 $\kappa_{\rm surv}>0$이다. 따라서

$$
\boxed{S(D)=e^{-\kappa_{\rm surv}D}},\qquad \kappa_{\rm surv}>0
$$

이다. S4 같은 감쇠 방향 조건을 빼고 범위도 양의 실수 전체로 넓히면
$S(D)=e^D$도 S1--S3을 만족하므로 감쇠 부호를 결론낼 수 없다.

**지위: `Convention`.** $\widetilde D:=\kappa_{\rm surv}D$를
optical-depth 단위로 정의하고
$\widetilde S(\widetilde D):=S(\widetilde D/\kappa_{\rm surv})$로
재매개화하면 단위율 $\kappa_{\rm surv}=1$이고

$$
\boxed{\widetilde S(\widetilde D)=e^{-\widetilde D}}.
$$

정확한 것은 S1--S4 아래의 함수형 정리다. 어떤 물리적 결함이 가법이고
독립인지, 그리고 $D$를 어떤 operator로 계산하는지는 `Open Bridge`다.

### 1.1 세 다른 지수 연산자

$$
e^{-\ell_P\sqrt {A_E}}\quad\text{(Poisson)},
\qquad
e^{-s_HA_E}\quad\text{(heat)},
\qquad
e^{-i\widehat{\mathcal H}t/\hbar}\quad\text{(unitary time)}
$$

는 서로 바꾸어 쓸 수 없다. 첫 둘은 Euclidean semigroup이고 마지막은
Lorentzian 시간진화다.

## 2. $\pi$: compact phase와 측도

$U(1)$ 위상은 $\theta\sim\theta+2\pi$다. 따라서 Fourier mode,
loop measure와 topological charge의 정규화에 $2\pi$가 나타난다.

CE의

$$
C_{\rm CE}:=\alpha_s+\alpha_w+\alpha_{em}=\frac1{2\pi}
$$

는 “한 gauge 주기를 한 optical-depth 단위로 정규화한다”는 `Selection`인
경계조건이다.
서로 다른 running coupling의 합이 일반적으로 scheme invariant인 것은
아니므로 같은 $\mu$와 scheme에서만 시험한다.

QCD instanton에서는

$$
S_{\rm inst}=\frac{8\pi^2}{g_s^2}
=\frac{2\pi}{\alpha_s}
$$

이다. $8\pi^2/\alpha_s$로 쓰면 $g_s^2=4\pi\alpha_s$ 치환에서
$4\pi$를 누락한 것이므로, 정확한 instanton action보다 $4\pi$배 크게 쓴
것이다.

## 3. $i$: 위상과 CP

Lorentzian 경로적분은

$$
Z=\int\mathcal D\Psi\,e^{iS[\Psi]/\hbar}
$$

를 사용한다. $i$는 간섭을 가능하게 하지만 CP 위반의 크기나 최대 위상을
자동으로 고정하지 않는다. 물리적 CP violation은 재위상 뒤에도 남는
invariant, 예를 들어

$$
J=\operatorname{Im}(V_{us}V_{cb}V_{ub}^*V_{cs}^*)
$$

로 검사한다.

Euclidean continuation에서는 $t=-i\tau$와 contour 조건을 명시해
$e^{-S_E/\hbar}$를 얻는다. Lorentzian 부호와 Euclidean 최대원리를 같은 증명에
섞지 않는다.

## 4. $1$: identity와 보존

정규화된 density matrix와 확률은

$$
\operatorname{Tr}\rho=1,
\qquad
\sum_i p_i=1
$$

을 만족한다. 그러나 일반 FLRW Friedmann closure 장부는 neutrino를 다른
항과 중복 계상하지 않는 convention에서

$$
\Omega_{\rm phys}:=\frac{\rho_{\rm tot}}{\rho_c}
=\Omega_b+\Omega_{\rm cdm}+\Omega_{\rm DE}+\Omega_r+\Omega_\nu,
\qquad
1=\Omega_{\rm phys}+\Omega_k
$$

이다. $\Omega_r=\Omega_\nu=\Omega_k=0$을 선택한 경우에만
$\Omega_b+\Omega_{\rm cdm}+\Omega_{\rm DE}=1$이다. 이것은
late-time truncated 3-sector `Selection`이며, 복사와 neutrino가 실제로
0이라는 명제가 아니다. 합이 1인 것은 closure 조건일 뿐 각 성분의 미시적
정체를 증명하지 않는다. 각 성분은 공변 stress tensor와 perturbation
response로 별도 정의해야 한다.

## 5. $0$: EOM과 branch

작용의 stationary condition은

$$
\frac{\delta S}{\delta\Psi}=0
$$

이다. 수치 계산에서는 residual 0과 boundary condition을 함께 검사한다.

Hodge-type 차원식만 쓰기 전에 필요한 기하 구조를 표시해야 한다. $V$를
양의 내적과 orientation을 가진 $d$차원 실수 벡터공간이라 하면

$$
*:\Lambda^2V^*\longrightarrow\Lambda^{d-2}V^*
$$

가 정의된다. 국소적·선형·$SO(d)$-equivariant closure의 출력이 다음
단계의 1-form 타입이어야 한다고 선택하면 $d-2=1$이고 $d=3$이다. 성분 수
검산은

$$
\binom d2=d
\quad\Longleftrightarrow\quad
d(d-3)=0
$$

은 형식적으로 $d=0$과 $d=3$을 가진다. $d\ge2$이고 2-form이 비영인
model class에서는 $d=3$이 유일하다. 이 결론은 metric·orientation과
Hodge-type closure를 조건으로 한 `Exact conditional`이다. 차원 일치만으로
metric 없는 자연스러운 $GL(V)$-동형이 생긴다는 뜻은 아니다.

## 6. 다섯 문법을 담는 공변 작용

계량 부호 $(-,+,+,+)$, reduced Planck mass
$M_P=(8\pi G)^{-1/2}$, 곡률 convention을 A 강의와 같게 고정한다. core
branch의 최소 물리 모형은

$$
S=\int d^4x\sqrt{-g}\left[
\frac{M_P^2}{2}R_g+\mathcal L_{\rm SM}
-\frac12(\nabla\phi)^2-V(\phi)
-\frac{\xi_{\rm core}}2R_g\phi^2
-\lambda_{\rm HP}\phi^2H^\dagger H
\right]
$$

이다. 이는 [A 강의 2절](A_연역적_유도.md#2-공변-eft)의 canonical
two-derivative truncation $Z_\phi=1$, $c_i=0$이며
$F_{\rm core}(\phi)=M_P^2-\xi_{\rm core}\phi^2>0$인 장 영역만 사용한다. portal은
no-$1/2$ 규약을 사용하며, $-\lambda^{(1/2)}_{H\phi}\phi^2H^\dagger
H/2$ 규약과 비교할 때에는
$\lambda^{(1/2)}_{H\phi}=2\lambda_{\rm HP}$로 바꾼다. 여기서

- $e$는 Euclidean weight와 bootstrap character에,
- $\pi$는 compact phase와 loop normalization에,
- $i$는 Lorentzian amplitude에,
- $1$은 state와 density normalization에,
- $0$은 EOM과 constraint에

나타난다. 이 대응은 문법의 역할이고, coupling 값은 입력·matching·RG로
고정한다.

## 7. 최신 Track A 수치 사슬

**provenance: `Calibration input`.** 외부 입력은

$$
\boxed{\alpha_s^{\overline{\rm MS}}(M_Z)=0.1180}.
$$

**논리 지위: `Definition`. provenance: `Calibration output`.** Track-A 등록량은

$$
s_A^2:=4\alpha_s^{4/3}=0.2315097758,
$$

$$
\delta_A:=s_A^2(1-s_A^2)=0.1779129995,
$$

**지위: `Selection + Exact conditional`.** 양의 additive fold operator에서
$\delta_{\rm fold}:=\delta_A$를 선택하면 trace additivity로

$$
D_A:=3+\delta_A=3.1779129995
$$

를 얻는다. 물리적 projector 값
$\delta_{\rm proj}=s_W^2(1-s_W^2)$와 $\delta_A$의 동일시는 아직
RG·threshold·scheme를 요구하는 `Open Bridge`다.

이 fold trace $D_A$가 다음 절의 next-generation 행렬 $\mathsf K$의 공통
행합이라는 명제도 별도 `Open Bridge`다. 아래 고정점 정리는 그 공통 행합을
조건으로 넣은 뒤에만 닫힌다.

**지위: `Exact conditional`.** 상위 재귀는 비음수 next-generation 행렬
$\mathsf K$의 벡터식이다. 한 type이거나
$\mathsf K\boldsymbol1=D_A\boldsymbol1$인 공통 행합 균일 sector에서만

$$
x=e^{-D_A(1-x)}
$$

로 줄어든다. 그 저분율 branch는

$$
\boxed{x_\star=-\frac{W_0(-D_Ae^{-D_A})}{D_A}=0.0486382585}.
$$

이때 $x_\star$는 수학적 고정점이며 아직 $\Omega_b$가 아니다.

**지위: `Phenomenology/Bridge`.** 분할 functional은

$$
R_{\rm dark}:=\alpha_sD_A(1+x_\star\delta_A)=0.3782386966
$$

로 선택한다. 완전한 밀도 장부에서

$$
\Omega_{\rm rem}:=\Omega_{\rm phys}-\Omega_b-\Omega_r-\Omega_\nu
=1-\Omega_k-\Omega_b-\Omega_r-\Omega_\nu
$$

를 먼저 계산하고

$$
\Omega_{\rm cdm}=\Omega_{\rm rem}
\frac{R_{\rm dark}}{1+R_{\rm dark}},
\qquad
\Omega_{\rm DE}=\Omega_{\rm rem}\frac1{1+R_{\rm dark}}
$$

로 나눈다. 여기에 8절의 E1--E4, fold-to-energy matching $\beta_E=1$, 저분율
branch와 flat-slice 장부를 추가 채택하여 $\Omega_b=x_E=x_\star$로 읽고,
동시에
$\Omega_r=\Omega_\nu=\Omega_k=0$으로 자른 late-time 3-sector
benchmark에서만

$$
\boxed{(\Omega_b,\Omega_{\rm cdm},\Omega_{\rm DE})
=(0.0486382585,0.2610881744,0.6902735671)}.
$$

현재 표시 정밀도의 모든 숫자는
[D 정합성 원장](D_정합성_원장.md)의 식과 독립 테스트에서 재계산한다.

## 8. E1--E4가 $x_\star$를 물리 readout과 연결하는 법

path-count 확률과 에너지분율은 일반적으로 다르다. 깊이 $D$에서
정규화된 양의 상태 $\varrho_D$, 양의 총에너지 operator $H_D$, 고정 baryon
sector projector $\Pi_b$를 두고

$$
H_{b,D}:=H_D^{1/2}\Pi_bH_D^{1/2},
\qquad
x_E(D):=
\frac{\operatorname{Tr}(\varrho_DH_{b,D})}
{\operatorname{Tr}(\varrho_DH_D)}
$$

로 정의한다. 분모는 유한한 양수여야 한다. 고전적 사건
$\mathcal A_D$로 쓸 때에는 $H_{b,D}=H_D\mathbf1_{\mathcal A_D}$가 같은
식이다. 다음 조건을 모두 요구한다.

1. **E1, baryon sector:** $H_{b,D}$가 late-time baryon stress-energy
   sector를 나타낸다. event projector로 읽으려면 $[H_D,\Pi_b]=0$ 또는
   동등한 공변 sector 분해를 확인한다.
2. **E2, 동일 장부:** 분자와 분모를 같은 관측 초곡면·같은 comoving
   영역에서 계산한다.
3. **E3, energy-biased semigroup:** $x_E:[0,\infty)\to(0,1]$,
   $x_E(0)=1$이고 $x_E(D_1+D_2)=x_E(D_1)x_E(D_2)$이며 연속이다.
   비자명성은 어떤 $D>0$에서 $x_E(D)<1$이라는 뜻이다.
4. **E4, feedback:** $c_E>0$를 fold-to-energy depth 변환으로 두고
   $D_{\rm act}=c_E(1-x_E)D_A$다.

S1--S4에 쓴 것과 같은 함수형 정리를 E3에 독립 적용하면
$x_E(D)=e^{-\kappa_E D}$인 $\kappa_E>0$를 얻는다. 이는 survival
character의 $\kappa_{\rm surv}$와 자동으로 같은 rate가 아니다. 따라서
$x_E=e^{-\beta_E D_A(1-x_E)}$, $\beta_E:=\kappa_E c_E$다.
$\kappa_E=1$은 좌표 convention일 뿐 $\beta_E$를 고정하지 않는다. 별도
matching `Selection` $\beta_E=1$을 선택하면
$x_E=e^{-D_A(1-x_E)}$다. 저분율 branch의 유일성으로 $x_E=x_\star$가
따른다. $\beta_E=1$의 동역학적 validation은 `Open Bridge`다. E1--E2는

$$
\frac{\Omega_b}{\Omega_{\rm phys}}=x_E
$$

를 주고, flat-slice $\Omega_k=0$, 따라서 $\Omega_{\rm phys}=1$을
추가할 때에만

$$
\boxed{\Omega_b=x_E=x_\star}
$$

라고 쓴다. 이는 현재 gate가 `Open`인 `Bridge`이며 정의에 의한 항등식이
아니다. 또한 $x_E$가
우주 시간에 따라 1에서 0.05로 변한다는 주장은 별도 $D(t)$와 전달방정식,
freeze-out 초곡면 없이 따르지 않는다.

## 9. $i$가 필요한 두 완성 모형

### 9.1 flavour

하나의 unitary construction에서 CKM/PMNS를 얻고 전체 행렬의 unitarity와
Jarlskog invariant를 검사한다. 개별 원소를 서로 다른 수치식으로 맞추지
않는다.

### 9.2 strong CP

shift-symmetric pseudoscalar의

$$
\left(\bar\theta+\frac {a_{\rm ax}}{f_a}\right)G\widetilde G
$$

coupling과 QCD potential이 vacuum minimum에서 effective theta를 0으로
만든다. CP-even scalar나 $i$ 자체는 이 일을 하지 않는다.

## 10. $e$가 필요한 독립 inflation branch

이 절의 inflation 장 $\varphi$와 coupling $\xi_{\rm inf}$는 6절 core의
$\phi,\xi_{\rm core}$와 다르다. 별도 `Phenomenology` branch로

$$
F_{\rm inf}(\varphi)=M_P^2+\xi_{\rm inf}\varphi^2,
\qquad
\xi_{\rm inf}:=\alpha_s^{1/3}=0.4904868132
$$

를 선택한다. $\xi_{\rm inf}$ 등록은 `Selection`이며, core의 minus-sign
$F_{\rm core}=M_P^2-\xi_{\rm core}\phi^2$와 같은 작용에서 자동으로
나오지 않는다. quartic 배경을 large-$\xi_{\rm inf}$ 근사 없이 적분하고
관측량에는 leading-order slow-roll 식을 쓰면, 지정 입력
$N_*=57.1999$, $A_s=2.10\times10^{-9}$에서

$$
n_s=0.96617114,
\qquad r=0.00434561,
\qquad\lambda_4=1.3434991\times10^{-10}
$$

을 얻는다. $A_s$를 이용해 $\lambda_4$를 정규화했으므로 $A_s$를 다시
예측으로 세지 않는다. $n_s,r$도 지정 branch의 조건부 출력이며
reheating·RG·고차 slow-roll 보정은 `Open`이다.

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

개별 중앙값 근접과 공동 적합을 구분한다. 최신 고정 density vector,
$w_0=-1,w_a=0,H_0=67.4$, external $r_d=147.09$ Mpc를 명시 주입하고
13-component compressed DESI DR2 covariance만 계산한 BAO-only partial
gate는

$$
\chi^2=40.20145086,
\qquad p=1.28283168\times10^{-4}
$$

로 `Rejected`다. 이것은 CMB·SN·growth를 포함한 공동 likelihood가 아니며 그
full joint gate는 `Open`이다. 따라서 현재의 late-time truncated 분할 대수는
지정 ansatz 안에서 닫혀 있지만 full cosmology 자료의 최종 모형은 아니다.
다음 시도는 같은 공변 scalar-tensor action에서 background와 perturbation을
함께 풀고 추가 파라미터 penalty를 지불해야 한다.

## 13. 자동 검산

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_chapter1_document_contract.py `
  tests\test_chapter1_numeric_contract.py `
  -q -p no:cacheprovider
```

이 명령은 삭제된 전역 gate에 의존하지 않고 1장 문서 구조와 다음 수치를
다시 계산한다.

- Track A와 Track B의 근
- $s_A^2,\delta_A,D_A,x_\star,R_{\rm dark},\boldsymbol\Omega$
- 타입·지위·상대 링크와 S1--S4/E1--E4 가정의 문서 계약
- 벡터 재귀에서 scalar 식으로 줄이는 공통 행합 조건
- 선언한 CKM sine seed와 $\delta_q$에서 $J$로 가는 계산
- external/EH $r_d$ 두 분기의 DESI DR2 BAO-only $\chi^2,p$
- 별도 plus-sign inflation branch의 $n_s,r,\lambda_4$
- $H_0,T_{\rm CMB}$ 입력을 포함한 $\eta_b$ 변환

다섯 상수는 이 계산의 언어를 제공하고, 작용·입력·likelihood가 그 언어를
검증 가능한 물리 모형으로 완성한다.
