# CE 우주 밀도 다리: 공변 작용·current 후보 독립 경로 연구

Status: COMPLETE

접근일: 2026-08-15

## 1. 질문과 판정 원칙

이 문서는 계약 B1--B6에 대해, 다음 세 명제를 서로 섞지 않고 조사한다.

1. Poisson 고정점 방정식을 정지조건으로 갖는 공변 작용을 **구성할 수
   있는가**.
2. 그 고정점을 conserved particle current의 **조성비**로 만들 수 있는가.
3. 그 조성비가 추가 정규화 없이 현재의
   $\Omega_b=\rho_b/(3M_{\rm Pl}^2H^2)$가 되는가.

첫 번째 질문의 긍정은 존재구성이다. 두 번째 질문은 물질의 정체, 반응률과
보존법칙을 더 요구한다. 세 번째 질문은 다시 절대 abundance, 질량,
Friedmann 배경과 평가 hypersurface를 요구한다. 따라서 앞 질문의 긍정을
뒤 질문의 긍정으로 승격하지 않는다.

이 조사에서는 관측된 $\Omega_b$ 또는 $H_0$를 potential, coupling,
reaction rate에 넣지 않았다. 다만 이미 알려진 수치적 근접성 때문에
어떤 readout을 택했는지를 별도의 target-awareness 항목으로 감사한다.

## 2. 공통 표기와 무차원 감사

자연단위 $c=\hbar=1$, metric 부호 $(-+++)$를 쓴다. $x,D,y$는
무차원이고 $[F]=[M]=1$, number density $[n]=3$, energy density
$[\rho]=4$, 반응률 $[\Gamma]=4$이다. reduced Planck mass를 써서
$\rho_{\rm crit}=3M_{\rm Pl}^2H^2$로 둔다.

$$
f_D(z)=\exp[D(z-1)],\qquad
v_D(x)=x\log x-x+D\left(x-\frac{x^2}{2}\right)+C.
$$

| 코어 인자 | 차원 벡터 $(M,L,T,\Theta)$ | 무차원? | 필요한 정규화 |
|---|---:|---:|---|
| $x$ in $\log x$ | $(0,0,0,0)$ | yes | 없음 |
| $D(z-1)$ in $\exp[D(z-1)]$ | $(0,0,0,0)$ | yes | $D,z$를 무차원으로 정의 |
| $y=n_b/(n_b+n_r)$ | $(0,0,0,0)$ | yes | 동일 차원의 두 number density |
| $\Omega_b=\rho_b/(3M_{\rm Pl}^2H^2)$ | $(0,0,0,0)$ | yes | 임계밀도로 나눔 |
| $m_x^2=(M^4/F^2)v_D''$ | mass dimension $2$ | 해당 없음 | exp/log 인자가 아님 |
| $\kappa t$ in reaction relaxation | $(0,0,0,0)$ | yes | $[\kappa]=1$ |

차원 상태는 정합하다. 그러나 이는 각 식의 물리적 정당성이나
$x=\Omega_b$를 보장하지 않는다.

코드 검증은
`uv run --extra dev python -m pytest tests\test_dimensionless.py -q`에서
15개가 통과했고,
`python reality_stone\python\reality_stone\clarus\dimensionless.py`도
exit code 0이었다. 기존 checker가 이 문서의 신규 reacting/projector
식들을 직접 등록 검사한 것은 아니므로, 이 결과는 공통 CE 무차원 회귀가
깨지지 않았다는 범위로만 해석한다.

## 3. 1차 문헌이 제공하는 최소 구조

### 3.1 보존 perfect fluid

Brown의 vector-density 형식에서
$J^\mu=\sqrt{-g}\,nU^\mu$이고, 작용은 다음과 같다.

$$
S_{\rm B}=\int d^4x\left[-\sqrt{-g}\,\rho(n,s)
+J^\mu\left(\partial_\mu\varphi+s\partial_\mu\theta
+\beta_A\partial_\mu\alpha^A\right)\right].
$$

$\varphi$ 변분은 $\partial_\mu J^\mu=0$, 즉
$\nabla_\mu(nU^\mu)=0$을 주며 metric 변분은

$$
T^{\mu\nu}=(\rho+p)U^\mu U^\nu+pg^{\mu\nu},\qquad
p=n\frac{\partial\rho}{\partial n}-\rho
$$

를 준다. Schutz의 velocity-potential 작용과 Taub의 변분원리는 같은
보존 perfect-fluid 문제의 선행 형식이다. 이 문헌은 current와 stress
tensor를 정합적으로 만드는 법을 제공하지만, 특정 species의 우주론적
abundance는 제공하지 않는다.

### 3.2 여러 current와 반응

보존 pull-back 형식에서 matter-space three-form이 자기 species의
matter-space 좌표에만 의존하면 current는 구성상 닫혀 있다. 서로 다른
matter space에 의존하게 하거나 별도 creation clock을 넣으면
$\nabla_\mu N_i^\mu=\Gamma_i$가 가능한 interacting/reacting-fluid
형식이 된다. 두 species의 변환이라면 stoichiometry를 단순화해

$$
\nabla_\mu N_b^\mu=\Gamma,
\qquad
\nabla_\mu N_r^\mu=-\Gamma
$$

로 둘 수 있고 total current는 보존된다. Diffeomorphism invariant한 전체
작용에서는 total stress가 보존되지만, 개별 sector의 stress에는 같은
크기와 반대 부호의 exchange current가 생긴다. Iosifidis--Jensko--Koivisto
및 Andersson--Comer의 원 논문은 이런 작용의 존재를 보이지만, CE에 필요한
특정 $\Gamma$나 species 정체를 고정하지는 않는다.

### 3.3 scalar--fluid coupling

Pull-back 유체와 scalar로 만들 수 있는 저차 invariant는 number density
$n$, scalar kinetic invariant $Y$, 그리고
$Z=U^\mu\nabla_\mu\phi$이다. 따라서 일반적인 저차 작용은
$L(n,Y,Z,\phi)$로 쓸 수 있다. Pourtsidou--Skordis--Copeland는 이
공간에서 algebraic coupling과 derivative/velocity coupling을 분류했고,
Koivisto--Saridakis--Tamanini는 background와 선형 섭동을 유도했다.
후자의 중요한 제한은 dust에 유효 sound speed를 만들지 않으려면 fluid
Lagrangian의 $n$ 의존성이 선형이어야 한다는 것이다. 즉 coupling의
존재만으로 안정성이나 structure formation 적합성이 보장되지 않는다.

### 3.4 master equation과 response-field action

Doi의 classical many-particle second-quantized 형식과 Peliti의
birth--death path integral은 Markov 반응망을 response-field action으로
옮긴다. Martin--Siggia--Rose 역시 고전 확률 동역학의 response-field
표현을 제공한다. 이들은 확률 생성함수와 extinction event를 작용 언어로
표현하지만, 그 작용은 일반 Lorentzian 물질 작용이나 Hilbert stress
tensor와 동일하지 않다.

## 4. 후보 A — 무차원 scalar potential 임베딩

### 4.1 작용, EOM과 stress tensor

계약의 최소 후보는

$$
S_A=\int d^4x\sqrt{-g}\left[-\frac{F^2}{2}
g^{\mu\nu}\partial_\mu x\partial_\nu x-M^4v_D(x)\right]
$$

이다. 변분하면

$$
F^2\Box x-M^4v_D'(x)=0,
\qquad
v_D'(x)=\log x+D(1-x),
$$

$$
T^{(x)}_{\mu\nu}=F^2\partial_\mu x\partial_\nu x
-g_{\mu\nu}\left[\frac{F^2}{2}(\partial x)^2+M^4v_D(x)\right]
$$

를 얻는다. 평탄 FLRW의 균일 mode는

$$
\ddot x+3H\dot x+\frac{M^4}{F^2}v_D'(x)=0
$$

을 따른다. 따라서 $v_D'(x)=0$은 정확히
$x=\exp[-D(1-x)]$이다. 이 단계에는 관측 밀도나 $H_0$가 없다.

### 4.2 가지, 영역과 안정성

$D>1$에서 정지점은 작은 근 $q_*$와 $x=1$이다.

$$
v_D''(x)=\frac1x-D.
$$

$D-1-\log D>0$이므로 작은 근은 $q_*<1/D$이고
$v_D''(q_*)>0$이다. 반면 $v_D''(1)=1-D<0$이다. 따라서 명시한
영역 $(0,1]$ 안에서 $q_*$는 안정한 최소점이고 $x=1$은 불안정한
끝점 정지점이다. 선형 perturbation은

$$
\delta\ddot x_k+3H\delta\dot x_k+
\left(\frac{k^2}{a^2}+m_x^2\right)\delta x_k=0,
\qquad
m_x^2=\frac{M^4}{F^2}\left(\frac1{q_*}-D\right)>0
$$

를 따른다. $F^2>0$이면 ghost가 없고 canonical scalar의
$c_s^2=1$이므로 gradient instability가 없으며, $H>0$이면 homogeneous
mode에 마찰이 있다.

이 안정성은 **지정 영역에 조건부**이다. 같은 식을 $x>1$로 연장하면
$-Dx^2/2$ 때문에 potential은 아래로 무한히 열려 있다. $x\leq1$과
$x>0$를 action 자체가 강제하지 않으므로 hard wall, constrained target
space, 또는 bounded field reparameterization이 추가되어야 한다. 또한
$x\to0^+$에서 potential은 유한하지만 $v_D'$는 발산한다.

상수 $C$는 scalar EOM에는 안 들어가지만 중력에서는 무해하지 않다.
$M^4C$만큼 vacuum energy와 Friedmann 방정식을 옮긴다. 그러므로 $C$를
“동역학에 무관한 상수”라고만 말할 수 있고 “우주론에 무관한 상수”라고
말할 수는 없다.

### 4.3 보존법칙과 $\Omega_b$ 동일시 문제

$x=q_*$인 균일 정지해의 stress는

$$
\rho_x=M^4v_D(q_*),\qquad p_x=-M^4v_D(q_*)
$$

이므로 $w_x=-1$이다. 이는 $p_b\simeq0$이고
$\rho_b\propto a^{-3}$인 baryon dust가 아니다. $C$를 골라
$v_D(q_*)=0$으로 만들면 scalar background energy가 사라질 뿐,
baryon energy가 생기지 않는다. 따라서 이 작용은 무차원 VEV를 만들지만
baryon current나 baryon abundance를 만들지 않는다.

### 4.4 자유도, target-awareness와 kill test

- 추가 구조: $D$의 미시적 출처, field domain, $F/M$, vacuum offset $C$,
  초기조건의 basin.
- target-awareness: potential은 관측 중심값이 아니라 고정점 방정식의
  antiderivative로 설계되었다. 그러므로 **방정식-aware**이지만 그 자체로
  관측 target-aware는 아니다. 그러나 수치적 근접성을 본 뒤
  $x\mapsto\Omega_b$를 고른 readout은 target-aware이다.
- 독립 산출: 두 가지 branch, $q_*$의 국소 안정성,
  $m_x^2F^2/M^4=1/q_*-D$만 조건부로 산출된다.
- kill A1: 관측값을 쓰지 않은 bounded-below UV/EFT completion이 같은
  $q_*$를 유지하는가.
- kill A2: radiative correction가 $v_D'$와 작은 근을 이동시키지 않게 하는
  symmetry 또는 기술적 자연성이 있는가.
- kill A3: 이 sector를 baryon으로 부를 경우 $w=-1$ 반례를 피하는 별도
  current가 있는가.
- kill A4: $C,F,M$을 우주 관측에 맞춰 고르지 않고 독립적으로 고정할 수
  있는가.

판정: B1과 B3의 scalar 존재구성은 성립하고 B2는 명시 영역에 조건부로
성립한다. B4와 B5의 density 주장은 성립하지 않는다.

## 5. 후보 B — Lagrange constraint를 가진 two-current 조성비

### 5.1 보존형 작용

두 Brown current $J_b^\mu,J_r^\mu$와

$$
n_i=\frac{\sqrt{-J_i^\mu g_{\mu\nu}J_i^\nu}}{\sqrt{-g}},
\qquad
y=\frac{n_b}{n_b+n_r}
$$

를 둔다. 가장 직접적인 존재구성은

$$
S_B=S_A+\int d^4x\left[-\sqrt{-g}\,\rho(n_b,n_r,s_i,x)
+\sum_{i=b,r}J_i^\mu\mathcal C^{(i)}_\mu
+\sqrt{-g}\,\Lambda(y-x)\right],
$$

$$
\mathcal C^{(i)}_\mu=\partial_\mu\varphi_i
+s_i\partial_\mu\theta_i+\beta_{iA}\partial_\mu\alpha_i^A
$$

이다. $[\Lambda]=4$이고 모든 항의 dimension은 4이다. 이 작용은
$y=x$를 강제하고 두 current의 보존을 동시에 구현할 수 있다.

그러나 이 구성만으로 $y=q_*$가 선택되지는 않는다. $\varphi_i$ 변분으로
$\nabla_\mu N_i^\mu=0$이고, 균일 FLRW에서는

$$
n_i=\frac{N_i}{a^3},\qquad
y=\frac{N_b}{N_b+N_r}
$$

이다. $y$는 두 conserved charge의 초기 비율이다. $y=x$ constraint를
추가하면 scalar 방정식에서 $\Lambda$가 potential force를 상쇄할 수 있어,
potential이 charge 비율을 $q_*$로 바꾸지 못한다. 즉 이 형태는 임의의
초기 비율에 이름표를 붙이는 존재구성이다.

### 5.2 반응형 개선과 composition attractor

조성 자체를 potential minimum으로 이완시키려면 두 current가 반응해야
한다. total number $n=n_b+n_r$를 보존하면서

$$
\dot n_b+3Hn_b=\Gamma,
\qquad
\dot n_r+3Hn_r=-\Gamma,
\qquad
\dot y=\frac{\Gamma}{n}
$$

로 둔다. 예를 들어 관측값을 쓰지 않은 constitutive law

$$
\Gamma=-\kappa n\,v_D'(y),\qquad \kappa>0
$$

는

$$
\dot y=-\kappa v_D'(y)
$$

를 주며, $(0,1)$의 basin에서 $q_*$를 안정하게 하고 $y=1$을 불안정하게
한다. 선형 이완률은
$\kappa v_D''(q_*)>0$이다. 같은 구조는 mixture free energy에
$n\mu_*v_D(y)$를 넣고 chemical affinity에 비례하는 반응률을 두는 방식으로
열역학적으로 쓸 수 있다.

이 결과는 중요한 **존재 증명**이다. 즉 Poisson 근을 conserved total
current의 late-time number composition으로 만드는 반응망은 구성할 수
있다. 하지만 $\kappa$, reaction channel, entropy production law와 species
정체가 새 입력이다. 이상적인 보존 Brown action만으로는 dissipative
attraction이 생기지 않는다. reacting-fluid action, doubled
Schwinger--Keldysh/MSR variables 또는 다른 비평형 closure가 필요하다.

### 5.3 number fraction은 critical-density fraction이 아니다

$y=q_*$가 닫혀도 일반적으로

$$
y=\frac{n_b}{n_b+n_r}
\ne
\frac{m_bn_b}{m_bn_b+m_rn_r}
\ne
\frac{\rho_b}{3M_{\rm Pl}^2H^2}=\Omega_b.
$$

첫 등호를 얻으려면 species masses와 internal energy를 고정해야 한다.
두 번째 등호를 얻으려면 reservoir가 현재의 **모든** 임계 에너지를
구성하고 평탄 Friedmann constraint를 만족해야 한다. 그러나 radiation,
dust와 vacuum energy는 서로 다르게 redshift하므로 하나의 보존 dust
current 비율이 모든 epoch의 $\rho_{\rm tot}$ 비율이 될 수 없다.

특히 baryon current가 보존되고 질량이 일정하면

$$
\dot\rho_b+3H\rho_b=0,
$$

$$
\frac{d\ln\Omega_b}{d\ln a}
=-3-2\frac{d\ln H}{d\ln a}.
$$

평탄 GR에서
$d\ln H/d\ln a=-\tfrac32(1+w_{\rm eff})$이면

$$
\frac{d\ln\Omega_b}{d\ln a}=3w_{\rm eff}.
$$

따라서 $x=q_*$가 상수여도 $\Omega_b$는 total background가 정확히
dust인 구간을 제외하면 상수가 아니다. 한 epoch에서만
$x=\Omega_b$라고 하려면 그 epoch를 고르는 covariant event 또는
freeze-out hypersurface가 추가되어야 한다.

더 근본적으로, 반응이 끝난 뒤
$\rho_b(t_0)=m_bN_b/a_0^3$의 normalization $N_b$는 total conserved
charge이다. $q_*$는 $N_b/N$를 고를 수 있어도 $N$이나 $H(t_0)$를
고르지 못한다. 이것이 B4--B5의 절대 abundance no-go이다.

### 5.4 안정성, target-awareness와 kill test

- thermodynamic stability: 고정 total $n$에서 composition Hessian이
  양수여야 하며, 위 최소 모형에서는
  $n\mu_*v_D''(q_*)>0$이다.
- hydrodynamic stability: $\partial^2\rho/\partial n_i\partial n_j$의
  적절한 convexity와 sound-speed eigenvalue가 비음수가 되어야 한다.
  $v_D''>0$ 하나로는 이를 보장하지 않는다.
- reaction stability: Onsager coefficient 또는 $\kappa$가 양수이고 entropy
  production이 비음수가 되어야 한다.
- 추가 구조: species, masses/EOS, total charge normalization, reaction
  network와 rate, entropy/temperature sector, freeze-out event.
- target-awareness: $y$를 선택한 이유가 이미 본
  $q_*\simeq\Omega_b$ 근접성뿐이면 bridge는 target-aware이다. 반응망이
  독립 미시모형에서 먼저 정해져야 이 문제를 벗어난다.
- kill B1: 미시 반응에서 $\Gamma=-\kappa nv_D'(y)$ 또는 동등한 detailed
  balance가 유도되는가.
- kill B2: total stress 보존과 $\nabla_\mu S^\mu\ge0$가 동시에 닫히는가.
- kill B3: number fraction을 energy fraction으로 바꾸는 masses와 EOS가
  독립적으로 정해지는가.
- kill B4: total charge와 reheating entropy가 관측 $\Omega_b,H_0$ 없이
  정해지는가.
- kill B5: 반응 종료 후 baryon current가 보존되고
  $\rho_b\propto a^{-3}$가 background와 perturbation 모두에서 유지되는가.

판정: B는 $q_*$를 **number composition attractor**로 만드는 가장 직접적인
경로지만, 그 결과를 $\Omega_b(t_0)$로 승격할 수는 없다.

## 6. 후보 C — Poisson branching master equation / Doi--Peliti 경로

### 6.1 고정점의 미시 확률 의미

offspring 분포를 처음부터

$$
p_k=e^{-D}\frac{D^k}{k!},\qquad
f_D(z)=\sum_{k=0}^{\infty}p_kz^k=e^{D(z-1)}
$$

로 둔 Galton--Watson 계보에서 extinction probability는

$$
q=f_D(q)=e^{-D(1-q)}
$$

를 만족한다. 이 route는 scalar potential보다 원래 확률 명제에 가깝다.
$q_*$는 한 계보가 결국 사라질 확률이고 $q=1$은 전체 구간의 다른
고정점이다. generation map의 미분은

$$
f_D'(q_*)=Dq_*<1,
\qquad
f_D'(1)=D>1
$$

이므로 두 가지의 안정성도 계약과 일치한다.

### 6.2 response-field action

각 개체가 rate $\gamma$로 Poisson offspring에 교체되는 zero-dimensional
continuous-time branching을 쓰면, 부호 convention을 제외한
Doi--Peliti coherent-state action은

$$
S_C=\int dt\left[
\bar\phi\,\dot\phi
-\gamma\phi\{f_D(\bar\phi)-\bar\phi\}
\right]
$$

로 쓸 수 있다. response saddle의 정지조건에
$f_D(\bar\phi)=\bar\phi$가 들어간다. 공간 확산과 팽창 유체를 넣은
effective 표현은 개략적으로

$$
S_C^{\rm eff}=\int d^4x\sqrt{-g}\left[
\bar\phi\,U^\mu\nabla_\mu\phi
-K h^{\mu\nu}\nabla_\mu\bar\phi\nabla_\nu\phi
-\gamma\phi\{f_D(\bar\phi)-\bar\phi\}
\right],
$$

$$
h^{\mu\nu}=g^{\mu\nu}+U^\mu U^\nu
$$

처럼 쓸 수 있다. stochastic density convention
$[\bar\phi]=0$, $[\phi]=3$, $[\gamma]=1$, $[K]=-1$을 택하면 각 항의
mass dimension은 4이고 지수 인자는 무차원이다. 그러나 $U^\mu$가 preferred
reaction time을 정하므로 이는 일반좌표 공변 표기를 가질 수 있어도
local Lorentz invariant한 fundamental matter action은 아니다.

### 6.3 stress tensor와 밀도 사상의 결손

$\bar\phi$는 독립적인 response/contour field이고 $S_C$는 확률의 생성
functional이다. 실수 Lorentzian scalar의 kinetic energy처럼 ghost
판정을 할 대상이 아니며, 이를 metric으로 변분한 양을 물리적 baryon
stress tensor라고 동일시할 수도 없다. energy와 momentum은 실제 branching
objects를 운반하는 substrate의 action에서 따로 나와야 한다.

또한 extinction probability와 공간 또는 에너지분율은 서로 다른
measure이다. 많은 독립 계보에서 소멸한 계보의 빈도가 $q_*$로 가는 것은
law of large numbers의 결과일 수 있지만,

$$
\text{extinct-lineage frequency}
\longrightarrow
\frac{\rho_b}{\rho_{\rm crit}}
$$

에는 causal patch measure, 계보당 energy weight, ergodicity, coarse-graining
slice가 모두 더 필요하다. 계보가 소멸했을 때 baryon이 남는지 사라지는지
조차 별도 물리 규칙이다.

### 6.4 자유도, target-awareness와 kill test

- 추가 구조: branching object, generation/event clock, iid Poisson 법칙,
  $D$의 미시 유도, rate $\gamma$, substrate stress, spacetime measure,
  freeze-out/coarse-graining.
- target-awareness: Poisson extinction 식 자체는 CE 확률 코어에서 먼저
  정의되어 있다. 하지만 extinction measure를 baryon energy measure로
  고르는 순간은 별도 target-aware bridge이다.
- 독립 산출: offspring law가 주어졌을 때 extinction probability와 total
  progeny statistics는 예측할 수 있다. 우주론적 $\Omega_b$는 아니다.
- kill C1: 실제 미시 사건이 독립·정상 Poisson이고 평균 $D$가 action 또는
  scattering amplitude에서 유도되는가.
- kill C2: generation time과 causal patch measure가 covariant하게 정의되는가.
- kill C3: substrate의 conserved total stress와 baryon current가 별도
  action에서 닫히는가.
- kill C4: extinction frequency와 energy density의 동일시가 임의 weight
  선택 없이 유도되는가.
- kill C5: response-field saddle을 양자 Lorentzian field나 물리적 VEV로
  오독하지 않는가.

판정: C는 $q_*$의 가장 자연스러운 **확률적 의미**를 보존하지만,
공변 baryon density bridge로는 가장 많은 추가 구조가 필요하다.

## 7. 후보 D — conserved baryon current와 scalar--fluid coupling

### 7.1 최소 algebraic coupling

Brown baryon current와 후보 A를 결합하고 dust energy를

$$
\rho_b(n_b,x)=m_0A(x)n_b
$$

로 두자. $A(x)$는 무차원 양의 함수이다. 작용은

$$
S_D=S_A+\int d^4x\left[
-\sqrt{-g}\,m_0A(x)n_b+J_b^\mu\mathcal C^{(b)}_\mu
\right]
$$

이다. 이는 Pourtsidou--Skordis--Copeland의 algebraic scalar--fluid
class의 단순한 dust 특수형이다. 변분하면

$$
\nabla_\mu(n_bU_b^\mu)=0,
$$

$$
F^2\Box x-M^4v_D'(x)-m_0n_bA'(x)=0
$$

를 얻는다. 평탄 FLRW에서 정지조건은

$$
M^4v_D'(x)+m_0n_bA'(x)=0
$$

이다. fluid Lagrangian이 $n_b$에 선형이므로 background pressure는
$p_b=n_b\partial\rho_b/\partial n_b-\rho_b=0$이다. 전체 stress는
보존되고 scalar와 baryon sector 사이에는 $A'(x)$에 비례하는 energy--momentum
exchange가 생긴다. particle number conservation과 개별 energy conservation은
같은 명제가 아니라는 점이 중요하다.

### 7.2 density-dependent minimum 문제

$x=q_*$가 모든 cosmological density에서 그대로 최소점이 되려면

$$
A'(q_*)=0
$$

이어야 한다. 그렇지 않으면 $n_b\propto a^{-3}$가 변할 때 effective
minimum도 이동하여 Poisson 근이 보존되지 않는다. $A'(q_*)=0$이면

$$
m_{\rm eff}^2=\frac{1}{F^2}\left[
M^4v_D''(q_*)+m_0n_bA''(q_*)
\right]>0
$$

가 tachyon 회피 조건이다. canonical scalar에는 기본적으로 ghost와
gradient 문제가 없지만, coupled scalar--fluid 전체의 kinetic/gradient
matrix와 matter growth는 별도 계산이 필요하다.

그러나 $A'(q_*)=0$은 abundance 문제를 해결하지 않는다. 균일 해에서

$$
\rho_b(t_0)=m_0A(q_*)\frac{N_b}{a_0^3},
$$

이고 $N_b$는 conserved integration constant, $m_0A(q_*)$는 질량
normalization이다. potential은 어느 것도 고르지 않는다. $A(q_*)$를
원하는 density에 맞춰 선택하면 target을 coupling에 숨기는 금지된
구성이 된다.

### 7.3 일반 $L(n,Y,Z,x)$ 경로의 대가

더 일반적인 algebraic $f(n,x)$ 또는 derivative $Z=U^\mu\partial_\mu x$
coupling도 가능하다. 하지만 함수 하나를 자유롭게 고를 때마다 background
fixed point, sound speed, fifth force와 perturbation growth를 함께 바꾼다.
특히 nonlinear $n$ dependence는 dust에 유효 pressure/sound speed를 만들 수
있으므로 structure formation이 강하게 제한한다. derivative coupling은
momentum exchange를 만들며 kinetic matrix의 양의 고유값을 직접 검사해야
한다.

### 7.4 자유도, target-awareness와 kill test

- 추가 구조: $A(x)$ 또는 일반 $f(n,Y,Z,x)$, $m_0$, conserved charge
  $N_b$, $F/M/C$, 초기조건, screening 또는 coupling suppression.
- target-awareness: $A(q_*)$나 $A'(q_*)$를 수치 일치 후 고르면 매우 큰
  함수 선택 자유도를 이용한 target-aware 구성이다.
- 독립 산출: coupling을 사전 고정한 뒤에는 mass variation, fifth force,
  background energy exchange와 growth modification가 반증 가능한 산출이다.
- kill D1: 모든 relevant $n_b$에서 $q_*$가 유지되거나 계산 가능한 이동만
  하는가.
- kill D2: 전체 quadratic action의 no-ghost, no-gradient, subluminality 또는
  명시한 EFT cutoff 조건이 닫히는가.
- kill D3: dust sound speed, CMB/BBN의 baryon mass 안정성, equivalence
  principle과 fifth-force 제한을 통과하는가.
- kill D4: $N_b$, reheating entropy와 $H(t_0)$가 $A(x)$ 바깥에서
  독립적으로 유도되는가.
- kill D5: coupling 함수 후보군과 선택 규칙을 관측 비교 전에 동결했는가.

판정: D는 physical current와 total stress를 가장 깨끗하게 제공하지만,
scalar VEV에서 absolute baryon abundance로 가는 정리는 제공하지 않는다.

## 8. 후보 E — energy-weighted event projector

### 8.1 정확한 확률--에너지 항등식

어떤 event $E$의 indicator를 $I_E$라 하고
$P(E)=\mathbb E[I_E]=q\in(0,1)$라 하자. 동일한 ensemble의 각 sample에
유한한 양의 energy weight $W$를 부여하고

$$
\Omega_E=\frac{\mathbb E[I_EW]}{\mathbb E[W]},
\qquad 0<\mathbb E[W]<\infty
$$

로 energy-weighted fraction을 정의하면 정확히

$$
\Omega_E-q
=\frac{\operatorname{Cov}(I_E,W)}{\mathbb E[W]}
$$

이다. 조건부 평균을

$$
\mu_E=\mathbb E[W\mid E],
\qquad
\mu_{\bar E}=\mathbb E[W\mid E^c]
$$

로 쓰면

$$
\operatorname{Cov}(I_E,W)
=q(1-q)(\mu_E-\mu_{\bar E})
$$

이므로

$$
\boxed{\Omega_E=q
\quad\Longleftrightarrow\quad
\mu_E=\mu_{\bar E}}
$$

이다. 이는 $q\in(0,1)$에서 필요충분조건이다. 완전한 통계적 독립
$I_E\perp W$는 충분조건이지만 필요조건보다 강하다.

이 항등식은 probability를 energy fraction으로 바꾸는 데 숨겨진 가정을
정확히 노출한다. 단순히 $P(E)=q$라는 이유로 $\Omega_E=q$가 되는 것이
아니라, event 양쪽의 조건부 평균 energy가 같아야 한다.

### 8.2 product factorization과 equal-energy symmetry

가장 단순한 존재구성은 selector와 physical energy sector가 factorize되는
경우이다.

$$
\mathcal P(E,\Psi)=\mathcal P_{\rm sel}(E)\mathcal P_W(\Psi),
\qquad
W=W[\Psi,g].
$$

경로적분 언어에서는 개략적으로
$Z=Z_{\rm sel}Z_W$이고 selector가 $W$를 운반하는 field와 coupling하지
않는다. 이때 $I_E$와 $W$는 독립이므로 $\Omega_E=q$가 정확하다.

하지만 완전 factorization에서는 $E$가 physical energy에 아무 영향도
주지 않는 epiphenomenal label이다. 그 label을 “baryon”이라고 부르고
$\rho_b:=I_EW$라고 정하는 순간 물리 사상은 다시 공리로 들어온다.

조금 더 물리적인 존재구성은 두 sector $b,r$가 internal symmetry 아래
서로 바뀌고 그 symmetry가 stress tensor와 commute하는 경우이다. 즉

$$
S_b[\Psi_b,g]\simeq S_r[\Psi_r,g],
\qquad
T^{(b)}_{\mu\nu}\simeq T^{(r)}_{\mu\nu}
$$

로 energy spectrum과 homogeneous EOS가 같고, Poisson selector만
$P(b)=q$를 정한다. 그러면 label별 conditional mean energy가 같아
matter-sector energy fraction이 $q$가 될 수 있다. 이 route는 후보 B의
equal-mass/equal-EOS two-current를 symmetry로 보호하는 특수형이다.

### 8.3 왜 $\Omega_b/\Omega_m$에는 자연스럽고 $\Omega_b$에는 부족한가

두 sector가 모두 separately conserved dust이고 단위 event당 질량분포가
같다면

$$
\frac{\rho_b}{\rho_b+\rho_r}=q
$$

가 모든 dust-dominated epoch에서 유지될 수 있다. $r$을 다른 matter
sector로 해석하면 이는 자연스럽게

$$
f_b=\frac{\Omega_b}{\Omega_m}
$$

형태의 **matter composition fraction**이다. 그러나 이는
$\Omega_b=\rho_b/\rho_{\rm crit}$와 다른 observable이다. 이 route로
$\Omega_b=q$까지 가려면 추가로 $\Omega_m=1$이거나 denominator에 들어가는
모든 non-baryonic energy를 equal-energy complement로 포함해야 한다.

후자는 dark energy가 공존할 때 구조적으로 문제가 된다. 서로 보존되는
상수 EOS sector에서 조건부 평균 weight는

$$
\mu_i(a)=\mu_i(a_f)
\left(\frac{a}{a_f}\right)^{-3(1+w_i)}
$$

처럼 변한다. $a_f$에서 $\mu_E=\mu_{\bar E}$를 맞춰도
$w_E\ne w_{\bar E}$이면 다음 epoch에는 equality가 깨진다. baryon dust는
$w_b\simeq0$인 반면 vacuum-like dark energy는 $w\simeq-1$이므로, 두
conditional energy를 모든 시각에 같게 하는 exact exchange symmetry는
동시에 서로 다른 EOS를 허용하지 못한다.

평탄 Friedmann constraint
$\rho_{\rm tot}=3M_{\rm Pl}^2H^2$를 쓰면 total-energy share는 형식상
$\Omega_b$가 된다. 그러나 complement가 CDM, radiation과 DE의 혼합이면
그 conditional mean은 단일 dust weight가 아니며 redshift law도 baryon과
다르다. equality가 우연히 한 hypersurface에서만 성립할 수는 있지만,
그 hypersurface를 “현재”로 고르는 것은 평가시각/normalization을 다시
넣는 것이다.

따라서 equal-energy symmetry는 같은 EOS를 가진 matter doublet의
composition에는 자연스럽지만, baryon과 전체 critical-energy complement를
교환하는 symmetry로는 자연스럽지 않다. symmetry를 깨서 DE를 허용하면
$\operatorname{Cov}(I_E,W)$가 일반적으로 0이 아니며, 그 breaking 계수가
새 자유도가 된다.

### 8.4 공변성, 보존법칙과 finite-sample 문제

$W$를 FLRW comoving observer가 측정한 양의 local energy
$T_{\mu\nu}U^\mu U^\nu$의 적분으로 정의하면 background에서는 공변 scalar
weight를 만들 수 있다. 그러나 slice 전체의 energy, causal patch와
ensemble measure는 별도로 고정해야 한다. 일반 GR에는 임의 시공간에서
유일한 양의 local gravitational-energy density가 없으므로 $W$는 matter
stress 또는 명시한 quasi-local prescription에 한정해야 한다.

많은 독립 event의 empirical weighted ratio가 위 기대값으로 수렴하려면
ergodicity와 적어도 유한한 first moment가 필요하다. variance까지 유한해야
통상적인 $N^{-1/2}$ fluctuation 추정이 가능하다. heavy-tail weight나 event
사이 상관이 있으면 한 우주의 spatial fraction이 ensemble ratio를 재현하지
않을 수 있다.

### 8.5 자유도, target-awareness와 kill test

- 추가 구조: event의 physical 의미, weight functional $W$, ensemble/patch
  measure, slicing, $E$와 $E^c$ sector의 action, equal-energy symmetry와 그
  breaking.
- target-awareness: $q$를 본 뒤 $W$를 골라 covariance를 0으로 만드는 것은
  임의 projector engineering이다. $E,W$와 symmetry가 관측 비교 전에
  고정되어야 한다.
- 독립 산출: exact symmetry가 있으면 matter composition fraction,
  symmetry breaking이 있으면
  $\Omega_E-q=\operatorname{Cov}(I_E,W)/\mathbb E[W]$라는 편차를
  독립적으로 예측할 수 있다.
- kill E1: $\mu_E=\mu_{\bar E}$가 정의가 아니라 action의 exact symmetry
  또는 factorization에서 따라오는가.
- kill E2: 그 symmetry가 reheating, Standard Model interaction과 RG
  correction 뒤에도 유지되는가.
- kill E3: baryon과 complement가 다른 EOS를 가질 때 현재 equality를
  고르는 time-boundary parameter가 숨어 있지 않은가.
- kill E4: $W$가 양의 물리 energy이고 같은 covariant hypersurface에서
  정의되는가.
- kill E5: denominator가 $\rho_m$인지 $\rho_{\rm crit}$인지 사전에
  구분했는가. $f_b=q$를 $\Omega_b=q$로 바꾸어 쓰면 즉시 기각한다.
- kill E6: finite-sample fluctuation와 isocurvature/segregation을 계산했는가.

판정: E의 확률 항등식은 `[정리]`이다. equal-energy dust doublet은
$q_*$를 matter composition으로 바꾸는 정직한 존재구성이다. 그러나 DE와
다른 EOS를 포함한 critical-density complement에는 같은 symmetry가
유지되지 않으므로 $\Omega_b=q_*$의 자연 유도는 아니다.

## 9. 후보 비교

| 항목 | A: scalar VEV | B: two-current/reaction | C: Doi--Peliti | D: scalar--fluid | E: weighted projector |
|---|---|---|---|---|---|
| $q_*$가 나오는 이유 | engineered potential minimum | engineered chemical minimum | Poisson extinction | scalar minimum | event probability |
| local covariant Lorentzian action | 조건부 yes | 보존형 yes; 이완은 추가 | no, stochastic effective | yes | sector action에 조건부 |
| explicit baryon current | no | species 공리 후 yes | no | yes | physical sector를 붙이면 yes |
| total stress conservation | scalar yes | 완성 작용에서 yes | substrate 없이는 미정 | yes | sector action에 달림 |
| $\rho_b\propto a^{-3}$ | no | 평형/반응 종료 후 가능 | no | 가능 | equal-EOS dust이면 가능 |
| $q_*$가 composition을 고름 | no | reaction law 아래 yes | extinction frequency | no | equal-energy symmetry 아래 yes |
| $q_*\Rightarrow\Omega_b(t_0)$ | no | no | no | no | no; matter fraction까지만 자연스러움 |
| 주된 자유도 | scales, offset, domain | EOS, rate, charge, freeze-out | object, rate, measure | coupling, mass, charge | event, weight, symmetry, slice |
| 가장 강한 장점 | B1--B3 단순성 | 실제 composition attractor | Poisson 의미 보존 | current+stress 틀 | probability--energy 조건을 정확히 노출 |
| 가장 강한 반례 | $w=-1$ | number와 critical fraction 불일치 | 물질 action 아님 | abundance integration constant | DE와 equal-energy/EOS symmetry 충돌 |

### 독립 순위

1. **공변 current/stress의 기반:** D와 Brown형 B가 공동 1순위이다.
2. **$q_*$의 미시 의미:** C가 1순위이다.
3. **composition attractor:** reacting B가 동역학적 1순위이고, E는
   equal-energy symmetry가 실제로 있을 때의 대수적 1순위이다.
4. **절대 $\Omega_b$ 자연 유도:** 통과한 후보가 없다.

후보를 결합하면 존재구성은 더 강해질 수 있지만, 자유도를 세 경로에서
가져와 관측값 하나를 맞추면 설명력이 오히려 감소한다. 결합 action은
symmetry 또는 미시 반응망이 함수 형태와 계수를 먼저 고를 때만 승격
가능하다.

## 10. B1--B6 형식 판정

| 계약 명제 | 판정 | 근거 |
|---|---|---|
| B1 변분 임베딩 | `[정리: 존재구성]` | 후보 A의 E-L 정지조건이 정확히 고정점 식 |
| B2 가지와 안정성 | `[정리: 조건부]` | $(0,1]$, $D>1$, $F^2>0$에서 $q_*$ 최소, $1$ 불안정; global domain completion은 미완성 |
| B3 공변 동역학 | `[정리: scalar sector]` | dimension-four action, EOM, stress, FLRW perturbation이 닫힘; $C$와 domain은 추가 입력 |
| B4 물리 사상 | `[미완성: no-go 확정]` | scalar VEV alone에는 current, mass, total abundance와 $H$가 없음; weighted projector도 equal conditional energy라는 추가 공리가 필요 |
| B5 보존법칙 | `[정리: 제한적 no-go]` | conserved dust는 $a^{-3}$지만 constant $q_*$가 일반 배경의 $\Omega_b$를 고정하지 못함; reacting B는 number composition까지만 가능 |
| B6 승격 조건 | `[미완성]` | species, reaction, absolute normalization, freeze-out/evaluation event와 blind protocol 미고정 |

현재 승격 가능한 `[예측]`은 없다. 다음 항목만 분리해 유지할 수 있다.

- `[정리]` 고정점 식의 scalar potential 존재구성.
- `[정리]` 명시 영역에서 branch Hessian과 canonical perturbation 안정성.
- `[정리]` reacting two-current가 적절한 positive kinetic law 아래
  $q_*$ number composition attractor를 가질 수 있다는 존재구성.
- `[정리]` energy-weighted projector에서 $\Omega_E=q$의 필요충분조건은
  두 event sector의 conditional mean energy가 같다는 항등식.
- `[공리 후보]` 어떤 current가 baryon인지, 어떤 reservoir와 반응하는지,
  reaction/free-energy 함수가 왜 $v_D$인지.
- `[미완성]` total abundance와 Friedmann normalization에서
  $\Omega_b(t_0)$를 산출하는 다리.

## 11. 가장 작은 후속 유도 순서

가능성을 가장 적은 자유도로 시험하려면 다음 순서를 권한다.

1. **미시 event 정의:** Lorentz-covariant microphysics에서 offspring 또는
   conversion channel을 하나 고정하고 $D$를 계산한다. iid Poisson이
   나오지 않으면 후보 C와 $v_D$ 기반 chemical law를 즉시 기각한다.
2. **reacting two-current action:** baryon-number-carrying current와 partner
   current의 matter space, EOS, stoichiometry를 고정한다. $\Gamma$와 entropy
   production을 action/constitutive relation에서 유도한다.
3. **composition theorem:** 넓은 초기조건에서 $y\to q_*$인지 background와
   perturbation 모두에서 증명한다. 동시에 weighted projector의
   $\mu_E=\mu_{\bar E}$가 독립 symmetry에서 나오는지 검사한다. 이때 얻는
   명제는 아직 matter composition이지 $\Omega_b=q_*$가 아니다.
4. **freeze-out 또는 exact equilibrium:** baryon current가 언제부터
   보존되는지 covariant event로 고정한다. 현재 cosmic time을 직접
   넣어서는 안 된다.
5. **absolute abundance:** conserved total charge 또는 baryogenesis
   asymmetry와 reheating entropy를 같은 미시모형에서 계산한다. 이 단계가
   $N_b/a^3$를 정한다.
6. **Einstein--Boltzmann 진화:** 계산된 stress tensor로 $H(a)$를 풀고,
   그 뒤에 처음으로 $\Omega_b(a)=\rho_b/(3M_{\rm Pl}^2H^2)$를 읽는다.
7. **blind kill test:** action, parameter prior, evaluation event와 비교량을
   숨겨진 자료 공개 전에 동결한다.

이 순서에서 1--3단계가 성공하면 CE는 “Poisson 고정점에서 조성비가
생긴다”는 물리 다리를 얻는다. 4--6단계까지 닫혀야 비로소
$\Omega_b$ 산출을 주장할 수 있다.

## 12. 재개·중단 기준

### 재개 조건

- $D$와 Poisson/반응 생성함수를 산출하는 명시적 미시 Lagrangian 또는
  검증 가능한 reaction network.
- species-resolved current, EOS, entropy current와 freeze-out condition.
- 관측 $\Omega_b,H_0$를 쓰지 않은 total charge/reheating normalization.
- full scalar--fluid quadratic action의 stability 계산.
- event projector를 쓸 경우 $W$, hypersurface와 equal-energy symmetry의
  관측 독립적 정의.

### 즉시 중단 조건

- $x:=\Omega_b$ 또는 $y:=\Omega_b$를 정의로 넣은 뒤 동일시를 유도라고
  부르는 경우.
- $A(q_*)$, $C$, $M/F$, $\kappa$ 또는 freeze-out time을 현재 관측값에
  맞춰 정한 경우.
- conserved current가 없는데 scalar vacuum energy를 baryon dust라고 부르는
  경우.
- extinction probability를 spacetime energy fraction과 같은 measure라고
  근거 없이 놓는 경우.
- $\Omega_b/\Omega_m=q$인 equal-dust 결과를
  $\Omega_b=q$라고 바꾸어 쓰는 경우.
- 여러 coupling/readout 중 관측값에 가장 가까운 것만 남기는 경우.

## 13. 결론

공변 작용의 **존재**는 충분히 가능하다. 후보 A는 고정점 식을 안정한
무차원 scalar VEV로 구현하고, Brown/Schutz 및 scalar--fluid 문헌은
particle current와 total stress를 공변적으로 붙일 수 있게 한다. 더 나아가
reacting two-current 후보 B는 관측값을 rate에 넣지 않고도 $q_*$를 안정한
number composition으로 만드는 명시적 존재구성을 제공한다. 후보 E는
확률이 energy fraction이 되기 위한 정확한 조건을 제시하며, equal-energy
dust doublet에서는 matter composition을 symmetry로 보호할 수 있다.

그러나 어느 후보도 $q_*$ 하나에서 $\Omega_b(t_0)$를 자연 유도하지 않는다.
남는 결손은 수학적 기교가 아니라 물리량의 종류가 다르다는 데 있다:
VEV, extinction probability, number composition, matter energy fraction과
critical-density fraction은 서로 다른 객체다. DE가 공존하면 baryon과
complement의 EOS가 달라 후보 E의 equal-energy symmetry도 일반적으로
깨진다. 가장 유망한 다음 단계는 B와 D를
결합한 reacting covariant two-current를 미시 반응망에서 고정하고, 그 뒤
별도로 total abundance와 reheating/Friedmann normalization을 유도하는
것이다.

## 14. 1차 문헌

아래는 원 논문 또는 공식 preprint만 사용했다. 모든 URL의 접근일은
2026-08-15이다.

1. A. H. Taub, “General Relativistic Variational Principle for Perfect
   Fluids,” *Phys. Rev.* **94**, 1468 (1954).
   DOI: [10.1103/PhysRev.94.1468](https://doi.org/10.1103/PhysRev.94.1468).
2. B. F. Schutz, “Perfect Fluids in General Relativity: Velocity Potentials
   and a Variational Principle,” *Phys. Rev. D* **2**, 2762 (1970).
   DOI: [10.1103/PhysRevD.2.2762](https://doi.org/10.1103/PhysRevD.2.2762).
3. J. D. Brown, “Action functionals for relativistic perfect fluids,”
   *Class. Quantum Grav.* **10**, 1579 (1993).
   [arXiv:gr-qc/9304026](https://arxiv.org/abs/gr-qc/9304026),
   DOI: [10.1088/0264-9381/10/8/017](https://doi.org/10.1088/0264-9381/10/8/017).
4. S. Dubovsky, L. Hui, A. Nicolis, D. T. Son, “Effective field theory for
   hydrodynamics: thermodynamics, and the derivative expansion,”
   *Phys. Rev. D* **85**, 085029 (2012).
   [arXiv:1107.0731](https://arxiv.org/abs/1107.0731),
   DOI: [10.1103/PhysRevD.85.085029](https://doi.org/10.1103/PhysRevD.85.085029).
5. A. Pourtsidou, C. Skordis, E. J. Copeland, “Models of coupled dark matter
   to dark energy,” *Phys. Rev. D* **88**, 083505 (2013).
   [arXiv:1307.0458](https://arxiv.org/abs/1307.0458),
   DOI: [10.1103/PhysRevD.88.083505](https://doi.org/10.1103/PhysRevD.88.083505).
6. N. Andersson, G. L. Comer, “A covariant action principle for dissipative
   fluid dynamics: From formalism to fundamental physics,”
   *Class. Quantum Grav.* **32**, 075008 (2015).
   [arXiv:1306.3345](https://arxiv.org/abs/1306.3345),
   DOI: [10.1088/0264-9381/32/7/075008](https://doi.org/10.1088/0264-9381/32/7/075008).
7. T. S. Koivisto, E. N. Saridakis, N. Tamanini, “Scalar-Fluid theories:
   cosmological perturbations and large-scale structure,” *JCAP* **09**
   (2015) 047. [arXiv:1505.07556](https://arxiv.org/abs/1505.07556),
   DOI: [10.1088/1475-7516/2015/09/047](https://doi.org/10.1088/1475-7516/2015/09/047).
8. D. Iosifidis, E. Jensko, T. S. Koivisto, “Relativistic interacting fluids
   in cosmology,” *JCAP* **11** (2024) 043.
   [arXiv:2406.01412](https://arxiv.org/abs/2406.01412),
   DOI: [10.1088/1475-7516/2024/11/043](https://doi.org/10.1088/1475-7516/2024/11/043).
9. M. Doi, “Second quantization representation for classical many-particle
   system,” *J. Phys. A* **9**, 1465 (1976).
   DOI: [10.1088/0305-4470/9/9/008](https://doi.org/10.1088/0305-4470/9/9/008).
10. L. Peliti, “Path integral approach to birth-death processes on a lattice,”
    *J. Physique* **46**, 1469 (1985).
    DOI: [10.1051/jphys:019850046090146900](https://doi.org/10.1051/jphys:019850046090146900).
11. P. C. Martin, E. D. Siggia, H. A. Rose, “Statistical Dynamics of
    Classical Systems,” *Phys. Rev. A* **8**, 423 (1973).
    DOI: [10.1103/PhysRevA.8.423](https://doi.org/10.1103/PhysRevA.8.423).
12. Planck Collaboration, “Planck 2018 results. VI. Cosmological parameters,”
    *Astron. Astrophys.* **641**, A6 (2020).
    [arXiv:1807.06209](https://arxiv.org/abs/1807.06209),
    DOI: [10.1051/0004-6361/201833910](https://doi.org/10.1051/0004-6361/201833910).
