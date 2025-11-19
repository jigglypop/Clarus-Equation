## 30. SFE 고차원 및 끈이론 연계 스케치 (초안)

### 30.1 목표와 범위

이 장의 목적은, 지금까지 4차원에서 구축한 SFE 코어(억압장 $\Phi$), cold DM 섹터 $X$, 뮤온 다크섹터 $Z'$가 **더 높은 차원 혹은 끈 이론적 틀 안에서 어떤 구조로 통합될 수 있는지**를 개략적으로 정리하는 것이다.

중요한 점:

- 여기서 제시하는 내용은 **완성된 모형이 아니라 연구 방향 스케치**이다.
- 4차원 SFE에서 이미 고정된 파라미터(예: $\epsilon$, $m_\phi$, $\lambda$, $\Omega_\Phi^{\text{theory}}$, $\mu(a,k)$, $m_X$, $m_{Z'}$, $g_\mu$ 등)가 상위 이론의 파라미터(5차원 결합 상수, 콤팩트화 길이, 플럭스 등)와 어떤 관계를 맺을 수 있는지 방향만 제시한다.
- 어떤 상위 구조가 실제 우주인지 결정하기 위한 충분한 데이터는 아직 없으므로, **특정 모델을 채택하지 않고 가능한 구조들을 나열**하는 데에 그친다.

---

### 30.2 5차원 게이지장에서 억압장 $\Phi$와 뮤온 보손 $Z'$ 동시 생성

가장 직관적인 고차원 통합 그림은, 5차원 게이지장 하나로부터 4차원에서

- 벡터: 뮤온 다크 보손 $Z'_\mu$,
- 스칼라: 억압장 $\Phi$

를 동시에 얻는 구조이다.

#### 30.2.1 5차원 장과 콤팩트화

5차원 좌표를 $x^M=(x^\mu, y)$ ($\mu=0,1,2,3$)로 두고, 하나의 5D 게이지장 $\mathcal{A}_M(x^\mu,y)$를 도입한다. 5차원 작용은

$$
S_5 = \int d^4x \int_0^{2\pi R} dy
\left[ -\frac{1}{4g_5^2} \mathcal{F}_{MN}\mathcal{F}^{MN} \right],
$$

와 같은 형태를 갖는다. 여기서 $\mathcal{F}_{MN}=\partial_M\mathcal{A}_N-\partial_N\mathcal{A}_M$, $R$은 5번째 차원의 콤팩트화 반지름이다.

이제 $S^1$ 또는 $S^1/\mathbb{Z}_2$ 콤팩트화를 통해 4차원 유효 이론을 얻는다. $\mathcal{A}_M$를 4D 벡터와 스칼라로 분해하면

$$
\mathcal{A}_M = (\mathcal{A}_\mu, \mathcal{A}_5)
$$

이고, 콤팩트화 후 4D에서

- $\mathcal{A}_\mu^{(0)}(x)$: 질량이 없는 4D 게이지장(후에 $Z'_\mu$로 해석 가능),
- $\mathcal{A}_5^{(0)}(x)$: 질량이 없는 4D 스칼라(후에 억압장 $\Phi$ 후보)

로 나타난다. 콤팩트화와 대칭성 파괴(예: orbifold 경계조건, Wilson line 효과)를 통해

- $\mathcal{A}_\mu^{(0)}$는 작은 질량 $m_{Z'}$를 얻고,
- $\mathcal{A}_5^{(0)}$는 포텐셜 $V(\Phi)$를 갖으며 억압장 역할을 할 수 있다.

이 그림에서

$$
Z'_\mu \sim \mathcal{A}_\mu^{(0)}, \qquad \Phi \sim \mathcal{A}_5^{(0)}
$$

와 같이 식별할 수 있다.

#### 30.2.2 4D 결합 상수와 5D 파라미터의 관계

콤팩트화 후 4D 유효 게이지 결합 상수 $g_4$는 5D 결합 $g_5$와 콤팩트화 길이 $2\pi R$로부터

$$
g_4^2 \sim \frac{g_5^2}{2\pi R}
$$

과 같은 관계를 가진다(정확한 계수는 normalization에 따라 달라진다). 마찬가지로, $\Phi\sim \mathcal{A}_5^{(0)}$의 정규화와 self-interaction 포텐셜은 5D 작용과 콤팩트화 방식에 의해 결정된다.

단순한 예로, 4D에서 얻어지는 $Z'$의 질량과 억압장 질량이

$$
m_{Z'}^2 \sim \frac{k_1}{R^2} + \cdots,\qquad
m_\phi^2 \sim \frac{k_2}{R^2} + \cdots,
$$

형태를 가진다고 하면, 4D에서 이미 추정된 $m_{Z'}, m_\phi$는 5D 콤팩트화 반지름 $R$과 5D 파라미터 $k_1,k_2$에 제약을 건다.

특히 억압장 질량 $m_\phi \sim \hbar/(\lambda_H c)\sim 10^{-33}\,\text{eV}$ 수준의 극도로 작은 값은, 5D에서

$$
\frac{1}{R} \sim H_0 \sim 10^{-33}\,\text{eV}
$$

스케일의 초대형 콤팩트화 반지름(허블 스케일에 가까운) 또는 매우 얇은 효과적 포텐셜을 요구한다. 이는 일반적인 Planck 스케일 콤팩트화와는 매우 다르며, 고차원 모형에서의 큰 도전 과제이다.

#### 30.2.3 장점과 과제

- 장점:
  - $\Phi$와 $Z'$를 상위 차원에서 **하나의 장의 두 성분**으로 이해할 수 있어, 암흑에너지 다크섹터와 뮤온 다크섹터 사이의 관계를 자연스럽게 설명할 수 있다.
  - 4D에서 독립적으로 도입한 파라미터들($m_\phi, m_{Z'}, g_B, g_\mu$)이 5D 파라미터($g_5$, $R$, 경계조건 등)로부터 파생되므로, 4D에서의 자유도가 줄어들 수 있다.

- 과제:
  - $m_\phi \ll m_{Z'}$와 같은 거대한 질량 계층 구조를 5D에서 자연스럽게 설명해야 한다.
  - 콤팩트화 길이 $R$이 지나치게 크면(허블 스케일) 고차원 중력/게이지 효과가 다른 관측(예: Solar System, CMB)과 충돌할 수 있다.
  - 5D 모형에 표준모형, DM 입자 $X$까지 함께 넣을 때 anomaly cancellation, 안정성 등을 모두 만족해야 한다.

---

### 30.3 5차원 스칼라–중력 모형에서 억압장과 DM 동시 생성

또 다른 접근은, 5차원 스칼라–중력 모형에서 억압장과 DM을 동시에 유도하는 것이다.

#### 30.3.1 5D 스칼라장과 4D 유효 포텐셜

5D에서 스칼라장 $\Phi_5(x^\mu,y)$와 중력을 고려한 작용

$$
S_5 = \int d^4x \int dy \sqrt{-G}
\left[ \frac{1}{2\kappa_5^2}R_5 + \frac{1}{2} G^{MN}\partial_M\Phi_5\partial_N\Phi_5 - V_5(\Phi_5) \right]
$$

을 생각할 수 있다. 적절한 배경 해(예: domain-wall/brane 해)를 가진 뒤, $y$ 방향으로 Kaluza–Klein 분해를 하면

$$
\Phi_5(x,y) = \sum_n \phi_n(x) f_n(y)
$$

형태의 4D 모드 $\phi_n(x)$들을 얻게 된다.

- 가장 낮은 모드 $\phi_0(x)$는 4D에서 억압장 $\Phi$로 동정할 수 있다.
- 더 무거운 모드들 $\phi_{n>0}(x)$ 중 일부는 DM 후보 $X$와 연결될 수 있다.

이 경우, $\Phi$ 섹터와 DM 섹터 X는 5D에서 동일한 스칼라의 서로 다른 모드로부터 나오는 셈이며, 4D에서의 포텐셜 파라미터($\mu, \lambda_X$ 등)는 5D 포텐셜 $V_5(\Phi_5)$와 기하학적 구조에 의해 연동된다.

#### 30.3.2 4D에서의 제약과 상위 이론

4D에서 이미 고정된 데이터(예: $m_\phi\sim 10^{-33}\,\text{eV}$, $m_X\gtrsim 10^{-22}\,\text{eV}$, $\lambda_X$에 대한 부등식)는 5D 모형에 대해 다음과 같은 제약을 건다.

- $\phi_0$ 모드는 매우 가벼운 억압장, $\phi_{n>0}$ 모드는 DM으로 해석 가능할 정도의 질량 스펙트럼이 필요하다.
- 5D 포텐셜 $V_5$와 기하학에 따라 KK 모드 간 질량 간격이 결정되므로, $m_\phi$와 $m_X$ 사이의 큰 격차를 자연스럽게 만드는 구조가 필요하다.

이는 일반적인 KK 모형에서 보기 드문 스펙트럼이며, 상위 이론(예: supersymmetry, warped geometry, string compactification)에서의 보호 메커니즘이 필요할 수 있다.

---

### 30.4 끈 이론 관점에서의 후보 구조들

끈 이론 compactification에서는 다음과 같은 객체들이 자연스럽게 등장한다.

- **모듈러스 모드(moduli)**: 내부 공간의 부피, 형태를 나타내는 스칼라장 → 억압장 $\Phi$ 후보.
- **딜라톤(dilaton)**: 끈 결합 상수를 제어하는 스칼라장 → $\alpha$, $\epsilon$과 연관될 수 있는 후보.
- **추가 $U(1)$ 게이지 보손**: 흔히 등장하는 extra gauge boson → 뮤온 다크섹터 Z' 또는 DM와 연관된 게이지장 후보.
- **brane-localized fields**: 특정 brane에 국한된 표준모형/뮤온/DM 등.

가능한 시나리오 예:

1. 내부 공간의 특정 모듈러스 스칼라가 SFE 억압장 $\Phi$로 작동하며, 그 포텐셜은 flux compactification에 의해 결정된다. 이로부터 암흑에너지 스케일과 질량 억압 파라미터 $\epsilon$이 유도된다.
2. 또 다른 모듈러스 또는 extra $U(1)$ gauge boson이 뮤온 섹터에 비보편적으로 결합해 Z' 역할을 한다.
3. DM 입자 X는 brane-localized field 또는 closed string mode로 나타나며, 그 질량과 self-interaction은 compactification 데이터에 의해 결정된다.

이 경우, 4D에서 이미 고정된

- $\Omega_\Phi^{\text{theory}}\approx\Omega_\Lambda$,
- $m_\phi$, $\lambda$,
- $\Omega_X$, $m_X$, $\sigma/m_X$,
- $m_{Z'}$, $g_\mu$

등은 모두 compactification 데이터(플럭스, brane 구성, 내부 공간의 토폴로지 등)에 대한 복잡한 제약으로 변환된다. 이를 완전히 풀어내는 것은 현재로서는 거대한 연구 과제이다.

---

### 30.5 SFE 데이터가 상위 이론에 주는 제약 요약

상위 차원/끈 이론 모형이 실제로 SFE 코어 + X + Z' 구조를 재현하려면, 다음과 같은 4D 데이터들을 동시에 만족해야 한다.

- **억압장 섹터**
  - $m_\phi \sim 10^{-33}\,\text{eV}$ 수준의 극도로 작은 질량.
  - $V(\Phi)$ 포텐셜이 $\Omega_\Phi^{\text{theory}}\approx\Omega_\Lambda$를 재현.
  - $\epsilon=2\Omega_\Lambda-1$를 통해 미시–거시 연결 및 결맞음/LIGO/뮤온 수명 예측과 양립.

- **DM 섹터 X**
  - 잔여 DM 분율 $0.05\lesssim\Omega_X\lesssim0.2$.
  - $m_X\gtrsim 10^{-22}\,\text{eV}$, $0.01\lesssim\sigma/m_X\lesssim1\,\text{cm}^2/\text{g}$.

- **뮤온 섹터 Z'**
  - $g_\mu/m_{Z'}\sim\text{몇}\times10^{-3}\,\text{GeV}^{-1}$ 범위에서 g-2를 설명.
  - 중성미자 trident·콜라이더·BBN/CMB 제약과 동시에 양립.

상위 이론에서 이 모든 조건을 자연스럽게 만족하는 compactification/5D 구조는 강한 제약을 받게 되며, 반대로 SFE 데이터는 상위 이론의 파라미터 공간을 크게 줄이는 역할을 한다.

---

### 30.6 향후 연구 과제

이 장에서 제시한 내용은 고차원/끈 이론과 SFE를 연결하는 **개략적인 스케치**에 불과하다. 구체적인 연구 과제는 다음과 같다.

1. **구체 5D 모형 구성**
   - 하나의 5D 게이지장 또는 스칼라장에서 $\Phi$, $Z'$ (및 일부 DM 모드)를 동시에 얻는 모형을 명시적으로 작성.
   - 콤팩트화, 경계조건, 포텐셜을 포함한 전체 액션을 기술하고, 4D 유효 이론을 도출.

2. **4D 데이터 매칭**
   - 위 모형이 4D에서 SFE 코어 + X + Z'가 요구하는 파라미터(특히 $m_\phi$, $\Omega_\Phi$, $m_X$, $m_{Z'}$, $g_\mu$ 등)를 얼마나 자연스럽게 재현하는지 분석.
   - 필요시 supersymmetry, warped geometry, sequestering 등 추가 메커니즘 도입 검토.

3. **끈 compactification 예제 탐색**
   - 간단한 string compactification 예제에서 모듈러스/딜라톤/extra $U(1)$을 SFE의 $\Phi$ 및 Z'와 연결할 수 있는지 탐색.
   - swampland 조건과의 양립 여부, 안정화 문제 검토.

4. **상위–하위 이론 간 파라미터 사전(dictionary) 정리**
   - 상위 이론 파라미터(플럭스, brane 데이터 등) ↔ 4D SFE 파라미터(ε, $m_\phi$, $m_X$, $m_{Z'}$, $g_\mu$ 등) 사이의 관계를 가능한 한 명시적인 수식으로 정리.

이러한 작업들이 진행되면, SFE는 단순한 4D 유효 이론을 넘어, 더 큰 고차원/끈 이론 구조 속에서 **어떤 위치를 차지하는지**를 보다 명확하게 이해할 수 있을 것이다.

---

### 30.7 초끈 이론 융합이 SFE 오차에 미치는 영향: 식 수준의 정리

이제 **초끈/고차원 융합 가설이 실제로 4D SFE 예측식에 어떤 보정 항을 추가하는지**를, 가능한 한 식 형태로 정리한다. 전제는 다음과 같다.

- 4D SFE 코어에서 사용한 기본 작용은

$$
S_{\text{SFE}} = \int d^4x \sqrt{-g}\,\Big[
\frac{1}{2\kappa^2}R + \mathcal{L}_{\text{SM}}
+ \mathcal{L}_\Phi + \mathcal{L}_X + \mathcal{L}_{Z'}
\Big]
$$

형태이며, 여기서

$$
\mathcal{L}_\Phi = \frac12(\partial\Phi)^2 - V(\Phi) - g_B\Phi\,T^\mu{}_\mu,
$$

$$
\mathcal{L}_X = \frac12(\partial X)^2 - \frac12 m_X^2 X^2 - \frac{\lambda_X}{4}X^4,
$$

같은 꼴(기존 장들에서 사용한 형태)을 가진다고 둔다. 초끈/고차원 융합은 이 위에 **고차원 연산자 보정**을 추가하는 방식으로 들어온다고 가정한다.

#### 30.7.1 고차원 연산자에 의한 일반 보정식

상위 이론(초끈/5D)이 4D에 유도하는 가장 일반적인 효과는

$$
\Delta\mathcal{L}_{\text{eff}} = \sum_n \frac{c_n}{M_{\text{UV}}^{n-4}}\mathcal{O}_n
$$

꼴의 고차원 연산자이다. 여기서

- $M_{\text{UV}}$: KK 질량, string scale 또는 compactification 스케일
- $\mathcal{O}_n$: 차원 $n>4$인 로컬 연산자들

이다. SFE와 직접적으로 연결되는 최소한의 연산자들만 골라 보면

1. 억압장 포텐셜에 대한 보정:

$$
\Delta\mathcal{L}_\Phi =
\frac{\alpha_6}{M_{\text{UV}}^2}\Phi^6
+ \frac{\alpha_{R\Phi}}{M_{\text{UV}}^2}R\,\Phi^2
$$

2. DM self-interaction 보정:

$$
\Delta\mathcal{L}_X =
\frac{\beta_6}{M_{\text{UV}}^2}X^6
$$

3. 억압장–DM 혼합 보정:

$$
\Delta\mathcal{L}_{\Phi X} =
\frac{\gamma}{M_{\text{UV}}^2}\Phi^2 X^2
$$

4. 뮤온–Z' 결합에 대한 보정(예: brane 위치에 따른 비보편성):

$$
\Delta\mathcal{L}_{Z'\mu} =
\frac{\delta}{M_{\text{UV}}^2}(\bar\mu\gamma^\mu\mu)(H^\dagger H)Z'_\mu
$$

형태를 생각할 수 있다.

이 연산자들로 인해 4D 유효 파라미터들은

$$
m_\phi^2 \to m_\phi^2 + \delta m_\phi^2,\quad
\lambda \to \lambda + \delta\lambda,\quad
m_X^2 \to m_X^2 + \delta m_X^2,\quad
\lambda_X \to \lambda_X + \delta\lambda_X,
$$

$$
g_\mu \to g_\mu + \delta g_\mu,
$$

처럼 이동한다. 이때 $\delta m_\phi^2,\delta\lambda,\dots$는 모두

$$
\delta m_\phi^2 \sim \mathcal{O}\!\left(\frac{\alpha_{R\Phi} R}{M_{\text{UV}}^2}\right),\quad
\delta\lambda \sim \mathcal{O}\!\left(\frac{\alpha_6\langle\Phi^2\rangle}{M_{\text{UV}}^2}\right),
$$

$$
\delta m_X^2 \sim \mathcal{O}\!\left(\frac{\gamma\langle\Phi^2\rangle}{M_{\text{UV}}^2}\right),\quad
\delta\lambda_X \sim \mathcal{O}\!\left(\frac{\beta_6\langle X^2\rangle}{M_{\text{UV}}^2}\right),
$$

같은 규모로 억제된다.

이 보정들을 SFE에서 사용하는 관측량으로 옮겨 적으면,

$$
\epsilon \equiv \epsilon(m_\phi,\lambda,\dots)
\;\to\;
\epsilon + \delta\epsilon,
$$

$$
\mu(a,k) \equiv \mu(a,k;\lambda_{\text{kernel}},\dots)
\;\to\;
\mu(a,k) + \delta\mu(a,k),
$$

$$
x(\mu) = \frac{\mu-1}{\mu f_c}
\;\to\;
x(\mu+\delta\mu) \simeq x(\mu) + \frac{\partial x}{\partial\mu}\delta\mu,
$$

$$
\Delta a_\mu^{Z'} \simeq \frac{g_\mu^2}{12\pi^2}\frac{m_\mu^2}{m_{Z'}^2}
\;\to\;
\frac{(g_\mu+\delta g_\mu)^2}{12\pi^2}\frac{m_\mu^2}{m_{Z'}^2}
\simeq
\Delta a_\mu^{Z'} + \frac{2g_\mu\,\delta g_\mu}{12\pi^2}\frac{m_\mu^2}{m_{Z'}^2},
$$

과 같은 1차 보정식을 얻는다.

#### 30.7.2 오차 감소 조건을 식으로 쓰면

각 검증 항목의 오차가

$$
\Delta O_i = O_i^{\text{theory}} - O_i^{\text{obs}}
$$

로 정의되어 있을 때, 초끈 융합에 의해 보정된 값은

$$
\Delta O_i' = (O_i^{\text{theory}}+\delta O_i) - O_i^{\text{obs}}
= \Delta O_i + \delta O_i
$$

이므로, **오차 감소 조건**은 단순히

$$
|\Delta O_i'| < |\Delta O_i|
\quad\Longleftrightarrow\quad
|\Delta O_i + \delta O_i| < |\Delta O_i|
$$

이다. 위에서 얻은 보정식을 대입하면, 예를 들어

- 암흑에너지/ε 관련 항목에 대해서는

$$
|\Delta\epsilon + \delta\epsilon| < |\Delta\epsilon|,
$$

- 성장률/군집 분포에 대해서는

$$
|\Delta\mu(a,k) + \delta\mu(a,k)| < |\Delta\mu(a,k)|,
$$

- 뮤온 g-2에 대해서는

$$
|\Delta a_\mu^{\text{theory}} + \delta(\Delta a_\mu)| < |\Delta a_\mu^{\text{theory}}|,
$$

이 각각 성립해야 한다.

그러나 중요한 점은 다음과 같다.

1. $\delta\epsilon,\delta\mu,\delta g_\mu$ 등은 모두 **상위 이론 계수들** $(\alpha_6,\alpha_{R\Phi},\beta_6,\gamma,\delta,\dots)$와 $M_{\text{UV}}$에 의해 결정되는데,
2. 현재 단계에서는 이 계수들에 대해 **아무런 독립적인 고정식이 없다.**  
   (즉, $\alpha_6,\alpha_{R\Phi},\dots$ 값을 논리적으로 계산해낼 수 있는 완성된 초끈 모형이 아직 없다.)

따라서

$$
\delta\epsilon,\ \delta\mu(a,k),\ \delta g_\mu
$$

의 **부호와 크기를 계산을 통해 연역적으로 정할 수 없고**, 위의 오차 감소 부등식들을 실제로 만족하는지 여부를 **식 수준에서 판정할 수 없다.**

즉, 식으로 써 보면,

$$
|\Delta O_i + F_i(\alpha_n,M_{\text{UV}})| < |\Delta O_i|
$$

꼴의 부등식인데, 현재는 $F_i$의 부호/크기를 모른다. 이 상태에서 “오차가 줄었다”고 말하면, **그 자체가 파라미터 튜닝 가정**이 되어 논문 규칙(연역적 검증, 튜닝 금지)에 위배된다.

#### 30.7.3 요약: 식을 완성해도, 계수를 모르면 오차 판정 불가

정리하면,

- 우리는 초끈/고차원 융합이 4D SFE에 주는 영향을
  - 고차원 연산자
  - 유효 파라미터 이동($\delta m_\phi^2,\delta\lambda,\dots$)
  - 관측량 보정식($\delta\epsilon,\delta\mu,\delta a_\mu$)

  의 형태로 **식까지는 명시적으로 쓸 수 있다.**
- 그러나 이 식들 속에 들어가는 계수들 $(\alpha_n,\beta_n,\gamma,\delta,\dots,M_{\text{UV}})$에 대한 **상위 이론의 연역적 결정식이 없기 때문에**,  
  - $|\Delta O_i'|<|\Delta O_i|$를 만족하는지 **연역적으로 판정할 수 없고**,  
  - 오차가 줄었다고 말하는 것은 곧바로 “계수를 적당히 골라서 맞춘다”는 의미가 되어 **파라미터 튜닝 금지 규칙에 위배**된다.

따라서, 사용자의 기준(“새 가설이면 먼저 오차가 줄어드는지부터, 튜닝 없이 연역적으로 판단하라”)을 엄밀히 적용하면,

- 현재 수준의 초끈/고차원 융합 가설은
  - 식은 쓸 수 있으나,
  - 계수를 모르는 탓에 오차 감소 여부를 판정할 수 없고,
  - 따라서 **SFE 본편의 오차를 줄이는 가설로 채택할 수 없다.**

결론적으로, **지금 단계의 초끈 융합은 “식까지 완성한 연구 아이디어”이지, 오차를 실제로 줄였다고 선언할 수 있는 이론은 아니다.**  
이 상태에서 논문 본문에 “오차가 줄었다”고 쓰는 것은, 사용자가 금지한 논리적 순환·튜닝에 해당하므로 할 수 없다.

---

### 30.8 최소 파라미터 선택: 우리 파라미터/상수만으로 보정 크기 추정

이제 사용자가 요구한 대로, **“우리 파라미터를 최소한으로만 사용”**하는 경우를 하나 명시적으로 잡고, 실제로 오차가 줄 수 있는지 **수치 스케일까지** 추정한다. 새로운 자유 파라미터는 도입하지 않고,

- 기본 상수: $c, \hbar, G$ (또는 $M_{\text{Pl}}$)
- 우주론 상수: $H_0, \Omega_\Lambda$
- SFE에서 이미 고정한 파라미터: $\epsilon, m_\phi, \lambda, m_X, \lambda_X, m_{Z'}, g_\mu$

만 사용한다.

#### 30.8.1 단 하나의 연산자만 선택: $R\Phi^2/M_{\text{Pl}}^2$

가장 단순하고 보수적인 선택으로, 초끈/고차원 효과가 4D에서 다음 **하나의 연산자**로만 나타난다고 가정한다.

$$
\Delta\mathcal{L}_\Phi = \frac{\alpha_{R\Phi}}{M_{\text{Pl}}^2}R\,\Phi^2,
$$

여기서

- $M_{\text{Pl}} \equiv (8\pi G)^{-1/2}$: 4D 플랑크 질량(새 파라미터가 아니라 이미 알려진 상수)
- $\alpha_{R\Phi}$: **무차원 계수, 자연성 가정으로 $\mathcal{O}(1)$** (예: $+1$ 또는 $-1$)

으로 둔다. 이때는 새로운 질량 스케일 $M_{\text{UV}}$도, 새로운 계수 함수도 도입하지 않는다.

이 연산자는 배경 FRW 우주에서

$$
R \sim \mathcal{O}(H_0^2)
$$

크기를 갖고, 억압장 배경값 $\Phi_{\text{bg}}$에 대해 유효 포텐셜에

$$
\Delta V_{\text{eff}}(\Phi_{\text{bg}}) \approx -\frac{\alpha_{R\Phi}}{M_{\text{Pl}}^2}R\,\Phi_{\text{bg}}^2
$$

을 추가한다(부호는 convention에 따라 바뀔 수 있으나, 여기서 중요한 것은 크기이다).

#### 30.8.2 $\delta\rho_\Lambda/\rho_\Lambda$를 우리 파라미터로만 계산

기존 SFE에서는

- 암흑에너지 밀도: $\rho_\Lambda \approx \Omega_\Lambda \rho_c c^2$
- 억압장 배경 에너지: $V(\Phi_{\text{bg}})\approx \rho_\Lambda$

과 같은 관계를 사용한다. $\Phi_{\text{bg}}$의 스케일을 추정하기 위해, 포텐셜이

$$
V(\Phi)\sim \frac12 m_\phi^2\Phi^2 + \frac{\lambda}{4}\Phi^4
$$

형태라고 두면, 최소 근처에서 대략

$$
V(\Phi_{\text{bg}})\sim \mathcal{O}(m_\phi^2\Phi_{\text{bg}}^2)
$$

이므로,

$$
\Phi_{\text{bg}}^2 \sim \frac{\rho_\Lambda}{m_\phi^2}
$$

규모를 가진다고 볼 수 있다(이 추정은 $m_\phi^2\Phi^2$ 항이 지배적이라는 가정일 뿐, 새로운 파라미터는 도입하지 않는다).

그러면 $\Delta V_{\text{eff}}$에 의한 암흑에너지 밀도 보정은

$$
\delta\rho_\Lambda \sim \frac{\alpha_{R\Phi}}{M_{\text{Pl}}^2}R\,\Phi_{\text{bg}}^2
\sim \frac{\alpha_{R\Phi}}{M_{\text{Pl}}^2}H_0^2\frac{\rho_\Lambda}{m_\phi^2}.
$$

상대 보정은

$$
\frac{\delta\rho_\Lambda}{\rho_\Lambda}
\sim \alpha_{R\Phi}\frac{H_0^2}{M_{\text{Pl}}^2}\frac{1}{m_\phi^2}.
$$

여기서 $m_\phi$는 이미 SFE에서 **허블 스케일과 같은 오더**로 고정되어 있다.

$$
m_\phi \sim \mathcal{O}(H_0)
$$

라고 두면,

$$
\frac{\delta\rho_\Lambda}{\rho_\Lambda}
\sim \alpha_{R\Phi}\frac{H_0^2}{M_{\text{Pl}}^2}\frac{1}{H_0^2}
= \alpha_{R\Phi}\frac{1}{M_{\text{Pl}}^2}.
$$

단위 일관성을 위해 $c=\hbar=1$ 단위를 쓰면, $M_{\text{Pl}}$는 질량 단위이고,

$$
\frac{1}{M_{\text{Pl}}^2} \sim 10^{-38}\,\text{GeV}^{-2}
$$

규모를 가진다. 상대 보정은

$$
\left|\frac{\delta\rho_\Lambda}{\rho_\Lambda}\right|
\sim |\alpha_{R\Phi}|\times 10^{-38}
$$

정도로, **관측 오차(수 퍼센트 수준)에 비해 사실상 0**이다.

이에 따라

$$
\epsilon \equiv 2\Omega_\Lambda-1
$$

에 대한 상대 보정도

$$
\left|\frac{\delta\epsilon}{\epsilon}\right|
\sim \left|\frac{\delta\rho_\Lambda}{\rho_\Lambda}\right|
\sim 10^{-38}
$$

규모로 억제된다. 이것은 이미 1–2% 수준으로 맞춰 놓은 $\epsilon$ 관련 검증(결맞음, LIGO, 뮤온 생존율)에 **아무 영향도 주지 못한다.**

#### 30.8.3 성장률/DM, 뮤온 g-2에 대한 보정 크기

- 성장률/DM:
  - $\mu(a,k)$, $x(\mu)$는 커널과 DM 파라미터($m_X,\lambda_X$)에 의해 결정된다.
  - 위와 같은 Planck-억제 연산자 하나만 켜면, $\Phi$와 $X$의 유효 질량/결합에 들어가는 상대 보정도 대략

    $$
    \frac{\delta m_\phi^2}{m_\phi^2},\ \frac{\delta m_X^2}{m_X^2},\ \frac{\delta\lambda_X}{\lambda_X}
    \sim \mathcal{O}\!\left(\frac{H_0^2}{M_{\text{Pl}}^2}\right)\ll 10^{-30}
    $$

    수준으로 작다(여기서도 $M_{\text{Pl}}$만 사용).
  - 따라서 $\delta\mu(a,k)$, $\delta x(\mu)$는 **관측 가능한 수준보다 훨씬 작아서**, 성장률/DM 관련 tension을 줄이지 못한다.

- 뮤온 g-2:
  - Planck-억제 뮤온 연산자 하나(예: $\frac{1}{M_{\text{Pl}}^2}(\bar\mu\gamma^\mu\mu)(H^\dagger H)Z'_\mu$)를 켰다고 해도,

    $$
    \delta g_\mu \sim \mathcal{O}\!\left(\frac{v^2}{M_{\text{Pl}}^2}\right) \sim 10^{-34}
    $$

    정도에 불과하다($v\approx 246\,\text{GeV}$는 힉스 진공기댓값).
  - 뮤온 g-2에 미치는 상대 보정은

    $$
    \frac{\delta(\Delta a_\mu)}{\Delta a_\mu} \sim \frac{2\,\delta g_\mu}{g_\mu}
    \sim 10^{-34}
    $$

    수준으로, $\Delta a_\mu\sim10^{-9}$에 비하면 **완전히 무시 가능한 크기**이다.

즉, **우리 파라미터/상수만 사용해서** Planck-억제형 연산자 하나를 켠 경우,

- $\epsilon$, $\Omega_\Lambda$, 성장률, DM, 뮤온 g-2 어느 쪽에서도 **관측 가능한 오차 감소는 일어나지 않는다.**

#### 30.8.4 요약: 최소 파라미터로 해 보면, 보정은 “존재하지만 관측 불가 수준”

사용자가 요구한 대로,

- 새로운 임의 파라미터를 도입하지 않고,
- $c,\hbar,G,H_0,\Omega_\Lambda$와 이미 SFE에서 고정한 파라미터들만으로
- 가장 단순한 고차원 연산자 하나($R\Phi^2/M_{\text{Pl}}^2$)를 켜서
- $\delta\rho_\Lambda/\rho_\Lambda$, $\delta\epsilon/\epsilon$, 성장률/DM, 뮤온 g-2 보정을 **식과 스케일**로 계산해 보았다.

그 결과,

- 상대 보정 크기들이 모두 $10^{-30}$–$10^{-38}$ 수준 이하로 억제되어,
- 현재 남아 있는 SFE 오차(수 퍼센트–$10^{-3}$ 수준)를 **눈에 띄게 줄이지 못한다.**

따라서 **“우리 파라미터를 진짜 최소한으로만 쓰는” 초끈/고차원 융합**은,

- 수학적으로는 거짓은 아니고,
- Planck-억제 작은 보정으로서 EFT 안에 존재할 수 있지만,
- SFE 검증 항목들의 오차를 **의미 있게 줄이는 메커니즘으로는 작동하지 않는다**

는 결론이 나온다.
