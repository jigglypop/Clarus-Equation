# Reality_Stone--CE 조건부 수학 인터페이스

이 문서는 Reality Stone이라는 물리적 이름을 경사 흐름·다양체 갱신·무차원 Gibbs 재가중·잔류 pushforward의 수학 interface와 구분한다. 닫힌 것은 명시한 metric/measurable 구조의 계산 정의이며, 물리 개념·구현 비유·검증 지위는 별도 층으로 유지한다.

독자는 경로공간 bridge와 residual readout 문서를 먼저 읽는다. 경사 흐름, 이산 갱신, Gibbs·pushforward를 거쳐 허용되는 무차원 사상과 금지된 물리 동일시·관련 문서 순서로 읽는다.

이 문서는 리만 최적화와 CE의 무차원 재가중 문법 사이에서 실제로 정의되는
부분만 기록한다. 학습 알고리즘과 물리 법칙의 동일시, 양자 측정, 중력원 또는
우주론적 에너지의 식별은 여기서 따라오지 않는다.

## 1. 리만 경사 흐름

리만 경사 흐름은 manifold, metric, differentiable functional의 정의역에서 쓰는 수학 모델이다. Reality Stone의 물리적 힘·시간 진화로 읽으려면 action, unit, causality와 관측 contract가 필요하다.

**[정의]** $(\mathcal M,g)$를 유한차원 Riemannian manifold,
$F\in C^1(\mathcal M;\mathbb R)$를 목적함수라 한다. Riemannian
gradient는 모든 $v\in T_x\mathcal M$에 대해

$$
g_x(\operatorname{grad}_gF(x),v)=dF_x(v)
$$

를 만족하는 유일한 tangent vector다.

**[정리]** 미분가능 곡선 $x(t)$가 존재 구간에서

$$
\dot x(t)=-\operatorname{grad}_gF(x(t))
$$

를 만족하면

$$
\frac{d}{dt}F(x(t))
=-\|\operatorname{grad}_gF(x(t))\|_g^2\leq0.
$$

**증명.** chain rule과 gradient의 정의를 적용한다.
$$
\frac{d}{dt}F(x(t))
=dF_{x(t)}(\dot x(t))
=-\|\operatorname{grad}_gF(x(t))\|_g^2.
\quad\square
$$

이 정리는 선택한 $g,F$의 흐름에 대한 감소 정리다. 학습된 metric이
데이터의 “참 곡률”이거나 $F$가 물리적 작용이라는 결론은 포함하지 않는다.

## 2. 다양체 위의 이산 갱신

이산 갱신은 step size, retraction/connection, state shape를 갖는 구현 규칙이다. 수렴은 curvature·Lipschitz·step schedule 같은 가정에 의존하며 코드 실행만으로 보장되지 않는다.

**[정의]** $x_n\in\mathcal M$, $v_n\in T_{x_n}\mathcal M$,
$\eta_n,\alpha_n\in\mathbb R$라 하고, 아래 tangent vector가
$\exp_{x_n}$의 정의역 안에 있다고 하자.

$$
x_{n+1}
=
\exp_{x_n}\!\left[
-\eta_n\bigl(\operatorname{grad}_gF(x_n)+\alpha_nv_n\bigr)
\right].
$$

이는 좌표와 무관한 update다. $v_n$을 residual, momentum 또는 외부
제어로 읽는 것은 알고리즘 선택이다.

**[정의]** $\sigma:\mathcal M\to\mathbb R$가 무차원이면 양의 gate

$$
w(x):=e^{-\sigma(x)}
$$

를 정의할 수 있다. 이를 step 크기에 넣은

$$
x_{n+1}
=
\exp_{x_n}\!\left[-\eta_nw(x_n)\operatorname{grad}_gF(x_n)\right]
$$

도 tangent-space update로 잘 정의된다. $w(x)$를 다양체의 점
$\exp_x(\cdots)$ 자체에 곱하는 연산은 일반 다양체에는 정의되지 않는다.

**[경험식]** 실제 Reality_Stone 계산에서 $g$, $F$, $v_n$,
$\eta_n$과 $\sigma$를 데이터로부터 정하는 규칙은 학습 모형이다.
그 일반화 성능은 독립 자료와 사전 고정한 평가량으로 검증해야 한다.

## 3. 무차원 Gibbs 재가중

Gibbs 재가중의 지수는 무차원 energy와 명시한 temperature를 필요로 한다. 확률 weight를 물리 amplitude 또는 에너지 보존 법칙으로 동일시하지 않는다.

**[정의]** $X$를 compact metric space, $\mu_0$를 $X$의 모든
비어 있지 않은 열린집합에 양의 질량을 주는 확률측도,
$\mathcal I:X\to\mathbb R$를 연속인 무차원 cost라 한다. $\beta>0$도
무차원 매개변수라 두고

$$
\mu_\beta(dx)
=
\frac{e^{-\beta\mathcal I(x)}}{Z_\beta}\,\mu_0(dx),
\qquad
Z_\beta=\int_Xe^{-\beta\mathcal I}\,d\mu_0
$$

로 정의한다. Compactness와 continuity 때문에 $0<Z_\beta<\infty$다.

**[정리]** $M=\operatorname*{argmin}_X\mathcal I$라 하면 모든 열린
이웃 $U\supset M$에 대해

$$
\mu_\beta(X\setminus U)\longrightarrow0
\qquad(\beta\to\infty).
$$

**증명.** $X\setminus U=\varnothing$이면 자명하다. 그렇지 않으면
$X\setminus U$가 compact이고 $M$과 만나지 않으므로
$$
\delta:=\min_{X\setminus U}\mathcal I-\min_X\mathcal I>0.
$$
연속성으로 어떤 비어 있지 않은 열린집합 $V\subset U$와
$0<\varepsilon<\delta$에 대해
$\mathcal I\leq\min_X\mathcal I+\varepsilon$ on $V$다.
Full support로 $\mu_0(V)>0$이고
$$
\mu_\beta(X\setminus U)
\leq
\frac{\mu_0(X\setminus U)}{\mu_0(V)}
e^{-\beta(\delta-\varepsilon)}
\to0.
\quad\square
$$

이 정리는 positive Gibbs measure의 최소점 농축이다. 복소 위상을 갖는
Lorentzian path integral, Born rule 또는 실제 측정 instrument를
유도하지 않는다.

## 4. 잔류량의 measurable pushforward

measurable pushforward는 finite/subprobability 측도와 integrable kernel을 출력 field로 보내는 정의다. kernel의 locality·covariance·identifiability가 없으면 독립 물리장·구현 feature의 의미를 결론낼 수 없다.

**[정의]** $Y$를 measurable space, $K:Y\times X\to\mathbb R$를
각 $y$에 대해 $\mu_\beta$-적분 가능한 measurable kernel이라 한다.
Measurable한 비선택 집합 $N\subset X$에 대해

$$
r_\beta(y)
:=
\int_NK(y,x)\,\mu_\beta(dx)
$$

로 정의한다.

**[정리]** $K$가 jointly measurable이고
$\int_N|K(y,x)|\,\mu_\beta(dx)<\infty$이면 $r_\beta(y)$는 정의된다.
또한 $|K(y,x)|\leq h(x)$인 공통 적분가능 지배함수가 있고
$K(y_j,x)\to K(y,x)$가 거의 모든 $x$에서 성립하면
$r_\beta(y_j)\to r_\beta(y)$다.

**증명.** 첫 명제는 Lebesgue 적분의 정의, 둘째는 dominated convergence
theorem이다. $\square$

이 적분은 측도의 요약량일 뿐 독립적인 시공간 장이나 stress tensor가
자동으로 생긴다는 뜻이 아니다. $Y$, $K$, normalization과 관측
연산자의 선택은 별도 자료다.

## 5. 물리적 사용의 정확한 경계

물리적 명명은 수학 interface의 정의역보다 넓은 해석이다. 다음 구분은 허용되는 무차원 대응과 정리가 아닌 모델 선택을 분리해 비유가 증거를 앞지르지 않게 한다.

### 5.1 허용되는 무차원 사상

허용되는 사상은 단위·normalization·입출력 contract를 보존하는 계산적 대응이다. 이 대응이 자연의 field ontology를 증명하는 것은 아니다.

**[공리: 유클리드 모형]** 실제로 정의된 Euclidean action $S_E$가 있고
기준 작용으로 $\hbar$를 택할 때만

$$
\mathcal I_E=\frac{S_E}{\hbar},
\qquad
e^{-\mathcal I_E}
$$

를 무차원 가중치로 사용할 수 있다. 통계역학에서는 같은 역할을
$E/(k_BT)$가 한다. 곡률을 지수에 넣으려면

$$
\widetilde R=\frac R{R_c}=RL_c^2
$$

처럼 먼저 무차원화한다.

### 5.2 정리가 아닌 물리 사상

물리 사상은 CE가 채택한 해석 또는 미완성 bridge로 표기한다. 반증 가능한 observable·baseline·external input이 없으면 경험 주장으로 승격하지 않는다.

다음 항목은 **[미완성]**이다.

- CE 경로공간의 topology, sigma-algebra와 countably additive prior
- 공변 작용, gauge fixing, boundary term과 renormalization
- Euclidean measure의 reflection positivity와 Lorentzian continuation
- 양자 상태에서 outcome 확률로 가는 CPTP instrument
- measurable residual $r_\beta$에서 국소 장과
  $T_{\mu\nu}$로 가는 metric variation
- 학습 metric, Ricci curvature와 물리 시공간 곡률 사이의 사상

이 자료가 없는 상태에서 AI의 soft selection을 자연의 측정 법칙으로,
optimizer의 residual을 에너지 밀도로, 또는 곡률 gate를 양자 decoherence로
옮기지 않는다.

## 6. 문서 연결

관련 문서는 수학 정의, CE bridge, 구현 검증, 물리 해석의 source role을 나눠 제공한다. 이 문서를 단독으로 읽어 다른 문서의 가정·지위를 생략하지 않는다.

- [형식적_수학_모델과_증명.md](형식적_수학_모델과_증명.md):
  toy ODE, 직접법과 Gibbs 농축
- [../검증_원장/참조_이론물리_보존_원장.md](../검증_원장/참조_이론물리_보존_원장.md):
  공변 Hessian, Ward identity, 스펙트럼·인과성·wormhole 정리
- [../9_등호이전/05_CE_브리지.md](../9_등호이전/05_CE_브리지.md):
  경로공간 후보를 쓰기 위한 조건부 브리지
