# 04c. PreEq 보편 스킴

## 0. 목표

이 문서는 다음 주장을 순수수학 명제로 분해한다.

> "등호 이전 구조는 등호가 등장하는 거의 모든 수학에 적용 가능하다."

분해 결과:

1. **인코딩 보편성**: metric target을 갖는 모든 등호 \(F(x)=G(x)\)는 정칙 defect functional을 받는다. 이 층은 정리로 닫힌다.
2. **자명성 경고**: 해집합 수준의 보편성은 공짜다. 내용은 defect의 정칙성(l.s.c., Lipschitz, coercivity)에 있다.
3. **정보 정리**: 같은 zero set을 주는 defect들은 manifest 극한은 공유하지만 finite-\(\beta\) 구조는 다르다. 즉 PreEq 데이터는 등호 자체보다 엄격히 많다.
4. **적용 경계**: 계산 불가능성, 측도 가능성, size 문제는 PreEq가 우회하지 못한다.

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| residual/metric 인코딩 | `Exact` | 정리 1.1, 1.2 |
| 해집합 수준 보편성 | `Exact`이지만 자명 | 주의 1.3 |
| 선형계 manifest = Moore-Penrose | `Exact` | 정리 2.1 |
| 축약사상 고정점 농축 | `Exact under assumptions` | 정리 2.2 |
| 유한 논리(SAT/max-SAT) 농축 | `Exact` | 정리 2.3 |
| defect는 등호보다 많은 정보 | `Exact` | 정리 3.1, 예시 3.2 |
| 변분 등호의 defect 선택 | `Selection` | 4절 |
| 계산 가능성/알고리즘 비용 | PreEq로 개선 안 됨 | 5절 |
| 범주적 functorial 승격 | 부분 `Exact`, 전체 `Open` | 정리 6.1 |

## 1. 보편 인코딩

PreEq 스킴의 입력은 항상 쌍 \((X,\delta)\)다. \(X\)는 후보공간, \(\delta:X\to[0,\infty]\)는 defect functional이며 해집합은

$$
Z(\delta)=\delta^{-1}(0)
$$

이다. 여기에 prior \(\mu_0\)와 \(\beta\)를 더하면 [01_공리와증명.md](01_공리와증명.md)-[02c_Gamma수렴과Gibbs농축.md](02c_Gamma수렴과Gibbs농축.md)의 Gibbs 농축 기계가 작동한다.

### 정리 1.1: residual 인코딩

\(X\)가 topological space, \((Y,d)\)가 metric space, \(F,G:X\to Y\)가 continuous라고 하자.

$$
\delta(x):=d\big(F(x),G(x)\big)
$$

로 두면 \(\delta\)는 continuous이고

$$
Z(\delta)=\{x:F(x)=G(x)\}
$$

는 닫힌집합이다.

증명:

\(d:Y\times Y\to[0,\infty)\)는 continuous이고 \((F,G):X\to Y\times Y\)도 continuous이므로 합성 \(\delta\)는 continuous다. \(Z(\delta)=\delta^{-1}(\{0\})\)는 닫힌집합의 연속 역상이라 닫혀 있다. 끝.

해석:

> 수학에서 등호는 거의 항상 "두 사상이 metric(또는 normed) target에서 같다"로 제시된다. 다항방정식, 선형계, 고정점, 미분방정식 잔차, 고유값 문제, 함수방정식이 모두 이 꼴이다. 따라서 이 모든 등호는 자동으로 continuous defect를 받고 PreEq의 입력이 된다. 이것이 보편성 주장의 닫히는 절반이다.

### 정리 1.2: metric 인코딩

\((X,d)\)가 metric space이고 \(R\subset X\)가 nonempty closed set이면

$$
\delta(x):=d(x,R)=\inf_{r\in R}d(x,r)
$$

은 1-Lipschitz이고 \(Z(\delta)=R\)이다.

증명:

1-Lipschitz는 삼각부등식에서 표준이다. \(\delta(x)=0\)이면 \(x\)로 수렴하는 \(R\)의 열이 있고 \(R\)이 닫혀 있으므로 \(x\in R\)이다. 역은 자명하다. 끝.

### 주의 1.3: 해집합 보편성은 자명하다

임의의 집합 \(X\)와 임의의 부분집합 \(R\subset X\)에 대해 \(\delta=\infty\cdot\mathbf 1_{X\setminus R}\)로 두면 \(Z(\delta)=R\)이다. 즉 "모든 등호가 PreEq에 들어간다"는 주장은 해집합 수준에서는 **내용이 없다**.

내용은 다음에 있다.

| 추가 구조 | 무엇을 사게 되는가 |
|---|---|
| \(\delta\) l.s.c. + compact sublevel | 02의 농축 정리, minimizer 존재 |
| \(\delta\) Lipschitz/continuous | 05g continuity route의 recovery |
| \(\delta\)의 정량적 하한 (예: 정리 2.2) | 근사해에서 해까지의 거리 통제 |
| prior \(\mu_0\) | 퇴화 등호의 selection (정리 2.1), 잔류장 |

## 2. 적용 catalog

각 항목은 선언이 아니라 닫힌 정리로 쓴다.

### 정리 2.1: 선형 등호의 manifest는 Moore-Penrose 해다

\(A\in\mathbb R^{m\times n}\), \(b\in\mathbb R^m\), 등호 \(Ax=b\)에 대해

$$
\delta(x)=|Ax-b|^2,
\qquad
\mu_0=\mathcal N(0,I_n),
\qquad
d\mu_\beta\propto e^{-\beta\delta}d\mu_0
$$

로 두면:

1. \(\mu_\beta\)는 Gaussian이다.
2. \(\beta\to\infty\)에서 평균은 Moore-Penrose 해 \(A^+b\)로 수렴한다.
3. \(\ker A=0\)이면 \(\mu_\beta\Rightarrow\delta_{A^+b}\)다.
4. \(\ker A\ne0\)이면 row-space 방향으로만 분산이 0으로 가고, 등호가 결정하지 않는 \(\ker A\) 방향에는 prior 분산이 그대로 남는다.

증명:

SVD \(A=U\Sigma V^\top\)를 잡고 \(y=V^\top x\), \(c=U^\top b\)로 좌표를 바꾼다. prior는 회전불변이라 \(y\sim\mathcal N(0,I)\)이고

$$
\delta=\sum_{i\le r}(\sigma_iy_i-c_i)^2+\sum_{i>r}c_i^2
$$

로 좌표별로 분리된다(\(r=\operatorname{rank}A\)). 따라서 \(\mu_\beta\)는 좌표별 Gaussian 곱이다.

\(i\le r\): 밀도 \(\propto\exp(-\beta(\sigma_iy_i-c_i)^2-y_i^2/2)\)는 평균

$$
m_i(\beta)
=
\frac{2\beta\sigma_ic_i}{2\beta\sigma_i^2+1}
\xrightarrow{\beta\to\infty}
\frac{c_i}{\sigma_i},
$$

분산 \((2\beta\sigma_i^2+1)^{-1}\to0\)을 갖는다.

\(i>r\): \(\delta\)가 \(y_i\)에 의존하지 않으므로 \(\mu_\beta\)는 prior 그대로, 평균 0, 분산 1이다.

평균 벡터는 \(\sum_{i\le r}(c_i/\sigma_i)v_i=A^+b\)로 수렴한다. \(\ker A=0\)이면 모든 좌표의 분산이 0으로 가므로 Dirac 약수렴이다. 끝.

해석:

> 퇴화 등호(해가 무한히 많거나 없는 경우)에서 PreEq의 선택은 prior가 닫는다. 해가 있으면 minimum-norm 해, 없으면 least-squares 해가 manifest된다. 그리고 등호가 결정하지 않는 방향은 **잔류장으로 남는다** (4번 항목). 이는 03a의 conditional readout과 같은 구조이고, Bayesian 선형회귀/ridge는 정확히 이 스킴의 finite-\(\beta\) 층이다.

### 정리 2.2: 축약사상 고정점

\((X,d)\)가 complete metric space, \(T:X\to X\)가 Lipschitz 상수 \(L<1\)의 축약사상, \(x_*\)가 유일 고정점이라 하자. defect \(\delta(x)=d(x,Tx)\)는

$$
d(x,x_*)\le\frac{\delta(x)}{1-L}
$$

을 만족한다. 따라서 \(X\)가 proper(닫힌 공이 compact)이면 \(\delta\)는 good rate function이고, recovery prior에 대해 \(\mu_\beta\Rightarrow\delta_{x_*}\)다.

증명:

삼각부등식과 축약성으로

$$
d(x,x_*)
\le
d(x,Tx)+d(Tx,Tx_*)
\le
\delta(x)+L\,d(x,x_*),
$$

이항하면 하한이 나온다. 이 하한은 sublevel \(\{\delta\le c\}\)를 반지름 \(c/(1-L)\)의 닫힌 공 안에 가두므로 proper 공간에서 compact closure를 갖고, \(\delta\)는 continuous(\(T\) Lipschitz)라 l.s.c.다. 02a의 농축 정리를 적용하면 끝.

### 정리 2.3: 유한 논리 등호

명제변수 \(n\)개의 CNF \(\varphi\)에 대해 \(X=\{0,1\}^n\), \(\delta(x)=\)위반된 절의 수, \(\mu_0=\)uniform으로 두면 01의 유한 농축 정리에 의해

$$
\mu_\beta
\xrightarrow{\beta\to\infty}
\text{uniform on }\operatorname*{argmin}\delta.
$$

\(\varphi\)가 충족 가능하면 manifest는 해의 균등분포이고, 불충족이면 max-SAT 최적해의 균등분포다.

증명:

\(X\)가 유한집합이고 모든 후보의 prior weight가 양수이므로 01의 정리(05h 정리 2.1과 동일)가 그대로 적용된다. 끝.

해석:

> 이산수학과 논리의 등호도 같은 기계에 들어간다. "해 없음 \(\to\) 최소잔차 manifest"(02의 구조)가 max-SAT으로 정확히 실현된다.

### 미분방정식, 고유값, 함수방정식

\(u'=f(u)\)류의 등호는 residual norm \(\delta(u)=\|u'-f(u)\|_{L^2}\)로 정리 1.1의 인스턴스가 된다. Galerkin 잔차 최소화, least-squares 솔버, PINN 손실함수는 전부 이 defect의 finite-\(\beta\) 또는 zero-temperature 처리다. 고유값 문제는 \(\delta(v,\lambda)=|Av-\lambda v|^2\) (정규화 \(|v|=1\))로 들어간다. 경로공간 위의 자연스러운 defect는 [05c_pathspace_closure_checklist.md](05c_pathspace_closure_checklist.md) 이후의 조건들로 닫는다.

## 3. Defect는 등호보다 많은 정보다

### 정리 3.1: 같은 zero set, 같은 manifest, 다른 finite-\(\beta\)

\(\delta_1,\delta_2\)가 같은 zero set \(Z\ne\varnothing\)을 갖고, 둘 다 good rate이며 recovery prior \(\mu_0\)에 대해 \(Z\cap\operatorname{supp}\mu_0\ne\varnothing\)이라 하자. 그러면 두 Gibbs family는 같은 집합 \(Z\)로 농축한다. 그러나 finite \(\beta\)에서 두 측도는 일반적으로 다르며, 비선택 후보의 순위도 다를 수 있다.

증명:

농축은 05e 정리 1.1을 각각 적용하면 된다(최소값 0이 \(Z\)에서 달성된다). 차이는 예시 3.2로 보인다. 끝.

### 예시 3.2

\(X=\{a,b,c\}\), uniform prior, 두 defect:

$$
\delta_1=(0,\,1,\,2),
\qquad
\delta_2=(0,\,2,\,1).
$$

zero set은 둘 다 \(\{a\}\)이고 manifest도 \(\delta_a\)로 같다. 그러나 모든 finite \(\beta\)에서

$$
\mu_\beta^{(1)}(b)>\mu_\beta^{(1)}(c),
\qquad
\mu_\beta^{(2)}(b)<\mu_\beta^{(2)}(c).
$$

즉 비선택 잔류측도 \(\mu_{\mathrm{ns}}\)의 모양이 반대다. [05a_phi_pushforward.md](05a_phi_pushforward.md)의 잔류장 \(\phi\)는 defect 선택에 의존한다.

해석:

> "등호 이후"(\(Z\)와 manifest)는 defect의 동치류에서 불변이고, "등호 이전"(근사해 순위, 잔류장, 유한 \(\beta\) 분율)은 defect 자체를 요구한다. 이 폴더가 등호 이후 수학에 환원되지 않는 이유가 이 정리다. 보편성 주장의 정확한 형태는 다음이다.
>
> $$
> \boxed{
> \text{모든 metric 등호는 PreEq 입력을 받지만, PreEq 데이터는 등호의 보존량이 아니라 추가 구조다.}
> }
> $$

## 4. 변분 등호의 defect 선택

Euler-Lagrange 등호 \(\nabla S=0\)에는 자연스러운 defect가 둘 있다.

| defect | zero set | manifest |
|---|---|---|
| \(\delta_1=|\nabla S|\) | 모든 critical point | critical set 전체로 농축 |
| \(\delta_2=S-\inf S\) | minimizer만 | minimizer로 농축 |

같은 "등호"에서 출발해도 defect 선택이 manifest를 바꾼다(이번에는 zero set 자체가 다르다). CE가 \(E_{\mathrm{fold}}=W-W_{\min}\), 즉 \(\delta_2\) 계열을 채택한 것은 `Selection`이며, 05i 정리 4.2가 그 선택을 scaled Brownian prior의 LDP rate로 정당화한 것이다.

## 5. 적용 경계

PreEq가 우회하지 못하는 것을 명시한다.

| 경계 | 내용 |
|---|---|
| 계산 불가능성 | Diophantine 등호의 해 존재는 비결정적(Hilbert 10번 문제)이다. \(Z(\delta)\ne\varnothing\) 판정을 PreEq가 대신해 주지 않는다. 스킴은 의미론이지 알고리즘이 아니다 |
| 알고리즘 비용 | 정리 2.3의 manifest는 존재하지만, 그것을 sampling하는 annealing의 mixing time은 최악의 경우 지수적이다. NP-hardness는 그대로다 |
| 측도 가능성 | \(\delta\)가 가측이 아니거나 \(X\)에 표준 Borel 구조가 없으면 Gibbs 기계가 시작되지 않는다 |
| size | "모든 집합 위의 functor 등호"처럼 proper class 크기의 후보공간은 prior를 받을 수 없다. small/locally small 제한이 필요하다 |
| 등호의 강도 | HoTT의 identity type처럼 등호 자체가 공간(경로공간)인 설정과의 대응은 시사적이지만 현재 정리가 없다. `Bridge/Open` |

## 6. 범주적 위치

[04_PreEq_범주.md](04_PreEq_범주.md)의 구조에 인코딩 층을 붙인다.

### 정리 6.1: defect 비증가 사상은 근사해를 보존한다

객체를 쌍 \((X,\delta_X)\) (\(\inf\delta_X=0\)), 사상을

$$
f:(X,\delta_X)\to(Y,\delta_Y),
\qquad
\delta_Y\circ f\le\delta_X
$$

인 가측사상으로 두면 이는 category를 이룬다(항등사상과 합성이 조건을 보존한다). 모든 사상 \(f\)와 \(\eta>0\)에 대해

$$
f\big(R_\eta(\delta_X)\big)\subset R_\eta(\delta_Y),
\qquad
f\big(Z(\delta_X)\big)\subset Z(\delta_Y).
$$

증명:

\(\delta_X(x)<\eta\)이면 \(\delta_Y(f(x))\le\delta_X(x)<\eta\)다. \(\inf\delta_Y\le\inf\delta_X=0\)이므로 \(R_\eta\)의 정의와 일치한다. zero set 보존은 \(\eta\downarrow0\)이다. 합성이 조건을 보존하는 것은 부등식의 연쇄다. 끝.

남은 것 (`Open`): 이 category에서 Gibbs family 배정 \((X,\delta,\mu_0)\mapsto(\mu_\beta)_\beta\)를 [04a_Markov_Kleisli.md](04a_Markov_Kleisli.md)의 weight Kleisli로 가는 functor로 승격하는 것. 사상이 prior를 어떻게 밀어야 하는지(\(f_*\mu_0^X\)와 \(\mu_0^Y\)의 호환 조건)가 추가 데이터로 필요하다.

## 7. 닫힌 것과 남은 것

닫힌 것:

| 항목 | 상태 |
|---|---|
| residual/metric 인코딩 | 정리 1.1, 1.2 |
| 자명성 경계 분리 | 주의 1.3 |
| 선형 등호 manifest = \(A^+b\), 잔류 방향 분리 | 정리 2.1 |
| 고정점 defect의 거리 하한과 농축 | 정리 2.2 |
| 유한 논리 농축 | 정리 2.3 |
| defect가 등호보다 많은 정보라는 분리 정리 | 정리 3.1, 예시 3.2 |
| defect 비증가 category와 근사해 보존 | 정리 6.1 |

남은 것:

| 병목 | 다음 작업 |
|---|---|
| Gibbs 배정의 functor 승격 | prior 호환 조건 설계, 04a 연결 |
| 정리 2.1의 코드 회귀 | `pre_eq` finite 코어로 Gaussian 좌표 분리 검증 |
| HoTT identity type 대응 | 시사 수준. 정리 없음 |
| 무한차원 변분 등호 | 05 시리즈가 담당, kinetic 인코딩은 05i로 닫힘 |

## 8. 결론

$$
\boxed{
F(x)=G(x)
\;\xrightarrow{\;d\;}\;
\delta=d(F,G)
\;\xrightarrow{\;\mu_0,\beta\;}\;
\mu_\beta
\;\xrightarrow{\;\beta\to\infty\;}\;
Z(\delta)\ \text{manifest}
}
$$

보편성의 닫힌 형태: metric 등호는 전부 이 파이프라인에 들어가고(정리 1.1), 퇴화는 prior가 닫으며(정리 2.1), 이산/연속/변분을 가리지 않는다(정리 2.2, 2.3). 그러나 PreEq 데이터는 등호의 함수가 아니라 추가 구조이고(정리 3.1), 계산 가능성의 벽은 그대로다(5절). "거의 모든 등호 수학에 적용 가능"은 이 두 단서와 함께 정리로 성립한다.
