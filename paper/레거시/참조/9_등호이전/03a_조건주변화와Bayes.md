# 03a. 조건 주변화와 Bayes Readout

이 문서는 03장의 joint Gibbs 상태에서 조건, 값, 조건부 분포를 어떤 순서로 읽는지 정의하고 Bayes 분해를 유한공간에서 증명한다. 핵심은 prior·likelihood·posterior의 대수적 분해와, 어떤 변수를 표면에 manifest할지 정하는 readout 선택 및 물리적 인과 해석을 분리하는 데 있다.

독자는 03장의 조건–값 joint 상태와 유한 확률의 합 규칙을 알고 있으면 된다. 먼저 양의 marginal에서만 정의되는 조건부 확률을 고정하고, marginal free energy와 projected minimizer를 거쳐 정보 손실·측정 비유의 한계로 읽는 순서이며, 연속 영측도 조건화는 이 문서 범위 밖이다.

## 0. 목표

03장은 조건과 값이 함께 후보가 되는 joint manifest를 정의했다. 여기서는 같은 결합 상태에서 어느 좌표를 주변화하거나 조건부로 읽는지가 서로 다른 산출을 만든다는 점을 정리하며, Bayes 항등식이 조건의 물리적 원인이나 인과 화살표를 제공하지 않음을 미리 구분한다.

핵심 질문:

> 조건 $C$를 먼저 읽는가, 값 $a$를 먼저 읽는가, 아니면 $(C,a)$ 쌍 전체를 읽는가?

형식 출처:

| 항목 | 판정 | 이유 |
|---|---|---|
| joint Gibbs 상태 | `[정의]` | 유한공간 정규화 |
| 조건/값 marginal | `[산출]` | 합으로 계산 |
| conditional readout | `[정리]` | Bayes 분해 |
| readout 선택 | `[공리: 모델 선택]` | 무엇을 표면에 manifest할지의 선택 |

## 1. 세팅

모든 합과 조건부 확률이 영측도 문제 없이 정의되도록 조건공간과 값공간을 유한집합으로 둔다. $\rho_0$는 joint prior, Gibbs 가중은 에너지 기반 likelihood factor, $\rho_\beta$는 posterior와 같은 대수 구조를 갖지만 이 명명만으로 실제 데이터 생성 모델이 지정되지는 않는다.

조건공간 $K$와 값공간 $A$는 공집합이 아닌 유한집합이다.

초기 joint 상태는

$$
\rho_0(k,a)\ge0,
\qquad
\sum_{k\in K}\sum_{a\in A}\rho_0(k,a)=1
$$

이다. support는

$$
S=\operatorname{supp}\rho_0
\subset K\times A
$$

이다.

joint energy는

$$
E:K\times A\to\mathbb R_{\ge0}
$$

이다.

Gibbs joint 상태는

$$
\rho_\beta(k,a)
=
\frac{e^{-\beta E(k,a)}\rho_0(k,a)}
{Z_\beta},
\qquad
Z_\beta=
\sum_{(l,b)\in K\times A}
e^{-\beta E(l,b)}\rho_0(l,b)
$$

이다.

## 2. 세 가지 readout

정규화된 joint 상태를 하나만 정해도 무엇을 관측 변수로 택하는지에 따라 서로 다른 분포를 얻는다. 다음 세 readout은 합과 조건부 확률의 정의이며, 관측 장치가 어느 변수를 실제로 인과적으로 선택한다는 사실은 별도의 모델·실험으로 검증되어야 한다.

### 2.1 Joint readout

joint readout은 조건과 값을 함께 보존하므로 가장 많은 상관 정보를 남긴다. 여기서의 zero-temperature 최소집합은 support 위의 에너지 최소쌍 집합이며, 유일성이 없으면 단일 쌍의 manifest를 결론낼 수 없다.

쌍 전체를 읽으면 후보공간은 $K\times A$다.

$$
(k,a)\sim\rho_\beta
$$

zero-temperature manifest 집합은

$$
S_*=
\operatorname*{argmin}_{(k,a)\in S}E(k,a)
$$

이다.

### 2.2 조건 readout

조건 readout은 값 좌표를 유한합으로 제거해 어떤 조건이 높은 posterior 질량을 갖는지 본다. 이 주변화는 값을 조건으로 원인화하는 연산이 아니며, 같은 marginal을 만드는 서로 다른 joint 상태를 구분하지 못한다.

조건만 읽으면 조건 marginal을 쓴다.

$$
\nu_\beta(k)
=
\sum_{a\in A}\rho_\beta(k,a)
$$

### 2.3 값 readout

값 readout은 조건 좌표를 제거해 값의 posterior를 얻는다. 이 연산 뒤에는 어떤 조건이 그 값을 지지했는지 일반적으로 되살릴 수 없으므로, 값의 집중을 조건 선택의 증거로 해석해서는 안 된다.

값만 읽으면 값 marginal을 쓴다.

$$
\mu_\beta(a)
=
\sum_{k\in K}\rho_\beta(k,a)
$$

## 3. Bayes 분해

Bayes 분해는 결합 확률을 marginal과 양의 확률을 가진 조건부로 인수분해하는 대수적 항등식이다. 아래 정의는 분모가 양수일 때만 성립하며, 영질량 조건 또는 연속 조건공간의 영측도 사건에서는 regular conditional probability 같은 추가 구조 없이는 이 비율식을 사용할 수 없다.

조건 prior와 값 prior를

$$
\nu_0(k)=\sum_a\rho_0(k,a),
\qquad
\mu_0(a)=\sum_k\rho_0(k,a)
$$

로 둔다.

$\nu_0(k)>0$이면

$$
\rho_0(a|k)=\frac{\rho_0(k,a)}{\nu_0(k)}
$$

이고, $\mu_0(a)>0$이면

$$
\rho_0(k|a)=\frac{\rho_0(k,a)}{\mu_0(a)}
$$

이다.

마찬가지로 $\nu_\beta(k)>0$, $\mu_\beta(a)>0$일 때

$$
\rho_\beta(a|k)=\frac{\rho_\beta(k,a)}{\nu_\beta(k)},
\qquad
\rho_\beta(k|a)=\frac{\rho_\beta(k,a)}{\mu_\beta(a)}
$$

이다.

**정리 3.1**  
다음 Bayes 분해가 성립한다.

$$
\rho_\beta(k,a)
=
\nu_\beta(k)\rho_\beta(a|k)
=
\mu_\beta(a)\rho_\beta(k|a)
$$

또한 $\nu_0(k)>0$이면

$$
\rho_\beta(a|k)
=
\frac{e^{-\beta E(k,a)}\rho_0(a|k)}
{\sum_{b\in A}e^{-\beta E(k,b)}\rho_0(b|k)}.
$$

**증명.**

첫 식은 conditional probability의 정의다. 둘째 식은

$$
\rho_\beta(a|k)
=
\frac{\rho_\beta(k,a)}{\sum_b\rho_\beta(k,b)}
$$

에 Gibbs 정의를 대입하면 $Z_\beta$와 $\nu_0(k)$가 약분되어 나온다. $\square$

## 4. 조건 free energy

조건을 먼저 읽는 경우에는 각 조건 내부의 값 후보를 partition function으로 접은 effective energy가 필요하다. 이 free energy는 유한 합과 양의 조건 prior에서 정의되는 산출이며, 온도와 prior가 바뀌면 값이 바뀌므로 물리적 잠재에너지 또는 인과 비용으로 동일시되지 않는다.

조건 $k$가 주어졌을 때 내부 값 partition을

$$
Z_A(k;\beta)
=
\sum_{a\in A}e^{-\beta E(k,a)}\rho_0(a|k)
$$

로 둔다. $\nu_0(k)>0$일 때 조건 free energy는

$$
F_K^\beta(k)
=
-\frac1\beta\log Z_A(k;\beta)
$$

이다.

그러면 조건 marginal은

$$
\nu_\beta(k)
=
\frac{\nu_0(k)e^{-\beta F_K^\beta(k)}}
{\sum_l\nu_0(l)e^{-\beta F_K^\beta(l)}}.
$$

**정리 4.1**  
$\nu_0(k)>0$이면

$$
F_K^\beta(k)
\to
E_K(k)
:=
\min_{a:\rho_0(k,a)>0}E(k,a)
$$

이다.

**증명.**

04장의 log-sum-exp 정리를 $a$-합에 적용하면 된다. $\square$

해석:

> 조건만 먼저 읽는다는 것은 각 조건 안에서 가능한 값들을 먼저 soft-min으로 접고, 그 조건 free energy를 비교한다는 뜻이다.

## 5. 값 free energy

값을 먼저 읽는 경우에는 역할을 바꾸어 조건 후보를 접는다. 대칭적인 대수 형식은 조건과 값이 의미론적으로 교환 가능하다는 뜻은 아니며, 실제 모델에서 어느 축이 intervention·observation인지에 따라 prior와 likelihood의 해석은 달라질 수 있다.

값 $a$에 대해서도 대칭적으로

$$
Z_K(a;\beta)
=
\sum_{k\in K}e^{-\beta E(k,a)}\rho_0(k|a)
$$

를 두고

$$
F_A^\beta(a)
=
-\frac1\beta\log Z_K(a;\beta)
$$

라 둔다. 그러면

$$
\mu_\beta(a)
=
\frac{\mu_0(a)e^{-\beta F_A^\beta(a)}}
{\sum_b\mu_0(b)e^{-\beta F_A^\beta(b)}}.
$$

또한

$$
F_A^\beta(a)
\to
E_A(a)
:=
\min_{k:\rho_0(k,a)>0}E(k,a).
$$

## 6. Projected minimizer 정리

zero-temperature에서 joint 최소집합으로의 농축은 각 projection의 농축을 함의하지만, 역방향은 성립하지 않는다. 다음 정리는 유한 support와 이미 증명된 joint 농축을 쓰는 형식 결과이며, marginal의 단일 모드만으로 유일 joint 원인을 식별할 수 없다는 반례 가능성을 남긴다.

joint 최소집합을

$$
S_*=
\operatorname*{argmin}_{(k,a)\in S}E(k,a)
$$

라 둔다.

조건 projection과 값 projection은

$$
K_*=\{k:\exists a,\ (k,a)\in S_*\}
$$

$$
A_*=\{a:\exists k,\ (k,a)\in S_*\}
$$

이다.

**정리 6.1**  
$\beta\to\infty$에서

$$
\rho_\beta(S_*)\to1,
\qquad
\nu_\beta(K_*)\to1,
\qquad
\mu_\beta(A_*)\to1.
$$

**증명.**

첫 번째 식은 01장의 유한공간 농축 정리를 $K\times A$에 적용한 것이다.

두 번째 식은

$$
\nu_\beta(K_*)
=
\sum_{k\in K_*}\sum_a\rho_\beta(k,a)
\ge
\sum_{(k,a)\in S_*}\rho_\beta(k,a)
=
\rho_\beta(S_*)
\to1
$$

에서 따른다. 세 번째 식도 같은 방식이다. $\square$

## 7. Readout의 비가역성

주변화는 상태공간을 단순하게 만들지만 조건–값 상관을 버리는 비가역 연산이다. 다음 표는 유한 최소집합에서의 정보 관계를 비교하며, 표의 pattern이 연속 측도에서의 식별성·인과 방향까지 확장된다는 주장은 아니다.

joint readout은 condition/value marginal보다 강하다.

| 상황 | joint | 조건 marginal | 값 marginal |
|---|---|---|---|
| 유일 $(k_*,a_*)$ | $\delta_{(k_*,a_*)}$ | $\delta_{k_*}$ | $\delta_{a_*}$ |
| 같은 조건, 여러 값 | 여러 $(k_*,a)$ | $\delta_{k_*}$ | 여러 값 |
| 여러 조건, 같은 값 | 여러 $(k,a_*)$ | 여러 조건 | $\delta_{a_*}$ |
| 여러 조건-값 | 여러 쌍 | projection | projection |

따라서 값만 manifest 되었다고 해서 어떤 조건이 manifest 되었는지는 일반적으로 복원할 수 없다. 조건만 manifest 되어도 값이 반드시 유일한 것은 아니다.

## 8. 등호 이전 해석

등호 이전 언어에서 표면에 나타난 식과 해를 함께 기록하면 joint readout으로 해석할 수 있고, 한쪽만 기록하면 marginal 또는 conditional readout이 된다. 이는 확률 모델의 표현 구분이지 측정이 조건을 만들거나 Bayes 갱신이 물리적 시간 인과를 구현한다는 증명은 아니며, CE 측정 operator 연결은 별도 미완성 다리다.

`x+1=2`와 `x=1`이 표면에 같이 나타나는 경우는 joint readout이다.

하지만 실제로는 세 층이 분리될 수 있다.

| readout | 의미 |
|---|---|
| $\nu_\beta(k)$ | 어떤 조건/문제가 manifest 되는가 |
| $\rho_\beta(a|k)$ | 그 조건 아래 어떤 값이 manifest 되는가 |
| $\rho_\beta(k,a)$ | 조건-값 쌍 자체가 manifest 되는가 |

이 분리는 06장의 측정 operator 후보와도 연결된다. 측정 결과만 읽는 것과, 어떤 측정 조건이 선택되었는지를 함께 읽는 것은 다른 일이다.

## 9. 결론

결론적으로 joint posterior는 가장 많은 정보를 보존하고, marginal·conditional readout은 명시한 양의 분모와 유한합 아래에서만 안전하게 정의된다. 연속 조건공간의 measure-zero conditioning, 데이터 provenance에 근거한 likelihood, 그리고 통계적 Bayes 구조에서 물리 인과로 넘어가는 단계는 이 문서가 닫지 않은 미완성 범위다.

조건공간 도구의 핵심은 다음이다.

$$
\rho_\beta(k,a)
\quad\Rightarrow\quad
\nu_\beta(k),\ \mu_\beta(a),\ \rho_\beta(a|k),\ \rho_\beta(k|a).
$$

joint manifest는 가장 많은 정보를 보존한다. marginal readout은 표면을 단순하게 만들지만, 조건 또는 값의 잔류 정보를 잃을 수 있다.
