# 03a. 조건 주변화와 Bayes Readout

## 0. 목표

03장은 조건과 값이 함께 후보가 되는 joint manifest를 정의했다. 이 문서는 joint 상태에서 무엇을 읽느냐에 따라 결과가 어떻게 달라지는지 닫는다.

핵심 질문:

> 조건 \(C\)를 먼저 읽는가, 값 \(a\)를 먼저 읽는가, 아니면 \((C,a)\) 쌍 전체를 읽는가?

현재 판정:

| 항목 | 판정 | 이유 |
|---|---|---|
| joint Gibbs 상태 | `Exact` | 유한공간 정규화 |
| 조건/값 marginal | `Exact` | 합으로 정의 |
| conditional readout | `Exact` | Bayes 분해 |
| readout 선택 | `Selection` | 무엇을 표면에 manifest할지의 선택 |

## 1. 세팅

조건공간 \(K\)와 값공간 \(A\)는 공집합이 아닌 유한집합이다.

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

### 2.1 Joint readout

쌍 전체를 읽으면 후보공간은 \(K\times A\)다.

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

조건만 읽으면 조건 marginal을 쓴다.

$$
\nu_\beta(k)
=
\sum_{a\in A}\rho_\beta(k,a)
$$

### 2.3 값 readout

값만 읽으면 값 marginal을 쓴다.

$$
\mu_\beta(a)
=
\sum_{k\in K}\rho_\beta(k,a)
$$

## 3. Bayes 분해

조건 prior와 값 prior를

$$
\nu_0(k)=\sum_a\rho_0(k,a),
\qquad
\mu_0(a)=\sum_k\rho_0(k,a)
$$

로 둔다.

\(\nu_0(k)>0\)이면

$$
\rho_0(a|k)=\frac{\rho_0(k,a)}{\nu_0(k)}
$$

이고, \(\mu_0(a)>0\)이면

$$
\rho_0(k|a)=\frac{\rho_0(k,a)}{\mu_0(a)}
$$

이다.

마찬가지로 \(\nu_\beta(k)>0\), \(\mu_\beta(a)>0\)일 때

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

또한 \(\nu_0(k)>0\)이면

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

에 Gibbs 정의를 대입하면 \(Z_\beta\)와 \(\nu_0(k)\)가 약분되어 나온다. \(\square\)

## 4. 조건 free energy

조건 \(k\)가 주어졌을 때 내부 값 partition을

$$
Z_A(k;\beta)
=
\sum_{a\in A}e^{-\beta E(k,a)}\rho_0(a|k)
$$

로 둔다. \(\nu_0(k)>0\)일 때 조건 free energy는

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
\(\nu_0(k)>0\)이면

$$
F_K^\beta(k)
\to
E_K(k)
:=
\min_{a:\rho_0(k,a)>0}E(k,a)
$$

이다.

**증명.**

04장의 log-sum-exp 정리를 \(a\)-합에 적용하면 된다. \(\square\)

해석:

> 조건만 먼저 읽는다는 것은 각 조건 안에서 가능한 값들을 먼저 soft-min으로 접고, 그 조건 free energy를 비교한다는 뜻이다.

## 5. 값 free energy

값 \(a\)에 대해서도 대칭적으로

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
\(\beta\to\infty\)에서

$$
\rho_\beta(S_*)\to1,
\qquad
\nu_\beta(K_*)\to1,
\qquad
\mu_\beta(A_*)\to1.
$$

**증명.**

첫 번째 식은 01장의 유한공간 농축 정리를 \(K\times A\)에 적용한 것이다.

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

에서 따른다. 세 번째 식도 같은 방식이다. \(\square\)

## 7. Readout의 비가역성

joint readout은 condition/value marginal보다 강하다.

| 상황 | joint | 조건 marginal | 값 marginal |
|---|---|---|---|
| 유일 \((k_*,a_*)\) | \(\delta_{(k_*,a_*)}\) | \(\delta_{k_*}\) | \(\delta_{a_*}\) |
| 같은 조건, 여러 값 | 여러 \((k_*,a)\) | \(\delta_{k_*}\) | 여러 값 |
| 여러 조건, 같은 값 | 여러 \((k,a_*)\) | 여러 조건 | \(\delta_{a_*}\) |
| 여러 조건-값 | 여러 쌍 | projection | projection |

따라서 값만 manifest 되었다고 해서 어떤 조건이 manifest 되었는지는 일반적으로 복원할 수 없다. 조건만 manifest 되어도 값이 반드시 유일한 것은 아니다.

## 8. 등호 이전 해석

`x+1=2`와 `x=1`이 표면에 같이 나타나는 경우는 joint readout이다.

하지만 실제로는 세 층이 분리될 수 있다.

| readout | 의미 |
|---|---|
| \(\nu_\beta(k)\) | 어떤 조건/문제가 manifest 되는가 |
| \(\rho_\beta(a\mid k)\) | 그 조건 아래 어떤 값이 manifest 되는가 |
| \(\rho_\beta(k,a)\) | 조건-값 쌍 자체가 manifest 되는가 |

이 분리는 06장의 측정 operator 후보와도 연결된다. 측정 결과만 읽는 것과, 어떤 측정 조건이 선택되었는지를 함께 읽는 것은 다른 일이다.

## 9. 결론

조건공간 도구의 핵심은 다음이다.

$$
\rho_\beta(k,a)
\quad\Rightarrow\quad
\nu_\beta(k),\ \mu_\beta(a),\ \rho_\beta(a|k),\ \rho_\beta(k|a).
$$

joint manifest는 가장 많은 정보를 보존한다. marginal readout은 표면을 단순하게 만들지만, 조건 또는 값의 잔류 정보를 잃을 수 있다.

