# CE 코어 강화 루프

상태: `Working canonical / loop 2 complete`  
범위: 생존함수, 자기·타공간 재귀, 공간 차원 선택, 전자약 유효 깊이  
코드 게이트: `core_axioms.py`, `multispace_bootstrap.py`  
실행: `python examples/physics/core_axiom_loop.py`,
`python examples/physics/multispace_recursion_gate.py`  
회귀 검증: `pytest tests/test_core_axioms.py tests/test_multispace_bootstrap.py`

## 0. 목적

이 문서의 목적은 약한 주장을 단순히 하향 분류하는 것이 아니다. 각 간극에
대해 다음 루프를 반복해, 반례를 실제로 제거하는 최소 구조를 찾는다.

$$
\boxed{
\text{claim}
\to
\text{countermodels}
\to
\text{minimal added principle}
\to
\text{theorem}
\to
\text{executable gate}
\to
\text{held-out physical test}
}
$$

루프의 규칙은 두 가지다.

1. 새 원리를 추가해 정리가 강해졌다면, 무엇이 추가되었는지 숨기지 않는다.
2. 대수적 residual이 작다는 사실과 자연의 관측량에 대한 식별을 합치지 않는다.

현재 상위 코어는 인접 공간을 포함한 다음 벡터식으로 묶는다.

$$
x_i
=
S_i\!\left(\sum_j A_{ij}K_j(x_j)\right).
$$

여기서

- \(S\): 누적 깊이에 대한 생존법칙,
- \(K\): 현재 생존율이 만드는 내부 억압률,
- \(A_{ii}\): 공간 \(i\)의 자기재귀 깊이,
- \(A_{ij}\;(i\ne j)\): 공간 \(j\)를 읽는 타공간 재귀 깊이,
- \(x_i\): 각 공간/종류의 생존 고정점

이다. 기존 스칼라식은 \(A=[D_{\mathrm{eff}}]\)인 1종류 특수 경우다.
CE의 현재 국소 후보는

$$
S_i(z)=e^{-z},
\qquad
K_i(x_i)=1-x_i
$$

이며, \(A\) 자체의 물리적 구성은 별도 공간결합 문제다.

## 1. 루프 S: 지수 생존법칙

### 1.1 반례 공간

무차원성, \(S(0)=1\), \(0<S(D)\le1\)만으로는 지수형이 나오지 않는다.
예를 들어

$$
S_1(D)=e^{-D^2},
\qquad
S_2(D)=\frac1{1+D}
$$

도 모두 무차원이고 양수이며 정규화되어 있다.

따라서 “무차원이므로 \(e^{-D}\)”는 증명이 아니다.

### 1.2 추가 원리: 연결 국소성

깊이 \(D_1\)인 경로 구간 뒤에 깊이 \(D_2\)인 독립 구간을 연결한다고
하자. 깊이는 가법적이고, 두 구간 사이에 추가 memory나 공통 잠재변수가
없다면

$$
D_{\mathrm{tot}}=D_1+D_2,
\qquad
S(D_1+D_2)=S(D_1)S(D_2)
$$

여야 한다. 즉 \(S\)는 가법 monoid
\((\mathbb R_{\ge0},+)\)에서 곱셈 monoid \(((0,1],\times)\)로 가는 양의
character다.

이 조건은 “독립성”이라는 추상 단어보다 강하다. 다음 실험량으로 직접
실패시킬 수 있다.

$$
\Delta_{\mathrm{comp}}(D_1,D_2)
:=
\log S(D_1+D_2)-\log S(D_1)-\log S(D_2).
$$

### 1.3 정리 S1

**정리 S1 (양의 생존 character).**  
\(S:\mathbb R_{\ge0}\to(0,1]\)가

1. \(S(0)=1\),
2. \(S(D_1+D_2)=S(D_1)S(D_2)\)

를 만족하면 어떤 \(\kappa\ge0\)에 대해

$$
S(D)=e^{-\kappa D}
$$

이다. 어떤 \(D_0>0\)에서 \(S(D_0)<1\)이면 \(\kappa>0\)이다.

**증명.** \(f(D)=-\log S(D)\)로 두면 \(f\ge0\)이고
\(f(D_1+D_2)=f(D_1)+f(D_2)\)다. 비음수 가법함수는
\(\mathbb R_{\ge0}\)에서 단조이므로 Cauchy 함수방정식의 선형해
\(f(D)=\kappa D\)만 남는다. 따라서 \(S(D)=e^{-\kappa D}\). \(\square\)

CE는 깊이의 단위를 “한 단위 깊이당 한 e-fold”로 잡아
\(\kappa=1\)로 쓴다. 이것은 함수형의 유도와 별개인 좌표/단위 정규화다.

### 1.4 게이트 결과

현재 감사 격자에서 composition residual은 다음과 같다.

| 후보 | \(\max|S(a+b)-S(a)S(b)|\) |
|---|---:|
| \(e^{-D}\) | \(2.78\times10^{-17}\) |
| \(e^{-D^2}\) | \(2.19\times10^{-1}\) |
| \((1+D)^{-1}\) | \(9.0\times10^{-2}\) |

즉 지수형은 단지 잘 맞는 후보가 아니라, 연결 국소성 gate를 정확히
통과하는 후보이다.

## 2. 루프 K: 왜 \(K(x)=1-x\)인가

### 2.1 반례 공간

외부 선택자가 없고 \(K(0)=1\), \(K(1)=0\)이어도

$$
K_1(x)=1-x,
\qquad
K_2(x)=(1-x)^2,
\qquad
K_3(x)=1-x^2
$$

가 모두 가능하다. 따라서 “가장 단순한 kernel”만으로는 유일성이 없다.

### 2.2 추가 원리: 완전 이분할과 혼합 호환성

생존과 억압이 서로 겹치지 않고 전체 경로량을 소진하는 이분할이면

$$
x+\sigma=1
$$

이다. 더 일반적으로 kernel \(K\)를 먼저 열어 둔 뒤, 서로 다른 두 경로
앙상블을 확률 \(t\)와 \(1-t\)로 무작위 혼합하는 연산과 suppression
측정이 교환한다고 요구한다.

$$
K(tx+(1-t)y)=tK(x)+(1-t)K(y),
\qquad 0\le t\le1.
$$

이 조건은 scalar \(x\)가 suppression을 결정하는 충분통계라는 선언이다.
비국소 memory나 숨은 상태가 있다면 이 gate는 실패한다.

### 2.3 정리 K1

**정리 K1 (affine complement uniqueness).**  
\(K:[0,1]\to[0,1]\)가

1. \(K(0)=1\),
2. \(K(1)=0\),
3. 모든 \(x,y,t\in[0,1]\)에 대해 혼합 호환성을 만족

하면

$$
\boxed{K(x)=1-x}
$$

가 유일하다.

**증명.** \(x=(1-x)0+x1\)로 쓰고 혼합 호환성을 적용하면

$$
K(x)=(1-x)K(0)+xK(1)=1-x.
$$

\(\square\)

이 정리는 모든 가능한 비국소 kernel을 배제하지는 않는다. 정확히는
“현재 생존율 하나가 충분통계이며, 무작위 ensemble 혼합과 호환되는 scalar
kernel”의 유일성을 증명한다.

### 2.4 게이트 결과

| 후보 | mixture-affinity residual |
|---|---:|
| \(1-x\) | \(2.22\times10^{-16}\) |
| \((1-x)^2\) | \(2.5\times10^{-1}\) |

따라서 기존의 “최소 선택”은 이제
`완전 이분할 + scalar sufficiency + mixture affinity` 아래의 조건부
유일 정리로 강화된다.

## 2A. 루프 X: 옆 공간의 타공간 재귀

### 2A.1 스칼라 충분통계의 한계

\(K(x)=1-x\)의 유일성은 각 채널 안의 국소 kernel에 관한 정리다.
서로 다른 공간의 상태가 하나뿐인 scalar \(x\)에 모두 들어간다는 뜻은
아니다. 옆 공간을 열면 생존상태는 벡터

$$
\boldsymbol x=(x_1,\ldots,x_n)
$$

이고, 재귀 깊이는 비음수 행렬 \(A\)가 된다.

### 2A.2 다형 Poisson fold-count 구성

조건부로 각 directed channel의 fold trigger를

$$
N_{i\leftarrow j}\mid x_j
\sim
\operatorname{Poisson}\!\left(A_{ij}(1-x_j)\right)
$$

라 두고, 서로 다른 \(j\)의 trigger stream이 독립이며 총 trigger가 0일
때만 공간 \(i\)가 생존한다고 하자. 독립 Poisson 합의 성질로

$$
\boxed{
x_i
=
\exp\!\left[-\sum_j A_{ij}(1-x_j)\right]
}
$$

가 나온다. 따라서 지수 생존과 선형 complement는 같은 미시가설에서
동시에 나온다.

- \(A_{ii}\): 자기재귀,
- \(A_{ij}\), \(i\ne j\): 타공간 재귀,
- \(A\)의 directed cycle: 실제 되먹임 고리

다. 큰 비대각 원소 하나만 있어도 되먹임이 생기는 것은 아니다. 되돌아오는
경로가 없는 DAG 결합은 nilpotent일 수 있고, 이때 재귀 임계값을 넘지
않는다.

### 2A.3 정리 X1: 최소 고정점과 Perron 임계값

\(F_A:[0,1]^n\to[0,1]^n\)를

$$
(F_A(\boldsymbol x))_i
=
\exp\!\left[-\sum_jA_{ij}(1-x_j)\right]
$$

로 둔다. 이는 유한 다형 Poisson branching process의 확률생성함수다.
그러므로

1. \(\boldsymbol 1\)은 항상 고정점이다.
2. \(\boldsymbol x^{(0)}=\boldsymbol0\),
   \(\boldsymbol x^{(m+1)}=F_A(\boldsymbol x^{(m)})\)는 단조 증가하여
   성분별 최소 고정점 \(\boldsymbol q\)로 수렴한다.
3. \(\boldsymbol q\)는 branching extinction probability 벡터다.
4. \(A\)가 irreducible이면
   \(\rho(A)\le1\)에서 \(\boldsymbol q=\boldsymbol1\)이고,
   \(\rho(A)>1\)에서 \(\boldsymbol q<\boldsymbol1\)인 비자명 가지가
   생긴다.
5. 고정점의 선형 안정성은

$$
J(\boldsymbol x^*)=\operatorname{diag}(\boldsymbol x^*)A,
\qquad
\rho(J)<1
$$

로 판정한다. 기존의 \(Dx^*<1\)은 이 식의 \(1\times1\) 경우다.

reducible \(A\)에서는 strongly connected component별로 임계값을
판정해야 하며, 일부 성분만 \(\boldsymbol q_i<1\)일 수도 있다.

### 2A.4 언제 다시 스칼라식이 되는가

대각 부분공간

$$
\Delta=\{\boldsymbol x:x_1=\cdots=x_n=x\}
$$

가 \(F_A\) 아래 불변일 필요충분조건은 모든 행합이 같은 것이다.

$$
\sum_jA_{ij}=D
\quad\text{for every }i.
$$

이때만

$$
x_i=x
\quad\Longrightarrow\quad
x=e^{-D(1-x)}
$$

로 정확히 축약된다. 행합이 다르면 하나의 \(D_{\mathrm{eff}}\)와 하나의
\(x\)로 줄이는 순간 타공간 정보를 잃는다. 즉 기존 CE 스칼라식은
“재귀가 하나뿐”이라는 명제가 아니라 `equal-row-sum invariant sector`
또는 실제 1종류 모형이다.

### 2A.5 실행 gate

| 결합 \(A\) | \(\rho(A)\) | 최소 고정점 | 의미 |
|---|---:|---:|---|
| \(\begin{psmallmatrix}0&1.8\\1.8&0\end{psmallmatrix}\) | 1.8 | \((0.26757,0.26757)\) | 자기항 없이 타공간 재귀만으로 초임계 |
| \(\begin{psmallmatrix}1.6&0.9\\0.3&1.2\end{psmallmatrix}\) | 1.95678 | \((0.14219,0.35767)\) | 비대칭 결합, scalar 축약 불가 |
| \(\begin{psmallmatrix}0&5\\0&0\end{psmallmatrix}\) | 0 | \((1,1)\) | 큰 일방향 영향이나 닫힌 재귀 없음 |

실행 코드는
`reality_stone/python/reality_stone/clarus/multispace_bootstrap.py`다.
아직 열린 물리 문제는 어떤 공간/종류를 index \(i\)로 삼고 실제
action/Hessian에서 \(A_{ij}\)를 어떻게 계산하는가다.

### 2A.6 연속 옆 공간

행렬식은 격자화된 표현이다. 연속 공간 \(\Sigma\)에서는 양의 적분 kernel
\(\mathcal A(r,r')\)를 써서

$$
x(r)
=
\exp\!\left[
-\int_\Sigma
\mathcal A(r,r')\{1-x(r')\}\,d\mu(r')
\right]
$$

로 올라간다. 격자 가중치까지 포함한
\(A_{ij}\simeq\mathcal A(r_i,r_j)w_j\)가 현재 코드의 행렬이다.
공간적으로 균일한 scalar sector가 정확하려면 row integral

$$
\int_\Sigma\mathcal A(r,r')\,d\mu(r')=D
$$

가 \(r\)에 무관해야 한다. 주기적 균일 격자는 이 조건을 만족하지만,
경계가 있는 열린 격자는 일반적으로 위치별 생존 profile을 남긴다.

### 2A.7 \(d+\delta\)의 spectral 강화 후보

타공간 관점은 \(D_{\mathrm{eff}}=d+\delta\)에도 새 유도 경로를 준다.
\(B\)가 비음수 normalized transfer operator라서

$$
B\boldsymbol1=\boldsymbol1
$$

이고, 전체 재귀행렬을

$$
A=dI+\delta B
$$

로 두면

$$
A\boldsymbol1=(d+\delta)\boldsymbol1.
$$

또한 \(B\)가 row-stochastic이므로 \(\rho(B)=1\)이고, 비음수
\(d,\delta\)에서 Perron 모드는

$$
\boxed{\rho(A)=d+\delta}
$$

다. 따라서 균일 최소 고정점은 정확히

$$
x=e^{-(d+\delta)(1-x)}
$$

를 만족한다. 두 공간이 서로 교환되는 최소 예

$$
B=
\begin{pmatrix}
0&1\\
1&0
\end{pmatrix}
$$

에서는 대칭 모드 깊이가 \(d+\delta\), 반대칭 모드 깊이가
\(d-\delta\)다. 기존 스칼라 계산은 대칭 Perron 모드만 읽은 것이다.

이 구성의 장점은 \(+\delta\)의 단위계수가 transfer normalization에서
나온다는 점이다. 그러나 \(B\)와 \(\delta\)를 실제 CE+SM quadratic
operator에서 유도하지 않으면 여전히 조건부 구성이다. 특히 다른 고유모드
\(d+\delta\lambda_k(B)\)의 물리적 역할은 스칼라 일치 하나로 제거되지
않는다.

## 3. 루프 F: 고정점과 가지 선택

정리 S1과 K1을 결합하면

$$
\boxed{x=e^{-D(1-x)}}
$$

가 나온다. 이 식의 수치해를 찾는 것보다 먼저 모든 가지를 열거해야 한다.

### 3.1 정리 F1

\(D>1\)에서 \([0,1]\) 안의 실수 고정점은 두 개다.

$$
x_{\mathrm{low}}
=-\frac{W_0(-De^{-D})}{D},
\qquad
x_{\mathrm{id}}
=-\frac{W_{-1}(-De^{-D})}{D}=1.
$$

반복 사상 \(F_D(x)=e^{-D(1-x)}\)의 고정점 multiplier는

$$
F_D'(x^*)=Dx^*
$$

이다. 따라서

- \(x_{\mathrm{low}}<1/D\): 안정 수축 가지,
- \(x_{\mathrm{id}}=1\): \(D>1\)에서 불안정 가지

다.

가지 선택은 관측값에 가까운 해를 고르는 규칙이 아니라,
`stable non-trivial branch`라는 사전 고정된 동역학 규칙으로 닫을 수 있다.

### 3.2 민감도

비자명 가지에서

$$
\boxed{
\frac{dx}{dD}
=-\frac{x(1-x)}{1-Dx}
}
$$

이다. 분모 \(1-Dx\)는 동시에 고정점의 조건수와 분기점까지의 거리를
측정한다. 현재 후보 \(D=3.1777573\), \(x=0.0486468\)에서는

$$
Dx\simeq0.15459
$$

로 분기점에서 충분히 떨어져 있다.

### 3.3 무엇이 아직 물리가 아닌가

여기까지는 \(x\)라는 무차원 고정점에 대한 수학이다.

$$
x\leftrightarrow\Omega_b
$$

는 아직 별도 관측 bridge다. 고정점 residual이 작다는 사실은 이 bridge를
증명하지 않는다.

## 4. 루프 H: \(d=3\)의 최소 재귀 타입 폐쇄

### 4.1 용어 교정

\(*:\Lambda^2\to\Lambda^2\)인 진짜 2-form self-duality는 \(d=4\)에서
나온다. CE가 필요한 것은 self-duality가 아니라

$$
*:\Lambda^2V^*\longrightarrow\Lambda^1V^*
$$

형태의 **type closure**다.

### 4.2 추가 원리: 무추가구조 재귀 폐쇄

공간의 한 점에서 두 경로 접선은 bivector
\(u\wedge v\in\Lambda^2V\)를 만든다. 이 접힘 결과가 다음 재귀 단계의
입력과 같은 vector/covector 타입으로 돌아가야 한다고 요구한다.

조건은 다음과 같다.

1. 국소적이고 선형이다.
2. 공간 회전에 대해 등변적이다.
3. 계량과 orientation 외의 배경 텐서를 넣지 않는다.
4. 출력은 다시 하나의 접선/covector 채널이다.

### 4.3 정리 H1

Hodge 별표는

$$
*:\Lambda^2V^*\to\Lambda^{d-2}V^*
$$

이므로 출력이 \(\Lambda^1V^*\)이려면

$$
d-2=1
\quad\Longrightarrow\quad
\boxed{d=3}.
$$

동일하게

$$
\dim\Lambda^2V^*=\frac{d(d-1)}2=\dim V^*=d
$$

의 비자명 양의 정수해도 \(d=3\)뿐이다.

\(d=7\)의 vector cross product는 추가 \(G_2\) 3-form 구조를 선택해야
하므로 3번 조건을 만족하지 않는다. 이를 “비결합적이어서 비물리적”이라고
배제해서는 안 된다.

이 정리의 정확한 범위는 다음과 같다.

- `Exact conditional`: 위 재귀 타입 폐쇄 요구 아래 \(d=3\).
- 아직 별도 원리: 자연의 fold response가 왜 이 타입 폐쇄를 요구하는가.
- 귀결되지 않음: \(N_c=d\), \(N_{\mathrm{gen}}=d\), 표준모형 게이지군.

후자의 동일시는 별도 동역학 bridge가 필요하다.

## 5. 루프 D: \(D_{\mathrm{eff}}=d+\delta\)

### 5.1 닫힌 전자약 대수

EWSB 중성 질량행렬을 \(M_Z^2\)로 정규화하면, 물리적으로 구별된
\(W^3/B\) gauge subspace에서

$$
\widehat{\mathcal M}^2
=
\begin{pmatrix}
\cos^2\theta_W & -\sin\theta_W\cos\theta_W\\
-\sin\theta_W\cos\theta_W & \sin^2\theta_W
\end{pmatrix}
$$

이다. 이는 trace 1, determinant 0인 rank-one projector다. 유일한
비대각 cross-channel의 정규화된 intensity는

$$
\boxed{
\delta
=
\left|\widehat{\mathcal M}^2_{12}\right|^2
=
\sin^2\theta_W\cos^2\theta_W
}
$$

이다.

이 대수는 SM EWSB를 조건으로 정확하다. 다만 임의 기저회전에 불변인
“raw off-diagonal 값”이라는 뜻은 아니다. \(W^3\)와 \(B\)라는 물리적으로
지정된 gauge subspace projector 사이의 coherence로 읽어야 한다.

### 5.2 additive channel construction

CE의 현재 강화 후보는 normalized fold-depth operator를

$$
\mathcal D_{\mathrm{fold}}
=
I_d\oplus C_Z^\dagger C_Z
$$

로 두는 것이다. 각 독립 공간 채널은 단위 intensity를 한 번씩 기여하고,
중성 cross-channel은 정규화된 quadratic intensity를 한 번 기여한다.
그러면 trace additivity로

$$
\operatorname{Tr}\mathcal D_{\mathrm{fold}}
=d+\|C_Z\|_{\mathrm{HS}}^2
=d+\delta
$$

가 나온다.

이 구성은 임의의 계수 \(c\)를 사후 삽입하지 않는 명시적 operator
realization이다. 코드 gate는 다음을 확인한다.

- normalized neutral matrix의 trace \(=1\),
- determinant \(=0\),
- off-diagonal intensity \(=\delta\),
- \(g\leftrightarrow g'\) 교환 불변,
- 독립 channel 순서에 대한 합 불변.

타공간 재귀를 명시하면 같은 덧셈 구조를 spectral하게 쓸 수도 있다.
정규화된 이웃 전달 연산자 \(B\boldsymbol1=\boldsymbol1\)에 대해

$$
A=dI+\delta B
$$

로 두면 균일 Perron 모드의 고유깊이가 \(d+\delta\)다. 이 경우
\(D_{\mathrm{eff}}\)는 trace가 아니라 재귀 네트워크의 지배 고유값이다.
두 구성은 아직 서로 독립적인 증거가 아니라, 실제 Hessian이 어느 operator
functional을 선택하는지를 묻는 경쟁적 강화 후보다.

### 5.3 다음 증명 병목

아직 남은 핵은 이 operator를 정의하는 것이 아니라 CE+SM의 실제 quadratic
Hessian에서 동일한 block이 나오는지다. 일반형을 먼저 열면

$$
D_{\mathrm{eff}}
=d+c_1\delta+c_2\delta^2+\cdots
$$

이고, 다음 루프는 아래 계산으로 \(c_1=1\), \(c_{n\ge2}\)의 억제를
도출해야 한다.

1. CE master action과 SM EWSB action의 quadratic fluctuation operator 작성.
2. spatial, neutral mixing, charged, fermion block을 모두 열거.
3. gauge fixing과 ghost를 포함한 normalized trace 또는 heat-kernel functional 고정.
4. Ward identity 아래 gauge/scheme 독립성 검사.
5. \(\partial D_{\mathrm{eff}}/\partial\delta|_{\delta=0}=1\)인지 계산.
6. \(W^\pm\), loop correction, cross block이 사라지거나 별도 항으로 남는지 확인.
7. matching scale을 관측 비교 전에 고정.

이 단계가 성공하면 \(d+\delta\)는 “잘 맞는 ansatz”에서
“명시적 Hessian의 spectral trace”로 승격된다. 실패하면 실패한 block이
정확히 어떤 추가 항을 요구하는지 다음 루프의 입력이 된다.

## 6. 현재 의존성 그래프

$$
\begin{array}{c}
\text{path concatenation locality}
\Longrightarrow S(D)=e^{-\kappa D}
\\[2mm]
\text{binary exhaustivity + mixture affinity}
\Longrightarrow K(x)=1-x
\\[2mm]
\Downarrow
\\[-1mm]
\boldsymbol x
=
\exp[-A(\boldsymbol1-\boldsymbol x)]
\Longrightarrow
\text{minimal extinction branch; threshold }\rho(A)=1
\\[2mm]
\text{equal row sums of }A
\Longrightarrow
x=e^{-D(1-x)}
\\[3mm]
\text{minimal Hodge type closure}
\Longrightarrow d=3
\\[2mm]
\text{SM neutral coherence}
\Longrightarrow \delta=s_W^2c_W^2
\\[2mm]
\text{quadratic additive fold operator}
\Longrightarrow D_{\mathrm{eff}}=d+\delta
\\[2mm]
\Downarrow
\\[-1mm]
x_{\mathrm{low}}\simeq0.0486468
\\[3mm]
\text{observable identification}
\stackrel{\text{separate bridge}}{\Longrightarrow}
\Omega_b
\end{array}
$$

이 그래프에서 마지막 화살만 관측 bridge다. 앞의 화살도 모두 무가정인
것은 아니지만, 각 가정이 어떤 반례군을 제거하는지가 이제 명시되어 있다.

## 7. 실행 결과와 해석

현재 실행 결과:

```text
composition residuals
  exponential      2.77555756156e-17
  stretched_p2     0.219253242796
  power            0.09
mixture-affinity residuals
  complement       2.22044604925e-16
  powered_p2       0.25
conditional core chain
  EW coherence intensity  0.1777573116
  effective depth         3.1777573116
  low fixed point         0.0486467805076
  low stability           0.154587662444
  identity stability      3.1777573116
  inverse sin^2(theta)    0.23122
  bivector->vector d      (3,)
```

해석:

- 지수형과 선형 complement는 각각의 구조 gate에서 대조군을 제거한다.
- 작은 고정점은 안정하고 \(x=1\)은 같은 \(D>1\)에서 불안정하다.
- 자기항이 없어도 reciprocal cross-space cycle의 \(\rho(A)>1\)이면
  비자명 최소 고정점이 생긴다.
- 기존 스칼라식은 \(A\)의 행합이 같은 불변 부분공간에서만 정확하다.
- EWSB coherence를 한 번 세는 conditional chain은 정방향과 역방향이
  수치적으로 닫힌다.
- 이 결과는 아직 \(\Omega_b\) 식별을 채점하지 않는다.

## 8. 다음 루프

우선순위는 다음과 같다.

1. `A-loop`: 실제 action/Hessian에서 자기·타공간 결합
   \(A_{ij}\)와 strongly connected components를 계산.
2. `D-loop`: scalar invariant sector가 존재할 경우 CE+SM quadratic
   Hessian에서 공통 행합과 unit coefficient를 계산.
3. `variance-loop`: \(\langle e^{-\Phi}\rangle\)와
   \(e^{-\langle\Phi\rangle}\) 사이의 cumulant/variance bound를 실제 모형에서 계산.
4. `bridge-loop`: \(\boldsymbol x\)의 어느 projection이
   \(\Omega_b\)인지 대안 물질화 channel과 독립 데이터에서 비교.
5. `model-selection loop`: \(S,K,A\) 후보군과 관측 역할
   (`input/calibration/selection/confirmation/prospective`)을 manifest로
   사전등록.

각 루프는 식을 더 복잡하게 만드는 것이 목적이 아니다. 대안 공간을 먼저
열고, 새로운 원리가 실제로 그 공간을 줄이는 경우에만 코어에 남긴다.
