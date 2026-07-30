# CE 코어 강화 루프

상태: `Working canonical / loop 3 complete, Q-loop structural gate active`

범위: 생존함수, 자기·타공간 재귀, 공간 차원 선택, 전자약 유효 깊이,
양자 jump에서 고전 분지과정으로의 조건부 축약

코드 게이트: `core_axioms.py`, `multispace_bootstrap.py`,
`core_model_selection.py`, `quantum_jump_bridge.py`

실행:

```powershell
python examples/physics/core_axiom_loop.py
python examples/physics/multispace_recursion_gate.py
python examples/physics/core_model_selection_gate.py
python examples/physics/quantum_jump_bridge_gate.py
```

회귀 검증:

```powershell
python -m pytest -q tests/test_core_axioms.py tests/test_multispace_bootstrap.py tests/test_core_model_selection.py tests/test_quantum_jump_bridge.py
```

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
- \(A_{ii}\): 종류 \(i\)의 자기재귀 깊이,
- \(A_{ij}\;(i\ne j)\): 종류 \(i\)에서 시작해 다음 세대 종류 \(j\)로
  가는 타공간 재귀 깊이,
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

먼저 화살표 규약을 고정한다. 이 문서와 코드는 **행이 출발 종류**다.
즉 \(A_{ij}\)는 종류 \(i\) trigger 하나가 만드는 다음 세대 종류 \(j\)
trigger의 평균 개수다. 물리적 incoming-influence 행렬을 행에 도착 종류로
적었다면 여기의 \(A\)는 그 전치행렬이다.

조건부로 잠재적인 다음 세대 fold trigger 수를

$$
N_{j\leftarrow i}
\sim
\operatorname{Poisson}(A_{ij})
$$

라 두자. 종류 \(j\)에서 시작한 후속 cascade가 소멸하지 않을 확률은
\(1-x_j\)다. Poisson thinning에 의해 소멸하지 않는 자식 cascade 수는
평균 \(A_{ij}(1-x_j)\)인 Poisson 변수가 된다. 서로 다른 \(j\)의 stream이
독립이고 이 활성 자식 cascade가 모두 0일 때 종류 \(i\) cascade가
소멸한다. 따라서

$$
\boxed{
x_i
=
\exp\!\left[-\sum_j A_{ij}(1-x_j)\right]
}
$$

가 나온다. 여기서 \(x_i\)의 엄밀한 branching 의미는 **종류 \(i\)에서
시작한 fold-trigger cascade의 최종 소멸확률**이다. 이를 물리 경로의
zero-trigger 생존율로 읽는 단계는 별도 식별 bridge다. 이 조건 아래
지수형과 선형 complement는 같은 미시가설에서 동시에 나온다.

- \(A_{ii}\): 자기재귀,
- \(A_{ij}\), \(i\ne j\): \(i\to j\) 타공간 재귀,
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
5. 고정점의 Jacobian은

$$
J(\boldsymbol x^*)=\operatorname{diag}(\boldsymbol x^*)A,
$$

이고 선형 안정성은 다음 세 경우로 나뉜다.

   - \(\rho(J)<1\): 국소 점근 안정,
   - \(\rho(J)>1\): 불안정,
   - \(\rho(J)=1\): 선형화만으로 판정할 수 없는 임계 경우.

기존의 \(Dx^*<1\)은 첫 조건의 \(1\times1\) 경우다. irreducible
supercritical Poisson 모형의 최소 소멸근에서는
\(\rho(\operatorname{diag}(\boldsymbol q)A)<1\)가 정리로 성립한다.
다만 이는 branching 세대 반복의 안정성이다. 물리적 시간 안정성과
동일시하려면 세대 갱신이 실제 시간진화를 나타낸다는 bridge가 더 필요하다.

reducible \(A\)에서는 각 strongly connected component의 내부
Perron 반지름뿐 아니라 도달가능성을 함께 봐야 한다. 행-출발 규약에서
\(q_i<1\)일 필요충분조건은 종류 \(i\)가 어떤 supercritical component에
도달할 수 있는 것이다. 자체 component가 subcritical이어도 downstream
supercritical component로 들어갈 수 있으면 \(q_i<1\)이다. 이 SCC와
upstream reachability를 solver가 명시적으로 계산한다.

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

로 정확히 축약된다. 행합이 다르면 이 **균일 대각 축약**은 성립하지
않고 하나의 \(D_{\mathrm{eff}}\)와 하나의 \(x\)로 줄이는 순간 타공간
정보를 잃는다. 다만 rank-one invariant curve나 equitable partition
같은 다른 저차원 축약까지 배제하는 정리는 아니다. 즉 기존 CE 스칼라식은
“재귀가 하나뿐”이라는 명제가 아니라 `equal-row-sum invariant sector`
또는 실제 1종류 모형이다.

### 2A.5 실행 gate

| 결합 \(A\) | \(\rho(A)\) | 최소 고정점 | 의미 |
|---|---:|---:|---|
| \(\begin{psmallmatrix}0&1.8\\1.8&0\end{psmallmatrix}\) | 1.8 | \((0.26757,0.26757)\) | 자기항 없이 타공간 재귀만으로 초임계 |
| \(\begin{psmallmatrix}1.6&0.9\\0.3&1.2\end{psmallmatrix}\) | 1.95678 | \((0.14219,0.35767)\) | 비대칭 결합, scalar 축약 불가 |
| \(\begin{psmallmatrix}0&5\\0&0\end{psmallmatrix}\) | 0 | \((1,1)\) | 큰 일방향 영향이나 닫힌 재귀 없음 |
| \([1]\) | 1 | \((1)\) | critical Poisson class를 해석적으로 종료 |
| \(\operatorname{diag}(2,1)\) | 2 | \((0.20319,1)\) | 초임계와 임계 class가 분리된 부분 생존 |
| \(\begin{psmallmatrix}0&1\\0&2\end{psmallmatrix}\) | 2 | \((0.45076,0.20319)\) | 자체 아임계 type도 초임계 class에 도달하면 비자명 |

실행 코드는
`reality_stone/python/reality_stone/clarus/multispace_bootstrap.py`다.
solver는 SCC 도달가능성으로 확실히 소멸하는 type을 먼저 고정하고,
나머지 최소 가지를 0에서 시작하는 safeguarded Newton으로 계산한다.
따라서 정확한 임계 class의 느린 Picard 수렴과 near-critical 정체를
분리해 처리한다.
아직 열린 물리 문제는 어떤 공간/종류를 index \(i\)로 삼고 실제
action의 vertex와 환경 spectral density에서 CP jump process를 거쳐
\(A_{ij}\)를 어떻게 계산하는가다.

### 2A.6 연속 옆 공간

행렬식은 격자화된 표현이다. 연속 공간 \(\Sigma\)에서는 가측인 양의
적분 kernel \(\mathcal A(r,r')\)를 쓰고, 각 행 적분이 본질적으로
유계라고 가정하여

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
연속형에서 Perron 임계정리까지 쓰려면 양의 적분연산자의 compactness와
irreducibility 같은 조건을 추가해야 한다.

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
\(d,\delta\)에서 균일 Perron 고유벡터의 고유깊이는

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
않는다. \(B\)가 reducible이면 Perron eigenspace도 유일하지 않을 수
있다. 유일한 양의 Perron 방향에는 irreducibility, 엄격한 지배 모드에는
primitivity가 추가로 필요하다.

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

## 5A. 루프 Q: 복소 양자진폭에서 양의 Poisson 재귀까지

### 5A.1 수식 전 질문

환기구 비유는 연기 신호가 양의 사건 수라고 가정한다. 그러나 원래
경로적분은 복소 진폭이며 서로 강화되거나 상쇄된다. Hessian의 비대각
원소도 기저에 따라 부호와 크기가 바뀔 수 있다. 따라서 “비대각 원소를
제곱하면 바로 Poisson 재귀율”이라는 규칙은 일반 정리가 아니다.

이 루프의 질문은 다음이다.

> CE+SM의 복소·부호 있는 양자 동역학이 어떤 조건에서 완전양의 고전
> jump 과정으로 축약되고, 그 jump 과정이 어떤 추가 조건에서 독립
> Poisson genealogy가 되는가?

### 5A.2 먼저 열어 두는 반례

- 폐쇄된 두 준위 계는 지수감쇠가 아니라 coherent oscillation을 보인다.
- 매우 짧은 시간의 양자 생존확률은 일반적으로 시간의 이차항부터
  변하므로 constant-hazard 지수형과 다르다.
- 유한하거나 구조화된 bath는 revival과 memory backflow를 만들 수 있다.
- 공통 bath의 collective jump는 서로 다른 type의 count를 상관시킨다.
- 같은 master equation도 측정 unraveling에 따라 서로 다른 jump record를
  가질 수 있다.
- off-diagonal mass mixing은 고유기저에서 대각화될 수 있으므로 원소의
  제곱 자체는 기저불변 decay rate가 아니다.

이 반례들은 CE action을 바로 폐기하지 않는다. 대신 현재 Poisson
고정점으로 가는 중간 bridge가 성립하는지를 검사한다.

### 5A.3 조건부 축약 사슬

강화 후보는 다음 객체들을 건너뛰지 않는다.

$$
\boxed{
\text{CE+SM action}
\to
\text{physical correlator/spectral density}
\to
\text{CP reduced dynamics}
\to
\text{classical jump rates }W
\to
\text{next-generation matrix }A
}
$$

첫 번째 화살에는 물리적 system–environment 분할, 초기상관 제어,
빠르게 감쇠하는 bath correlation, 약결합 시간척도와 secular
coarse-graining이 필요하다. gauge와 ghost를 제거한 물리 Hilbert
공간에서 이 조건들이 성립하면 reduced generator를 GKSL 형태로 쓸 수
있고, Kossakowski spectral matrix의 양의 고유값이 jump eigenrate가 된다.

“Born–Markov”라는 이름만으로는 충분하지 않다. 완전양성을 잃는 비세속
근사도 있으므로 Choi positivity와 trace preservation을 직접 gate로 둔다.

### 5A.4 no-jump 지수형의 정확한 범위

jump operator를 \(L_r\)라 하고

$$
\Gamma=\sum_rL_r^\dagger L_r
$$

라 쓰면 no-jump 생존확률은 누적 hazard의 지수로 항상 표현할 수 있다.
그러나 시간에 대한 단순 지수형은 조건부 no-jump 상태에서 hazard가
상수일 때만 정확하다. 충분조건 후보는 선택한 sector의 projector \(P\)에
대해

$$
P\Gamma P=\kappa P
$$

가 성립하는 것이다.

따라서 \(D\)를 실제 누적 hazard인 optical depth로 정의하면
\(S=e^{-D}\)는 정확하다. 반면 정적 channel count \(d+\delta\)가 이
optical depth와 같다는 명제는 여전히 별도 유도다.

### 5A.5 양자 jump에서 고전 type rate로

decoherence가 선택한 직교 type basis를 \(\{|i\rangle\}\)라 하자.
population과 coherence가 닫혀 분리될 때 row-source 고전 rate를

$$
W_{ij}
=
\sum_r
\left|\langle j|L_r|i\rangle\right|^2,
\qquad i\ne j
$$

로 정의할 수 있다. 여기서도 행 \(i\)가 출발 type이고 열 \(j\)가
도착 type이다.

dephasing projection을 \(\mathcal P\)라 할 때 정확한 고전 폐쇄에는

$$
(I-\mathcal P)\mathcal L\mathcal P=0,
\qquad
\mathcal P\mathcal L(I-\mathcal P)=0
$$

가 필요하다. 근사 폐쇄라면 두 leakage norm과 decoherence gap으로 오차를
제어해야 한다.

연속시간 전이율 \(W\)는 offspring 기대행렬 \(A\)와 같은 객체가 아니다.
부모 type \(i\)의 수명과 출산률을 포함해

$$
A_{ij}
=
\int_0^\infty
S_i(t)b_{ij}(t)\,dt
$$

같은 next-generation 정의가 추가로 필요하다. 개체별 독립성, reset,
세대별 동일 법칙, 자원 비경쟁성이 깨지면 현재 단순 Poisson branching이
아니라 renewal, Hawkes 또는 compound-Poisson 후보로 이동한다.

### 5A.6 CE portal에서 바로 드러나는 조건

현재 정본의 \(Z_2\) 보존과 \(v_\Phi=0\) 아래에서

$$
-\lambda_{\mathrm{HP}}|H|^2\Phi^2
=
-\frac{\lambda_{\mathrm{HP}}}{2}(v+h)^2\Phi^2
$$

를 진공에서 전개하면 \(h\)-\(\Phi\) quadratic cross-Hessian은 0이다.
portal은 \(\Phi\) 질량을 바꾸지만 선형 \(h\leftrightarrow\Phi\) mixing
block을 만들지 않는다.

따라서 실제 CE–SM rate는 quadratic Hessian 하나가 아니라
\(h\Phi^2\), \(h^2\Phi^2\) vertex, loop self-energy, physical cut와
phase space에서 계산해야 한다. 특히 한 사건이 \(\Phi\) 두 개를 만드는
과정은 offspring가 한 개씩 독립 발생하는 모형이 아니라 batch가 있는
compound-Poisson 후보가 된다.

전자약 \(W^3/B\) 비대각 질량항도 coherent mixing이다. 그것의 제곱을
곧바로 decay rate로 읽지 않고, 물리 고유상태, on-shell matrix element,
phase space와 흡수 self-energy를 거쳐야 한다.

### 5A.7 실행 gate

| Gate | 계산 대상 | 통과 조건 |
|---|---|---|
| `Q0-action` | quadratic operator와 cubic/quartic vertex | pole spectrum·Ward/BRST 통과, \(Z_2\) 진공의 cross-Hessian 0 재현 |
| `Q1-spectral` | bath correlator, spectral matrix, 흡수 self-energy | Hermitian residual 작음, spectral matrix 양의 준정부호, gauge/scheme 안정 |
| `Q2-Markov` | bath correlation tail과 시간척도 분리 | coarse window에서 exact reduced map과 CP semigroup의 holdout 오차 통과 |
| `Q3-CP` | finite-time reduced map의 Choi matrix와 trace | Choi 최소고유값과 trace residual이 사전등록 허용치 안 |
| `Q4-no-jump` | 생존곡선과 hazard | constant-hazard가 holdout에서 대안 Weibull/renewal보다 선택되거나 invariant-sector 정리 통과 |
| `Q5-classical` | population–coherence leakage | 두 leakage norm이 목표 정밀도보다 작고 고전 population 예측이 holdout 통과 |
| `Q6-offspring` | lineage, batch size, factorial cumulant, cross-covariance | 단위 batch, Fano 1, 고차 factorial cumulant 0, type stream 조건부 독립 |
| `Q7-branching` | 여러 초기 type의 extinction trajectory | training에서 정한 \(A\)로 독립 Monte Carlo extinction을 예측 |
| `Q8-robustness` | gauge, scale, pointer grouping, coarse window | \(A\), Perron 반지름, 최소 고정점이 사전등록 범위 안에서 안정 |

현재 최소 실행 게이트
`reality_stone/python/reality_stone/clarus/quantum_jump_bridge.py`는 이미
주어진 후보 객체에 한해 다음을 감사한다.

- Kossakowski 행렬의 Hermitian/양의 준정부호 조건
- 선택한 type 기저에서 population–coherence 양방향 leakage
- no-jump sector의 불변성과 상태 독립 constant hazard
- \(W_{ij}=\sum_r|\langle j|L_r|i\rangle|^2\)의 행-출발 방향
- 별도로 주어진 offspring birth rate와 평균수명에서
  \(A_{ij}=\tau_i b_{ij}\)로 가는 행별 변환

이는 `Q1` 후단의 대수 조건, `Q4`의 불변 sector 충분조건, `Q5`의 정확
폐쇄 조건을 실행 가능하게 만든 최소 구조 게이트다. `Q0`의 CE+SM 작용
계산, `Q1`의 물리 spectral density, `Q2-Q3`의 reduced-map 유도,
`Q6-Q8`의 독립 계보·holdout·강건성은 아직 구현되지 않았다. 따라서
게이트가 통과해도 보고서는
`ce_sm_derivation_complete=False`,
`poisson_branching_derived=False`를 유지한다.

Q-loop의 현재 판정은
`Open physical derivation / executable conditional structural gates`다.
이 루프가 실패하면 action 전체가 자동 기각되는 것이 아니라, Poisson
bridge를 density-matrix, compound-Poisson 또는 memory-kernel 모형으로
교체한다.

## 6. 현재 의존성 그래프

한 줄짜리 유도처럼 보이면 열린 물리 bridge와 닫힌 수학 정리가 섞인다.
따라서 세 사슬로 나누어 읽는다.

### 6.1 물리적 생성 사슬 — 현재 열려 있음

$$
\boxed{
\text{CE+SM action}
\xrightarrow[\text{open}]{Q0-Q1}
\text{physical spectral density}
\xrightarrow[\text{open}]{Q2-Q5}
\text{classical rate }W
\xrightarrow[\text{open}]{Q6}
\text{offspring matrix }A
}
$$

현재 코드는 이 사슬의 입력이 주어졌을 때 필요한 구조만 검사한다. 사슬의
화살 자체를 CE+SM에서 유도한 것은 아니다.

### 6.2 \(A\)가 주어진 뒤의 수학 사슬

$$
\left.
\begin{array}{l}
\text{path concatenation locality}
  \Rightarrow S(D)=e^{-\kappa D}\\
\text{binary exhaustivity + mixture affinity}
  \Rightarrow K(x)=1-x\\
A\ge0\text{ and independent Poisson offspring}
\end{array}
\right\}
\Longrightarrow
\boldsymbol x
=
\exp[-A(\boldsymbol1-\boldsymbol x)].
$$

이 조건 아래 최소 소멸 고정점과 SCC 도달가능성을 계산할 수 있고,
irreducible \(A\)의 임계값은 \(\rho(A)=1\)이다.

### 6.3 균일 스칼라 특수화와 관측

$$
\begin{array}{c}
\text{equal row sums of }A
\Longrightarrow
x=e^{-D(1-x)}
\\[2mm]
\left.
\begin{array}{l}
\text{minimal Hodge type closure}\Rightarrow d=3\\
\text{declared SM neutral subspace}\Rightarrow\delta=s_W^2c_W^2\\
\text{normalized additive transfer ansatz}
\end{array}
\right\}
\Longrightarrow
D_{\mathrm{eff}}=d+\delta
\quad\text{(conditional)}
\\[2mm]
\Downarrow
\\[-1mm]
x_{\mathrm{low}}\simeq0.0486468
\xrightarrow[\text{separate observational bridge}]{}
\Omega_b
\end{array}
$$

여기서는 \(D_{\mathrm{eff}}=d+\delta\)로 가는 화살과
\(x_{\mathrm{low}}\leftrightarrow\Omega_b\)로 가는 화살이 모두 별도
bridge다. 앞의 조건부 정리와 같은 증명 상태로 읽어서는 안 된다.

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

model-selection gate
  candidate_count         27
  algebraic_status        PASS
  selection_status        UNDERIDENTIFIED
  independent selection observations  1

quantum-jump structural gate
  scope                   conditional_quantum_jump_structure_only
  status                  STRUCTURAL_CONDITIONAL_PASS
  CE+SM derivation        False
  Poisson branching       False
  coherent leakage counterexample  2
  collective leakage counterexample  1
```

해석:

- 지수형과 선형 complement는 각각의 구조 gate에서 대조군을 제거한다.
- 작은 고정점은 안정하고 \(x=1\)은 같은 \(D>1\)에서 불안정하다.
- 자기항이 없어도 reciprocal cross-space cycle의 \(\rho(A)>1\)이면
  비자명 최소 고정점이 생긴다.
- 기존 스칼라식은 \(A\)의 행합이 같은 균일 대각 불변 부분공간에서
  정확하다. 다른 저차원 invariant manifold의 가능성은 별도다.
- EWSB coherence를 한 번 세는 conditional chain은 정방향과 역방향이
  수치적으로 닫힌다.
- 이 결과는 아직 \(\Omega_b\) 식별을 채점하지 않는다.
- 사전등록된 스칼라 후보 27개는 대수적으로 계산 가능하지만 독립 selection
  관측이 하나뿐이므로 현재 모형 선택 판정은 `UNDERIDENTIFIED`다.
- 양자 jump 구조 게이트의 `PASS`는 입력된 고전 cycle 예제가 조건을
  만족한다는 뜻이다. coherent Hamiltonian과 collective jump 반례는 같은
  게이트에서 실제로 실패하며, CE+SM 유도 완료 플래그는 의도적으로
  `False`다.

## 8. 다음 루프

우선순위는 다음과 같다.

1. `Q-loop`: 현재 구조 게이트의 입력을 CE+SM action, interaction
   vertex, spectral density에서 실제로 계산하고, decoherence와
   coarse-graining을 거쳐 비음수 jump rate 및 독립 offspring가
   나오는지 검증.
2. `A-loop`: 위 양의 rate가 존재할 경우 실제 action/Hessian에서 자기·타공간 결합
   \(A_{ij}\)와 strongly connected components를 계산.
3. `D-loop`: homogeneous scalar invariant sector가 존재할 경우 CE+SM quadratic
   Hessian에서 공통 행합과 unit coefficient를 계산.
4. `variance-loop`: \(\langle e^{-\Phi}\rangle\)와
   \(e^{-\langle\Phi\rangle}\) 사이의 cumulant/variance bound를 실제 모형에서 계산.
5. `bridge-loop`: \(\boldsymbol x\)의 어느 projection이
   \(\Omega_b\)인지 대안 물질화 channel과 독립 데이터에서 비교.
6. `vector model-selection loop`: 현재 사전등록된 scalar 후보군을
   \(A\)의 topology, boundary, reducibility 후보까지 확장하고
   `confirmation/prospective` 관측을 추가.

각 루프는 식을 더 복잡하게 만드는 것이 목적이 아니다. 대안 공간을 먼저
열고, 새로운 원리가 실제로 그 공간을 줄이는 경우에만 코어에 남긴다.
