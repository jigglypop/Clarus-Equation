# 12. 합성 face와 simplicity 블록 폐쇄

이 장은 `유한 블록 RG와 스파인 고정점`에서 남긴 두 의무 가운데

1. face가 **어디에 붙는가**,
2. local simplicity가 coarse block에서도 유지되는가

를 가장 작은 유한 모형에서 검산한다.

먼저 결론을 고정한다. composition face, finite bivector inverse, shared spacelike-face hard gluing은 선언한 정상ㆍtransportㆍmatching 조건 아래의 정리다. 반면 local simplicity만으로 glued geometry, incidence/simplicity만으로 unique actionㆍmeasure, finite flat data만으로 continuum Einstein--Hilbert/두 편극이 나온다는 세 부모 명제는 완전 반례로 삭제됐다. “full fixed point가 아직 열려 있다”는 말로 이 삭제된 함의를 보존하지 않는다.

새 Lorentzian continuum 이론을 제안하려면 cross/shape matching, actionㆍmeasure, refinement/RG limit, continuum criterion, EH limit, two-DOF criterion을 독립 공리 또는 정리로 제시해야 한다. 그것은 이 유한 결과의 자동 완성이 아니라 새 모형 계약이다.

## 12.1 face endpoint를 임의로 뽑지 않는다

fine causal relation을 edge

\[
u\longrightarrow m,\qquad m\longrightarrow v
\]

로 두고, 한 번의 block quotient가 이 두 step을 coarse continuation

\[
u\longrightarrow v
\]

로 표현한다고 하자.

그러면 fine factorization과 coarse edge의 동등성을 기록하는 최소 2-cell은 삼각형

\[
f=(u,m,v)
\]

이다. oriented boundary는

\[
\partial f
=
e_{um}+e_{mv}-e_{uv}
\]

이다.

따라서 현재 face attachment rule은

\[
\boxed{
(u\to m,\;m\to v,\;u\to v)
\Longrightarrow
\text{one composition face }(u,m,v)
}
\]

로 둔다.

이는 causal category의 nerve에서 composable morphism pair와 그 composite가 만드는
2-simplex의 block 제한이다. face endpoint를 별도 무작위 kernel로 선택하지 않는다.

### 조건부 정리 후보: Composition-Face Lemma

fine edge 집합 \(E_{\rm fine}\)과 coarse edge 집합 \(E_{\rm coarse}\)가 같은
acyclic causal order와 양립한다고 하자. 모든 factorization

\[
(u,m),(m,v)\in E_{\rm fine},
\qquad
(u,v)\in E_{\rm coarse}
\]

에 face \(f=(u,m,v)\)를 하나 대응시키면 다음이 성립한다.

1. attachment는 relabeling-equivariant하다.
2. 모든 face는 \(u\prec m\prec v\)인 causal cell이다.
3. face boundary는 fine path와 coarse path의 차이를 정확히 기록한다.
4. 동일 coarse edge의 여러 microscopic factorization은 그 edge를 공유하는
   triangle fan으로 표현된다.

face 수가 \(M\)인 local fan은

\[
V=M+2,\qquad E=2M+1,\qquad F=M
\]

이므로

\[
\chi=V-E+F=1
\]

인 contractible local disk다.

이 결과는 **face의 local attachment**를 정하지만, 여러 fan을 어떤 4차원
cell incidence로 glue할지는 아직 정하지 않는다.

## 12.2 Q-spine face count를 incidence로 읽기

11장의 임계 Q-spine에서 한 spine epoch당 face event 수는

\[
F_1\sim \operatorname{Poisson}(\mu),
\qquad
\mu=D-1.
\]

서로 독립인 \(b\)개의 spine epoch를 한 coarse edge block으로 묶으면

\[
\boxed{
F_b\sim\operatorname{Poisson}\!\left(b(D-1)\right)
}
\]

이다.

4차원 cellular/spin-foam 해석에서 coarse edge는 3-cell에 dual하다. 비퇴화
3차원 polyhedron은 최소 네 개의 face를 가져야 하므로 먼저

\[
F_b\ge4
\]

를 필요조건으로 검사한다.

CE benchmark

\[
D=3.1777584234
\]

에서

\[
\mu=D-1=2.1777584234
\]

이고

\[
P(F_b\ge4)
=
1-e^{-b\mu}
\sum_{k=0}^{3}\frac{(b\mu)^k}{k!}.
\]

수치는 다음과 같다.

| block depth \(b\) | \(\mathbb E[F_b]\) | \(P(F_b\ge4)\) |
|---:|---:|---:|
| 1 | 2.177758 | 0.176292 |
| 2 | 4.355517 | 0.632744 |
| 3 | 6.533275 | 0.890420 |
| 4 | 8.711034 | 0.973998 |
| 5 | 10.888792 | 0.994655 |

따라서 raw one-epoch object는 대부분 비퇴화 3-cell을 만들 face incidence가
부족하다.

\[
\boxed{
b_{95}=4,\qquad b_{99}=5
}
\]

가 각각 \(95\%\), \(99\%\) 이상의 `at least four faces` 조건을 처음 만족한다.

이 수치를 곧바로 플랑크 길이와 동일시하지 않는다. 다만 **최소 고전 렌더링
프레임은 한 microscopic event가 아니라 여러 Q-spine epoch의 block이어야
한다**는 구체적 진단을 준다.

## 12.3 exact simplicial valence no-go

4-simplex dual complex의 tetrahedral edge를 정확히 쓰려면 face valence가

\[
F=4
\]

로 고정되어야 한다.

그러나 Poisson 변수에 대해

\[
\sup_{\lambda>0}P(\operatorname{Poisson}(\lambda)=4)
\]

는 \(\lambda=4\)에서 달성되며

\[
\boxed{
e^{-4}\frac{4^4}{4!}
=
0.195366\ldots
}
\]

뿐이다.

따라서 어떤 block depth를 고르더라도 unconditioned independent-Poisson law가
exact tetrahedral valence에 높은 확률로 집중할 수 없다.

\[
\boxed{
\text{raw Poisson count}
\not\Longrightarrow
\text{simplicial 4D complex}
}
\]

이 강한 branch는 폐기한다.

살아 있는 선택지는 다음 둘이다.

1. exact simplicial sector를 별도 topology projector로 강하게 조건부화한다.
2. variable-valence 3-cells를 허용하는 general cellular/polyhedral spin foam을
   사용하고 \(F\ge4\), closure, simplicity, nondegeneracy를 요구한다.

현재 최소 branch는 두 번째를 우선한다.

## 12.4 finite Plebanski simplicity audit

이 장의 계산 코드는 full Lorentzian quantum amplitude 대신 Euclideanized local
self-dual algebra를 사용한다.

4차원 tetrad \(e^I\)에서 self-dual 2-form triple을

\[
\Sigma^i(e)
=
e^0\wedge e^i
+\frac12\epsilon^i{}_{jk}e^j\wedge e^k
\]

로 둔다.

일반 triple \(B^i\)에 대해

\[
X_{ij}[B]
:=
B^i\wedge B^j
\]

를 정의한다. Plebanski simplicity는

\[
\boxed{
X_{ij}
=
\frac{\operatorname{tr}X}{3}\delta_{ij}
}
\]

이다.

정규화 residual을

\[
\epsilon_{\rm simp}(B)
=
\frac{
\left\|
X-\frac{\operatorname{tr}X}{3}I
\right\|_F
}{
\|X\|_F
}
\]

로 둔다. 비퇴화 geometric triple \(\Sigma^i(e)\)에서는 machine precision까지

\[
\epsilon_{\rm simp}=0
\]

이 재현된다.

## 12.5 local simplicity는 block 아래 닫히지 않는다

두 local cell의 triple을 \(B^i,C^i\)라 하자. 각각 simple이면

\[
B^i\wedge B^j=v_B\delta^{ij},
\qquad
C^i\wedge C^j=v_C\delta^{ij}.
\]

block variable을

\[
Q^i=B^i+C^i
\]

로 정의하면

\[
Q^i\wedge Q^j
=
(v_B+v_C)\delta^{ij}
+
Y^{ij},
\]

여기서

\[
\boxed{
Y^{ij}
=
B^i\wedge C^j+C^i\wedge B^j
}
\]

이다.

그러므로 block simplicity의 필요충분조건은

\[
\boxed{
Y^{ij}
-
\frac{\operatorname{tr}Y}{3}\delta^{ij}
=0
}
\]

이다.

즉

\[
\boxed{
\text{local simplicity of }B,C
\not\Longrightarrow
\text{simplicity of }B+C
}
\]

이며 추가 cross-cell condition이 정확히 필요하다.

동일한 tetrad geometry에서 scale만 다른

\[
C^i=\alpha B^i
\]

를 gauge-aligned해 합치면 cross residual은 0이고 block simplicity가 보존된다.
반면 서로 다른 nonconformal tetrad에서 생성한 두 triple은 각각 local residual이
0이어도 block residual이 일반적으로 0이 아니다.

고정 seed의 1,000개 random nondegenerate tetrad pair 진단에서는 모든 표본이
\(10^{-6}\)보다 큰 block residual을 보였고, residual 중앙값은 약

\[
0.0871
\]

이었다. 이 수치는 보편상수가 아니라 local-simplicity-only prescription의
비폐쇄성을 보여주는 재현 진단이다.

## 12.6 최소 simplicity amplitude 후보

finite block에서 다음 soft projector를 사용한다.

\[
W_\sigma(B,C)
=
\exp\left[
-\frac{
\epsilon_{\rm simp}(B)^2
+\epsilon_{\rm simp}(C)^2
+\epsilon_\times(B,C)^2
+\epsilon_{\rm simp}(B+C)^2
}{
2\sigma^2
}
\right],
\]

여기서

\[
\epsilon_\times
\propto
\left\|
Y-\frac{\operatorname{tr}Y}{3}I
\right\|_F
\]

이다.

\[
\sigma\to0
\]

에서 이 weight는 local simplicity뿐 아니라 block/cross simplicity까지 만족하는
sector에 집중한다.

full Lorentzian cellular amplitude는 최소한 다음 조건들을 함께 가져야 한다.

\[
\boxed{
\begin{aligned}
&\text{causal composition-face incidence},\\
&F_e\ge4\text{ and nondegeneracy},\\
&\sum_{f\supset e}\epsilon_{ef}B_f=0
\quad\text{(closure)},\\
&\exists n_e:\ n_e\lrcorner(*B_f)=0
\quad\text{(linear simplicity)},\\
&\text{parallel-transported shared-face matching},\\
&\text{cross/block simplicity under coarse graining}.
\end{aligned}
}
\]

이 문서의 Gaussian weight는 마지막 obstruction을 검사하는 finite algebra
proxy다. EPRL/FK 또는 Barrett--Crane amplitude를 새로 유도했다는 뜻이 아니다.

## 12.7 유한 결과와 삭제된 부모 주장

### 해결

1. face endpoint attachment는 block composition triangle로 canonical하게 정할 수
   있다.
2. \(D-1\) face intensity를 coarse-edge factorization multiplicity로 읽을 수 있다.
3. one-epoch Q-spine은 비퇴화 polyhedral incidence에 대부분 부족하다.
4. CE benchmark에서 \(95\%\) incidence block은 4 epoch, \(99\%\) block은
   5 epoch다.
5. raw Poisson law는 exact simplicial valence에 집중할 수 없다.
6. local simplicity의 block 실패를 정확한 cross matrix \(Y^{ij}\)로 분리했다.

local simplicity의 block 실패는 “아직 prove하지 못한 closure”가 아니라, cross residual $Y^{ij}$가 일반적으로 남는 완전 반례다. 같은 incidence에서 $S=0$과 quadratic action은 서로 다른 saddle/Hessian/$Z$를 갖고, invariant normalized Gaussian measures도 분산이 다르다. flat BF closure witness는 common simplicity sector에 들지 않는다. 더구나 $R$과 $R+\alpha R^2$는 flat data를 공유하면서 scalaron을 구별한다. 따라서 finite face data가 unique action/measure, continuum EH, exactly two local degrees of freedom을 entail한다는 주장은 삭제한다.

이후 cross/shape matching을 동적으로 유지하는 amplitude나 RG flow를 만들 수는 있다. 그러나 그것은 global matching, measure, refinement criterion과 함께 새로 선언해 시험할 모형이며, local projector의 결과로 부르면 안 된다.

## 12.8 재현

구현:

```text
examples/physics/causal_face_simplicity.py
```

회귀:

```text
tests/test_causal_face_simplicity.py
```

집중 실행:

```powershell
.codex/hooks/python.cmd pytest tests/test_causal_face_simplicity.py -q
```

현재 격리 실행 결과는

```text
31 passed
```

였다.

참고할 기존 기하 구조는 Barrett--Crane의 bivector reconstruction,
closure+simplicity를 통한 geometric tetrahedron, 그리고 coarse graining에서
simplicity/shape matching이 자동 보존되지 않는다는 spin-foam 문헌이다.

- Barrett & Crane, *Relativistic spin networks and quantum gravity*:
  https://arxiv.org/abs/gr-qc/9709028
- Baratin & Oriti, *Quantum simplicial geometry in the group field theory formalism*:
  https://arxiv.org/abs/1108.1178
- Anzà & Speziale, *A note on the secondary simplicity constraints*:
  https://arxiv.org/abs/1409.0836
- Dittrich, *The continuum limit of loop quantum gravity - a framework for solving the theory*:
  https://arxiv.org/abs/1609.02429

## 12.9 현재 판정

- face attachment topology: **CONDITIONAL RESULT — composition nerve closed**.
- nondegenerate face incidence block: **finite probability result closed**.
- exact simplicial Poisson topology: **FAIL / 폐기**.
- Euclideanized local simplicity: **재현 PASS**.
- local simplicity under block: **FAIL**.
- cross-simplicity soft projector: **candidate implemented**.
- Lorentzian closure+shape matching+Plebanski RG fixed point: **새 모형 계약이 필요한 별도 문제; 현 유한 결과의 함의 아님**.
