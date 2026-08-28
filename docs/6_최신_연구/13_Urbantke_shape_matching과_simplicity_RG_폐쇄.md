# 13. Urbantke shape matching과 simplicity RG 폐쇄

이 장은 12장에서 확인한

\[
\text{local simplicity}
\not\Rightarrow
\text{block simplicity}
\]

문제를 한 단계 더 좁힌다.

질문은 다음과 같다.

> 서로 다른 microscopic cell의 simple self-dual 2-form을 parallel transport한 뒤
> 어떤 조건에서 coarse block도 하나의 metric으로 렌더링되는가?

현재 결과는 다음처럼 분리한다.

- 하나의 common conformal-metric orbit: block 아래 정확히 닫힘.
- 그 orbit 안의 SO(3) frame rotation과 positive scale: 제거·합성 가능.
- common-orbit block operation: associative.
- coherent nonmatching geometry: block 수를 늘려도 사라지지 않음.
- fixed-seed centered-tetrad finite diagnostic: block residual이 감소하는 출력 확인.
- local/0D data가 unique Lorentzian amplitude 또는 RG attractor를 고른다는 부모 주장: 완전 반례로 삭제.

## 13.1 simple triple이 만드는 metric

Euclideanized self-dual Plebanski triple을

\[
B^i\in\Lambda^2,
\qquad i=1,2,3
\]

라 하자.

비퇴화 triple은 Urbantke construction으로 metric density

\[
\widetilde g_{\mu\nu}[B]
=
\frac1{12}
\epsilon_{ijk}
\epsilon^{\alpha\beta\gamma\delta}
B^i_{\mu\alpha}
B^j_{\beta\gamma}
B^k_{\delta\nu}
\]

를 만든다.

표준 geometric triple

\[
B^i=\Sigma^i(e)
\]

에서는

\[
\widetilde g_{\mu\nu}
=
\det(e)\,g_{\mu\nu}(e)
\]

이므로 determinant를 제거하면 conformal metric을 복원할 수 있다.

현재 finite audit에서는 positive Euclidean branch만 사용하고

\[
\widehat g[B]
=
\frac{\widetilde g[B]}{\det(\widetilde g[B])^{1/4}}
\]

로 정규화한다.

두 triple의 conformal mismatch는

\[
\boxed{
\epsilon_g(B,C)
=
\frac12\left\|\widehat g[B]-\widehat g[C]\right\|_F
}
\]

로 둔다. 이 값은 internal self-dual basis rotation과 positive rescaling에
불변이다.

## 13.2 internal SO(3) alignment

두 triple 사이의 cross-wedge matrix를

\[
M_{ij}[B,C]
=
B^i\wedge C^j
\]

로 둔다.

singular-value decomposition

\[
M=U\Sigma V^\mathsf T
\]

의 proper orthogonal polar factor

\[
R=UV^\mathsf T\in SO(3)
\]

를 사용해

\[
C_{\rm align}=RC
\]

로 내부 frame을 정렬한다.

만약

\[
C^i=\alpha Q^i{}_jB^j,
\qquad
\alpha>0,
\quad Q\in SO(3)
\]

이면 polar alignment는 정확히

\[
R=Q^\mathsf T,
\qquad
C_{\rm align}=\alpha B
\]

를 복구한다. 따라서

\[
B+C_{\rm align}
=(1+\alpha)B
\]

도 정확히 simple하다.

## 13.3 Common-Metric Orbit Closure Theorem

reference triple \(B_0\)의 orbit를

\[
\mathcal O(B_0)
=
\left\{
\alpha RB_0:
\alpha>0,
\ R\in SO(3)
\right\}
\]

로 정의한다.

> **조건부 정리.** positive Euclidean self-dual/orientation branch에서 \(\epsilon_{\rm simp}(B_0)=0\)인 비퇴화 reference triple을 두고, 모든 microscopic triple \(B_a\)가
> \(\mathcal O(B_0)\)에 속하면, 각 triple을 \(B_0\) frame으로 정렬해 합한
> block은
>
> \[
> B_{\rm block}
> =
> \sum_a R_aB_a
> =
> \left(\sum_a\alpha_a\right)B_0
> \]
>
> 이며 Plebanski simplicity를 정확히 만족한다.

**증명.** 같은 positive Euclidean self-dual/orientation branch에서 각 \(B_a=\alpha_aR_aB_0\)에 선언한 polar alignment와 parallel transport를 적용하면 \(B_a^{\rm align}=\alpha_aB_0\)다. 따라서 block은 \((\sum_a\alpha_a)B_0\)이고, \(\epsilon_{\rm simp}(B_0)=0\)과 양의 scale 때문에 simplicity를 보존한다. 덧셈의 결합법칙으로 grouping과 무관하게 같은 총 scale을 얻으므로 block operation은 associative다. \(\square\)

따라서 common orbit 안에서는 blocking order와 grouping이 결과를 바꾸지 않는다.

\[
(B_1\oplus B_2)\oplus B_3
=
B_1\oplus(B_2\oplus B_3).
\]

즉

\[
\boxed{
\text{common conformal metric}
+
\text{parallel-transported internal alignment}
\Longrightarrow
\text{exact simplicity RG closure}
}
\]

이다.

이 결과는 hard shape-matching sector가 적어도 하나 존재함을 보인다.

## 13.4 common orbit 밖의 정확한 실패

각각

\[
\epsilon_{\rm simp}(B)=0,
\qquad
\epsilon_{\rm simp}(C)=0
\]

인 geometric triple이라도

\[
\widehat g[B]\ne\widehat g[C]
\]

이면 internal SO(3) rotation만으로 metric mismatch를 제거할 수 없다.

최적 alignment 뒤 candidate가 reference의 scale multiple인지 보는 orbit residual을

\[
\epsilon_{\rm orb}
=
\frac{
\|C_{\rm align}-\alpha_*B\|_F
}{
\|C_{\rm align}\|_F
}
\]

로 둔다.

\[
\epsilon_g=0,
\quad
\epsilon_{\rm orb}=0
\]

이면 common orbit에 있고 exact closure가 성립한다. 반대로 일반적인
nonconformal tetrad pair에서는 두 residual이 모두 비영이고 block simplicity도
깨진다.

따라서

\[
\boxed{
\text{local simplicity alone is not a sufficient RG projector.}
}
\]

## 13.5 coherent mismatch no-go

simplicity residual은 전체 rescaling에 불변이다.

\[
\epsilon_{\rm simp}(\lambda B)
=
\epsilon_{\rm simp}(B).
\]

따라서 같은 mismatched pair를 \(N\)번 coherent하게 반복해

\[
B_N=N(B+C_{\rm align})
\]

로 만들면

\[
\boxed{
\epsilon_{\rm simp}(B_N)
=
\epsilon_{\rm simp}(B+C_{\rm align})
}
\]

이다.

즉 block size를 키우는 것만으로 coherent shape mismatch가 평균되어 사라지지
않는다. 이 결과는 단순 central-limit 직관을 strong RG theorem으로 쓰는 것을
막는다.

## 13.6 centered mismatch의 조건부 감소

별도의 확률적 보조 가정으로, frame-aligned *2-form* fluctuation을

\[
B_a=\bar B+\delta B_a
\]

로 두고

\[
\mathbb E[\delta B_a]=0
\]

로 실제로 두고 약상관·유한분산을 가정하면

\[
\sum_{a=1}^{N}\delta B_a=O(\sqrt N),
\qquad
\sum_{a=1}^{N}\bar B=O(N)
\]

이 경우에 한해 normalized mismatch가

\[
O(N^{-1/2})
\]

방향으로 줄 수 있다는 조건부 heuristic을 얻는다. 이 식은 nonlinear한 \(\Sigma(e)\) map과 polar alignment 뒤의 실제 audit에 자동 적용되지 않는다.

현재 코드는 2-form fluctuation을 직접 중심화하지 않는다. fixed seed에서 centered **tetrad perturbation**을 만든 뒤 nonlinear \(\Sigma(e)\)와 polar alignment를 적용한다. 그 finite diagnostic에서 sample size

\[
N=(8,16,32,64)
\]

에 대한 평균 block residual은

\[
(0.0021661,
0.0017175,
0.0011860,
0.0009697)
\]

이었고 log-log fitted power는

\[
\boxed{-0.4013}
\]

이었다.

이는 fixed-seed centered-tetrad finite output일 뿐, \(-1/2\) 정리나 보편지수를 뜻하지 않는다. 위 B-fluctuation heuristic의 전제도 검증하지 않는다.

## 13.7 최소 soft matching amplitude

finite audit에서는

\[
W(B,C)
=
\exp\left[
-\frac12
\left(
\frac{\epsilon_g^2}{\sigma_g^2}
+
\frac{\epsilon_{\rm orb}^2}{\sigma_o^2}
+
\frac{\epsilon_{\rm block}^2}{\sigma_b^2}
\right)
\right]
\]

를 후보로 사용한다.

\[
\sigma_g,\sigma_o,\sigma_b\to0
\]

에서 common-orbit simple sector에 집중한다.

그러나 이것은 full Lorentzian amplitude가 아니다. 실제 shared 3-cell gluing에는
최소한 다음 항목이 함께 필요하다.

\[
\boxed{
\begin{aligned}
&\text{face closure},\\
&\text{linear simplicity with a cell normal},\\
&\text{parallel transport across the shared cell},\\
&\text{secondary simplicity / shape matching},\\
&\text{Lorentzian reality conditions},\\
&\text{nondegenerate oriented volume}.
\end{aligned}
}
\]

## 13.8 이번 단계의 판정

### 닫힌 것

1. simple triple에서 conformal metric을 Urbantke 방식으로 복원했다.
2. internal SO(3) frame mismatch는 polar alignment로 정확히 제거된다.
3. common positive-scale conformal-metric orbit는 block 아래 정확히 닫힌다.
4. 그 orbit의 block operation은 associative하다.
5. coherent metric mismatch는 반복 blocking으로 억제되지 않는다는 no-go가
   정확히 성립한다.
6. fixed-seed centered-tetrad finite diagnostic은 감소 출력을 보였다.

### Lorentzian 새 모형 계약

local 또는 0D data가 unique Lorentzian shared-face amplitude나 common-metric sector의 RG attractor를 선택한다는 부모 주장은 action/measure 비식별성과 coherent mismatch 반례로 삭제한다. common-orbit closure와 finite audit은 이 선택을 제공하지 않는다.

Lorentzian convolution을 쓰는 새 모형은 face closure, cell normal, linear/secondary simplicity, parallel transport, Lorentzian realityㆍorientationㆍnondegeneracy/volume, action과 measure/contour, refinement kernel, RG observable과 attractor criterion을 명시 입력으로 둔다. 이는 선택된 convolution을 local/0D 자료에서 산출한다고 주장하지 않는 `[정의: 새 모형 계약]`이다.

## 13.9 재현

구현:

```text
examples/physics/urbantke_shape_matching_rg.py
```

회귀:

```text
tests/test_urbantke_shape_matching_rg.py
```

집중 실행:

```powershell
.codex/hooks/python.cmd pytest tests/test_urbantke_shape_matching_rg.py -q
```

격리 실행 결과:

```text
14 passed
```

## 13.10 문헌 경계

이 장은 Urbantke metric reconstruction과 spin-foam simplicity/shape matching의
알려진 구조를 사용하지만, EPRL/FK 또는 Barrett--Crane amplitude를 새로
유도했다고 주장하지 않는다.

- Anzà & Speziale, *A note on the secondary simplicity constraints*:
  https://arxiv.org/abs/1409.0836
- Dupuis, Ryan & Speziale, *Discrete gravity models and Loop Quantum Gravity*:
  https://arxiv.org/abs/1204.5394
- Reisenberger, *A left-handed simplicial action for Euclidean general relativity*:
  https://arxiv.org/abs/gr-qc/9609002
- Dittrich, *The continuum limit of loop quantum gravity - a framework for solving the theory*:
  https://arxiv.org/abs/1609.02429

## 13.11 상태

- Urbantke conformal metric audit: **PASS**.
- exact common-orbit block closure: **CONDITIONAL THEOREM**.
- associative blocking inside the orbit: **PASS**.
- coherent mismatch suppression: **FAIL**.
- fixed-seed centered-tetrad finite diagnostic: **FINITE DETERMINISTIC OUTPUT**.
- Lorentzian shared-cell matching RG attractor: **새 모형 계약의 명시 입력; local/0D 자료의 함의 아님**.
