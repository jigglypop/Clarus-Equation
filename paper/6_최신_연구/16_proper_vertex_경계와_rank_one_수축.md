# 16. proper vertex의 경계와 rank-one 수축

이 장은 비평탄 Plebanski 힌지에 proper vertex 공식을 곧바로 붙일 수 없다는 경계를 먼저
확정하고, 그 경계를 넘지 않는 평탄 단일-vertex 점근과 세 cell의 조건부 수축을 분리한다. 필요한
이유는 간단하다. curved de Sitter cell에서 확인한 것은 고전 장방정식과 holonomy이고, published
proper-vertex 정리는 spacelike 경계를 가진 **평탄한 하나의 Lorentzian 4-simplex**의 큰-spin
점근 정리다. 둘은 같은 incidence를 쓸 수 있어도 같은 amplitude가 아니다.

이 글은 한 해 전에 이 주제를 처음 배운 독자를 기준으로 쓴다. 먼저 기존 곡률 증인에 공식을
붙일 수 없는 이유와, 그 대신 준비한 정확한 유리수 경계를 보인다. 다음으로 정리의 가정을
만족하는 대칭 한 simplex와 spin family를 만든다. 마지막으로 세 vertex를 곱할 수 있게 하는
rank-one 입력을 명시하고, 그것이 표준 state sum이나 curved amplitude가 아님을 경계로 남긴다.
형식 지위와 수치는 [차원 분류 원장 C4](../검증_원장/참조_차원_분류_원장.md#c4-proper-boundary-12a)에
고정되어 있다.

## 16.1 같은 incidence는 같은 amplitude가 아니다

[15장](15_같은_이력의_비평탄_Plebanski_힌지.md)의 세 cell은

$$
(0,1,2,3,4),\qquad(1,2,3,4,5),\qquad(0,1,2,3,5)
$$

이라는 조합론을 고정했다. 그러나 그 장의 stereographic de Sitter 좌표는 일정한 양의 곡률을
표현하려고 선택한 것이다. 표준 Lorentzian proper vertex의 단일-vertex 정리를 쓰려면 각
boundary tetrahedron이 spacelike이고, 하나의 flat Lorentzian 4-simplex에 glue되는 경계자료가
필요하다. 기존 좌표를 이 조건에 넣으면 적어도 한 boundary tetrahedron의 유도 Gram metric이
양의 정부호가 아니다. 그래서 같은 이름의 꼭짓점을 쓴다는 이유만으로 proper vertex를 curved
cell에 붙일 수 없다.

이 문제를 피하려고 새 양자 진화를 주장하지 않는다. 같은 세 cell의 incidence 위에, 단지
flat 단일-vertex 정리의 경계 가정을 검사할 별도 고전 배치를 놓는다. 기준 척도를 $1$로 고정한
유리수 좌표는

$$
\begin{aligned}
x_0&=(0,0,0,0),&x_1&=(1/100,1,0,0),\\
x_2&=(1/25,0,1,0),&x_3&=(9/100,0,0,1),\\
x_4&=(4/25,1,1,1),&x_5&=(1/4,4,-3,-3)
\end{aligned}
$$

이다. 모든 수는 길이나 spin이 아니라 좌표비다. Minkowski Gram matrix를 exact rational
arithmetic으로 계산하면, 서로 다른 boundary tetrahedron은 $12$개이고 모두 양의 정부호다.
세 4-simplex의 Gram determinant는 차례로

$$
-\frac1{2500},\qquad-\frac{3969}{2500},\qquad-\frac9{25}
$$

이며 각각 한 개의 음 eigenvalue와 세 개의 양 eigenvalue를 가진다. 따라서 세 cell은
nondegenerate Lorentzian이고 열두 tetrahedron은 spacelike다. 이것은 standard proper vertex의
**고전 경계 가정**을 만족한다는 정확한 산출이다. amplitude의 값, spinor, measure, 양자 시간
발전은 여기서 계산하지 않는다.

## 16.2 한 flat 4-simplex에서 정확한 spin family를 만든다

외부 점근 정리의 가정을 손으로 확인할 수 있게, 네 꼭짓점은 $t=0$의 regular Euclidean
tetrahedron에 두고 다섯째 apex를 그 중심 위에 둔다. base edge 길이를 $L$, 시간 높이를 $T$라
할 때 선택은

$$
\frac{T^2}{L^2}=\frac5{256},\qquad
\frac{\ell_{\rm apex}^2}{L^2}=\frac{91}{256}
$$

이다. 이때 base triangle과 apex triangle의 면적제곱 비는 각각 $3/16$과 $27/1024$이므로 면적의
비는 $8:3$이다. 따라서 고정 shape의 최소 정수 spin은

$$
(k_{\rm base},k_{\rm apex})=(8,3),\qquad j_f(m)=m k_f,\qquad m\in\mathbb N_{>0}.
$$

여기서 $m$과 $j_f$는 무차원 SU(2) spin이다. spin을 면적의 차원 있는 값으로 착각하지 않도록,
물리 길이 $L$은 각 $m$에 맞춰 재척도한다. 큰-spin 극한도 $L$을 고정한 연속 극한이 아니라
$m\to\infty$인 이 무차원 family다.

대칭을 선택해도 admissibility는 따로 확인해야 한다. base tetrahedron의 leading Gram minors는

$$
1,\qquad\frac34,\qquad\frac12
$$

이고 side tetrahedron의 것은

$$
1,\qquad\frac34,\qquad\frac{17}{1024}
$$

이다. 모두 양수다. 전체 4-simplex Gram determinant는 $-5/512$이고 signature는 한 음ㆍ세
양이다. 네 base spin으로 된 intertwiner와 $(8m,3m,3m,3m)$ side intertwiner는 polygon 및
parity 조건을 만족하므로 다섯 coherent intertwiner를 잡을 수 있다.

여기서 한번 실패한 후보도 경계를 분명히 한다. 비율을 절반으로 읽어 $(4,3/2,3/2,3/2)$를
쓰면 side tetrahedron의 네 spin 합이 정수가 아니어서 invariant가 없다. 그래서 $4:3/2$ 후보는
거부되고, $(8,3)$이 최소 정수 재척도다.

## 16.3 외부 proper-vertex 정리가 말하는 범위

Engle과 Zipfel의 [Eq. (53)](https://arxiv.org/abs/1502.04640)는 boundary spin
$j_{ab}$와 coherent boundary data $\xi_{ab}$에 대해, 한 group 변수를 gauge fixing한
$SL(2,\mathbb C)^5$ Haar 적분으로 Lorentzian proper vertex를 정의한다. 표기를 압축하면

$$
A_v^{(+)}(j,\xi)=
\int_{SL(2,\mathbb C)^5}\!\delta(X_0)
\prod_{a=0}^{4}dX_a\;
\prod_{a<b}\mathcal K^{(+)}_{ab}(X_a,X_b;j_{ab},\xi_{ab}).
$$

여기서 $dX_a$는 Haar measure, $\delta(X_0)$는 한 vertex-frame의 gauge fixing이며,
$\mathcal K^{(+)}_{ab}$는 그 식에 적힌 EPRL injection, boundary coherent state와 proper-sector
projector를 함께 가진 face kernel이다. 이 식은 integral의 **정의**다. 위의 유리수 기하는 이
적분을 수치 평가하지 않았고, kernel 안의 coherent spinor와 phase도 materialize하지 않았다.

이 family가 flat Lorentzian 4-simplex 경계로 glue되고 위 admissibility를 만족하므로,
EngleㆍVilenskyㆍZipfel의 [Theorem 3](https://arxiv.org/abs/1505.06683)를 적용할 수 있다.
그 정리는 proper projector가 선택한 단일 Feynman/Regge 항의 큰-spin 점근을 제공한다. 고정
shape의 무차원 phase를

$$
S_{\rm Regge}(k)=\gamma\sum_f k_f\Theta_f
$$

로 정하면, 이 글에서 쓰는 외부 정리의 결과는

$$
A_v^{(+)}(m k_f)\sim C\,m^{-12}
\exp\!\left(i m S_{\rm Regge}(k)\right),\qquad m\to\infty.
$$

이다. $\gamma$, $k_f$, dihedral boost angle $\Theta_f$, $mS_{\rm Regge}$는 모두 무차원이다.
$C$의 Hessian prefactor와 적분의 수치값을 이 저장소가 계산했다는 뜻이 아니다. 이 절은 외부
stationary-phase 정리를 새로운 수치 결과로 바꾸지 않고, 그 정리의 경계 가정이 이 scaling
family에서 성립함을 확인한다.

## 16.4 세 vertex의 곱이 되는 조건부 수축

세 cell을 함께 쓸 때는 각 cell의 apex를 모두 vertex $1$로 **선언**한다. 이 공통 apex 선택은
global edge shape와 $19$개 서로 다른 triangle의 spin 배정을 하나로 만든다. $1$을 포함한
triangle에는 $3m$, 포함하지 않은 triangle에는 $8m$을 준다. 세 shared side tetrahedron은 모두
$(8m,3m,3m,3m)$이므로 같은 경계자료를 갖는다. apex를 $(1,2,1)$처럼 바꾸면 이 일치가 깨진다.
그 음성 대조군에서는 factorization과 아래 점근식을 적용하지 않는다.

본 논문에서 제안하는 수축은 각 internal tetrahedron에 정규화한

$$
P_e=|\iota_e^{\rm Regge/LS}\rangle
\langle\iota_e^{\rm Regge/LS}|,qquad
\langle\iota_e|\iota_e\rangle=1
$$

을 삽입한다. bra/ket dualization, 양립하는 local time orientation과 positive Regge-phase
branch, fixed internal spinㆍface weight, product Haar measure와 vertex별 gauge fixing도 모두
선언한다. 이 projector는 완전한 intertwiner 합이 아니라 한 방향만 남기는 rank-one 선택이다.

이 선언 아래에는 각 shared boundary contraction이 $1$이므로 세 independent proper vertex가
곱으로 분해된다. 따라서 앞 절의 외부 단일-vertex 점근을 세 번 곱해

$$
\begin{aligned}
A^{\rm cond}_3(m)
&\sim \prod_{v=1}^{3}
C_v m^{-12}\exp\!\left(i mS_{\rm Regge}(k)\right)
&& \text{세 rank-one 수축의 선언}\\
&=C_1C_2C_3\,m^{-36}
\exp\!\left(i m\,3S_{\rm Regge}(k)\right)
&& \text{무차원 phase와 거듭제곱의 곱}.
\end{aligned}
$$

이것이 얻은 조건부 결론이다. internal spin을 합하지 않았고, 완전한 internal intertwiner
basis를 적분하지 않았으며, 수축의 수치값도 계산하지 않았다. 그러므로 이는 standard EPRL
multi-vertex state sum이 아니다. de Sitter 곡률을 가진 cell의 Chern--Simons/proper amplitude도
아니다.

## 16.5 다음에 닫아야 할 다리

이 장은 proper vertex의 flat 한계와 조건부 rank-one 수축을 명확히 했을 뿐이다. [17장](17_곡률_단일가지_가우스_템플릿.md)은
곡률 있는 단일 branch의 exact local Gaussian template과 실제 block으로 옮기기 위해 빠진
보조정리를 분리한다. 비평탄 Plebanski 힌지에서 amplitude를 논하려면 곡률을 담는 curved
Chern--Simons block을 실제로 정의해야 한다. 이어서 그 적분의 branch와 thimble을 제어하고, proper sector가 곡률 있는
경계에서 살아남는지를 보여야 한다. 그 뒤에야 internal spin/intertwiner 합을 가진 state sum,
refinement, Einstein--Hilbert 지배와 two-DOF를 차례로 물을 수 있다. 접힌 sector의 에너지와
암흑 표현은 그보다 더 뒤의 별도 readout 문제다.

## 16.6 재현 범위

정확 유리수 경계, spin family, rank-one 음성 대조군은
[proper_vertex_boundary.py](../../examples/physics/proper_vertex_boundary.py)와
[test_proper_vertex_boundary.py](../../tests/test_proper_vertex_boundary.py)에 있다.

```powershell
.codex/hooks/python.cmd pytest tests/test_proper_vertex_boundary.py -q
```

원장에 기록된 focused 결과는 `24 passed`이며, curved witness의 focused 결과는 `15 passed`,
source parse는 `418 PASS`다. 이 회귀는 경계의 exact arithmetic과 선언한 수축 규칙을 검사한다.
외부 single-vertex theorem의 재증명, standard multi-vertex amplitude, curved amplitude의
계산은 검사하지 않는다.
