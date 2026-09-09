# 15. 같은 이력의 비평탄 Plebanski 힌지

이 장은 앞 장의 평탄한 유한 증인을 버리고 새 이론을 붙이는 글이 아니다. 같은 두 4-simplex 이력에 꼭 한 개의 simplex를 더해, 원래는 끊겨 있던 한 힌지 주위의 고리를 닫는다. 그 닫힌 고리 위에 일정한 양의 곡률을 가진 Plebanski/Einstein 해를 놓고, 곡률이 실제로 항등이 아닌 holonomy를 만든다는 데까지를 보인다. 따라서 여기서 전진한 것은 “평탄한 한 예”에서 “같은 이력을 보존한 비평탄한 한 예”까지다. 이 장의 순서는 고리의 조합론, 장방정식, holonomy, 남은 증명의 순서다.

형식 지위와 수치는 [차원 분류 원장 C4](../검증_원장/참조_차원_분류_원장.md#c4-curved-same-history-11a)에 고정되어 있다. 이 글은 그 원장을 고치지 않는다. CE의 서사에서는 환경이 강제한 선택이 하나의 기하 이력을 고르고, 선택되지 않은 성분은 접힌 채 남으며, 그것을 우주론적 암흑 표현으로 읽는 마지막 단계가 뒤따른다. 여기서 닫는 것은 첫 단계의 기하적 후보 하나뿐이다. 접힘의 보존, 더구나 암흑에너지 readout은 이 계산으로 얻지 않는다.

## 15.1 왜 두 simplex만으로는 곡률 고리를 말할 수 없었나

hinge란 여러 4-simplex가 공유하는 삼각형이다. 여기서는 labelled triangle을
$f=(1,2,3)$로 고정한다. 삼각형 주위를 한 바퀴 돈다는 말이 성립하려면, 그 삼각형을 지나는
4-simplex들의 **link**가 원이어야 한다. 종이에 삼각형의 둘레를 따라 방을 차례로 붙였다고
생각하면 된다. 방들이 사슬로 끝나면 출발점으로 돌아오지 못하고, 고리로 닫혀야만 한 바퀴
운반을 정의할 수 있다.

평탄 증인이 이미 쓰던 두 방은

$$
\sigma_L=(0,1,2,3,4),\qquad
\sigma_R=(1,2,3,4,5)
$$

이다. $f$를 제외하고 각 방에 남는 꼭짓점 쌍을 link의 변으로 읽으면

$$
\operatorname{Lk}(f)=[0,4]\cup[4,5]
$$

가 된다. 즉 $0-4-5$라는 열린 길이다. 이 사실은 결함이 아니라 이전 평탄 증인의 정확한
경계다. 열린 길에는 닫힌 Regge dual face도, 그 face를 도는 운반도 없다.

여기서는 원래 두 simplex를 그대로 두고

$$
\sigma_C=(0,1,2,3,5)
$$

를 하나 붙인다. 새 simplex는 남는 쌍 $(0,5)$를 더하므로 link는

$$
[0,4]\cup[4,5]\cup[5,0]\simeq S^1
$$

가 된다. 고정한 여섯 labelled 꼭짓점에서 한 cell만 추가하는 경우, 세 개보다 적게는 열린 두
변을 원으로 닫을 수 없으므로 이 simplex 추가가 최소다. 이 추가는 분명히 새 **확장 증인**과
확장 이력을 만든다. “같은 이력”은 그것이 원래 이력과 문자 그대로 동일하다는 뜻이 아니라,
원래 두 cell과 그 typed trace를 바꾸지 않고 포함한 확장이라는 뜻이다. 이 보존 관계가 중요하다.
causal 2-cell에서 이 triangle으로 가는 대응은 계속 선언된 duality이며, 이 조합론만으로
discrete Levi--Civita transport를 유도한 것은 아니다.

## 15.2 단위를 먼저 없애고, 일정 곡률 해를 놓는다

길이 단위가 식 속에 섞이면 곡률과 우주상수를 비교할 수 없다. 기준 길이 $L_{\rm ref}$를
고정하고 좌표와 두 매개변수를

$$
y^I=\frac{x^I}{L_{\rm ref}},\qquad
\kappa=K L_{\rm ref}^2,\qquad
\bar\lambda=\Lambda L_{\rm ref}^2
$$

로 쓴다. 여기서 $K$는 일정한 sectional curvature, $\Lambda$는 우주상수다. 인증한
stereographic patch에서는 $0<\kappa<4$이며, Plebanski 방정식의 관계는
$\bar\lambda=3\kappa$다. 따라서 수치 구현은 단위를 가진 $K$와 $\Lambda$가 아니라 이 세
무차원 양만 받는다.

Minkowski metric $\eta_{IJ}=\operatorname{diag}(-1,1,1,1)$, $s=\eta_{IJ}y^Iy^J$를 두고

$$
\Omega=\left(1+\frac{\kappa s}{4}\right)^{-1},\qquad
e^I=\Omega\,dy^I,
$$

$$
\omega^{IJ}=\frac{\kappa\Omega}{2}
\left(y^I dy^J-y^Jdy^I\right)
$$

로 둔다. $e^I$는 각 점에서 자를 놓는 tetrad이고, $\omega^{IJ}$는 그 자를 옆 점으로
옮길 때의 규칙이다. 이 선택은 de Sitter metric의 stereographic 표현이다.

**계산.** $d\Omega=-(\kappa/2)\Omega^2y_I dy^I$를 먼저 대입하면 $de^I$의 두 항과
$\omega^I{}_J\wedge e^J$의 항이 정확히 상쇄한다. 따라서

$$
T^I=de^I+\omega^I{}_J\wedge e^J=0.
$$

같은 미분을 $d\omega^{IJ}+\omega^I{}_K\wedge\omega^{KJ}$에 적용하면 $y$에 의존하는
항이 상쇄하고

$$
R^{IJ}=\kappa\,e^I\wedge e^J
$$

만 남는다. 이것은 표본점에서 맞는 수치적 우연이 아니라 이 patch 전체에서 하는 해석 계산이다.
좌표 patch 밖이나 $\kappa\ge4$의 이 유한 convex-hull 인증을 주장하지 않는다.

## 15.3 Plebanski 변수로 다시 읽기

복소 self-dual convention을 다음처럼 고정한다.

$$
\Sigma^i=i\,e^0\wedge e^i-\frac12\epsilon^i{}_{jk}e^j\wedge e^k,
\qquad
A^i=i\omega^{0i}-\frac12\epsilon^i{}_{jk}\omega^{jk}.
$$

이는 tetrad 기하를 chiral 2-form과 연결로 바꿔 쓰는 사전이다. 위에서 얻은 $T=0$은 이
사전에서 $D_A\Sigma=0$을 주고, 일정 곡률식은

$$
F^i=\kappa\Sigma^i
$$

를 준다. 또 $\Sigma^i\wedge\Sigma^j$의 trace-free 부분은 0이다. 이것이 simplicity다.
다시 말해 세 개의 독립 2-form을 마음대로 고른 것이 아니라 하나의 tetrad에서 만들어졌다는
제약이다.

우주상수 Plebanski 방정식은 이 convention에서

$$
F^i=\left(\Psi^i{}_j+\frac{\bar\lambda}{3}\delta^i{}_j\right)\Sigma^j
$$

로 읽는다. 여기서는 Weyl multiplier $\Psi=0$이므로, 앞 식과 비교하면
$\bar\lambda=3\kappa$일 때 정확히 맞는다. 그러므로 이 patch는 Lorentzian Einstein
endpoint도 만족한다. 반대로 $\bar\lambda\ne3\kappa$이면 남는 residual이 생긴다. 이 역방향
검사는 “곡률을 넣었으니 Einstein일 것”이라는 추측을 막는다.

## 15.4 곡률은 실제로 한 바퀴 돌아 항등에서 벗어난다

고정한 primal triangle은 $(1,2,3)$이며, 좌표 변의 길이를 $a=1$로 둔다. 이 triangle의
평면 $y^0=y^3=0$에서는 연결이 한 $J_{12}$ 생성자 방향에만 놓인다. 따라서 경로의 서로 다른
점에서 생기는 연결이 서로 commute하고 path ordering이 빠진다. 세 oriented boundary segment
$(1,2)$, $(2,3)$, $(3,1)$에서 이 연결을 적분한 값을 합치면, 일반 기호
$u=\kappa a^2/4$ 아래 선택한 방향 $(1,2,3,1)$의 정확한 운반 각은

$$
\phi_\kappa=
\frac{\kappa a^2/2}{\sqrt{(1+u/2)(u/2)}}
\arctan\sqrt{\frac{u/2}{1+u/2}}.
$$

따라서 holonomy는 $(1,2)$ 공간 평면의 회전
$H_f=\exp(\phi_\kappa J_{12})$다. $\kappa=0$이면 $H_f=1$이고, $\kappa>0$이면
$H_f\ne1$이다. 예를 들어 $\kappa=a=1$에서는
$\phi_\kappa\simeq0.4290007392$다. 이는 고정 labelled face 위에서 정의한 primal
holonomy다. 임의 triangulation의 Regge deficit을 계산했다는 주장은 아니다.

닫힌 link에는 별도의, 더 조심스러운 dual 확인도 있다. 각 simplex 또는 tetrahedron의
꼭짓점을 de Sitter hyperboloid에 심고, 그 점들의 **정규화한 양의 ambient 합**을 centre로
정한다. 순서는

$$
\sigma_L\to(1234)\to\sigma_R\to(1235)\to\sigma_C\to(0123)\to\sigma_L
$$

이다. 이웃한 두 flag는 포함 관계에 있다. 그래서 두 centre의 양의 선형결합은 큰 flag의
꼭짓점에 대한 양의 계수를 계속 가지며, 선택한 geodesic은 그 positive cone 안에 남는다.
각 구간에는 hyperboloid 위의 정확한 parallel transport

$$
P_{X\to Y}(v)=v-
\frac{\kappa(v\mathbin\cdot Y)}{1+\kappa X\mathbin\cdot Y}(X+Y)
$$

를 쓴다. 여섯 구간을 곱하면 비항등 Lorentz holonomy가 나온다. 이것은 abstract dual loop를
projective barycentric하게 실현한 결과다. centre의 선택도, geodesic branch도 입력이며,
**Regge deficit angle과 같다고 유도하지 않는다.**

## 15.5 무엇을 계산으로 확인했고, 무엇을 아직 증명하지 않았나

해석적으로 증명한 것은 patch 전체의 $T=0$, $R=\kappa e\wedge e$, simplicity,
$D_A\Sigma=0$, $F=\kappa\Sigma$다. 구현은 이를 대신 증명하지 않는다. 구현은 여섯 꼭짓점과
세 simplex barycentre, 합계 아홉 점에서 위 항등식을 다시 평가한다. 고정 회귀의 최대 field
residual은 약 $1.8\times10^{-16}$이고, abstract dual loop의 비항등 residual은 약
$0.0707000$이다. 이 수치는 코드 변경을 잡는 회귀 기록이지 기계가 하는 기호 증명은 아니다.

음성 대조군도 같은 계약 안에 둔다. closing simplex를 빼면 link는 다시 열린 길이라 dual
loop를 만들 수 없다. $\kappa=0$이면 primal holonomy가 항등이 된다. $\bar\lambda\ne3\kappa$이면
Plebanski residual이 남는다. $(1,2,3)$ 아닌 face label은 거부한다. 이 중 하나가 달라지면
이 장의 조건부 정리를 적용하지 않는다.

따라서 정확한 claim ceiling은 다음이다. **원래 두 simplex를 보존하고 한 simplex로 닫은
같은 typed 이력 위에, 하나의 constant-curvature 비평탄 Lorentzian Plebanski/Einstein
유한 조건부 존재 증인이 있다.** 이것은 bare $0$차원 자료가 곡률이나 작용을 유일하게
도출했다는 말도, 일반 curved geometry의 정리도, discrete Levi--Civita/Regge curvature의
정리도 아니다.

다음 증명 순서는 이 ceiling을 넘어서는 데 맞춘다. [16장](16_proper_vertex_경계와_rank_one_수축.md)은
표준 proper vertex를 curved cell에 그대로 붙일 수 없는 이유와, flat 단일-vertex 경계 및
조건부 rank-one 수축까지를 분리한다. 곡률 있는 sector의 실제 amplitude와 measure/contour는
여전히 닫아야 한다. 그 다음 refinement에서
distributional rigging-map criterion을 검사하고, 그 한계에서 Einstein--Hilbert 지배와 정확히
두 개의 massless spin-2 자유도를 분리해 보여야 한다. 마지막으로 접힌 sector의 에너지 보존,
우주론적 readout, 독립 관측 계약을 세워야 암흑 표현을 말할 수 있다. 이 순서를 거꾸로 하거나
한 유한 holonomy로 건너뛸 수 없다.

## 15.6 재현 경로

구현은 [curved_plebanski_hinge.py](../../examples/physics/gravity/curved_plebanski_hinge.py), 집중
회귀는 [test_curved_plebanski_hinge.py](../../tests/test_curved_plebanski_hinge.py)에 있다.
Windows에서는 다음 명령으로 확인한다.

```powershell
.codex/hooks/python.cmd pytest tests/test_curved_plebanski_hinge.py -q
```

원장에 기록된 focused 결과는 `15 passed`, source parse는 `416 PASS`다. 이 검사는 고정한
유한 계약의 구현 일관성만 검사하며, continuum quantum gravity 또는 암흑에너지의 증명으로
승격하지 않는다.
