# L8 preregister (한 스텝 평가보다 먼저 기록)

이 파일은 등록 집합 $S$, 호스트 칸, 내부 커널 $K$의 형을 이름만 붙인다.
다음 튜플 숫자, $(m',b')$ 값, 점유 점수, 궤적은 없다.

선행 기하 (인용만; 선행
`_workspace/ce/agi-l7-region-loop-20260815/artifacts/l7_preregister.md`,
`_workspace/ce/agi-l6-activity-closure-20260815/artifacts/l6_preregister.md`,
L3 open-set
`_workspace/ce/agi-l3-ne2-open-set-20260814/artifacts/o_e2_preregister.md`):

$$
R_0
=
\left[\frac25,\frac35\right]
\times
\left[\frac49,\frac6{11}\right],
\qquad
U_0=\operatorname{int}(B_{\mathrm{c}})
=
\left(\frac{13}{30},\frac{17}{30}\right)
\times
\left(\frac{137}{297},\frac{157}{297}\right).
$$

$U_0\subset R_0$은 선행 기하. 이 파일이 새 헐을 만들지 않는다.

## 호스트 칸

$$
H
=
\bigl(t,E,Z^{\mathrm{S}},Z^{\mathrm{A}},\sigma,I\bigr),
\tag{L8.1}
$$

$t\in\mathbb{N}$, $E\in[0,1]^2$, $Z^{\mathrm{S}},Z^{\mathrm{A}}\in[0,1]^3$,
$\sigma,I\in\{0,1\}$. L7이 이미 전진시키는 칸과 같은 종류다.
새 우주론이 아니다. 셋째 큐브가 아니다.

본체 인덱스는 선행과 같다. $W=I$에서 센서는 왼쪽 행, 작용은 오른쪽 행.

$$
e^{(1)}=(1,0),\qquad e^{(2)}=(0,1),
\qquad
Ie^{(1)}=(1,0),\qquad Ie^{(2)}=(0,1).
$$

그러므로 $u_I^{\mathrm{S}}(e^{(2)})=0$, $u_I^{\mathrm{A}}(e^{(2)})=1$.

## 내부 커널 형

$K$의 공역은 $H$의 공간이다. 등록 $K$는 참 한 스텝 $\Phi$ 자신이고
형은 $H\to H$다. $\hat U=K(Z)$의 칸 종류 조건
(`artifacts/program-plan.md`, 선행 계획)을 이 호스트에 옮긴 것이다.
$K$는 셋째 큐브가 아니고 BrainRuntime이 아니다.

비트값 맵 $K_{\mathrm{bit}}$의 공역은 $\{0,1\}$이다. 그 형은 이 파일에서
이름만 붙인다. 평가 숫자는 없다.

## 등록 집합 $S$

두 점은 평가 전에 적는다. 둘 다 $E=e^{(2)}$, $\sigma=1$, $I=1$, $t=0$,
$Z^{\mathrm{S}}=Z^{\mathrm{A}}$이고, 활동은 L6 등록 쌍이다.

$$
P_{\star}
=
\Bigl(\tfrac12,\tfrac{49}{99},\tfrac34\Bigr),
\qquad
P_{\circ}
=
\Bigl(\tfrac{7}{15},\tfrac{49}{99},\tfrac34\Bigr).
\tag{L8.2}
$$

$$
H_{\star}=(0,e^{(2)},P_{\star},P_{\star},1,1),
\qquad
H_{\circ}=(0,e^{(2)},P_{\circ},P_{\circ},1,1).
$$

$$
S=\{H_{\star},H_{\circ}\}.
$$

열린 상자 소속 (기하만; 상 없음):

$$
\frac{13}{30}
<
\frac{7}{15}
=
\frac{14}{30}
<
\frac12
=
\frac{15}{30}
<
\frac{17}{30},
$$

$$
\frac{137}{297}
<
\frac{49}{99}
=
\frac{147}{297}
<
\frac{157}{297}.
$$

그러므로 두 점의 $(m,b)$는 $U_0$에 있고 표지는 $q=3/4$다.

## 등록 구동

$W=I$이고 이름 $I=1$이므로 등록 한 스텝에서

$$
u^{\mathrm{A}}
=
I\,u_I^{\mathrm{A}}(e^{(2)})
=
1,
\qquad
u^{\mathrm{S}}
=
u_I^{\mathrm{S}}(e^{(2)})
=
0.
$$

각 큐브는 구동 (L4.1)의 $F_{1/4}$를 따른다. $t\mapsto t+1$.
등록 한 스텝 안에서 $E$는 고정. 비트는 고정. wash 없음.
다른 채널은 없다.

L7 판독은 $o^{\mathrm{A}}=\mathbf 1[(m^{\mathrm{A}},b^{\mathrm{A}})\in R_0]$이다.
현재 비트는 현재 $(m^{\mathrm{A}},b^{\mathrm{A}})$만 본다. 이 파일은
그 비트를 채점하지 않는다.

## 등록하지 않는 것

- 다음 튜플 숫자, $(m',b')$ 값, $q'$ 값.
- $T=32$ 상, 점유 점수, wash, 순환 구동.
- 셋째 큐브, BrainRuntime, 자율 $A$, AGI.
- $E=e^{(1)}$ 한 스텝, $I=0$ 한 스텝.
- 마우스·초파리·zebrafish 동형.
