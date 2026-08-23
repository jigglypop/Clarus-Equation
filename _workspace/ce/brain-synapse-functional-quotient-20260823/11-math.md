# 11-math — BA-SRM2 함수공간 몫기하 감사

Status: COMPLETE

## M1 — 정의역 가정과 PSD

$M:\mathcal H\to\mathcal Y$는 관심 domain에서 Fréchet 미분 가능하고 $DM_x$는
bounded라고 가정한다. $C\succ0$이고 $DM_xu\in\operatorname{dom}(C^{-1/2})$인
tangent에 대해

$$
G_x(u,v)=
\langle C^{-1/2}DM_xu,C^{-1/2}DM_xv\rangle_{\mathcal Y}
$$

를 정의한다. 그러면

$$
G_x(u,u)=\lVert C^{-1/2}DM_xu\rVert^2\ge0
$$

이므로 $G_x$는 대칭 PSD다.

## M2 — 유한 관측 no-go

$J_{m,x}:\mathcal H\to\mathbb R^m$이면

$$
\operatorname{rank}G_x^{(m)}
=\operatorname{rank}J_{m,x}\le m.
$$

$\mathcal H$가 무한차원이고 $J_{m,x}$가 bounded finite-rank이면 kernel은 닫힌
무한차원 부분공간이다. 따라서 전체 $\mathcal H$의 data-identified SPD metric은
불가능하다. $\lambda I$는 prior geometry를 더할 뿐 관측 rank를 회복하지 않는다.

## M3 — pointwise quotient와 전역 조건

$u\sim v$를 $u-v\in\ker J_{m,x}$로 정의하면

$$
\langle[u],[v]\rangle_{\rm obs}:=G_x^{(m)}(u,v)
$$

는 $T_x\mathcal H/\ker J_{m,x}$에서 양의 정부호다. 그러나 점마다 rank가 바뀌면
quotient dimension도 바뀌어 Riemannian manifold가 되지 않는다. 전역 승격에는
constant-rank neighborhood와 smooth closed kernel subbundle이 필요하다.

## M4 — finite sieve ceiling

train-only sieve $P_d\mathcal H\simeq\mathbb R^d$와 $m$개 독립 output을 쓰면

$$
G_d(q)=J_d(q)^TR_d^{-1}J_d(q),
\qquad
\operatorname{rank}G_d(q)\le\min(d,m).
$$

$d$를 크게 선언해도 관측 rank가 늘지 않는다. rank $r<d$이면 rank-$r$ quotient만
보고해야 한다.

## M5 — coordinate transport

가역 affine rechart $q'=Aq+b$에서는

$$
J'=JA^{-1},\qquad
G'=A^{-T}GA^{-1},\qquad
g_{\rm ref}'=A^{-T}g_{\rm ref}A^{-1}.
$$

kernel response map을 쓸 경우 kernel distance의 reference tensor도 함께 transport해야
한다. isotropic Euclidean RBF를 scale/shear chart에서 새로 계산하면 같은 estimator가
아니며 affine gauge 불변성이 없다.

## M6 — 시간 방향과 metric 분리

$G_x$는 response distribution의 대칭적 구별가능성이다. causal semiflow와 directed
delay는 일반적으로 역대칭이 아니므로 Riemannian distance와 동일시하지 않는다.
order는 causal prediction과 order-shuffle control로 별도 검정한다.

## M7 — 무차원성

history channel은 미리 선언한 전압, 전류, 시간, frequency와 length scale로 각각
정규화한다. output을 물리 단위로 남기면 $C$가 output unit의 제곱을 가져
$J^TC^{-1}J$ line element가 무차원이 된다. 무차원성은 차원 정합이며 생물학적
정당성이나 예측 성공을 증명하지 않는다.

