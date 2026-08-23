# BA-SRM3 mathematics

Status: `CONDITIONAL_THEOREM / EMPIRICAL_RANK_UNTESTED`

[정의] train-only sieve coordinate를 $a\in\mathbb R^d$, future response를
$Y\in\mathbb R^{16}$, conditional mean을 $M(a)$, frozen predictive residual covariance를
$R\succ0$라고 한다.

[정의]

$$
J(a)=DM(a),\qquad G(a)=J(a)^TR^{-1}J(a).
$$

[정리] 모든 $u$에 대해

$$
u^TG(a)u=\lVert R^{-1/2}J(a)u\rVert^2\ge0,
$$

이므로 $G$는 PSD다. 또한

$$
\operatorname{rank}G(a)=\operatorname{rank}J(a)\le\min(d,16).
$$

따라서 $d>16$이어도 이 측정으로 보이는 pointwise quotient rank는 16을 넘지 않는다.
전체 history Hilbert space의 SPD metric을 유한 event output으로 식별할 수 없다.

[조건부 정리] 어떤 neighborhood에서 $J$가 일정 rank이고 kernel이 매끄러운 closed
subbundle을 이루면 $T_aH_d/\ker J(a)$에 $G$가 유도하는 양의 정부호 내적을 줄 수 있다.
이 전제는 데이터에서 rank/gauge/support gate를 통과하기 전에는 성립했다고 가정하지 않는다.

[미완성] 실제 $J$, $R$, rank, constant-rank support와 predictive advantage는 아직 계산하지
않았다. numerical ridge나 covariance floor는 이 미완성을 증명으로 바꾸지 않는다.
