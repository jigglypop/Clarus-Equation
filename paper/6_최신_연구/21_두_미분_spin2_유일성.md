# 21. 두 미분 spin-2 유일성: 무엇이 Fierz--Pauli를 고르는가

이 장은 어떤 미시 이론이 중력을 만들었다고 주장하지 않는다. 대신 저에너지에서 이미 **국소적이고 두 번 미분되는**, 질량 없는 대칭 tensor의 실수 quadratic action이 주어졌다고 가정할 때, 그 action이 어떤 모양이어야 두 편광 후보가 되는지를 분류한다. 순서는 다섯 개의 가능한 항을 쓰고, gauge 조건만으로 남는 자유를 확인한 다음, 실제 action이 가져야 할 대칭성으로 그 자유가 어떻게 사라지는지 보이는 것이다.

여기서 $h_{\mu\nu}=h_{\nu\mu}$는 Minkowski 배경 $\eta_{\mu\nu}$ 위의 작은 tensor장이고, $q_\mu$는 운동량이다. 이 분류는 locality와 translation invariance, $\eta_{\mu\nu}$와 $q_\mu$만 사용, parity-even, tensor장 하나, 실수 quadratic action, $q$의 정확히 2차 동차성, 고차미분·nonlocal 항 없음이라는 범위에서만 완전하다. 그 범위 밖의 미시 kernel을 이 안에 넣는 일은 아직 별도 입력이다.

## 21.1 출발점은 다섯 개의 계수다

$h=\eta^{\rho\sigma}h_{\rho\sigma}$라 쓰면 위 가정 아래 가능한 운동방정식 symbol은 다음 다섯 항의 합이다.

$$
\begin{aligned}
E_{\mu\nu}={}&a q^2h_{\mu\nu}
+b(q_\mu q^\rho h_{\rho\nu}+q_\nu q^\rho h_{\rho\mu})
+c q_\mu q_\nu h\\
&+d\eta_{\mu\nu}q^\rho q^\sigma h_{\rho\sigma}
+e\eta_{\mu\nu}q^2h.
\end{aligned}
$$

이는 모든 중력 이론을 나열한 식이 아니라, 선언한 작은 상자 안에서 가능한 가장 일반적인 식이다. 따라서 이 장의 질문은 “왜 자연이 이 상자를 택했는가”가 아니라 “상자 안에서 무엇이 남는가”다.

## 21.2 gauge만으로는 하나가 남지 않는다

질량 없는 좌표 중복은 $\delta h_{\mu\nu}=q_\mu\xi_\nu+q_\nu\xi_\mu$로 쓴다. 이 방향이 방정식의 영방향이어야 한다는 조건은

$$
a+b=0,\qquad b+c=0,\qquad d+e=0
$$

이다. 독립 조건이 셋이므로 다섯 계수에는 두 자유도가 남는다. 특히 $(1,-1,1,0,0)$은 gauge-null이지만 Fierz--Pauli가 아니다. “gauge가 있으니 Einstein 식이다”라는 빠른 결론이 실패하는 완전한 반례다.

이 차이는 글의 문법과 비슷하다. 금지된 문장을 지우는 규칙만으로는 문단의 앞뒤가 서로 맞는다는 보장이 없다. gauge-nullness는 중복 좌표를 없애지만, 그 식이 하나의 실수 action에서 나왔는지는 아직 말하지 않는다.

## 21.3 action의 양쪽 일치가 마지막 계수를 고른다

실수 quadratic action의 Hessian은 대칭 tensor 성분의 자연스러운 weighted inner product에서 formal self-adjoint여야 한다. 이 요구는 $c=d$를 더한다. 같은 내용을 방정식 쪽에서 말하면 off-shell Bianchi identity $q^\mu E_{\mu\nu}=0$을 요구하는 일이다. 어느 쪽을 더해도 독립 조건은 넷이 되고 한 개의 비율만 남는다.

$$
(a,b,c,d,e)=A(1,-1,1,1,-1).
$$

$A=1$일 때 repository의 $2G^{(1)}_{\mu\nu}$와 일치하며, 부분적분과 전체 배율을 제외하면 Fierz--Pauli quadratic action이다. $A$의 부호와 크기는 이 논리로 고정되지 않는다. 이 결과는 [19장](19_선형화_Einstein_두_편광_수용_정리.md)의 $10\to6\to2$ 수용 기준과 맞지만, 그 기준이나 [20장](20_게이지_보존_Fierz_Pauli_격자_refinement.md)의 격자 극한에서 새로 유도된 것은 아니다.

**증명 범위.** 계수 조건의 rank는 gauge만으로 $3$, gauge와 self-adjointness 또는 Bianchi를 함께 쓰면 $4$다. 그러므로 계수공간 차원은 각각 $2$, $1$이며 위 ray가 유일하다. 구현한 exact 선형대수 검사는 이 선언된 분류를 재현한다.

## 21.4 아직 고르지 못한 것

고차 항을 허용하면 이 유일성은 적용되지 않는다. 예를 들어 curvature-squared 계열은 추가 mode를 가질 수 있다. 이는 이 CE 모형에 그런 mode가 있다는 말이 아니라, 두 미분 가정이 실제로 필요한 경계임을 보여 준다. [Stelle (1995)](https://arxiv.org/abs/hep-th/9509142)가 이 경계를 설명한다. 질량 없는 spin-2의 더 넓은 consistency 맥락은 [Deser (2004)](https://arxiv.org/abs/gr-qc/0411023)와 [Rodina (2016)](https://arxiv.org/abs/1612.06342)를 따른다.

남은 전진 과제는 미시 refinement kernel이 정말 이 다섯 계수 ansatz에 들어감을 보이고, 고차·nonlocal 보정을 통제하며, nonlinear Einstein--Hilbert 유효작용이 지배함을 증명하는 일이다. 이 장은 그 목적지의 좁은 IR 표지판이지, 미시에서 목적지까지의 도로가 아니다.

## 21.5 재현 범위

계수 조건, gauge-only 반례, weighted self-adjointness와 overall-sign 미결정은 [two_derivative_spin2_uniqueness.py](../../examples/physics/two_derivative_spin2_uniqueness.py) 및 [대응 테스트](../../tests/test_two_derivative_spin2_uniqueness.py)에 있다. 원장은 focused test `23 passed`와 source parse `430 PASS`를 기록한다. 이 회귀는 선언한 ansatz의 대수만 검사한다.
