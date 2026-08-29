# 22. refinement Hessian, Ward 항등식과 Schur 유효커널

이 장은 거친 기술과 미세한 기술이 같은 action을 적는다고 **입력으로** 주었을 때, 작은 흔들림의 Hessian과 gauge 영방향이 언제 거친 쪽으로 내려오는지 설명한다. 핵심은 두 갈래다. embedding이 선형이면 단순한 pullback이 되고, 비선형이면 미세 action의 stationary point에서만 추가항이 사라진다. 그 뒤 내부 변수를 적분해 없애면 Schur complement가 남는다.

## 22.1 Hessian pullback에는 숨은 항이 있다

거친 변수 $x$, 미세 변수 $F(x)$와 두 action의 관계를 $\Gamma_c(x)=\Gamma_f(F(x))$로 둔다. $J=dF$는 embedding Jacobian이고 $H_c,H_f$는 두 action의 Hessian이다. chain rule은

$$
H_c=J^{\mathsf T}H_fJ+\sum_A(\partial_A\Gamma_f)\,d^2F^A
$$

를 준다. 그래서 $F$가 선형이면 두 번째 항이 처음부터 없고, $F$가 비선형이어도 미세 방정식이 성립하는 stationary point에서는 없어진다. 둘 다 아닌 경우에는 단순 pullback을 쓰면 틀린다. 실제 한 성분 반례에서 $\partial\Gamma_f=3$, $H_f=5$, $J=2$, $d^2F=4$이면 pullback 값은 $20$이지만 진짜 Hessian은 $32$다. 빠진 값 $12$가 바로 비선형 extra term이다.

## 22.2 Ward 항등식은 잔차를 셀 때만 내려온다

coarse/fine gauge generator를 $G_c,G_f$, gauge-parameter intertwiner를 $R$이라 하자. pullback, intertwining, fine Ward identity가 어느 정도 맞는지를 각각 $R_H,R_I,R_W$로 적으면

$$
H_cG_c=R_HG_c+J^{\mathsf T}H_fR_I+J^{\mathsf T}R_WR.
$$

따라서 세 잔차가 정확히 영이면 fine 쪽의 gauge 영방향이 coarse 쪽에도 남는다. 대칭 Hessian에서는 왼쪽 Ward 항등식도 함께 따른다. 이것은 “refinement라면 gauge가 저절로 보존된다”가 아니라, 무엇을 맞춰야 보존되는지를 적은 검사식이다. 이 결과의 일반적 배경은 [Asante--Dittrich--Steinhaus (2022)](https://arxiv.org/abs/2211.09578)와 proper-vertex Hessian의 [Shirazi--Engle--Vilensky (2015)](https://arxiv.org/abs/1511.03644)에 있다. 두 문헌은 이 저장소 witness의 미시 provenance가 아니다.

## 22.3 미세한 방 여러 개를 거친 방 하나로 정리하기

fine quadratic Hessian을 경계 변수 $x$와 내부 변수 $y$로 나누어

$$
H_f=\begin{pmatrix}A&B\\B^{\mathsf T}&C\end{pmatrix}
$$

라 하자. $C$가 가역이고 선언한 conditioning 기준을 만족하면, 내부 방 $y$의 Gaussian 적분 또는 stationary elimination 뒤 경계만의 유효 kernel은

$$
H_{\rm eff}=A-BC^{-1}B^{\mathsf T}
$$

가 된다. 비유하면 미세한 방 여러 개를 없애고 거친 방 하나에 남은 반응을 기록하는 일이다. $A$만 남기면 내부 방이 경계에 되돌려 주는 힘을 잃는다. saddle embedding $J=[I;-C^{-1}B^{\mathsf T}]$에 대해 $J^{\mathsf T}H_fJ=H_{\rm eff}$이고, $G_f=JG_c$, $H_fG_f=0$이면 $H_{\rm eff}G_c=0$도 정확히 따른다.

구현한 10-boundary/3-internal Fierz--Pauli target witness는 이 산술을 정확히 회복한다. 그러나 target을 먼저 고른 뒤 만든 constructed witness는 inverse witness다. Fierz--Pauli가 미시 dynamics에서 **emerge**했다는 증명이 아니다.

## 22.4 남은 실제 물리 입력

rigging pairing의 cylindricity만으로 action equality나 Hessian pullback은 나오지 않는다. 실제 proper/EPRL multi-cell block, indefinite·gauge-null Hessian의 contour와 measure, microscopic higher term, nonlinear Einstein--Hilbert 유효작용은 모두 남아 있다. 따라서 이 장은 “어떤 supplied kernel이 Ward 구조를 지키며 coarse kernel로 내려가는가”를 말할 뿐, CE refinement가 그런 kernel을 이미 만들었다고 말하지 않는다.

## 22.5 재현 범위

chain-rule extra term, residual Ward bound, Gaussian Schur elimination과 constructed target witness는 [stationary_refinement_ward_bridge.py](../../examples/physics/stationary_refinement_ward_bridge.py) 및 [gaussian_refinement_schur_kernel.py](../../examples/physics/gaussian_refinement_schur_kernel.py)에 있다. 원장의 focused 결과는 각각 `15 passed`, `18 passed`다.
