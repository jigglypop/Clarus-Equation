# Q0.0–Q0.3 최소 공변 작용 Manifest

## 1. 범위

이 manifest는 전체 CE+SM 이론이 아니라 다음 네 항목을 한 convention에서
검산하는 통제 모형이다.

1. 장·배경·부호·경계조건 고정
2. field-space 좌표변환에서 공변 Hessian 복원
3. Higgs 및 $Z_2$ singlet tadpole
4. $R_\xi$ gauge-fixing, Goldstone, Faddeev--Popov ghost 장부

이 통제 모형의 통과를 전체 $SU(3)\times SU(2)\times U(1)$,
동적 중력, fermion/Yukawa, 재규격화 완료로 해석하지 않는다.

## 2. Convention

| 항목 | 고정값 |
|---|---|
| signature | $(-,+,+,+)$ |
| background | $g_{\mu\nu}=\eta_{\mu\nu}$, 경계항 소멸 |
| gauge group | broken Abelian $U(1)$ control |
| complex scalar | $H=(v+h+i\chi_G)/\sqrt2$ |
| CE control singlet | real $\phi$, $Z_2:\phi\mapsto-\phi$ |
| covariant derivative | $D_\mu=\partial_\mu-i g A_\mu$ |
| gauge mass | $m_A=gv$ |
| gauge parameter | $\xi_g>0$ |

Goldstone $\chi_G$와 계산축 $\chi$는 다른 양이다.

## 3. Bare tree-level control action

$$
S_0=\int d^4x\left[
-\frac14F_{\mu\nu}F^{\mu\nu}
-(D_\mu H)^*(D^\mu H)-V_H(H)
-\frac12\partial_\mu\phi\partial^\mu\phi-V_\phi(\phi)
-\frac{\lambda_{H\phi}}2\phi^2H^*H
\right],
$$

$$
V_H=-\mu_H^2H^*H+\lambda_H(H^*H)^2,
\qquad
V_\phi=\frac{m_{\phi,0}^2}{2}\phi^2+\frac{\lambda_\phi}{4}\phi^4.
$$

안정성의 충분조건으로
$\lambda_H>0$, $\lambda_\phi>0$ 및
$\lambda_{H\phi}>-2\sqrt{\lambda_H\lambda_\phi}$를 둔다. 통제 배경은

$$
\langle H\rangle=v/\sqrt2,
\qquad \langle\phi\rangle=0,
\qquad \mu_H^2=\lambda_Hv^2
$$

이다. 마지막 조건은 Higgs tadpole을 0으로 만든다. singlet tadpole은
potential이 \(\phi\)에 대해 짝함수이고 \(\phi=0\) 배경을 택했기 때문에
0이다. exact \(Z_2\) 자체가 이 배경을 자동으로 전역 최소점으로 정하지는
않으므로, 적어도

$$
m_{\phi,\mathrm{eff}}^2
=m_{\phi,0}^2+\frac{\lambda_{H\phi}}2v^2>0
$$

와 전역 최소점 비교를 추가로 요구한다.

## 4. $R_\xi$ gauge와 ghost

gauge-fixing function을

$$
F_g=\partial_\mu A^\mu-\xi_gm_A\chi_G,
\qquad
\mathcal L_{\rm gf}=-\frac1{2\xi_g}F_g^2
$$

로 고정한다. 부호는 위 $D_\mu$ convention에서 scalar kinetic의
$A^\mu\partial_\mu\chi_G$ 혼합을 상쇄하도록 expansion gate가 직접
검사한다. FP 작용은

$$
\mathcal L_{\rm FP}=-\bar c\,
\left.\frac{\delta F_g}{\delta\alpha}\right|_{\alpha=0}c
$$

로 정의한다. quadratic background에서 Goldstone과 ghost의 gauge 의존
질량은 모두 $\xi_gm_A^2$를 포함해야 한다. convention을 바꾸면
$D_\mu,F_g,\mathcal L_{\rm gf},\mathcal L_{\rm FP}$를 한꺼번에 바꾼다.

## 5. Field-space Hessian control

Cartesian 좌표 $q^I=(h,\chi_G,\phi)$에서는
$G_{IJ}=\delta_{IJ}$다. 비선형 좌표 $Q^A(q)$로 바꿀 때 일반 Hessian
$\partial_A\partial_BS$는 tensor가 아니므로

$$
\mathcal H_{AB}=\nabla_A\nabla_BS
=\partial_A\partial_BS-\Gamma^C{}_{AB}\partial_CS
$$

를 사용한다. Q0.1 통과조건은 Cartesian 결과를 좌표변환한 값과
$\mathcal H_{AB}$가 같고, connection을 뺀 대조군은 off-shell에서
달라지는 것이다.

## 6. Tadpole·vertex 분리

$h=\chi_G=\phi=0$에서

$$
\frac{\partial V}{\partial h}=0,
\qquad
\frac{\partial V}{\partial\phi}=0,
\qquad
\frac{\partial^2V}{\partial h\partial\phi}=0.
$$

그러나 portal expansion에는
$\lambda_{H\phi}v h\phi^2/2$와
$\lambda_{H\phi}h^2\phi^2/4$가 남는다. cross-Hessian 0을 상호작용
부재로 읽지 않는다.

## 7. Gate 결과의 의미

| 필드 | 통과 의미 |
|---|---|
| `control_q0_0_pass` | 위 범위·장·작용·부호·배경이 manifest와 일치 |
| `control_q0_1_pass` | 공변 Hessian 좌표변환 대조 통과 |
| `control_q0_2_pass` | Higgs·singlet tadpole과 portal vertex 장부 통과 |
| `control_q0_3_pass` | gauge 혼합 상쇄와 Goldstone/ghost 질량 장부 통과 |

네 항목이 참이어도 `full_ce_sm_complete`, `stress_tensor_derived`,
`renormalized_spectrum_complete`는 별도 gate다. 전체 완성에는 fermion,
non-Abelian ghost, 동적 metric, regulator/counterterm, Slavnov--Taylor
identity를 같은 action provenance에서 닫아야 한다.
