# CE 핵심 정리 증명 — 우주론/GR

<!-- 출처: docs/검증_원장/참조_핵심_정리_증명.md — §12,15,17,19,24,25 이동 | 이동일: 2026-08-23 (MULTIREPO_PLAN.md P2-6) -->

이 문서는 CE 문서에서 반복 사용하는 순수 수학 정리 중 우주론·GR 도메인이
주로 소비하는 것을 보존한다. 물리적 동일시나 관측 비교는 포함하지 않는다.
정리 번호는 원파일 `참조_핵심_정리_증명.md`의 번호를 유지한다.

<a id="noether-stress"></a>

## 12. 미분동형사상 불변 작용의 on-shell stress 보존

**[정리]** 경계항이 사라지는 조건 아래 미분동형사상 불변인
$S[g,\varphi]$를 두고

$$
T_{\mu\nu}:=-\frac2{\sqrt{-g}}\frac{\delta S}{\delta g^{\mu\nu}},
\qquad
E_\varphi:=\frac1{\sqrt{-g}}\frac{\delta S}{\delta\varphi}
$$

로 정의하자. 스칼라장에 대해 Noether 항등식은 부호 규약에 맞추어

$$
\nabla^\mu T_{\mu\nu}=E_\varphi\nabla_\nu\varphi
$$

이고, 운동방정식 $E_\varphi=0$에서
$\nabla^\mu T_{\mu\nu}=0$이다.

**증명.** compact support를 갖는 무한소 벡터장 $\xi^\mu$가 생성하는
미분동형사상에서 $\delta g_{\mu\nu}=2\nabla_{(\mu}\xi_{\nu)}$,
$\delta\varphi=\xi^\nu\nabla_\nu\varphi$이다. $\delta S=0$에 대입하고 한 번
부분적분하면

$$
0=\int\sqrt{-g}\,
\bigl(-\nabla^\mu T_{\mu\nu}+E_\varphi\nabla_\nu\varphi\bigr)
\xi^\nu d^dx.
$$

$\xi$가 임의이므로 항등식이 따른다. $\square$

따라서 임의의 Hessian을 stress tensor로 부르는 것은 이 정리의 결론이
아니다. 작용의 metric variation으로 정의한 총 stress에만 적용된다.

<a id="canonical-scalar-flrw"></a>

## 15. canonical scalar FLRW와 phantom 경계

**[정리]** 부호 $(-+++)$의 평탄 FLRW에서 균일한 canonical scalar의
작용을

$$
S_\phi=\int d^4x\sqrt{-g}
\left[-\frac12\nabla_\mu\phi\nabla^\mu\phi-V(\phi)\right]
$$

로 두자. 그러면

$$
\ddot\phi+3H\dot\phi+V_{,\phi}=0,\qquad
\rho_\phi=\frac12\dot\phi^2+V,\qquad
p_\phi=\frac12\dot\phi^2-V
$$

이고 $\dot\rho_\phi+3H(\rho_\phi+p_\phi)=0$이다. $\rho_\phi>0$이면

$$
w_\phi+1=\frac{\dot\phi^2}{\rho_\phi}\geq0.
$$

따라서 양의 kinetic term을 가진 이 최소 모형은 $w=-1$을 아래로 통과할
수 없다.

**증명.** Euler--Lagrange 변분으로 Klein--Gordon 방정식을 얻는다.
metric variation으로
$T_{\mu\nu}=\nabla_\mu\phi\nabla_\nu\phi-g_{\mu\nu}
[\frac12(\nabla\phi)^2+V]$를 얻고 FLRW에 대입하면 $\rho,p$가 따른다.
연속방정식은 Klein--Gordon 방정식에 $\dot\phi$를 곱한 것과 같다.
마지막 부등식은 $\rho+p=\dot\phi^2$에서 따른다. $\square$

<a id="vacuum-stress"></a>

## 17. 상수 진공항의 stress tensor

**[정리]** 물질 작용에 상수 $V_0$가

$$
S_0=-\int d^4x\sqrt{-g}\,V_0
$$

로 들어가면 $T^{(0)}_{\mu\nu}=-V_0g_{\mu\nu}$이다.
FLRW 정지 관찰자에게 $\rho_0=V_0$, $p_0=-V_0=-\rho_0$이다.
$V_0\ne0$일 때만 비율 $w_0:=p_0/\rho_0$가 정의되며 그 값은
$-1$이다.

**증명.**
$\delta\sqrt{-g}=-\frac12\sqrt{-g}g_{\mu\nu}\delta g^{\mu\nu}$를
stress tensor의 정의에 대입한다. perfect-fluid 분해와 비교하면
$\rho=V_0,p=-V_0$를 얻는다. $\square$

이 정리는 $V_0$의 절대값이나 복사보정에 대한 안정성을 설명하지 않는다.

<a id="starobinsky-slow-roll"></a>

## 19. Starobinsky형 potential의 slow-roll 근사

**[정리: 근사 범위]** Einstein gravity에 최소 결합된 정준 단일장이
배경 에너지를 지배하고, $V_0>0$인 potential을

$$
V(\phi)=V_0\left(1-e^{-\sqrt{2/3}\,\phi/M_{\rm Pl}}\right)^2
$$

로 갖는다고 하자. $y=e^{-\sqrt{2/3}\phi/M_{\rm Pl}}$가
$0<y_N<y_{\rm end}<1$인 plateau, adiabatic
Bunch--Davies 초기상태, tree-level 선형 섭동과 leading potential
slow-roll을 적용한다. $N\gg1$인 horizon-exit 구간에서

$$
n_s=1-\frac2N+O\!\left(\frac{\log N}{N^2}\right),\qquad
r=\frac{12}{N^2}+O\!\left(\frac{\log N}{N^3}\right),
$$

$$
\alpha_s^{\rm(run)}=-\frac2{N^2}
+O\!\left(\frac{\log N}{N^3}\right).
$$

**계산.** $y=e^{-\sqrt{2/3}\phi/M_{\rm Pl}}$로 두면

$$
\epsilon_V=\frac43\frac{y^2}{(1-y)^2},\qquad
\eta_V=\frac43\frac{-y+2y^2}{(1-y)^2},
$$

이고

$$
N=\frac34\left[
\frac1{y_N}+\log y_N-\frac1{y_{\rm end}}-\log y_{\rm end}
\right].
$$

따라서 $y_N=3/(4N)+O(\log N/N^2)$이다. 이를
$n_s=1-6\epsilon_V+2\eta_V$, $r=16\epsilon_V$,
$d/d\log k\simeq-d/dN$에 대입하면 표시한 근사를 얻는다.
$\square$

$N$은 reheating 이력에 의존하고 $V_0$는 스칼라 진폭으로 고정된다.
따라서 potential을 선택하지 않은 CE 코어의 무입력 예측으로 읽지 않는다.

<a id="dust-lambda-age"></a>

## 24. 평탄 dust+$\Lambda$ 우주의 나이

**[정리]** $a_0=1$, $H_0>0$,
$\Omega_{m,0},\Omega_{\Lambda,0}>0$,
$\Omega_{m,0}+\Omega_{\Lambda,0}=1$인 expanding Big-Bang branch에서

$$
E(a)^2=\Omega_{m,0}a^{-3}+\Omega_{\Lambda,0}
$$

이면

$$
H_0t_0=
\frac{2}{3\sqrt{\Omega_{\Lambda,0}}}
\operatorname{arsinh}
\sqrt{\frac{\Omega_{\Lambda,0}}{\Omega_{m,0}}}.
$$

**증명.** $dt=da/(aH)$를 적분하고 $u=a^{3/2}$로 치환하면

$$
H_0t_0
=\int_0^1\frac{a^{1/2}\,da}
{\sqrt{\Omega_{m,0}+\Omega_{\Lambda,0}a^3}}
=\frac23\int_0^1\frac{du}
{\sqrt{\Omega_{m,0}+\Omega_{\Lambda,0}u^2}},
$$

이고 마지막 적분이 표시한 역쌍곡사인이다. $\square$

<a id="oscillating-scalar-dust"></a>

## 25. 빠르게 진동하는 quadratic scalar의 dust 극한

**[정리: adiabatic 근사]** 평탄 FLRW에 최소 결합된 canonical scalar가
$m>0$, $V=m^2\phi^2/2$를 갖는다고 하자. 정확한 유효 주파수를

$$
\omega^2(t):=m^2-\frac32\dot H-\frac94H^2
$$

로 두자. 관심 구간 $I$에서 $\omega^2>0$이고

$$
\eta:=\sup_{t\in I}\max\!\left\{
\frac{|H|}{m},\frac{|\dot H|}{m^2},
\frac{|\dot\omega|}{\omega^2}
\right\}\ll1
$$

이라 하자. 이것만으로 장시간 WKB 근사를 결론내리지 않고, 비영 해
$\psi:=a^{3/2}\phi$가 $I$ 전체에서 다음 균일 WKB 전개를 갖는다고
추가로 가정한다. 어떤 상수 $C\ne0$, $C_{\rm WKB}<\infty$, 위상
$\dot\theta=\omega$, $2\pi$-주기인 two-scale 나머지
$R_i(t,\vartheta)$가 존재하여 실제 궤도 $\vartheta=\theta(t)$에서

$$
\psi=C\omega^{-1/2}
[\cos\theta+R_0(t,\theta)],\qquad
\dot\psi=-C\omega^{1/2}
[\sin\theta+R_1(t,\theta)],
$$

$$
\sup_{(t,\vartheta)\in I\times[0,2\pi]}
(|R_0(t,\vartheta)|+|R_1(t,\vartheta)|)
\le C_{\rm WKB}\eta.
$$

slow time $t$를 고정한 two-scale 위상평균을

$$
\langle f\rangle_\theta(t):=\frac1{2\pi}
\int_0^{2\pi}f(t,\vartheta)\,d\vartheta
$$

로 정의하면

$$
\frac{\langle p_\phi\rangle_\theta}
{\langle\rho_\phi\rangle_\theta}=O(\eta),\qquad
a^3\langle\rho_\phi\rangle_\theta=C_\rho[1+O(\eta)],
\qquad C_\rho=\frac{C^2m}{2}>0
$$

이다. 따라서 leading adiabatic order $\eta\to0$에서
$\langle p_\phi\rangle_\theta=0$,
$\langle\rho_\phi\rangle_\theta\propto a^{-3}$이다.

**계산.** $\psi=a^{3/2}\phi$로 두면 Klein--Gordon 방정식은

$$
\ddot\psi+
\left[m^2-\frac32\dot H-\frac94H^2\right]\psi=0
$$

이 된다. 가정한 균일 WKB 전개와 위상평균에서
$\langle\cos^2\vartheta\rangle_\theta=
\langle\sin^2\vartheta\rangle_\theta=1/2$이고 leading 교차항이
사라지며, 나머지에서 오는 항은 모두 상대적으로 $O(\eta)$다.
또 $\omega/m=1+O(\eta)$이므로
$\langle\dot\phi^2\rangle_\theta=
m^2\langle\phi^2\rangle_\theta[1+O(\eta)]$이고, 표시한 압력과
에너지 밀도 관계가 따른다. $\square$

임의의 유한 시간창 평균은 이 정리의 평균 연산자가 아니다. 길이
$\Delta t$인 창을 쓰면 일반적으로 끝점 오차
$O((m\Delta t)^{-1})$와 slow-background drift 오차를 별도로 더해야 한다.
또 점별 adiabatic 비율이 작다는 조건만으로는 parametric resonance를
배제하지 못한다. 그런 배경에서는 위의 균일 WKB 나머지 가정 자체가
성립하지 않으므로 이 정리를 적용하지 않는다.
