# 자기측정 잔여량 암흑에너지의 수학 감사

Status: COMPLETE

## 1. operational 입력과 독립 공리

앞선 고정 dephasing 정리에서

$$
u=e^{-\theta},\qquad c=1-e^{-\theta},\qquad
C_{\rm self}=C_\infty c
$$

는 모두 무차원이다. 이 결과만으로는 국소 장, stress tensor 또는 에너지
밀도를 만들 수 없다. 따라서 Cauchy surface $\Sigma_*$에서

$$
\Theta|_{\Sigma_*}=\mathcal R_\Theta[\theta,\mu_F]
$$

를 주는 retention map과, 이후 source-free 공변 작용을 별도 공리로 둔다.

## 2. 채택 작용과 변분

부호 $(-,+,+,+)$에서

$$
S=\int d^4x\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}R-
\frac{f^2}{2}(\nabla\Theta)^2-ho_*e^{-\Theta}
\right]+S_m+S_r
$$

로 둔다. $f>0$, $\rho_*>0$이다. 변분하면

$$
M_{\rm Pl}^2G_{\mu\nu}=T^{(m)}_{\mu\nu}+T^{(r)}_{\mu\nu}
+f^2\nabla_\mu\Theta\nabla_\nu\Theta
-g_{\mu\nu}\left[\frac{f^2}{2}(\nabla\Theta)^2+V\right],
$$

$$
f^2\Box\Theta-V_{,\Theta}=0,
\qquad V=\rho_*e^{-\Theta},\qquad V_{,\Theta}=-V.
$$

따라서 FLRW의 homogeneous 식은

$$
\boxed{\ddot\Theta+3H\dot\Theta-\frac{V}{f^2}=0}
$$

이고, 정지 초기상태에서는 $\ddot\Theta=V/f^2>0$이다. 장 방정식 위에서

$$
\nabla_\mu T^{\mu}{}_{\nu}{}^{(\Theta)}
=(f^2\Box\Theta-V_{,\Theta})\nabla_\nu\Theta=0
$$

이므로 sector 보존이 닫힌다.

## 3. FLRW 밀도와 상태방정식

$$
\rho_\Theta=\frac{f^2}{2}\dot\Theta^2+V,
\qquad
p_\Theta=\frac{f^2}{2}\dot\Theta^2-V,
$$

$$
3M_{\rm Pl}^2H^2=\rho_m+\rho_r+\rho_\Theta,
$$

$$
-2M_{\rm Pl}^2\dot H=\rho_m+\frac43\rho_r+f^2\dot\Theta^2.
$$

$K=f^2\dot\Theta^2/2\ge0$이므로

$$
-1\le w_\Theta=\frac{K-V}{K+V}\le1.
$$

즉 source-free 정준 모형은 phantom $w<-1$을 만들지 않는다.

## 4. 지수 퍼텐셜 고정점

$$
\phi=f\Theta,\qquad \lambda=\frac{M_{\rm Pl}}f,
\qquad V(\phi)=\rho_*e^{-\lambda\phi/M_{\rm Pl}}
$$

및

$$
x=\frac{\dot\phi}{\sqrt6M_{\rm Pl}H},\qquad
y=\frac{\sqrt V}{\sqrt3M_{\rm Pl}H}
$$

를 사용하면

$$
x'=-3x+\sqrt{\frac32}\lambda y^2+
\frac32x\left(1+x^2-y^2+\frac{\Omega_r}{3}\right),
$$

$$
y'=-\sqrt{\frac32}\lambda xy+
\frac32y\left(1+x^2-y^2+\frac{\Omega_r}{3}\right),
$$

$$
\Omega_r'=\Omega_r(-1+3x^2-3y^2+\Omega_r).
$$

scalar-dominated fixed point는

$$
x_*=\frac{\lambda}{\sqrt6},\qquad
y_*=\sqrt{1-\frac{\lambda^2}{6}},\qquad
\Omega_{m*}=\Omega_{r*}=0
$$

이다. 존재 조건은 $\lambda^2\le6$이고 Jacobian 고유값은

$$
\boxed{\lambda^2-3,\quad\lambda^2-4,\quad
\frac{\lambda^2-6}{2}}.
$$

따라서 물질과 복사를 포함한 미래 안정성은 $\lambda^2<3$, 가속은

$$
\boxed{w_\Theta=-1+\frac{\lambda^2}{3}<-\frac13
\iff\lambda^2<2}
$$

이며, 고정점에서

$$
\boxed{\Theta'=\lambda^2}.
$$

$\lambda=0$은 $f=\infty$인 $\Lambda$ 극한이지 유한 $f$ 작용의 원소가 아니다.

## 5. 완전 반례: 누적 기회비용 자체의 자율 퍼텐셜

literal 경로

$$
V_L(\Theta)=\rho_*(1-e^{-\Theta})
$$

에서는 $V_{L,\Theta}=\rho_*e^{-\Theta}>0$이고

$$
f^2(\ddot\Theta+3H\dot\Theta)+V_{L,\Theta}=0.
$$

임의의 유한 $\Theta_0$와 정지 초기조건에 대해

$$
\boxed{\ddot\Theta(t_0)=-\frac{\rho_*e^{-\Theta_0}}{f^2}<0}.
$$

이는 $H$와 무관하게 누적 측정깊이의 비감소 조건을 즉시 위반한다. 따라서
“누적 기회비용 자체가 source-free 정준 장의 에너지”라는 부모 주장은
완전 반례로 삭제한다. 채택 작용의 $V\propto e^{-\Theta}$는 $c$가 아니라
$u=1-c$인 **남은 자기-비구별성/비선택 잔여량**에 대응한 새 공리다.

## 6. 선형 안정성과 성장식의 범위

정준 장이므로

$$
f^2>0,\qquad c_s^2=1,\qquad
m_{\rm eff}^2=V_{,\phi\phi}=\frac{V}{f^2}
=\frac{\lambda^2V}{M_{\rm Pl}^2}\ge0,
$$

이며 strict inequality는 물리적 $\lambda>0$ branch에서 성립하고
$\lambda=0$ limit control에서는 $m_{\rm eff}^2=0$이다. 허용 domain에는
ghost, short-wavelength gradient instability, tachyonic mass가 없다. 다만

$$
D''+\left(2+\frac{H'}H\right)D'-\frac32\Omega_mD=0
$$

은 GR, 최소결합, pressureless matter, late-time negligible radiation,
$k\gg aH$, $c_s=1$ scalar clustering 억제의 smooth-DE 근사에서만 쓴다.
full Einstein--Boltzmann 결론이 아니다.

## 7. 재현 가능한 배경 IVP

$$
\psi=\frac{\phi}{M_{\rm Pl}},\qquad q=\psi',\qquad
A=\frac{\rho_*}{3M_{\rm Pl}^2H_0^2},\qquad E=\frac H{H_0}
$$

로 놓으면

$$
\boxed{E^2\left(1-\frac{q^2}{6}\right)=
\Omega_{m0}e^{-3N}+\Omega_{r0}e^{-4N}+Ae^{-\lambda\psi}}
$$

이고

$$
\frac{H'}H=-\frac12(3\Omega_m+4\Omega_r+q^2),
$$

$$
\psi'=q,\qquad q'=-(3+H'/H)q+3\lambda\Omega_V.
$$

$N_i=\ln10^{-4}$에서 $\psi_i=q_i=0$으로 시작해 각 $\lambda$의 $A$를
$E(0)=1$에 맞춘다. bisection은 $F(A)$의 일반 단조성 정리가 아니므로 실제
bracket의 부호변화와 반복 중 bracket 보존을 기계적으로 검사한다.

## 8. DESI BAO profile

$m=s g(\lambda)$, $s>0$이면

$$
\widehat s=\frac{g^TC^{-1}d}{g^TC^{-1}g},
$$

$$
\chi^2_{\rm prof}=d^TC^{-1}d-
\frac{(g^TC^{-1}d)^2}{g^TC^{-1}g}.
$$

$\widehat s>0$를 별도 검사한다. 13-vector에서 $\lambda=0$ limit control은
$k=1$, dof $=12$이고, 같은 자료로 $\lambda$도 고르면 $k=2$, dof $=11$이다.

$$
\Delta{\rm AIC}=\Delta\chi^2+2,\qquad
\Delta{\rm BIC}=\Delta\chi^2+\ln13.
$$

$\rho_*$는 flat present-density shooting으로 고정되므로 BAO fit parameter로
세지 않지만, 그 절대값이 이론에서 예측됐다는 뜻도 아니다.

## 9. 형식 지위

| claim | status | reason |
|---|---|---|
| 채택 action의 stress, KG, on-shell 보존 | [정리] | 직접 변분 |
| 지수 퍼텐셜 fixed point, 안정·가속 조건 | [정리] | autonomous Jacobian으로 증명 |
| canonical perturbative sign과 $c_s^2=1$ | [정리] | 명시한 action과 domain에서 성립 |
| smooth-DE 성장식 | [조건부 산출] | subhorizon 등 근사 필요 |
| $u=e^{-\theta}$를 $V/\rho_*$에 대응 | [채택 공리] | operational-to-field 기원 미유도 |
| $\mathcal R_\Theta$와 persistent retention | [미완성] | microscopic map 없음 |
| $\rho_*$ 또는 오늘의 abundance | [외부 보정] | 정보량에서 차원 있는 scale이 나오지 않음 |
| $V_L=\rho_*(1-e^{-\Theta})$의 자율 누적장 | [삭제: 완전 반례] | finite rest data에서 $\ddot\Theta<0$ |
| 본 모형이 phantom $w<-1$을 생성 | [삭제] | positive canonical kinetic이면 $w\ge-1$ |
| 양자경로가 관측 암흑에너지임을 증명 | [미완성] | retention·scale·full likelihood 부재 |

## 10. 제2 수학 감사: Cauchy 완결성과 정확한 shift 퇴화

### 10.1 일반 초기값 문제의 누락 자료

스칼라 방정식은 2계 쌍곡형 방정식이므로 일반 Cauchy 문제에는

$$
\Theta|_{\Sigma_*},\qquad
\Pi_\Theta|_{\Sigma_*}:=f^2n^\mu\nabla_\mu\Theta|_{\Sigma_*}
$$

의 두 함수가 필요하다. 따라서 $\Theta|_{\Sigma_*}=\mathcal R_\Theta$
와 법선 미분의 부호만으로는 일반 해를 정할 수 없다. 완결된 조건부 공리는

$$
\left(\Theta,\Pi_\Theta\right)|_{\Sigma_*}
=\left(\mathcal R_\Theta[\theta,\mu_F],
\mathcal R_\Pi[\theta,\mu_F]\right)
$$

의 쌍이어야 한다. 수치 branch의 $\Theta_i=0$, $q_i=0$은 해당 FLRW 해를
닫지만, operational map에서 유도된 값이 아니라 외부 선택 초기조건이다.

### 10.2 정확한 퇴화와 식별가능성 상한

임의의 상수 $\Delta$에 대해

$$
\widetilde\Theta=\Theta+\Delta,\qquad
\widetilde\rho_*=\rho_*e^\Delta
$$

로 두면

$$
\widetilde\rho_*e^{-\widetilde\Theta}
=\rho_*e^\Delta e^{-(\Theta+\Delta)}
=\rho_*e^{-\Theta},
\qquad
\nabla_\mu\widetilde\Theta=\nabla_\mu\Theta.
$$

그러므로 채택 작용, stress tensor, KG 방정식, background $H(a)$,
smooth-growth 산출과 BAO 벡터가 모두 정확히 불변이다. 특히 $\rho_*$를 오늘의
flat abundance에 맞추어 shooting하면 임의의 $\Theta_i$는 $\rho_*$에 흡수되므로,
background와 BAO는 operational field origin 또는 0차원·양자 기원을 식별하지
못한다. 현재 수치는 표준 exponential quintessence의 조건부 수치로는 유효하지만
그 operational 기원의 검증은 아니다.

operational 원점을 식별하려면 $\rho_*$ normalization과 절대 $\Theta$ level을
독립적으로 함께 예측·고정하여 둘의 곱만 맞추는 shooting을 금지하거나,
$\Theta$의 절대값에 민감한 새 결합·observable이 필요하다. 단순한 gradient,
초기 운동량 또는 표준 선형 perturbation 자체는 상수 shift 퇴화를 깨지 않는다.
$\lambda=M_{\rm Pl}/f$와 $\rho_*$ 역시 현재 이론에서 유도된 값이 아니라 독립
입력 또는 보정값이다.

| claim | corrected status | reason |
|---|---|---|
| $(\mathcal R_\Theta,\mathcal R_\Pi)$가 일반 Cauchy 자료를 준다 | [채택 공리] | 두 map의 microscopic derivation 없음 |
| constant-shift 퇴화 | [정리] | action 수준의 정확한 불변성 |
| 자유 $\rho_*$ 아래 background·BAO가 operational 원점을 식별한다 | [삭제: 완전 반례] | shift가 동일 관측량을 생성 |
| 기존 배경·BAO 수치 | [조건부 산출] | exponential-quintessence branch로는 유지 |
