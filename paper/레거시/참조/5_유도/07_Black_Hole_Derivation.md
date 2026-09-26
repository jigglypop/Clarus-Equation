# 상수장 scalar--tensor 블랙홀 분기

이 문서는 일반 scalar--tensor 유효작용에서 **상수 scalar 분기**가 존재할
조건과 그 분기에서 보존되는 블랙홀 정리만 정리한다. 특정 CE 상수로
중력상수를 바꾸어 shadow, ringdown 또는 증발률의 수치 편이를 예측하지
않는다.

독자는 Einstein 방정식, Killing horizon과 기본 블랙홀 열역학을 안다고
가정한다. 7.1–7.2절은 상수장 branch가 실제 해가 되는 조건을 먼저
고정하고, 7.3–7.5절은 그 조건 아래 보존되는 Einstein 해와 열역학을
도출한다. 7.6절은 이 축약으로 넘어갈 수 없는 hair 문제를 분리하며,
7.7–7.8절은 외부 기준척도의 제한된 역할을 요약한다.

관련 문서:

- [암흑에너지와 scalar--tensor 유도](04_Dark_Energy_Derivation.md)
- [변분 master action](06_Master_Action_Universal_Derivation.md)

## 7.1 작용과 정의역

상수장 branch를 논하려면 먼저 작용의 부호, 차원, 그리고 scalar가
상수라는 말의 정확한 범위를 고정해야 한다. 이 절의 정의역은 뒤의
Einstein 축약과 열역학 계산 모두에 공통으로 쓰인다.

**[공리] 모형 선택:** 부호 규약 $(-+++)$, 자연단위
$c=\hbar=k_B=1$, reduced Planck mass
$M_{\rm Pl}^{-2}=8\pi G$를 쓰고
$$
S=\int d^4x\,\sqrt{-g}\left[
\frac{M_{\rm Pl}^2}{2}F(\phi)R
-\frac12K(\phi)(\nabla\phi)^2
-V(\phi)+\mathcal L_m(g,\Psi)
\right]
$$
를 택한다. 이 절에서는 $\mathcal L_m$이 $\phi$에 직접 의존하지
않는다고 가정한다.

질량차원은
$$
[x^\mu]=-1,\quad [\phi]=[M_{\rm Pl}]=1,\quad
[F]=[K]=0,\quad [V]=4
$$
이다. 따라서 지수나 로그에 차원 있는 양을 직접 넣지 않는다.

**[정의]** $\phi=\phi_0$가 시공간 전체에서 상수인 해를
상수장 분기라 한다. 다음 기호를 쓴다.
$$
F_0=F(\phi_0),\qquad K_0=K(\phi_0),\qquad
V_0=V(\phi_0).
$$

## 7.2 상수장 분기의 존재 조건

작용과 기호를 고정했으므로, 다음에는 $\phi=\phi_0$라는 ansatz가 장방정식
전체와 양립하는 정확한 조건을 확인한다. metric 방정식만 축약하는 것으로는
상수장 해를 얻을 수 없으며 scalar 방정식도 같은 시공간에서 성립해야 한다.

계량과 scalar의 Euler--Lagrange 방정식은
$$
\begin{aligned}
M_{\rm Pl}^2F G_{\mu\nu}
={}&T_{\mu\nu}^{(m)}
+K\nabla_\mu\phi\nabla_\nu\phi
-g_{\mu\nu}\!\left[\frac12K(\nabla\phi)^2+V\right]\\
&+M_{\rm Pl}^2
(\nabla_\mu\nabla_\nu-g_{\mu\nu}\Box)F ,
\\
0={}&K\Box\phi+\frac12K'(\nabla\phi)^2
+\frac{M_{\rm Pl}^2}{2}F'R-V'.
\end{aligned}
$$

**[정리] 상수장 축약:** $F_0>0$이고
$$
\frac{M_{\rm Pl}^2}{2}F'_0R(x)-V'_0=0
\tag{7.1}
$$
가 해 전체에서 성립하면, 상수장 방정식은
$$
G_{\mu\nu}+\Lambda_{\rm eff}g_{\mu\nu}
=8\pi G_{\rm eff}T_{\mu\nu}^{(m)}
\tag{7.2}
$$
로 정확히 축약된다. 여기서
$$
G_{\rm eff}=\frac{1}{8\pi M_{\rm Pl}^2F_0},
\qquad
\Lambda_{\rm eff}=\frac{V_0}{M_{\rm Pl}^2F_0}.
\tag{7.3}
$$

증명은 $\nabla_\mu\phi_0=\nabla_\mu F_0=0$을 위 두 장방정식에
대입하는 것으로 끝난다. 식 (7.1)은 pointwise 조건이다. 따라서
$R(x)$가 일정하지 않은 물질 배경에서는 일반적으로
$F'_0=V'_0=0$ 같은 더 강한 조건이 필요하다.

**[산출]** 진공 Einstein 공간에서는
$R=4\Lambda_{\rm eff}$이므로 scalar 조건은
$$
2M_{\rm Pl}^2F'_0\Lambda_{\rm eff}-V'_0=0
$$
이다. 특히 점근적으로 평탄한 진공 분기는
$$
V_0=0,\qquad V'_0=0,\qquad F_0>0
\tag{7.4}
$$
이면 존재한다.

**[정리] 안정성의 필요조건:** 양의 graviton kinetic term에는
$F_0>0$이 필요하다. Einstein frame에서 scalar ghost를 피하려면
$$
\mathcal K_E(\phi_0)
=\frac{K_0}{F_0}
+\frac{3M_{\rm Pl}^2}{2}
\left(\frac{F'_0}{F_0}\right)^2>0
\tag{7.5}
$$
이어야 한다. 이는 충분한 선형 안정성 정리가 아니다. 유효질량,
경계조건과 모든 perturbation mode의 spectrum도 별도로 검사해야 한다.

## 7.3 Schwarzschild 분기

이제 점근 평탄한 진공 조건을 더하면, 앞 절의 Einstein 축약은 표준
정적 구면대칭 해를 허용한다. 아래 결과는 이 추가 대칭과 경계조건을
사용한 branch의 결과다.

**[산출]** 식 (7.4)와 $T_{\mu\nu}^{(m)}=0$ 아래에서 (7.2)는
점근적으로 평탄한 진공 Einstein 방정식이다. 정적 구면대칭,
연결된 외부영역을 택하면 Birkhoff 정리에 의해
$$
ds^2=-f(r)dt^2+\frac{dr^2}{f(r)}+r^2d\Omega_2^2,
\qquad
f(r)=1-\frac{2G_{\rm eff}M}{r}
\tag{7.6}
$$
를 얻는다.

$$
r_h=2G_{\rm eff}M,\qquad
A_H=4\pi r_h^2
\tag{7.7}
$$
이다. $M>0$이고 asymptotic time을 단위 norm으로 정규화하면
$$
\kappa=\frac12f'(r_h)
=\frac{1}{4G_{\rm eff}M},
\qquad
T_H=\frac{\kappa}{2\pi}
=\frac{1}{8\pi G_{\rm eff}M}.
\tag{7.8}
$$

**[정리] Wald entropy:** 이 작용의 stationary bifurcate Killing
horizon entropy는
$$
S_{\rm W}
=2\pi M_{\rm Pl}^2
\int_H d^2x\,\sqrt h\,F(\phi).
\tag{7.9}
$$
상수장 분기에서는
$$
S_{\rm W}
=2\pi M_{\rm Pl}^2F_0A_H
=\frac{A_H}{4G_{\rm eff}}.
\tag{7.10}
$$
이는 단순히 $G$를 임의 치환한 식이 아니라, 주어진 작용에 대한
Noether-charge 결과다.

## 7.4 비영 cosmological constant와 다른 Einstein 해

점근 평탄 조건을 풀면 같은 상수장 축약은 비영 유효 cosmological constant를
가진 Einstein 공간에도 적용된다. 이 절은 지평선의 수와 온도가 달라지는
경우를 분리하고, 어떤 다른 해가 조건부로 lift되는지 밝힌다.

**[산출]** 상수장 조건을 만족하는 진공 구면대칭 분기에서는
$$
f(r)=1-\frac{2G_{\rm eff}M}{r}
-\frac{\Lambda_{\rm eff}r^2}{3}.
\tag{7.11}
$$
지평선은 $f(r)=0$의 양의 실근이다. de Sitter 부호에서는 black-hole
horizon과 cosmological horizon의 온도가 일반적으로 다르므로,
하나의 전역 평형온도를 자동으로 부여할 수 없다.

**[정리] 조건부 보존:** 식 (7.1)이 성립하고 물질작용이 동일하면,
(7.2)의 모든 Einstein 해는 같은 $G_{\rm eff}$와
$\Lambda_{\rm eff}$를 갖는 상수장 해로 lift된다. 따라서 Maxwell
작용을 포함한 Reissner--Nordström 또는 Kerr--Newman, 진공의 Kerr도
그 전제 아래 보존된다. 질량, 각운동량과 전하는 해당 작용의
asymptotic charge 정의로 정해야 한다.

고정된 bare $G$와 임의의 $F_0$를 비교해 길이를 일정 비율로
변형하는 규칙은 별도 관측 예측이 아니다. 실험으로 정한
$G_{\rm eff}$와 $\Lambda_{\rm eff}$를 일치시키면 이 상수장
분기의 background horizon과 최소결합 test ray의 shadow는 같은 Einstein
metric의 값과 동일하다. 그러나 full perturbation spectrum에는
$\delta\phi$와 metric--scalar mixing이 추가될 수 있으므로 quasinormal
mode까지 자동으로 같지는 않다. 예를 들어 $F'_0=0$이고
$\delta\phi=0$가 일관된 선형 축약일 때에만 순수 metric sector가 GR과
분리된다.

## 7.5 열역학과 증발의 조건부 결과

기하학적 지평선 해를 얻은 뒤에도 열역학과 증발률은 경계 charge와 방출
종의 선택에 의존한다. 다음 결과는 그 추가 전제를 명시한 stationary 또는
반고전 branch에 한정된다.

**[정리]** 고정된 asymptotic coupling, 적분 가능한 Hamiltonian
charge와 stationary bifurcate horizon을 갖는 해의 family에서는
Wald의 first law가
$$
\delta M
=\frac{\kappa}{2\pi}\,\delta S_{\rm W}
+\Omega_H\delta J+\Phi_H\delta Q
\tag{7.12}
$$
형태로 성립한다. 각 항은 선택한 matter action과 경계조건에 맞는
charge여야 한다.

**[산출] 반고전 scaling:** 점근적으로 평탄한 Schwarzschild 분기에서
방출 가능한 종의 질량 문턱 $m_i$ 등은
$\boldsymbol z(M)=\{G_{\rm eff}Mm_i,\ldots\}$ 같은 무차원 조합으로
묶는다. greybody factor와 방출 종을 무차원 함수
$\alpha(\boldsymbol z)$로 모으면
$$
\frac{dM}{dt}
=-\frac{\alpha(\boldsymbol z(M))}{G_{\rm eff}^2M^2}.
\tag{7.13}
$$
$\alpha(\boldsymbol z(M))=\alpha_0>0$를 일정하게 근사하는
질량구간에서는
$$
\tau(M_i\to M_f)
=\frac{G_{\rm eff}^2}{3\alpha_0}
\left(M_i^3-M_f^3\right).
\tag{7.14}
$$
종의 문턱, greybody spectrum, backreaction과 양자중력 종말조건을
주지 않으면 $\alpha$와 전체 수명은 정해지지 않는다.

## 7.6 비상수 scalar와 hair

상수장 축약이 성공했다는 사실은 공간적으로 변하는 scalar를 배제하지
않는다. 바로 그 경우에는 앞 절의 Einstein 해를 그대로 가져올 수 없고,
원래의 coupled 경계값 문제로 돌아가야 한다.

**[미완성]** $\phi=\phi(r,\theta)$인 분기는 식 (7.2)로 축약되지
않는다. 다음을 동시에 풀어야 한다.

1. coupled metric--scalar 장방정식,
2. regular future-horizon 경계조건,
3. 선택한 asymptotic flat/de Sitter/AdS 조건,
4. $F>0$, kinetic positivity와 선형 perturbation 안정성,
5. ADM 또는 covariant phase-space charge의 유한성.

no-hair 결론은 $F,K,V$, matter coupling과 경계조건에 의존한다.
상수장 분기의 존재만으로 hair의 부재나 존재를 결론낼 수 없다.

## 7.7 외부 benchmark의 역할

앞 절의 미완성은 임의의 질량 기준을 넣어 메울 수 없다. 다음 benchmark는
좌표 정규화나 scan에만 쓸 수 있으며, 작용계수나 관측 편이를 유도하는
새 입력으로 승격하지 않는다.

**[공리] benchmark 선택:** 필요하면
$$
M_\star=29.65\,{\rm MeV}
$$
를 좌표 정규화 또는 parameter scan의 외부 기준척도로 둘 수 있다.
허용되는 조합의 예는
$$
G_{\rm eff}M_\star^2,\qquad
M_\star r_h,\qquad
\frac{\Lambda_{\rm eff}}{M_\star^2},\qquad
\frac{M}{M_\star}
\tag{7.15}
$$
이며 모두 무차원이다.

이 선택은 $F_0,V_0,M$, scalar pole 또는 black-hole observable을
고정하지 않는다. 그런 연결에는 독립 작용계수, matching 조건 또는
관측 입력이 필요하다.

## 7.8 보존 범위

상수장 조건 아래에서 실제로 보존된 결과와, coupled scalar 문제로 남은
항목을 마지막으로 분리한다. 다음 표는 이 문서가 주장하지 않는 독립 CE
수치 예측까지 함께 명시한다.

| 항목 | 문서에서 보존되는 내용 |
|---|---|
| 상수장 존재조건 | [정리] 식 (7.1) |
| Einstein 축약 | [정리] 식 (7.2)--(7.3) |
| Schwarzschild--(A)dS | [산출] 식 (7.6), (7.11) |
| Hawking 온도와 Wald entropy | [산출] 식 (7.8), [정리] 식 (7.9) |
| 다른 Einstein black hole | [정리] 동일 matter action 아래의 lift |
| 비상수 scalar hair | [미완성] coupled boundary-value problem |
| 독립 CE 수치편이 | 상수장 분기에서는 산출되지 않음 |
