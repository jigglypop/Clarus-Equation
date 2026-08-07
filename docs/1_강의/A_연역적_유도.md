# CE 연역 강의: 공변 작용에서 검증량까지

## 0. 이 강의의 계약

이 문서는 현 checkout의 최신 연역 사슬만 사용한다. 출발점은 숫자 하나가
아니라 다음 네 종류의 자료다.

1. 수학적 정의와 정리
2. 물리 모형을 고정하는 action·projector·branch
3. scale와 scheme이 표시된 외부 입력
4. 입력과 겹치지 않는 검증·holdout

현 checkout에서 재현 가능한 1장 수치, 입력 provenance와 수용 gate는
[D 정합성 원장](D_정합성_원장.md)을 따른다. 삭제된 전역 JSON manifest와
과거 212문서 gate를 1장의 재현 조건으로 요구하지 않는다.

각 항목은 논리적 지위, provenance 역할, 현재 validation 상태의 세 축으로
기록한다. 서로 다른 축을 한 종류의 증거처럼 세지 않는다.

| 축 | 표기 | 뜻 |
|---|---|---|
| 논리 | `Definition` | 기호나 계산량의 뜻을 정한다. 정의만으로 자연의 값이 예측되지는 않는다. |
| 논리 | `Exact` | 정의와 표시된 수학만으로 결론이 닫힌다. |
| 논리 | `Exact conditional` | 명시한 가정 아래에서만 정리가 닫힌다. |
| 논리 | `Convention` | 단위·부호·정규화의 선택이다. 자연의 예측이 아니다. |
| 논리 | `Selection` | 여러 가능한 모형 가운데 이 장에서 채택한 가지다. |
| 논리 | `Bridge` | 수학적 대상과 물리량을 잇는 추가 가정 또는 계산이다. |
| 논리 | `Phenomenology` | 관측과 비교하기 위해 고정한 유효식·benchmark다. |
| provenance | `Calibration input` | 외부 자료나 benchmark에서 넣은 값이다. 독립 예측으로 다시 세지 않는다. |
| provenance | `Calibration output` | calibration input과 표시된 식으로 재계산한 값이다. |
| validation | `Open` / `Rejected` | 아직 닫히지 않았거나 지정된 gate에서 배제된 주장이다. |

`Open`과 `Rejected`는 논리적 역할에 덧붙는 현재 gate 상태다. 예를
들어 `Open Bridge`는 연결식의 역할은 명확하지만 그 연결을 물리 작용이나
자료로 아직 닫지 못했다는 뜻이다. 따라서
`Exact conditional + Selection + Bridge`로 얻은 숫자를 줄여서
`Exact prediction`이라고 부르지 않는다. 각 등호 옆에서 어느 층이 쓰였는지
드러내는 것이 이 장의 정합성 계약이다.

## 1. 타입이 지정된 기본 대상

같은 기호가 서로 다른 대상을 가리키지 않도록 먼저 타입을 고정한다.

| 기호 | 대상 | 역할 |
|---|---|---|
| $g_{\mu\nu}$ | Lorentzian metric | causal cone과 중력 |
| $\phi(x)$ | 독립 CP-even scalar | core EFT의 국소 물리장 |
| $\varphi(x)$ | inflation branch scalar | $\phi$와 동일시하지 않은 별도 benchmark 장 |
| $\chi_E(\varphi)$ | Einstein-frame canonical scalar | $\varphi$의 장 재정의 |
| $q_E$ | Euclidean smoothing field | 확산·정칙화 functional |
| $A_E\ge0$ | Euclidean Hilbert 공간의 self-adjoint generator | Poisson/heat spectral calculus |
| $\ell_P,s_H$ | Poisson/heat smoothing depth | 물리 시간이 아닌 비음수 계산축 |
| $\kappa_{\rm surv}$ | survival 감쇠율 | 중력 결합과 다른 양의 inverse-depth 상수 |
| $\kappa_E$ | energy-biased character $x_E(D)$의 감쇠율 | $\kappa_{\rm surv}$와 자동 동일시하지 않는 양의 inverse-depth 상수 |
| $c_E,\beta_E:=\kappa_E c_E$ | fold depth에서 energy depth로 가는 양의 변환과 무차원 곱 | $\beta_E=1$은 별도 matching `Selection` |
| $R_g:=R[g]$ | Ricci scalar | 길이 차원 $-2$의 metric curvature |
| $x_\star$ | scalar bootstrap의 저분율 고정점 | 수학적 고정점 |
| $x_E$ | 에너지 편향 생존분율 | E1--E4 뒤의 물리 readout 후보 |
| $\Omega_{\rm phys}:=\rho_{\rm tot}/\rho_c$ | 곡률을 제외한 총 물리 에너지 밀도분율 | $1=\Omega_{\rm phys}+\Omega_k$ |
| $R_{\rm dark}$ | $\Omega_{\rm cdm}/\Omega_{\rm DE}$ | 무차원 암흑성분 분할비 |
| $D_A$ | Track-A 선택 깊이 | $3+\delta_A$인 조건부 고정점 제어변수 |

$\phi$, $\varphi$, $\chi_E$, $q_E$, $R_g$는 등호로 바꾸어 쓸
수 없다. $R_g$와 $R_{\rm dark}$는 차원부터 다르다. 서로 연결하려면 작용의
coupling, 장 재정의 또는 단위가 표시된 readout map이 필요하다. 과거의
중성깊이 표기는 물리적 projector에서 자동 유도됐다는 인상을 주므로,
이 장에서는 Track-A 선택값을 일관되게 $D_A$로 쓴다.

## 2. 공변 EFT

이 절의 convention은

$$
(-,+,+,+),\qquad
[\nabla_\mu,\nabla_\nu]V^\rho
=R^\rho{}_{\sigma\mu\nu}V^\sigma,
\qquad
R_{\mu\nu}=R^\rho{}_{\mu\rho\nu},
\qquad
M_P=(8\pi G)^{-1/2}
$$

이다. 즉 $M_P$는 reduced Planck mass다. 곡률 부호를 바꾸면 비최소결합의
표시 부호도 함께 번역해야 하므로, 부호를 숨긴 채 두 작용을 비교하지 않는다.
공변 EFT와 inflation 식은 $c=\hbar=1$ 단위를 쓰되 경로적분 지수에는 차원
장부를 보이려고 $\hbar$를 복원한다. 12절의 수밀도 변환에서는 SI 단위를
전부 복원한다.

최소 Lorentzian 작용은

$$
S=S_{\rm EH}+S_{\rm SM}
+\int d^4x\sqrt{-g}\,\mathcal L_\phi
$$

로 두고

$$
\mathcal L_\phi=
-\frac{Z_\phi}{2}(\nabla\phi)^2-V(\phi)
-\frac{\xi_{\rm core}}2R_g\phi^2
-\lambda_{\rm HP}\phi^2H^\dagger H
+\sum_{\Delta_i>4}\frac{c_i}{\Lambda^{\Delta_i-4}}\mathcal O_i.
$$

따라서 중력계수는

$$
F_{\rm core}(\phi)=M_P^2-\xi_{\rm core}\phi^2
$$

이다. $Z_\phi>0$이고 $F_{\rm core}>0$인 장 영역을 기본 healthy domain으로
둔다. 전자는 Jordan-frame scalar kinetic의 ghost를 피하고 후자는 통상적인
graviton kinetic 부호를 유지한다. 비최소결합 뒤 Einstein-frame 전체 kinetic
행렬의 양성은 선택한 background에서 추가로 확인해야 한다. scalar EOM은

$$
Z_\phi\Box\phi-V'(\phi)-\xi_{\rm core}R_g\phi
-2\lambda_{\rm HP}\phi H^\dagger H
+\sum_i\frac{c_i}{\Lambda^{\Delta_i-4}}
\frac{\delta\mathcal O_i}{\delta\phi}=0
$$

이다. 여기서 $\delta\mathcal O_i/\delta\phi$는 Euler--Lagrange functional
derivative다. $\mathcal O_i$에 $\nabla\phi$가 들어가면 단순 편미분
$\partial\mathcal O_i/\partial\phi$로 바꿀 수 없다.

portal은 이 문서군의 폭 계산과 같은
$\mathcal L_{\rm portal}=-\lambda_{\rm HP}\phi^2H^\dagger H$ 규약을 쓴다.
$-\lambda^{(1/2)}_{H\phi}\phi^2H^\dagger H/2$ 규약을 인용할 때에는

$$
\lambda^{(1/2)}_{H\phi}=2\lambda_{\rm HP}
$$

로 변환해야 한다. 두 규약에 같은 숫자를 넣으면 cubic vertex가 2배,
그 제곱에 비례하는 tree-level 폭이 4배 달라진다.

stress tensor는 정의에 의해

$$
T_{\mu\nu}^{(\phi)}
=-\frac2{\sqrt{-g}}\frac{\delta S_\phi}{\delta g^{\mu\nu}}
$$

이고, 비최소결합의 기여는

$$
\xi_{\rm core}\left(G_{\mu\nu}\phi^2
+g_{\mu\nu}\Box\phi^2-\nabla_\mu\nabla_\nu\phi^2\right)
$$

를 포함한다. on-shell diffeomorphism invariance에서
$\nabla^\mu T_{\mu\nu}^{\rm total}=0$가 따른다.

## 3. Lorentzian 진화와 Euclidean smoothing

세 연산자를 구분한다.

$$
\begin{array}{ll}
\text{Lorentzian Cauchy:}&(\Box-m^2)\phi=J,\\
\text{Poisson semigroup:}&\partial_{\ell_P}q_E=-\sqrt {A_E}\,q_E,
\quad q_E(\ell_P)=e^{-\ell_P\sqrt {A_E}}q_E(0),\\
\text{heat semigroup:}&\partial_{s_H}q_E=-A_Eq_E,
\quad q_E(s_H)=e^{-s_HA_E}q_E(0).
\end{array}
$$

Poisson kernel과 heat kernel은 서로 다르고, Euclidean smoothing이 곧
물리적 시간감쇠나 QFT RG라는 뜻은 아니다. Lorentzian 폐쇄계의 에너지는
보존되며 damping을 쓰면 bath stress까지 합쳐 보존한다.

$A_E\ge0$는 지정한 Euclidean Hilbert 공간의 self-adjoint operator로 둔다.
그러면 spectral calculus로 두 semigroup가 정의된다. $\ell_P$와 $s_H$는
차원이 서로 다를 수 있고, 둘 중 어느 것도 11절의 Einstein-frame 장
$\chi_E$가 아니다.

## 4. $d=3$ 선택 정리

**가정 H.** $V$를 양의 정부호 내적과 orientation을 가진 $d$차원 실수
벡터공간이라 하자. 접힘 입력은 2-form이고 출력은 다음 단계의 1-form이며,
그 사상은

1. 국소적이고 선형이며,
2. $SO(d)$ 회전에 대해 등변적이고,
3. 주어진 metric과 orientation 외의 배경 tensor를 쓰지 않는다.

이때 Hodge star는

$$
*:\Lambda^2V^*\longrightarrow\Lambda^{d-2}V^*
$$

를 준다. 출력 타입이 $V^*=\Lambda^1V^*$와 같으려면 $d-2=1$, 따라서
$d=3$이다. 성분 수 검산도

$$
\binom d2=d
$$

를 주고, $d\ge1$에서 $\Lambda^2V^*\ne0$인 비자명 해는 $d=3$이다.
차원 일치만으로는 자연스러운 동형이 생기지 않는다. 위 결론은 metric과
orientation을 포함한 가정 H 안에서만 `Exact conditional`이다.

이 정리는 공간 차원을 고르는 조건부 closure일 뿐 색의 수 $N_c=3$이나
표준모형 gauge group을 유도하지 않는다. 다음 gauge group과 한 세대의
left-handed Weyl 표현은 별도 `Selection`으로 지정한다.

$$
SU(3)_c\times SU(2)_L\times U(1)_Y
$$

$$
Q_L:(\mathbf3,\mathbf2)_{1/6},\quad
u_R^c:(\bar{\mathbf3},\mathbf1)_{-2/3},\quad
d_R^c:(\bar{\mathbf3},\mathbf1)_{1/3},
$$

$$
L_L:(\mathbf1,\mathbf2)_{-1/2},\qquad
e_R^c:(\mathbf1,\mathbf1)_1.
$$

이 표현에 대해서는 실제 anomaly 합이 다음처럼 0이다. 여기서
$T(\mathbf3)=T(\mathbf2)=1/2$다.

$$
\begin{aligned}
[SU(3)]^3 &: 2-1-1=0,\\
[SU(3)]^2U(1) &:2\left(\frac16\right)\frac12
 -\frac23\frac12+\frac13\frac12=0,\\
[SU(2)]^2U(1) &:3\left(\frac16\right)\frac12
 -\frac12\frac12=0,\\
[U(1)]^3 &:6\left(\frac16\right)^3
 +3\left(-\frac23\right)^3+3\left(\frac13\right)^3
 +2\left(-\frac12\right)^3+1^3=0,\\
[\mathrm{grav}]^2U(1) &:6\left(\frac16\right)
 +3\left(-\frac23\right)+3\left(\frac13\right)
 +2\left(-\frac12\right)+1=0.
\end{aligned}
$$

$SU(2)$ doublet은 색을 센 $3$개의 $Q_L$과 $1$개의 $L_L$, 모두 4개다.
$SU(2)^3$ perturbative anomaly는 pseudoreality 때문에 사라지고,
Witten global anomaly도 없다. 선택적인
$\nu_R^c:(\mathbf1,\mathbf1)_0$와
Higgs·CE scalar는 위 chiral anomaly 합을 바꾸지 않는다. 이 검산은
**선택한 표준모형 표현이
anomaly-free라는 사실**을 보일 뿐, Hodge 정리가 그 표현을 만들었다는
주장은 아니다.

## 5. 결합상수의 두 독립 트랙

CE 경계조건과 표준모형 관계는

$$
C_{\rm CE}=\alpha_s+\alpha_w+\alpha_{em}=\frac1{2\pi},
$$

$$
s_A^2:=4\alpha_s^{4/3},
\qquad
\alpha_{em}=\alpha_ws_W^2,
\qquad
s_A^2\mathrel{\stackrel{\rm Open\ Bridge}{\longleftrightarrow}}
s_W^2(\mu,\text{scheme})
$$

이다. 첫 합 규칙은 같은 scale·scheme에서 시험하는 CE `Selection`이다.
여기서 \(s_A^2\)는 CE에 등록한 neutral-mixing 양이고 마지막 화살표는
RG·threshold·scheme 변환을 요구하는 미완성 matching 문제다. 이를 채택해
미지수를 $(\alpha_s,\alpha_w,\alpha_{em},s_W^2)$로 줄여도 세 독립식만으로는
닫히지 않는다.

### 5.1 Track A

$$
\boxed{\alpha_s^{\overline{\rm MS}}(M_Z)=0.1180}
$$

을 입력하면

$$
\boxed{s_A^2:=4\alpha_s^{4/3}=0.2315097758}
$$

가 조건부 등록 출력이다. 이는 그 자체로 on-shell \(s_W^2\),
\(\overline{\rm MS}\)의 \(\hat s_Z^2\), effective \(\bar s_\ell^2\) 중 어느
것도 아니다. 물리적 electroweak fit과 비교하려면 같은 scale와 convention으로
보내는 scheme map을 먼저 고정해야 한다.

### 5.2 Track B

독립

$$
\alpha_{em}^{\overline{\rm MS}}(M_Z)=1/127.95
$$

를 입력하면

$$
s_{W,B}^2:=4(\alpha_s^{(B)})^{4/3},
\qquad
\alpha_{w,B}:=\frac{\alpha_{em}}{s_{W,B}^2}
$$

라는 matching을 같은 scale·scheme에서 조건부로 채택한다. 이는
$s_{W,B}^2\leftrightarrow s_W^2(\mu,\text{scheme})$의 validation 상태가
`Open`인 `Bridge`를
Track-B benchmark 안에 넣은 것이며, 독립적인 표준모형 예측이 아니다. 그때

$$
f_B(\alpha_s^{(B)})=\alpha_s^{(B)}
+\frac{\alpha_{em}}{4(\alpha_s^{(B)})^{4/3}}
+\alpha_{em}-\frac1{2\pi}=0
$$

을 푼다. 양의 두 근은

$$
\alpha_{s,{\rm low}}^{(B)}=0.0528678687,
\qquad
\alpha_{s,{\rm SM}}^{(B)}=0.1173186647.
$$

$\alpha_s^{(B)}>\alpha_{w,B}>\alpha_{em}$라는 사전 지정 hierarchy가 SM-like
가지를 선택한다. Track A의 입력과 Track B의 출력을 동시에 독립 성공으로
세지 않는다.

## 6. neutral projector, 등록량, fold depth의 분리

### 6.1 표준모형 projector 대수 — `Exact conditional on SM EWSB`

EWSB 중성 질량행렬을 $(W^3,B)$ gauge basis에서

$$
M_N^2=\frac{v^2}{4}
\begin{pmatrix}g^2&-gg'\\-gg'&g'^2\end{pmatrix}
$$

로 둔다. $M_N^2$의 고유값은 $0$과
$M_Z^2=v^2(g^2+g'^2)/4$이므로

$$
P_Z:=\frac{M_N^2}{\operatorname{Tr}M_N^2}
$$

는 massive neutral 방향의 rank-one spectral projector다. 또한

$$
P_W=\begin{pmatrix}1&0\\0&0\end{pmatrix},\qquad
P_B=\begin{pmatrix}0&0\\0&1\end{pmatrix}
$$

로 두면

$$
p_W:=\operatorname{Tr}(P_WP_Z)=\frac{g^2}{g^2+g'^2}=c_W^2,
\qquad
p_B:=\operatorname{Tr}(P_BP_Z)=\frac{g'^2}{g^2+g'^2}=s_W^2.
$$

따라서 지정된 gauge basis에서의 normalized coherence intensity는

$$
\boxed{\delta_{\rm proj}:=p_Wp_B=s_W^2(1-s_W^2)}.
$$

이 등식은 표준모형 중성 projector의 정확한 대수다. 그러나
$\delta_{\rm proj}$는 아직 decay rate, Poisson offspring rate 또는 CE
fold depth가 아니다. charged sector, fermion, ghost와 loop를 없앴다는
뜻도 아니다.

### 6.2 Track-A 등록량 — `Definition`

**provenance:** `Calibration output`.

Track A는 물리적 약혼합각과 다른 내부 등록량

$$
s_A^2:=4\alpha_s^{4/3},\qquad
\boxed{\delta_A:=s_A^2(1-s_A^2)}
$$

를 정의한다. $s_A^2$를 $s_W^2(\mu,\text{scheme})$로 보내는 map과
$\delta_A=\delta_{\rm proj}$의 성립 여부는 RG·threshold·scheme 계산을
요구하는 `Open Bridge`다. 이 장은 두 값을 등호로 놓지 않는다.

### 6.3 additive fold-depth 선택 — `Exact conditional + Selection`

양의 fold block $C_{\rm fold}^\dagger C_{\rm fold}$를 선택하고

$$
\mathcal D_{\rm fold}
=I_3\oplus C_{\rm fold}^\dagger C_{\rm fold},
\qquad
\lVert C_{\rm fold}\rVert_{\rm HS}^2=\delta_{\rm fold}
$$

로 정의하면 trace additivity 때문에

$$
D_{\rm fold}:=\operatorname{Tr}\mathcal D_{\rm fold}
=3+\delta_{\rm fold}
$$

가 정확히 따른다. canonical Track-A benchmark는

$$
\delta_{\rm fold}:=\delta_A
$$

를 **선택**한다. 따라서

$$
\delta_A=0.1779129995,
\qquad
\boxed{D_A:=3+\delta_A=3.1779129995}.
$$

이는 선택한 양의 operator의 trace 정리다. 실제 CE+SM action에서
`vertex → spectral density → 완전양의 jump rate → next-generation
operator`를 유도하는 단계는 `Open Bridge`이며, 그 계산 전에는
$D_A$를 자연의 유일한 유효깊이라고 부르지 않는다.

## 7. vector bootstrap과 scalar 축약 정리

### 7.1 상위 벡터식

재귀 type이 $n$개라면 출발 type $i$가 다음 세대 type $j$를 만드는 평균
수를 비음수 행렬 $\mathsf K_{ij}\ge0$로 둔다. type별 독립 Poisson
offspring를 가정하면 최소 소멸확률 벡터는

$$
x_i=\exp\!\left[-\sum_{j=1}^n
\mathsf K_{ij}(1-x_j)\right],
\qquad i=1,\ldots,n
$$

의 최소 고정점이다. 이 식은 지정한 branching model 안에서
`Exact conditional`이다. $\mathsf K$를 CE+SM action에서 얻는 것은 별도
`Open Bridge`다.

### 7.2 scalar 식이 정확한 두 경우

첫째, $n=1$이고 $\mathsf K=[D]$이면 즉시

$$
x=F_D(x):=e^{-D(1-x)}
$$

를 얻는다. 둘째, 여러 type이더라도 모든 행합이 같아

$$
\sum_j\mathsf K_{ij}=D\quad\text{for every }i
$$

이면 대각 부분공간
$\{\boldsymbol x:x_1=\cdots=x_n\}$가 불변이다. 실제로
$x_j=x$를 대입하면 모든 성분이 같은 $F_D(x)$가 된다. 반대로 행합이
다르면 일반적으로 하나의 scalar 식으로 줄일 수 없다. 임의의 투영
$\mathcal P$에 대해서도 보통

$$
\mathcal P(F_{\mathsf K}(\boldsymbol x))
\ne F_D(\mathcal P(\boldsymbol x))
$$

다. 이 장의 scalar benchmark는 **한 type 또는 공통 행합 균일 sector를
선택했다는 조건 아래에서만** 사용한다.

### 7.3 저분율 branch의 존재와 유일성

$D>1$에서 $I_D=[0,1/D]$를 잡자. $F_D$는 $I_D$를 자기 자신으로 보내며

$$
0\le F_D'(x)=D F_D(x)
\le D e^{1-D}<1
$$

이므로 수축이다. 따라서 $I_D$에는 유일한 고정점 $x_\star$가 있다.
Lambert $W$로 풀면

$$
\boxed{x_\star
=-\frac{W_0(-De^{-D})}{D}},
$$

이고 다른 실수 가지 $W_{-1}(-De^{-D})=-D$는 경계해 $x=1$을 준다.
Track-A 선택 $D=D_A$에서는

$$
\boxed{x_\star=0.0486382585},\qquad
D_Ax_\star=0.1545681540<1.
$$

일반 벡터식의 Jacobian은

$$
J(\boldsymbol x^*)=
\operatorname{diag}(\boldsymbol x^*)\mathsf K
$$

이므로 $\rho(J)<1$이면 고정점 반복이 국소 안정하다. 이는 반복 알고리즘의
안정성이지 물리 시간진화의 안정성이 아니다. $x_\star$도 이 시점에는
수학적 소멸 고정점일 뿐 바리온 에너지분율이 아니다.

## 8. E1--E4 에너지 readout bridge

생존함수의 가정을 S1--S4, 에너지 readout의 가정을 E1--E4로 부른다.
서로 다른 공리군을 모두 “B1--B4”라고 부르는 기존 문서의 namespace
충돌을 이 장에서는 허용하지 않는다.

### 8.1 S1--S4 지수 생존 character

$S:[0,\infty)\to(0,1]$에 다음을 가정한다.

1. **S1, 정규화:** $S(0)=1$.
2. **S2, 연결 곱성:** $S(D_1+D_2)=S(D_1)S(D_2)$.
3. **S3, 정칙성:** $S$는 측정가능하다. 연속성을 가정해도 충분하다.
4. **S4, 비자명성:** 어떤 $D_0>0$에서 $S(D_0)<1$이다.

$f(D):=-\log S(D)$로 두면 S2에서 $f$는 가법함수다. S3에 의해
$f(D)=\kappa_{\rm surv}D$이고, $0<S\le1$에서
$\kappa_{\rm surv}\ge0$, S4에서 $\kappa_{\rm surv}>0$이다. 따라서

$$
\boxed{S(D)=e^{-\kappa_{\rm surv}D}},\qquad \kappa_{\rm surv}>0
$$

가 `Exact conditional`로 따른다.
$\widetilde D:=\kappa_{\rm surv}D$와
$\widetilde S(\widetilde D):=S(\widetilde D/\kappa_{\rm surv})$를 정의하는
`Convention`을 채택하면
$\widetilde S(\widetilde D)=e^{-\widetilde D}$인 단위율 표현을 얻는다. 실제
물리 과정의 비용이 가법인지, 독립 구간의 생존 가중치가 곱해지는지는
이 함수형 정리의 결론이 아니라 `Open Bridge`다.

### 8.2 양의 에너지 측도와 E1--E4

깊이 $D$마다 먼저 정규화된 양의 확률측도 $\mu_D$를 둔다. 이를 형식적
Euclidean 경로적분으로 나타낼 때에는 gauge fixing과 regularization 뒤의
가중치가 실수·비음수이고

$$
0<Z_D:=\int e^{-S_E[\gamma]/\hbar}\mathcal D\gamma<\infty,
\qquad
d\mu_D(\gamma)=Z_D^{-1}e^{-S_E[\gamma]/\hbar}\mathcal D\gamma
$$

를 가정해야 한다. sign/phase problem 때문에 양의 측도가 되지 않으면 아래
확률 정리는 적용할 수 없다. 이제 $H_D(\gamma)\ge0$가
$0<\int H_Dd\mu_D<\infty$를 만족한다고 하자. 생존 사건을
$\mathcal A_D$라 하면 energy-biased measure와 생존 에너지분율은

$$
d\nu_D(\gamma)=
\frac{H_D(\gamma)}{\int H_Dd\mu_D}\,d\mu_D(\gamma),
\qquad
x_E(D):=\nu_D(\mathcal A_D)
$$

이다. 원시 path-count 확률
$p_D=\mu_D(\mathcal A_D)$와 $x_E(D)$는 일반적으로 다르다.

다음 가정을 모두 표시한다.

1. **E1, baryon readout:**
   $H_{b,D}=H_D\mathbf1_{\mathcal A_D}$다.
2. **E2, 동일 장부:** 같은 관측 초곡면·같은 comoving 영역에서
   $\rho_b=\int H_{b,D}d\mu_D$,
   $\rho_{\rm tot}=\int H_Dd\mu_D$로 정의한다.
3. **E3, energy-biased semigroup:**
   $x_E(D_1+D_2)=x_E(D_1)x_E(D_2)$이고,
   $x_E:[0,\infty)\to(0,1]$는 연속이며 $x_E(0)=1$, 어떤 $D>0$에서
   $x_E(D)<1$이다.
4. **E4, feedback depth:**
   fold-trace 단위와 E3의 energy-depth 단위를 잇는 $c_E>0$를 두고
   $D_{\rm act}=c_E(1-x_E)D_A$다.

S1--S4에 쓴 것과 같은 함수형 증명을 E3에 독립적으로 적용하면
$f_E(D):=-\log x_E(D)=\kappa_E D$이고 $\kappa_E>0$이다. 이 표기는
energy-biased character의 rate를 새로 이름 붙인 것이며,
$\kappa_E=\kappa_{\rm surv}$를 가정한 것이 아니다. 실제 실현값을
$x_E:=x_E(D_{\rm act})$로 정의하고 E4를 대입하면

$$
\boxed{x_E=e^{-\beta_E(1-x_E)D_A}},
\qquad
\beta_E:=\kappa_E c_E>0.
$$

optical-depth 단위를 한 단위당 한 e-fold로 정하는 `Convention`
$\kappa_E=1$은 가능하지만, depth 좌표를 바꾸면 $c_E$가 역으로 변하므로
무차원 곱 $\beta_E$는 사라지지 않는다. fold-trace 수치 $D_A$와 energy
optical depth의 matching $\beta_E=1$을 별도 `Selection`으로 채택하고
저분율 branch를 선택하면, $x_E$와 $x_\star$는
같은 수축구간의 같은 방정식을 만족한다. 유일성 때문에 비로소

$$
\boxed{x_E=x_\star}
$$

가 따른다. 이는 같은 문자를 두 번 정의한 등식이 아니라 E1--E4,
$\beta_E=1$, 균일 scalar sector, 저분율 branch 아래의 조건부
결론이다. 특히 $\beta_E=1$을 CE+SM 동역학에서 확보하는 현재 gate는
`Open Bridge`다.

형식적으로 E1--E2를 가정하면

$$
\frac{\Omega_b}{\Omega_{\rm phys}}
=\frac{\rho_b}{\rho_{\rm tot}}=x_E.
$$

$\Omega_{\rm phys}$는 곡률항을 포함하지 않는다. 추가로 관측면에서
$\Omega_k=0$, 따라서 Friedmann closure $\Omega_{\rm phys}=1$을 채택할
때에만

$$
\boxed{\Omega_b=x_E=x_\star=0.0486382585}
$$

라고 쓸 수 있다. E1이 근사적으로
$H_{b,D}=H_D\mathbf1_{\mathcal A_D}+\Delta_D$라면 오차는

$$
\left|\frac{\Omega_b}{\Omega_{\rm phys}}-x_E\right|
\le
\frac{\int|\Delta_D|d\mu_D}{\int H_Dd\mu_D}
$$

로 검증해야 한다.

## 9. 암흑성분 분할과 완전한 밀도 장부

현재 FLRW 장부는 neutrino를 radiation 또는 matter 중 어디에 넣는지 먼저
고정한 뒤

$$
\Omega_{\rm phys}:=\Omega_b+\Omega_{\rm cdm}+\Omega_{\rm DE}
+\Omega_r+\Omega_\nu,
\qquad
1=\Omega_{\rm phys}+\Omega_k
$$

로 쓴다. 여기서 $\Omega_\nu$는 $\Omega_r$이나 $\Omega_{\rm cdm}$에 이미
포함하지 않은 neutrino 성분만 뜻한다. 중복 계상하지 않는다.

Track-A의 암흑비 functional은

$$
\boxed{R_{\rm dark}:=
\frac{\Omega_{\rm cdm}}{\Omega_{\rm DE}}
=\alpha_sD_A(1+x_\star\delta_A)=0.3782386966}
$$

로 **선택**한다. 이 식은 `Phenomenology/Bridge`이며 Friedmann 방정식에서
유도된 항등식이 아니다. 알려진 나머지 장부를 뺀

$$
\Omega_{\rm rem}:=\Omega_{\rm phys}-\Omega_b-\Omega_r-\Omega_\nu
=1-\Omega_k-\Omega_b-\Omega_r-\Omega_\nu
$$

를 정의하면 대수적으로

$$
\Omega_{\rm cdm}=\Omega_{\rm rem}
\frac{R_{\rm dark}}{1+R_{\rm dark}},
\qquad
\Omega_{\rm DE}=\Omega_{\rm rem}
\frac1{1+R_{\rm dark}}
$$

이다. 현재 수치표는
$\Omega_r=\Omega_\nu=\Omega_k=0$으로 둔 **late-time truncated
3-sector benchmark**다. 그 절단 안에서만

$$
\boxed{
(\Omega_b,\Omega_{\rm cdm},\Omega_{\rm DE})
=(0.0486382585,0.2610881744,0.6902735671)}
$$

이고 합이 정의상 정확히 1이다. 일반 dark energy에는
$\Omega_{\rm DE}$를 쓰며, $w=-1$ branch를 별도로 채택할 때만
$\Omega_{\rm DE}=\Omega_\Lambda$라 쓴다. 이 배경 분할의 물리적 통과는
같은 공변 action으로 CMB·BAO·SN·성장률·렌즈를 공동 적합해야 판정된다.

## 10. 입자물리 완성 branch

### 10.1 flavour

질량행렬 하나를

$$
Y_f=U_{fL}\,\operatorname{diag}(y_{f1},y_{f2},y_{f3})U_{fR}^\dagger
$$

로 정의하고
$V_{\rm CKM}=U_{uL}^\dagger U_{dL}$,
$U_{\rm PMNS}=U_{eL}^\dagger U_{\nu L}$를 같은 unitary construction에서
계산한다. 각 행렬원소마다 서로 다른 보정식을 붙이지 않는다. unitarity,
Jarlskog invariant와 전체 covariance가 공동 gate다.

### 10.2 strong CP

CP-even singlet만으로 $\bar\theta$를 0으로 만들 수 없다. 완성 branch는
shift-symmetric pseudoscalar $a_{\rm ax}$를 도입한다.

$$
\mathcal L_{a_{\rm ax}}=-\frac12(\partial a_{\rm ax})^2
+\frac{\alpha_s}{8\pi}
\left(\bar\theta+\frac {a_{\rm ax}}{f_a}\right)
G^a_{\mu\nu}\widetilde G^{a\mu\nu}.
$$

QCD susceptibility가 만드는

$$
V(a_{\rm ax})=\chi_{\rm QCD}
\left[1-\cos\left(\bar\theta+\frac {a_{\rm ax}}{f_a}\right)\right]
$$

의 최소에서
$\bar\theta+a_{\rm ax}/f_a=0\pmod{2\pi}$가 된다. instanton action은
$g_s^2=4\pi\alpha_s$ convention에서 $8\pi^2/g_s^2=2\pi/\alpha_s$다.

### 10.3 바리오제네시스

CP-even portal은 1차 상전이를 도울 수 있지만 CP source가 아니다. 최신
EWBG 모형은 $(H^\dagger H)W\widetilde W/\Lambda_{\rm CP}^2$ 연산자,
finite-temperature bounce, diffusion equation과 sphaleron washout을
한 수송계로 푼다. 상세식은
[`../2_경로적분과_응용/08_바리온_비대칭.md`](../2_경로적분과_응용/08_바리온_비대칭.md)에
있다.

## 11. 독립적인 positive-coupling inflation benchmark

이 절의 $\varphi$와 $\xi_{\rm inf}$는 2절의 core scalar와
$\xi_{\rm core}$가 아니다. 두 branch를 같은 장이라고 주장하려면 하나의
작용에서 부호 번역과 parameter map을 먼저 보여야 한다. 현재는 별도
`Phenomenology` branch로 다음 Jordan-frame 작용을 정의한다. 아래에서
$R_J:=R[g_J]$다.

$$
S_{\rm inf}=\int d^4x\sqrt{-g_J}\left[
\frac12F_{\rm inf}(\varphi)R_J
-\frac12(\nabla\varphi)^2
-\frac{\lambda_4}{4}\varphi^4\right],
$$

$$
F_{\rm inf}(\varphi)=M_P^2+\xi_{\rm inf}\varphi^2,
\qquad
\xi_{\rm inf}:=\alpha_s^{1/3}=0.4904868132.
$$

마지막 식은 Track-A registered matching인 `Selection`이며 작용에서 유도된
등식이 아니다. $F_{\rm inf}>0$인 영역에서

$$
\Omega_{\rm conf}^2:=\frac{F_{\rm inf}}{M_P^2},
\qquad
g^E_{\mu\nu}=\Omega_{\rm conf}^2g^J_{\mu\nu}
$$

로 변환한다. Einstein-frame kinetic term을 canonical하게 만들려면

$$
\left(\frac{d\chi_E}{d\varphi}\right)^2
=\frac{M_P^2}{F_{\rm inf}}
+\frac32M_P^2
\left(\frac{F_{{\rm inf},\varphi}}{F_{\rm inf}}\right)^2
$$

이고, 위 $F_{\rm inf}$를 대입하면

$$
\left(\frac{d\chi_E}{d\varphi}\right)^2
=\frac{1+\xi_{\rm inf}(1+6\xi_{\rm inf})\varphi^2/M_P^2}
{(1+\xi_{\rm inf}\varphi^2/M_P^2)^2}.
$$

Einstein-frame potential은

$$
U(\varphi)=\frac{\lambda_4\varphi^4/4}{\Omega_{\rm conf}^4}
$$

다. 배경 궤적은 finite-$\xi_{\rm inf}$ 식으로 적분하지만 관측량은
leading-order potential slow-roll 식

$$
\epsilon_V=\frac{M_P^2}{2}\left(\frac{U_{,\chi_E}}U\right)^2,
\qquad
\eta_V=M_P^2\frac{U_{,\chi_E\chi_E}}U,
$$

$$
N_*=\frac1{M_P^2}
\int_{\chi_{E,{\rm end}}}^{\chi_{E,*}}
\frac{U}{U_{,\chi_E}}d\chi_E,
\qquad
n_s=1-6\epsilon_{V*}+2\eta_{V*},
\qquad
r=16\epsilon_{V*}
$$

으로 계산한다. 따라서 결과를 “exact observable”이라고 부르지 않는다.
$N_*=57.1999$, $A_s=2.10\times10^{-9}$를 명시적 입력으로 쓴 benchmark는

$$
\boxed{n_s=0.96617114,
\qquad r=0.00434561,
\qquad\lambda_4=1.3434991\times10^{-10}}.
$$

$A_s$는 $\lambda_4$ 정규화에 사용했으므로 별도 예측으로 세지 않는다.
위에 표시한 최소 작용에는 direct SM portal이 없으므로 그 portal이 만드는
perturbative decay·annihilation rate는 0이다. metric과 비최소결합을 통한
gravitational particle production까지 0이라는 뜻은 아니며, 그것은 별도
시공간 계산이 필요한 `Open`이다. 예를 들어 별도 입력
$-\lambda_{H\varphi}\varphi^2H^\dagger H$를 추가하고 exact $Z_2$를
유지하면 단입자 붕괴는 금지되지만
$\varphi\varphi\leftrightarrow HH$ annihilation과 portal preheating은
가능하다. 이 coupling의 크기, reheating·RG·loop 보강은 모두 `Open`이다.

## 12. 현재 바리온 수밀도

E1--E4 bridge의 $\Omega_b$를 바리온-광자 비로 바꾸는 식은

$$
\eta_b=
\frac{\Omega_b[3H_0^2/(8\pi G)]}
{m_{\rm bar}[2\zeta(3)/\pi^2][k_BT_{\rm CMB}/(\hbar c)]^3}.
$$

$H_0=67.4\,{\rm km\,s^{-1}Mpc^{-1}}$,
$T_{\rm CMB}=2.7255\,{\rm K}$와 평균 바리온 질량 근사
$m_{\rm bar}=m_p$를 넣으면

$$
\eta_b^{\rm density}=6.0412\times10^{-10}.
$$

여기서 SI의 $3H_0^2/(8\pi G)$는 mass-equivalent critical density
${\rm kg\,m^{-3}}$로 썼다. 에너지밀도로 쓰면 분자에 $c^2$, 한 바리온
에너지에 $m_pc^2$가 함께 붙어 같은 비에서 상쇄된다.
이는 차원 입력을 포함한 변환값이다. EWBG transport가 같은 값을
재현하는지가 독립 closure test다.

## 13. 입력·출력 표

| 항목 | 역할 |
|---|---|
| $\alpha_s(M_Z)=0.1180$ | Track A calibration input |
| 가정 H 아래 $d=3$ | Hodge-type closure의 `Exact conditional`; gauge group 유도가 아님 |
| SM 표현과 gauge group | anomaly 합을 통과한 별도 `Selection` |
| $\delta_{\rm proj}$ | 지정 SM EWSB 부분공간의 `Exact conditional` 대수 |
| $s_A^2,\delta_A$ | Track-A registered output; $\delta_{\rm proj}$와의 map은 `Open Bridge` |
| $\delta_{\rm fold}:=\delta_A$, $D_A=3+\delta_A$ | additive fold operator의 `Selection/Exact conditional` |
| $x_\star$ | 한 type 또는 공통 행합 sector의 수학적 저분율 근 |
| $x_\star\leftrightarrow x_E\leftrightarrow\Omega_b$ | E1--E4, $\beta_E=1$, 균일 scalar sector, 저분율 branch와 flat-slice 장부를 요구하는 `Open Bridge` |
| $R_{\rm dark},\boldsymbol\Omega$ | late-time truncated `Phenomenology`; 지정 DESI DR2 BAO-only partial gate는 `Rejected`, CMB·SN·growth 공동 gate는 `Open` |
| $M_P,v,H_0,T_{\rm CMB},A_s,N_*$ | 각 차원 모형의 명시적 scale/input |
| $n_s,r$ | 별도 plus-sign inflation branch의 leading slow-roll 출력 |
| $\lambda_4$ | $A_s$로 보정된 coupling |
| CMB·BAO·SN·growth, EDM, collider, GW | likelihood/holdout |

## 14. 반복 검증

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_chapter1_document_contract.py `
  tests\test_chapter1_numeric_contract.py `
  -q -p no:cacheprovider
```

첫 gate는 A·B·C·D의 링크·타입·namespace·부호·밀도 장부와 현행 Track-A,
Track-B, plus-sign inflation 수치를 독립 재계산한다. 기존 수학 구조 회귀는
다음으로 별도 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests\test_core_axioms.py `
  tests\test_bootstrap_solver.py `
  tests\test_dimensionless.py `
  tests\test_core_model_selection.py `
  tests\test_cosmology_ratio_audit.py `
  tests\test_a1_q0_action_bridge.py `
  tests\test_ce_residual_forward_model.py `
  tests\test_clarus_negative_source_search.py `
  -q -p no:cacheprovider
```

기존 일부 구조 테스트는 과거 반올림 상수를 사용하므로 현행 수치의 정본으로
인용하지 않는다. 현행 숫자는 첫 gate와 D 정합성 원장이 담당한다. 코드
residual은 구현의 검증이고, 관측 likelihood는 자연에 대한 별도 검증이다.
