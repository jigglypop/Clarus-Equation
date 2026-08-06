# CE 연역 강의: 공변 작용에서 검증량까지

## 0. 이 강의의 계약

이 문서는 2026-08-06 최신본의 연역 사슬만 사용한다. 출발점은 숫자 하나가
아니라 다음 네 종류의 자료다.

1. 수학적 정의와 정리
2. 물리 모형을 고정하는 action·projector·branch
3. scale와 scheme이 표시된 외부 입력
4. 입력과 겹치지 않는 검증·holdout

최신 수치는
[`../0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json`](../0_검증과감사/CANONICAL_NUMERIC_MANIFEST_2026-08-06.json),
관측 provenance는
[`../0_검증과감사/OBSERVATIONAL_BASELINE_2026-08-06.md`](../0_검증과감사/OBSERVATIONAL_BASELINE_2026-08-06.md)를
따른다.

## 1. 타입이 지정된 기본 대상

같은 기호가 서로 다른 대상을 가리키지 않도록 먼저 타입을 고정한다.

| 기호 | 대상 | 역할 |
|---|---|---|
| $g_{\mu\nu}$ | Lorentzian metric | causal cone과 중력 |
| $\phi(x)$ | 독립 CP-even scalar | CE EFT의 국소 물리장 |
| $\Phi_H[\gamma,\eta]$ | Hessian probe readout | 경로공간의 투영값 |
| $q$ | Euclidean smoothing field | 확산·정칙화 functional |
| $R$ | Ricci scalar | metric curvature |
| $x$ | 무차원 에너지 생존분율 | bootstrap 변수 |
| $D_N$ | neutral-projector 유효깊이 | 고정점 제어변수 |

$\phi$, $\Phi_H$, $q$, $R$은 등호로 바꾸어 쓸 수 없다. 연결하려면 작용의
coupling 또는 명시적 readout map이 필요하다.

## 2. 공변 EFT

최소 Lorentzian 작용은

$$
S=S_{\rm EH}+S_{\rm SM}
+\int d^4x\sqrt{-g}\,\mathcal L_\phi
$$

로 두고

$$
\mathcal L_\phi=
-\frac{Z_\phi}{2}(\nabla\phi)^2-V(\phi)
-\frac\xi2R\phi^2
-\frac{\lambda_{H\phi}}2\phi^2H^\dagger H
+\sum_{\Delta_i>4}\frac{c_i}{\Lambda^{\Delta_i-4}}\mathcal O_i.
$$

metric signature는 $(-,+,+,+)$다. scalar EOM은

$$
Z_\phi\Box\phi-V'(\phi)-\xi R\phi
-\lambda_{H\phi}\phi H^\dagger H
+\sum_i\frac{c_i}{\Lambda^{\Delta_i-4}}
\frac{\partial\mathcal O_i}{\partial\phi}=0.
$$

stress tensor는 정의에 의해

$$
T_{\mu\nu}^{(\phi)}
=-\frac2{\sqrt{-g}}\frac{\delta S_\phi}{\delta g^{\mu\nu}}
$$

이고, 비최소결합의 기여는

$$
\xi\left(G_{\mu\nu}\phi^2
+g_{\mu\nu}\Box\phi^2-\nabla_\mu\nabla_\nu\phi^2\right)
$$

를 포함한다. on-shell diffeomorphism invariance에서
$\nabla^\mu T_{\mu\nu}^{\rm total}=0$가 따른다.

## 3. Lorentzian 진화와 Euclidean smoothing

세 연산자를 구분한다.

$$
\begin{array}{ll}
\text{Lorentzian Cauchy:}&(\Box-m^2)\phi=J,\\
\text{Poisson semigroup:}&\partial_\chi q=-\sqrt A,q,
\quad q(\chi)=e^{-\chi\sqrt A}q(0),\\
\text{heat semigroup:}&\partial_s q=-Aq,
\quad q(s)=e^{-sA}q(0).
\end{array}
$$

Poisson kernel과 heat kernel은 서로 다르고, Euclidean smoothing이 곧
물리적 시간감쇠나 QFT RG라는 뜻은 아니다. Lorentzian 폐쇄계의 에너지는
보존되며 damping을 쓰면 bath stress까지 합쳐 보존한다.

## 4. $d=3$ 선택 정리

추가 metric 없이 자연스러운 nonzero isomorphism

$$
\Lambda^2V^*\simeq V^*
$$

을 요구하는 모형 클래스에서는

$$
\binom d2=d
$$

이고 $d\ge1$의 비자명 해는 $d=3$이다. 이 결과는 해당 closure class
안에서 정확하다. 표준모형 gauge group은 이 숫자만으로 자동 생성되지
않으므로, 하강 label $3\to2\to1$과 함께 anomaly-free representation을

$$
SU(3)_c\times SU(2)_L\times U(1)_Y
$$

로 지정하고 anomaly 합을 직접 검사한다.

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
s_A^2\mathrel{\stackrel{\rm match}{=}}s_W^2(\mu,\text{scheme})
$$

이다. 여기서 \(s_A^2\)는 CE에 등록한 neutral-mixing 양이고 마지막 등호는
RG·threshold·scheme 변환을 요구하는 별도 matching 조건이다. 이를 채택해
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
f(a)=a+\frac{\alpha_{em}}{4a^{4/3}}
+\alpha_{em}-\frac1{2\pi}=0
$$

을 푼다. 양의 두 근은

$$
a_{\rm low}=0.0528678687,
\qquad
a_{\rm SM}=0.1173186647.
$$

$\alpha_s>\alpha_w>\alpha_{em}$라는 사전 지정 hierarchy가 SM-like
가지를 선택한다. Track A의 입력과 Track B의 출력을 동시에 독립 성공으로
세지 않는다.

## 6. neutral projector와 유효깊이

EWSB 중성 질량행렬을 $(W^3,B)$ gauge basis에서

$$
M_N^2=\frac{v^2}{4}
\begin{pmatrix}g^2&-gg'\\-gg'&g'^2\end{pmatrix}
$$

로 둔다. $P_W,P_B$를 gauge charge가 지정하는 projector,
$P_Z=M_N^2/\operatorname{Tr}M_N^2$를 비영 spectral projector라 하면

$$
p_W=\operatorname{Tr}(P_WP_Z)=c_N^2,
\qquad
p_B=\operatorname{Tr}(P_BP_Z)=s_N^2.
$$

neutral mixing index를

$$
\boxed{\delta_N=p_Wp_B=s_N^2(1-s_N^2)}
$$

로 정의한다. charged projector $P_C$는 $P_W,P_B$의 neutral
부분공간과 직교하므로 이 functional에 0을 주지만 전체 EWSB spectrum에서
제거되지는 않는다. 여기서 \(s_N^2\)는 이 지정 projector의 내부 좌표다.
Track A의 registered boundary \(s_N^2=s_A^2\)를 선택하면

$$
\delta_N=0.1779129995,
\qquad
\boxed{D_N=3+\delta_N=3.1779129995}.
$$

## 7. bootstrap 정리

에너지 가중 생존분율은

$$
x=e^{-(1-x)D_N}
$$

을 만족한다. $D_N>1$일 때 $[0,1]$에는 저분율 해와 경계해 $x=1$이
있다. $I_D=[0,1/D]$에서는 $F_D(x)=e^{-D(1-x)}$가 수축이므로 저분율
해가 유일하다.

$$
\boxed{x=-\frac{W_0(-D_Ne^{-D_N})}{D_N}=0.0486382585}
$$

이고

$$
D_Nx=0.1545681540<1,
\qquad
D(x)=-\frac{\ln x}{1-x}=D_N.
$$

동역학적 실현은

$$
V_D(x)=\frac{x^2}{2}-\frac1D e^{-D(1-x)},
\qquad
\partial_xV_D=x-F_D(x)
$$

와 logit gradient flow를 사용한다. 이로써 고정점 반복과 물리 시간을
같은 것으로 부르는 오류를 피한다.

## 8. B2 에너지 readout

Euclidean 준비 측도를 $d\mu=Z^{-1}e^{-S_E/\hbar}\mathcal D\gamma$,
관측 초곡면의 총에너지를 $E_{\rm tot}$라 하면

$$
d\nu_E(\gamma)=
\frac{E_{\rm tot}(\gamma)d\mu(\gamma)}
{\int E_{\rm tot}d\mu}
$$

가 에너지 가중 측도다. 고정 baryon projector의 에너지를 $E_b$라 두고

$$
x=\frac{\langle E_b\rangle_\mu}
{\langle E_{\rm tot}\rangle_\mu}
$$

로 bootstrap 변수를 정의한다. 평탄 FLRW의 관측면에서
$\rho_{\rm tot}=\rho_c$이므로

$$
\boxed{\Omega_b(a_0)=x=0.0486382585}.
$$

이는 단순 path-count 확률과 에너지분율을 혼동한 등식이 아니다.

## 9. 암흑성분 분할

최신 Track A 분할 functional을

$$
\boxed{R=\frac{\Omega_{DM}}{\Omega_{DE}}
=\alpha_sD_N(1+x\delta_N)=0.3782386966}
$$

로 고정한다. 평탄성에서

$$
\boxed{
\begin{aligned}
\Omega_b&=x,\\
\Omega_{DM}&=(1-x)\frac{R}{1+R}=0.2610881744,\\
\Omega_{DE}&=(1-x)\frac1{1+R}=0.6902735671.
\end{aligned}}
$$

합은 정확히 1이다. 이 배경 분할이 물리적으로 통과하려면 같은
stress-sector action으로 CMB·BAO·SN뿐 아니라 성장률·렌즈·halo를 함께
재현해야 한다.

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
shift-symmetric pseudoscalar $a$를 도입한다.

$$
\mathcal L_a=-\frac12(\partial a)^2
+\frac{\alpha_s}{8\pi}
\left(\bar\theta+\frac a{f_a}\right)
G^a_{\mu\nu}\widetilde G^{a\mu\nu}.
$$

QCD susceptibility가 만드는

$$
V(a)=\chi_{\rm QCD}
\left[1-\cos\left(\bar\theta+\frac a{f_a}\right)\right]
$$

의 최소에서
$\bar\theta+a/f_a=0\pmod{2\pi}$가 된다. instanton action은
$g_s^2=4\pi\alpha_s$ convention에서 $8\pi^2/g_s^2=2\pi/\alpha_s$다.

### 10.3 바리오제네시스

CP-even portal은 1차 상전이를 도울 수 있지만 CP source가 아니다. 최신
EWBG 모형은 $(H^\dagger H)W\widetilde W/\Lambda_{\rm CP}^2$ 연산자,
finite-temperature bounce, diffusion equation과 sphaleron washout을
한 수송계로 푼다. 상세식은
[`../2_경로적분과_응용/08_바리온_비대칭.md`](../2_경로적분과_응용/08_바리온_비대칭.md)에
있다.

## 11. finite-$\xi$ 인플레이션

Jordan-frame action의 비최소결합을

$$
\frac12(M_P^2+\xi\phi^2)R,
\qquad
\xi=\alpha_s^{1/3}=0.4904868132
$$

로 두면 canonical Einstein-frame field는

$$
\left(\frac{d\chi}{d\phi}\right)^2
=\frac{1+\xi(1+6\xi)\phi^2/M_P^2}
{(1+\xi\phi^2/M_P^2)^2}
$$

를 따른다. $N_*=57.1999$, $A_s=2.10\times10^{-9}$를 명시적 입력으로
쓴 quartic benchmark의 재계산은

$$
\boxed{n_s=0.96617114,
\qquad r=0.00434561,
\qquad\lambda_4=1.3434991\times10^{-10}}.
$$

$A_s$는 $\lambda_4$ 정규화에 사용했으므로 별도 예측으로 세지 않는다.
exact $Z_2$에서는 단입자 붕괴가 금지되며 재가열은
$\phi\phi\leftrightarrow HH$ annihilation 또는 preheating으로 계산한다.

## 12. 현재 바리온 수밀도

B2의 $\Omega_b$를 바리온-광자 비로 바꾸는 식은

$$
\eta_b=
\frac{\Omega_b[3H_0^2/(8\pi G)]}
{m_b[2\zeta(3)/\pi^2][k_BT_{\rm CMB}/(\hbar c)]^3}.
$$

$H_0=67.4\,{\rm km\,s^{-1}Mpc^{-1}}$,
$T_{\rm CMB}=2.7255\,{\rm K}$와 $m_b=m_p$를 넣으면

$$
\eta_b^{\rm density}=6.0412\times10^{-10}.
$$

이는 차원 입력을 포함한 변환값이다. EWBG transport가 같은 값을
재현하는지가 독립 closure test다.

## 13. 입력·출력 표

| 항목 | 역할 |
|---|---|
| $\alpha_s(M_Z)=0.1180$ | Track A calibration input |
| $d=3$, neutral projector, $W_0$ low branch | model selection |
| $M_P,v,H_0,T_{\rm CMB},A_s,N_*$ | 각 차원 모형의 명시적 scale/input |
| $s_A^2,\delta_N,D_N,x,R,\boldsymbol\Omega$ | Track A 계산 출력; 물리적 \(s_W^2\) scheme map은 Open |
| $n_s,r$ | 지정 inflation benchmark 출력 |
| $\lambda_4$ | $A_s$로 보정된 coupling |
| CMB·BAO·SN·growth, EDM, collider, GW | likelihood/holdout |

## 14. 반복 검증

```powershell
powershell -ExecutionPolicy Bypass -File docs/0_검증과감사/run_full_consistency_gate.ps1
```

이 gate는 212개 Markdown의 구조와 링크를 검사하고, 결합상수 두 트랙,
bootstrap 두 branch, Gamma count/energy 측도, 밀도벡터, finite-$\xi$
인플레이션을 독립 재계산한다. 코드 residual은 구현의 검증이고,
관측 likelihood는 자연에 대한 별도 검증이다.
