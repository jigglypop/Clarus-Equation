# 36. QFT 재설계 연구 계획서: 관계적 관측량에서 반고전적 회복까지

이 계획서는 양자장론(QFT)을 버리는 계획이 아니다. 검증된 저에너지 QFT를 반드시 되찾는다는 제약 아래, 고정된 시공간ㆍ전역 Fock 공간ㆍ외부 시간에 기대는 현재의 표현을 더 근본적인 관계적 구조에서 회복할 수 있는지 시험하는 계획이다. 만들려는 것은 **관계적 경계자료, 관측량 대수, 양의 상태, 영역 과정, 제약식**으로 이루어진 최소 QFT-next의 명세와, 그것을 탈락시킬 수 있는 계산 순서다.

이 연구가 필요한 이유는 양자론과 일반상대론이 각각 잘 맞더라도, 동적으로 변하는 기하 위에서 무엇이 관측량ㆍ부분계ㆍ시간ㆍ측정인지를 같은 언어로 주지 않기 때문이다. 진행 순서는 공변 작용에서 시작해 제약식을 직접 유도하고, 관계적 관측량과 측정의 확률 구조를 만든 뒤, 절단 독립성과 표준 QFT 및 반고전적 Einstein 방정식의 회복을 차례로 판정한다. 어느 단계에서든 필수 조건이 실패하면 다음 단계로 넘어가지 않는다.

## 한 문장 목표, 현재 판정, 비목표

**목표.** 양자 기하와 물질이 함께 변하는 경우에도 양의 확률, 게이지 불변 관측량, 경계 합성, 저에너지 QFT와 일반상대론의 반고전적 한계를 동시에 만족하는 최소 QFT-next 후보를 구성하거나 반증한다.

**현재 판정.** 고정 배경에서의 표준 QFT는 보존해야 할 성공한 유효 이론이다. 그러나 단일 전역 Fock 공간, 좌표점에 붙은 정확한 국소장, 곱공간으로의 부분계 분해, 모든 관찰자에게 공통인 외부 시간은 양자 중력의 근본 구조로 채택하기에 충분하지 않다. 이 문서는 그 부족분을 메우는 설계안이며, 아직 새 이론이나 관측 예측이 아니다.

**비목표.** 이 계획서는 보편적인 플랑크 시간 틱을 선언하지 않고, 숨은 상태가 자동으로 중력원ㆍ암흑물질ㆍ암흑에너지가 된다고 주장하지 않으며, 기존 CE의 유한 모형을 연속 QFT나 일반상대론의 증명으로 승격하지 않는다. 이 문서는 새로운 우주론 수치나 입자 스펙트럼을 맞추는 작업도 시작하지 않는다.

## 목표 계약

| 항목 | 이 계획에서 고정하는 내용 |
|---|---|
| 최종 목표 | 관계적 QFT-next가 표준 곡률 시공간 QFT와 반고전적 Einstein 방정식을 제어된 극한에서 회복하는지 판정한다. |
| 현재 하위 목표 | 하나의 실제 일반공변 작용 후보를 고르고, clock/rod와 기하ㆍ물질의 정준 제약을 직접 유도한다. |
| 완료 조건 | 제약 대수가 이상 없이 닫히고, 관계적 관측량ㆍ양의 상태ㆍCP 측정ㆍ경계 합성이 정의되며, 고정 배경 QFT와 Einstein EFT 회복 오차를 명시한다. 또는 그중 하나의 kill gate가 명확히 실패한다. |
| 고정 제약 | Born 확률의 양성, 완전양성(CP) 측정, 저에너지 Lorentz 대칭, 재규격화 구조, 총 응력의 단일 출처와 이중계산 금지를 보존한다. |
| 정지 규칙 | 실패한 조건을 새 매개변수나 우주론 readout으로 보정하지 않는다. 실패 원인과 반례 범위를 기록하고 해당 경로를 중단한다. |

## 먼저 정하는 기호와 질문

영역 $R$은 관찰ㆍ상호작용ㆍ경계가 지정된 시공간 부분을 뜻한다. $\mathfrak A(R)$은 $R$에서 실제로 조작하거나 읽을 수 있는 관측가능량의 복소 $*$-대수다. 상태 $\omega$는 각 관측가능량의 평균과 확률을 주는 함수이며, $A\in\mathfrak A$에 대해 다음을 만족해야 한다.

$$
\omega(1)=1,
\qquad
\omega(A^\dagger A)\geq0.
\tag{36.1}
$$

여기서 양성은 음의 확률을 막는 최소 조건이다. $Z_R$은 영역 $R$의 경계 사이를 붙이는 진폭 또는 과정이고, $\mathcal E_R$은 측정 결과를 포함할 수 있는 CP instrument다. $\hat C_\mu$는 좌표 선택이 물리 상태를 바꾸지 않는다는 일반공변 제약이다. 이 기호들은 완성된 이론의 공리가 아니라, 아래 연구에서 시험할 최소 타입이다.

사용자가 든 ‘10개를 보지만 나머지 100개가 중첩으로 남는’ 그림은 입자 수로 일반화하지 않는다. 전체 관측량 대수 $\mathfrak A$와 한 관찰자가 접근하는 부분대수 $\mathfrak A_{\rm obs}\subset\mathfrak A$를 구분하고, 그 관찰자가 갖는 정보는 전체 상태의 제한

$$
\omega_{\rm obs}
= \omega\big|_{\mathfrak A_{\rm obs}}
\tag{36.2}
$$

으로 쓴다. 접근하지 못한 자유도와 중첩은 전체 $\omega$의 상관관계에 남는다. 따라서 ‘보지 않은 양자가 사라진다’는 규칙을 넣지 않는다. 국소 QFT에서는 영역을 언제나 $\mathcal H_R\otimes\mathcal H_{\bar R}$로 나눌 수 없다는 점도 이 선택의 이유다. [Driessler (1977)](https://doi.org/10.1007/BF01609853)는 국소 대수의 type-III 성질을 보이는 표준 배경이다.

## 표준 QFT에서 보존할 것과 일반화할 것

| 보존할 구조 | 일반화할 구조 | 이유와 회복 조건 |
|---|---|---|
| 복소 선형성, 중첩, Born 확률 | 전역 상태벡터를 기본 객체로 삼는 방식 | 상태는 우선 $\omega$로 두고, 필요한 반고전 부문에서만 GNS 표현을 구성한다. |
| 유니터리 닫힌계 진화와 CP 측정 | 외부 시간 $t$에 따른 단일 $U(t)$ | 관계적 시계 또는 경계 과정으로 바꾸되, 고정 배경에서는 통상 유니터리/채널 합성을 회복한다. |
| 국소성, Lorentz 공변성, no-signalling | 좌표점의 정확한 국소장과 고정된 전역 인과순서 | 약한 중력ㆍ반고전 기하 부문에서 미시적 인과성을 회복한다. |
| OPE, RG, Hadamard 단파장 구조, 국소 공변 재규격화 | 하나의 선호 진공과 입자 목록 | 진공과 입자는 상태ㆍ배경 의존 표현으로 나타나야 한다. [Hollands–Wald](https://arxiv.org/abs/gr-qc/0111108) 구조를 회복 기준으로 쓴다. |
| 게이지 대칭과 총 응력 보존 | 날카로운 $\mathcal H_R\otimes\mathcal H_{\bar R}$ 부분계 | 부분대수, 경계 자유도, edge charge와 gluing으로 바꾼다. |

고정된 고전 배경 위 QFT를 시공간에서 대수로 가는 공변 함수자로 정리하는 방법은 이미 있다. [Brunetti–Fredenhagen–Verch](https://arxiv.org/abs/math-ph/0112041)의 locally covariant QFT는 이 연구의 회복 목표이지, 양자 기하 자체의 완성 답은 아니다. 또한 임의의 Cauchy 절단 사이의 진화를 하나의 고정 Fock 공간에서 항상 유니터리하게 구현할 수 있다는 가정은 쓸 수 없다. [Torre–Varadarajan](https://arxiv.org/abs/hep-th/9811222)의 결과가 바로 이 단순화를 막는다.

## 최소 QFT-next 설계 후보

이 계획이 시험할 최소 타입은 다음이다.

$$
\boxed{
(b,\ \mathfrak A,\ \omega,\ Z_R\ \text{또는}\ \mathcal E_R,\ \hat C_\mu)
}
\tag{36.3}
$$

여기서 $b$는 기하, clock/rod 장 $X^A$, 게이지 및 경계 자료를 포함한 관계적 경계자료다. 이 다섯 항목은 각각 ‘어디를 비교하는가’, ‘무엇을 측정하는가’, ‘어떤 확률인가’, ‘영역을 어떻게 합성하는가’, ‘좌표 바꿈을 어떻게 제거하는가’를 맡는다.

좌표 $y$의 값 자체를 관측 위치로 쓰지 않고, 물리적 시계ㆍ자를 나타내는 스칼라장 $X^A$가 $\xi^A$를 가리키는 조건에서 관측량 $\mathcal O$를 읽는 후보는 다음과 같다.

$$
\mathcal O_f[X=\xi]
=
\int d^4y\,\sqrt{-g}\,
f\!\left(X^A(y)-\xi^A\right)\mathcal O(y).
\tag{36.4}
$$

식 (36.4)는 아직 결과가 아니라 M3의 산출물 타입이다. M3는 $f$의 compact support, 각 reference patch에서 $\det(\partial_\mu X^A)\neq0$ 또는 그에 동등한 clock/rod map의 비퇴화 조건, 선택한 gauge dressing, 그리고 해당 dressing이 지나는 edge/boundary sector를 함께 명세해야 한다. 이 네 항목이 없으면 식 (36.4)는 정의가 아니다. 중력 게이지 불변 관측량에는 장거리 dressing이 필요해 정확한 compact locality가 일반적으로 막힐 수 있다. [Donnelly–Giddings](https://doi.org/10.1103/PhysRevD.94.104038)는 이 경계의 핵심 출처다.

영역 포함과 경계 합성은 다음 조건을 만족해야 한다.

$$
\begin{aligned}
R\subset R' &\Longrightarrow
\iota_{RR'}:\mathfrak A(R)\hookrightarrow\mathfrak A(R'),
&& \text{관측량 포함,}\\
\iota_{R'R''}\circ\iota_{RR'}&=\iota_{RR''},
&& \text{포함의 일관성,}\\
Z_{R_2\circ_\Sigma R_1}&=Z_{R_2}\circ_\Sigma Z_{R_1},
&& \text{공통 경계 $\Sigma$에서의 합성.}
\end{aligned}
\tag{36.5}
$$

표현에 의존하지 않는 측정의 기본 타입은 Heisenberg 그림의 normal CP map

$$
\mathcal I_m:\mathfrak A_{\rm out}\longrightarrow\mathfrak A_{\rm in},
\qquad
p(m)=\omega\!\left(\mathcal I_m(1)\right),
\qquad
\sum_m\mathcal I_m(1)=1
\tag{36.6a}
$$

이다. 여기서 normal은 선택한 von Neumann 완비화에서 증가하는 유계 양의 망의 극한을 보존한다는 뜻이다. Kraus 연산자 $K_m$는 GNS 표현 또는 그와 동등한 표현에서 사용할 수 있는 경우의 표현이며,

$$
p(m)=\omega(K_m^\dagger K_m),
\qquad
\sum_mK_m^\dagger K_m=1
\tag{36.6}
$$

을 만족해야 한다. 닫힌 영역에는 유니터리 또는 등거리 합성을 요구하고, 열려 있거나 결과를 읽는 영역에는 CP instrument를 요구한다. 이 구분은 전역 중첩의 보존과 관찰자의 조건부 기록을 동시에 유지한다.

가장 이른 결정 시험은 제약식의 닫힘이다.

$$
\begin{aligned}
\hat C[\xi]\,|\Psi_{\rm phys}\rangle&=0,\\
[\hat C[\xi],\hat C[\eta]]
&=i\hbar\hat C[[\xi,\eta]_{\rm HD}]
+\mathcal A[\xi,\eta],
&& \mathcal A[\xi,\eta]=0.
\end{aligned}
\tag{36.7}
$$

여기서 $[\xi,\eta]_{\rm HD}$는 hypersurface-deformation 조합이고, $\mathcal A$는 양자 이상항이다. 이상항이 남으면 좌표 절단에 따라 예측이 달라질 수 있으므로 이 후보는 탈락한다.

회복 단계에서는 작은 양자 기하 요동 $\epsilon$에 대해

$$
\begin{aligned}
\langle\hat g_{\mu\nu}\rangle_{\omega_\epsilon}
&=g_{\mu\nu}+O(\epsilon),
& \frac{\Delta g}{|g|}&\to0,\\
\frac{\delta\Gamma_{\rm eff}}{\delta g^{\mu\nu}}&=0
&\Longrightarrow\quad
G_{\mu\nu}+\Lambda g_{\mu\nu}
&=8\pi G\langle T_{\mu\nu}\rangle+O(\epsilon)
\end{aligned}
\tag{36.8}
$$

을 제어된 근사로 얻어야 한다. M6에 들어가기 전에 $\epsilon$의 정의, metric fluctuation을 재는 norm 또는 topology, $O(\epsilon)$의 오차 예산, 그리고 $\Gamma_{\rm eff}$가 어떤 topology에서 고정 배경 유효작용으로 수렴하는지를 고정한다. 예를 들어 smearing한 metric correlator의 분산, 국소 관측량의 약한 수렴, 유효작용 변분의 잔차를 서로 다른 항목으로 보고해야 한다. 식 (36.8)은 이 계획에서 가정하는 출발점이 아니라 6ㆍ7단계의 통과 조건이다.

## 채택하지 않는 지름길

보편 플랑크 틱은 모든 관찰자에게 같은 갱신 순서를 강제해 선호 foliation을 만들 수 있으므로 채택하지 않는다. 이산성이 필요하다면 관계적 사건ㆍ국소 인과순서ㆍ연속 극한의 Lorentz 회복을 별도로 보여야 한다.

근본 비선형 mean-field도 채택하지 않는다. 평균장 방정식은 반고전적 근사로는 유용하지만, 근본 상태 진화를 비선형으로 바꾸면 Born 확률ㆍ완전양성ㆍ무신호 조건을 해칠 수 있다. CE의 E53류 계산은 그러므로 회복 단계의 testbed이지 QFT-next의 기본 법칙이 아니다.

정확한 compact locality 역시 채택하지 않는다. 중력 dressing과 경계 전하가 있는 이론에서는 정확한 지역 tensor factorization을 먼저 선언할 수 없다. 대신 반고전 기하 부문에서 관계적으로 spacelike인 두 관측량에 대해 교환자가 사라지는지 확인한다.

$$
P_\gamma[A(R),B(S)]P_\gamma\longrightarrow0
\quad
\text{as }\Delta g\to0,
\tag{36.9}
$$

여기서 $P_\gamma$는 고전 기하 $\gamma$ 근방의 반고전 부문을 고르는 projector다.

## 순서형 마일스톤과 최소 산출물

각 마일스톤은 전 단계가 통과했을 때만 시작한다. ‘가능해 보임’은 통과가 아니다.

### 2026-09-01 진행상황

M0의 기준선ㆍ용어 감사, M1의 공변 네-스칼라 기준장 작용 선택, M2의 **고전** ADM 제약 및 hypersurface-deformation algebra 계산까지 완료했다. 이 결과의 주장 지위ㆍ정확한 계수ㆍ경계는 [E54 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e54-a)에 고정했고, 처음 읽는 독자를 위한 유도는 [37. 공변 기준장과 고전 제약](37_QFT_M0_M2_공변_기준장과_고전_제약.md)에 분리했다. M2의 quantum anomaly-free 대수와 physical inner product는 아직 **미완성**이므로, 아래 M3--M9의 미래 단계와 kill gate는 변경하지 않는다.

M2의 양자 admission 감사에서는 필요한 명제를 더 좁혀 고정했다. 공통 불변 조밀 정의역, regulator와 ordering, 제약 연산자, regulator를 제거한 교환자, 비자명한 물리 상태와 양의 inner product가 한 구성 안에서 제시되어야 한다. 현재 정본에는 이 입력이 없으므로 M2는 **미통과**다. 이는 계산된 $\mathcal A\ne0$이나 전체 모델 클래스의 no-go가 아니며 M3 진입을 허용하지도 않는다. 정확한 계약과 판정은 [E54-H1--H4](../검증_원장/참조_양자_보존_원장.md#qnb-e54-h1)에 둔다.

대안 경로는 reduced/deparametrized quantization, Dirac/RAQㆍmaster constraint, perturbative BRST/EFT, discrete/refinement의 넷으로 분리했다. 네 보통 Klein--Gordon 기준장을 그대로 쓰는 reduced-LQG 경로는 현 M1 작용에서 통과하지 않는다는 범위 제한 반례가 있다. 이 결과는 모든 양자화의 no-go가 아니므로 다른 세 경로를 자동 삭제하지 않는다. 각 경로의 추가 입력과 kill condition은 E54-H3의 portfolio가 정본이다.

reduced 경로의 첫 고전 단계는 [E55 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e55-a)에서 닫았다. 세 rod momentum을 먼저 제거하면 clock momentum은 일반적으로 단순 square root가 아니라 $A P_T^2+2B P_T+D=0$의 두 branch를 따른다. 비직교 rod patch $64$개와 두 branch의 제약 대입 검사는 `tests/test_qft_reference_reduction.py`에서 **2 passed**다. 이 결과는 국소 고전 reduction이며 M2 양자 gate 통과가 아니다.

[E56 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e56-a)은 $A,B,D$가 strongly commuting self-adjoint operator인 부분에서 $h_s$의 self-adjointness와 unitary evolution을 닫는다. 반면 noncommuting Hermitian $A,B$에는 $A^{-1}B$조차 Hermitian이 아닌 $2\times2$ 반례가 있으므로 고전 근의 naive operator 치환은 폐기했다. E55--E56 focused 검사는 **4 passed**이며 실제 field-operator commutator나 physical inner product의 증명이 아니다.

[E57--E59 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e57-a)은 양자 경로를 세 번 더 좁힌다. interacting scalar와 tilted clock의 한-cell regulator에서는 $[B,D]\ne0$이므로 generic strong-commuting 경로가 탈락한다. symmetric scalar form은 positive self-adjoint extension을 갖지만 zero-kernel이 비어 있고, finite master constraint는 kernel 동치를 보장하지만 gap closing만으로 continuum kernel을 만들지 못한다. focused 검사는 **8 passed**다. 이 부정 결과들은 full gravity와 continuum rigging map이 필요한 정확한 위치를 고정한다.

[E60 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e60-a)은 정본 [분포적 refinement toy](18_분포적_refinement와_rigging_map.md)를 master-constraint 언어로 다시 검사했다. 유한 절단의 rank-one 영공간은 존재하지만 zero-extension에서 정규화 projector는 strong하게 $0$으로 사라진다. 반면 $N\langle\phi,\Pi_N\psi\rangle$의 분포적 rigging pairing은 $c_{00}$에서 비자명하게 남고 quotient completion은 $\mathbb C$다. 이 결과와 embeddingㆍ정규화 반례의 focused 검사는 기존 rigging-map 검사와 합쳐 **34 passed**다. 그러나 이는 중력 제약을 포함하지 않은 정확한 toy이며, regulator-independent continuum gravity Hilbert space나 anomaly-free HDA를 만들지 않는다. 따라서 master-constraint 경로의 내부 toy gate만 통과했고 M2는 통과하지 않았다.

[E61 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e61-a)은 BRST 경로에서 먼저 섞이면 안 되는 네 판단을 분리했다. 유한 quartet의 nilpotency와 $H^0/H^1$, closed breaking의 exactness, quotient form의 descent와 양성은 각각 독립 gate다. 실제 M1의 $\Lambda=m=0$ flatㆍconstant-$X^A$ 자유장 one-particle sector에서는 선형 Einstein Ward map $KR=0$, 두 TT quotient와 $\mu_X/k_{\rm ref}>0$인 다섯 scalar kinetic direction을 계산했다. 회전 null momentum, non-exact anomaly, 음의 norm과 Ward 파괴를 포함한 focused 검사는 **18 passed**다. 그러나 constant $X^A$는 reference Jacobian이 퇴화하고 loop ST/QME도 계산하지 않았으므로, 이는 finite/tree gate의 제한적 통과이지 관계적 M2 통과가 아니다.

[E62 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e62-a)은 실제로 비퇴화하는 local reference 배경을 세웠다. Spatially-flat unit-lapse FLRW patch에서 $X^0=T(t)$, $X^i=\beta x^i$를 택하면 clock은 $w=1$, 세 rod는 합쳐 $w=-1/3$의 isotropic stress를 주며 Einstein constraint와 $a^3u$, $ab=\beta/\mu_X$의 무차원 charge가 보존된다. $u\beta\ne0$인 finite-curvature 구간에서 $\det(\partial_\mu X^A)\ne0$이고 timelike clock도 성립한다. 무차원 residual, 퇴화 branch, Ricci-scalar cutoff와 RK4 진화 검사는 **15 passed**다. 다만 이 chart는 local/noncompact이고, nonzero clock branch에는 유한 과거 singularity bound가 있으며, bare kinetic 양성은 perturbation 안정성이나 BRST 양성이 아니다.

[E63 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e63-a)은 E62 배경의 strict high-frequency gate를 분리했다. Minimal harmonic power counting에서 metric--scalar background-gradient mixing은 $O(k)$, curvature/mass는 $O(k^0)$이고, 선언한 two-TT+five-scalar $O(k^2)$ block은 $K=G=\operatorname{diag}(m^2/4,m^2/4,1,\ldots,1)$과 $c^2=1$을 준다. 임의 spatial 방향 TT frame, wrong-sign internal metric, gradient flip, derivative-mixing threshold와 low-$k$ tachyon 반례의 focused 검사는 **15 passed**다. 이는 declared frozen principal sector의 통과이며 full lapse/shift reduction, finite-$k$ 안정성, harmonic constraint propagation이나 quantum positivity가 아니다.

[E64 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e64-a)은 finite-$k$ TT sector를 닫았다. Exponential unimodular tensor metric에서 canonical rod trace Hessian은 $m_T^2=2\mu_X^2\beta^2/(M_P^2a^2)>0$을 주며, declared ADM velocity Hessian과 periodic Christoffel--Ricci spectral integral은 두 편광에서 $K_T=G_T=m^2/4$, $c_T^2=1$을 독립 재현했다. 방향을 맞추지 않은 첫 검산의 $G_T=16.0714$ mismatch를 gate가 reject했고, 회전 정렬 뒤 $\epsilon$ㆍgridㆍwave-number spread까지 포함한 검사는 **12 passed**다. 이 결과는 TT quadratic sector에만 유효하고 cutoff은 supplied scale 비교일 뿐이다.

[E65 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e65-a)은 transverse rod phonon과 metric shift를 함께 남긴 finite-$k$ vector sector를 닫았다. 고정 vector gauge의 주기모드에서 세 rod의 full gradient와 convective velocity, ADM $K_{ij}K^{ij}-K^2$를 직접 수축해 $(\dot\pi,S)$ Hessian을 다시 추출했다. Shift의 algebraic equation을 제거하면 두 transverse polarization에 $K_V>0$, $c_V^2=1$, $m_V^2=2\mu_X^2\beta^2/(M_P^2a^2)=m_T^2$가 남는다. 임의 wavevector, 두 polarization, exact-action finite difference, $k\to0/\infty$, 잘못된 rod sign과 shift를 먼저 0으로 둔 반례까지 묶은 검사는 **14 passed**다. 이 결론은 E62의 local FLRWㆍ고정 gaugeㆍfrozen finite-$k$ ansatz에 한정되며, $k=0$과 시간의존 성장ㆍstrong coupling은 아직 미해결이다.

[E66 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e66-a)은 clockㆍlongitudinal rodㆍlapseㆍlongitudinal shift의 finite-$k$ scalar sector를 닫았다. Flat scalar gauge의 exact ADM 주기모드에서 $(q',r',\alpha,\theta,q,r)$ 6×6 Hessian을 직접 추출하고, nondynamical $(\alpha,\theta)$ block을 Schur 보완으로 제거했다. 양의 $\Lambda$ 기준점에서는 reduced kinetic이 양이고 두 coupled frozen pole이 정확히 $\bar\omega^2=\kappa^2$와 $\kappa^2+M_s^2$로 인수분해된다. $\chi=0$ 배경의 $\chi$는 별도 free spectator로만 추가한다. Exact Hessian, factorization, low/high-$k$, wrong sign, 음의 $\Lambda$의 constraint pole과 “kinetic 양성인데 tachyon” 반례를 묶은 검사는 **15 passed**다. 이는 고정 gauge의 frozen bare quadratic 결과이며 $k=0$, $\beta=0$, 시간의존 성장과 strong coupling은 아직 닫지 않았다.

[E67 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e67-a)은 frozen pole을 사전고정 finite-time screening으로 확장했다. E62 background와 tensorㆍvectorㆍ두 coupled scalarㆍfree $\chi$의 canonical fundamental matrix를 같은 RK4 substep에서 적분하고, 21개 중간 checkpoint의 1000/2000-step refinement, $\Phi^TJ\Phi=J$, $\det\Phi=1$, scalar 동등 2차식 residual과 coefficient domain을 함께 검사했다. $\tau\in[0,0.5]$, $\bar k=\{0.05,0.2,1,3\}$ witness의 6개 검사는 모두 통과했고, 최대 refinement $2.66\times10^{-14}$, 2차식 residual $5.89\times10^{-11}$, symplectic residual $9.77\times10^{-15}$, 선언 scaling의 최대 growth $3.113$이었다. 같은 RK4ㆍ같은 coefficient 계열의 내부 일관성 검사이므로 continuous $k$ band, 독립 적분, physical norm, asymptotic/nonlinear 안정성을 뜻하지 않는다.

[E68 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e68-a)은 cubic admission의 첫 함정을 제거했다. 단일 $\cos(kx)$는 $\langle\cos^3(kx)\rangle=0$이라 상호작용이 있어도 false negative를 만들므로, $(k,k,-2k)$ momentum-conserving real triad로 바꿨다. E66의 $r=ks$ 좌표를 exact ADM one-mode action과 $4.44\times10^{-16}$까지 교차검증한 뒤, static $(q,r)$의 $2\times2\times2$ off-shell cubic tensor를 추출하고 각 external leg를 양의 $K^{-1/2}$로 정규화했다. Flat scalar gauge와 time-independent spatial-rod unitary pullback의 tensor는 선언 오차 안에서 일치했고, step refinementㆍ동일한 두 $k$ leg 교환ㆍcoordinate-measure 음성대조를 포함한 검사는 **5 passed**다.

그러나 이것은 강결합 판정이 아니라 정적 입구 전구체다. 현재 tensor에는 time-dependent/on-shell vertex, second-order lapse/shift와 gauge completion, 모든 frequency/sign assignment, vectorㆍtensorㆍmixed sector가 없다. 네 momentum에서 정준 tensor norm이 유한하다는 사실과 $M_P/\mu_X=10$ power counting은 reduced cutoff을 도출하지 않는다. 따라서 physical strong-coupling scale, one-loop ST/QME, BRST physical Hilbert와 M2는 계속 **미완성**이고 M3--M9는 동결한다. Unitary-gauge EFT와 gravitational-Higgs 문헌은 비교 원리만 제공하며 그 모델별 계수를 이 작용으로 가져오지 않는다.

[E68-G--K 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e68-g)은 이 정적 전구체를 frozen-event dynamic scalar jet으로 확장했다. Lapse cosine $0\ldots4k$와 zero-mean shift sine $k\ldots4k$의 nonlinear projected equations를 풀어 second-order constraint tensor를 기록했고, $N'^y=JN^x-\dot y$와 fixed-$y$ metric derivative를 포함한 time-dependent rod-unitary pullback을 구성했다. Constraint step/grid residual은 각각 $4.30\times10^{-7}$, $9.47\times10^{-11}$, flat/unitary exact action 차이는 $1.13\times10^{-17}$이다. $-\dot y$를 빼거나 constraint를 0으로 둔 음성대조는 각각 $2.04\times10^{-5}$와 $0.1719$로 실패했다.

Dynamic $z=(q',r',q,r)$ cubic Lagrangian tensor를 full gyroscopic pencil의 KG-normalized 두 mode에 투영해, 고정 spatial $(k,k,-2k)$에서 branch/frequency sign 64개를 열거했다. 네 $\kappa$의 최대 magnitude는 $0.03205,0.02465,0.02053,0.02722$이고 두 gauge 차이는 최대 $4.25\times10^{-8}$이며 focused 검사는 **8 passed**다. 이는 frozen quadratic equation을 만족하는 local Lagrangian projection이지 interaction Hamiltonian 또는 S-matrix가 아니다. 따라서 full E68과 strong-coupling scale은 여전히 **미완성**이다.

[E68-L--O 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e68-l)은 velocity-dependent cubic tensor를 canonical phase space로 옮겼다. $p=a^3(Kv-Ry+\partial_v\ell_3)$에서 $v_0=K^{-1}(a^{-3}p+Ry)$를 얻고, analytic 후보 $T^H=-a^3T^L(M_k,M_k,M_{2k})$를 exact reduced action의 momentum을 직접 미분ㆍ역풀이한 finite-amplitude Legendre transform과 비교했다. 네 $\kappa$의 analytic/direct tensor residual은 $8.26\times10^{-6},3.11\times10^{-6},1.50\times10^{-6},9.23\times10^{-7}$이고, stepㆍgridㆍpullback-gauge residual의 최댓값은 각각 $1.67\times10^{-5},5.34\times10^{-9},1.45\times10^{-8}$이다. Pure quadratic $H_2$의 가짜 cubic 신호는 0이고, $M_k$를 생략하거나 $R$ 부호를 뒤집은 음성대조는 모두 실패했다. Active/inactive 성분을 따로 검사한 강화 gate와 **12 passed in 53.72s** 회귀, 독립 수학ㆍ형식 감사까지 통과했다.

이 산출은 고정된 0--4 auxiliary-harmonic truncation과 네 finite momentum의 local phase-space consistency다. Unitary 값은 같은 flat canonical 변수로 pullback한 functional의 일치이지 독립 unitary canonical Hamiltonian이 아니다. Field-redefinition/EOMㆍboundary quotient, finite-time in-in observable, quartic contact와 cubic exchange, vector/tensor/mixed sector가 없으므로 S-matrix나 physical strong-coupling scale로 승격하지 않는다.

[E68-P--R 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e68-p)은 signed E66 pencil의 right-null EOM ideal과 finite-time boundary를 분리했다. Deterministic EOM-exact 변형의 64-mode residual은 네 $\kappa$에서 최대 $7.30\times10^{-14}$인 반면, Hermitian pencil에서 서로 같은 하나의 wrong-$P^T$/gyroscopic-sign 음성대조는 최소 $2.01\times10^{-3}$, 일반 non-EOM 변형은 최소 $6.52\times10^{-2}$다. Stepㆍgridㆍgauge residual 최댓값은 $2.44\times10^{-7},9.50\times10^{-9},4.25\times10^{-8}$이고 focused 검사는 **2 passed in 22.82s**다.

각 momentum의 luminal $k+k-2k=0$ assignment 2개는 frozen boundary endpoint가 0이지만 나머지 62개는 unit-norm deterministic $B$에 대해 $0.251$--$0.833$의 정규화 endpoint를 남긴다. 직접 적분과 endpoint 식의 차이는 최대 $8.03\times10^{-10}$이다. 따라서 finite-time 계산에서 total derivative를 자동 소거하는 규칙은 탈락했다. 다만 이 $A,B$는 full ADM action에서 유도한 field redefinition이 아니므로 complete equivalence theorem이나 physical correlator가 아니다.

[E68-S--U 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e68-s)은 all-creation cubic Hamiltonian vertex를 반복-leg Fock 계수와 함께 유한시간 Dyson kernel에 넣었다. 네 $\kappa$의 여덟 branch triple에서 Hermitian $2\times2$ exact evolution, closed form과 first-Dyson 항을 비교한 active transition 상대오차는 최대 $3.83\times10^{-6}$이다. Vertex stepㆍgridㆍgauge는 최대 $1.72\times10^{-7},7.52\times10^{-9},3.96\times10^{-8}$이고, exact kernel/closed-form residual은 $1.39\times10^{-16},5.49\times10^{-18}$ 이하이다. Wrong-frequency, wrong repeated-leg와 $\lambda=20$ control도 gate를 실패했다.

첫 사후 수학 감사는 별도 boundary 열에서 E68-Q의 $F_B=\frac12By^3$ 계수가 빠져 값이 정확히 2배인 결함을 찾았다. Bulk와 섞이지 않은 열만 $1/2$로 교정하고 직접 회귀를 추가해 최종 검사는 **3 passed in 14.23s**다. 교정된 boundary 범위는 네 $\kappa$에서 각각 $0.0922$--$0.5890$, $0.0925$--$0.3771$, $0.0716$--$0.2583$, $0.0511$--$0.1776$이며 여전히 bulk와 합산하지 않는다. 이 결과는 finite-time two-state first-order consistency이지 physical 3점 함수가 아니다.

[E68-V--X 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e68-v)은 처음의 cubic-exchange 공명 판정을 보강했다. 첫 $\kappa=0.2$ 실행에서 raw $|\mathcal M_{\rm res}|=1.65436\times10^{-8}$가 나왔지만, 같은 성분의 step residual이 $5.44294\times10^{-8}$여서 signal/error가 $0.303946$에 그쳤다. 따라서 이를 비영 공명 결합으로 해석하지 않았고 임계치를 낮추지도 않았다. 대신 cubic order에서 충분한 선형 auxiliary-constraint 해와 $h=(0.08,0.04,0.02,0.01)$의 Richardson 외삽을 실패 뒤 보강 계약으로 고정한 null-aware 경로를 사용했다.

네 $\kappa=\{0.05,0.10,0.20,0.40\}$에서 유일한 exact-luminal branch triple은 모두 raw signal/error $<1$, $|R_6|\le E_{\rm null}$, relative null envelope $<10^{-5}$를 만족했다. 그러므로 허용되는 결론은 해당 frozen scalar triad의 matrix element가 선언한 Richardsonㆍgridㆍgauge 오차 안에서 0과 양립한다는 것뿐이다. Finite-time kernel의 공명극한 $K(\Delta\to0)=\Delta\tau$와 nonresonant bookkeeping weight $X=|\mathcal M|^2/\Delta$는 별도로 검증했지만, regulator의 $1/(i\varepsilon)$ 성장 자체는 물리 poleㆍ폭ㆍ진폭의 증거로 세지 않는다. Wrong-frequency와 wrong-$1/\sqrt2$ 음성대조는 정상 진폭을 각각 적어도 $0.848894$, $0.414214$만큼 바꾸었고 focused 회귀는 **2 passed in 15.47s**였다.

따라서 현재 ansatz에서는 비영 공명 coupling을 근거로 local exchange elimination을 기각하는 반례 경로가 탈락하여 `local_exchange_elimination_rejected=False`다. 이것은 local elimination의 존재ㆍ정칙성ㆍ정확성이나 다른 momentumㆍ다른 sector의 정확한 영점을 증명하지 않는다. $X$도 두 time ordering, intermediate-state 합, finite-time second-Dyson kernel을 갖춘 exchange amplitude나 local Wilson coefficient가 아니다. Complete scalar exchange, quartic contact, physical in-in correlator, strong-coupling scale, loop ST/QME, BRST physical Hilbert, HDA/M2는 계속 **[미완성]**이며 M3--M9는 동결한다.

[E68-Y--Z 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e68-y)은 이 다음
두 subgate를 닫았다. 먼저 velocity-dependent cubic의 induced Legendre 항을 포함한

$$
H_4=a^3\left[-\ell_4(v_0,y)+{1\over2}
(\partial_v\ell_3)^TK^{-1}(\partial_v\ell_3)\right]
$$

을 exact constraintㆍmomentum inversion의 finite-amplitude fourth derivative와
비교했다. 네 $\kappa$와 두 scalar branch의 analytic/direct relative residual
최댓값은 $8.27\times10^{-6}$, stepㆍgridㆍgauge 최댓값은
$3.18\times10^{-9},2.73\times10^{-9},5.94\times10^{-9}$이고 induced-term
omission은 수치오차의 최소 $9.82\times10^3$배였다. 따라서 이 diagonal
normal-ordered contact에서 $H_4=-L_4$만 쓰는 경로는 탈락했다. Focused 회귀는
**2 passed in 21.48s**다.

그 contact와 두 rotating $|1_{2k,c}\rangle$ intermediate를
$H=H_0+\lambda H_3+\lambda^2H_4$ finite star에 넣고
$\lambda=(1,1/2,1/4)$에서 exact evolution과 비교했다. 네 momentum의 최대
normalized exact error는 각각
$4.15\times10^{-7},3.72\times10^{-7},2.95\times10^{-6},
4.46\times10^{-5}$로 모두 $10^{-4}$ 아래였다. 최소 음성대조/수치오차 비는
각각 $198.2,120.6,42.4,26.9$다. 공명 certificate는 branch key와 production
matrix element까지 일치해야 적용되며 null/resolved/unclassifiedㆍmissingㆍmismatch
경로를 fail-closed 회귀로 고정했다. 두-$\theta$ square 적분은 triangle과 독립으로
계산하고 Richardson $N\to2N$ stability까지 gate에 넣었다. 최종 focused 회귀는
cubic exchange **3 passed in 14.06s**, quartic+exchange
**3 passed in 43.21s**이고 독립 재감사도 선언 범위의 잔여 P0/P1/P2가 없다고
판정했다.

이 PASS의 범위는 고정된 두 real harmonic의 **branch-keyed rotating
$s$-channel finite star**다. Counterrotating/signed monomial과 다른 intermediate
Fock state, off-diagonal $2\to2$ contactㆍcrossed routing, SK/in-in correlator,
continuumㆍcutoffㆍrenormalization은 포함하지 않는다. 그러므로 이를 complete
projected $H_3$ exchange, full quartic Hamiltonian, S-matrix 또는 strong-coupling
scale로 부르지 않으며 M2와 M3--M9는 계속 동결한다.

[E68-AA--AB 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e68-aa)은
네 real-harmonic oscillator에서 64개 signed cubic assignment를 normal-order한 뒤,
두 diagonal initial state가 한 번의 $H_3$ 삽입으로 도달하는 Fock target을
algebraically 닫았다. Initial state마다 candidate 12개ㆍactive 4개가 나왔고,
occupation-factor 조립은 독립 Kronecker Fock matrix와 최대
$1.39\times10^{-17}$까지 일치했다. 두 rotating target은 E68-V--X의 branch-keyed
certificate를 재사용하고, 나머지 target의 near resonance는 개별적으로
fail-closed한다.

첫 256/512와 512/1024 grid는 $\kappa=0.05$에서 각각
$1.64\times10^{-8}$, $1.16\times10^{-8}$로 $10^{-8}$ 문턱을 넘어서 실패했다.
임계치를 낮추지 않고 1024/2048로 올렸다. 이어 사후 수학 감사가 active count가
영수증에는 있지만 PASS 조건에 빠진 P1을 찾아 `==4` 조건과 회귀를 추가했고,
네 momentum을 처음부터 다시 실행해 최종 **4/4 PASS**를 얻었다. 최대 exact-star
normalized error는 $4.47\times10^{-5}$, 최대 vertex grid residual은
$9.49\times10^{-9}$, 최소 음성대조/오차 비는 $1.19\times10^3$이고 focused
회귀는 **2 passed in 40.43s**다.

이 결과는 frozen four-oscillatorㆍ채택한 normal-order conventionㆍ유한시간의
**all-signed diagonal-survival consistency**다. Off-diagonal scattering, full Fock
evolution, complete exchange, full quartic matrix, continuumㆍSK/in-inㆍregulatorㆍ
countertermㆍstrong coupling은 포함하지 않는다. 따라서 E68 전체나 M2로
승격하지 않고 M3--M9도 계속 동결한다.

[E69-A--C 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e69-a)은
supplied $O(\hbar)$ breaking을 exactㆍclosed non-exactㆍnon-closed로 구분하는
finite ST cohomology 입장 게이트를 닫았다. 세 differential rank는 $(1,1,1)$,
cohomology 차원은 $(0,1,1,0)$이고, $\Delta=(3/8,0,0)$은
$\Xi=(0,3/8,0)$으로 residual 0에 제거됐다. Anomaly control의 image distance와
non-closed control의 closure residual은 각각 1이다.

첫 감사에서 exact zero remainder만 쓴 자명한 basis check와 unit singular만 쓴
rank evidence의 P1/P2를 찾았다. Nonzero quotient coordinate 0.60의 비직교 basis
불변성, retained singular/threshold $10^{10}$, nilpotency control/tolerance
$10^9$, near-threshold rank ambiguity 탐지를 PASS에 연결해 보강했다. 최종
focused 회귀는 **3 passed in 0.12s**, 독립 재감사의 잔여 P0/P1/P2는 없다.
이는 supplied finite coefficient complex의 algebraic anomaly admission이며 실제
loop integralㆍcontinuum $H^1(s|d)$ㆍST/QMEㆍCTPㆍphysical HilbertㆍM2가 아니다.

[E69-D--F 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e69-d)은
arXiv:1706.02622v7의 Einstein--massless-quantum-scalar Eq. (28)을 로컬
payload로 잠그고 exact rational vector $(43/60,1/40,1/6,1,1)$을 재현했다.
여기서 $Q=R_{\mu\nu}R^{\mu\nu}$, $X=(\nabla\phi)^2$, $Y=\Box\phi$이며,
source background EOM으로 $X=2R$, $Q=R^2$, $Y=0$을 얻어 Eq. (30)의
$D_{\rm on}=203R^2/40$을 확인했다. On-shell 4점과 off-shell 3점의 항등식
residual은 정확히 0이고, Eq. (28)/(30) 혼동, $\bar\phi=0$에서 scalar loop를
제거하는 shortcut, $X^2$ㆍ$RX$ 누락, 잘못된 EOM과 $Y$ 차수 대조는 모두
nonzero로 검출됐다.

첫 수학 감사가 단항식 차원을 $L^{-4}$로 하드코딩한 P1을 찾아냈다. 이를
$[R_{\mu\nu}]=[R]=[X]=[Y]=L^{-2}$에서 ordered factor의 지수를 실제 합산하는
계산으로 바꾸고 훼손된 $[X]$ 대조를 추가했다. 최종 focused 회귀는
**5 passed in 0.06s**, production receipt는 PASS다. 이 결과는 local
transcription checksum과 supplied 4차원 EOM 아래의 source reproduction이다.
SHA-256은 내려받은 논문 artifact나 원격 source의 독립 인증 hash가 아니며,
Gauss--Bonnet flag도 finite-boundary completion을 뜻하지 않는다. Heat-kernel
traceㆍghost determinant의 재구성, loopㆍrenormalization, pure Einstein,
continuum ST/QMEㆍCTPㆍphysical HilbertㆍHDA/M2는 여전히 미완성이므로
M3--M9는 계속 동결한다.

[E69-G--I 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e69-g)은 source
Eq. (23)/(27)의 $n$-의존 coefficient formula를 $n=4$에서 각각 평가하고,
ghost weight $-2$와 4D integrated-bulk Gauss--Bonnet quotient를 거쳐
Eq. (28)을 재구성했다. 공통 $1/360$ 이전 vector는
$g_{\rm 2+gf}=(382,-1102,595,-720,-60,720,360)$,
$g_{\rm gh}=(-22,172,80,-360,-60,180,0)$,
$g_{\rm raw}=(426,-1446,435,0,60,360,360)$이다. 한 source lane의 잘못된
$P=-4$ 산출은 원문 분모와 최종 $P=0$ 소거에 모두 모순되어 폐기하고
$P=-2$로 정정했다.

세 bulk representative는 exact residual 0, 깨진 대표는 $71/420$이고,
wrong ghost/basis/GB, $R^2$ㆍ$RX$ㆍ$X^2$ 생략, $n=5$ 혼용과 차원 훼손
controls가 모두 nonzero다. 감사 보강 뒤 focused 회귀는 **5 passed in
0.08s**, production receipt는 PASS이며 최종 P0/P1은 없다. 이는 source
coefficient formula assembly일 뿐 Laplace operatorㆍEq. (22) traceㆍheat-kernel
coefficientㆍghost determinant의 독립 도출이 아니고, finite boundaryㆍ
$n\ne4$ evanescentㆍrenormalizationㆍST/QMEㆍHDA/M2도 미완성이다.

[E69-J--L 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e69-j)은 source
Eq. (19)와 supplied Eq. (22), ghost Eq. (24)--(26)의 trace input을
rational-polynomial pair로 조립해 Eq. (23)/(27)의 일곱 성분을 각각
항등 재현했다. 총 14개 cross-product residual polynomial은 0이고,
$n=3,4,5$의 42 exact component가 모두 맞았으며 $n=2$ pole은 거부됐다.
$n=4$ 상수 복사 impostor mismatch는 Eq. (23) $538/45$, Eq. (27) $1/20$이다.

여덟 symbolic 음성대조는 각각 1개 이상 nonzero component를 냈고, 모든
invariant와 universal formula contribution의 $L^{-4}$ 차원도 primitive
exponent에서 산출했다. Focused 회귀는 **6 passed in 0.11s**, production
receipt는 PASS이며 독립 감사 P0/P1은 없다. 이는 supplied trace identity의
source-consistency 증거이지 Eq. (19), Eq. (22) trace tensor, ghost determinant와
weight의 독립 도출이 아니다. Total derivativeㆍboundaryㆍevanescentㆍloopㆍ
renormalizationㆍST/QMEㆍHDA/M2는 계속 미완성이다.

[E69-M--O 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e69-m)은 source-v7
Eq. (24)--(26)의 ghost potential과 vector curvature action을 Euclidean
$n=3,4,5$ exact rational fixture에서 직접 수축했다. Generic/zero-vector 두
fixture의 네 식, 총 24개 residual은 모두 0이고 세 curvature audit와 훼손
거부도 통과했다. Frobenius/Ricci/$vv^{\mathsf T}$/cross-term/$W$-index/generic
fixture/rank-deficient identity 음성대조는 모두 nonzero다.

첫 수학 감사가 불변량 차원 기저의 $X$ 표기 오류와
$\operatorname{tr}\mathbb I=n$의 자명한 `n-n` 검사를 P1로 찾아냈다. 기저를
$(E,Q,R,X,P,X^2)$로 명시하고 실제 $\delta_{ab}$ 행렬 trace 및 rank-$(n-1)$
대조를 추가한 뒤 focused 회귀는 **7 passed in 0.18s**, production receipt는
PASS, 수학ㆍ형식 재감사는 P0/P1 없음이다. 이 gate는 $W^2$에 보이지 않는
$W$의 선형 전체 부호를 판정하지 않으며, FP operator/determinantㆍghost weightㆍ
Lorentzian/globalㆍheat kernel/loopㆍboundaryㆍST/QMEㆍHilbertㆍHDA/M2는
계속 미완성이다.

[E69-P--R 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e69-p)은
$\operatorname{Sym}^2V\oplus1$의 raw representation matrix를 직접 만들어
$n=3,4,5$의 identity trace와 세 genericㆍ두 Weyl-added curvature trace를
검산했다. Bundle rank는 $(7,11,16)$이고 총 8개 exact residual은 모두 0이다.
$n=4,5$의 별도 Weyl tensor는 nonzeroㆍRicci-flat이며 이를 버리는 shortcut은
mismatch 208로 실패했다.

기저 정규화, half/omitted/relative-sign action, curvature index, Frobenius trace,
scalar identity와 dimension controls도 모두 nonzero이고 focused 회귀는
**7 passed in 8.99s**, production receipt는 PASS다. 세 독립 감사의 P0/P1은
없다. 이는 local finite Euclidean representation의 두 trace만 닫으며
Eq. (18), Eq. (22)의 $\mathcal Y$ 두 trace, determinantㆍheat kernelㆍloopㆍ
Lorentzian/globalㆍST/QMEㆍHilbertㆍHDA/M2는 계속 미완성이다.

[E69-S--U 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e69-s)은 source
Eq. (17)--(18)의 raw DeWitt metric과 potential을 local normal-coordinate
point에서 직접 구성했다. Rank $(7,11,16)$ metric inverse residual은 모두 0,
$n=2$ pole은 거부됐고 9개 fixture의 trY 및 raw-trY2/bulk quotient, 총
18개 exact residual이 모두 0이다. Flat
$H=\operatorname{diag}(1,2,4)$ 표본은
$\mathfrak D=-28$, raw-minus-bulk $=-112=4\mathfrak D$를 냈다.

14개 잘못된 metric/basis/block/trace/divergence/coefficient/Weyl/$n$ 변형은
모두 nonzero이고 focused 회귀는 **8 passed in 2.77s**, production receipt는
PASS다. Source 감사에서 원문의 partial Hessian 표기를 global하게 공변화할
위험을 찾아 local normal-coordinate point로 계약과 hash를 다시 잠그고
재실행했다. 최종 수학ㆍ형식ㆍ1차출처 감사에는 P0/P1이 없다. Source가
IBP를 명시했다거나 pointwise Eq. (22), boundary 또는 integrated action을
증명했다는 뜻은 아니며 operatorㆍheat kernelㆍloopㆍHDA/M2도 미완성이다.

[E69-V--X 원장](../검증_원장/참조_양자_보존_원장.md#qnb-e69-v)은 source
Eq. (11)/(13)의 linearized gauge jet에서 Eq. (24)--(26)의 local FP action을
구성하고, 별도 exterior algebra에서 finite Berezin determinant를 직접
계산했다. 9개 curvature/vector fixture의 commutator와 두 FP 관계, 총
72개 exact component는 모두 0이고 $N=1,2,3$ Berezin/Leibniz 값은
$(2,7,16)$이다. Singular reference는 ratio 전에 거부하며 scale ratio는
$(2,4,8)$, 선언한 effective-action exponent의 비는 $-2$다.

12개 부호ㆍtraceㆍscalarㆍRicciㆍorientationㆍinverseㆍmultiplicity 대조는
모두 nonzero이고 focused 회귀는 **8 passed in 0.21s**, production receipt는
PASS다. 첫 수학 감사가 gauge parameter의 길이 차원을 P1로 지적해
$[\xi]=L$, $[\delta\chi]=L^{-1}$, $[\Delta_{\rm FP}]=L^{-2}$로 바로잡았고,
수정 안정판의 수학ㆍ형식ㆍ1차출처 감사에는 P0/P1이 없다. 이는 local finite
FP/Berezin consistency이며 global determinantㆍfunctional measureㆍheat
kernelㆍloopㆍBRST/QMEㆍHilbertㆍHDA/M2는 계속 미완성이다.

| 순서 | 연구 패키지: 입력 → 작업 → 최소 산출물 | 통과 조건 | 탈락ㆍ중단 조건 | 선행 의존성 |
|---|---|---|---|---|
| M0 | 기준선ㆍ용어: 표준 QFT, LCQFT, 제약, CP, 관계적 관측량의 정의와 회복 기준을 한 계약으로 고정한다. | 기호ㆍ단위ㆍ상태/관측량/과정의 타입이 혼동 없이 정리된다. | 입자 수, 좌표 라벨, 관찰자 기록을 같은 객체로 섞으면 중단한다. | 없음 |
| M1 | 공변 작용과 clock/rod: 기하 $g_{\mu\nu}$, 물질, reference field $X^A$, 게이지ㆍ경계항을 가진 실제 작용 하나를 고른다. | 작용, 변분원리, 경계조건, 자유도 수가 명시된다. | ghost, 잘못된 경계 변분, 또는 gauge/reference 선택을 바꿔도 남는 관측 가능한 물리적 preferred frame이 필수라면 탈락한다. 관계적 clock을 gauge/reference로 고르는 일 자체는 탈락 조건이 아니다. | M0 |
| M2 | 정준 제약과 이상항: M1 작용을 ADM 또는 동등한 방법으로 분해하여 $\hat C_\mu$와 괄호 대수를 계산한다. | 식 (36.7)의 이상항 0 또는 조절 가능한 소거 증명을 제시한다. | anomaly, 음의 norm, gauge fixing에 따른 물리 예측 차이가 나면 탈락한다. | M1 |
| M3 | 관계적 관측량 대수: 식 (36.4)의 compactly supported $f$, 비퇴화 reference patch, dressing, edge/boundary sector, 영역 포함을 구체화한다. | gauge-invariant 관측량과 inclusion map을 명시하고 고정 배경 극한을 보인다. | 정확한 국소 tensor factorization을 억지로 가정하거나 dressingㆍedge sector가 정의되지 않으면 중단한다. | M2 |
| M4 | 상태ㆍGNSㆍ측정: 양의 $\omega$에서 표현을 구성하고, 실제 측정 영역의 CP instrument를 정의한다. | 식 (36.1), (36.6), 비선택 상태 보존과 no-signalling 검사를 통과한다. | 음의 확률, 비CP map, 사후선택을 물리적 신호로 쓰는 경우 탈락한다. | M3 |
| M5 | 경계 gluing과 절단 독립성: 같은 관계적 경계자료를 공유하고 같은 관측량 정의ㆍedge 조건을 보존하는 foliation/경계분해의 범위를 입력으로 고정한다. 두 분해의 관측량 대수 사이 comparison isomorphism $\Phi_{\Sigma\to\Sigma'}$를 구성하여 식 (36.5)를 계산한다. | $\Phi_{\Sigma\to\Sigma'}$ 아래 대응 관측량의 확률과 합성이 일치한다. | 허용 범위 안에서 foliation dependence, comparison isomorphism 부재, 또는 gluing 비결합성이 남으면 탈락한다. | M4 |
| M6 | 고정 배경 QFT 회복: M6 시작 전에 $\epsilon$, metric fluctuation norm/topology, $O(\epsilon)$ 오차 예산, $\Gamma_{\rm eff}$ 수렴 판정을 고정하고, 고전 기하 부문에서 LCQFT, Hadamard, OPE, RG, 국소 공변 재규격화를 비교한다. | 평탄ㆍ곡률 배경의 표준 QFT 예측과 사전 고정한 오차 예산을 제시한다. | 표준 QFT 한계, Lorentz 대칭, 재규격화 회복 또는 사전 고정한 수렴 기준에 실패하면 탈락한다. | M5 |
| M7 | 반고전 Einstein 회복: M6의 norm/topology와 오차 예산을 유지한 채 총 유효작용과 응력텐서의 변분, Ward identity, backreaction을 시험한다. | 식 (36.8), 총 응력 보존, source의 단일 provenance와 유효작용 수렴이 성립한다. | stress 이중계산, 비보존, 불안정한 중력 부호 또는 수렴 기준 위반이 나오면 탈락한다. | M6 |
| M8 | 기존 CE testbed 재배치: E41은 유한 Gram/관측 접근 예시, E44는 CTPㆍinfluence 과정, E53은 반고전 평균장 회복의 제한된 수치 testbed로 둔다. | 각 모형의 입력ㆍ출력ㆍ연속 이론으로 승격할 수 없는 경계가 명시된다. | 유한 성공을 QFT/GR 증명으로 읽으면 중단한다. | M4, M7 |
| M9 | Clarus interaction/source와 우주론: M7과 M8 뒤에만 $J_C$, $T^{\rm Clarus}_{\mu\nu}$, cosmological readout을 별도 작용에서 시험한다. 평가 전 별도 프로토콜로 observable, likelihood, data split을 사전등록하고 holdout을 고정한다. | source, Ward, 관측 예측, 사전등록한 독립 holdout이 함께 제시된다. | 숨은 확률ㆍ기록을 에너지로 재명명하거나 기존 응력을 중복 가산하거나 사전등록 뒤 holdout 정의를 바꾸면 탈락한다. | M7, M8 |

## 공통 kill gates

다음 중 하나는 설계안의 우회 가능한 불편함이 아니라 실패 판정이다.

- 제약 대수의 이상항이 사라지지 않는다.
- 양의 확률 또는 CP instrument가 깨진다.
- 허용된 foliationㆍclock 선택에 따라 관측 확률이 달라진다.
- 평탄 또는 곡률 고정 배경 QFT의 Lorentz 공변성ㆍHadamard/OPE/RG 구조를 회복하지 못한다.
- 보편 틱이나 숨은 선호 frame이 관측 가능한 Lorentz 위반을 강제한다.
- 중력 dressing이 필요한데 정확한 compact locality를 공리로 고집해 모순이 생긴다.
- full state, retained field, influence functional에서 이미 계산한 응력을 다시 더해 총 응력을 이중계산한다.

## CE와의 연결 및 현재 지위

CE의 출발 동기는 [선택과 접힘](../5_유도/00_선택과_접힘.md)에 정리된 **끼임(환경이 강제하는 선택) → 접힘(비선택 성분의 보존) → 암흑 표현(접힌 에너지의 우주론 readout)** 이다. 이 계획은 그 서사를 중력원으로 즉시 바꾸지 않는다. 먼저 비선택 성분을 전체 상태와 접근 부분대수의 관계로 정확히 적고, 그 다음에만 공변 관측량ㆍsourceㆍ응력ㆍ우주론이라는 다리를 하나씩 시험한다.

[현재 상태와 열린 문제](05_현재_상태와_열린_문제.md), [플랑크 렌더링 경계와 0D→GR 브리지](10_플랑크_렌더링_경계와_0D_GR_브리지.md), [양자 극장 개장](../5_유도/09_양자_극장_개장/00_읽기_순서.md)은 각각 기존 후보와 제한된 증인을 제공한다. 이 계획은 그 결과의 지위나 수치를 변경하지 않는다. E41/E44/E53은 각각 관측 접근의 유한 예시, 환경을 적분한 과정의 예시, 제한된 반고전 평균장 계산으로만 재배치한다.

현재 형식 지위는 다음과 같이 고정한다.

| 항목 | 현재 지위 |
|---|---|
| LCQFT, type-III 국소대수의 경계, dressing 비국소성, 재규격화 회복 요구 | **[정리]** 또는 기존 이론의 확립된 결과 |
| 식 (36.3)의 타입, 관계적 경계자료, gluing 요구 | **[정의 후보]** |
| 양성ㆍCPㆍ반고전적 미시적 인과성 회복을 기본 제약으로 두는 선택 | **[공리 후보]** |
| M1 공변 작용과 고전 ADM/HDA | **[정리: 조건부/범위 제한]** |
| 양자 제약 닫힘ㆍ양의 physical Hilbert, 관계적 대수, 완전한 회복 증명 | **[미완성]** |
| 독립 새 관측 | **[예측: 없음]** |

## E69-A 사전등록과 완료 계약

다음 subgate는 **E69-A: supplied $O(\hbar)$ ST breaking의 유한 cohomology
입장 게이트**다. 이는 실제 Feynman 적분에서 one-loop breaking을 산출하는
계산이 아니다. 먼저 E61의 quartet complex를 확장한 고정 coefficient-space에서
“closed”, “counterterm으로 제거 가능”, “closed지만 non-exact anomaly control”,
“non-closed consistency failure”를 서로 구분할 수 있는지 검증한다.

Ghost-number complex를

$$
C^{-1}\xrightarrow{B_{-1}}C^0\xrightarrow{B_0}C^1
\xrightarrow{B_1}C^2,\qquad
C^0=(x,q,B),\quad C^1=(c,a,u),\quad C^2=(v)
\tag{36.12}
$$

로 두고 $B_{-1}\bar c=B$, $B_0q=c$, $B_1u=v$만 비영으로 고정한다.
따라서 $B_0B_{-1}=0$, $B_1B_0=0$이고, $a$는 의도적으로 넣은 closed
non-exact $H^1$ representative다. 이는 continuum local BRST complex가 아니라
gate가 anomaly 후보를 잘못 제거하지 않는지 검사하는 유한 음성대조다.

부호는

$$
\mathcal S(\Gamma_0+\hbar\Gamma_1)
=\hbar\Delta+O(\hbar^2),\qquad
\Gamma_1^R=\Gamma_1-\Xi,\qquad
\Delta^R=\Delta-B_0\Xi
\tag{36.13}
$$

여기서 $\hbar$는 물리 단위를 가진 상수가 아니라 perturbative order를 표시하는
formal dimensionless parameter다. $\Gamma_1,\Delta,\Xi$의 수치 성분도 각
operator의 선언 reference scale로 나눈 dimensionless coefficient coordinate다.
이 정규화는 차원 정합만 보장하며 continuum operator matching을 뜻하지 않는다.

로 고정한다. 독립 입력 breaking은 $\Delta_{\rm in}=(3/8,0,0)$으로 하며,
solver에는 정답 counterterm을 주지 않는다. $B_1\Delta_{\rm in}=0$과
$\Delta_{\rm in}\in\operatorname{im}B_0$를 각각 검사한 뒤 pseudoinverse가
$\Xi_*$를 구하고 $\|\Delta_{\rm in}-B_0\Xi_*\|$를 기록한다. 별도
$\Delta_{\rm anom}=(0,1,0)$은 closed지만 image distance가 1이어야 하고,
$\Delta_{\rm open}=(0,0,1)$은 $\|B_1\Delta_{\rm open}\|=1$로 consistency를
실패해야 한다. Wrong counterterm sign과 $B_1B_0\ne0$ 변형도 fail-closed
대조로 둔다.

통과 조건은 (i) 모든 shapeㆍghost numberㆍfinite/provenance 필드가 존재하고,
(ii) nilpotency residual $<10^{-12}$, (iii) supplied breaking의 closure와 제거 후
residual $<10^{-12}$, (iv) anomaly control의 closure $<10^{-12}$ 및 image
distance $>0.9$, non-closed control residual $>0.9$, (v) 고정된 well-conditioned
basis change 뒤 rank, $H^1$ 차원, exact/non-exact 판정이 불변이고 covariance
residual $<10^{-10}$인 것이다. SVD singular spectrumㆍrank thresholdㆍcondition
number를 영수증에 남기며 missing, NaN, rank ambiguity는 자동 FAIL한다.

계약에는 $1,R,R^2,R_{\mu\nu}R^{\mu\nu},R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma},
\Box R,(\nabla\phi)^2,m^2\phi^2,\xi R\phi^2,\phi^3,\phi^4$의 최소 진단
catalogue를 기록한다. 그러나 Gauss--Bonnetㆍ유한시간 경계항과 scalar--gravity
고차 연산자를 포함한 complete basis라고 표시하지 않으며, 모든 coefficient는
“미계산”으로 둔다. UV regulator, loop integral, local counterterm 계수,
regulator independence, BV measure Laplacian, CTP doubling 플래그도 모두
미계산으로 fail-closed 고정한다.

이 subgate의 PASS 문구는 “지정한 finite BRST/ST complex에서 supplied breaking이
$B_0$-exact이고 선언 counterterm으로 제거되며, 별도의 nontrivial class와
non-closed vector를 올바르게 거부했다”까지만 허용한다. 실제 one-loop
renormalization, continuum $H^1(s|d)$, ST/QME anomaly cancellation, curved-space
local covariance, in-in/CTP, positive physical Hilbert와 HDA/M2는 계속
**[미완성]**이다. M3--M9는 동결한다.

## E69-B 사전등록과 완료 계약

완료한 subgate는 **E69-B: arXiv:1706.02622v7 Eq. (28)--(30)의
Einstein--massless-quantum-scalar source reproduction**이다. 현재 v7의 field
content는 pure Einstein이 아니라 metricㆍconnection과 minimally coupled
massless quantum scalar를 함께 적분한 이론이다. 따라서 $\bar\phi=0$ 배경만으로
scalar determinant를 제거했다고 해석하지 않는다. 이전에 후보로 검토한
$(7/10,1/60)$ pure-gravity vector는 이 v7 source와 맞지 않아 부모 경로에서
폐기하고 mismatch control로만 사용한다.

Local transcription lock은
`arXiv:1706.02622v7`, 2021-09-11, harmonic/de Donder gauge,
arXiv metadata title `One-loop divergences in first order Einstein-Hilbert gravity`,
Eq. (28), Eq. (30), quantum scalar multiplicity 1과 canonical transcription
SHA-256
`37653a585f767212830cf49ba21cc9661c6509fa3f49d4f78c4a16ce5c869189`
로 고정한다. Ordered basis를

$$
Q=R_{\mu\nu}R^{\mu\nu},\qquad
X=(\nabla\phi)^2,\qquad Y=\Box\phi,
$$

$$
D={43\over60}Q+{1\over40}R^2+{1\over6}RX+X^2+Y^2,
\qquad
\Delta S={1\over(4\pi)^2\epsilon}\int d^4x\sqrt{|g|}\,D
\tag{36.14}
$$

로 둔다. 차원 계약에서는 기준길이 $L_*$에 대해
$\bar R=RL_*^2$, $\bar Q=QL_*^4$, $\bar X=XL_*^2$,
$\bar Y=YL_*^2$, $\bar D=DL_*^4$로 해석한다. 구현은 primitive length
exponent에서 ordered monomial 지수를 실제 합산한다. 따라서 coefficient와
$\epsilon$은 무차원이고 모든 합은 같은 $L^{-4}$ 차원이다. 이 차원 검사는
source의 물리적 정당성이나 regulator independence를 증명하지 않는다.

같은 source의 4차원 background EOM은

$$
{1\over2}g_{\mu\nu}R-R_{\mu\nu}
-{1\over4}g_{\mu\nu}X
+{1\over2}\nabla_\mu\phi\nabla_\nu\phi=0,
\qquad Y=0.
\tag{36.15}
$$

Trace와 원래 tensor 식을 분리해 $X=2R$, $Q=R^2$, $Y=0$을 유도하고,
직접 치환으로 $D_{\rm on}=203R^2/40$을 얻는다. 두 번째 독립 경로는
$\delta Q=Q-R^2$, $\delta X=X-2R$에 대한 exact polynomial identity

$$
D-{203\over40}R^2
={43\over60}\delta Q+{25\over6}R\delta X
+(\delta X)^2+Y^2
\tag{36.16}
$$

를 유리수 sample에서 검증한다. $R\in\{-2,-1/3,0,5/7\}$의 on-EOM 점과
별도 off-shell $(R,Q,X,Y)$ 점을 모두 `fractions.Fraction`으로 계산해 floating
tolerance 없이 항등식 residual 0을 요구한다.

통과 조건은 (i) source IDㆍ날짜ㆍ제목ㆍequationㆍfield contentㆍpayload hash가
모두 일치하고, (ii) coefficient vector가 정확히
$(43/60,1/40,1/6,1,1)$이며, (iii) trace/tensor/scalar EOM 두 단계와 직접 치환,
residual identity가 모두 $203/40$을 재현하고, (iv) source가 Gauss--Bonnet을
사용했다는 사실과 유한 경계 completion 미계산을 별도 flag로 남기는 것이다.
Stale coefficient, scalar multiplicity $1\to0$, $X^2$ 또는 $RX$ 누락,
$X=R$, $Q=R^2/4$, Eq. (28)/(30) 혼용, $\bar\phi=0\Rightarrow$ scalar-loop 제거,
source hash 변경은 각각 fail-closed 음성대조다.

`derivation_status`는 `source_reproduction_only`로 고정하고
`loop_integral_evaluated`, `heat_kernel_trace_derived`, `ghost_determinant_derived`,
`regularization_scheme_implemented`, `independent_feynman_diagram_check`,
`renormalization_proof`, `boundary_counterterm_computed`는 모두 false로 둔다.
따라서 PASS는 “v7 Eq. (28)의 exact rational 전사와 supplied EOM 아래 Eq. (30)의
축약을 재현했다”까지만 뜻한다. 독립 one-loop 도출, pure-gravity 계수,
ST/QME anomaly cancellation, finite-boundary/in-in renormalization, physical Hilbert,
HDA/M2는 계속 **[미완성]**이고 M3--M9는 동결한다.

실행 결과 coefficient와 EOM 대입, on-shell rational 4점, off-shell rational
3점, 일곱 음성대조가 exact arithmetic으로 통과했다. 사후 차원 P1 보강 뒤
focused 회귀는 **5 passed in 0.06s**이고 production runner는
`declared_one_loop_source_reproduction_gate_passed=true`다. 형식 감사의 P0/P1은
없고, 수학 감사에서 남긴 provenance와 Gauss--Bonnet 지적은 각각 “local
transcription checksum”과 “source-use flag only”라는 claim ceiling으로
격리했다.

## E69-C 사전등록과 완료 계약

완료한 subgate는 **E69-C: source-locked Eq. (23)/(27)의 exact Fraction
assembly와 4차원 bulk Gauss--Bonnet quotient**다. 이 단계의 목적은 이미
전사한 Eq. (28)을 답으로 넣고 다시 비교하는 것이 아니다. Eq. (23)의
gaugeㆍmetricㆍscalar 최소 연산자 기여와 Eq. (27)의 ghost 기여를 각각
$n=4$에 대입하고, source가 명시한 ghost 부호ㆍ중복도
$a_2-2a_2^{\rm gh}$ 및 Eq. (29)의 bulk quotient를 거쳐 Eq. (28)에
도달하는지를 exact rational arithmetic으로 검사한다.

Source lock은 arXiv metadata title
`One-loop divergences in first order Einstein-Hilbert gravity`와 HTML 내부
heading `One-Loop in first order quantum gravity`를 별도 필드로 보존한다.
Field contentㆍgaugeㆍprefactor $[(4\pi)^2\epsilon]^{-1}$, Eq. (23), (27)--(29),
ghost weight $-2$와 canonical local transcription SHA-256
`88cb5281c058f0983281d2e20017be987de6e2ab6bb53af41fa6fcc205ae9f17`도
잠근다. 여기서

$$
E=R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma},\quad
Q=R_{\mu\nu}R^{\mu\nu},\quad
P=R^{\mu\nu}\nabla_\mu\phi\nabla_\nu\phi,\quad
X=(\nabla\phi)^2,\quad Y=\Box\phi
$$

로 두고 ordered basis를

$$
(E,Q,R^2,P,RX,X^2,Y^2)
\tag{36.17}
$$

로 고정한다. Source Eq. (23)과 Eq. (27)의 $n$-의존 coefficient를 문자 그대로
유리함수로 구현한 뒤 $n=4$에서 공통 $1/360$ 이전 integer vector

$$
g_{\rm 2+gf}=(382,-1102,595,-720,-60,720,360),
\tag{36.18}
$$

$$
g_{\rm gh}=(-22,172,80,-360,-60,180,0)
\tag{36.19}
$$

를 각각 얻어야 한다. 따라서 signed assembly는

$$
g_{\rm raw}=g_{\rm 2+gf}-2g_{\rm gh}
=(426,-1446,435,0,60,360,360).
\tag{36.20}
$$

특히 $P$ 계수는 background EOM으로 버리는 것이 아니라
$-720-2(-360)=0$으로 ghost subtraction 자체에서 소거되어야 한다.

Eq. (29)는 pointwise identity로 사용하지 않는다. 경계와 위상 기여를 계산하지
않는 이 gate에서는 오직 4차원 적분 bulk의 formal quotient

$$
E-4Q+R^2=\text{total derivative}
\tag{36.21}
$$

를 적용한다. Coefficient map은

$$
(a_E,a_Q,a_{R^2},a_P,a_{RX},a_{X^2},a_{Y^2})
\mapsto
(4a_E+a_Q,-a_E+a_{R^2},a_P,a_{RX},a_{X^2},a_{Y^2})
\tag{36.22}
$$

이고, Eq. (36.20)의 $1/360$ vector는 출력 basis
$(Q,R^2,P,RX,X^2,Y^2)$에서

$$
\left({43\over60},{1\over40},0,{1\over6},1,1\right)
\tag{36.23}
$$

로 가야 한다. $P=0$을 별도 보존한 뒤에만 Eq. (28)의 5차원 basis와
비교하여, $P$를 조기에 삭제한 false PASS를 막는다.

통과 조건은 다음과 같다.

1. Source metadataㆍequationㆍfield contentㆍordered basisㆍcanonical local
   transcription payload가 잠금과 일치한다.
2. 모든 $n=4$ 대입, common $1/360$, ghost sign/multiplicity, raw vector,
   quotient map, Eq. (28) 비교를 `fractions.Fraction`으로 수행하고 tolerance
   없는 exact equality를 요구한다.
3. $a_2+2a_2^{\rm gh}$, $a_2-a_2^{\rm gh}$, ghost 미포함, $P$의 조기 삭제,
   wrong GB sign, Eq. (28)을 raw vector로 오인하는 대조, curvature basis 순서
   교환, $R^2$ㆍ$RX$ㆍ$X^2$ 개별 생략, $n=5$ raw vector 혼용을 모두 거부한다.
4. $[E]=[Q]=[R^2]=[P]=[RX]=[X^2]=[Y^2]=L^{-4}$를 primitive factor 지수의
   실제 합으로 산출하고, 의도적으로 $[X]$를 훼손한 대조가 실패해야 한다.
5. 적어도 세 rational curvature/scalar point에서 GB-related raw density와
   reduced density가 exact 일치하고, GB 관계를 깨뜨린 점은 nonzero residual을
   내야 한다.

`derivation_status`는 `source_coefficient_assembly_only`로 고정한다.
`heat_kernel_trace_derived`, `ghost_determinant_derived`,
`loop_integral_evaluated`, `regularization_scheme_implemented`,
`finite_boundary_completed`, `evanescent_terms_controlled`,
`independent_source_artifact_authenticated`, `renormalization_proof`,
`continuum_st_qme_proved`, `local_covariance_proved`, `in_in_ctp_completed`,
`positive_physical_hilbert_proved`, `quantum_hda_m2_proved`는 모두 false다.

따라서 PASS는 “잠근 source Eq. (23)/(27)의 계수를 exact rational로 조합하고,
명시한 4D bulk Gauss--Bonnet quotient 아래 Eq. (28)의 coefficient vector로
환원했다”까지만 뜻한다. 이는 heat-kernel trace나 ghost determinant의 독립 도출,
one-loop renormalization의 새 증명, finite-boundaryㆍ$n\ne4$ evanescent 결과,
pure Einstein, continuum ST/QME, local covariance, CTP, physical Hilbert 또는
HDA/M2의 증명이 아니다. Full E69은 **[미완성]**이며 M3--M9는 계속 동결하고
새 관측 예측은 없다.

실행 결과 Eq. (23), Eq. (27), signed raw와 reduced vector가 모두 사전고정값에
exact 일치했다. 형식 감사에서 항 생략ㆍbasis permutationㆍ$n=5$ 혼용을
receipt에 직접 드러내라는 P1을 보강한 뒤 focused 회귀는 **5 passed in
0.08s**이고, production runner는
`declared_source_coefficient_assembly_gate_passed=true`다. 최종 독립 수학ㆍ
형식 감사의 P0/P1은 없다. 세 rational sample은 선택한 bulk quotient
representative만 검사한다는 P2와 파일명의 `reconstruction`이 operator/trace
도출로 오독될 수 있다는 위험은 `source_coefficient_assembly_only` status와
모든 상위 claim flag false로 격리했다.

## E69-D 사전등록과 완료 계약

완료한 subgate는 **E69-D: v7 Eq. (19)과 supplied trace identity에서
Eq. (23)/(27)을 symbolic rational-polynomial로 재현하는 gate**다. 목적은
source의 최종 counterterm을 다시 전사하는 것이 아니다. Eq. (19)의 universal
minimal-operator formula에 source Eq. (22)의 gaugeㆍmetricㆍscalar trace
identity를 대입하여 Eq. (23)의 $n$-의존 coefficient를 항등식으로 얻고,
ghost sector에서는 Eq. (24)--(26)의 operator trace contraction을 같은 formula에
대입해 Eq. (27)을 얻는지를 검사한다.

기호 충돌을 피하려고
$E=R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}$,
$Q=R_{\mu\nu}R^{\mu\nu}$,
$P=R^{\mu\nu}\nabla_\mu\phi\nabla_\nu\phi$,
$X=(\nabla\phi)^2$, $Y=\Box\phi$로 두고, minimal operator potential은
$\mathcal Y$로 쓴다. Ordered invariant basis는

$$
(E,Q,R^2,P,RX,X^2,Y^2)
\tag{36.24}
$$

다. Eq. (19)의 common factor $1/360$을 제외한 조합은

$$
\begin{aligned}
\mathcal H[\mathcal Y,W]
&=(2E-2Q+5R^2)\operatorname{tr}\mathbb I
+180\operatorname{tr}\mathcal Y^2\\
&\quad-60R\operatorname{tr}\mathcal Y
+30\operatorname{tr}W_{\mu\nu}W^{\mu\nu}.
\end{aligned}
\tag{36.25}
$$

Source Eq. (22)의 trace input은

$$
\begin{aligned}
\operatorname{tr}\mathbb I
 &=\frac{n(n+1)}2+1,\\
\operatorname{tr}\mathcal Y
 &=\frac{n(n-1)}2R+\frac{8+3n-n^2}{4}X,\\
\operatorname{tr}\mathcal Y^2
 &=3E+\frac{n^2-8n+4}{n-2}Q
 +\frac{n^3-5n^2+8n+4}{2(n-2)}R^2\\
 &\quad-\left[\frac{2n(n-4)}{n-2}+4\right]P
 +\frac{n^3-7n^2+10n+8}{2(2-n)}RX\\
 &\quad+\frac{n^3-n^2+14n-40}{8(n-2)}X^2+2Y^2,\\
\operatorname{tr}W_{\mu\nu}W^{\mu\nu}
 &=-(n+2)E
\end{aligned}
\tag{36.26}
$$

로 잠근다. $n\ne2$에서 Eq. (36.25)를 전개ㆍ통분한 common-$1/360$
numerator vector는 source Eq. (23)의

$$
\begin{aligned}
g_{\rm 2+gf}(n)=\bigg(
&482-29n+n^2,\,
\frac{724-1440n+181n^2-n^3}{n-2},\\
&\frac{5(140+264n-145n^2+25n^3)}{2(n-2)},\,
-\frac{360(-4-2n+n^2)}{n-2},\\
&-\frac{15(32+62n-37n^2+5n^3)}{n-2},\,
\frac{45(n^3-n^2+14n-40)}{2(n-2)},\,360
\bigg)
\end{aligned}
\tag{36.27}
$$

와 성분별 rational-polynomial identity여야 한다.

Ghost branch에서는 Eq. (24)--(26)의 source-supplied operator와 curvature
convention에서

$$
\operatorname{tr}\mathbb I=n,\qquad
\operatorname{tr}\mathcal Y=-R+X,\qquad
\operatorname{tr}\mathcal Y^2=Q-2P+X^2,\qquad
\operatorname{tr}W_{\mu\nu}W^{\mu\nu}=-E
\tag{36.28}
$$

를 조립한다. Eq. (36.25)에 대입한 결과는 source Eq. (27)의

$$
g_{\rm gh}(n)
=(2n-30,180-2n,5n+60,-360,-60,180,0)
\tag{36.29}
$$

와 항등 일치해야 한다. 두 vector 모두 $1/360$을 보존한다. Downstream
ghost weight $-2$는 source/path-integral convention으로 잠그지만 이 gate에서
새로 유도하지 않는다.

구현은 외부 CAS에 의존하지 않는 exact polynomial/rational pair를 사용한다.
덧셈ㆍ곱셈ㆍscale 뒤 두 rational function의 분자ㆍ분모를 교차곱해 정규화된
polynomial 차이가 identically zero인지 검사한다. 별도로 $n=3,4,5$를
`fractions.Fraction`으로 평가해 symbolic identity와 source Eq. (23)/(27)
function을 각각 교차 확인한다.

통과 조건은 다음과 같다.

1. Eq. (19), (22)--(27), 두 제목 필드, theoryㆍgaugeㆍbasisㆍformula payload의
   canonical local checksum
   `684ace59f009a4ce2a3c680b835df786ea9bab0803ce308365645b5331811ebc`이
   일치하고 missing symbolㆍzero denominatorㆍ
   basis 순서 변경은 receipt 생성 전에 실패한다.
2. Eq. (23)/(27)의 모든 symbolic cross-product residual polynomial이 exact
   zero이며 $n=3,4,5$의 모든 component도 exact 일치해야 한다.
3. $n=2$ pole을 명시적으로 거부하고, $n=4$ 값만 복사한 impostor는
   $n=3,5$에서 nonzero mismatch를 내야 한다.
4. $-60R\operatorname{tr}\mathcal Y$ 누락, $\operatorname{tr}W^2$ 부호 반전,
   $\operatorname{tr}\mathbb I$의 scalar $+1$ 누락, Eq. (22)의 $P$ㆍ$RX$
   성분 누락, ghost $-2P\to+2P$와 ghost $\operatorname{tr}W^2$ 부호 반전을
   각각 nonzero symbolic mismatch로 검출한다.
5. 모든 basis term과 Eq. (36.25)의 네 contribution이 $L^{-4}$임을 primitive
   exponent에서 산출하고, $[X]$ 훼손 대조를 거부한다.

이 gate에서 Eq. (19), Eq. (22), Eq. (24)--(26), total-derivative omission
convention과 downstream ghost weight $-2$는 source-supplied input이다.
Eq. (19)가 “usual four-dimensional” formula로 소개되면서 $n$을 유지한다는
source의 범위도 그대로 기록한다. $n=3,5$ 검사는 유리함수 대수의 verification
point일 뿐 임의 차원의 heat-kernel theorem이나 evanescent operator의 물리
처리를 뜻하지 않는다.

`derivation_status`는 `source_trace_identity_assembly_only`로 고정한다.
`universal_heat_kernel_formula_derived`, `eq22_trace_tensors_derived`,
`ghost_determinant_derived`, `ghost_weight_derived`,
`loop_integral_evaluated`, `regularization_scheme_implemented`,
`finite_boundary_completed`, `evanescent_terms_controlled`,
`independent_source_artifact_authenticated`, `renormalization_proof`,
`continuum_st_qme_proved`, `local_covariance_proved`, `in_in_ctp_completed`,
`positive_physical_hilbert_proved`, `quantum_hda_m2_proved`는 모두 false다.

따라서 PASS는 “잠근 v7 Eq. (19)과 supplied Eq. (22), Eq. (24)--(26)의 trace
input을 symbolic rational-polynomial로 조합해 Eq. (23)과 Eq. (27)의
$n$-의존 coefficient를 항등 재현했다”까지만 뜻한다. 이는 heat-kernel trace
tensor나 ghost determinant의 독립 도출, loopㆍrenormalization,
finite-boundary/$n\ne4$ 물리 결과, pure Einstein, continuum ST/QME,
local covariance, CTP, physical Hilbert 또는 HDA/M2의 증명이 아니다.
Full E69은 **[미완성]**이며 M3--M9는 계속 동결하고 새 관측 예측은 없다.

실행 결과 Eq. (23)/(27)의 14개 symbolic residual polynomial과 42개 exact
spot component가 모두 0이고, $n=2$ source-domain pole 및 모든 음성대조가
사전고정대로 작동했다. Focused 회귀는 **6 passed in 0.11s**, production
runner는 `declared_source_trace_identity_assembly_gate_passed=true`다.
수학 감사는 rational cross-product와 pole semantics, 각 contribution slot,
control count와 차원 계산을 독립 검산해 P0/P1 없음으로 판정했다. 형식 감사도
checksumㆍsupplied formulaㆍ$n$-차원ㆍtotal-derivative claim ceiling이 모두
fail-closed임을 확인했다.

## E69-E 사전등록과 완료 계약

완료한 subgate는 **E69-E: ghost vector operator의 finite-exact trace
contraction 검산**이다. E69-D가 source Eq. (22)의 trace identity를 입력으로
삼아 Eq. (27)을 조합했다면, 이 단계는 Eq. (24)--(26)의 ghost vector
operator에서 쓰인 네 trace contraction을 별도 유한 기하 표본으로 직접
확인한다. 대상은 Euclidean orthonormal frame의 점별 대수이며, 미분 연산자,
Faddeev--Popov determinant 또는 ghost weight를 계산하지 않는다.

차원 $n\in\{3,4,5\}$의 Euclidean frame에서 off-diagonal 성분을 가진 대칭
유리수 행렬 $S_{ab}$와 유리수 vector $v_a$를 고정한다. Kronecker delta로
지표를 올리고 내리고, algebraic curvature tensor를

$$
R_{abcd}
=\delta_{ac}S_{bd}-\delta_{ad}S_{bc}
-\delta_{bc}S_{ad}+\delta_{bd}S_{ac}
\tag{36.30}
$$

로 둔다. Ricci contraction과 scalar curvature는

$$
R_{bd}=\sum_aR_{abad},\qquad R=\sum_bR_{bb}
\tag{36.31}
$$

이고, invariants는

$$
\begin{aligned}
E&=\sum_{a,b,c,d}R_{abcd}R_{abcd},&
Q&=\sum_{a,b}R_{ab}R_{ab},\\
X&=\sum_av_a^2,&
P&=\sum_{a,b}R_{ab}v_av_b
\end{aligned}
\tag{36.32}
$$

다. Eq. (26)의 ghost potential과 source convention에 맞춘 vector-bundle
curvature matrix는

$$
\mathcal Y_{ab}=-R_{ab}+v_av_b,\qquad
(W_{\mu\nu})_a{}^b=R_{ab\mu\nu}
\tag{36.33}
$$

로 고정한다. 검사할 항등식은

$$
\begin{aligned}
\operatorname{tr}\mathbb I&=n,\\
\operatorname{tr}\mathcal Y&=-R+X,\\
\operatorname{tr}\mathcal Y^2&=Q-2P+X^2,\\
\sum_{\mu,\nu}\operatorname{tr}(W_{\mu\nu}W_{\mu\nu})&=-E.
\end{aligned}
\tag{36.34}
$$

세 번째 식은
$\operatorname{tr}(-{\rm Ric}+vv^{\mathsf T})^2
=Q-2v^{\mathsf T}{\rm Ric}\,v+(v^{\mathsf T}v)^2$다. 마지막 식은
$R_{ab\mu\nu}R_{ba\mu\nu}=-E$라는 matrix trace이며, 단순 Frobenius 합
$\sum R_{ab\mu\nu}^2=+E$와 구별한다.

각 $n$에서 $S$와 $v\ne0$인 generic `Fraction` fixture를 고정하고
$E,Q,R,X,P$가 모두 nonzero인지 검사한다. 같은 $S$에 $v=0$인 별도 fixture도
실행해 $X=P=0$, $\operatorname{tr}\mathcal Y=-R$,
$\operatorname{tr}\mathcal Y^2=Q$ 극한을 확인한다. 모든 비교는 floating
tolerance 없이 exact residual 0을 요구한다. Frameㆍdimensionㆍtensor formula와
fixture payload의 canonical local SHA-256은
`38657a0defe69d3391f1affede36221d65f241a6cd413d4263a9ad735aa45488`로
첫 실행 전에 잠근다.

입력 curvature에는 두 지표쌍 각각의 antisymmetry, pair-exchange symmetry,
first Bianchi identity, Ricci symmetry와 Eq. (36.31) contraction을 PASS에
연결한다. 이는 trace 식이 우연히 맞는 비곡률 배열을 받지 않게 한다.
음성대조는 다음을 모두 포함한다.

- $W$ matrix trace 대신 Frobenius contraction을 써 $+E$를 얻는 변형.
- 올바른 $\sum_aR_{abad}$ 대신 $\sum_aR_{baad}=-R_{bd}$를 쓰는 Ricci 변형.
- $\mathcal Y=-{\rm Ric}-vv^{\mathsf T}$, $vv^{\mathsf T}$ 누락,
  또는 $\operatorname{tr}\mathcal Y^2$의 $-2P$ 누락.
- $(W_{\mu\nu})_a{}^b$의 사전고정 index placement를 바꾼 변형.
- $v=0$ fixture만 통과시키고 $v\ne0$ fixture를 누락하는 변형.
- $\mathbb I=\delta_{ab}$의 마지막 대각 원소를 지워 rank $n-1$로 만든 변형.

$W_{\mu\nu}\mapsto-W_{\mu\nu}$의 전체 부호 반전은 제곱 trace에서 보이지
않는다. 따라서 이 gate는 source가 선언한 matrix-index convention 아래
$W^2$ contraction을 검산하지만, covector/vector commutator의 선형 전체 부호를
독립 판정하지 않는다.

$[\mathrm{Ric}]=[R]=[\mathcal Y]=[W]=[X]=L^{-2}$와
$[E]=[Q]=[P]=[X^2]=L^{-4}$는 primitive factor에서 실제 합산한다. 기저를
$(E,Q,R,X,P,X^2)$로 명시하면 exponent vector는
$(-4,-4,-2,-2,-4,-4)$이고, $[X]$를 $L^{-1}$로 훼손한 대조는
$(-4,-4,-2,-1,-3,-2)$를 내어 PASS를 거부해야 한다.

`derivation_status`는 `finite_ghost_trace_contraction_only`로 고정한다.
`fp_operator_derived`, `fp_determinant_derived`, `ghost_weight_derived`,
`eq19_heat_kernel_derived`, `loop_integral_evaluated`,
`regularization_scheme_implemented`, `finite_boundary_completed`,
`evanescent_terms_controlled`, `independent_source_artifact_authenticated`,
`renormalization_proof`, `continuum_st_qme_proved`,
`local_covariance_proved`, `in_in_ctp_completed`,
`positive_physical_hilbert_proved`, `quantum_hda_m2_proved`는 모두 false다.

실행은 사전등록을 그대로 구현했다. $3$개 차원, generic/zero-vector 두 fixture,
네 trace의 **24개 exact residual이 모두 0**이고 curvature symmetryㆍBianchiㆍ
Ricci audit는 모두 PASS, 한 curvature 성분 훼손은 거부됐다. 음성대조의 합산
mismatch는 Frobenius $159435184/33075$, wrong Ricci $408$, wrong outer sign
$157427/1800$, outer-product 누락 $186165551/4320000$, cross-term 누락
$9529/225$, wrong $W$ index $39858796/11025$, generic fixture 누락
$5413/240$, rank-deficient identity $3$이다. $W\mapsto-W$의 squared-trace
residual은 예상대로 0이고 `w_linear_sign_determined=false`다. Focused 회귀는
**7 passed in 0.18s**, production runner는
`declared_finite_ghost_trace_contraction_gate_passed=true`이며 수정 안정판의
독립 수학ㆍ형식 감사에는 P0/P1이 없다.

따라서 PASS는 “사전고정한 Euclidean finite curvature/vector fixture에서
Eq. (24)--(26)에 대응하는 ghost potential과 vector curvature action의 네
대수적 trace contraction이 Eq. (36.34)를 exact rational로 만족한다”까지만
뜻한다. 이것은 Eq. (24)--(26)의 독립 유도, Faddeev--Popov determinant,
ghost factor $-2$, Eq. (19), one-loop divergence, total derivativeㆍfinite
boundary, renormalization, continuum ST/QME, local covariance, CTP,
physical Hilbert 또는 HDA/M2의 증명이 아니다. Full E69은 **[미완성]**이며
M3--M9는 계속 동결하고 새 관측 예측은 없다.

## E69-F 사전등록과 완료 계약

완료한 subgate는 **E69-F: Eq. (20)--(22)의
$\operatorname{Sym}^2V\oplus1$ bundle curvature trace를 finite-exact로 검산하는
gate**다. E69-E가 ghost vector bundle을 검사했다면, 이 단계는 metric
fluctuation의 symmetric rank-two tensor와 quantum scalar의 합 bundle에서
$\operatorname{tr}\mathbb I$와 $\operatorname{tr}W^2$만 직접 구성한다.
Eq. (22)의 $\operatorname{tr}\mathcal Y$와
$\operatorname{tr}\mathcal Y^2$는 이 gate의 입력도 산출도 아니다.

차원 $n$의 Euclidean orthonormal frame에서

$$
\mathcal B=\operatorname{Sym}^2V\oplus\mathbb R,\qquad
\dim\mathcal B=\frac{n(n+1)}2+1
\tag{36.35}
$$

로 둔다. $\operatorname{Sym}^2V$는 unordered pair $i\le j$의 raw coordinate
basis를 쓴다. $i=j$ basis는 $B^{ii}_{ab}=\delta_{ai}\delta_{bi}$,
$i<j$ basis는
$B^{ij}_{ab}=\delta_{ai}\delta_{bj}+\delta_{aj}\delta_{bi}$이고, 출력
coordinate는 $c_{ij}(T)=T_{ij}$다. 이는 orthonormalized tensor basis가
아니므로 injection과 extraction에 같은 raw convention을 유지한다.

Source Eq. (20)--(21)에 대응하는 tensor block action은

$$
(W_{\mu\nu}h)^{ab}
=R^a{}_{c\mu\nu}h^{cb}+R^b{}_{c\mu\nu}h^{ac},\qquad
W_{\mu\nu}^{\rm scalar}=0
\tag{36.36}
$$

로 고정한다. Eq. (21)의 $1/2$는 ordered $(\rho,\sigma)$ 합에서 대칭 tensor를
표현하는 계수이므로, 위 unordered raw basis action에 추가 $1/2$를 곱하지
않는다. 실제 bundle identity와 각 $W_{\mu\nu}$ 행렬을 구성해

$$
\operatorname{tr}_{\mathcal B}\mathbb I
=\frac{n(n+1)}2+1,\qquad
\sum_{\mu,\nu}\operatorname{tr}_{\mathcal B}
(W_{\mu\nu}W_{\mu\nu})=-(n+2)E,\quad
E=R_{abcd}R_{abcd}
\tag{36.37}
$$

를 검사한다. 오른쪽 trace는
$\sum_{\mu,\nu,p,q}(W_{\mu\nu})_{pq}(W_{\mu\nu})_{qp}$이며 entrywise
Frobenius 제곱합이 아니다.

$n=3,4,5$의 generic rational fixture는 Eq. (36.30)의
$R^{(S)}=\delta\owedge S$를 사용한다. 이것만으로는 $n\ge4$의 Weyl-sensitive
오류를 잡지 못하므로 $n=4,5$에는 첫 네 축에서

$$
C_{1212}=1,\quad C_{1313}=-1,\quad
C_{2424}=-1,\quad C_{3434}=1
\tag{36.38}
$$

이고 나머지를 Riemann 대칭으로 생성한 exact tensor를 별도 고정한다. $n=5$는
이를 첫 네 축에 그대로 embed한다. 매 실행에서 antisymmetry, pair exchange,
first Bianchi와

$$
C\ne0,\qquad \operatorname{Ric}(C)=0,qquad
R^{(S+C)}=R^{(S)}+C
\tag{36.39}
$$

를 확인하고, Eq. (36.37)은 세 generic fixture와 두 $S+C$ fixture 모두에서
exact residual 0이어야 한다. Frameㆍdimensionㆍbasisㆍactionㆍ$S$와 $C$의
성분을 담은 canonical payload는 첫 production 실행 전에
`23826e568c1fd9e995437e9fb088f23372e7ca97167003b19a4016989e70e1a7`로
잠근다.

음성대조는 다음을 모두 PASS 결선에 연결한다.

- scalar identity block 누락 또는 rank-deficient bundle identity.
- $i<j$ 출력 coordinate를 $2T_{ij}$로 읽는 off-diagonal normalization 변형.
- Eq. (36.36) 전체에 잘못된 $1/2$를 곱하거나 두 tensor slot 중 하나를 누락한
  변형, 두 번째 slot의 상대 부호를 바꾼 변형.
- $R^a{}_{c\mu\nu}$의 index placement를 바꾼 변형.
- matrix-product trace를 entrywise Frobenius 제곱합으로 바꾼 변형.
- $S+C$에서 $C$를 버리고 Ricci/$S$만으로 curvature action을 재구성하는 변형.
- generic fixture를 빼고 bundle dimension만 확인하는 변형.

$W_{\mu\nu}\mapsto-W_{\mu\nu}$의 전체 부호는 Eq. (36.37)에 보이지 않는다.
따라서 이 gate는 선형 commutator convention의 전체 부호를 독립 판정하지
않는다. $[\mathbb I]=L^0$, $[R]=[W]=L^{-2}$,
$[E]=[\operatorname{tr}W^2]=L^{-4}$를 primitive factor에서 계산하고,
$[R]$을 $L^{-1}$로 훼손한 차원 대조는 PASS를 거부해야 한다.

`derivation_status`는 `finite_sym2_bundle_curvature_trace_only`로 고정한다.
`eq22_trY_derived`, `eq22_trY2_derived`, `eq18_operator_derived`,
`gauge_fixing_derived`, `functional_determinant_derived`,
`heat_kernel_trace_derived`, `fp_determinant_derived`,
`ghost_weight_derived`, `loop_integral_evaluated`,
`regularization_scheme_implemented`, `finite_boundary_completed`,
`evanescent_terms_controlled`, `independent_source_artifact_authenticated`,
`renormalization_proof`, `continuum_st_qme_proved`,
`local_covariance_proved`, `in_in_ctp_completed`,
`positive_physical_hilbert_proved`, `quantum_hda_m2_proved`는 모두 false다.

실행 결과 bundle rank는 $(7,11,16)$이고 세 identity 및 세 genericㆍ두
Weyl-added curvature trace의 **8개 exact residual이 모두 0**이다. Generic,
pure-Weyl, Weyl-added를 합한 7개 curvature audit가 모두 PASS하고, Weyl
fixture는 nonzeroㆍRicci-flat이며 한 성분 훼손은 거부됐다. Scalar curvature
block도 모든 fixture에서 0이다.

Scalar identity, off-diagonal normalization, half action, second-slot 누락,
relative slot sign, wrong curvature index, Frobenius trace, Weyl drop의 mismatch는
각각 $3$, $37364587/675$, $1020539221/44100$, $9468018587/396900$,
$95787383/3675$, $2393281099/44100$, $1463281703/22050$, $208$이다.
Generic fixture liveness는 $79717592/33075>0$이고, $W\mapsto-W$의 squared
residual은 0이라 `w_linear_sign_determined=false`를 유지한다. Focused 회귀는
**7 passed in 8.99s**, production runner는
`declared_finite_sym2_curvature_trace_gate_passed=true`이며 세 독립 감사 모두
P0/P1 없음으로 판정했다.

따라서 PASS는 “사전고정한 finite Euclidean algebraic-curvature fixture에서 raw
$\operatorname{Sym}^2V\oplus1$ representation을 구성해 source Eq. (22)의
$\operatorname{tr}\mathbb I$와 $\operatorname{tr}W^2$ target을 exact rational로
재현했다”까지만 뜻한다. 이는 Eq. (18)의 potential/minimal operator,
Eq. (22)의 나머지 두 $\mathcal Y$ trace, gauge fixing, determinant, heat kernel,
one-loop divergence, boundaryㆍevanescent completion, renormalization, continuum
BRST/ST/QME, local covariance, CTP, physical Hilbert 또는 HDA/M2의 증명이
아니다. Full E69은 **[미완성]**이며 M3--M9는 계속 동결하고 새 관측 예측은
없다.

## E69-G 사전등록과 완료 계약

완료한 subgate는 **E69-G: Eq. (17)--(18)의 raw potential matrix에서 Eq. (22)의
$\operatorname{tr}\mathcal Y$와 bulk
$\operatorname{tr}\mathcal Y^2$ representative까지 가는 finite-exact
quotient gate**다. 원문은 Eq. (22) 직전에 적분부분적분 과정을 밝히지 않지만,
Eq. (18)의 mixed Hessian block을 직접 제곱하면 pointwise
$H_{\mu\nu}H^{\mu\nu}$가 남는다. 따라서 raw pointwise trace와 source bulk
표현을 처음부터 같은 식으로 놓지 않고 그 차이를 독립 계산한다.

Euclidean orthonormal **local normal-coordinate point**와 E69-F의 raw unordered
$\operatorname{Sym}^2V\oplus\mathbb R$ basis를 유지한다. Source Eq. (17)의
DeWitt block과 그 역은

$$
C_{ab,cd}=\frac14(\delta_{ac}\delta_{bd}+\delta_{ad}\delta_{bc}
-\delta_{ab}\delta_{cd}),\qquad
C^{-1}_{ab,cd}=\delta_{ac}\delta_{bd}+\delta_{ad}\delta_{bc}
-\frac{2}{n-2}\delta_{ab}\delta_{cd}
\tag{36.40}
$$

다. Raw basis tensor를 $B_p^{ab}$라 하면
$G_{pq}=B_p^{ab}C_{ab,cd}B_q^{cd}$, scalar block $G_{\phi\phi}=1$로
행렬을 직접 만들고 exact Gauss--Jordan inverse가
$G^{-1}G=\mathbb I$인지 검사한다. $n=2$ pole은 명시적으로 거부한다.

$v_a=\nabla_a\bar\phi$로 두고, source가 mixed block에 쓴
$\partial_a\partial_b\bar\phi$는 이 normal-coordinate point에서만
$H_{ab}=\partial_a\partial_b\bar\phi=\nabla_a\nabla_b\bar\phi$로 식별한다.
이를 global covariantization으로 주장하지 않는다. 이어
$Z=\operatorname{tr}H$, $X=v_av_a$로 두고 source Eq. (18)을

$$
\begin{aligned}
\mathcal Y^{hh}_{\mu\nu,\rho\sigma}
={}&C_{\mu\nu,\rho\sigma}\left(R-\frac12X\right)
-\frac12(R_{\mu\rho\nu\sigma}+R_{\nu\rho\mu\sigma})\\
&+\frac12(\delta_{\mu\nu}R_{\rho\sigma}
 +\delta_{\rho\sigma}R_{\mu\nu})\\
&-\frac14(\delta_{\mu\rho}R_{\nu\sigma}
 +\delta_{\mu\sigma}R_{\nu\rho}
 +\delta_{\nu\rho}R_{\mu\sigma}
 +\delta_{\nu\sigma}R_{\mu\rho})\\
&-\frac14\bigl(
\delta_{\mu\nu}v_\rho v_\sigma+\delta_{\rho\sigma}v_\mu v_\nu
-\delta_{\mu\rho}v_\nu v_\sigma-\delta_{\mu\sigma}v_\nu v_\rho\\
&\hspace{29mm}
-\delta_{\nu\rho}v_\mu v_\sigma-\delta_{\nu\sigma}v_\mu v_\rho
\bigr),
\end{aligned}
\tag{36.41}
$$

$$
\mathcal Y^{h\phi}_{ab}=\mathcal Y^{\phi h}_{ab}
=H_{ab}-\frac12\delta_{ab}Z,\qquad
\mathcal Y^{\phi\phi}=X
\tag{36.42}
$$

로 전사한다. Covariant raw matrix는 basis tensor를 양쪽에 실제 수축해 만들고
$A=G^{-1}\mathcal Y$를 계산한다. 첫 target은 pointwise 항등식

$$
\operatorname{tr}A
=\frac{n(n-1)}2R+\frac{8+3n-n^2}{4}X
\tag{36.43}
$$

다. 두 번째 source-supplied bulk representative는

$$
\begin{aligned}
\mathcal B_{22}(n)={}&3E
+\frac{n^2-8n+4}{n-2}Q
+\frac{n^3-5n^2+8n+4}{2(n-2)}R^2\\
&-\left(\frac{2n(n-4)}{n-2}+4\right)P
+\frac{n^3-7n^2+10n+8}{2(2-n)}RX\\
&+2Z^2
+\frac{n^3-n^2+14n-40}{8(n-2)}X^2 ,
\end{aligned}
\tag{36.44}
$$

여기서 $E=R_{abcd}R_{abcd}$, $Q=R_{ab}R_{ab}$,
$P=R_{ab}v_av_b$다. Raw matrix trace와의 차이를 비교할 quotient는 source
formula에서 역으로 맞추지 않고 Eq. (42)의 mixed block과 공변미분 교환자로
사전고정한다.

$$
\mathfrak D=H_{ab}H_{ab}-Z^2+P
=\nabla_\mu(v_\nu H^{\mu\nu}-v^\mu Z),\qquad
\operatorname{tr}(A^2)-\mathcal B_{22}(n)=4\mathfrak D .
\tag{36.45}
$$

마지막 divergence 식은
$[\nabla^2,\nabla_\nu]\bar\phi=R_{\nu\lambda}v^\lambda$ convention 아래의
bulk identity다. Gate residual은
$\operatorname{tr}(A^2)-\mathcal B_{22}-4\mathfrak D$이며,
$\operatorname{tr}(A^2)=\mathcal B_{22}$를 pointwise로 요구하지 않는다.

Fixture는 $n=3,4,5$의 세 generic $\delta\owedge S$, $n=4,5$의 두
Weyl-added curvature에 nonzero rational $v$와

$$
H_{ii}=i+2,\qquad H_{ij}=\frac1{i+j+5}\quad(i\ne j)
\tag{36.46}
$$

를 결합한다. 같은 세 generic curvature에서 $v=0$ fixture도 실행하고,
$n=3$ flatㆍ$v=0$ㆍ$H=\operatorname{diag}(1,2,4)$ 표본은
$\mathfrak D=-28$, raw-minus-bulk $=-112$를 내야 한다. 총 9개 fixture마다
Eq. (36.43)과 Eq. (36.45)의 exact residual 0, 즉 18개 exact component를
요구한다. Generic 표본은
$E,Q,R,X,P,H_{ab}H_{ab},Z,\mathfrak D$의 필요한 nonzero liveness를
검사하고 pure Weyl tensor의 Ricci-flatness도 다시 감사한다.
SourceㆍframeㆍbasisㆍDeWitt metric/inverseㆍEq. (18) 전사ㆍfixtureㆍquotient
payload는 첫 실행 전에
993123d20fc3f95d52d013fe7bdf7951867a6d8e2b40e53c662d93f42527af40로
잠근다.

음성대조는 $C^{-1}$ trace coefficient 훼손, $G^{-1}$ 대신 raw Euclidean
identity 사용, off-diagonal basis normalization 변경, $\mathcal Y^{hh}$ 한
성분/index 훼손, mixed block 누락ㆍ상대 부호 변경, Eq. (36.42)의
$-\delta_{ab}Z/2$ 부호 변경, scalar block 누락,
$\operatorname{tr}(A^2)$를 $(\operatorname{tr}A)^2$로 교체, Ricci contraction
부호 변경, quotient coefficient $4$ 변경, pointwise
$\mathfrak D=0$ 강제, Weyl drop와 $n=4$ coefficient를 $n=3,5$에 복사하는
변형을 포함한다. 두 mixed block을 동시에 부호 반전하면 squared trace에
보이지 않으므로 그 선형 전체 부호는 판정하지 않는다.

$[G]=[C]=[C^{-1}]=L^0$, $[v]=L^{-1}$,
$[R_{abcd}]=[H]=[X]=[\mathcal Y]=L^{-2}$,
$[\operatorname{tr}A]=L^{-2}$와
$[E]=[Q]=[R^2]=[P]=[RX]=[H^2]=[Z^2]=[X^2]
=[\mathfrak D]=[\operatorname{tr}A^2]=L^{-4}$를 primitive factor에서
계산하고 gradient 또는 Hessian exponent 훼손을 거부한다.

derivation status는 finite_sym2_potential_bulk_quotient_only로 고정한다.
source_eq22_pointwise_identity_proved, integration_by_parts_source_explicit,
finite_boundary_completed, endpoint_terms_computed,
eq18_operator_derived, gauge_fixing_derived,
functional_determinant_derived, heat_kernel_trace_derived,
fp_determinant_derived, ghost_weight_derived, loop_integral_evaluated,
regularization_scheme_implemented, evanescent_terms_controlled,
independent_source_artifact_authenticated, renormalization_proof,
continuum_st_qme_proved, local_covariance_proved, in_in_ctp_completed,
positive_physical_hilbert_proved, quantum_hda_m2_proved는 모두 false다.

실행 결과 rank $(7,11,16)$ raw metric의 exact inverse residual은 모두 0이고
$n=2$ pole은 거부됐다. 세 generic-vector, 세 generic-zero-vector, 두
Weyl-added, 한 flat-Hessian의 총 9개 fixture에서 trY와 Eq. (36.45)의
**18개 exact residual이 모두 0**이다. 일곱 curvature audit, potential matrix
symmetry, Weyl Ricci-flatness와 모든 liveness 검사도 PASS했다. Flat 표본은
raw trace $-14$, source bulk $98$, 차이 $-112=4(-28)$을 재현했다.

Wrong DeWitt/raw metric, off-diagonal basis, 훼손한 $Y^{hh}$, mixed 누락ㆍ
상대부호, Hessian trace 부호, scalar 누락, trace-square 혼동, Ricci 부호,
quotient coefficient, pointwise 강제, Weyl drop, $n=4$ 복사의 14개 mismatch는
모두 nonzero다. 두 mixed block 동시 부호 반전 residual만 0이므로 선형 부호는
판정하지 않는다. Focused 회귀는 **8 passed in 2.77s**, production runner는
declared_finite_sym2_potential_bulk_gate_passed=true다. 1차출처 감사 뒤
local normal-coordinate convention과 hash
993123d20fc3f95d52d013fe7bdf7951867a6d8e2b40e53c662d93f42527af40으로
재잠그고 처음부터 재실행했으며, 최종 수학ㆍ형식ㆍ출처 감사에는 P0/P1이 없다.

따라서 PASS는 “source-v7 Eq. (17)--(18)의 고정된 local normal-coordinate
Euclidean raw
matrix에서 $\operatorname{tr}A$를 pointwise 재현하고,
$\operatorname{tr}(A^2)$와 source Eq. (22)의 supplied bulk representative
차이가 선언한 $4\mathfrak D$ quotient임을 $n=3,4,5$ exact fixture에서
확인했다”까지만 뜻한다. 원문이 이 적분부분적분 단계를 명시했다는 주장,
pointwise Eq. (22) 전체 항등식, 유한 boundary/endpoint 또는 integrated action
동일성, full operatorㆍgauge fixingㆍdeterminantㆍheat kernelㆍloopㆍ
renormalizationㆍST/QMEㆍHilbertㆍHDA/M2의 증명이 아니다. Full E69은
**[미완성]**이며 M3--M9는 계속 동결하고 새 관측 예측은 없다.

## E69-H 사전등록과 완료 계약

다음 subgate는 **E69-H: linearized Faddeev--Popov variation과 finite Berezin
determinant에서 ghost weight를 분리 유도하는 gate**다. Source는 Eq. (11),
Eq. (13), Eq. (24)--(26)과 최종 $-2$를 제시하지만 그 사이의 FP Jacobian과
Grassmann 적분을 전개하지 않는다. 따라서 source 전사, 채택 convention,
독립 finite algebra를 구분한다.

Local Euclidean normal-coordinate point에서 quantum/background split의
선형부를

$$
\delta h_{\mu\nu}=\nabla_\mu\xi_\nu+\nabla_\nu\xi_\mu,\qquad
\delta\varphi=\xi^\rho v_\rho,\qquad
\chi_\nu=\nabla^\mu h_{\mu\nu}-\frac12\nabla_\nu h-\varphi v_\nu
\tag{36.47}
$$

로 고정한다. 원문 Eq. (11)의 scalar $\phi/\bar\phi$ 표기는 모호하므로
$\delta\varphi=\xi\cdot v$는 background split의 **채택 선형화**이며 source가
명시적으로 유도했다고 쓰지 않는다.

$K_{\mu\nu\rho}=\nabla_\mu\nabla_\nu\xi_\rho$와 arbitrary symmetric second jet
$S_{(\mu\nu)\rho}$를

$$
K_{\mu\nu\rho}
=S_{\mu\nu\rho}
+\frac12R_{\mu\nu\rho\sigma}\xi^\sigma,\qquad
K_{\mu\nu\rho}-K_{\nu\mu\rho}
=R_{\mu\nu\rho\sigma}\xi^\sigma
\tag{36.48}
$$

로 구성한다. Source curvature convention에서 Eq. (36.47)을 성분별로
변분해

$$
\begin{aligned}
\delta\chi_\nu
&=\sum_\mu K_{\mu\mu\nu}
 +\sum_\mu K_{\mu\nu\mu}
 -\sum_\mu K_{\nu\mu\mu}
 -v_\nu(v\cdot\xi)\\
&=\Box\xi_\nu+R_{\nu\rho}\xi^\rho-v_\nu v_\rho\xi^\rho,\\
\Delta_{\rm FP}&:=-\frac{\delta\chi}{\delta\xi}
=-\mathbb I\Box-\operatorname{Ric}+vv^{\mathsf T}
\end{aligned}
\tag{36.49}
$$

를 검사한다. 마지막 정의의 overall minus는 source Eq. (24)--(26)에 맞춘
채택 convention이다. 따라서 유도 대상은 principal term에 대한 potential의
상대 부호이며, $\Delta_{\rm FP}\mapsto-\Delta_{\rm FP}$의 전역 determinant
phase나 log branch는 판정하지 않는다.

Dimensionful normal coordinate와 무차원 $h_{\mu\nu},\varphi$ convention에서는
$[\xi]=L$, $[\nabla]=[v]=L^{-1}$로 둔다. 따라서
$[\Box\xi]=[\operatorname{Ric}\xi]=[vv^{\mathsf T}\xi]=L^{-1}$이고
$[\Delta_{\rm FP}]=L^{-2}$다. 첫 focused 실행 뒤 독립 수학 감사가
$[\xi]=L^0$로 둔 action-term 라벨의 P1을 찾아 이 표기로 수정했다. FP 계수,
exact residual과 determinant/Berezin 결과는 바뀌지 않는다.

각 $n=3,4,5$에서 E69-E의 generic rational curvatureㆍ$v$ㆍ$\xi$와 별도
rational symmetric $S_{\mu\nu\rho}$를 잠근다. Expanded gauge variation과
Eq. (36.49)의 residual은 exact 0이어야 하고, $v=0$, Ricci-flat Weyl,
flat-curvature 표본도 따로 실행한다. Gauge trace coefficient $1/2$ 변경,
scalar gauge term 누락/부호 반전, commutator 부호, Ricci contraction/index,
$\Delta=+\delta\chi$, generic fixture 누락을 음성대조로 둔다.

통계 레인은 differential-operator 계산과 독립이다. Grassmann generator
$(\bar c_1,\ldots,\bar c_N,c_1,\ldots,c_N)$의 exterior algebra와 top-form
orientation을 처음부터 고정하고, truncated exponential을 직접 전개해

$$
\int D\bar c\,Dc\,
\exp(-\bar c_iM_{ij}c_j)=\det M
\tag{36.50}
$$

를 $N=1,2,3$의 nonsingular rational matrix에서 exact하게 검사한다. Determinant
우변은 별도 Leibniz permutation 합으로 계산한다. $N=1,3$은 exponent sign
오류를 잡고, off-diagonal $N=2$ 표본은 diagonal shortcut을 막으며,
singular matrix는 zero mode로 명시적으로 reject한다.

Absolute log나 dimensionful determinant를 계산하지 않는다. 같은 차원의
nonsingular reference $M_0,A_0$와 positive real-boson matrix $A$에 대해
dimensionless ratio만 두고

$$
W_{\rm gh}(M;M_0)
=-\log\left|\frac{\det M}{\det M_0}\right|,\qquad
W_{\rm b}(A;A_0)
=+\frac12\log\left(\frac{\det A}{\det A_0}\right)
\tag{36.51}
$$

라는 선언된 Euclidean Gaussian convention을 쓴다. $M\mapsto\lambda M$,
$A\mapsto\lambda A$에서 log-ratio coefficient는 각각 $-N$과 $N/2$이므로

$$
w_{\rm gh/b}=\frac{-1}{1/2}=-2
\tag{36.52}
$$

다. Wrong Berezin orientation, $\exp(+\bar cMc)$, determinant inverse,
real-ghost half weight, species multiplicity 변경, reference ratio 누락,
singular matrix 허용을 모두 kill control로 둔다. $M\mapsto-M$과 basis
permutation/rescaling의 determinant ratio 및 odd-$N$ sign/log-branch 한계도
receipt에 남긴다.

SourceㆍframeㆍEq. (11)/(13)/(24)--(26)ㆍlinearizationㆍjetㆍBerezin orderingㆍ
matrix fixturesㆍreference-ratio convention은 첫 실행 전에 canonical
SHA-256
0f98583d2bc462d2f4252499a0e3f59c6573169b717f66f4c682849c0298ff49로
잠근다. derivation status는
finite_linear_fp_variation_and_berezin_weight_only로 고정한다.
linearized_background_split_assumed는 true이고,
fp_derivation_source_explicit, grassmann_measure_source_explicit,
ghost_minus_two_derivation_source_explicit, action_prefactor_derived,
global_fp_operator_completed, boundary_conditions_completed,
zero_mode_sector_resolved, functional_measure_derived,
functional_determinant_computed, log_branch_resolved,
brst_bv_measure_proved, heat_kernel_derived, loop_integral_evaluated,
renormalization_proof, continuum_st_qme_proved, local_covariance_proved,
in_in_ctp_completed, positive_physical_hilbert_proved,
quantum_hda_m2_proved는 모두 false다.

따라서 PASS는 “고정한 local finite gauge-jet basis에서 linearized FP
operator와 Eq. (26) potential의 상대 부호를 구성하고, 별도 finite Berezin
identity 및 dimensionless logdet scaling convention에서 relative ghost weight
$-2$를 확인했다”까지만 뜻한다. Eq. (11)의 완전한 split, source-explicit FP
유도, Eq. (24)의 $1/4$ normalization, global/functional determinant,
Grassmann path-integral measure, zero-mode와 boundary, BRST/BVㆍST/QME,
heat kernelㆍloopㆍrenormalizationㆍHilbertㆍHDA/M2의 증명이 아니다. Full
E69은 **[미완성]**이며 M3--M9는 계속 동결하고 새 관측 예측은 없다.

실행 결과 세 generic-vector, 세 generic-zero-vector, 두 pure-Weyl, 한
flat-vector의 총 9개 fixture와 여섯 curvature audit가 PASS했다. 아홉
commutator residual, expanded gauge variation 36성분, FP operator relation
36성분의 총 **72개 exact component는 모두 0**이고 gauge-parameter rescaling
residual도 0이다. 독립 exterior-algebra 적분과 Leibniz determinant는
$N=1,2,3$에서 각각 $(2,7,16)$으로 같고 singular matrix는 reference ratio
전에 거부됐다. Transposeㆍdiagonal similarityㆍpermutation basis를 보존하며
$\det(2M)/\det M=(2,4,8)$, relative weight는 정확히 $-2$다.

Wrong commutator/gauge trace/scalar omission/scalar flip/Ricci sign/FP sign과
positive exponent/wrong orientation/determinant inverse, ghost inverse/half/
doubled multiplicity의 12개 대조는 모두 nonzero다. 첫 focused 실행 뒤 독립
수학 감사가 $[\xi]=L^0$ 차원 라벨의 P1을 찾아 dimensionful-coordinate
convention의 $[\xi]=L$로 수정했다. 수정 뒤 focused 회귀는
**8 passed in 0.21s**, production runner는
declared_finite_fp_berezin_gate_passed=true다. 수학ㆍ형식ㆍ1차출처 감사의
최종 판정은 모두 P0/P1 없음이다. 출처가 FP Jacobian, Grassmann measure,
Eq. (24)의 $1/4$, 또는 $-2$의 Gaussian 유도를 명시했다는 승격은 계속
금지한다.

## 지금 바로 할 한 가지 — E69-I 사전등록

다음 subgate는 **E69-I: independently reconstructed raw operator trace에서
finite heat-kernel coefficient를 합성하는 gate**다. Source Eq. (19)의

$$
\operatorname{tr}a_2=
\frac{2E-2Q+5R^2}{360}\operatorname{tr}\mathbb I
+\frac12\operatorname{tr}\mathcal Y^2
-\frac16R\operatorname{tr}\mathcal Y
+\frac1{12}\operatorname{tr}W^2
\tag{36.53}
$$

는 **source-supplied heat-kernel theorem input**으로 쓴다. 이 식 자체를
유도했다고 표시하지 않는다. 반면 우변의 trace는 source Eq. (22) coefficient
table을 넣지 않고 E69-E/F/G의 실제 finite matrix contraction에서 다시
계산한다. Ghost weight $-2$는 E69-H의 별도 finite Berezin exponent 결과를
채택한다.

Ordered coefficient basis를

$$
\mathcal B_7=(E,Q,R^2,P,RX,X^2,Z^2),\quad
P=R_{\mu\nu}v^\mu v^\nu,\quad X=v^2,\quad Z=\Box\phi
\tag{36.54}
$$

로 고정한다. Bosonic potential은 raw matrix의
$\operatorname{tr}\mathcal Y$와 $\operatorname{tr}\mathcal Y^2_{\rm raw}$를
계산한 뒤 E69-G에서 분리한

$$
\operatorname{tr}\mathcal Y^2_{\rm bulk}
=\operatorname{tr}\mathcal Y^2_{\rm raw}
-4\mathfrak D,\qquad
\mathfrak D=H_{\mu\nu}H^{\mu\nu}-Z^2+P
\tag{36.55}
$$

를 선언한 integrated-bulk quotient로 쓴다. Ghost는 실제
$\mathcal Y_{\rm gh}=-\operatorname{Ric}+vv^{\mathsf T}$ matrix와 vector-bundle
$W$를 직접 수축한다. Eq. (22), Eq. (23), Eq. (27)의 source coefficient는
fit 입력으로 접근하지 않고 fit이 끝난 뒤 oracle 비교에만 쓴다.

각 $n=4,5,6,7$에서 pure Weyl, identity-$S$, traceless-$S$, flat-$v$,
flat-$H$, traceless-$S+v$, identity-$S+v$와 다섯 seeded
$\delta\owedge S+tC/v/H$ 표본, 총 12개 exact fixture를 고정한다. Invariant
design matrix가 rank 7이어야 하고, 일곱 독립 행으로 구한 coefficient가
나머지 다섯 overdetermined 행에서도 residual 0이어야 한다. Weyl이 없는
$n=3$은 $E,Q,R^2$를 독립 식별할 수 없으므로 full-rank identification
차원으로 쓰지 않는다. $n=8$의 raw coefficient는 같은 독립 fit으로 얻되
$n$-의존 polynomial interpolation에는 넣지 않는 holdout이다.

각 차원의 independently fitted bosonic/ghost vector를 fit 완료 뒤에만
source Eq. (23)/(27)과 비교한다. 차원 의존성은

$$
p_i(n)=360(n-2)c_i(n),\qquad \deg p_i\le3
\tag{36.56}
$$

라는 명시적 rational-form degree bound를 **채택**해 $n=4,5,6,7$에서
interpolate하고 $n=8$에서 holdout 검증한다. 이는 declared degree bound 아래의
finite 검산이지 all-$n$ symbolic theorem이 아니다.

그 뒤에만

$$
c_{\rm raw}(n)=c_{\rm bos}(n)-2c_{\rm gh}(n)
\tag{36.57}
$$

를 만들고 $n=4$의 $P$ coefficient가 0인지 확인한다. 마지막으로 오직
4차원 integrated bulk에서

$$
E-4Q+R^2\sim0,\qquad
(a_E,a_Q,a_{R^2})\mapsto(4a_E+a_Q,-a_E+a_{R^2})
\tag{36.58}
$$

를 적용해 source Eq. (28)의
$(43/60,1/40,0,1/6,1,1)$을 비교한다. Generic fixture의 Euler density가
nonzero인지 검사해 Eq. (36.58)을 pointwise-zero identity로 쓰는 경로를
금지한다.

Kill controls는 $\mathfrak D$ quotient 누락과 잘못된 $+4\mathfrak D$,
$W^2$ 부호 반전, scalar identity rank 누락, ghost weight
$+2,-1,0$, bosonic $P$의 premature 삭제, $n=4$ vector를 $n=8$에 복사,
Weyl을 뺀 rank-deficient design, 한 raw density 훼손, pointwise GB 강제,
$n=2$ pole 허용과 source/upstream hash mutation이다. 모든
$\mathcal B_7$, $\mathfrak D$, Eq. (36.53) density는 $L^{-4}$,
coefficient와 $n$은 무차원이어야 하며 primitive exponent 훼손은 PASS를
거부한다.

Sourceㆍupstream E69-D/E/F/G/H hashㆍframeㆍbasisㆍdimensionsㆍ12-fixture
generatorㆍEq. (36.53)--(36.58) convention은 첫 synthesis 실행 전에
canonical SHA-256
b7f30c9948b1853335a519d65b5c6796b2fd8ec58fd7532ecdad714b4bc0d83c로
잠근다. derivation status는
finite_raw_trace_to_source_coefficient_synthesis_only다.
source_eq22_coefficients_used_as_fit_input,
source_eq23_eq27_used_as_fit_input, eq19_theorem_independently_derived,
all_n_symbolic_identity_proved, 두 quotient의 pointwise_zero,
finite_boundary_completed, global_minimal_operator_derived,
functional_measure_derived, functional_determinant_computed,
heat_kernel_proper_time_integral_derived, loop_integral_evaluated,
regularization_scheme_implemented, evanescent_terms_controlled,
renormalization_proof, continuum_st_qme_proved, local_covariance_proved,
in_in_ctp_completed, positive_physical_hilbert_proved,
quantum_hda_m2_proved는 모두 false다.

초기 payload
814e6197b7b5373c11b45cbec338afb6905b796c21d8340e8891e7d6e892191b의
첫 실행은 coefficient 계산 전에 기존 Weyl helper가 $n=6$을 거부해
중단됐다. 따라서 고정 4D Ricci-flat $C$를 $n>4$의 추가 축에 0으로 embed하는
규칙을 fixture formula에 명시하고 위 새 hash로 다시 잠갔다. 이 수정은
실패한 실행의 수치 결과를 사용하지 않았다.

따라서 PASS는 “고정한 local finite Euclidean raw operator/trace fixture에서
independent trace를 Eq. (19)의 supplied map으로 합성하고, exact rational
coefficient fit과 holdout이 source Eq. (23)/(27)/(28)에 일치함을 확인했다”까지만
뜻한다. Eq. (19)의 독립 증명, all-background/all-$n$ theorem, global
determinant, boundaryㆍevanescent completion, continuum one-loop
renormalizationㆍBRST/ST/QMEㆍHilbertㆍHDA/M2의 증명이 아니다. Full E69은
**[미완성]**이며 M3--M9는 계속 동결하고 새 관측 예측은 없다.

## 1차 문헌

- [Brunetti, Fredenhagen, Verch, *The Generally Covariant Locality Principle*](https://arxiv.org/abs/math-ph/0112041)
- [Torre, Varadarajan, *Functional Evolution of Free Quantum Fields*](https://arxiv.org/abs/hep-th/9811222)
- [Driessler, *On the Type of Local Algebras in Quantum Field Theory*](https://doi.org/10.1007/BF01609853)
- [Donnelly, Giddings, *Observables, gravitational dressing, and obstructions to locality*](https://doi.org/10.1103/PhysRevD.94.104038)
- [Hollands, Wald, *Existence of local covariant time ordered products of quantum fields in curved spacetime*](https://arxiv.org/abs/gr-qc/0111108)
- [Giesel, Vetter, *Reduced Loop Quantization with four Klein--Gordon Scalar Fields as Reference Matter*](https://arxiv.org/abs/1610.07422)
- ['t Hooft, Veltman, *One-loop divergencies in the theory of gravitation*](https://www.numdam.org/article/AIHPA_1974__20_1_69_0.pdf)
- [Alvarez, Anero, Santos-Garcia, *One-loop divergences in first order Einstein-Hilbert gravity*, v7](https://arxiv.org/html/1706.02622v7)
- [Slavnov, *Ward identities in gauge theories*](https://doi.org/10.1007/BF01090719)
- [Becchi, Rouet, Stora, *Renormalization of Gauge Theories*](https://doi.org/10.1016/0003-4916(76)90156-1)
- [Batalin, Vilkovisky, *Gauge algebra and quantization*](https://doi.org/10.1016/0370-2693(81)90205-7)
- [Calzetta, Hu, *Nonequilibrium quantum fields: Closed-time-path effective action, Wigner function, and Boltzmann equation*](https://doi.org/10.1103/PhysRevD.37.2878)
- [Giulini, Marolf, *On the Generality of Refined Algebraic Quantization*](https://arxiv.org/abs/gr-qc/9812024)
- [Marolf, *Group Averaging and Refined Algebraic Quantization: Where are we now?*](https://arxiv.org/abs/gr-qc/0011112)
- [Thiemann, *Quantum Spin Dynamics VIII. The Master Constraint*](https://arxiv.org/abs/gr-qc/0510011)
- [Zinn--Justin, *Renormalization of Gauge Theories*](https://arxiv.org/abs/hep-th/9906115)
- [Barnich, Brandt, Henneaux, *Local BRST cohomology in gauge theories*](https://arxiv.org/abs/hep-th/0002245)
- [Barvinsky et al., *Renormalization of gauge theories in the background-field approach*](https://arxiv.org/abs/1705.03480)
- [Alvarez--Gaumé, Witten, *Gravitational Anomalies*](https://doi.org/10.1016/0550-3213(84)90066-X)
- [Han, Ma, *Master Constraint Operator in Loop Quantum Gravity*](https://arxiv.org/abs/gr-qc/0510014)
- [Cheung et al., *The Effective Field Theory of Inflation*](https://arxiv.org/abs/0709.0293)
- [Baumann, Green, *Equilateral Non-Gaussianity and New Physics on the Horizon*](https://arxiv.org/abs/1102.5343)
- [Demir, Pak, *General Tensor Lagrangians from Gravitational Higgs Mechanism*](https://arxiv.org/abs/0904.0089)
- [Oda, *Higgs Mechanism for Gravitons*](https://arxiv.org/abs/1003.1437)
- [Maldacena, *Non-Gaussian features of primordial fluctuations in single field inflationary models*](https://arxiv.org/abs/astro-ph/0210603)
- [Kamefuchi, O'Raifeartaigh, Salam, *Change of Variables and Equivalence Theorems in Quantum Field Theories*](https://doi.org/10.1016/0029-5582(61)90056-6)
- [Chisholm, *Change of variables in quantum field theories*](https://doi.org/10.1016/0029-5582(61)90077-8)
- [Cohen et al., *On-Shell Covariance of Quantum Field Theory Amplitudes*](https://arxiv.org/abs/2202.06965)
