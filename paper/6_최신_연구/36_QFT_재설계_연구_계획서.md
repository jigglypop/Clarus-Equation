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

## 지금 바로 할 한 가지

다음 subgate는 **normal-ordered all-signed cubic Fock closure와 rotating
truncation의 직접 비교**다. 공간 basis는 E68이 실제로 사용한 real cosine/sine
harmonic이므로 continuum의 독립 $\pm k$ oscillator라고 재해석하지 않는다. 네
oscillator의 순서를 $(k,0),(k,1),(2k,0),(2k,1)$로 고정하고, signed projected
vertex $V^{\sigma_1\sigma_2\sigma_3}_{ab;c}$에서
$O_{m,+}=a_m$, $O_{m,-}=a_m^\dagger$로 두어

$$
H_3^{\rm NO}={1\over2}\sum_{a,b,c=0}^1
\sum_{\sigma_1,\sigma_2,\sigma_3=\pm1}
V^{\sigma_1\sigma_2\sigma_3}_{ab;c}
{:O_{k,a,\sigma_1}O_{k,b,\sigma_2}
O_{2k,c,\sigma_3}:}
\tag{36.12}
$$

를 채택한다. $1/2$는 두 $k$ harmonic leg의 symmetric Taylor 계수이며 normal
ordering은 이 subgate의 명시적 양자화 규약이다. Ordering contraction을 action에서
유도했다고 주장하지 않는다.

두 initial state
$|I_0\rangle=|2,0;0,0\rangle$,
$|I_1\rangle=|0,2;0,0\rangle$ 각각에 대해 64 signed assignment를 모두
적용하고, occupation factor가 0이 아닌 algebraic target 전체를

$$
N_a=\{|n\rangle:\langle n|H_3^{\rm NO}|I_a\rangle\ne0\}
\tag{36.13}
$$

로 생성한다. 한 삽입에서 $k$ occupation 합은 최대 4, $2k$ occupation 합은
정확히 1이므로 임의 cutoff를 조정하지 않고 이 reachable closure를 쓸 수 있다.
Occupation-factor 조립과 독립 tensor-product creation/annihilation matrix를
비교하고, candidate target 수ㆍactive target 수ㆍ각 target의 기여를 영수증에
남긴다.

Diagonal survival exchange는

$$
\mathcal A_{aa}^{(2),\rm full}
=-\sum_{n\in N_a}
|\langle n|H_3^{\rm NO}|I_a\rangle|^2
I(E_a-E_n,E_n-E_a;\Delta\tau)
\tag{36.14}
$$

로 계산한다. E68-Z의 두 $|1_{2k,c}\rangle$만 남긴 rotating 열과
(36.14)의 counterrotatingㆍnumber-scattering 열을 합치기 전에 별도 기록한다.
같은 reachable star의 exact Hermitian matrix exponential과
$H_0+\lambda H_3^{\rm NO}+\lambda^2H_4$를
$\lambda=(1,1/2,1/4)$에서 비교한다.

통과 조건은 (i) 64 signed assignment와 sign-flip conjugation 완전성,
(ii) combinatorial/explicit-Fock residual $<10^{-12}$, Hermiticity
$<10^{-10}$, (iii) step $<2\times10^{-4}$, grid $<10^{-8}$,
rod-unitary pullback $<10^{-6}$, (iv) analytic/triangle kernel residual
$<10^{-10}$, (v) exact-star normalized error $<10^{-4}$와 사전고정한
quarter-scaling residual, (vi) rotating subset이 E68-Z의 retained matrix
elements를 같은 certificate 규약 아래 재현하는 것이다. Wrong $1/2$, unordered
contraction 삽입, counterrotating target 하나 누락, occupation cap 축소와
rotating-only 대조를 각각 독립 음성대조로 둔다. Full-minus-rotating이 자체
수치오차의 10배를 넘으면 rotating-only 부모 경로를 이 finite ansatz에서
기각하고, 넘지 않으면 차이를 비검출로 기록할 뿐 임계치를 낮추지 않는다.

누락 target, factorial 중복, 비Hermitian signed 조립, exact-star refinement 실패가
나오면 이 경로를 중단한다. 통과해도 두 diagonal initial state와 두 real
harmonic의 normal-ordering convention에 한정된다. Off-diagonal
$|1,1\rangle$ scattering, full quartic matrix, continuum momentum conservation,
SK/in-in correlator, vector/tensor/mixed sector, loop ST/QME, BRST physical
Hilbert와 HDA/M2는 계속 **[미완성]**이며 M3--M9는 동결한다.

## 1차 문헌

- [Brunetti, Fredenhagen, Verch, *The Generally Covariant Locality Principle*](https://arxiv.org/abs/math-ph/0112041)
- [Torre, Varadarajan, *Functional Evolution of Free Quantum Fields*](https://arxiv.org/abs/hep-th/9811222)
- [Driessler, *On the Type of Local Algebras in Quantum Field Theory*](https://doi.org/10.1007/BF01609853)
- [Donnelly, Giddings, *Observables, gravitational dressing, and obstructions to locality*](https://doi.org/10.1103/PhysRevD.94.104038)
- [Hollands, Wald, *Existence of local covariant time ordered products of quantum fields in curved spacetime*](https://arxiv.org/abs/gr-qc/0111108)
- [Giesel, Vetter, *Reduced Loop Quantization with four Klein--Gordon Scalar Fields as Reference Matter*](https://arxiv.org/abs/1610.07422)
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
