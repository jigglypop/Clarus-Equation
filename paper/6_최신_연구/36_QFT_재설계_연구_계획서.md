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
| 실제 공변 작용, 제약 닫힘, 관계적 대수, 완전한 회복 증명 | **[미완성]** |
| 독립 새 관측 | **[예측: 없음]** |

## 지금 바로 할 한 가지

다음 작업은 M1이다. 기하, 물질, clock/rod를 포함하는 **실제 일반공변 작용 후보 하나**를 고르고, 정준 운동량과 모든 제약을 직접 유도한다. 처음부터

$$
P_T+H=0
$$

을 가정하지 않는다. 보통의 스칼라 시계는 먼저 $P_T^2$ 꼴의 제약을 줄 수 있으므로, 선형 시간 제약은 작용ㆍ게이지ㆍbranch 선택에서 정말 나오는지 별도로 증명해야 한다. 이 계산이 통과하기 전에는 플랑크 틱, QFT-next 동역학, Clarus source, 암흑부문 readout으로 진행하지 않는다.

## 1차 문헌

- [Brunetti, Fredenhagen, Verch, *The Generally Covariant Locality Principle*](https://arxiv.org/abs/math-ph/0112041)
- [Torre, Varadarajan, *Functional Evolution of Free Quantum Fields*](https://arxiv.org/abs/hep-th/9811222)
- [Driessler, *On the Type of Local Algebras in Quantum Field Theory*](https://doi.org/10.1007/BF01609853)
- [Donnelly, Giddings, *Observables, gravitational dressing, and obstructions to locality*](https://doi.org/10.1103/PhysRevD.94.104038)
- [Hollands, Wald, *Existence of local covariant time ordered products of quantum fields in curved spacetime*](https://arxiv.org/abs/gr-qc/0111108)
