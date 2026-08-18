# 06. 측정 문제와 Born bridge

이 문서는 finite outcome Gibbs readout과 표준 양자 instrument를 비교해 CE의 측정 bridge가 요구하는 추가 가정을 분리한다. Hilbert 공간·상태·outcome sigma-algebra·CPTP instrument가 없는 Gibbs 재가중은 Born 확률이나 물리 측정의 유도가 아니다.

독자는 03a의 conditional readout과 05의 경로측도 bridge를 먼저 읽는다. 유한 outcome, Gibbs 조건화, instrument 기준선, branch·field·조건 후보를 거쳐 물리적 미완성 문제를 확인한다.

## 0. 범위

이 문서의 수학은 유한 확률 후보의 재가중이며, 양자 측정의 인과·decoherence·기록 형성은 별도 층이다. 대응표는 동일시가 아니라 필요한 bridge 조건을 드러낸다.

유한 후보의 Gibbs 농축, Born probability assignment와 물리적 측정
instrument는 서로 다른 구조다. 이 장은 세 구조를 분리한다.

## 1. 유한 outcome 후보

유한 outcome 모델은 확률분포와 에너지의 정의역을 명시한다. Hilbert amplitude나 phase를 잊은 확률 모델은 간섭·tensor product·measurement update를 자동으로 보존하지 않는다.

**[정의]** 유한차원 Hilbert space $\mathcal H$, orthonormal basis
$\{|i\rangle\}_{i=1}^n$와 단위벡터
$$
|\psi\rangle=\sum_{i=1}^nc_i|i\rangle,
\qquad
\sum_i|c_i|^2=1
$$
를 고정한다. Outcome label space는 $A=\{1,\dots,n\}$다.

**[공리: 확률 모형]** 초기 prior를
$$
\mu_0(i)=|c_i|^2
$$
로 택한다. 또는
[06a_Born_prior_유도.md](06a_Born_prior_유도.md)의 강한
refinement axioms를 모두 채택하면 이 식은 그 axioms의 조건부 정리가
된다. 어느 경우에도 실제 장치가 그 refinement를 구현한다는 결론은
따로 필요하다.

## 2. 조건부 Gibbs 재가중

Gibbs 재가중은 prior와 energy로 posterior 같은 분포를 정의한다. 이는 Bayes 대수 구조일 뿐, Born rule의 원리나 physical collapse를 증명하지 않는다.

**[공리: 모델 선택]** 무차원 cost
$$
\mathcal I_{\rm meas}:A\to\mathbb R
$$
와 무차원 $\beta>0$를 택하고
$$
\mu_\beta(i)
=
\frac{e^{-\beta\mathcal I_{\rm meas}(i)}|c_i|^2}
{\sum_j e^{-\beta\mathcal I_{\rm meas}(j)}|c_j|^2}
$$
로 정의한다. 물리적 에너지 $E_i$를 쓸 때에는
$\mathcal I_{\rm meas}=E_i/E_*$ 또는 열적 모형의
$E_i/(k_BT)$처럼 무차원화한다.

**[정리]** Support 안의 최소집합
$$
A_*=
\operatorname*{argmin}_{i:|c_i|^2>0}
\mathcal I_{\rm meas}(i)
$$
에 대해
$$
\mu_\beta(A\setminus A_*)\to0
\qquad(\beta\to\infty).
$$
유일 최소자 $k$가 있으면 $\mu_\beta\to\delta_k$다.

**증명.** $S=\{i:|c_i|^2>0\}$라 하자.
$S\setminus A_*=\varnothing$이면 자명하다. 그렇지 않으면 유한집합에서
최소자 밖에는 양의 energy gap
$$
\Delta=
\min_{i\in S\setminus A_*}
\bigl(\mathcal I_{\rm meas}(i)-\min_S\mathcal I_{\rm meas}\bigr)>0
$$
$A\setminus S$의 질량은 항상 0이고, 분자·분모의 유한합을 비교하면
나머지 바깥 질량은
$Ce^{-\beta\Delta}$ 이하이다. $\square$

이 정리는 이미 주어진 prior의 zero-temperature selection이다. 반복
실험의 single-shot Born sampling이나 물리적 collapse를 유도하지 않는다.

## 3. 표준 instrument 기준선

표준 instrument는 상태공간, outcome, CP map과 normalization을 함께 요구한다. 이 기준선이 있어야 CE readout과 물리 측정의 차이 및 circularity를 검사할 수 있다.

**[정의]** 유한 outcome quantum instrument는 completely positive
maps $\{\mathcal J_k\}$의 족이며
$\sum_k\mathcal J_k$가 trace preserving이다. Kraus 표현에서는
$$
\mathcal J_k(\rho)
=
\sum_\alpha M_{k\alpha}\rho M_{k\alpha}^\dagger,
\qquad
\sum_{k,\alpha}M_{k\alpha}^\dagger M_{k\alpha}=I.
$$
Outcome probability와 조건부 상태는
$$
p(k)=\operatorname{Tr}\mathcal J_k(\rho),
\qquad
\rho_k=\frac{\mathcal J_k(\rho)}{p(k)}
\quad(p(k)>0)
$$
다.

이 구조는 선형성, complete positivity, normalization과 상태 update를
한꺼번에 고정한다. Gibbs formula $\mu_\beta$는 classical probability
재가중만 정의하므로, 그 자체로 quantum instrument가 아니다.

## 4. Branch-dependent reweighting의 경계

branch별 가중치는 이미 어떤 branch decomposition과 prior가 주어졌을 때만 정의된다. Born weight를 전제한 prior로 Born weight를 다시 얻으면 순환 논증이 된다.

**[정의]** 무차원 $\varphi_k$에 대해
$$
\widetilde p_k
=
\frac{p(k)e^{-\varphi_k}}
{\sum_jp(j)e^{-\varphi_j}}
$$
는 분모가 양수일 때 well-defined probability vector다.

**[산출]** 모든 $\varphi_k$가 같으면
$\widetilde p_k=p(k)$다. 서로 다르면 일반적으로 Born probability와
다르다.

상태 $\rho$에 의존하는 normalization을 사후 적용한 map은 일반적으로
선형 CPTP channel이 아니다. 이를 물리 측정으로 쓰려면

1. $\varphi_k$를 장치 action에서 계산하고,
2. 해당 확률과 상태 update를 만드는 CP instrument를 구성하며,
3. 비선택 map의 trace preservation과 spacelike no-signalling을
   확인해야 한다.

국소 비선택 channel의 spacelike marginal 정리는
[../검증_원장/참조_이론물리_보존_원장.md](../검증_원장/참조_이론물리_보존_원장.md) 5절에
있다.

## 5. 장 후보와 Hessian readout

Hessian 또는 잔류장 readout은 경로측도에서 정의할 수 있는 kernel 구성이다. 이를 독립 양자장·decoherence 환경·관측 apparatus로 읽으려면 locality·action·renormalization과 실험 입력이 더 필요하다.

**[공리: EFT 후보]** 독립 scalar field를 택한다면 예를 들어
$$
S_\phi
=
\int d^4x\sqrt{-g}
\left[
-\frac12\nabla_\mu\phi\nabla^\mu\phi
-\frac12m_\phi^2\phi^2
-\frac12\xi R\phi^2
-V_{\rm int}(\phi)
\right]
$$
같은 action을 출발점으로 둘 수 있다. 부호, boundary term, state와
renormalization을 함께 지정해야 한다.

한편 Jacobi/Hessian의 scalar projection
$$
\Phi_{\rm eff}[\gamma,\eta]
=
\frac{\langle\eta,\mathcal J_\gamma\eta\rangle}
{\langle\eta,\eta\rangle}
$$
은 probe $\eta\ne0$를 고정한 readout이다. 독립장 $\phi$,
$\Phi_{\rm eff}$, 곡률 $R$은 서로 다른 대상이다. 이들을 동일시하려면
별도의 **[공리: 물리 사상]**이 필요하다.

## 6. 측정 조건 자체가 후보인 모형

측정 조건을 후보공간에 넣으면 setting과 outcome의 joint distribution을 기술할 수 있다. 이는 자유 선택·개입 인과·contextuality 문제를 해결하거나 물리 setting을 생성한다는 뜻이 아니다.

**[정의]** 유한한 장치 후보집합 $K$와 outcome 집합 $A$에 대해
joint prior $\rho_0\in\mathcal P(K\times A)$, 무차원 cost
$$
\mathcal I(M,i)
=
\mathcal I_{\rm outcome}(M,i)
+\lambda\mathcal I_{\rm apparatus}(M)
$$
를 정의할 수 있다. $\lambda$도 무차원이다.

이 유한 모형에는 03장의 joint Gibbs 농축 정리를 적용할 수 있다.
그러나 $M$을 실제 apparatus intervention으로 읽는 것은
**[공리: 물리 사상]**이다.

## 7. 남은 물리 문제

남은 문제는 Hilbert dynamics, decoherence, instrument, branch prior의 독립적 정당화와 Born frequency의 실증 연결이다. 이 공백은 Gibbs 농축의 수학적 성공으로 메우지 않는다.

- 06a refinement isometry와 실제 ancilla·장치의 대응
- CE path amplitude에서 Hilbert probability assignment로 가는 사상
- $\mathcal I_{\rm meas}$의 공변 장치 action
- Gibbs 재가중을 실현하는 CPTP instrument
- 독립 preparation을 반복했을 때의 outcome frequency theorem
- 환경 spectral density와 Lindblad/GKSL 유효범위

이 항목은 **[미완성]**이다. 최소자 농축과 Born sampling은 서로 대체하지
않는다.
