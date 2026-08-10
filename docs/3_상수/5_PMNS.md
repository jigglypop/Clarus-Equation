# PMNS 경험식을 일관된 중성미자 EFT로 완성

## 1. 기준선과 경험식

**[정의]** TBM 기준선은

\[
\sin^2\theta_{12}^{(0)}=\frac13,\qquad
\sin^2\theta_{23}^{(0)}=\frac12,\qquad
\sin^2\theta_{13}^{(0)}=0
\]

이다. 이는 unitary 기준행렬이지 CE 동역학의 결과가 아니다.

**[경험식]** \(0\leq\delta\leq1/4\)에 대해

\[
s_{13}^2=\frac{\delta}{8},\qquad
s_{12}^2=\frac13\left(1-\frac{3\delta}{8}\right),\qquad
s_{23}^2=\frac12\left(1+\frac{7\delta}{8}\right)
\]

를 후보 매개화로 둔다. \(\delta=0.17776\)을 넣으면 각각 약
\(0.02222\), \(0.3111\), \(0.5778\)이다.

이 세 계수의 수치 선택은 경험적이지만, 그 뒤의 unitary·질량행렬 구조는
정확히 닫을 수 있다.

## 2. unitary 완성

**[정의]** \(c_{ij}=\sqrt{1-s_{ij}^2}\)로 두고 임의의 Dirac 위상
\(\delta_{\rm CP}\)에 대해

\[
U_{\rm PMNS}
=R_{23}(\theta_{23})
U_{13}(\theta_{13},\delta_{\rm CP})
R_{12}(\theta_{12})P_M
\]

로 정의한다. 여기서 \(P_M=\operatorname{diag}
(1,e^{i\alpha_{21}/2},e^{i\alpha_{31}/2})\)는 Majorana 위상행렬이다.

**[정리]** 각 인자가 unitary이므로 \(U_{\rm PMNS}\)도 정확히
unitary다. 따라서 세 경험식은 서로 독립인 각 목록이 아니라 하나의
unitary mixing ansatz를 정의한다. \(\delta_{\rm CP}\)와 두 Majorana
위상은 아직 자유다.

## 3. Majorana 질량행렬

**[공리]** 양의 중성미자 질량
\(D_\nu=\operatorname{diag}(m_1,m_2,m_3)\)를 입력한다.

**[산출]**

\[
M_\nu=U_{\rm PMNS}^*D_\nu U_{\rm PMNS}^\dagger
\]

로 두면 \(M_\nu^T=M_\nu\)이고

\[
U_{\rm PMNS}^TM_\nu U_{\rm PMNS}=D_\nu.
\]

[구성 증명](../참조/핵심_정리_증명.md#flavor-realization)

따라서 경험각과 임의의 질량·위상을 실현하는 복소 대칭 질량행렬은 항상
존재한다.

## 4. gauge-invariant EFT embedding

**[공리: Weinberg branch]** 표준모형 위에 차원 5 연산자

\[
\mathcal L_5
=\frac{\kappa_{ij}}{2\Lambda}
(\overline{L_i^c}\,\widetilde H^*)
(\widetilde H^\dagger L_j)+\mathrm{h.c.},
\qquad \kappa^T=\kappa
\]

를 둔다.

**[산출]** \(H=(0,v/\sqrt2)^T\) 이후

\[
M_\nu=\frac{v^2}{2\Lambda}\kappa.
\]

따라서 위 질량행렬은

\[
\kappa=\frac{2\Lambda}{v^2}
U_{\rm PMNS}^*D_\nu U_{\rm PMNS}^\dagger
\]

로 gauge-invariant 저에너지 EFT에 embedding된다.

## 5. 남은 선택 문제 **[미완성]**

- 왜 \(\delta\)가 세 각에 \(1/8,3/8,7/8\)로 분배되는가
- \(\kappa\)를 만드는 UV seesaw와 flavor symmetry가 무엇인가
- 질량 ordering, 절대 질량과 CP 위상을 무엇이 선택하는가
- RG 흐름이 경험식을 어느 scale에서 보존하는가

즉 “일관된 중성미자 이론이 존재하는가”는 닫혔고, “CE가 그 특정
매개변수를 선택하는가”가 남아 있다.
