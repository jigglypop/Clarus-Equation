# 비선택 양자경로의 기회비용 함수 연구 계약

Status: COMPLETE

PREDECESSOR: `_workspace/ce/zero-dimensional-fold-memory-field-20260825`

절차 메모: run 초기화 직후 source/math lane이 계약 동결보다 먼저 완료되었다.
따라서 이 run은 탐색적 형식 연구이며 사전등록된 경험 예측으로 인용하지 않는다.
관측 데이터나 암흑부문 수치에는 맞추지 않는다.

## 1. 연구 질문

선택된 quantum-instrument outcome의 보완 경로를 “기회비용”으로 정량화할 수
있는가? 그 값이

1. 무차원 정보 readout,
2. 열역학적 자유에너지 또는 work bound,
3. metric variation을 갖는 중력 source

가 되는 조건을 분리한다. “에너지 없는 에너지”라는 표현은 먼저 정보적 shadow
price로 해석하고, 실제 에너지·암흑물질·암흑에너지 동일성은 독립 게이트에 둔다.

## 2. 타입과 기호

quantum instrument를 $\{\mathcal I_a\}$, 입력을 $\rho$, 선택 outcome을 $o$라
한다.

$$
p_a=\operatorname{tr}\mathcal I_a(\rho),
\qquad
\rho_a=\frac{\mathcal I_a(\rho)}{p_a}
$$

이며 비선택 집합은 $U=\{a:a\ne o\}$, $p_U=1-p_o$다. $p_a$, entropy,
relative entropy와 surprisal은 무차원이다. 에너지, 온도, 작용, 시간과
에너지밀도 scale은 각각 독립 외부 구조로 장부에 올린다.

기존 carrier pushforward

$$
\mu_F(B)=\int_{\Gamma_{\rm ns}}
w(\gamma)\mathbf1_B(F(\gamma))\nu_{\rm ns}(d\gamma)
$$

를 PREDECESSOR로 사용하되, 이 run은 $w$에 기회비용 functional을 곱할 수 있는지
검사한다.

## 3. 사전 고정 후보 경로

### R1. 정보 기회비용

다음을 서로 다른 후보로 비교한다.

$$
C_{\rm agg}(o)=-\ln p_U,
$$

$$
C_{\rm w}(o)=\sum_{a\in U}p_a[-\ln p_a],
$$

$$
H(p)=-\sum_ap_a\ln p_a,
\qquad
D(q\|r)=\sum_{a\in U}q_a\ln\frac{q_a}{r_a},
\quad q_a=\frac{p_a}{p_U}.
$$

모든 log 인자는 무차원이어야 한다. zero probability, reference support,
coarse-graining과 instrument dependence를 감사한다.

### R2. 열역학적 환산

Hamiltonian $H$, bath temperature $T>0$와 Gibbs state
$\gamma_T=e^{-H/(k_BT)}/Z$가 지정될 때

$$
F_T(\rho)=\operatorname{tr}(\rho H)-k_BT S(\rho),
$$

$$
\Delta F_T(\rho)=F_T(\rho)-F_T(\gamma_T)
=k_BT D(\rho\|\gamma_T)
$$

를 검사한다. $k_BT[-\ln p]$는 지정된 기록·소거 protocol에서의 work scale
후보이지 outcome이 본래 가진 에너지라고 사전 가정하지 않는다.

### R3. 반사실적 energy regret

outcome value가 실제 energy 또는 free energy로 지정될 때만

$$
C_E(o)=\sum_{a\in U}p_a[E_a-E_o]
$$

또는 양의 부분·최선 대안 version을 검토한다. 부호를 양으로 만들기 위한
$[\cdot]_+$ 또는 supremum은 별도 효용 공리로 취급한다.

### R4. influence/effective action

비선택·환경 자유도를 적분한 influence functional과 Euclidean ratio 후보를

$$
e^{iS_{\rm IF}/\hbar}=\int\mathcal D\chi_+\mathcal D\chi_-
e^{i(S_+-S_-)/\hbar},
$$

$$
\Gamma_{\rm ns}=-\hbar\ln(Z_{\rm ns}/Z_{\rm ref})
$$

로 둔다. $\Gamma_{\rm ns}$는 action 차원이며, energy가 되려면 시간 또는
inverse-temperature scale이 추가로 필요하다고 사전 고정한다.

### R5. 중력 bridge

기회비용 scalar $C$가 실제 source가 되려면 독립 에너지밀도 scale
$\epsilon_*$와 diffeomorphism-covariant action을 선언한다.

$$
S_{\rm opp}[g,\chi;C]
=-\int d^4x\sqrt{-g}\,
V_{\rm opp}(C,\chi,\nabla\chi;\epsilon_*),
$$

$$
T_{\mu\nu}^{\rm opp}
=-\frac{2}{\sqrt{-g}}
\frac{\delta S_{\rm opp}}{\delta g^{\mu\nu}}.
$$

$C$만으로 $\epsilon_*$, pressure, anisotropic stress 또는 conservation을
결정할 수 있는지 감사한다.

## 4. 고정 검산 예

두 outcome $p=(0.8,0.2)$와 선택 $o=0$을 쓴다. 다음을 계산한다.

1. $H(p)$,
2. $-\ln p_U$,
3. $p_U[-\ln p_U]$,
4. $q=(1)$의 conditional entropy,
5. $r=(1/2,1/2)$에 대한 full-outcome KL 예,
6. $E_0=0$, $E_1=\Delta$일 때 expected regret $0.2\Delta$.

analytic identity tolerance는 $10^{-12}$다.

## 5. 반증 조건

다음 부모 주장은 정의역 안의 반례가 있으면 제거한다.

1. 확률 또는 entropy만으로 energy 차원이 생긴다.
2. 비선택 outcome weight가 선택 branch의 Einstein source에 자동 가산된다.
3. scalar cost density 하나가 pressure와 전체 stress tensor를 유일하게 정한다.
4. $-\hbar\ln Z$가 시간·온도 scale 없이 energy다.
5. Landauer bound가 모든 측정 또는 미실현 경로의 실제 저장 에너지다.
6. continuum path entropy가 regulator, reference와 coarse-graining 없이 유한하고
   고유하다.

## 6. 주장 상한

이 run이 닫을 수 있는 최대 주장은 “비선택 경로에 무차원 정보 기회비용을
일관되게 정의하고, 지정된 bath/Hamiltonian/protocol에서만 이를 자유에너지 또는
work scale로 환산할 수 있으며, 중력원은 별도 covariant action과 energy-density
scale의 metric variation을 요구한다”까지다. 암흑물질·암흑에너지 동일성,
abundance와 관측 적합은 예측하지 않는다.
