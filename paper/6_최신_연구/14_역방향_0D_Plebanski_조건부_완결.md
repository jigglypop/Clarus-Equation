# 14. 하나의 typed 이력으로 읽는 0D--Plebanski 유한 증인

이 장의 결론은 단순하지만 범위가 좁다. 좌표 없는 $0$차원 seed에서 일반상대론 전체가 저절로 나온다는 결론은 얻지 않았다. 대신 **선언한 typed 규칙을 가진 하나의 이력**을 끝까지 따라가면, 그 이력 안에서 선택된 평탄 $\Lambda=0$ Lorentzian Plebanski/Einstein 해를 실제로 만들 수 있음을 보였다. 이전처럼 서로 독립인 조합론ㆍreadoutㆍGibbsㆍ기하 검사를 나란히 놓는 방식은 감사에서 실패했다. 같은 번호를 달았어도 값이 서로 흘러가지 않으면 하나의 증명이 아니기 때문이다. 지금의 증인은 한 typed trace에서 나온 같은 shared tetrahedron, 같은 Gram defect, 같은 $q$, $\theta$, class record를 다음 단계가 다시 사용한다. 같은 이력을 보존한 비평탄 constant-curvature 증인은 [15장](15_같은_이력의_비평탄_Plebanski_힌지.md)에서 이어진다.

정확한 형식 지위는 [차원 분류 원장 C4](../검증_원장/참조_차원_분류_원장.md#조건부-0dlorentzian-plebanskieinstein-폐쇄)에 고정돼 있다. 이 장은 원장을 바꾸지 않고, 처음 읽는 사람이 그 지위를 따라갈 수 있게 설명한다. CE의 큰 서사인 **끼임 → 접힘 → 암흑 표현** 가운데 여기서 형식적으로 다루는 것은 첫째와 둘째다. 접힌 가능성이 남는다는 유한 정리까지가 접힘의 범위다. 그것이 암흑에너지로 읽힌다는 마지막 단계는 아직 열려 있다.

## 14.1 먼저 도착점에서 확인할 것

끝점은 평탄한 $\Lambda=0$ Einstein 기하다. Plebanski의 chiral $BF$ 작용에서 $\Sigma^i$는 tetrad $e^I$로 만든 2-form이고, $A^i$는 연결, $F^i[A]$는 곡률이다. $\Psi_{ij}$는 대칭ㆍtrace-free multiplier라고 둔다.

$$
S[\Sigma,A,\Psi]
=\frac1{8\pi G}\int_M\left[
\Sigma^i\wedge F^i[A]
-\frac12\Psi_{ij}\Sigma^i\wedge\Sigma^j
\right].
$$

비퇴화 gravitational branch와 Lorentzian reality 조건을 **선택**하면 $\Sigma$는 $e\wedge e$로 복원된다. variation은 $D_A\Sigma=0$ 및 $F^i=\Psi^i{}_j\Sigma^j$를 주며, 선택한 평탄 해에서는 $F=0$, 따라서 vacuum Einstein 방정식이 성립한다. 이것은 [Plebanski (1977)](https://doi.org/10.1063/1.523215)의 고전적 충분조건을 한 유한 기하에 적용한 것이다. seed가 이 작용, branch, reality 조건을 유일하게 골랐다는 뜻은 아니다.

그 유한 기하는 두 Lorentzian 4-simplex가 tetrahedron $(1,2,3,4)$ 하나를 공유하도록 붙인 것이다. 선택된 member는 그 tetrahedron 양쪽에서 같은 세 face vector와 같은 bivector/Gram 자료를 쓴다. 그래서 “왼쪽에서 복원한 metric”과 “오른쪽에서 복원한 metric”이 별개의 샘플이 아니라 같은 면을 공유하는 한 기하인지 직접 검사할 수 있다. identity holonomy를 둔 선택 member는 평탄하다. 같은 이력에서 나온 다른 member들은 $\theta=\sqrt{\Delta}>0$의 face holonomy를 가져 곡률 대안으로 남는다. 평탄한 끝점을 골랐다고 곡률 대안을 삭제한 것이 아니다.

EPRL 점근은 이 구별을 더 엄격하게 만든다. 표준 vertex의 large-spin 분석에는 여러 saddle/방향 sector가 나타난다. [Engle--Zipfel (2015)](https://arxiv.org/abs/1502.04640)과 [Engle--Vilensky--Zipfel (2015)](https://arxiv.org/abs/1505.06683)의 proper vertex는 **한 4-simplex 점근**에서 원하는 항을 고르는 장치다. 이 사실은 이 장의 finite sector 선택과 양립하지만, 일반 triangulation의 Einstein 한계 정리는 아니다.

## 14.2 한 단계씩 거꾸로: 선택된 기하가 요구하는 장부

선택된 0-defect member를 먼저 놓고 뒤로 가면 필요한 값의 순서가 보인다.

1. 같은 metric을 공유하려면 shared tetrahedron의 bivector와 Gram 자료가 양쪽에 정확히 맞아야 한다. 이 조건은 local simplicity만으로 자동으로 나오지 않으므로, unit timelike normal, linear/secondary simplicity, orientation, reality, shape matching을 모형 입력으로 선언한다.

2. 이 shared-face mismatch를 무차원 수 $\Delta$로 정의한다. 선택 member에서는 $\Delta=0$이고, 두 대안 member에서는 $\Delta>0$이다. 같은 $\Delta$가 뒤에서 곡률 각 $\theta=\sqrt{\Delta}$, Gibbs weight, stationary-phase defect action의 입력이 된다. 따라서 세 계산은 ID만 공유하는 것이 아니라 값을 공유한다.

3. 대안 member의 base weight는 같은 distortion $x\in\mathbb R^3$에서 $q=\exp(-|x|^2/2)$로 정한다. $\beta\Delta$가 무차원이 되도록 $\beta=100$을 택하고,

$$
p_\beta(h)=\frac{q_h e^{-\beta\Delta_h}}{Z_\beta}
$$

로 다시 가중한다. 유한한 $\beta$에서는 $q_h>0$인 모든 대안이 여전히 양의 확률을 가진다. good set의 $\Delta=0$, bad set의 gap $\delta>0$이면

$$
P_\beta(G^c)\leq
\frac{Q_{\rm bad}}{Q_{\rm good}}e^{-\beta\delta}.
$$

이 부등식은 선택으로의 집중을 보일 뿐 대안을 지우지 않는다.

4. stationary phase도 discrete history 목록에 곧바로 적용한 것이 아니다. 같은 shared-face defect를 연속적으로 늘인 $x\in\mathbb R^3$ 확장에서, common tetrahedron frame을 고정하고 real $\mathbb R^3$ Gaussian contour를 **선언**한다. $\lambda=100$인 비퇴화 Hessian을 검사해 $x=0$이 그 연속 확장의 국소 stationary point임을 보인다. 이것은 유한 history의 확률 정리와 다른 종류의 계산이다.

5. Planck readout은 $L^2/L^2$라는 무차원 비로 만든다. 각 member의 squared length를 Planck area로 나눈 뒤, 원점 $0$에 고정한 half-open bin에 넣는다. 같은 bin이라는 것은 보이는 label을 공유한다는 뜻일 뿐, 원 history를 state space에서 제거한다는 뜻이 아니다. 따라서 이 quotient도 접힘을 삭제로 바꾸지 않는다.

6. 서로 다른 readout class에는 직교 class record를 붙인다. environment를 trace하면 class 사이 off-diagonal은 0이지만, rendered class 밖 대각 norm은 양수로 남는다. 이것이 이 장에서 말하는 **접힘**의 정확한 유한 의미다. 물리 환경이 그런 record isometry를 어떻게 만드는지는 여기서 증명하지 않는다.

## 14.3 더 뒤로: 그 값들이 하나의 trace에서 나오는 방법

이제 같은 값을 낳는 한 이력을 본다. seed에는 좌표, metric, 시간, 거리 모두 없다. 다만 typed split/merge rewrite와 rank-$4$ interaction을 **선택**한다. rank-$4$ simplex interaction은 한 4-simplex에 다섯 boundary atom, 스무 strand end, 열 개의 paired codimension-two face를 준다. 두 4-simplex $(0,1,2,3,4)$와 $(1,2,3,4,5)$를 붙이면 shared tetrahedron은 $(1,2,3,4)$가 된다.

이 trace에는 네 causal composition 2-cell이 있다. 이 장은 그것들을 shared tetrahedron의 네 triangle에 보내는 대응을 **선언한 duality**로 사용한다. 유도된 discrete Levi--Civita transport라고 부르지 않는다. 선언한 proper $SO^+(1,3)$ transport를 각 triangle 경계에서 곱해

$$
U_f=\prod_{e\subset\partial f}U_e
$$

를 정의하면 $U_f\neq1$은 그 선언된 2-cell 위의 finite holonomy curvature다. 선택 member는 $U_f=1$, 나머지는 $U_f\neq1$이다. causal face, triangle, shared tetrahedron, holonomy가 처음부터 끝까지 같은 typed trace에 매여 있다는 점이 이전 독립 부품 묶음과의 차이다.

$B_2$와 $F_2$를 같은 2-form type으로 택하고 background metric 없이 $B_2\wedge F_2$를 top form으로 쓴다면 차수는 $2+2=4$다. 여기에 비퇴화 signature $(-,+,+,+)$를 **추가로 선언**하면 $3+1$ Lorentzian 표현을 얻는다. 이는 form-degree 조건부 정리다. bare seed가 다른 rank, 다른 form, 다른 signature를 배제했다는 뜻이 아니다.

## 14.4 다시 앞으로 읽는 유한 증명

앞 절들을 순서대로 합치면 다음의 한정된 명제가 성립한다.

**유한 linked-witness 정리.** typed rank-$4$ rewrite, causal-2-cell$\leftrightarrow$triangle 선언, proper Lorentzian transport, common-metric/Plebanski sector, 원점 고정 Planck-area readout, 직교 class record, 그리고 위의 $q$, $\Delta$, $\theta$, $\beta=\lambda=100$ 계약을 고정하자. 그러면 두 4-simplex와 세 deformation member로 이뤄진 한 typed history에서 다음이 동시에 성립한다.

- 선택 member는 shared tetrahedron의 bivector/Gram을 맞추고 $U_f=1$인 평탄 $\Lambda=0$ Lorentzian Plebanski/Einstein 해다.
- 다른 member들은 같은 trace와 같은 tetrahedron에 매인 $\Delta>0$, $\theta>0$ 곡률 대안이며, finite $\beta$에서 양의 지지를 유지한다.
- Planck binning과 record trace는 미시 member를 지우지 않으면서 selected/folded class를 구별한다.
- 연속 defect 확장의 Gaussian stationary-phase 계산은 선택점의 비퇴화 국소 stationary 성질을 확인한다.

**증명.** rank-$4$ pairing과 two-simplex gluing에서 동일 trace와 shared tetrahedron을 얻는다. 이 tetrahedron의 face vectors로 bivector와 Gram을 복원하고 양쪽 matching을 검사한다. 같은 distortion으로 $\Delta$, $q$, $\theta$, squared-length readout, class record를 차례로 계산한다. $\Delta=0$ member의 identity holonomy와 chiral Plebanski residual을 확인해 평탄 Einstein 끝점을 얻는다. $\Delta>0$ member의 nonidentity holonomy, Gibbs 부등식의 full support, record trace 보존은 각각 곡률 대안ㆍ접힌 norm 보존을 준다. 마지막으로 같은 defect의 $x\in\mathbb R^3$ Hessian이 비퇴화임을 확인한다. 각 결과가 앞 결과의 같은 값에 의존하므로, 이는 독립 검사의 conjunction이 아니라 한 이력의 값 흐름이다. $\square$

## 14.5 이 정리가 말하지 않는 것

이 정리는 세 가지를 증명하지 않는다.

- $0$차원 seed가 이 typed rewrite, $B_2/F_2$, Lorentz signature, Plebanski action 또는 measure/contour를 유일하게 선택한다는 명제. chain/fork와 action/measure 반례가 그 지름길을 막는다.
- 위 한 평탄 증인이 generic curved geometry, refinement-consistent continuum, Einstein--Hilbert 지배, 정확히 두 massless spin-$2$ 자유도를 낳는다는 명제. 평탄 자료만으로 $R$과 $R+\alpha R^2$를 구별할 수 없다는 반례가 있다.
- 접힌 norm이 우주론에서 암흑에너지로 읽힌다는 명제. 그것에는 별도의 readout map, 보존 법칙, 진화, 독립 관측 계약이 필요하다.

특히 continuum은 열린 다리다. [15장](15_같은_이력의_비평탄_Plebanski_힌지.md)은 같은 이력에서 하나의 비평탄 constant-curvature 유한 증인까지 닫지만, 실제 proper amplitude, refinement, Einstein--Hilbert 지배와 two-DOF는 여전히 제공하지 않는다. [Finocchiaro--Oriti (2020)](https://arxiv.org/abs/2004.07361)와 [Carrozza (2024)](https://arxiv.org/abs/2404.07834)는 GFT renormalization의 방법과 남은 과제를 정리하지만, 4D Lorentzian EPRL/GFT에서 refinement-consistent Einstein dynamics와 정확히 두 자유도가 나온다는 정리를 제공하지 않는다. [Bruno et al. (2026)](https://arxiv.org/abs/2603.16999)은 너무 강한 Hilbert-space 수렴 요구가 TQFT로 굳을 수 있음을 보이고 distributional rigging-map 길을 제시한다. 그 길 역시 Einstein dynamics의 증명이 아니라 다음에 검증할 criterion이다.

그러므로 “조건부”는 얼버무린 미완성 표지가 아니다. **명시한 계약 아래 한 유한 평탄 존재 증인을 끝까지 증명했다**는 뜻이다. 계약 자체가 자연에서 유도되는가와 continuum으로 살아남는가는 다음 연구의 별도 질문이다.

## 14.6 재현 범위

증명서는 [구현](../../examples/physics/zerod_plebanski_closure.py)과 [회귀 검사](../../tests/test_zerod_plebanski_closure.py)에 있다. 집중 검사는 다음과 같다.

```powershell
.codex/hooks/python.cmd pytest tests/test_zerod_plebanski_closure.py -q
```

기록된 결과는 `27 passed`다. 인접 Lorentzian reconstructionㆍshared-faceㆍaction no-goㆍcontinuum no-go까지 묶은 검사는 `112 passed`다. 이 숫자는 선언한 유한 계약과 구현의 일관성을 확인할 뿐, 자연 또는 continuum quantum gravity의 증명은 아니다.
