# 14. 역방향 0D--Plebanski 조건부 완결

이 장이 보이는 것은 bare $0$차원 seed가 저절로 일반상대론을 낳는다는 명제가 아니다. 이미 고정한 rank-$4$ 생성 규칙, Lorentzian 중력 sector, Plebanski 작용과 측도, refinement 및 적외선 수용 조건을 함께 채택하면, 그 입력들로부터 $3+1$ Lorentzian Plebanski/Einstein 기하까지 이어지는 **조건부 충분성 사슬**을 재현 가능하게 조립할 수 있음을 보인다. 이 결론의 정확한 형식 지위와 반증 경계는 [차원 분류 원장 C4](../검증_원장/참조_차원_분류_원장.md#조건부-0dlorentzian-plebanskieinstein-폐쇄)에 고정돼 있으며, 이 장은 그 원장을 바꾸지 않는다.

순서를 거꾸로 잡는 이유는 출발점에서 빠진 것을 말로 채우지 않기 위해서다. 끝점인 Einstein 방정식에는 어떤 기하ㆍ작용ㆍ저에너지 조건이 필요한지가 알려져 있다. 그 필요 입력을 한 칸씩 뒤로 옮기면, $0$차원 관계 규칙만으로는 정해지지 않는 선택이 어디인지 드러난다. 경로는 `Plebanski gravitational sector → tetrad/metric → stationaryㆍdecoherent common-metric sector → Planck readout quotient → rank-4 split/merge 2-complex → 0D seed`이다. 이 장은 CE의 서사인 **끼임 → 접힘 → 암흑 표현** 가운데 앞의 두 단계를 조건부 기하로 정리한다. 접힌 에너지의 우주론 readout과 암흑부문 경험 확인은 여기서 닫지 않는다.

## 14.1 무엇을 뜻하는가

여기서 $0$차원은 아주 작은 공간을 뜻하지 않는다. 좌표, 거리, 계량, 시간 간격이 아직 주어지지 않은 관계적 seed를 뜻한다. 따라서 seed 하나에서 길이 또는 차원을 측정할 수 없다. split과 merge는 그 seed의 typed relation을 합성하는 생성 규칙이며, 이 규칙에 transport와 amplitude를 추가해야 face, 곡률, 기하를 말할 수 있다.

접힘은 선택되지 않은 이력이 없어지는 일을 뜻하지 않는다. 거친 readout에서 한 class로 표시되지 않는 microscopic 성분이 reduced state의 양의 대각 norm으로 남는다는 뜻이다. 이 용어는 collapse나 하나의 존재론적 branch를 뜻하지 않는다. 독자는 이 구별을 유지해야 이후의 Gibbs 집중을 “나머지 가능성의 삭제”로 오독하지 않는다.

이 장에서 쓰는 $B^{IJ}$는 내부 Lorentz 지수 $I,J=0,1,2,3$를 갖는 2-form, $A^{IJ}$는 연결 1-form, $F^{IJ}[A]$는 그 곡률 2-form이다. $e^I$는 tetrad 1-form이고 $g_{\mu\nu}=\eta_{IJ}e^I{}_{\mu}e^J{}_{\nu}$가 복원된 metric이다. 이 기호들은 seed에 처음부터 있지 않다. 아래의 Lorentzian sector와 nondegeneracy 가정 안에서만 등장한다.

## 14.2 끝점: Plebanski 중력 sector에서 Einstein 방정식으로

출발점은 10장에서 이미 채택한 chiral Plebanski형 $BF$ 작용이다. internal index $i,j=1,2,3$의 multiplier $\Psi_{ij}$는 대칭이고 trace-free, 즉 $\Psi_{ij}=\Psi_{ji}$ 및 $\delta^{ij}\Psi_{ij}=0$이라고 둔다. 이 제약을 먼저 밝혀야 multiplier variation이 어느 simplicity condition을 주는지 분명해진다.

$$
\begin{aligned}
S[\Sigma,A,\Psi]
=\frac1{8\pi G}\int_M\left[
\Sigma^i\wedge F^i[A]
-\frac12\left(\Psi_{ij}+\frac\Lambda3\delta_{ij}\right)
\Sigma^i\wedge\Sigma^j
\right].
\end{aligned}
$$

여기서 $\Sigma^i$는 chiral 2-form이고 $F^i[A]$는 $A^i$의 곡률이다. multiplier의 trace-free 대칭 variation은

$$
\Sigma^i\wedge\Sigma^j
-\frac13\delta^{ij}\Sigma^k\wedge\Sigma_k=0
$$

라는 simplicity condition을 준다. 비퇴화 gravitational branch에서는 $\Sigma$가 tetrad의 self-dual 2-form과 동등해진다. 이 고전적 출발점은 [Plebanski (1977)](https://doi.org/10.1063/1.523215)에 있다.

경계에서 $\delta A$가 사라진다고 하고 세 변수의 variation을 모으면 다음 식을 얻는다.

$$
\begin{aligned}
8\pi G\,\delta S
&=\int_M\delta\Sigma^i\wedge
\left[F^i-\left(\Psi^i{}_j+\frac\Lambda3\delta^i{}_j\right)\Sigma^j\right]
&& \text{$\Sigma$ variation} \\
&\quad-\int_M D_A\Sigma^i\wedge\delta A^i
&& \text{부분 적분과 경계조건} \\
&\quad-\frac12\int_M\delta\Psi_{ij}\,\Sigma^i\wedge\Sigma^j
&& \text{trace-free symmetric $\Psi$ variation}.
\end{aligned}
$$

따라서 Euler--Lagrange equations는 $D_A\Sigma^i=0$ 및

$$
F^i[A]=\left(\Psi^i{}_j+\frac\Lambda3\delta^i{}_j\right)\Sigma^j
\tag{1}
$$

를 만족한다. nondegenerate simplicity branch에서 첫 식은 compatible self-dual connection을 정하고, 식 (1)은 Weyl curvature $\Psi$와 cosmological 부분을 분리한다. tetrad로 복원하면 vacuum Einstein equation $G_{\mu\nu}+\Lambda g_{\mu\nu}=0$이다. 표준 metric matter action을 더하고 $c$를 복원하는 관례에서는 $G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G T_{\mu\nu}/c^4$가 된다. 이 절은 선택한 action과 sector의 고전적 충분조건을 재현했을 뿐, 아래 microscopic model이 이 action을 유일하게 선택했다고 주장하지 않는다.

spin-foam 쪽에서는 simplicity가 $BF$ 이론의 자유도를 중력 sector로 제한하는 구조를 사용한다. [Barrett et al.](https://arxiv.org/abs/0907.2440) 및 [Barrett et al.](https://arxiv.org/abs/0902.1170)은 EPRL 4-simplex amplitude의 large-spin 점근에서 Regge 작용과 여러 saddle sector가 함께 나옴을 분석했다. 따라서 표준 EPRL amplitude의 cosine형ㆍmultiple-sector 구조를 한 개의 Einstein history로 바꾸어 읽을 수 없다. [Engle--Vilensky--Zipfel](https://arxiv.org/abs/1505.06683)의 proper vertex는 그 점근 전개에서 한 항을 고른다. 다른 경로는 gravitational-sector projector와 record의 정의를 명시 입력으로 공급하고, 그 뒤 서로 다른 coarse class에 decoherence를 적용하는 것이다. decoherence만으로는 gravitational sector를 고르지 않는다.

## 14.3 한 metric을 공유하는 조건

국소 face마다 $B$가 simple하다는 사실만으로 이웃 cell들이 하나의 metric을 공유하지는 않는다. 비퇴화 common-metric sector에는 unit timelike normal, linear simplicity, face closure, secondary simplicity, shared-face shape matching, proper orientation, Lorentzian reality와 nonzero oriented volume이 함께 필요하다. 이 조건 아래 finite bivector에서 tetrad와 face metric을 역복원하고, 이웃 cell의 공통 face를 같은 기하로 glue할 수 있다.

이 조건은 정리의 가정이지 coarse graining의 자동 결과가 아니다. 특히 13장의 Euclidean common-orbit closure는 그 제한된 algebra 안에서 exact block closure를 보이지만, Lorentzian shared-cell matching이 RG attractor라는 결론을 주지 않는다. 반복한 mismatch가 저절로 평균되는 명제도 반례로 삭제돼 있다. 여기서는 common-metric을 defect가 영인 조건으로 둔다. 14.6절의 유한 $\beta$에서는 이를 soft weight로 집중시키고, hard constraint는 $\beta\to\infty$의 이상적 한계에서만 뜻한다.

## 14.4 stationary phase는 어떤 선택을 하는가

각 discrete history $h$에 action $S_h$와 measure/contour를 선언하면 amplitude는 $\exp(iS_h/\hbar)$의 합 또는 적분으로 정의된다. $S_h/\hbar$는 무차원이어야 한다. 큰 action scale에서 phase가 빠르게 변하는 이력은 서로 상쇄되고, stationary history는

$$
\delta S_h=0
$$

을 만족한다. 이 식은 stationary phase의 조건이며, 그것만으로 common metric, proper orientation, continuum limit를 고르지 않는다. 그러므로 이 장의 contract는 stationary condition, 유한 $\beta$의 soft common-metric weight 또는 그 $\beta\to\infty$ hard-limit, 그리고 선택한 proper Lorentzian sector를 동시에 요구한다.

이 구별은 EPRL 점근의 경계와 맞닿는다. large-spin 결과가 Regge 위상을 포함해도 multiple saddle과 orientation sector가 남는다. proper vertex 또는 명시한 gravitational-sector projector를 공급하지 않으면, decoherence만으로 한 sector를 고르지 못한다. [Han et al.](https://arxiv.org/abs/2110.10670)은 Lorentzian spin-foam에서 복소 임계점을 수치적으로 찾아 curved Regge 기하와 연결한 증거를 제시한다. 이는 continuum theorem이 아니다.

## 14.5 끼임 뒤에도 접힘은 보존된다

유한 history basis $\{|h\rangle\}$와 environment record $\{|r_h\rangle\}$를 두고, coarse class $a$에 대한 상태를

$$
|\Psi\rangle=\sum_h c_h|h\rangle|r_h\rangle,
\qquad
\sum_h|c_h|^2=1
$$

로 둔다. 서로 다른 coarse class가 직교 record를 가질 때, environment를 trace한 reduced matrix의 class 사이 원소는

$$
\begin{aligned}
(\rho_{\rm red})_{hh'}
&=c_hc_{h'}^*\langle r_{h'}|r_h\rangle \\
&=0
&& \text{서로 다른 coarse class의 record 직교성}.
\end{aligned}
$$

반면 trace는 $\sum_h|c_h|^2=1$로 보존된다. rendered class 밖 history들의 대각항 합도 음수가 되거나 사라지지 않는다. 이것이 정확한 유한 folded-norm 정리의 내용이다. 물리 환경이 언제 이 직교 isometry를 제공하는지는 별도 동역학 문제다.

## 14.6 constraint 집중은 삭제가 아니다

common-metric 조건에서 벗어난 정도를 $\Delta_h\ge0$라 하고, 모든 history의 base weight를 $q_h>0$라 하자. $\beta\Delta_h$가 무차원인 Gibbs 분포는

$$
p_\beta(h)=\frac{q_h e^{-\beta\Delta_h}}{Z_\beta}
$$

이다. defect가 영인 good set을 $G$로, bad history의 최소 defect gap을 $\delta>0$로 둔다. $Q_{\rm good}=\sum_{h\in G}q_h$, $Q_{\rm bad}=\sum_{h\notin G}q_h$라 쓰면 다음 bound를 얻는다.

$$
\begin{aligned}
P_\beta(G^c)
&=\frac{\sum_{h\notin G}q_h e^{-\beta\Delta_h}}{Z_\beta} \\
&\le\frac{e^{-\beta\delta}Q_{\rm bad}}{Q_{\rm good}}.
\end{aligned}
\tag{2}
$$

**증명.** bad set에서는 $\Delta_h\ge\delta$이므로 분자는 $e^{-\beta\delta}Q_{\rm bad}$ 이하이다. 분모는 good set의 $\Delta_h=0$ 항을 포함하므로 $Z_\beta\ge Q_{\rm good}$이다. 두 부등식을 결합하면 식 (2)를 얻는다. $\square$

모든 유한 $\beta$에서 $q_h>0$이면 모든 $p_\beta(h)>0$이다. 따라서 이 정리는 끼임이 common-metric sector로 집중할 수 있음을 보이되, 접힌 history가 없어졌다고 말하지 않는다. $\Delta_h$를 어떤 물리 constraint로 택할지는 증명된 결과가 아니라 모형의 입력이다.

## 14.7 Planck readout과 2-complex

Planck resolution reading은 이 모형이 채택하는 정의이자 공리다. 그것은 존재가 그보다 짧은 길이에서 끊긴다는 선언이 아니다. 물리량 $O$와 같은 차원을 갖는 기준량 $O_P$를 사용해 $O/O_P$라는 무차원 label로 읽고, 같은 finite bin에 든 microscopic history를 동치화하는 classical-geometric readout resolution으로 **정의한다**. 길이에서는 $[O_P]=[\ell_P]=L$, 면적에서는 $[O_P]=[\ell_P^2]=L^2$다. 그러므로 quotient는 history를 지우지 않고 visible label만 합친다.

그 label 아래의 국소 조합은 rank-$4$ coordinate-free simplex interaction으로 둔다. 이 interaction은 다섯 boundary atom과 열 개의 paired codimension-two strand를 주며 4-simplex 조합론을 갖는다. [Reisenberger--Rovelli](https://arxiv.org/abs/gr-qc/0002083)는 이런 국소 vertex의 Feynman 전개가 2-complex들의 합으로 조직되는 spin-foam 경로를 제시한다. 선언한 split/merge composition에서 vertex, edge, face가 생기고, oriented face $f$의 transport는

$$
U_f=\prod_{e\subset\partial f}U_e
$$

로 정의된다. $U_f\ne1$이면 그 face는 finite holonomy curvature의 carrier다. 이 문장은 rank와 transport가 있을 때의 조합론적 결과다. singleton seed가 이 interaction이나 Lorentz group을 유일하게 고른다는 문장이 아니다.

## 14.8 왜 조건부로 $3+1$인가

$B_2$와 $F_2$가 같은 2-form type이고 background metric 없이 $B_2\wedge F_2$를 top form으로 쓴다고 가정하자. wedge의 차수는 $2+2=4$이므로 action density는 4-form이어야 한다. 여기에 시간 방향 하나를 선언하면 $3+1$ 표현을 얻는다. 이 form-degree lemma는 강력하지만 선택 원리는 아니다. bare $0$차원 seed가 다른 rank, 다른 form content, 다른 signature를 배제했다는 결론은 여기서 나오지 않는다.

## 14.9 Full-Chain Conditional Closure Theorem

**정리 (Full-Chain Conditional Closure).** 다음 입력을 고정하자.

1. 좌표와 metric이 없는 typed seed 및 rank-$4$ split/merge generator를 둔다.
2. 생성된 2-complex의 face에 proper $SO^+(1,3)$ transport, Lorentzian gravitational sector, Plebanski action과 measure/contour를 둔다.
3. $B_2\wedge F_2$ top-form 조건과 선언한 시간 방향을 둔다.
4. nondegenerate cell이 linear/secondary simplicity, closure, reality, proper orientation, shared-face shape matching을 만족하는 zero-defect common-metric condition을 둔다.
5. stationary-phase condition, 직교 environment record, 양의 finite Gibbs weight와 positive defect gap을 둔다. 유한 $\beta$에서는 4번을 soft하게 집중시키며, hard condition은 $\beta\to\infty$ 한계에서만 사용한다.
6. $[O_P]=[O]$인 Planck readout quotient를 두고, distributional refinement/rigging-map criterion 및 Einstein--Hilbert 지배ㆍ정확히 두 질량없는 spin-$2$ 자유도라는 적외선 acceptance gate를 둔다.

그러면 유한 단계에서는 split/merge에서 2-complex와 face holonomy가, record 조건에서 norm을 잃지 않는 decoherent coarse class가, Gibbs condition에서 zero-defect common-metric sector의 정량적 집중 bound가 나온다. nondegenerate common-metric stationary gravitational sector의 고전 방정식은 Lorentzian Plebanski/Einstein 기하로 환원한다. $3+1$은 3번의 form-degree 조건에서 얻는다.

**증명.** 1번과 2번에서 14.7절의 rank-$4$ 조합론 및 face holonomy를 얻는다. 3번에서 14.8절의 4-form 및 선언한 시간 방향이 $3+1$ 표현을 준다. 4번은 local bivector의 tetrad/metric 복원과 shared-face gluing의 가정이다. 5번의 record 직교성은 14.5절의 정확한 reduced off-diagonal 소멸과 trace 보존을 주며, positive gap은 14.6절의 식 (2)를 준다. stationary condition과 proper gravitational sector를 4번의 복원된 tetrad에 적용하면 14.2절의 variation이 Einstein 방정식을 준다. 마지막 6번은 이 유한 결론을 continuum GR의 증명으로 바꾸지 않고, 허용할 IR theory를 판정하는 별도 gate로 남긴다. $\square$

이 정리는 존재와 충분성의 조건부 정리다. 이 input들이 bare seed에서 유일하게 나왔다는 정리도, 자연이 이 sector를 채택했다는 경험 주장도 아니다.

## 14.10 되돌릴 수 없는 모형 선택과 continuum 경계

원장의 no-go는 세 종류의 지름길을 차단한다. chain/fork 자료는 bare $0$차원에서 unique growth나 차원이 따라오지 않음을 보인다. finite incidence와 simplicity는 unique action 또는 measure를 정하지 않는다. $R$과 $R+\alpha R^2$가 같은 flat stationary data를 공유하는 반례는 finite data가 Einstein--Hilbert 지배나 정확히 두 편극을 보장하지 않음을 보인다. 그래서 rank-$4$ generator, action/measure, proper sector, shape matching, refinement와 IR gate는 잉여 장식이 아니라 이 정리가 쓰는 불가약 모형 선택이다.

continuum에는 더 조심스러운 표현이 필요하다. triangulation마다 Hilbert space가 강하게 수렴한다고 요구하면 theory가 TQFT로 굳을 수 있다는 경계가 있다. [Bruno et al. (2026)](https://arxiv.org/abs/2603.16999)은 strong-convergence에서 TQFT로 가는 경계를 분석하고 distributional refinement/rigging-map 경로를 제안한다. 그것은 Einstein dynamics나 두 자유도의 IR limit를 증명하지 않는다. 그러므로 이 장의 continuum 문장은 “그 criterion과 acceptance gate가 통과할 때에만 GR 후보로 받아들인다”까지다.

## 14.11 재현 범위

유한 witness는 다음 구현과 회귀에서 C4의 algebraic/combinatorial 전제를 검사한다.

```text
examples/physics/zerod_plebanski_closure.py
tests/test_zerod_plebanski_closure.py
```

집중 실행 명령은 다음과 같다.

```powershell
.codex/hooks/python.cmd pytest tests/test_zerod_plebanski_closure.py -q
```

기록된 결과는 `20 passed`다. 이 통과는 face 조합, folded norm, Gibbs bound, finite stationary witness, local tetrad/shared-face 조건, local higher-curvature suppression gate와 constant-curvature Einstein identity의 구현 일관성을 확인한다. 그것은 continuum quantum gravity의 물리적 증명이나 CE의 암흑 표현의 경험적 확인이 아니다.

## 14.12 읽은 뒤 남는 질문

이 장의 완결은 “어떤 계약 아래 0D 관계적 언어를 Plebanski/Einstein 기하까지 모순 없이 연결하는가”에 한정된다. CE의 마지막 서사 단계인 암흑 표현은 별도 readout map, 보존 법칙, 우주론적 진화와 독립 관측 계약을 요구한다. 접힌 norm이 남는다는 정리와 그것이 암흑부문 에너지로 관측된다는 명제는 다른 문장이다. 그 간격을 닫기 전에는 이 장의 조건부 GR bridge를 우주론의 최종 증명으로 읽지 않는다.
