# Mathematics lane: one-way zero-dimensional boundary and quantum bootstrap

Status: COMPLETE  
Revision: 1 — 사용자 정정 `외부 0D -> 현재 3+1D`를 처음부터 재유도한다.

## 1. 기하학적 0D와 채널의 화살표를 분리한다

외부 경계의 기하학적 지지집합을

$$
Z=\{\star\}
$$

로 둔다. 한 점에는 서로 다른 공간 위치, 내부 거리 또는 공간적 접선 방향이
없다. 따라서 “0D에서 일방향”의 일관된 뜻은 0D 안의 방향이 아니라 두 sector
사이 사상의 domain/codomain 순서

$$
Z\xrightarrow{\mathcal E_{Z\to M}}M,
\qquad M\not\to Z
$$

이다. 이 화살표는 경계 조건, 양자채널 또는 인과적 feed-forward 순서를
표현한다.

여기서 **spacetime dimension과 internal state dimension은 다르다.** 0D
지지집합에도 유한한 내부 label/algebra를 별도 부여할 수 있다. 그러나 strict
0D가 시간 좌표나 외부 evolution parameter까지 자동으로 갖는 것은 아니다.
내부 상태를 시간에 따라 갱신하면 그것은 이미 $0+1$D dynamics 또는 외부
schedule을 추가한 모형이다.

**ZDO-1: 정의로 일관됨.** 방향은 $Z$의 내재 방향이 아니라 $Z\to M$ channel
orientation이다.

## 2. bare 0D 입력은 상태 준비 채널이다

가장 작은 입력 Hilbert 공간을 $\mathcal H_Z\cong\mathbb C$로 택하자. 모든
선형 map $\mathcal E:\mathbb C\to\mathcal B(\mathcal H_M)$은 어떤 연산자
$\sigma_M$에 대해

$$
\mathcal E(z)=z\sigma_M
$$

꼴이다. complete positivity는 $\sigma_M\succeq0$, trace preservation은
$\operatorname{Tr}\sigma_M=1$을 요구한다. 따라서

$$
\boxed{\mathcal E(z)=z\rho_M}
$$

이며 이는 fixed-state preparation channel이다.

이 정리는 다음 두 문장을 구별한다.

1. $\rho_M$에 time register, path label 또는 여러 subsystem의 상관관계를
   미리 인코딩하는 것은 가능하다.
2. 유일한 입력상태를 가진 bare $Z$가 서로 다른 입력 history를 판별하고,
   memory를 갱신하며, 자체 clock으로 다음 출력을 선택하는 것은 불가능하다.

두 번째 기능을 원하면 $\dim\mathcal H_Z>1$인 internal register, 시간에 따른
channel family $\mathcal E_t$, 환경 memory, 또는 $M$의 후속 동역학을 추가해야
한다. 이는 기하학적 0D를 반드시 포기한다는 뜻은 아니지만, bare static point
이상의 구조를 도입한다는 뜻이다.

**ZDO-2: 조건부 정리.** $\mathcal H_Z\cong\mathbb C$라는 최소 모형에서만
fixed-state preparation 결론이 따른다.

## 3. 선택/비선택 확률 장부

선택과 비선택을 두 outcome의 quantum instrument로 둔다.

$$
\mathcal I(\rho_Z)=
\sum_{a\in\{\mathrm{sel},\mathrm{ns}\}}
\mathcal E_a(\rho_Z)\otimes|a\rangle\langle a|,
$$

$$
\mathcal E_a\ \text{CP and trace-nonincreasing},
\qquad
\mathcal E_{\rm sel}+\mathcal E_{\rm ns}\ \text{CPTP}.
$$

정규화 전 상태와 확률은

$$
\widetilde\rho_a=\mathcal E_a(\rho_Z),
\qquad p_a=\operatorname{Tr}\widetilde\rho_a,
\qquad p_{\rm sel}+p_{\rm ns}=1
$$

이고, 조건부 상태는 $p_a>0$일 때

$$
\rho_a=\frac{\widetilde\rho_a}{p_a}
$$

이다. 마지막 나눗셈은 입력상태에 대해 비선형이므로, 물리적 channel 자체와
postselection update를 혼동하면 안 된다.

history outcome이 연속이면 비선택 instrument-valued measure
$\mathcal I_{\rm ns}(\Delta)$를 두고

$$
\nu_{\rm ns}(\Delta)
=\operatorname{Tr}[\mathcal I_{\rm ns}(\Delta)(\rho_Z)]
$$

로 subprobability measure를 얻는다. 이 단계까지는 표준 확률·상태 장부다.
그러나 선택된 결과와 함께 비선택 결과를 같은 우주의 중력원으로 보존하는
규칙은 여기서 나오지 않는다.

## 4. 닫힌 상호작용과 열린 단방향 cascade

### 4.1 단순 Hermitian pair coupling은 상호적이다

상류 후보 $A$와 하류 후보 $B$에 대해

$$
H_{\rm pair}=g a^\dagger b+g^*b^\dagger a
$$

이면 Heisenberg 식은

$$
\dot a=-igb,\qquad \dot b=-ig^*a
$$

처럼 양쪽 변수가 서로에게 들어간다. 따라서 이 단순한 닫힌 pair Hamiltonian은
exact $A\to B$ directionality의 구현이 아니다.

### 4.2 cascaded GKSL 구성

반면 source $A$의 출력이 target $B$의 입력으로 들어가는 Markov cascade는,
위상과 rate를 coupling operators $a,b$에 흡수했을 때,

$$
\dot\rho=-i[H_A+H_B+H_{\rm cas},\rho]
+\mathcal D[a+b]\rho,
$$

$$
H_{\rm cas}=\frac{1}{2i}(b^\dagger a-a^\dagger b),
\qquad
\mathcal D[c]\rho=c\rho c^\dagger-\frac12\{c^\dagger c,\rho\}
$$

로 쓸 수 있다. $H_{\rm cas}=H_{\rm cas}^\dagger$이고 dissipator가 하나의
Lindblad operator $a+b$로 쓰였으므로 전체 generator는 GKSL 형태다.

전개하면

$$
\dot\rho=\mathcal L_A\rho+\mathcal L_B\rho
+[a\rho,b^\dagger]+[b,\rho a^\dagger].
$$

target을 부분추적하면 partial trace의 target cyclicity 때문에

$$
\operatorname{Tr}_B\big([a\rho,b^\dagger]+[b,\rho a^\dagger]\big)=0,
$$

따라서

$$
\boxed{\dot\rho_A=\mathcal L_A\rho_A}
$$

이다. source reduced dynamics에는 target이 되먹임되지 않는다. 반대로 source를
부분추적하면

$$
\operatorname{Tr}_A[a\rho,b^\dagger]
+\operatorname{Tr}_A[b,\rho a^\dagger]
$$

가 일반적으로 0이 아니므로 target은 source와 source-target correlations에
의존한다.

이 식은 요구한 단방향성의 존재 증명이지, bare 0D point가 그 channel을
자동으로 만든다는 증명이 아니다. chiral travelling field, reservoir,
isolator, collision ordering 또는 measurement-feed-forward 같은 열린 구현이
필요하다. Markov 근사라면 propagation delay와 bath correlation time의 적용
범위도 밝혀야 한다.

**ZDO-3: 조건부 구성 정리.** open cascaded channel에서는 exact upstream
no-feedback이 가능하다. 그 환경과 에너지 장부는 추가 물리다.

## 5. 현재 차원 안의 directed neighbour bootstrap

directed edge $j\to i$마다

$$
L_{i\leftarrow j}=\sqrt{\kappa_{ij}}\,\sigma_i^+n_j,
\qquad
R_i=\sqrt{\gamma_i}\,\sigma_i^-
$$

를 둔다. density matrix가 occupation basis에서 diagonal이고, 이 대각
부분공간을 깨는 coherent Hamiltonian이 없을 때

$$
\rho=\sum_xP(x)|x\rangle\langle x|
$$

를 Lindblad 식에 대입하면 정확한 CTMC 전이율

$$
x\to x^{i,+}quad\text{at}\quad
b_i(x)=(1-x_i)\sum_{j:j\to i}\kappa_{ij}x_j,
$$

$$
x\to x^{i,-}quad\text{at}\quad
d_i(x)=\gamma_i x_i
$$

를 얻는다. coherent off-diagonal Hamiltonian을 추가하면 population만의 식이
일반적으로 닫히지 않으므로 위 결론은 그 조건 밖으로 확장하지 않는다.

노드 에너지를 $H_M=\sum_i\epsilon_i n_i$로 두면 upward jump는 일반적으로
$\epsilon_i$만큼 $M$의 excitation energy를 늘린다. $n_j$는 전이를 **gate**할
뿐 그 에너지원을 자동으로 제공하지 않는다. source field, pump 또는 reservoir
current가 별도로 필요하다.

차원은

$$
[\kappa_{ij}]=[\gamma_i]=T^{-1},
\qquad[L]=[R]=T^{-1/2}
$$

이고 $\mathcal D[L]$는 $T^{-1}$이다.

**ZDO-4: 조건부 정리.** 선언한 jump들과 diagonal/decohered sector에서
directed CTMC가 정확히 따른다.

## 6. 유한 DAG와 무한 branching을 구별한다

### 6.1 유한 directed graph

유한 DAG, 유한 seed, 외부 재주입 없음에서 다음을 구별한다.

1. $\gamma_i=0$이면 exclusion $x_i\in\{0,1\}$ 때문에 각 노드는 최대 한 번
   켜진다. birth 수는 도달 가능한 노드 수 이하이며 유한하다.
2. 모든 $\gamma_i>0$이면 vacuum은 absorbing이고 모든 상태에서 유한한 decay
   순서를 통해 도달 가능하다. 유한 CTMC에서 vacuum은 유일한 closed class가
   되어 거의 확실히 흡수된다.
3. 영구 점유 root, 반복 boundary drive 또는 $\gamma=0$ parent를 두면 유한
   graph에서도 downstream birth/death activity가 계속될 수 있다. 이것은
   무원인 self-bootstrap이 아니라 명시된 지속 seed/drive의 결과다.

따라서 유한 feed-forward graph만으로 영구 생존을 주장할 수 없다.

### 6.2 무한 독립 Poisson genealogy

각 실행된 노드가 fresh target에 독립적으로 Poisson$(D)$개의 자손을 만들고,
부모별 clock이 독립이며 collision과 target reuse를 무시하는 이상화에서는
offspring generating function이

$$
G(s)=\exp[D(s-1)]
$$

이다. 한 ancestor에서 $n$세대 안에 소멸할 확률을 $q_n$이라 하고 $q_0=0$으로
두면

$$
q_{n+1}=G(q_n)=\exp[D(q_n-1)].
$$

극한은 최소 고정점

$$
q=e^{-D(1-q)}
$$

이다. $D\le1$이면 $q=1$, $D>1$이면 $q<1$인 작은 해가 있고

$$
q=-\frac1D W_0(-De^{-D}),
\qquad 1-q>0.
$$

CE의 등록값

$$
D=3.1777584234099736
$$

을 넣으면

$$
q=0.048646719644028225,
$$

$$
1-q=0.9513532803559718,
\qquad Dq=0.15458752312007412<1,
$$

이며 fixed-point residual은 float64에서 0이다. 이 유도는 본질적으로 전방
세대 재귀다. reciprocal edge나 strongly connected component는 필요 없다.
대신 무한 fresh-target branching 가정이 필요하다.

**ZDO-5: 조건부 정리.** 수치는 맞지만 genealogy extinction/survival
probability일 뿐 cosmic density fraction이 아니다.

## 7. 비선택 출력에서 residual field로 가는 추가 사상

subnormalized 비선택 measure를 보존해

$$
\phi(x)=M_*\int_{\Gamma_{\rm ns}}
\widehat K(x,\gamma)\,\nu_{\rm ns}(d\gamma)
$$

를 선언할 수 있다. $\nu_{\rm ns}$와 $\widehat K$가 무차원이고 $[M_*]=E$이면
$[\phi]=E$이다. 그러나 이 식은 standard conditioning에서 나오지 않는다.
표준 instrument에서 selected outcome으로 조건화하면 다른 outcome은 그
조건부 상태에 자동으로 더해지지 않는다. 따라서 “비선택 경로도 같은 $M$에서
중력원으로 남는다”는 것은 CE의 독립 physical-map axiom이다.

특히 $\nu_{\rm ns}/p_{\rm ns}$로 조건부 정규화하면 전체 비선택 weight가
사라진다. 암흑 abundance를 논하려면 subnormalized weight, kernel amplitude,
$M_*$와 no-double-counting 규칙을 함께 유지해야 한다.

선언된 최소 scalar EFT는

$$
S_{\rm res}=\int d^4x\sqrt{-g}
\left[-\frac12(\nabla\phi)^2
-\frac12m^2\phi^2-V_\Lambda\right]
$$

이다. 빠른 quadratic oscillation은 조건부로 pressureless DM-like 평균을,
constant $V_\Lambda$는 정확히 $w=-1$인 DE-like 항을 줄 수 있다. 이는
readout의 존재 예이지 실제 암흑 성분의 동일성 증명이 아니다.

**ZDO-7: physical-map axiom.** 표준 양자역학에서의 유도는 미완성이다.

## 8. 에너지와 응력 보존의 두 경우

### 8.1 초기자료만 준비하는 경우

$Z\to M$이 한 번의 initial/boundary state preparation이고 이후 $M$ 안의
selected+residual sector가 닫혀 진화한다면 요구식은

$$
\nabla_\mu
\left(T_{\rm vis}^{\mu\nu}+T_{\rm res}^{\mu\nu}
+T_{\rm channel}^{\mu\nu}\right)=0
$$

이다. channel을 이미 적분 제거했다면 그 효과가 유효 작용과 초기조건에
일관되게 포함되어야 한다.

### 8.2 계속 주입하는 경우

$Z$가 $M$의 시간 동안 계속 에너지·운동량을 주입한다면

$$
\nabla_\mu T_M^{\mu\nu}=J_Z^\nu
$$

만 적고 끝낼 수 없다. Einstein tensor의 Bianchi identity와 양립하려면
source/channel/boundary를 포함한 전체 계에서

$$
\nabla_\mu T_{\rm total}^{\mu\nu}=0
$$

이 되도록 junction current 또는 수정된 중력 방정식을 제시해야 한다. 현재
모형은 $J_Z^\nu$, matching hypersurface 및 noise kernel을 아직 유도하지
않았다.

**ZDO-6: 미완성.** one-way CP dynamics의 존재와 우주론적 stress closure는
서로 다른 문제다.

## 9. abundance 비식별성

branching 자료 $(D,q)$와 instrument probability만으로는

$$
M_*,\quad\widehat K,\quad m,\quad A_\phi,\quad V_\Lambda,
\quad J_Z^\nu
$$

가 정해지지 않는다. 그런데 homogeneous scalar density는

$$
\rho_\phi=\frac12\dot\phi^2+\frac12m^2\phi^2+V_\Lambda
$$

에 의존한다. 같은 $q$를 고정한 채 $M_*$, $A_\phi$ 또는 $V_\Lambda$를 바꾸면
$\Omega_{\rm DM}$와 $\Omega_{\rm DE}$가 연속적으로 달라진다. 따라서

$$
q\ne\Omega_{\rm DM},\qquad1-q\ne\Omega_{\rm DE}
$$

이며, 추가 normalization law 없이 어느 재배치도 허용되지 않는다.

**ZDO-8: 완전한 비식별성 반례.** 현재 구조만으로 절대 abundance나 분할을
예측한다는 부모 주장은 기각된다.

## 10. 반대 화살표의 대조 반례

오직

$$
M\to Z,qquad Z\not\to M
$$

인 sink 모형이면 $M$의 reduced equation에는 손실/감쇠가 생길 수 있지만,
$Z$에서 $M$으로 돌아오는 새 residual field 또는 양의 stress source는 없다.
따라서 이 반대 방향만으로는 $M$ 안의 암흑물질형·암흑에너지형 중력원을 만들
수 없다. 그것을 만들려면 $Z\to M$ co-output 또는 별도의 modified-gravity
항을 다시 추가해야 한다.

## 11. 개정 상태표

| Claim | Mathematical status |
|---|---|
| ZDO-1: 0D의 일방향은 $Z\to M$ channel orientation이다 | definition, consistent |
| ZDO-2: $\mathcal H_Z\cong\mathbb C$ channel은 fixed-state preparation이다 | conditional theorem |
| ZDO-3: open cascade에서 upstream reduced dynamics는 downstream과 독립이다 | conditional construction theorem |
| ZDO-4: facilitated jumps의 diagonal sector는 directed birth/death CTMC다 | conditional theorem |
| ZDO-5: infinite Poisson genealogy는 $q=e^{-D(1-q)}$를 따른다 | conditional theorem |
| ZDO-6: boundary energy/current와 cosmological stress가 닫힌다 | incomplete |
| ZDO-7: nonselected history가 residual stress로 사상된다 | CE physical-map axiom |
| ZDO-8: $(D,q)$가 dark abundance를 정한다 | non-identifiability no-go |
| $M\to Z$-only sink가 $M$의 dark source를 만든다 | causal no-go |

가장 강하게 살아남는 결론은 다음뿐이다.

$$
\boxed{
\text{static 0D boundary preparation}
\to\text{open one-way channel}
\to\text{directed in-$M$ cascade}
}
$$

는 조건부로 수학적 구성이 가능하다. residual-to-gravity map, junction current,
미시적 에너지원과 cosmological normalization은 추가 공리 또는 미완성 다리다.

## Appendix A. 이전 common-bus 경로의 지위

이전 run의 reciprocal linear bus

$$
K=G\mathcal G G^\dagger,
\qquad\operatorname{rank}K\le r
$$

정리와 single-mode sparsity 반례는 수학적으로 유효하다. 다만 이것은 현재
사용자가 고정한 $Z\to M$ cascade의 중심 모형이 아니다. “모든 노드가 하나의
reciprocal mode에 결합한다”는 별도 후보를 기각하는 비교 결과로만 보존한다.
