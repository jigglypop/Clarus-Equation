# Research contract: one-way zero-dimensional boundary and quantum bootstrap

Status: COMPLETE  
Revision: 1 — 사용자 정정에 따라 공통 양방향 bus를 중심 모형에서 제외하고
`외부 0D → 현재 3+1D 시공간`의 단방향 경계 사상으로 재등록한다.

PREDECESSOR:

- `_workspace/ce/_archive/quantum-neighbor-bootstrap-dark-sector-20260825`
- `_workspace/ce/dark-sector-observational-census-derivation-20260825`

## 1. 연구 질문

사용자의 중심 가설을 다음처럼 고정한다.

> 현재 시공간 $M$의 바깥에 strict zero-dimensional boundary/source
> $Z=\{\star\}$가 있고, 물리적 화살표는 오직 $Z\to M$이다. $Z$에서 나온
> 양자적 출력은 선택 경로와 비선택 경로로 갈라지며, 선택 경로는 가시적
> 사건을, 비선택 경로는 residual sector를 이룬다. $M$ 안에서는 이미 실행된
> 양자가 다음 양자의 허용 전이를 여는 전방 bootstrap cascade를 이룰 수 있다.
> residual sector의 우주론적 readout이 암흑물질형 및 암흑에너지형일 수 있다.

여기서 **일방향**은 0차원 내부의 공간 방향이 아니다. 한 점에는 접선 방향이나
거리 방향이 없으므로, 방향은 두 sector 사이의 사상 또는 양자채널

$$
Z\xrightarrow{\mathcal E_{Z\to M}}M,
\qquad M\not\to Z
$$

에 부여한다. 이 run은 그 사상이 수학적으로 가능한지, 어떤 추가 동역학이
필요한지, 에너지·확률·응력 보존을 어떻게 닫아야 하는지, 그리고 무엇이 아직
암흑물질·암흑에너지의 관측 예측으로 이어지지 않는지를 판정한다.

## 2. 정의역과 용어

1. **외부 strict 0D $Z$:** 시공간 좌표, 내부 거리, 고유한 시간 미분을 갖지
   않는 경계점/원천. $Z$ 자체의 “앞·뒤”를 말하지 않는다.
2. **단방향 경계 사상:** $Z$의 경계 자료를 $M$의 상태 또는 장으로 내보내는
   completely positive map/instrument. $M$의 상태가 $Z$의 다음 출력을
   바꾸는 feedback 항은 금지한다.
3. **엄격한 1차원 입력 Hilbert 공간:** $\mathcal H_Z\cong\mathbb C$이면
   $\mathcal E_{Z\to M}$은 고정 상태 준비 $\mathcal E(1)=\rho_M$일 뿐이다.
   출력 $\rho_M$에 미리 정해진 time register나 path correlation을 인코딩할
   수는 있지만, bare input point가 입력에 따라 history를 선택하거나 clock과
   memory를 스스로 갱신하지는 못한다. 그런 갱신 규칙은 경계 조건, 채널 환경,
   또는 $M$의 동역학에 별도로 명시한다.
4. **선택/비선택 분기:** CP trace-nonincreasing maps로 이루어진 instrument

   $$
   \{\mathcal E_{\rm sel},\mathcal E_{\rm ns}\},\qquad
   \mathcal E_{\rm sel}+\mathcal E_{\rm ns}\ \text{is CPTP}
   $$

   로 쓴다. $\widetilde\rho_a=\mathcal E_a(\rho_Z)$는 정규화 전 출력이고
   $p_a=\operatorname{Tr}\widetilde\rho_a$이다. 조건부 상태
   $\rho_a=\widetilde\rho_a/p_a$는 $p_a>0$일 때만 정의한다. 두 출력은 모두
   $Z$에서 $M$ 쪽으로 나가며, 비선택 출력이 다시 $Z$로 돌아간다고 가정하지
   않는다.
5. **양자 옆의 양자가 실행:** $M$ 안의 directed edge $j\to i$에서 이미
   점유된 $j$가 $i$의 전이 채널을 열어 주는 것. 대표 jump는

   $$
   L_{i\leftarrow j}=\sqrt{\kappa_{ij}}\,\sigma_i^+n_j.
   $$

   이웃은 선언된 causal graph의 edge이며, “허용”은 에너지 공급과 동의어가
   아니다.
6. **residual map:** 비선택 history measure $\nu_{\rm ns}$를 $M$ 안의
   local/covariant field 및 stress tensor로 보내는 CE의 새 물리 사상

   $$
   \phi(x)=M_*\int_{\Gamma_{\rm ns}}
   \widehat K(x,\gamma)\,\nu_{\rm ns}(d\gamma).
   $$

   이것은 표준 조건부 양자역학만으로 자동 유도되지 않는다.

AGI, 뇌 모형, guard 및 제품 구현은 범위 밖이다.

## 3. 고정 모형

### 3.1 경계 출력

$Z$와 $M$ 사이의 첫 단계는 classical outcome register $A$를 포함해

$$
\mathcal I(\rho_Z)=
\sum_{a\in\{\mathrm{sel},\mathrm{ns}\}}
\mathcal E_a(\rho_Z)\otimes|a\rangle\langle a|
$$

로 쓴다. 비선택 record에서 history measure $\nu_{\rm ns}$를 만드는 단계는
별도 physical-map axiom이다. 이 식은 방향과 확률 장부를 정의하지만, 아직
Hamiltonian, 에너지원, 시공간 국소성 또는 residual stress tensor를 유도하지
않는다.

### 3.2 현재 차원 안의 전방 cascade

노드 상태 $x_i\in\{0,1\}$에 대해, 제시된 jumps만 남거나 Hamiltonian이 이
basis에서 diagonal이거나 충분한 decoherence로 대각 부분공간이 닫힐 때의
후보 전이율은

$$
b_i(x)=(1-x_i)\sum_{j:j\to i}\kappa_{ij}x_j,
\qquad d_i(x)=\gamma_i x_i.
$$

edge는 $Z$에서 멀어지는 세대 순서로 향한다. 유한 DAG와 무한 branching
genealogy를 구별한다. 독립 Poisson 자손, fresh target, 독립 clock, 충돌 무시
가정 아래 균일 평균 자손수 $D$의 extinction probability는

$$
q_{n+1}=\exp[D(q_n-1)],\qquad
q=\exp[-D(1-q)].
$$

이는 전방 세대 재귀이며 mutual coupling이나 strongly connected component를
요구하지 않는다.

### 3.3 단방향성의 물리 구현 후보

닫힌 두 subsystem의 Hermitian pair coupling만으로 exact source-to-target
directionality가 자동 생성된다고 가정하지 않는다. 허용 후보는 cascaded open
quantum system, chiral travelling field, reservoir engineering, 또는 측정과
feed-forward이다. 어느 경우든 환경의 noise, dissipative current, 에너지 입력과
boundary/junction condition을 장부에 포함해야 한다.

residual field가 $M$에서 중력원으로 작용하는 것은 $M\to Z$ feedback이 아니다.
그것은 경계에서 나온 두 번째 co-output이 이미 $M$ 안에서 진화한다는 뜻이다.

## 4. 등록 주장

| ID | 사전 지위 | 검증 질문 |
|---|---|---|
| ZDO-1 | Definition candidate | “0D에서 일방향”을 $Z$ 내부 방향이 아니라 sector map $Z\to M$으로 일관되게 정의할 수 있는가? |
| ZDO-2 | Conditional theorem candidate | $\mathcal H_Z\cong\mathbb C$인 CPTP map이 고정 상태 준비와 동치이며, bare 0D가 입력별 clock/history를 스스로 선택·갱신하지 못함을 보일 수 있는가? |
| ZDO-3 | Conditional theorem candidate | source가 target에 영향을 주되 target reduced dynamics가 source로 되먹임되지 않는 cascaded GKSL 구현이 존재하는가? |
| ZDO-4 | Conditional theorem candidate | directed facilitation jump가 diagonal sector에서 위 birth/death CTMC를 정확히 유도하는가? |
| ZDO-5 | Conditional theorem candidate | one-way Poisson genealogy에서 $q=e^{-D(1-q)}$와 $D>1$ 생존 조건이 따르는가? |
| ZDO-6 | Incomplete | 경계 주입과 $M$ 내부의 $\nabla_\mu T^{\mu\nu}=0$을 잇는 에너지/current junction condition은 무엇인가? |
| ZDO-7 | Axiom candidate | 비선택 출력이 residual field로 사상된다는 규칙을 CP 확률 장부와 모순 없이 추가할 수 있는가? |
| ZDO-8 | Incomplete | 위 구조가 절대 $\Omega_{\rm DM}$, $\Omega_{\rm DE}$, 그 분할 또는 섭동 스펙트럼을 예측하는가? |

## 5. 의무 반례와 판정 기준

1. strict 0D point 안에서 intrinsic spatial arrow 또는 Hamiltonian clock을 찾는
   시도를 기각한다.
2. $\dim\mathcal H_Z=1$인 channel이 여러 입력 history를 저장하거나 스스로
   갱신한다고 주장하는 것을 state-preparation 반례로 검사한다.
3. 닫힌 Hermitian source-target coupling이 exact 일방향이라고 주장할 때
   reciprocal back-action 또는 환경 누락을 검사한다.
4. 유한 directed acyclic graph에서 무한 자기실행을 주장하는 것을 유한 도달
   노드/흡수 반례로 검사한다.
5. 모든 local excitation이 0이고 경계 drive/seed가 없는데 excitation이
   생긴다는 주장을 energy/seed 반례로 검사한다.
6. 반대 화살표 $M\to Z$만 있고 $Z\to M$ 출력이 없는데 $M$의 암흑 중력원이
   생긴다는 주장을 causal decoupling 반례로 검사한다.
7. 같은 genealogy probability에 서로 다른 amplitude와 vacuum offset을 주어
   다른 cosmic abundance를 얻는 비식별성 반례를 유지한다.

기존 “모든 노드가 하나의 양방향 선형 common bus에 결합한다”는 rank-1 및
비국소성 반례를 이 run의 중심 모델로 쓰지 않는다. 그것은 사용자의 정정과
다른 비교용 기각 경로로만 보존한다.

## 6. 허용 오차와 기계 검사

- symbolic identity와 operator support 판정은 exact로 취급한다.
- CPTP/CP 판정은 Choi matrix의 최소 고유값이 정규화 오차 $-10^{-12}$ 이상이고
  trace-preservation residual이 $10^{-12}$ 이하여야 한다.
- float64 확률 고정점 residual은 $10^{-12}$ 이하만 허용한다.
- branching 수치는 iteration과 Lambert-$W$ 표현을 독립 비교한다.
- 지수·로그·확률·고정점 인자는 무차원이어야 하며, jump coefficient
  $\sqrt{\kappa}$는 $T^{-1/2}$, Lindblad generator는 $T^{-1}$ 차원이어야 한다.
- boundary energy/current와 $M$의 stress tensor는 같은 보존식 또는 명시된
  source/junction term으로 연결되어야 한다.

## 7. 대안 경로

1. $Z$를 bare Hilbert subsystem이 아니라 state-preparing boundary condition으로
   둔다.
2. exact directionality를 cascaded/chiral open channel로 구현한다.
3. 실행 순서는 $M$ 안의 directed acyclic causal graph 또는 branching tree가
   담당하고, 시간 매개변수는 $M$에 둔다.
4. residual history map은 경계 instrument와 결합하되 독립적인 local-covariant
   kernel 공리로 유지한다.
5. 완전한 미시 모형 대신 scalar residual EFT를 사용해 DM-like oscillation과
   DE-like constant term의 조건부 readout만 계산한다.

## 8. 주장 상한과 중단 조건

이 run이 허용하는 최강 결론은 다음과 같다.

> 외부 strict 0D를 동역학적 물체가 아니라 단방향 상태 준비 경계로 해석하고,
> open cascaded channel과 $M$ 내부의 directed facilitation을 별도로 명시하면
> $Z\to M$ one-way bootstrap은 수학적으로 구성 가능하다. 그러나 에너지원,
> noise/current, local-covariant residual map 및 우주론적 초기조건을 추가하지
> 않으면 실제 암흑물질·암흑에너지와 그 abundance는 유도되지 않는다.

표준 양자역학에서 cross-branch gravity가 나온다는 주장, 무상 에너지 생성,
bare 0D 안의 시간 진화, 유한 cascade의 영구 생존, 실제 암흑물질·암흑에너지와
비선택 경로의 실증적 동일성, 절대 abundance 예측은 허용하지 않는다. 완전한
반례가 있는 부모 주장은 삭제·tombstone한다. 구현은 개정된 `20-audit.md`의
Gate가 통과한 좁은 범위에서만 재개한다.
