# 우주론·양자론 무한 자기재귀 참조함수 정비 계약

Date: 2026-08-20  
Status: COMPLETE  
Scope: equation repair and formal-status audit

## 1. 목적

CE에서 `무한 자기재귀`라고 불린 구조를 다음 세 연산으로 분리하고, 각
연산의 정의역·치역·반복 변수·고정점·수렴 조건을 다시 고정한다.

1. Poisson 분지확률 사상 (F_D:[0,1]\to[0,1])
2. 양자 상태의 CPTP 사상 \(\mathcal E:\mathsf D(\mathcal H)\to
   \mathsf D(\mathcal H)\)
3. 우주론 상태의 시간진화 사상 또는 흐름 \(\Phi_{t_2,t_1}\)

공통 고정점 표기만으로 세 물리계를 동일시하지 않는다. 정비 결과는
`끼임 → 접힘 → 암흑 표현` 서사에서 어느 화살표가 정리이고 어느
화살표가 공리·미완성인지 보존해야 한다.

## 2. 판정 대상

- **SR-1**: 무한 재귀는 원시적인 자기참조 정의가 아니라 초기값이 있는
  유한 합성열 (x_{n+1}=T(x_n))과 그 극한으로 정의되어야 한다.
- **SR-2**: (F_D(x)=\exp[-D(1-x)])의 exp 인자는 무차원이고,
  (D\ge0, x\in[0,1])에서만 Poisson 생성함수 해석을 갖는다.
- **SR-3**: (D>1)에서 (x=1)과 최소근 (q_{\rm ext})을 구분하고,
  초기값·정의역 없이 "유일한 자기일관해"라고 쓰지 않는다.
- **SR-4**: 양자 사상의 무한 반복은 선형 CPTP 사상의 합성이며,
  일반적으로 Poisson 비선형 사상 (F_D)와 같지 않다.
- **SR-5**: 양자 반복의 수렴은 CPTP라는 사실만으로 보장되지 않는다.
  주변 스펙트럼, 주기 궤도, 고정점 부분공간을 별도로 검사한다.
- **SR-6**: quantum-to-branching은 CP reduced dynamics, population 폐쇄,
  Markov jump rate와 genealogy가 실제로 구성될 때만 허용한다.
  **[판정 수용, 2026-08-22, revise contract 1/2]** 20-audit §2에 따라 의무
  2건을 추가한다: (i) 실기록 outcome을 갖는 instrument/unravelling 지정
  (nonselective channel만으로는 branching matrix 비식별), (ii) reproduction
  count의 확률공간과 세대 조건부 독립성 명시. salvage된 좁은 조건부 정리
  (지정 instrument record + Markov counting + 조건부 독립 계보 + 식별된
  $A\ge0$ ⇒ $F_A$는 기록된 classical genealogy의 확률생성함수)는 유지하고,
  bridge 전체는 [미완성]으로 남는다.
- **SR-7**: 우주론 fixed point는 물리적 시간진화 방정식과 stress tensor가
  주어져야 attractor로 불릴 수 있다. (F_D)의 계산 반복 횟수를 우주
  시간·상전이·인플레이션으로 읽지 않는다.
- **SR-8**: (q_{\rm ext}\mapsto\Omega_b)와 암흑부문 분할은 확률
  고정점 정리의 결론이 아니다. 공변 current·yield·stress readout이
  없으면 `[공리]` 또는 `[미완성]`으로 유지한다.

## 3. 필수 반례

1. (x_0=1)이면 (D>1)에서도 영원히 자명근에 머무는 초기값 반례.
2. unitary quantum channel의 비수렴/주기 궤도 반례.
3. dephasing channel의 비유일 고정점 부분공간 반례.
4. 동일한 stationary set을 갖지만 서로 다른 동역학·potential의 비유일성.
5. 같은 (q_{\rm ext})에 서로 다른 에너지 weight/yield를 붙여 서로 다른
   \(\Omega\)를 만드는 확률-to-density 반례.

완전한 반례가 맞은 보편 부모 주장은 축소하거나 활성 정본에서 제거한다.

## 4. 수치·무차원 게이트

- exp/log 인자와 확률·고정점 상태는 무차원이어야 한다.
- scalar 및 multitype 반복의 residual, branch, basin을 검사한다.
- quantum channel은 trace, Hermiticity, positivity/CP 전제, fixed-space
  dimension, peripheral eigenvalue를 기록한다.
- cosmological flow는 시간 변수와 generator의 단위가 상쇄되는지 검사한다.
- 기계 검산은 수학 무결성만 증명하며 물리 bridge를 승격하지 않는다.

## 5. 변경 규칙

수학 감사와 1차 출처 검증이 끝나기 전 정본 식을 수정하지 않는다. 이후
최소 변경으로 공통 반복자 정의, 세 물리 타입의 분리, 잘못된 보편 표현을
교정하고 focused validation만 실행한다. M4-R 연구는 이 run 동안 정지된
상태로 보존한다.

## 7. 뇌 재귀 bridge 확장

사용자 지시에 따라 scope를 BrainRuntime의 다중시간척도 재귀까지
확장한다. 추가 판정 대상은 다음과 같다.

- **BR-1:** signed recurrent coupling
  \(W_{ij}:j\text{ sender}\to i\text{ receiver}\)와 nonnegative
  offspring matrix \(A_{ji}^{\rm brain}\)을 분리한다.
- **BR-2:** 빠른 state update는 명시한 projection 또는 실제 contraction
  조건으로 activation bound를 보장해야 한다.
- **BR-3:** lifecycle mask, TopK active set, STDP spike predicate와
  genealogy event 중 하나를 canonical event receipt로 고정한다.
- **BR-4:** offspring estimator는 causal parent-child assignment,
  delay, no-double-count와 zero-denominator 정책을 가져야 한다.
- **BR-5:** \(\rho(A)<1\)인 안정한 subcritical brain process에서는
  \(q_\infty=1\)이므로 유한 horizon persistence
  \(s_H=1-q^{(H)}\)를 주 readout으로 사용한다.
- **BR-6:** eligibility는 row-post/column-pre orientation을 보존하고,
  weight는 block 안에서 동결한 뒤 block 끝에 한 번만 갱신한다.
- **BR-7:** quantum layer는 instrument outcome
  \(y\mapsto u=\psi(y)\)만 BrainRuntime input으로 보낸다.
- **BR-8:** 우주론의 \(D_{\rm eff}\), \(q_{\rm ext}\)와 density fraction을
  brain target, TopK ratio, homeostatic setpoint로 사용하지 않는다.

필수 P0 반례는 activation bound 위반, receiver-row Dale sign,
bounded 2-cycle, event-definition mismatch, stationary Hawkes와
supercritical survival의 충돌이다. 전체 후보식은
artifacts/brain-recursive-bridge-equations.md에서 관리한다.

## 6. 산출물

- `10-sources.md`: 표준 양자채널·분지과정·우주론 동역학 출처
- `11-math.md`: 타입, 수렴, 가지, 반례, 무차원 감사
- `12-routes.md`: 가능한 공통 추상함수와 물리별 독립 구현 경로
- `20-audit.md`: 형식 지위와 수정 승인
- `30-implementation.md`: 실제 정본·코드 변경
- `31-validation.md`: 수치·회귀 결과
- `40-final-report.md`: 독자용 결론
