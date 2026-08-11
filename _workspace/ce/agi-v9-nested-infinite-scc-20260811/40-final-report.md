# V9 중첩 무한 SCC 최종 보고서

Status: COMPLETE

Gate: PASS — 형식 수학과 격리된 finite unit에 한함. V9 개발·확인·생물학·AGI gate는 BLOCKED/UNTESTED.

완료일: 2026-08-12 (Asia/Seoul)

## 최종 결론

사용자가 수정한 직관, 즉 “뇌를 여러 조각의 SCC로 분절하는 것이 아니라 하나의
무한 SCC를 중첩된 유한 recurrent view로 본다”는 생각은 정확한 graph 객체로
형식화할 수 있다.

그 객체는 한 고정 graph 안에 중첩된 여러 maximal SCC가 아니다. 정확한 정의는

\[
G_0\subsetneq G_1\subsetneq\cdots,\qquad
G_\infty=\bigcup_{n\ge0}G_n,
\]

이며 각 \(G_n\)은 비어 있지 않은 유한 강연결 subgraph다. 이때 \(G_\infty\)는
강연결이고, proper tower이면 countably infinite인 standalone graph 전체가 정확히
하나의 maximal SCC다. 반대로 모든 비어 있지 않은 countable strongly connected
graph는 증가하는 유한 강연결 subgraph 열로 exhaustion할 수 있다. 따라서 graph
이론의 양방향 핵심은 증명되었다.

그러나 이것은 물리적 뇌에 무한히 많은 neuron/agent가 있다는 명제가 아니다.
허용되는 해석은 `finite substrate + finite generator + indefinitely queryable virtual
tower`다. 실제 한 번의 실행은 항상 유한 깊이와 유한 causal cone만 계산한다.

## 증명된 것과 증명되지 않은 것

### 형식적으로 증명됨

1. 증가하는 유한 강연결 graph들의 직접합집합은 강연결이다.
2. countable strong graph와 finite strong exhaustion 사이의 역정리가 성립한다.
3. proper finite exhaustion의 standalone union은 하나의 countably infinite SCC다.
4. 한 고정 graph의 서로 다른 maximal SCC들은 equivalence class이므로 중첩될 수
   없다.
5. 모든 edge가 시간을 엄격히 증가시키는 forward event unroll은 horizon이
   무한이어도 DAG이며 event SCC들은 singleton이다.
6. finite in-degree, one-edge-per-tick locality, complete predecessor enumeration 아래
   유한 query와 유한 horizon의 backward causal cone은 유한하다. 최대 in-degree가
   \(\Delta\), query set이 \(S\), horizon이 \(T\)이면

   \[
   |C_T(S)|\le |S|\sum_{k=0}^{T}\Delta^k.
   \]

### 조건부로 증명됨

level state embedding \(J_n:X_n\hookrightarrow X_{n+1}\)과 update \(F_n\)이

\[
J_nF_n=F_{n+1}J_n
\]

을 정확히 만족해야 representative와 무관한 direct-limit update가 존재한다.
completion에서 하나의 공통 metric과 level-independent \(q<1\)이 있으면 Banach
fixed-point theorem으로 유일한 fixed point와 기하 수렴을 얻는다. finite lift의
uniform one-step defect가 \(\epsilon_n\)이면

\[
d(x_t,y_t)\le q^t d(x_0,y_0)
 +\frac{1-q^t}{1-q}\epsilon_n,
\qquad
d(y_n,x^*)\le\frac{\epsilon_n}{1-q}.
\]

이 구현의 \(q\)는 norm-free 수치가 아니라 모든 활성 level·coordinate를 합친
`global_coordinate_sup` norm의 block-row-sum bound다. Euclidean norm으로 같은
수치를 재해석해서는 안 된다.

### 여전히 미증명·미시험

- 실제 뇌가 이 scale-indexed tower와 embedding/update를 구현한다는 것;
- 임의 completed infinite state를 유한 generator가 exact quotient한다는 것;
- 무한 horizon 또는 fixed-point query가 항상 유한 비용으로 계산된다는 것;
- V9가 V5, V8, ACBSM 또는 matched recurrent control보다 예측에서 우월하다는 것;
- cognition, consciousness 또는 AGI.

## V1~V9 계보 판정

V9는 V1~V8의 실패를 이름만 바꾼 route가 아니다. V8 prospective R1 confirmation은
candidate `0.5377851901`, V5 `0.5400745832`, improvement `0.0022893931`, 95% interval
`[-0.0031918816, 0.0077706678]`로 실패했다. ACBSM은 training-only HOLD였고 선택된
fold에서 rank two가 rank one으로 collapse했다.

따라서 V9가 새 메커니즘으로 승격하려면 적어도 둘 이상의 persistent level state,
실제 reciprocal cross-scale message, state-only readout, same-input/different-history
mediation, level/cross-message lesion 효과, matched-information·matched-compute held-out
우위를 모두 새로 보여야 한다. output gain, analytic posterior, V5/V8/ACBSM output,
cosmetic SCC wrapper는 허용되지 않는다.

현재 V8 locked test `81100..81355`, ACBSM reserved block `82100..82355`, V9 evidence
block은 하나도 열지 않았다. V9 development는 `0/256 BLOCKED`, confirmation은
등록되지 않았다.

## 구현 결과

감사에서 허용한 격리 범위만 다섯 파일로 구현했다.

- `nested_scc_tower.py`: nested finite prefix generator, 독립 SCC audit, ideal
  predecessor rule, finite causal cone, positive-delay unroll, schedule-specific
  contraction와 compatibility certificate;
- `adaptive_scc_tower_controller.py`: finite grow-only \(D_{\max}\) controller,
  state-token-only readout, snapshot continuation, six one-step interventions;
- 두 focused test file과 deterministic non-evidence demo.

default fixture는 `previous_tick_jacobi` schedule에서

\[
q=0.24+0.16+0.14=0.54<0.95
\]

를 `global_coordinate_sup` norm으로 인증한다. state domain은 정확한 closed interval
`[-1,1]`이며 raw observations는 positive frozen reference scale로 먼저
무차원화한다.

중요하게도 default nonzero upward boundary coupling은 append-zero direct-limit
compatibility를 만족하지 않는다. 코드는 이를 완전하게 `REFUSED`하며 witness
defect를 낸다. exact compatibility는 zero-state/zero-input singleton 또는 구조적으로
`upward_gain == 0.0`인 등록 domain에만 발급된다. 그러므로 현 controller는 exact
infinite rollout이나 truncation certificate가 아니라 finite grow-only unit이다.

token은 controller identity, episode generation, tick, active depth, parameter hash,
전체 state와 delay buffer를 묶는다. snapshot HMAC은 같은 process 안의 정확한
continuation만 확인하며 외부 authentication이나 cross-process persistence를 뜻하지
않는다. `LevelReset`, `CutUp`, `CutDown`, `TimeShift`, `SignFlip`, `StateShuffle`은
각각 실제 다음 update tensor를 바꾸고 독립 storage를 사용한다.

## 독립 검증

최종 source SHA-256은 `30-implementation.md`의 다섯 항목과 모두 일치한다.

- focused tests, warnings-as-errors: **162 passed**;
- 별도 schema/contract consistency checks: **579 passed**;
- fixed SCC foundation·V9·dimensionless 관련 결합 suite: **192 passed**;
- Ruff check: PASS;
- Ruff format check: 5 files formatted;
- dimensionless tests: **10 passed**, checker exit `0`;
- deterministic demo: exit `0`;
- CE `gate`와 `build`: PASS.

최종 독립 감사에서 reviewed scope의 추가 P0/P1은 발견되지 않았다. 이전에 기록한
full-repository baseline은 missing ScienceDB/benchmark artifacts와 policy mirror drift
때문에 `2012 passed, 14 skipped, 28 failed, 41 errors`였고, V9/nested-SCC test failure는
없었다. 그 전체 baseline은 최종 lock에서 다시 실행하지 않았으며 관련 결함을 이
unit의 성공으로 재분류하지 않는다.

## 남은 P2와 개발 금지선

현재 unit PASS를 막지는 않지만 V9 development registration 전 다음을 닫아야 한다.

1. `depth_error_tolerance`와 `hysteresis_ticks`는 현재 structural grow-only controller에서
   dormant/reserved다. 제거하거나 실제 controller semantics에 연결해야 한다.
2. `generated_parameter_count`는 active/free coefficient, MAC 또는 matched-capacity
   count가 아니다. 정의와 독립 재계산 전에는 budget 지표로 사용할 수 없다.
3. underscore-prefixed state·delay·diagnostic field는 unit 내부다. evidence runner가
   model input, scorer 정보 또는 hidden control로 노출·소비해서는 안 된다.
4. process-local snapshot을 cross-process provenance나 persistence로 승격할 수 없다.
5. uniform truncation/tail error, matched comparator, seed-role manifest, 모든 개발
   DOF와 구현 hash를 별도 preregistration에서 잠그기 전에는 256-seed run을 시작할
   수 없다.

## 신경과학 해석의 상한

정확한 directed connectome에서 큰 recurrent core가 관찰된 것은 사실이다.
FlyWire v630의 등록된 threshold graph에서는 93.3% giant SCC가 보고되었고
([Lin et al. 2024](https://doi.org/10.1038/s41586-024-07968-y)), C. elegans chemical
graph에서는 `237/279`, chemical+reciprocal-electrical union에서는 `274/279` giant
SCC가 보고되었다
([Varshney et al. 2011](https://doi.org/10.1371/journal.pcbi.1001066)).

이 결과들은 finite brain graph의 광범위한 recurrence를 지지하지만 nested maximal
SCC, infinite direct-limit dynamics, 안정성 또는 V9 causal state를 입증하지 않는다.
Winding의 “nested recurrent architecture”는 hierarchical clustering과 return-loop
census이며 SCC tower가 아니다. BANC의 13 networks도 symmetrized weighted-undirected
spectral communities이지 SCC가 아니다. 따라서 `BRAIN-N1`은 valid mathematical
model, `BRAIN-N2`는 conditional engineering design, `BRAIN-N3`는 untested biological
hypothesis로 남는다.

## 최종 지위와 다음 단계

```text
NESTED GRAPH / EXHAUSTION THEOREMS   PROVED
FIXED-GRAPH NESTED MAXIMAL SCC       IMPOSSIBLE
FORWARD EVENT-UNROLL SCC             SINGLETON / NO-GO
DIRECT-LIMIT DYNAMICS                CONDITIONAL
ISOLATED FINITE UNIT                 COMPLETE / PASS
GENERIC APPEND-ZERO COMPATIBILITY    REFUSED
V9 DEVELOPMENT                       0/256 BLOCKED
V9-1 CAUSAL MECHANISM                UNTESTED
BIOLOGICAL INFINITE-SCC IDENTITY     UNTESTED / LITERAL FORM REJECTED
AGI                                  UNTESTED
```

다음 합법적 단계는 실행이 아니라 별도 development preregistration이다. 새 seed-role
manifest, frozen implementation/config/normalizer/comparator hashes, dormant DOF 정리,
capacity·MAC 정의, matched controls와 lesion floors, schedule-specific certificate를
먼저 고정하고 독립 pre-run audit를 통과해야 한다. 그 전에는 어떤 개발 seed도 열지
않는다.

CE_RUN=_workspace/ce/agi-v9-nested-infinite-scc-20260811
