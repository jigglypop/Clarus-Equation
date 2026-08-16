# 물리 독립 AGI 코어 V0: 대안 형식 경로

Status: COMPLETE

## 1. 실행 이유와 목표

`11-math.md`가 PIC-I1--PIC-I4의 현재 판정조건에 완전한 반례를 확인했으므로 routes를 실행한다. 목표는 AGI, 성능 또는 의식을 증명하는 것이 아니라 다음 좁은 목표를 만족하는 구현 경계를 찾는 것이다.

> 두 비동형 procedural rule family를 동일한 환경 비특정 orchestrator로 실행하고, evaluator truth noninterference, invalid-permit 무전이, full-state deterministic public ledger와 반환된 session의 linear-history single-use를 동시에 보장할 수 있는가.

수정 계약은 과거 immutable session snapshot에서 새 branch를 만드는 rollback/fork까지 V0가 차단한다고 주장하지 않는다. 이 강화 요구는 순수 reducer와 양립하지 않으며, 숨은 mutable global registry로 우회하면 full-state replay가 깨진다. 아래 route 비교는 이 no-go를 반영한다.

세 후보는 모두 이 반례들을 본 뒤 만든 `target-aware` 경로다. 비교한 후보 수는 3이므로 architecture-level look-elsewhere 규모는 3이다. 수치 결과의 선택은 없고 numerical parameter dof는 모두 0이다. 후보마다 새 공리는 정확히 하나다.

## 2. 강제 부분과 자유 부분

계약이 강제하는 것은 immutable learner records, `WorldAdapter` 경계, truth 비노출, permit을 거친 agent action, deterministic reference ledger, 두 구조적으로 다른 transition family다. 자유 부분은 world state를 mutable하게 보관할지, permit authority를 process 안의 seal로 둘지, evaluator를 process 밖으로 분리할지, reset을 genesis event와 lifecycle permit 중 무엇으로 나타낼지다.

다음 경로는 후보에서 제외했다.

- nominal frozen dataclass만 permit으로 쓰는 경로: 직접 생성과 `replace()` 반례로 사망;
- field 이름에서 `truth`만 금지하는 경로: `metadata` alias와 closure 반례로 사망;
- 현재 observation·seed·config만 replay key로 쓰는 경로: 다른 memory state 반례로 사망.

## 3. 후보 비교

| 순위 | 경로 | 새 공리 1개 | numerical dof | 구조 선택 수 | target-aware | 단계별 지위 | 교차 예측 | 죽이는 반증 시험 |
|---:|---|---|---:|---:|---|---|---|---|
| 1 | R1 pure event-sourced core | 모든 post-genesis world/agent 변화는 명시적 이전 session과 authorized event의 pure function이다 | 0 | 4 | 예 | 형식 구성 가능, 제품 판정 없음 | full-state replay byte equality; descendant replay 거부; 동일 old snapshot fork의 동일 child; 두 family 공통 reducer | 숨은 global nonce/time/RNG에 따라 ledger가 달라지거나 raw action이 reducer에 들어가면 기각 |
| 2 | R2 sealed mutable adapter | executor만 mutable adapter reference를 소유한다 | 0 | 5 | 예 | 조건부 구성 가능, ownership test 필요 | forged/stale/next-session replay/cross-world permit 모두 transition count 0; valid permit만 1 증가 | planner가 alias·callback·reflection으로 world를 바꾸거나 adapter가 permit 없이 advance하면 기각 |
| 3 | R3 process-isolated evaluator/world | learner와 evaluator/world는 canonical message protocol 외 공유 상태가 없다 | 0 | 5 | 예 | 형식 경계 강함, V0 최소 범위보다 큼 | poison truth와 shared-memory 공격 불가; same visible transcript decision equality | pickle/arbitrary object, shared file/global, nondeterministic scheduling이 learner 결과에 들어가면 기각 |

구조 선택 수는 후보 안에서 고정해야 하는 비수치 설계 선택의 개수다. R1은 canonical value grammar, genesis event, reducer state, ledger projection의 네 선택이다. R2는 permit binding set, authentication, nonce lifecycle, reference ownership, ledger projection의 다섯 선택이다. R3는 process roles, message schemas, scheduler, seed transport, ledger merge의 다섯 선택이다. 이 수는 model parameter count나 통계 dof가 아니다.

## 4. R1: pure event-sourced core

### 4.1 경로

world의 mutable object를 외부에 두지 않고 `used_nonces`까지 포함한 session $x_t=(w_t,U_t,t,\ldots)$ 자체를 immutable record로 둔다. reset은 숨은 mutation이 아니라 최초 `Genesis` event다. 그 뒤의 유일한 transition은 다음 reducer다.

$$
(x_{t+1},o_{t+1})
=
\mathcal R_F(x_t,\operatorname{Verified}_{x_t}(\pi_t)),
\qquad
U_{t+1}=U_t\cup\{\nu_t\}.
\tag{R1.1}
$$

invalid permit은 `Verified` 값을 만들지 못하므로 family reducer가 호출되지 않는다. permit은 episode, tick, world, session, action space, policy와 proposal commitment에 결합한다. family $F$는 evaluator-owned reducer 구현을 선택하지만 orchestrator는 공통 protocol만 본다. public ledger에는 event의 canonical projection과 이전 digest를 넣고 authority secret/tag는 넣지 않는다.

### 4.2 linear-history 정리와 전역 anti-rollback no-go

$\nu_t\in U_{t+1}$이고 모든 후속 session에서 $U$가 단조 증가하면, 반환된 $x_{t+1}$의 모든 descendant에 같은 permit을 다시 제출했을 때 `unused` 검사가 실패한다. 이것이 V0가 보장할 수 있는 single-use다.

반면 valid transition $\mathcal R_F(x,\pi)=x'\ne x$가 있을 때, 동일한 과거 snapshot $x$를 다시 입력한 두 번째 호출만 거부하려면 같은 pure function에 $\mathcal R_F(x,\pi)=x$도 요구해야 한다. 같은 입력의 출력은 하나뿐이므로 불가능하다. 따라서 old snapshot fork는 다음 deterministic branch로 허용된다.

$$
\mathcal R_F(x,\pi)=x'
\quad\Longrightarrow\quad
\mathcal R_F(x,\pi)=x'
\text{ on every reevaluation of the same explicit inputs}.
\tag{R1.2}
$$

숨은 process-global nonce registry $G$를 읽으면 실제 함수는 $\mathcal R_F(x,\pi,G)$가 된다. $G$를 replay state에서 제외한 채 첫 실행과 재실행의 값을 바꾸는 것은 R1과 PIC-I4의 반증이다. $G$를 명시적 immutable session에 넣으면 다시 linear-history 정리만 얻는다. process 전역 anti-rollback은 append-only durable executor나 monotonic custody service를 새 공리와 replay input으로 추가하는 후속 route다.

### 4.3 장점과 비용

이 경로는 reset 반례를 genesis로 분리하고, world의 숨은 background mutation을 구조적으로 없애며, deterministic ledger 귀납의 전제를 가장 직접적으로 만족한다. approved `procedural_world.py` 안에 두 reducer와 generic adapter를 둘 수 있어 새 파일이 필요하지 않다.

제약은 future multi-agent 또는 asynchronous world가 들어오면 모든 autonomous event도 명시적으로 event log에 넣어야 한다는 점이다. 또한 호출자가 보관한 old snapshot의 fork를 막지 않는다. 둘 중 하나를 숨은 process state로 처리하면 R1 공리가 깨진다.

### 4.4 교차 예측과 반증

- 두 비동형 family의 transition image-cardinality invariant가 달라도 같은 orchestrator source bytes가 실행된다.
- 같은 genesis, full agent state, event sequence, RNG state에서 ledger가 byte-identical하다.
- valid permit을 반환된 다음 session에 다시 내면 state와 transition count가 불변이다.
- 같은 old snapshot과 permit을 두 번 독립 평가하면 두 child session과 ledger가 byte-identical하다. 둘째 평가만 거부되면 숨은 상태가 있다는 뜻이다.
- invalid permit 전후 world record와 ledger tail이 같다. 단, rejection audit을 ledger에 남기기로 했다면 world record는 같고 rejection entry만 결정적으로 추가돼야 한다.
- reducer가 wall clock, global RNG, mutable module global nonce registry 또는 object address를 읽는 즉시 이 경로는 기각된다.

## 5. R2: sealed mutable adapter

### 5.1 경로

contract의 객체지향 형태를 유지하되 mutable adapter reference를 executor 안에 봉인한다. permit은 `11-math.md` 식 (5)의 episode, tick, world, session, action-space, policy, proposal, nonce와 authentication tag를 모두 가진다. verifier가 성공한 뒤에만 private adapter method를 호출한다.

$$
\operatorname{execute}(\pi)=
\begin{cases}
\operatorname{adapter\_advance}(a),&J(\pi)=1,\\
\operatorname{NoTransition},&J(\pi)=0.
\end{cases}
\tag{R2.1}
$$

reset은 일반 action theorem 밖의 lifecycle operation으로 좁힌다. 모든 agent-initiated post-reset transition만 permit theorem의 정의역에 둔다.

### 5.2 장점과 비용

기존 `WorldAdapter`, `ActionExecutor`, `SafetyKernel` 언어와 가장 가깝다. 반면 Python에서는 nominal visibility가 보안경계가 아니므로 생성자에서 raw adapter를 planner/model/memory에 전달하지 않았다는 object-graph 감사와 adversarial alias test가 필요하다. shared callback, global registry 또는 bound method가 새는 순간 exclusive-ownership 공리가 거짓이다.

runtime security nonce/tag가 public ledger replay를 깨지 않도록 ledger는 permit claim digest와 verification result만 기록한다. tag 자체를 기록하려면 deterministic development authority key와 exact key provenance를 별도로 고정해야 한다.

R2의 mutable executor가 process 전역 nonce registry를 소유하면 old snapshot fork를 같은 executor lifetime 안에서는 막을 수 있다. 그러나 그 registry를 full initial state와 replay input에서 숨기는 순간 PIC-I4가 성립하지 않는다. durable anti-rollback route로 승격하려면 registry snapshot, 복구, append-only성, crash consistency와 custody identity를 모두 외부 신뢰 공리로 열어야 하므로 V0 R1의 대체 구현으로 몰래 섞을 수 없다.

### 5.3 교차 예측과 반증

- forged action, wrong episode, stale tick, wrong adapter/session/policy/proposal와 반환된 다음 state의 replay permit은 모두 private transition call count를 바꾸지 않는다.
- valid permit 한 번만 count를 정확히 1 증가시킨다.
- planner가 받은 객체의 recursive reachability graph에 adapter 또는 bound mutation method가 있으면 R2는 기각된다.
- reset이나 autonomous step을 PIC-I2의 post-reset agent-action 결론에 몰래 포함하면 기각된다.

## 6. R3: process-isolated evaluator/world

### 6.1 경로

world/evaluator와 learner/orchestrator를 별도 process로 분리한다. 통신은 식 (1)의 canonical immutable grammar를 따르는 메시지만 허용한다. learner process에는 transition table, family id, evaluator score 또는 world object가 존재하지 않는다. executor process가 signed permit을 검증하고 world process에 전달한다.

$$
\mathrm{Learner}
\xleftrightarrow[\text{proposal}]{\text{observation}}
\mathrm{Executor}
\xleftrightarrow[\text{world step}]{\text{verified permit}}
\mathrm{World/Evaluator}.
\tag{R3.1}
$$

### 6.2 장점과 비용

closure, shared dict와 module global을 통한 truth 누수를 가장 강하게 줄인다. 그러나 process lifecycle, IPC ordering, timeout, failure recovery와 ledger merge라는 새 표면이 생기고 승인된 V0 여섯 파일보다 구현량이 커진다. 따라서 V0의 1순위 경로는 아니다.

### 6.3 교차 예측과 반증

- evaluator truth를 access 시 예외를 던지는 poison object로 바꿔도 learner process는 영향을 받지 않는다.
- 같은 canonical message transcript와 learner state에서 proposal bytes가 같다.
- arbitrary pickle, shared filesystem state, environment variable, shared memory 또는 nondeterministic message ordering이 learner decision에 영향을 주면 기각된다.

## 7. 두-family fixture의 공통 요구

세 경로 모두 다음 exact fixture를 사용할 수 있다.

$$
T_{\oplus}(s,a)=s\oplus a,
\qquad
T_{\rm set}(s,a)=a,
\qquad S=A=\{0,1\}.
\tag{R4.1}
$$

두 family의 action-map image cardinality multiset은 $(2,2)$와 $(1,1)$이므로 state/action relabel 아래에도 다르다. 이 invariant를 manifest에 고정하면 seed나 표면 symbol 변경을 새 family로 세는 것을 막을 수 있다.

그러나 두 fixture를 실행하는 것만으로 universal adapter independence가 나오지 않는다. 최소 kill test는 다음 conjunction이다.

1. orchestrator가 procedural family module을 import하지 않는다.
2. concrete adapter type, class name, family id와 숨은 transition callback에 접근하지 않는다.
3. 두 adapter를 동일 opaque forwarding wrapper로 감싸도 동작한다.
4. hidden truth만 바꾸고 visible history를 맞춘 prefix에서는 proposal과 learner ledger가 같다.
5. adapter의 extra attribute가 읽히면 예외를 내는 poison proxy를 통과한다.
6. valid permit은 반환된 다음 session의 `used_nonces` 때문에 다시 실행되지 않는다.
7. 같은 old session snapshot과 permit의 독립 재평가는 같은 child bytes를 만들며 전역 거부를 주장하지 않는다.
8. hidden mutable nonce registry를 읽거나 쓰면 실패하고, public ledger에는 authentication secret/tag가 없다.

## 8. 열린 가설의 처리

PIC-H1은 V1의 preregistered unseen-family 비교 없이는 어떤 route에서도 올라가지 않는다. R1--R3은 scaffold의 권한·정보·재현 경로일 뿐 planner 성능 경로가 아니다. PIC-H2--PIC-H5의 metric, SCC, language, self-hypothesis는 V0 구현 범위 밖이므로 이 routes에서 후보를 만들지 않았다. 그 미구현은 R1--R3의 승패로 해소되지 않는다.

## 9. 최종 route 판정

R1이 numerical dof 0, 새 공리 1개와 가장 작은 authority surface로 최초 반례들을 피하므로 1순위다. 그 대가로 single-use 정리는 반환된 session의 linear history에만 적용되고 deterministic old-snapshot fork는 허용한다. R2는 contract의 객체 구조를 가장 덜 바꾸지만 exclusive-reference 공리를 Python에서 동적 test로 보강해야 하며, durable registry를 추가하면 별도 external-custody route가 되므로 2순위다. R3은 정보경계가 강하지만 V0 구현량을 넘으므로 후속 confirmation/evaluator 격리 후보로 둔다.

이번 수정 수학 레인은 제품 코드나 test를 읽거나 실행하지 않았다. 따라서 이 순위는 형식 설계 비교이며 PIC-I1--PIC-I5 구현 통과, 성능, AGI 또는 물리 독립의 보편 정리를 뜻하지 않는다.
