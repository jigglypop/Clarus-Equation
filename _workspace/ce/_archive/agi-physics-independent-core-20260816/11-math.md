# 물리 독립 AGI 코어 V0: 독립 수학·형식 검산

Status: COMPLETE

## 1. 대상과 선행 경계

검산 대상은 `00-contract.md`의 PIC-I1--PIC-I5, 타입 불변식, learner/evaluator 정보 경계, permit-only transition, deterministic ledger, 두 rule family에 대한 adapter 교체 가능성이다. 선행 `_workspace/ce/agi-causal-recurrent-geometry-phase-a-20260816/40-final-report.md`가 이미 제한한 known-identity synthetic development 식별성, unknown-mix no-go와 confirmation 부재는 재유도하지 않았다. 특히 선행 결과는 hidden context, partial observation, 기억, AGI 또는 이 계약의 scaffold를 증명하지 않는다.

이 레인의 `COMPLETE`는 검산이 끝났다는 뜻이다. 구현 승인이나 형식 승격을 선언하지 않는다. 이번 수정 검산은 제품 코드와 제품 test를 읽거나 실행하지 않으므로 PIC-I1--PIC-I5의 실제 구현 통과도 판정하지 않는다.

## 2. 요약 판정

계약의 좁은 코어는 구성 가능하다. 최초 검산은 당시 판정 문구에서 다음 네 개의 완전한 논리 반례를 찾았다.

1. 같은 orchestrator를 두 adapter에 사용했다는 사실만으로 구체 환경 의존성이 제거되지는 않는다. orchestrator가 adapter의 concrete type이나 family id를 검사할 수 있다.
2. 모든 외부 변경이 permit을 요구한다는 문자 그대로의 문장은 `reset`이 permit 없이 world state를 바꾸는 즉시 반례를 갖는다.
3. learner-facing 객체에 `truth`라는 이름의 field가 없다는 사실은 truth alias, callback 또는 closure 누수를 막지 못한다.
4. 같은 현재 관측·seed·구성은 서로 다른 memory/model state를 가진 stateful agent의 ledger를 결정하지 못한다.

업데이트된 `00-contract.md`는 protocol-only orchestrator, lifecycle 경계, 동적 noninterference, full-state replay domain으로 이 네 반례의 부모 문구를 좁혔다. PIC-I5도 authenticated permit과 immutable `WorldSession.used_nonces`를 채택하면서 single-use의 범위를 **반환된 다음 session을 잇는 동일 linear history**로 제한했다.

이번 수정 검산의 새 결론은 다음과 같다.

1. 명시적 session state에 nonce 소비 원장을 넣으면 반환된 후속 session 계보에서 permit 재사용 거부를 조건부 정리로 증명할 수 있다.
2. 동일한 과거 immutable session snapshot을 호출자가 다시 넣어 별도 branch를 만드는 것까지 순수 deterministic reducer가 전역 차단할 수는 없다. 같은 입력에 첫 호출과 둘째 호출이 다른 출력을 내야 하므로 함수의 외연성과 모순이다.
3. 숨은 mutable global nonce registry는 이 모순을 해결하지 않는다. registry 값을 full replay state에서 숨기므로 같은 명시적 초기상태와 입력이 서로 다른 ledger를 만들 수 있어 R1과 PIC-I4를 깨뜨린다.
4. 전역 anti-rollback이 필요하면 append-only durable executor, monotonic counter 또는 custody service처럼 session 밖의 신뢰 상태가 필요하다. 이것은 V0의 좁은 정리 밖이다.

기존 독립 verifier는 최초 다섯 반례와 네 구성 증인을 정확 유한모형으로 재현했다.

```text
python _workspace/ce/agi-physics-independent-core-20260816/artifacts/verify_core_contract.py
```

출력의 모든 값은 `true`였고 외부 패키지나 제품 모듈을 import하지 않았다.

## 3. 형식 대상

### 3.1 레코드 값의 허용 정의역

`frozen` dataclass만으로는 불변성이 성립하지 않는다. field 안의 list, dict 또는 array는 계속 변할 수 있다. 따라서 learner-visible record의 값은 다음 재귀 문법으로 제한해야 한다.

$$
v ::= \mathrm{null}
\mid \mathrm{bool}
\mid \mathrm{int64}
\mid \mathrm{finite\_float64}
\mid \mathrm{NFC\_string}
\mid (v_1,\ldots,v_n)
\mid ((k_1,v_1),\ldots,(k_n,v_n)).
\tag{1}
$$

마지막 경우의 key는 NFC-normalized nonempty string이고 중복이 없으며 bytewise 정렬한다. `bool`은 `int`로 받지 않고, NaN과 Inf를 거부하고, signed zero를 하나의 canonical zero로 정규화해야 한다. 외부 list/dict/array를 받는다면 생성 시 깊은 복사 뒤 식 (1)의 tuple 표현으로 바꾼다. callback, generator, file handle, module, arbitrary object reference와 mutable view는 허용하지 않는다.

이 문법과 field별 exact schema가 고정되면 구조적 귀납으로 canonical serialization의 유일성이 따른다. 반대로 verifier의 `ShallowFrozen([1])`은 frozen object 내부 list가 `[1,2]`로 바뀌므로 `frozen=True`만으로는 계약의 불변 레코드가 아니다.

### 3.2 world와 learner view

world의 evaluator-owned 상태를 $w_t=(s_t,g)$로 쓴다. $s_t$는 전이 가능한 내부 상태이고 $g$는 transition table, hidden family label, 목표 판정용 truth 같은 evaluator 전용 자료다. learner가 볼 수 있는 것은 명시적 projection이다.

$$
o_t=V(s_t,g),
\qquad
h_t=(o_0,a_0,o_1,\ldots,a_{t-1},o_t).
\tag{2}
$$

learner state와 명시적 RNG state를 각각 $m_t,r_t$라 하면 허용되는 결정은 다음 형식이다.

$$
(a_t,m_{t+1})=L(h_t,m_t,r_t,C),
\tag{3}
$$

여기서 $C$는 공개 configuration이다. $L$의 인자, closure, global, shared mutable alias와 반환 객체 어디에도 $g$로 가는 참조가 없어야 한다.

정보경계의 동적 필요조건은 이름 감사가 아니라 다음 noninterference다.

$$
h_t(w)=h_t(w'),\quad m_t=m'_t,\quad r_t=r'_t
\Longrightarrow
(a_t,\ell_t)=(a'_t,\ell'_t),
\tag{4}
$$

여기서 $\ell_t$는 learner가 만든 ledger prefix다. 결정론적 reference implementation에는 등식을 쓰고, 확률적 구현에는 action/ledger 분포의 동등성을 써야 한다.

### 3.3 permit 검증 관계

최소 permit은 다음 자료에 결합돼야 한다.

$$
\pi=(
\mathrm{schema},e,t,h_W,h_X,h_A,h_P,h_Q,a,\nu,\tau
),
\tag{5}
$$

$e$는 episode id, $t$는 tick, $h_W$는 adapter/world instance commitment, $h_X$는 permit 발행 시점의 immutable `WorldSession` commitment, $h_A$는 action-space commitment, $h_P$는 immutable safety-policy commitment, $h_Q$는 proposal commitment, $a$는 canonical action, $\nu$는 nonce, $\tau$는 issuer authentication tag다. verifier $J$는 다음 conjunction을 모두 검사해야 한다.

$$
J(\pi)=J_{\rm auth}\land J_{\rm episode}\land J_{\rm tick}
\land J_{\rm world}\land J_{\rm session}\land J_{\rm action}\land J_{\rm policy}
\land J_{\rm proposal}\land J_{\rm unused}\land J_{\rm live}.
\tag{6}
$$

순수 R1에서 전체 world session을 $x_t=(w_t,U_t,t,\ldots)$로 쓰고, $U_t$를 canonical immutable `used_nonces` 집합으로 둔다. 허용 전이는 명시적 상태만 받는 함수 $\Phi$로 다음과 같이 정의할 수 있다.

$$
\Phi(x_t,\pi)=
\begin{cases}
x_{t+1}=(T(w_t,a),U_t\cup\{\nu\},t+1,\ldots),&J(x_t,\pi)=1,\\
x_t,&J(x_t,\pi)=0.
\end{cases}
\tag{7}
$$

식 (7)에서는 invalid permit이 world transition을 일으키지 않는다는 결론이 정의에 의해 따른다. 그러나 Python의 nominal type이나 frozen dataclass는 $J_{\rm auth}$와 $J_{\rm unused}$를 주지 않는다. verifier에서 `NaivePermit("forbidden", True)`를 직접 만들고 `replace()`로 action을 바꿀 수 있었다. 반면 HMAC, episode/tick/world/policy binding과 nonce 소비를 둔 유한 구성은 forged, wrong-episode와 **반환된 다음 상태에서의 replay**를 모두 거부했고 transition count는 유효 permit 한 번에서만 1이 됐다.

### 3.4 linear-history single-use 정리와 snapshot-fork no-go

**정리 1: 후속 session 계보의 single-use.** 식 (7)의 valid branch가 $\nu\in U_{t+1}$을 보존하고 모든 후속 전이가 $U_{k+1}\supseteq U_k$를 만족한다고 하자. $x_{t+1}$에서 시작하는 모든 descendant $x_k$, $k\ge t+1$에 같은 $\pi$를 제출하면 $J_{\rm unused}=0$이므로 world transition은 없다.

**증명.** $\nu\in U_{t+1}$이다. 단조성에 대한 귀납으로 모든 $k\ge t+1$에 $\nu\in U_k$다. 따라서 $J_{\rm unused}(x_k,\pi)=0$이고 conjunction 식 (6)이 거짓이어서 식 (7)의 invalid branch를 탄다. 이 정리는 호출자가 reducer가 반환한 session을 다음 입력으로 쓰는 linear history에 한정된다.

**정리 2: 순수 reducer의 전역 snapshot-fork 차단 불가능성.** 유효 permit $\pi$와 session $x$가 있어 $\Phi(x,\pi)=x'\ne x$라고 하자. 과거 snapshot $x$를 다시 입력하는 두 번째 호출을 전역 replay라는 이유로 거부하려면 같은 식이 $\Phi(x,\pi)=x$도 만족해야 한다. deterministic function은 같은 ordered pair $(x,\pi)$에 하나의 값만 대응시키므로 이는 $x'\ne x$와 모순이다.

가장 작은 정확 반례는 $X=\{0,1\}$, permit 집합 $\{p\}$, $\Phi(0,p)=1$이다. 첫 호출의 허용은 출력 $1$을 강제한다. 같은 immutable snapshot $0$과 같은 $p$의 두 번째 호출만 출력 $0$으로 바꾸려면 호출 이력이라는 추가 입력이 필요하다. 그 이력은 더 이상 순수 함수의 명시적 입력 밖에 있을 수 없다.

**따름정리.** 숨은 mutable registry $G$로 $\nu\notin G$일 때 허용하고 $\nu\in G$일 때 거부하면 계산은 실제로 $\Phi(x,\pi,G)$다. $G$를 `WorldSession`이나 식 (8)의 replay state에 넣지 않으면 동일한 명시적 $x,\pi$에서 결과가 달라져 R1과 PIC-I4의 전제를 위반한다. $G$를 명시적 immutable session state에 넣으면 식 (7)로 돌아오며 과거 $x$의 독립 branch 자체는 여전히 허용된다. 전역 anti-rollback에는 V0 밖의 durable custody가 필요하다.

### 3.5 deterministic ledger

완전한 초기 실행상태를 다음과 같이 둔다.

$$
z_0=(w_0,m_0,q_0,U_0,r_0,C,B),
\tag{8}
$$

$q_0$는 planner/model state, $U_0$는 `WorldSession` 안의 명시적 permit-consumption state, $r_0$는 모든 domain-separated RNG state, $B$는 실행한 code·schema·dependency commitment다. 각 tick 함수가 결정적이면 ledger entry $E_t$와 hash chain은 다음으로 고정할 수 있다.

$$
d_{-1}=0^{256},
\qquad
d_t=\operatorname{SHA256}(
\mathrm{schema}\Vert d_{t-1}\Vert\operatorname{Canon}(E_t)
).
\tag{9}
$$

**조건부 정리.** 두 실행의 $z_0$, learner-visible input history와 tick 함수가 같고, tick 함수가 time, global RNG, process-randomized iteration, address repr, concurrency race 또는 외부 mutable state를 읽지 않으면 모든 $E_t,d_t$가 같다.

**검산.** 초기값이 같을 때 $t=0$의 결정 함수 입력이 같으므로 $E_0$가 같고 식 (9)에서 $d_0$가 같다. $t=k$까지 같다고 가정하면 다음 내부상태와 입력이 같아 $E_{k+1}$가 같고 식 (9)로 $d_{k+1}$도 같다. 유한 horizon에 대한 귀납이다.

최초 PIC-I4의 `같은 관측·seed·구성` 문구는 식 (8)의 $m_0,q_0,U_0,r_0,B$와 과거 history를 생략했다. verifier에서 같은 observation, seed, config에 memory `(0,)`와 `(1,)`만 달리했더니 각각 `left`, `right`가 선택되어 canonical ledger가 달라졌다. 업데이트된 PIC-I4는 전체 초기상태와 history를 포함하므로 이 역사적 반례를 배제한다.

runtime-only permit authentication nonce나 session secret을 ledger에 그대로 직렬화하면 기능 trace가 같아도 byte replay가 깨질 수 있다. 해결하려면 (a) deterministic development authority envelope을 쓰거나, (b) 공개 ledger에는 secret tag가 아니라 permit claim digest와 검증 결과만 투영해야 한다. 어느 방식을 쓰는지 계약과 test가 고정해야 한다.

nonce 소비 여부를 숨은 process-global registry에서 읽으면 그 registry는 식 (8)에 빠진 상태가 된다. 예를 들어 $G_0=\varnothing$인 실행은 permit을 수용하고 $G_0=\{\nu\}$인 실행은 거부하므로, 나머지 $z_0$와 입력이 같아도 첫 ledger entry가 다르다. 따라서 hidden registry는 deterministic replay 정리의 결론을 직접 반증한다. 반대로 $U_0$를 `WorldSession`에 포함하면 각 branch는 재현 가능하고, 같은 과거 snapshot에서 갈라진 두 branch가 같은 child ledger를 만드는 것도 순수성의 예상 결과다.

## 4. 두 rule family의 구성 증인

상태와 행동을 각각 $S=A=\{0,1\}$로 두고 두 transition을 정의한다.

$$
T_{\oplus}(s,a)=s\oplus a,
\qquad
T_{\rm set}(s,a)=a.
\tag{10}
$$

각 고정 action이 만드는 state-map image cardinality의 multiset은 다음과 같다.

$$
I(T_{\oplus})=(2,2),
\qquad
I(T_{\rm set})=(1,1).
\tag{11}
$$

state와 action symbol을 전단사로 재명명해도 map의 image cardinality는 보존된다. 따라서 두 transition은 seed 변경이나 표면 symbol relabel로 서로 바뀌지 않는 비동형 family다. 초기상태 0과 action sequence $(1,0,1)$에 같은 `step(action)` orchestrator를 적용하면 trace는 각각 다음이다.

$$
(1,1,0),
\qquad
(1,0,1).
\tag{12}
$$

따라서 두 비동형 transition family를 하나의 adapter-shaped orchestrator로 실행하는 **존재 증인**은 있다. 이것이 모든 adapter에 대한 독립성을 증명하지는 않는다. 다음 orchestrator도 두 family에 대해 같은 코드 객체를 사용하지만 contract의 물리 독립 의도에는 어긋난다.

```python
if type(adapter).__name__ == "XorFamily":
    return xor_specific_path(adapter)
return set_specific_path(adapter)
```

즉 `두 family + 같은 orchestrator`는 PIC-I1의 필요 test witness일 수 있으나 충분조건이 아니다. concrete module import, `isinstance`/`type`/family-id dispatch, adapter-specific `hasattr`, hidden transition callback 접근이 없다는 parametricity/import boundary가 추가돼야 한다.

## 5. claim별 판정

| ID | 계약 지위 | 독립 판정 | P 등급 | 근거와 필요한 최소 수정 |
|---|---|---|---|---|
| PIC-I1 | [산출 후보] | 두 비동형 family 공통 API의 존재 증인은 식 (10)--(12)로 가능. 업데이트된 protocol-only·동일-source 조건은 기존 concrete-dispatch 반례를 배제할 수 있음 | P1 | opaque proxy, import/AST 경계와 실제 두-family 실행은 제품 focused test가 필요 |
| PIC-I2 | [산출 후보] | genesis/reset을 lifecycle로 분리한 post-genesis agent transition에는 식 (7)의 조건부 불변식이 가능 | P1 | valid verified event만 pure reducer에 도달하는지 제품 kill test가 필요 |
| PIC-I3 | [산출 후보] | 이름 감사 대신 식 (4)의 동적 noninterference로 좁힌 현재 문구는 구성 가능 | P1 | hidden truth paired-world proposal·learner-ledger byte equality를 제품에서 확인해야 함 |
| PIC-I4 | [산출 후보] | 식 (8)의 전체 초기상태와 history를 포함한 현재 문구에는 식 (9)의 조건부 귀납 정리가 적용됨 | P1 | 숨은 registry·time·global RNG 부재와 canonical hash-chain replay를 제품에서 확인해야 함 |
| PIC-I5 | [산출 후보] | 식 (5)--(7) 아래 반환된 후속 session 계보의 single-use는 정리 1로 성립. 과거 snapshot fork의 전역 차단은 정리 2로 불가능하며 계약이 이를 주장하지 않음 | P1 | session-bound auth, nonce 단조성, adversarial proposal과 next-session replay kill test가 필요; global anti-rollback은 범위 밖 |
| PIC-H1 | [미완성] | V0 scaffold에서 성능 결론 없음 | P1 | V1 preregistered unseen-family 비교, matched baseline, fresh unit-of-analysis 필요 |
| PIC-H2 | [미완성] | 범위 밖 유지 | P1 | metric 구현·대조군·kill test 없음 |
| PIC-H3 | [미완성] | 범위 밖 유지 | P1 | SCC adapter·압축 대조군·kill test 없음 |
| PIC-H4 | [미완성] | 범위 밖 유지 | P1 | DSL·의미 intervention·negative control 없음 |
| PIC-H5 | [미완성] | 범위 밖 유지 | P1 | `SelfHypothesis` 행동평가와 identity 분리 test 없음 |

PIC-I1--PIC-I5는 모두 `[산출 후보]`이고 위 표의 P1은 구현 certificate가 아직 이 수정 수학 레인에서 확인되지 않았다는 뜻이다. 수학적 구성 가능성과 제품 test 통과는 서로 다른 판정이다.

## 6. 숨은 공리와 자유도

| 항목 | 고정해야 하는 형식 선택 | 없을 때 생기는 문제 |
|---|---|---|
| recursive immutability | 허용 scalar/container grammar, deep copy, float normalization | frozen 내부 mutable alias로 과거 ledger byte가 사후 변경됨 |
| execution authority | executor의 mutable world reference 독점 | planner나 shared callback이 permit 없이 world를 변경 |
| lifecycle boundary | reset, autonomous tick, evaluator intervention의 지위 | PIC-I2의 전칭이 즉시 거짓 |
| permit binding | episode, tick, world, action space, policy, proposal, nonce, terminal state | cross-world substitution, stale/replay, policy swap |
| replay scope | `WorldSession.used_nonces`가 반환된 descendant에 단조 누적되는 linear history | old snapshot fork까지 막는다는 거짓 전역 single-use 주장 |
| anti-rollback custody | V0에는 없음; 필요하면 명시적 durable external state | hidden global registry로 순수성·replay를 깨뜨림 |
| information projection | evaluator truth에서 observation으로 가는 유일 함수 $V$ | metadata, closure, global 또는 seed channel 누수 |
| deterministic state | full initial state, history, code/schema hash, 모든 RNG role | PIC-I4 replay 조건 부족 |
| ledger projection | secret/runtime authority 자료를 canonical ledger에 넣는지 | secure random nonce와 byte replay가 충돌 |
| family inequivalence | seed/relabel과 구조적 family 차이 판정 | 같은 transition의 표면 변형을 두 family로 오인 |

`PrincipalIdentity`, `SelfHypothesis`, `Constitution`을 nominally 다른 class로 만드는 것만으로는 내부 mutable object의 alias를 막지 않는다. V0에서는 placeholder type separation만 검사하고 의미적 독립성 주장은 후속으로 남겨야 한다.

## 7. 구현 kill test

| 경계 | 반드시 죽여야 하는 잘못된 구현 | 최소 test |
|---|---|---|
| 타입 불변성 | frozen record가 list/dict/array alias를 보존 | 입력 container를 record 생성 후 변조하고 bytes/hash 불변 확인 또는 생성 시 거부 확인 |
| exact scalar type | `True`가 tick `1`로 수용됨 | bool, Decimal, numeric string, custom numeric 거부 |
| 정보경계 | truth가 `metadata` 또는 closure로 전달 | truth field 이름 없는 alias fixture와 poison truth object 접근 거부 |
| noninterference | 같은 visible history인데 hidden family id로 action이 갈림 | hidden truth만 다른 paired world의 proposal/ledger prefix byte equality |
| adapter 독립 | concrete type 또는 family id branch | 동일 opaque proxy와 class-name permutation에서 trace behavior 유지; orchestrator import/AST boundary 확인 |
| direct execution | planner가 adapter/world reference를 보유 | adversarial planner의 mutation method 호출 불가능 및 world transition count 0 |
| permit 위조 | frozen dataclass를 직접 만들거나 `replace`해 허용 | forged action/tag/policy/world/proposal 모두 거부 |
| permit replay | 반환된 다음 `WorldSession`에 같은 permit 재제출을 수용 | 첫 실행만 transition count 증가하고 descendant `used_nonces`에서 거부 |
| snapshot fork 의미 혼동 | 같은 과거 immutable session snapshot의 새 branch까지 전역 거부한다고 주장 | 동일 snapshot·permit 두 호출은 byte-identical child를 만들고, 이 동작을 replay 실패로 오판하지 않음 |
| hidden nonce registry | 명시적 session은 같은데 process-global 소비 이력에 따라 결과가 달라짐 | fresh explicit state에서 global registry 값을 바꿔도 결과가 달라지지 않거나 global 접근 즉시 실패 |
| stale/cross-world | 과거 tick·다른 episode/adapter permit 수용 | 모든 조합 거부, world byte snapshot 동일 |
| lifecycle | reset을 일반 action theorem에 몰래 포함 | genesis/reset event를 별도 감사하고 post-reset theorem 시작 tick 고정 |
| deterministic replay | global RNG, time, UUID, set iteration, address repr 사용 | 각 source를 monkeypatch해도 full-state replay bytes 동일하거나 source 접근 즉시 실패 |
| hash chain | 중간 entry 변조가 tail에 반영되지 않음 | 한 byte 변조 후 해당 entry 이후 digest 전부 변경 |
| authority/ledger 분리 | random secret nonce 때문에 public ledger가 매번 다름 | 서로 다른 session key에서 functional ledger projection equality |
| termination | terminal 이후 valid-looking permit 실행 | verifier `J_live=0`, transition count 불변 |

## 8. P0/P1/P2 원장

### P0

현재 업데이트된 계약에 열린 P0는 0개다. 다만 다음 강화 명제는 완전한 반례로 기각된다.

> 순수 deterministic event-sourced reducer가 동일한 과거 immutable snapshot의 재제출까지 process 전역에서 single-use로 차단한다.

정리 2의 $X=\{0,1\}$, $\Phi(0,p)=1$이 정확 반례다. 첫 호출 뒤 같은 ordered pair $(0,p)$의 두 번째 평가만 $0$을 반환하게 할 수 없다. 이 강화 명제는 현재 PIC-I5의 부모가 아니며 계약은 linear-history 보장으로 명시적으로 좁혔다.

최초 검산의 concrete-type dispatch, permit 없는 reset, truth alias, omitted learner state와 plain frozen permit 반례는 각각 현재 계약의 protocol-only 조건, lifecycle 정의역, 동적 noninterference, full-state replay, authenticated session permit 조건으로 배제됐다. 이들은 여전히 잘못된 구현을 죽이는 회귀 반례지만 현재 부모 문구에 열린 P0는 아니다.

### P1

1. 식 (1)의 recursive immutable grammar와 field별 exact validation이 실제 제품에 구현됐는지 이 레인은 확인하지 않았다.
2. 식 (4)의 dynamic noninterference와 closure/global/shared-reference 차단은 제품 paired-world kill test가 필요하다.
3. 식 (5)--(7)의 session binding, authentication, descendant single-use와 pure reducer 경계는 제품 kill test가 필요하다.
4. 식 (8)--(9)의 full-state replay, hidden registry 부재와 public ledger projection은 제품 byte replay test가 필요하다.
5. 두 family의 식 (11) invariant와 protocol-only 동일-source orchestrator는 제품 fixture·opaque proxy test가 필요하다.
6. PIC-H1--PIC-H5는 계약대로 미완성이고 V0 결과로 승격할 수 없다.

### P2

1. $F=(\mathcal O,\mathcal A,T,R,H,\Lambda)$에서 $R,H,\Lambda$의 정확한 type과 동치관계를 명시해야 한다. 특히 $H$를 horizon으로 오독하지 않도록 `hidden_state_structure` 같은 이름이 낫다.

## 9. 재현성

독립 검산 artifact:

- `_workspace/ce/agi-physics-independent-core-20260816/artifacts/verify_core_contract.py`

실행 명령:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python '_workspace/ce/agi-physics-independent-core-20260816/artifacts/verify_core_contract.py'
```

기록된 출력 요약:

```json
{"constructive_witnesses":{"authenticated_episode_tick_bound_single_use_permit":true,"canonical_hash_chain_replays_and_detects_tamper":true,"two_nonisomorphic_families_share_one_orchestrator":true,"visible_history_noninterference_is_stronger_than_field_name_audit":true},"counterexamples":{"frozen_dataclass_is_only_shallow":true,"no_truth_field_does_not_imply_noninterference":true,"plain_frozen_permit_is_forgeable":true,"reset_changes_world_without_action_permit":true,"same_observation_seed_config_omits_agent_state":true}}
```

수치 근사나 tolerance 선택은 사용하지 않았다. 모든 반례와 증인은 유한 정수 상태, exact tuple, SHA-256과 exact byte comparison으로 판정했다.

이번 revision의 snapshot-fork no-go는 제품 코드나 scratch 실행에 의존하지 않는다. 정리 2의 두 요구 $\Phi(0,p)=1$과 $\Phi(0,p)=0$이 동일 함수 입력에 동시에 부과되는 exact finite countermodel이다. 따라서 별도 수치 artifact나 tolerance가 없다. revision budget 기록은 `revisions/log`의 `math-verifier` 1회다.
