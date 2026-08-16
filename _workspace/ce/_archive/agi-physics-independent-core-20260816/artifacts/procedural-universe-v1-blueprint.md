# Procedural Universe V1 사전등록 청사진

Status: COMPLETE

Scope: **V1 PREREGISTRATION BLUEPRINT ONLY**

V0 implementation scope: **OUTSIDE — 이 문서는 V0 구현 범위에 포함되지 않는다.**

이 문서는 물리 독립 AGI 코어 V0가 닫힌 뒤 별도 후속 run에서 사용할 승격안이다. 현재 run의 승인 구현 파일, 제품 코드, V0 테스트, Phase A 봉인물과 결과를 변경하거나 확장할 권한을 부여하지 않는다. 여기에 적힌 표본수, threshold, generator와 evaluator는 V1 사전등록 전에 구현·감사·hash 고정되어야 하며, 현재 시점의 empirical 결과나 AGI 증거가 아니다.

## 1. 목적과 주장 경계

V1의 질문은 현실 물리법칙을 얼마나 많이 암기했는지가 아니라 다음 능력을 제한된 생성문법 안에서 측정할 수 있는지다.

> 동일한 frozen system이 제한된 능동 탐색으로 처음 보는 가상세계의 작동법칙을 알아내고, 그 법칙을 새로운 초기상태와 목표에 적용해 계획할 수 있는가.

V1에서 `물리 독립`은 임의의 가능한 우주 전체에 대한 보편성을 뜻하지 않는다. 아래에 고정된 유한 procedural grammar 안에서 관측 부호, 행동 의미, 전이법칙과 목표를 분리하여 바꿀 수 있다는 뜻이다. V1이 통과해도 허용되는 지위는 `PROCEDURAL_UNIVERSE_V1_GO`이며 현실 로봇 grounding, 연속 물리, 인간 수준 AGI, 의식 또는 불교적 성취로 승격하지 않는다.

V0의 PIC-I1--PIC-I5와 Phase A의 synthetic development GO는 독립 보존한다. V1 실패는 그 좁은 선행 산출을 삭제하지 않고, 선행 통과도 V1 점수나 confirmation 증거로 합산하지 않는다.

## 2. 평가 객체

하나의 가상세계는 다음 튜플로 정의한다.

$$
U=(E,X,R,A,\mathcal T,\Omega,G,H).
$$

- $E$: 4--12개의 typed entity;
- $X$: entity별 유한 이산 속성;
- $R$: 이웃, 연결, 포함과 같은 관계;
- $A$: 4--8개의 유한 행동 원자;
- $\mathcal T$: guarded rewrite rule로 표현된 전이법칙;
- $\Omega$: 부분관측, 잡음과 agent-facing 부호화;
- $G$: 목표 predicate와 금지조건;
- $H$: query horizon.

`family`는 단순 seed가 아니라 법칙 AST의 구조적 skeleton이다. 같은 family의 `block`은 skeleton을 공유하고 객체 수, 초기상태, 상수, relation graph, 토큰 치환과 support/query 상태만 다르게 뽑는다. 따라서 통계적 독립 단위는 frame, step, episode 또는 block이 아니라 family skeleton이다.

## 3. V1 최소 세계 문법

### 3.1 상태 자료형

V1은 exact causal truth와 유한 oracle 검사가 가능한 결정론적 이산세계로 시작한다.

- `categorical(k)`;
- `integer_mod(k)`;
- `bounded_integer(lo, hi)`;
- `boolean`;
- `relation(entity, entity)`;
- `location(node)`, 여기서 node는 Euclidean grid가 아닌 임의 유한 graph의 정점일 수 있다.

### 3.2 전이 연산자

최소 허용 연산자는 다음으로 고정한다.

- `assign`;
- `toggle`;
- `increment_mod`;
- `transfer`;
- `swap`;
- `move_along_relation`;
- `create_relation`;
- `delete_relation`;
- `delayed_effect`;
- `conditional_branch`;
- `priority`;
- `noop_on_invalid`.

한 tick의 기본 evaluation order는 다음과 같다.

1. agent action decode;
2. action precondition;
3. direct effect;
4. entity interaction;
5. autonomous rule;
6. delayed event queue;
7. viability와 terminal 판정;
8. observation rendering.

기본 순서는 family manifest에 명시한다. 일부 confirmation family는 사전 고정된 다른 update-order motif를 사용한다. V1은 우선 결정론만 다루며 확률적 rule은 별도 버전으로 분리한다.

### 3.3 Evaluator-only rule AST 예

```json
{
  "rule_id": "r_07",
  "phase": "interaction",
  "priority": 20,
  "when": {
    "and": [
      {"eq": [{"action_op": true}, {"token_ref": "op_3"}]},
      {"related": ["action.target", "actor", "rel_2"]},
      {"eq": [{"attr": ["action.target", "ch_1"]}, {"value_ref": "v_4"}]}
    ]
  },
  "effects": [
    {
      "op": "transfer",
      "from": "actor",
      "to": "action.target",
      "channel": "ch_5",
      "amount": 1
    },
    {
      "op": "delayed_effect",
      "delay": 2,
      "effect": {
        "op": "toggle",
        "entity": "action.target",
        "channel": "ch_2"
      }
    }
  ],
  "probability": {"numerator": 1, "denominator": 1}
}
```

## 4. Family schema와 novelty

Family manifest의 최소 필드는 다음과 같다.

```json
{
  "family_id": "fam_c_014",
  "grammar_version": "pu-1.0",
  "novelty_stratum": "structural",
  "rule_dependency_motif": "delayed_cycle_with_gate",
  "operators_used": ["transfer", "toggle", "delayed_effect"],
  "action_arity_signature": [1, 2, 1, 1],
  "latent_state_count": 2,
  "update_order_id": "uo_3",
  "canonical_ast_sha256": "<sha256>",
  "parent_train_motifs": [],
  "generator_constraints_sha256": "<sha256>"
}
```

Confirmation은 세 novelty stratum을 분리한다.

1. `compositional`: 원시 연산자는 train에서 보였지만 rule dependency graph와 시간적 조합이 새롭다.
2. `structural`: train에 없던 dependency motif 또는 연산자 하나를 포함한다. agent는 support 상호작용으로 이를 배워야 한다.
3. `representational`: 법칙 난이도는 대응 world와 같지만 관측 channel, entity ID, 행동 token, goal token과 slot order를 독립 재부호화한다.

Confirmation family는 agent 결과를 보기 전에 다음 조건을 만족해야 한다.

- train/dev canonical AST와 동형이 아니다;
- train/dev와 동일한 rule dependency graph가 아니다;
- 단순 상수, seed 또는 token 변경만으로 새 family를 선언하지 않는다;
- support와 query 초기상태 및 목표 hash가 겹치지 않는다;
- confirmation family 사이 canonical AST가 중복되지 않는다;
- renderer seed와 dynamics seed가 독립이다;
- family ID, split, novelty label, AST hash와 raw seed가 agent payload에 없다.

## 5. Agent-facing protocol

Agent가 받는 관측의 최소 구조는 다음과 같다.

```json
{
  "observation": {
    "step": 17,
    "self_token": "e_a91",
    "entities": [
      {
        "entity_token": "e_b37",
        "type_token": "t_4",
        "features": {
          "c_1": "v_7",
          "c_3": 2,
          "c_8": false
        },
        "relations": [
          {"relation_token": "q_2", "target_token": "e_a91"}
        ]
      }
    ],
    "visible_delayed_events": [],
    "budget_remaining": {
      "steps": 41,
      "tool_calls": 0
    }
  },
  "goal": {
    "all": [
      {
        "exists": {
          "type_token": "t_4",
          "feature_equals": {"channel": "c_3", "value": 0}
        }
      },
      {
        "relation_holds": {
          "source": "self",
          "relation_token": "q_6",
          "target_type_token": "t_2"
        }
      }
    ]
  },
  "available_actions": [
    {
      "op_token": "a_5",
      "argument_schema": ["entity_token"]
    },
    {
      "op_token": "a_8",
      "argument_schema": ["entity_token", "value_token"]
    }
  ]
}
```

Agent action은 다음 계약을 따른다.

```json
{
  "op_token": "a_5",
  "arguments": ["e_b37"]
}
```

Environment 응답의 public 부분은 다음으로 제한한다.

```json
{
  "observation": {},
  "terminal": false,
  "goal_satisfied": false,
  "public_event": "valid_action",
  "steps_remaining": 40
}
```

자연어 의미를 가진 `red`, `push`, `north`, `mass` 같은 token은 사용하지 않는다. `info`, reward shaping, exception text와 serialization에는 law, seed, family, split, oracle 또는 scorer 정보를 넣지 않는다. Query의 성공 여부는 terminal 또는 query 종료 때만 공개하고 중간 reward는 사용하지 않는다.

## 6. Evaluator manifest

V1 evaluator manifest는 최소 다음 필드를 고정한다.

```json
{
  "benchmark": {
    "id": "ce-procedural-universe-v1",
    "protocol_version": "1.0.0",
    "grammar_version": "pu-1.0",
    "split": "confirmation",
    "manifest_sha256": "<sha256>",
    "created_after_model_freeze": true
  },
  "system_boundary": {
    "checkpoint_sha256": "<sha256>",
    "container_digest": "<digest>",
    "system_prompt_sha256": "<sha256>",
    "external_memory_sha256": "<sha256>",
    "retrieval_snapshot_sha256": "<sha256>",
    "network_access": false,
    "human_assistance": false
  },
  "budget": {
    "support_steps_per_block": 256,
    "support_episodes": 4,
    "query_count_per_block": 12,
    "query_horizon": 64,
    "max_wall_seconds_per_step": 10,
    "max_context_tokens_per_block": 65536,
    "max_external_memory_bytes": 1048576,
    "max_retries": 0
  },
  "rng": {
    "future_beacon_id": "<id>",
    "secret_salt_commitment": "<sha256>",
    "namespaces": [
      "family_ast",
      "law_constants",
      "state",
      "support",
      "query",
      "observation_map",
      "action_map",
      "decoy",
      "bootstrap"
    ]
  },
  "family": {},
  "law_program_evaluator_only": {},
  "encoding_map_evaluator_only": {},
  "support_protocol": {},
  "query_protocol": {},
  "generator_acceptance": {},
  "scoring": {},
  "leakage_audit": {}
}
```

`law_program_evaluator_only`, 실제 encoding map, oracle plan과 모든 raw seed는 agent process로 직렬화하지 않는다. Agent-visible schema와 evaluator-only schema는 별도 타입으로 구현하고 serialization allow-list test를 둔다.

## 7. Block 실행 절차

한 block은 하나의 구체적인 law program과 그것을 공유하는 support/query 묶음이다.

1. 동일한 frozen checkpoint와 빈 block memory로 시작한다.
2. Agent에게 4개의 sandbox episode, 합계 256 step을 준다.
3. Sandbox에는 score용 목표와 중간 reward를 주지 않는다.
4. Agent는 행동을 선택하여 법칙을 능동 탐색한다.
5. Support 종료 시 허용된 전체 writable state를 snapshot한다.
6. 12개 query는 모두 동일 support snapshot에서 각각 fork한다.
7. Query끼리는 결과나 memory를 공유하지 않는다.
8. 각 query는 새 초기상태, 객체와 목표를 사용하고 horizon은 64다.
9. Query 중 checkpoint weight와 global memory write는 금지하고 fork-local state만 허용한다.
10. 한 family는 독립 law constant와 layout을 가진 4개 block으로 구성한다.

Query를 동일 snapshot에서 fork함으로써 앞 query의 정답을 뒤 query가 학습하는 누수를 막는다. System이 전체 writable state snapshot을 제공하지 못하면 V1 confirmation 대상이 될 수 없다.

## 8. Split과 custody

V1의 사전등록 기본 수량은 다음과 같다.

| Split | Family skeleton | Block/family | 역할 |
|---|---:|---:|---|
| pilot/train | 12 | 8 | 공개 generator, 학습과 디버깅 |
| development | 8 | 8 | 모델, baseline, budget과 threshold 선택 |
| confirmation/compositional | 12 | 4 | 새 법칙 조합 |
| confirmation/structural | 12 | 4 | 새 motif 또는 operator |
| confirmation/representational | 8 | 4 paired twins | 재부호화 강건성 |

Confirmation은 총 32개 독립 family, 128개 block과 1,536개 query다. 통계 resampling 단위는 32개 family이며 block과 query는 family 내부에서 먼저 평균한다.

Custody 절차는 다음과 같다.

1. checkpoint, system prompt, tool, memory, retrieval corpus, dependency와 evaluator code hash를 고정한다.
2. Confirmation family skeleton과 seed는 model freeze 뒤 미래 공개 난수 beacon과 custodian secret salt로 생성한다.
3. Secret salt의 commitment와 one-shot exclusive receipt를 실행 전에 남긴다.
4. `family_ast`, `law_constants`, `state`, `support`, `query`, `observation_map`, `action_map`, `decoy`, `bootstrap`은 SHA-256 domain-separated namespace를 사용한다.
5. 한 checkpoint에는 confirmation을 한 번만 연다.
6. 실패 뒤 수정하려면 새 버전, 새 beacon과 새 confirmation block을 사용한다.
7. 평가가 끝난 뒤 generator, seed derivation, salt와 receipt를 공개하여 replay 가능하게 한다.

Development 결과는 모델 선택 자료이며 confirmation 증거가 아니다.

## 9. Generator acceptance gate

Agent 결과를 보기 전에 생성된 family와 block은 다음을 모두 만족해야 한다.

- Truth-aware oracle query 성공률이 $0.95$ 이상이다.
- `RandomValid` 성공률이 $0.15$ 이하이다.
- No-support best policy 성공률이 $0.35$ 이하이다.
- 최소 한 쌍의 observational twin이 존재한다.
- Twin은 초기 passive observation 분포가 같지만 개입 뒤 최적 행동이 다르다.
- 각 핵심 rule에는 256 support step 안에 실행 가능한 diagnostic intervention이 하나 이상 존재한다.
- Query 최적해는 support state 또는 action sequence의 exact replay가 아니다.
- Invalid action 반복, reset 또는 timeout으로 목표를 달성할 수 없다.
- 주 interpreter와 독립 reference interpreter가 property test에서 동일한 transition을 낸다.
- Action/observation token 길이, JSON field order, payload byte 수와 응답 latency로 family 또는 split을 분류할 수 없다.

한 조건이라도 실패한 family는 candidate 결과를 열기 전에 폐기하고 같은 acceptance rule로 새 family를 생성한다. Candidate 결과를 본 뒤 난이도를 이유로 family를 교체하는 행위는 금지한다.

## 10. Baseline과 비교 공정성

모든 baseline은 candidate와 동일한 support/query, observation, action, test-time step, wall-clock, context와 external-memory budget을 사용한다.

- `B0 RandomValid`: 유효 행동 무작위;
- `B1 Reactive`: block memory가 없는 반응형 policy;
- `B2 MetaRL`: train family에서 학습된 recurrent model-free agent;
- `B3 LatentWM-MPC`: latent dynamics와 bounded model-predictive control;
- `B4 DSL-Bayes`: 공개된 grammar 위에서 rule posterior를 갱신하는 구조화 learner;
- `B5 Oracle`: true law AST를 받는 planner. 난이도와 ceiling 확인 전용이며 matched baseline 비교에는 넣지 않는다.

`B1`--`B4` 가운데 development에서 primary score가 가장 높은 하나를 $B^*$로 freeze한다. Confirmation 결과를 본 뒤 baseline을 교체하거나 parameter budget을 늘릴 수 없다. 가능하면 동일 backbone에서 block-memory 제거, horizon-one planning과 support-shuffle ablation도 함께 실행한다. Parameter 수가 정확히 같지 않으면 parameter, pretraining compute와 test-time compute를 별도 원장으로 공개한다.

## 11. Primary metric과 통계

각 query의 terminal 성공을 $s\in\{0,1\}$로 둔다. Family $f$에서 candidate, random과 oracle의 성공률을 각각 $\bar s_A(f)$, $\bar s_R(f)$와 $\bar s_O(f)$라 하면 family score를 다음처럼 정의한다.

$$
S_f=
\frac{\bar s_A(f)-\bar s_R(f)}
{\bar s_O(f)-\bar s_R(f)}.
$$

Generator acceptance 때문에 분모는 최소 $0.80$이다. $S_f$는 자르지 않은 원값으로 보존한다. Best matched baseline과의 paired 차이는 다음과 같다.

$$
\Delta_f=S_f(A)-S_f(B^*).
$$

Primary는 $S_f$와 $\Delta_f$다. Query step 수, invalid action, exploration entropy, inferred-law exact match와 intervention prediction NLL은 secondary이며 primary 실패를 뒤집지 않는다.

사전 고정 bootstrap seed로 novelty stratum을 보존한 family-level paired bootstrap 10,000회를 수행한다. Block이나 query를 독립 표본으로 재표집하지 않는다. 모든 LCB와 UCB는 단측 95% bound다.

### 11.1 Development 진행조건

다음을 모두 만족해야 confirmation manifest를 만들 수 있다.

1. Development 평균 $S_f\geq0.55$;
2. Development 평균 $\Delta_f\geq0.03$;
3. Representational twin의 평균 score 감소가 $0.15$ 이하;
4. 아래 leakage/property kill test 전부 통과;
5. Budget 위반 0건.

실패하면 `REVISE`이며 confirmation 생성과 실행은 금지한다.

### 11.2 Exact confirmation 판정

다음을 모두 만족할 때만 `PROCEDURAL_UNIVERSE_V1_GO`다.

1. $\operatorname{LCB}_{95}(\mathbb E_f[S_f])\geq0.65$;
2. $\operatorname{LCB}_{95}(\mathbb E_f[\Delta_f])\geq0.05$;
3. Compositional stratum의 $S_f$ LCB가 $0.60$ 이상;
4. Structural stratum의 $S_f$ LCB가 $0.50$ 이상;
5. Encoding-isomorphic twin의 paired score 감소에 대한 UCB가 $0.10$ 이하;
6. Correct-support와 wrong-law-support의 paired 차이 LCB가 $0.15$ 이상;
7. 필수 kill test 전부 통과;
8. Resource, one-shot, hash와 custody 감사 통과.

수치는 자연상수가 아니라 V1에서 사전 고정하는 engineering margin이다. 하나라도 실패하면 `PROCEDURAL_UNIVERSE_V1_STOP`이다. 성공한 하위 항목은 별도로 남기되 평균으로 실패를 상쇄하지 않는다.

## 12. Leakage와 shortcut kill test

### K1. Encoding isomorphism

동일한 law, state와 goal을 새로운 object ID, feature channel, value token, action token과 entity order로 재부호화한다. Paired score 감소의 UCB가 $0.10$을 넘으면 semantic-token prior 또는 serialization shortcut으로 판정하여 STOP한다.

### K2. Wrong-law support

Family $f$의 query에 observationally similar한 twin $f'$에서 만든 support snapshot을 연결한다. Correct-support 우위의 LCB가 $0.15$ 미만이면 support에서 법칙을 학습했다는 주장을 기각한다.

### K3. No-support

동일 query를 빈 memory에서 실행한다. Generator의 no-support ceiling은 $0.35$다. Candidate의 full-support와 no-support score 차이가 $0.10$ 미만이면 world-learning 주장을 기각한다.

### K4. Observational twin intervention

Passive observation은 같고 action result가 다른 두 world를 제시한다. 두 world에서 같은 개입 정책을 고집하여 paired 성공률이 $0.50$ 이하이면 causal grounding gate를 실패한다.

### K5. Metadata stripping

Agent-facing payload에 family, split, seed, novelty, AST, oracle, scorer 식별자, file path, generator exception 또는 raw PRNG state가 한 건이라도 있으면 run을 무효화한다. 응답 크기와 timing은 padding과 batching으로 family label과 독립화한다.

### K6. Goal/reward leakage

Intermediate shaped reward를 금지하고 query success는 terminal에서만 공개한다. Goal-token mapping은 dynamics와 action mapping에서 독립된 RNG를 쓴다. Reward-only 또는 initial-state-only probe가 acceptance ceiling을 넘으면 해당 family를 candidate 실행 전에 폐기한다.

### K7. Train duplication

Canonical AST isomorphism, rule dependency graph, support/query state와 goal hash를 train/dev와 비교한다. Exact AST 또는 graph duplicate 한 건이라도 있으면 confirmation block을 무효화한다. 상수나 token만 바뀐 복제는 새 family가 아니다.

### K8. Brute-force shortcut

`RandomValid` 성공률이 $0.15$를 넘으면 family를 폐기한다. Invalid action, reset, timeout 또는 error recovery가 유리한 state transition을 만들면 interpreter bug로 run을 무효화한다.

### K9. Query contamination

모든 query는 동일 support snapshot에서 독립 fork한다. 한 query의 observation, action, success 또는 memory가 다음 query에 기록되면 run을 무효화한다. Support와 query의 초기상태 및 goal hash 중복은 0건이어야 한다.

### K10. Canary와 evaluator 접근

Evaluator-only manifest에는 secret canary를 둔다. Candidate 출력, prompt, observation 또는 writable memory에 canary가 나타나면 run을 무효화한다. Network, evaluator filesystem, environment variable, wall-clock seed, 허용되지 않은 human 또는 tool 접근은 한 건만 있어도 STOP이다.

### K11. Independent interpreter

주 dynamics interpreter와 구현 경로가 다른 최소 reference interpreter를 둔다. 사전 생성 transition corpus와 random property trace에서 canonical next-state byte가 모두 일치해야 한다. 불일치가 confirmation 중 발견되면 score를 해석하지 않고 run을 무효화한다.

### K12. Test-time retry와 선택 보고

Checkpoint당 confirmation 실행은 한 번이고 query retry는 0이다. 여러 decoding seed, prompt 또는 planner 설정을 confirmation에서 실행해 최선만 고르는 행위를 금지한다. 실행 전 freeze된 단일 정책을 사용하지 않으면 STOP이다.

## 13. 구현 순서와 V1 재개 조건

이 청사진은 구현 승인이 아니다. 별도 V1 run은 다음 순서로만 시작한다.

1. Grammar와 JSON schema를 versioned contract로 고정한다.
2. 주 interpreter와 독립 reference interpreter를 구현한다.
3. Generator acceptance와 AST canonicalization property test를 닫는다.
4. Agent-visible/evaluator-only 타입과 serialization leak test를 닫는다.
5. Support snapshot과 query fork의 byte-replay를 검증한다.
6. `B0`--`B5`와 compute ledger를 구현한다.
7. Pilot/train과 development만 실행한다.
8. Development 진행조건을 만족한 뒤 model과 evaluator를 freeze한다.
9. Custodian이 future beacon으로 confirmation을 생성하고 one-shot 실행한다.

V1 재개 조건은 V0 run의 정식 종료, 별도 계약과 형식 감사, 위 schema의 구현 가능성 확인, custodian 지정, model/evaluator hash와 자원 원장의 사전등록이다.

## 14. 최종 허용 결론

`PROCEDURAL_UNIVERSE_V1_GO`가 뜻할 수 있는 최강 문장은 다음으로 제한한다.

> 이 frozen system은 고정된 이산 procedural grammar 안에서 보지 못한 법칙 family를 제한된 능동 탐색으로 학습하여 새로운 목표에 적용했고, 사전 고정된 matched baseline과 절대성능 기준을 one-shot confirmation에서 함께 통과했다.

그 밖의 현실 물리 일반화, 로봇 grounding, 장기 continual learning, 언어, 사회·윤리 제어, 자기모형, 인간 뇌의 근본 알고리즘, AGI 전체 또는 의식 주장은 각각 별도의 계약과 confirmation을 요구한다.
