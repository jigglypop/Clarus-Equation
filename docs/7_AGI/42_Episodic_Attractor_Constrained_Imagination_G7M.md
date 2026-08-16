# G7-M: episodic LTM과 constrained imagination 연구 루프

> 상태: **[산출] V1 validation FAIL / V1 locked test 미개봉; V2 validation PASS / V2 locked test PASS**
>
> 사전등록: V1
> [`episodic_ltm_dream_factorial_v1.json`](../../experiments/preregistration/episodic_ltm_dream_factorial_v1.json),
> V2
> [`episodic_ltm_dream_factorial_v2.json`](../../experiments/preregistration/episodic_ltm_dream_factorial_v2.json)
>
> 검증 산출물: V1
> [`episodic_ltm_dream_factorial_validation_v1.json`](../../artifacts/agi/episodic_ltm_dream_factorial_validation_v1.json),
> V2
> [`validation`](../../artifacts/agi/episodic_ltm_dream_factorial_validation_v2.json)ㆍ
> [`locked test`](../../artifacts/agi/episodic_ltm_dream_factorial_test_v2.json)ㆍ
> [`integrity ledger`](../../artifacts/agi/episodic_ltm_dream_factorial_integrity_v2.json)
>
> 관련 실험:
> [제한 기억ㆍreplayㆍ계층 계획 G7](31_Memory_Replay_Hierarchical_Planning_G7.md),
> [희소 인과 브리지 세계모형 G9-CB](41_Sparse_Causal_Bridge_World_G9CB.md)

## 1. 무엇을 시험했는가

한 줄로 말하면, 실제 episode를 오래 보존하는 경로와 관찰 조각을 제약 안에서
재조합하는 오프라인 경로를 분리한 뒤 함께 작동하는지를 시험했다.

V1은 다음 두 질문을 2×2 요인 실험으로 고정했다.

1. 장기 episodic store가 방해 뒤에도 과거 episode의 신원과 instance-specific
   hidden trajectory를 noisy partial cue에서 복원하는가?
2. 제한된 오프라인 재조합이 관찰하지 않은 유효 조합을 만들되, 그것을 실제
   경험이나 회상으로 위조하지 않는가?

**[공리: 모델 선택]** 이 실험은 known-slot, 작은 합성 환경, 고정 용량 기억,
NumPy CPU라는 선택 위에서만 수행한다. 이 선택은 인간 기억, 생물학적
해마-피질 기제, 수면이나 꿈의 의식, 일반 세계모형 또는 AGI의 공리가 아니다.

V5--V7은 실제 입력을 끊은 짧은 자유전개를 다뤘지만 episodic store와
cue completion을 구현하지 않았다. 반대로 이 V1은 planning이나 장기
world-model rollout을 구현하지 않았다. 어느 결과도 다른 축의 증거로
소급하지 않는다.

## 2. V1의 용어와 provenance

**[정의] Observed episode**는 환경에서 실제 관찰해 저장한 record다. V1의
episodic LTM에는 이 record만 들어갈 수 있다.

**[정의] Episodic reinstatement**는 partial cue에서 저장 episode의 identity와
instance-specific hidden trajectory를 복원하는 연산이다. identity를 맞히는 것과
trajectory를 안정적으로 복원하는 것은 서로 다른 gate다.

**[정의] Slow schema table**은 관찰 조각 사이의 공통 구조를 담는 느린
binding 표다. 이것은 episodic LTM과 동일하지 않다.

**[정의: V1 한정] Dream-like constrained offline recombination**은
관찰한 prefixㆍconnectorㆍsuffix 조각의 co-occurrence에서 component를 추정하고,
같은 component 안의 빠진 binding만 조합하는 1회 오프라인 연산이다. 출력에는
`synthetic/hypothetical` provenance를 붙이며, observed episode나 recalled
memory라고 부르지 않는다.

**[미완성]** 일반적인 reality constraint `Π_M`, 자유로운
counterfactual 생성, 시간축 world-model consolidation, 행동조건부 planning은
이 V1에 없다. 여기서 “dream-like”는 인간의 꿈을 가리키는 생물학적 용어가
아니라 제한된 계산 경로의 이름이다.

## 3. 사전등록한 2×2 설계

**[공리: 실험 계약]** 네 cell은 다음처럼 한 요인만 바꾼다.

| cell | persistent episodic LTM | offline recombination | 평가 때 허용된 정보 |
|---|---:|---:|---|
| `M00` | 없음 | 없음 | slow-schema fallback |
| `M10` | 있음 | 없음 | observed-only LTM + fallback |
| `M01` | 없음 | 있음 | synthetic missing-binding table |
| `M11` | 있음 | 있음 | observed-only LTM + synthetic missing-binding table |

`M01`은 `M11`과 같은 bounded wake snapshot으로 offline pass를 수행하지만,
평가 전 workspace를 지운다. 따라서 dream-only cell에 queryable LTM을 몰래
제공하지 않는다. 각 LTM cell은 context A와 B의 실제 observed episode 96개만
보존하고, 각 dream cell은 seed마다 빠진 binding 24개만 제안할 수 있다.

**[예측: V1 사전등록]**

- LTM 주효과는 과거 A의 identity와 hidden trajectory를 모두 개선한다.
- dream-like 주효과는 유효 새 조합의 coverage를 높이고 hidden NRMSE를 낮춘다.
- `M11`은 LTM과 dream의 두 절대 gate를 모두 통과하며 단일 요인 cell보다
  실질적으로 나빠지지 않는다.
- interaction은 추정하지만 양수일 필요는 없다. 95% interval의 하한이
  양수일 때만 synergy를 주장한다.

등록 이력과 byte-level lock은 다음과 같다.

| 항목 | 값 |
|---|---|
| 최초 사전등록 commit | `48be628` |
| registered seed를 열기 전 clerical clarification | `d2deae4` |
| 최종 registration raw SHA-256 | `6487156371e4c42877fa0813dd170fb000ce11fe05e51f34bceb74653159fac0` |
| 구현 commit | `81c7ce6` |
| train calibration SHA-256 | `9ab33380e47feaadc608ff147cb012763d1979e114afda7caaf87059f72be8cd` |
| validation FAIL 보존 commit | `0cc20f2` |
| validation artifact SHA-256 | `7b3bbf75349ae651fbed1211f15c1b4b26fa102ebdb7053876c12ab63525bf79` |

clarification은 metric pooling, RNG stream, query construction, join quantile,
paired contrast, provenance coverage와 calibration artifact를 명시한
pre-implementation 정정이다. hypotheses와 수치 gate는 바꾸지 않았고,
그 전에 registered seed 결과를 보지 않았다.

## 4. Validation V1 산출

**[산출]** open train seed 40개로 calibration을 동결한 뒤 validation seed
40개를 평가했다. 등록 check 73개 중 67개가 PASS, 6개가 FAIL이어서
all-of 판정은 **FAIL**이다. resource gate는 PASS였지만 성능 실패를 덮지
않는다. 이 판정 뒤 locked test seed 60개와 test artifact는 열지 않았다.

### 4.1 Cell 평균

| cell | old-A identity accuracy | old-A positive coverage | old-A hidden NRMSE | attractor convergence | extra-step stability | novel coverage | novel hidden NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M00` | 0.000000 | 0.000000 | 0.130655 | 1.000000 | 1.000000 | 0.000000 | 0.654725 |
| `M10` | 0.993229 | 0.996354 | 0.135414 | 0.418229 | 0.540104 | 0.000000 | 0.654725 |
| `M01` | 0.000000 | 0.000000 | 0.130655 | 1.000000 | 1.000000 | 1.000000 | 0.089741 |
| `M11` | 0.993229 | 0.996354 | 0.135414 | 0.418229 | 0.540104 | 1.000000 | 0.089741 |

`M00/M01`의 convergence와 stability 1.0은 episodic attractor가 좋다는 뜻이
아니다. 두 cell에는 persistent episodic retrieval 자체가 없으므로, 이 값은
비-LTM 경로의 trivial finite behavior다.

### 4.2 Paired 요인 효과

| 지표 | 주효과 평균 | paired 95% interval | 판정 |
|---|---:|---:|---|
| LTM: recall identity gain | +0.993229 | [0.989416, 0.997043] | PASS |
| LTM: hidden-NRMSE reduction | -0.004759 | [-0.007192, -0.002326] | FAIL |
| dream: novel coverage gain | +1.000000 | [1.000000, 1.000000] | PASS |
| dream: novel hidden-NRMSE reduction | +0.564985 | [0.559025, 0.570944] | PASS |

이 표의 NRMSE 효과는 paired seed별 **절대** 차이다. 별도의 magnitude gate가
사용한 cell-mean **상대** 감소율은 LTM -0.036422, dream 0.862934였다.

identity, recall error, novel coverage와 novel error의 factorial interaction 평균은
모두 0이었다(수치 roundoff 범위 포함). 따라서 coexistence는 관찰했지만
synergy 증거는 없다.

### 4.3 실패한 여섯 check

| check | 관찰값 | 등록 gate | 지위 |
|---|---:|---:|---|
| `L_main.hidden_reduction` | relative = -0.036422 | 최소 0.35 | FAIL |
| `L_main.hidden_ci` | lower = -0.007192 | 최소 0.0 | FAIL |
| `M10.convergence` | 0.418229 | 최소 0.99 | FAIL |
| `M10.extra_stability` | 0.540104 | 최소 0.99 | FAIL |
| `M11.convergence` | 0.418229 | 최소 0.99 | FAIL |
| `M11.extra_stability` | 0.540104 | 최소 0.99 | FAIL |

old-A hidden NRMSE의 절대값 0.135414는 등록 상한 0.45 안이었지만, matched
no-LTM cell의 0.130655보다 나빠졌다. 따라서 “identity를 찾았으니 trajectory도
복원했다”라고 결론낼 수 없다. 수치 고정점도 convergence와 extra-step
stability를 통과해야 episodic attractor라고 부를 수 있는데, 두 LTM cell 모두
그 조건을 크게 놓쳤다.

## 5. 성공한 경로와 실패한 경로를 분리한다

**[산출: 제한된 dream path 성공]** `M01/M11`은 seed마다 synthetic binding
24개를 받아들였고, novel coverage 1.0과 novel hidden NRMSE 0.089741을
달성했다. contextㆍcomponentㆍportㆍjoin constraint violation은 모두 0,
observed binding overwrite와 hash change도 0이었다. 이것은 오직
“관찰 조각으로 known missing slot을 채우는 constrained schema-table
augmentation”의 성공이다.

**[산출: LTM identity-only 주효과]** `M10/M11`은 old-A identity accuracy
0.993229와 positive coverage 0.996354를 달성했다. 그러나 trajectory NRMSE
감소는 음수였고 recurrent convergence와 extra-step stability가 실패했다.
따라서 V1에서 입증된 LTM의 추가 효능은 identity retrieval에 한정되며,
instance-specific trajectory reinstatement나 episodic attractor는
**[미완성]**으로 남는다.

**[산출: provenance와 false-memory guard]**

- unstored-lure false episode recall rate는 0.031771로 등록 상한 0.05 아래였다.
- accepted-wrong rate는 0.003125로 상한 0.05 아래였다.
- synthetic-to-LTM insert, heldout-target read, observed overwrite,
  invalid-query non-abstention과 nonfinite output은 모두 0이었다.
- LTM cell은 observed item 96개와 persistent trace 73,728 bytes를 사용했다.
- 외부 다운로드와 raw trajectory file write는 0, 전체 wall time은
  15.248초였다.

**[산출: 통합 판정]** `M11`은 dream gate를 유지했고 LTM identity 결과도
`M10`과 같아 factor 간 antagonism check는 통과했다. 그러나 사전등록은
`M11`이 두 절대 gate family를 모두 통과하도록 요구했다. LTM trajectory와
안정성 실패 때문에 통합 전체는 FAIL이다.

## 6. 문헌이 지지하는 범위

**[공리: 외부 입력]** 아래 1차 연구는 설계 영감과 반례 경계로만 사용한다.

| 연구 | V1에 준 영감 | 이 문서가 주장하지 않는 것 |
|---|---|---|
| [Horner et al. (2015)](https://www.nature.com/articles/ncomms8462) | partial cue와 pattern completion을 분리 측정 | V1 recurrent rule이 해마 기제라는 주장 |
| [Spens & Burgess (2024)](https://www.nature.com/articles/s41562-023-01799-z) | 빠른 episode trace와 느린 생성모델의 역할 분리 | 뇌가 같은 모델을 쓰거나 dream이 곧 replay라는 주장 |
| [Gupta et al. (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4460981/) | replay가 경험 궤적의 복사만은 아닐 수 있다는 관찰 | V1 synthetic binding이 동물 replay나 꿈이라는 주장 |
| [Wamsley (2022)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0264574) | dream report가 여러 waking source를 결합할 수 있다는 관찰 | 재조합이 항상 기억이나 일반화를 개선한다는 인과 주장 |
| [Schapiro et al. (2017)](https://www.nature.com/articles/s41598-017-12884-5) | 밤잠 뒤 shared-property memory가 개선되고 unique-property memory가 보존된 결과; nap 이득은 lower-frequency category에 한정 | dream 자체의 효과 또는 sleep의 보편적 consolidation 이득 |
| [Zaki et al. (2025)](https://www.nature.com/articles/s41586-024-08168-4) | co-reactivation이 기존 기억을 오염시킬 수 있다는 안전 경계 | V1의 0 overwrite가 일반 false-memory 문제를 해결했다는 주장 |

이 문헌들은 V1의 synthetic slot completion을 인간의 꿈으로 바꾸지 않으며,
V1 FAIL을 생물학 이론의 반증으로 만들지도 않는다.

## 7. V1 완결성 판정

| 주장 | V1 판정 |
|---|---|
| observed-only store가 noisy cue에서 old-A identity를 복원한다 | **[산출]**, 이 합성 family의 validation 한정 |
| LTM이 instance-specific trajectory를 matched no-LTM보다 개선한다 | **[산출] V1 예측 FAIL**, active claim에서 제거 |
| V1 recurrent retrieval이 episodic attractor다 | **[미완성]**, convergenceㆍstability FAIL |
| constrained offline pass가 missing binding을 복원한다 | **[산출]**, known-slot 합성 family 한정 |
| LTM과 dream-like path가 수치적으로 공존한다 | **[산출]**, interaction 0ㆍsynergy 주장 없음 |
| 꿈과 장기기억을 함께 탑재하면 전체 문제가 해결된다 | **[미완성]**, V1 validation FAIL |
| 인간 기억ㆍ꿈ㆍ수면, 일반 세계모형, planning 또는 AGI | **범위 밖** |

V1 validation 결과를 본 뒤 같은 seed와 gate를 조정해 PASS로 만들지 않는다.
다음 LTM 동역학이나 representation 변경은 새 버전, 새 validation/test seed,
새 raw SHA로 사전등록해야 한다. V1 자체가 validation all-of gate에 실패했으므로
V1 locked test는 미개봉으로 영구 보존하며, 후속 버전은 fresh test seed를 쓴다.

## 8. V2: 한 가지 변경만 한 재검증

**[예측: V2 사전등록 당시]** V1의 여러 trace를 부드럽게 섞는 recurrent
retrieval을 cue가 가장 잘 맞는 observed trace 하나의 hidden 좌표로 완성하는
연산으로만 바꾸면, fresh seed에서 V1 soft comparator보다 old-A hidden trajectory
복원이 좋아지면서 identityㆍcoverageㆍfalse-memory guard는 나빠지지 않는다.

**[공리: V2 모델 선택]** V2의 hard cue-anchored completion은 표준화된 observed
cue의 masked cosine으로 후보를 고르고 train-only threshold를 넘을 때만 수락한다.
수락하면 관찰된 cue 좌표는 그대로 고정하고, 가려진 좌표만 선택된 observed
trace 하나에서 복사한다. 이는 결정론적 content-addressed exemplar/template
completion이다. recurrent attractor, 생물학적 해마 기제 또는 인간 기억에 관한
공리가 아니다.

V2는 이 LTM 연산 외의 합성 세계, query, slow schema, dream-like 경로,
provenanceㆍconstraint gate를 V1과 같게 유지했다. 동결된 V1 soft retrieval도 같은
fresh V2 seed와 query에서 별도 train-only threshold로 다시 평가했다. 등록 seed를
열기 전 off-range seed `76001--76004`에서 공유 경로와 frozen V1 comparator의
byte-level 동등성을 확인했다.

| 잠금 항목 | SHA-256 또는 commit |
|---|---|
| V2 사전등록 commit / raw SHA-256 | `375515b` / `973e90111ee98862a5c9ffc3f86509b46ee4e263b5a977e7e1504e00109092b9` |
| 구현 commit | `d74a56b` |
| V2 module / CLI SHA-256 | `edace8bb0ff63eb04bf87518a2f69c4f2960d5a200fec2ac9f1b2bf2a89fa6b1` / `b971fdd8520bb4be447912cef89b8697aaec9eaf6424f2c13c1bdb5ff6d171b8` |
| implementation lock commit / artifact SHA-256 | `4980bb5` / `cb88993b595ad3e527e59d48ba50b471af14585cbadd1e810328bbd0e2208105` |
| train calibration artifact SHA-256 | `ccec7b07540099bba1dd3b482f7073c77115e6a16a591410c5e5d422fb0eb76b` |
| validation commit / artifact SHA-256 | `3e4fafb` / `56bf2253774634a1997b1b59c35912eb804182032fd2bceb0938e845b6527e30` |
| locked-test commit / artifact SHA-256 | `9d11f86` / `c13bce6a03913c8edaec20f038ef806fc0760f9e1602dbafab47693a9bf9c3de` |
| post-test integrity ledger SHA-256 | `d4153bcbf71ade1b1c14d35f2132addc381d455de9aafa17114d7a1c8e47cda6` |

train `80100--80139`, validation `81100--81139`, locked test `82100--82159`는
서로 겹치지 않는다. **[산출]** validation 40 seed에서 등록 performance check
84개와 resource check 7개가 모두 통과한 뒤에만 test를 열었다. test artifact의
lock은 위 validation artifact SHA-256을 가리키며, test 60 seed에서도 같은
84+7 all-of check가 모두 통과했다. 이 절차는 V2 test의 잠금에 관한 것이며,
실패한 V1의 locked test는 열지 않았다.

## 9. V2 locked-test 산출

### 9.1 2×2 cell 평균

| cell | old-A identity | positive coverage | old-A hidden NRMSE | one-step idempotence | repeat identity stability | novel coverage | novel hidden NRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M00` | 0.000000 | 0.000000 | 0.131152 | 1.000000 | 1.000000 | 0.000000 | 0.656863 |
| `M10` | 0.995486 | 0.995486 | 0.004074 | 1.000000 | 1.000000 | 0.000000 | 0.656863 |
| `M01` | 0.000000 | 0.000000 | 0.131152 | 1.000000 | 1.000000 | 1.000000 | 0.089235 |
| `M11` | 0.995486 | 0.995486 | 0.004074 | 1.000000 | 1.000000 | 1.000000 | 0.089235 |

**[산출: hard completion]** LTM의 old-A hidden-NRMSE 상대 감소는
0.968940이고, recall-error paired 주효과는 0.127079, 95% interval
[0.124896, 0.129261]이다. `M10/M11`의 accepted-wrong rate는 0,
unstored-lure false episode recall rate의 seed 평균은 0.027431로 등록된 0.05
gate를 통과했지만 seed 최댓값은 0.083333이었다. 따라서 lure false recall을 모든
환경에서 제거했다고 주장하지 않는다. one-step idempotence와
repeat identity stability 1.0, hidden idempotence error와 cue clamp error 0은
이 결정론적 완성 연산이 반복 호출에서 안정적이라는 산출이지 attractor
convergence의 산출이 아니다.

**[산출: V1 실패의 등록된 remediation]** 같은 fresh test seed의 frozen V1
soft retrieval은 hidden NRMSE 0.134971, V2 hard completion은 0.004074였다.
paired 개선은 0.130897, 95% interval [0.128358, 0.133437]이고 strict seed win은
60/60이다. identity 차이도 +0.002431, 95% interval
[0.000688, 0.004173]이었으며 coverage 차이는 0이었다. 따라서 V1에서 실패한
이 합성 family의 trajectory 복원 문제는 V2의 hard exemplar completion으로
고쳐졌다고 판정한다. V1 recurrent-attractor 주장이 사후에 성공으로 바뀌는 것은
아니다.

**[산출: dream-like 경로]** `M01/M11`은 seed마다 synthetic binding 24개를
사용해 novel coverage 1.0과 hidden NRMSE 0.089235를 얻었다. no-dream cell의
0.656863 대비 상대 감소는 0.864150이다. 출력 provenance는 `synthetic`이며,
synthetic-to-LTM insert, observed overwrite/hash change, heldout-target read와
contextㆍportㆍjoin violation count는 모두 0이고 no-cross-context component
check도 통과했다. 따라서 이
경로는 observed 조각의 known-slot constrained recombination이지 실제 episode,
회상된 기억, 자유로운 꿈 생성 또는 생물학적 수면이 아니다.

**[산출: 공존, synergy 아님]** `M11`은 LTM 절대 gate와 dream-like 절대 gate를
함께 통과했다. 네 등록 metric family의 factorial interaction은 0 또는 부동소수
roundoff 수준이었다. 두 경로의 공존은 확인됐지만 양의 interaction이나 synergy는
관찰되지 않았다.

## 10. V2 뒤의 완결성 경계

| 주장 | 현재 지위 |
|---|---|
| hard cue-anchored exemplar completion이 이 fixed 96-item known-slot 합성 family에서 V1 soft mixing보다 hidden trajectory를 잘 복원한다 | **[산출] V2 validationㆍlocked test PASS** |
| observed-only LTM과 constrained synthetic schema path가 같은 2×2 cell에서 각 절대 gate를 함께 만족한다 | **[산출] V2 PASS**, interaction 0ㆍsynergy 주장 없음 |
| V2 hard completion이 episodic attractor 또는 hippocampal mechanism이다 | **[미완성]**, 해당 동역학ㆍ생물 자료 없음 |
| V2 dream-like recombination이 인간의 꿈ㆍ수면ㆍreplay다 | **[미완성]**, 제한된 계산적 비유일 뿐 |
| 장기기억과 dream-like 경로가 일반 forgettingㆍsemantic consolidationㆍ세계모형ㆍ장기 planning을 해결한다 | **[미완성]**, 구현ㆍ평가하지 않음 |
| V2가 AGI의 증거다 | **범위 밖** |

V2 PASS는 V1 실패를 보존한 채, 정확히 한 변경의 fresh-seed remediation을
확인한 결과다. 다음 루프는 이 hard template을 biological attractor라고 다시
이름 붙이는 것이 아니라, 다른 episode 수ㆍ가변 slotㆍ분포 이동ㆍ장기 forgetting,
실제 world-model rollout과 행동 planning에서 반례를 찾도록 별도 사전등록해야 한다.
