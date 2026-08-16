# G9-CBM V2 final status audit

Status: COMPLETE

Gate: **REVISE**

## 1. 판정

V2 초안은 V1의 factorial estimand, metric denominator, provenance, common
budget, interaction sign 문제를 실질적으로 고쳤다. 특히 20-parameter core,
raw/standardized 경계, residual-only schema 저장, typed candidate return,
deterministic lesion, exact allocation ledger가 계약 본문에 들어갔다.

그러나 현재 문서만으로는 결과에 영향을 주는 네 선택이 아직 하나의 실행
객체로 고정되지 않는다. 따라서 **등록 구현ㆍtrain/calibration 개시 전
수정이 필요**하다. 이 판정은 설계 폐기가 아니라 contract-only revision이다.
아래 P0 네 건을 문구 그대로 닫고 raw preregistration SHA를 새로 고정하면
재감사 후 구현할 수 있다. PASS가 되더라도 그것은 구현 가능 판정일 뿐
C1--C5의 결과 예측이나 경험적 지지가 아니다.

이 감사에서는 코드, test, calibration, benchmark 또는 어떤 등록 seed도
실행하지 않았다.

## 2. `11-math-v2` P0-1--P0-7 반영 감사

| 항목 | 판정 | 계약 근거 | 독립 감사 |
|---|---|---|---|
| P0-1, fitted intercept 4개 | **PASS** | `00-contract-v2-draft.md:164-193,213-224,353-359` | codecㆍrolloutㆍcontrol이 같은 `(c_hat,d_hat,b_hat,G_hat)` 20-vector를 사용한다. |
| P0-2, raw/std 경계 | **PASS** | `:288-319,610-621` | recall은 raw를 반환하고 schema는 inverse-standardize 뒤 residual `0:4`만 rollout에 쓴다. |
| P0-3, fingerprint-free dream | **REVISE** | `:321-346` | 4-column join과 zero fingerprint storage는 반영됐지만, `P/C/S`의 표본 집계식, ordinary fallback 식, threshold 동률 부등호가 없다. 새 adapter의 출력을 바꿀 자유가 남는다. |
| P0-4, prefix/validity/API | **REVISE** | `:250-286,408-420,424-452,469-501` | evaluation prefix와 action index는 고정됐다. 다만 goal의 시작점 `x_0`이 이미 정의된 prefix `x[0]`인지 prediction anchor `x[12]`인지 충돌하고, typed request에 cost 계산용 `mu_x,sigma_x`가 없다. |
| P0-5, lesion 240→24 | **PASS** | `:529-538,635-657,846-898` | 288 enumeration, 48 observed rejection, first-24 lesion buffer, 216 padding과 denominator 24가 분리됐다. |
| P0-6, budget ownership | **PASS** | `:846-954` | 열거된 payload 합은 `393,216` bytes이고 cap `524,288` 아래다. 모든 cell/control에 동일 owner/shape/hash 규칙이 적용된다. |
| P0-7, inherited boundary/calibration | **PASS 조건부** | `:232-248,757-795` | scoped 12-row wrapper, 72-slot adapter와 calibration population은 고정됐다. 단, P0-3의 adapter 식을 먼저 고정해야 동일 adapter calibration이라는 문장이 실행 가능해진다. |

## 3. 남은 P0와 정확한 수정

### V2-P0-a — residual-only adapter가 수치적으로 미정

**Severity: P0.** `00-contract-v2-draft.md:333-346`은 join 거리만 정의한다.
다음 식을 계약에 추가해야 한다.

```text
R[c,p,i,j] = mean_h T_std[c,p,i,j,h,:,0:4]  for (i,j) in O
P[c,p,i]   = mean_{j:(i,j) in O} R[c,p,i,j,0:5,:]
C[c,p]     = mean_{(i,j) in O} R[c,p,i,j,5:7,:]
S[c,p,j]   = mean_{i:(i,j) in O} R[c,p,i,j,7:12,:]
Dream[c,p,i,j] = concatenate(P[c,p,i], C[c,p], S[c,p,j])
Fallback[c,p,phi] = mean_{(i,j) in O} R[c,p,i,j,phi,:]
```

Dream candidate order is canonical `(c,p,i,j)` and only `M` keys are candidates.
Acceptance is exactly `J_left <= tau_join AND J_right <= tau_join`; equality is
accepted. An empty `M` key resolves to its preallocated index and uses
`Fallback[c,p,phi]`; it is not learner-invalid merely because its hypothetical
occupancy bit is false. 이 식이 의도와 다르면 의도한 단 하나의 식을 대신
적어야 하며 구현 중 선택해서는 안 된다.

### V2-P0-b — planning anchor와 typed cost input 충돌

**Severity: P0.** `:263`은 prediction origin을 `x[12]`로 두지만 `:424-427`은
goal을 `x_0`에서 시작한다고 쓴다. 또한 `:429-452`의 candidate cost에는
`mu_x,sigma_x`가 필요하지만 `CandidateRequestV2` 목록 `:482-487`에 없다.

다음으로 교체한다.

```text
x_anchor := evaluation-prefix state x[12].
The public reference path g starts from x_anchor, not prefix state x[0].
CandidateRequestV2 contains read-only frozen mu_x[4] and sigma_x[4]; their
bytes and SHA equal the train calibration artifact in every cell/control.
```

`codec mu/sigma`도 store/schema 내부에 소유되는지 request field인지 typed
schema에 한 번만 명시한다. 이미 allocation ledger에 잡힌 배열이므로 byte
총량은 바뀌지 않는다.

### V2-P0-c — near-neighbour lure 생성기가 미정

**Severity: P0.** `:275-282`의 “normalized perturbation”에는 perturbation
크기ㆍ결합식이 없어 recall 난이도와 threshold population이 구현 선택에
따라 달라진다. 정규화된 target fingerprint `f`에 대해 다음처럼 고정한다.

```text
draw v ~ Uniform[-1,1]^4 from lure stream
v_perp = v - dot(v,f)*f; reject if ||v_perp||_2 <= 1e-12
v_perp = v_perp / ||v_perp||_2
f_lure = 0.85*f + sqrt(1-0.85^2)*v_perp
accept the first draw with max_other |dot(f_lure,f_other)| < 0.95
```

최대 10,000 draw 뒤 hard-fail은 유지한다. 이 식은 target cosine을 정확히
`0.85`로 만들어 기존 `[0.80,0.90]` 조건을 만족한다. 또한 NumPy bit generator와
버전을 implementation lock에 명시한다.

### V2-P0-d — pre-train feasibility와 locked-test chronology 충돌

**Severity: P0.** `:445-467`의 “before train ... every registered
origin/candidate”를 문자 그대로 수행하면 `:985-991`의 validation/test lock
전에 locked futures/cost를 생성할 수 있다. 다음 state-machine 문구로 바꾼다.

```text
Envelope and task-feasibility assertions run split-locally only when that split
is legally opened: train inside the one train opening, validation inside its one
registered run, and test only after unlock. No validation/test origin, future,
or cost is generated or read during the pre-train lock. Failure hard-fails the
opened split without resampling and never authorizes the next split.
```

Off-range handcrafted fixtures로 assertion code를 미리 시험할 수 있으나 등록
범위의 seed를 preflight/pilot으로 열어서는 안 된다.

## 4. A1--A10 판정

| Action | 판정 | 이유 |
|---|---|---|
| A1 factorial estimand | **PASS** | `:659-713`의 C1 marginal, 두 simple effect와 별도 `RR_joint`가 정합하다. |
| A2 metrics | **PASS** | `:564-657`에 `E_all`, evaluator-only `U_s`, `E_uv`, exact denominators, nonfinite 규칙이 있다. |
| A3 planning | **REVISE** | 수치 action/cost/regret/success는 닫혔으나 V2-P0-b를 고쳐야 동일한 origin과 cost input이 보장된다. |
| A4 leakage/API | **REVISE** | evaluator-after-hash와 taint gate는 강하다. typed request 보완과 V2-P0-d의 split-local lock이 필요하다. |
| A5 provenance | **PASS** | `:797-844`의 17개 integer invariant는 모두 exact zero이며 attempt/success가 분리됐다. |
| A6 thresholds | **REVISE** | selector는 train-only/unique지만 P0-aㆍP0-c가 calibration population을 완전히 정의해야 한다. |
| A7 common budget | **PASS** | numeric call/capacity/allocation equality가 닫혔다. |
| A8 comparisons | **PASS** | benefit sign, ratio-of-means, positive denominator, strict win/tie, t-value, relative 2%, no-synergy가 고정됐다. |
| A9 controls/P1 | **PASS** | `:518-562,705-748`에 shuffled, zero-q, lesion, zero-slot, absolute cap, persistence, lure upper CI/max, cross-context zero가 있다. |
| A10 source boundary | **PASS for V2; P1 doc follow-up** | `:38-45,1011-1012`는 Schapiro 결과를 생물학 gate로 쓰지 않는다. 기존 `20-audit.md:63-76`의 문서 42 오서술 수정은 출판 전 별도 P1이며 V2 구현 결과를 바꾸지 않는다. |

## 5. Claim boundary

- R1 고정은 **[공리: 모델 선택]**이며 정리ㆍ산출ㆍ성능 근거가 아니다.
- C1 marginal LTM, C2 두 matched dream effect, C3 M11/M00 planning은 P0 수정
  후에만 **[예측]**으로 닫힌다. 현재는 각각 결과 미실행이며 C2/C3의 실행
  명세가 일부 **[미완성]**이다.
- C4는 명시된 capability/taint/provenance channel에 대한 **[안전 예측]**이다.
  hard-zero test가 모든 가능한 누설 원인의 보편적 부재를 정리로 증명하지는 않는다.
- C5는 **[공리: 보고 규칙]**이다. 모든 metric-specific interaction을 보고하며
  generic `synergy` 결론은 등록되지 않았다.
- dimensionless `PASS` (`13-dimensionless-v2.md:5-37`)는 단위ㆍcoordinate
  consistency만 뜻한다. C1--C4의 진리나 자연ㆍ생물학적 타당성을 뜻하지 않는다.

## 6. Seedㆍ실행 장부와 계수

입력 문서들은 V2 registered/development seed 미실행을 일관되게 기록한다
(`00-contract-v2-draft.md:15-16`, `11-math-v2.md:13`,
`13-dimensionless-v2.md:7-9`). Dimensionless lane은 일반 unit test/checker를
실행했다고 별도 공개했다(`13-dimensionless-v2.md:39-48`); 이는 등록 seed,
calibration 또는 과학 결과 실행이 아니다. 허용된 입력에서 등록 artifact가
열렸다는 반대 증거는 없다.

- 감사한 상위 claim: **5** (C1--C5)
- 결과로 성립한 정리/산출: **0 / 0**
- 반영 감사한 math P0: **7** (PASS 5, REVISE 2; P0-7은 P0-3 종속)
- 새로 남은 executable P0: **4**
- A1--A10: **PASS 7, REVISE 3**
- 즉시 삭제할 부모 claim: **0**; 수정 전 구현ㆍ등록 seed 개시만 금지

최종 판정: 네 P0 수정과 새 raw SHA/fresh lock 전에는 구현하지 않는다.
수정 뒤 같은 입력 범위로 짧은 최종 재감사를 수행한다.
