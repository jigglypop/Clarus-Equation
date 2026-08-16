# Status audit — G9-CBM V1

Status: COMPLETE

## 1. 감사 범위와 결론

대상은 `00-contract.md`, `10-sources.md`, `11-math.md`, `12-routes.md`이다.
구현ㆍ교정ㆍvalidationㆍtest seed는 실행하지 않았다. 수치 성능의 PASS/FAIL도
판정하지 않았다.

`00-contract.md:117-122`의 자체 중지 조건과 `ce-closure-gate`에 따라 현재
계약은 구현 전 단계에서 닫히지 않았다. C1--C5는 모두 시험 가능한 명제로
고정되지 않았고, `11-math.md:48-55`의 P0-1--P0-8은 아직 계약에 반영되지
않았다. R1은 유망한 설계 제안이지만 동결 계약의 수정본이 아니다.

실행 판정: **구현ㆍ등록 seed 개방 금지**. 계약 변경은
`00-contract.md:105-107`에 따라 G9-CBM V2와 fresh seed를 요구한다.

## 2. C1--C5 형식 지위

| Claim | 파일:줄 | 현재 표기 | 실제 지위 | 판정과 필요한 조치 |
|---|---|---|---|---|
| C1 | `00-contract.md:58`, `:74-75` | 예측 | **미완성** | “marginal LTM”과 `M11-M00` joint contrast가 다르다. `11-math.md:100-131`의 `RR_L`, marginal paired CI, strict-win gate로 교체하고 joint contrast는 `G_joint`로 별도 보고한다. |
| C2 | `00-contract.md:59-60`, `:76-77` | 예측 | **미완성** | evaluator-only unseen-valid 집합과 분모가 없다. `11-math.md:133-148,218-231`처럼 두 matched dream contrast를 각각 판정하고 `U_s`를 후보에게 비공개로 고정한다. |
| C3 | `00-contract.md:61-62`, `:78-81` | 예측 | **미완성** | do-action 생성기, 후보 구성, 비용, regret, success가 없다. `11-math.md:150-168,260-320`을 수치까지 동결하지 못하면 C3와 모든 counterfactual/planning 문구를 삭제한다. |
| C4 | `00-contract.md:63-64`, `:82-87` | 안전 예측 | **미완성** | 금지사항은 있으나 capability 경계ㆍchronologyㆍpoison testㆍcell isolation이 없다. `11-math.md:322-380,382-438`의 API 시험과 hard-zero provenance count를 계약화해야 한다. |
| C5 | `00-contract.md:65-67`, `:88-89` | 보고 규칙 | **미완성** | interaction의 척도ㆍ부호ㆍmetric familyㆍ2% 의미가 없다. `11-math.md:60-98,170-185`의 benefit-oriented interaction과 relative no-antagonism 식을 고정해야 산출 가능하다. |

현재 성립한 정리나 산출은 없다. C1--C5는 가설로서 삭제할 필요는 없지만,
현 계약으로 검증됐다는 해석은 허용되지 않는다.

## 3. P0-1--P0-8과 R1 정합 감사

| ID | 상태 | R1이 제공한 것 | 남은 결함 / 판정 |
|---|---|---|---|
| P0-1 | **미해결** | R1은 LTM과 dream의 역할을 `q`와 `s(k)`로 분리한다 (`12-routes.md:203-247`). | `12-routes.md:7-12`도 joint `M11/M00` 목표를 유지한다. 역할 직교성은 marginal estimand가 아니다. C1 gate를 반드시 교체한다. |
| P0-2 | **부분 제안, 미해결** | typed API와 action core 초안이 있다 (`12-routes.md:64-106,159-201`). | `d,m,K`, 상태ㆍ행동 범위, 생성 전이, 후보 생성, goal/cost/success/invalid penalty가 수치로 동결되지 않았다. |
| P0-3 | **미해결** | H5를 H20 slice로 제한한다 (`12-routes.md:225-240`). | NRMSE normalizer, `E_all`, `E_uv`, invalid-rate 세 분모, zero/nonfinite 규칙이 없다. |
| P0-4 | **부분 제안, 미해결** | evaluator-after-return API와 poison 불변성을 제안한다 (`12-routes.md:64-112`). | 계약ㆍ테스트 SHAㆍchronologyㆍ24 cell-order 불변성ㆍcapability/taint 계측으로 아직 동결되지 않았다. |
| P0-5 | **부분 제안, 미해결** | G7-M V2 provenance 경계와 synthetic identity 금지를 재사용한다 (`12-routes.md:114-157`). | 계약의 `synthetic tagged recalled <=0.01`과 insert 의미는 그대로다. 위반 count, insert attempt, successful insert를 각각 정수 0으로 바꾼다. |
| P0-6 | **부분 제안, 미해결** | R1은 공통 train-derived threshold 3개와 route-specific scalar 0을 제안한다 (`12-routes.md:190-195,249-265`). | selection 알고리즘, planner/success/penalty 값, LTM/dream absolute gate가 계약에 없다. |
| P0-7 | **부분 제안, 미해결** | inert store와 no-op dream buffer/call padding을 제안한다 (`12-routes.md:149-152,267-279`). | `b_registered`의 모든 정수ㆍbyte capㆍhash equality가 없다. “same budget”은 아직 검증 불가다. |
| P0-8 | **미해결** | positive interaction을 가정하지 않는다고 적었다 (`12-routes.md:244-247,294-296`). | interaction sign, ratio-of-means, denominator invalidity, strict tie, multiplicity, relative 2% 식이 동결되지 않았다. |

따라서 R1은 **[예측: 선택된 구현 경로]**일 뿐 [정리]나 구현 허가가 아니다.
R1을 V2의 단일 route로 사전 고정하는 것은 허용되지만, R2--R4를 train에서
비교해 고르는 것은 `12-routes.md:262-265`에 따라 별도 selection protocol과
fresh seed 없이는 허용되지 않는다.

## 4. 필수 계약 개정 10건

1. **A1 — factorial estimand:** C1을 `11-math.md:100-131`의 marginal LTM 식으로 교체하고 두 simple effect도 보고한다.
2. **A2 — metric:** train-only `mu,q`, `E_all`, evaluator-only nonempty `U_s`, `E_uv`, H5=H20 prefix, nonfinite hard-fail을 `11-math.md:188-240`대로 고정한다.
3. **A3 — planning:** `p,m,H=20,K`, action bounds/order, public goal, dimensionless cost weights, `P_invalid`, success threshold, tie rule, do-transition와 shared-noise coupling을 `11-math.md:260-320`대로 고정한다.
4. **A4 — leakage:** append-only wake chronology, 최소 candidate API, evaluator-after-hash order, future/latent/validity/metadata/masked-cue poison tests, H5 prefix, cell-order/input-hash 불변성, pre-unlock read denial을 계약과 test에 명시한다.
5. **A5 — provenance:** `11-math.md:384-410`의 provenance tuple 4종과 hard invariant 9종을 seed/cell별 정수 0으로 요구한다. insert attempt와 success를 분리한다.
6. **A6 — recall/thresholds:** `|P_s|,|L_s|`, coverage, identity, `wrong_all`, `wrong_given_accept`, lure rate와 exact train-only threshold/join/blend selection을 `11-math.md:412-477`대로 고정한다.
7. **A7 — budget:** `11-math.md:480-516`의 numeric budget vector, inactive padding, caps, call/update counts, core/input hashes를 모두 등록한다.
8. **A8 — comparisons:** positive-benefit interaction sign, ratio-of-means, positive denominator, strict `>0` wins, ties, CI critical values, relative no-antagonism 식, synergy metric/multiplicity를 고정한다.
9. **A9 — controls/P1:** shuffled-binding와 unconstrained lesion의 paired decision rule, absolute H20 comparator/cap, one-calibration 조건부 추론, `upper95(false_lure)<=0.05`, max lure, cross-context recall=0을 채택하거나 해당 기제 귀속을 삭제한다.
10. **A10 — source boundary:** Schapiro 서술을 아래 §5대로 수정한다. 이는 계약 수치나 G7-M PASS/FAIL을 바꾸지 않는다.

## 5. Schapiro P1 범위

`10-sources.md:13,43-44`가 확인한 불일치는 **P1 문헌 서술만의 결함**이다.
문서 42 §6의 “sleep이 wake보다 일반화를 높이지 않았다”는 표현은 다음으로
교체해야 한다.

> Experiment 1에서는 sleep 뒤 shared-property memory가 개선되고
> unique-property memory가 보존됐으며, Experiment 2의 nap 이득은
> lower-frequency category에 한정됐다. 이 연구는 dream 자체나 보편적
> consolidation 이득을 검증하지 않았다.

이 수정은 G7-M V1 validation FAIL, G7-M V2 validation/test PASS, interaction
`0`, lure/provenance 수치, G9-CBM C1--C5 판정에 영향을 주지 않는다. S05를
부정 대조로 계속 쓰려면 삭제하고 별도의 검증된 1차 null study로 대체한다.

## 6. 개정 불가 시 조건부 삭제ㆍ축소 8건

1. C3, “counterfactual action selection”, planning, regret, success, invalid-action lure를 함께 삭제한다.
2. “without increasing false recall”을 삭제하고 “within preregistered ceilings”로 바꾼다. 원문을 유지하려면 lure false recall이 정확히 0이어야 한다.
3. LTM/dream 절대 gate를 열거하지 못하면 “both absolute component gates”를 삭제한다.
4. 일반적 “synergy”를 삭제하고 사전 선택된 metric-specific interaction만 보고한다.
5. 실제 frozen world/transition component를 갱신하고 H20에 영향을 보이지 못하면 dream path를 “known-slot constrained schema completion”으로 축소한다.
6. inherited hard completion을 attractor로 부르거나 G9-CB V5--V7 FAIL을 통합 지지 증거로 재표기하는 문구는 허용하지 않는다.
7. inactive padding을 채택하지 않으면 memory-content effect를 삭제하고 “component-plus-resource bundle effect”로 바꾼다.
8. shuffled/lesion control이 diagnostic이면 memory-content 및 constraint-mechanism 인과 귀속을 삭제한다.

## 7. 감사 계수

- 검사 비자명 명제: **5** (C1--C5)
- 정의/보고 요구: **5** claim label 중 C5 1건은 보고 규칙이며, 시험 명제로는 미고정
- 성립 정리: **0**
- 성립 산출: **0**
- 실제 `[미완성]`: **5**
- P0 결함군: **8** (미해결 3, 부분 제안이나 미동결 5)
- P1 요구: **7** (수학 6 + Schapiro 문헌 경계 1)
- P2 보고 요구: **4**
- 숨은 공리군: **15** — state type/unit; action type/bounds/scale; do-transition/noise; chronology; candidate construction/order; public goal; cost/weights/penalty; success threshold; metric normalizer; unseen-valid set; invalid/nonfinite denominators; threshold selection; leakage/cell isolation; budget ledger; interaction/aggregation/tie/multiplicity
- 검토된 반례ㆍ경계 계산: **7** (`11-math.md:569-639`)
- 즉시 삭제된 부모 주장: **0** — C1--C5는 아직 가설로 보존
- 개정 불가 시 조건부 삭제ㆍ축소군: **8**

## 8. 최종 판정

감사는 완료됐지만 계약은 실행 가능하게 닫히지 않았다. P0-1--P0-8을 모두
반영한 G9-CBM V2 preregistration, fresh seeds, implementation/tests/hash lock이
완료되기 전에는 train/calibrationㆍvalidationㆍtest를 열 수 없다.

Gate: **BLOCKED**
