# 10. PreEq LLM Manifest Verifier

이 문서는 후보 답변을 evidence manifest로 검증하는 조건부 구현 규약이다. 독자는 claim 분해, NLI와 precision/recall을 안다고 가정한다. 입력은 후보·근거·threshold이고 출력은 답변 또는 보류 판정이며, 성공은 synthetic과 실제 holdout에서 false-allow를 낮추는 것이고 실패는 manifest 누락·오탐·분포이동을 감추는 것이다.

## 목표

목표는 낮은 defect energy의 후보만 복원하는 것이며 낮은 점수가 사실성의 완전한 증명은 아니다.

LLM 후보 답변을 등호 이전 후보 상태로 보고, 근거 불일치 defect energy가 낮은 답변만 manifest answer로 선택한다. 이 문서는 `reality_stone.clarus.llm_pre_eq`, Rust/PyO3 `nn_llm_pre_eq_fwd`, `examples/pre_eq/llm_manifest_verifier.py`의 검증 기준을 고정한다.

> 지위 갱신: 이 문서의 count 기반 defect는 v0 smoke-test proxy다. `paper/4_공학적_활용/09_무차원_잔차장_환각억제.md`와 `paper/9_등호이전/01_공리와증명.md`를 모두 반영한 v2 수식 정본은 [11_CE_claim_residual_verifier_formula.md](11_CE_claim_residual_verifier_formula.md)에 둔다.

## 수식

아래 수식은 score의 정의역과 무차원 정규화를 고정한다. scale·evidence extractor가 바뀌면 threshold도 독립적으로 재검증해야 한다.

후보 답변 $y_i$와 evidence $E$에 대해

$$
p_\beta(y_i\mid E)
:=
\frac{\mu_0(y_i)\exp(-\beta\Delta(y_i,E))}
{\sum_j\mu_0(y_j)\exp(-\beta\Delta(y_j,E))}.
$$

$\mu_0$는 모델 prior 또는 후보 생성 prior이고, $\Delta$는 verifier defect다. 유한 $\beta$에서는 posterior MAP를 선택한다. $\beta=0$이면 posterior는 prior로 돌아가고, $\beta\to\infty$에서 최소 defect 후보로 농축된다.

## Claim-level fold

atomic claim 분해는 검증 단위를 만들지만 분해 오류가 미탐을 만들 수 있다. label policy는 자료를 보기 전에 고정하고 인간 audit으로 반증해야 한다.

실제 RAG/QA에서는 답변을 atomic claim으로 쪼갠 뒤 각 claim을 세 라벨 중 하나로 접는다.

| label | 의미 | defect 반영 |
|---|---|---|
| `supported` | evidence가 직접 지지 | coverage defect 감소 |
| `unsupported` | evidence에 근거 없음 | unsupported defect |
| `contradicted` | evidence와 충돌 | contradiction defect |

`ClaimAudit.to_candidate()`가 claim label들을 `CandidateAnswer`의 defect count로 접고, 이후 같은 PreEq posterior 수식에 들어간다.

현재 구현된 defect energy는

$$
\Delta
:=
\frac{3C_{\rm contradicted}+C_{\rm unsupported}}
{\max(1, C_{\rm supported}+C_{\rm unsupported}+C_{\rm contradicted})}
+\mathbf{1}_{T=0}
+\frac{0.2}{1+C_{\rm supported}}
+2C_{\rm instruction}
+2C_{\rm self}
+0.25C_{\rm uncertainty},
$$

여기서 $T=C_{\rm supported}+C_{\rm unsupported}+C_{\rm contradicted}$다. 음수 support credit은 제거했다. supported claim은 보상항이 아니라 coverage defect를 낮추는 방식으로만 작용한다.

수치 커널은 Rust/PyO3의 `nn_llm_pre_eq_fwd`가 있으면 Rust에서 energy와 posterior를 계산하고, Rust extension이 없으면 같은 수식의 numpy fallback을 사용한다.

## Manifest 판정

세 조건은 선택한 verifier의 accept 규칙이며, 하나라도 약하면 보류·검색·질문으로 개입해야 한다.

`PreEqVerifier`는 다음 세 조건을 모두 통과할 때만 답한다.

| 조건 | 의미 |
|---|---|
| $E_{\min}\le E_{\max}$ | 모든 후보가 너무 나쁘면 abstain |
| $\Delta_2-\Delta_1\ge g_{\min}$ | 1등과 2등 defect gap이 충분해야 함 |
| $\max_i p_\beta(y_i)\ge p_{\min}$ | posterior confidence가 충분해야 함 |

실패하면 답변 대신 `abstained=True`와 reason을 돌려준다.

## 현재 synthetic 검산

synthetic 검산은 코드 경로의 회귀 검사이지 실제 RAG 분포에서의 성능 증명은 아니다.

실행:

```powershell
uv run --extra dev python examples/pre_eq/llm_manifest_verifier.py
```

현재 결과:

```text
total 4
answered 4
abstained 0
correct 4
answer_rate 1.000000
accuracy_on_answered 1.000000
exact_accuracy 1.000000
hallucination_rate_on_answered 0.000000
baseline_accuracy 0.000000
baseline_hallucination_rate 1.000000
```

이 synthetic set은 high-prior hallucination 후보를 의도적으로 넣은 smoke test다. 즉 위 숫자는 실전 성능 주장이 아니라, verifier가 “fluent prior”보다 “evidence defect”를 우선하도록 작동하는지 확인하는 회귀 검산이다.

## Adversarial / noisy sweep

sweep은 noise·공격 조건에서의 오탐과 미탐을 함께 보고해야 하며 threshold 사후선택은 허용하지 않는다.

추가 sweep은 `llm_manifest_verifier_sweep.py`에 둔다.

```powershell
uv run --extra dev python examples/pre_eq/llm_manifest_verifier_sweep.py --mode adversarial --seed 20260621 --cases 1000
uv run --extra dev python examples/pre_eq/llm_manifest_verifier_sweep.py --mode noisy --seed 20260621 --cases 1000
```

`adversarial` mode는 high-prior hallucination이 항상 끼어 있는 prior-trap smoke test다.

| mode | seed | cases | best exact accuracy | answer rate | hallucination on answered | baseline accuracy |
|---|---:|---:|---:|---:|---:|---:|
| adversarial | 20260621 | 1000 | 0.916 | 0.999 | 0.083 | 0.000 |
| noisy | 20260621 | 1000 | 0.620 | 0.968 | 0.360 | 0.000 |

`noisy` mode는 더 어려운 synthetic 분포다. 정답에도 일부 unsupported/uncertainty defect가 있고, 오답도 일부 supported claim을 가진다. 5개 seed에서의 noisy 결과:

| seed | exact accuracy | answer rate | accuracy on answered | hallucination on answered |
|---:|---:|---:|---:|---:|
| 11 | 0.662 | 0.970 | 0.682 | 0.318 |
| 22 | 0.604 | 0.965 | 0.626 | 0.374 |
| 33 | 0.643 | 0.964 | 0.667 | 0.333 |
| 44 | 0.609 | 0.972 | 0.627 | 0.373 |
| 55 | 0.607 | 0.964 | 0.630 | 0.370 |

평균적으로 exact accuracy는 약 0.625, answered accuracy는 약 0.646, answered hallucination rate는 약 0.354다. 이 수치는 유한 $\beta$에서 posterior MAP를 선택하고 negative support credit을 제거한 뒤의 값이다. 공개 benchmark가 아니라 verifier 구조의 수치 회귀이며, 현재 noisy 분포에서는 아직 강한 verifier라고 주장할 수 없다.

## 테스트

테스트는 불변조건과 지정 입력의 재현성을 확인하며 새로운 사실의 entailment를 보장하지 않는다.

실행:

```powershell
uv run --extra dev python -m pytest tests/test_llm_pre_eq_verifier.py tests/test_llm_pre_eq_sweep.py -q
cargo test --manifest-path reality_stone/python/reality_stone/clarus/core/Cargo.toml llm_pre_eq
```

현재 결과:

```text
Python: 16 passed
Rust: 2 passed
```

테스트 항목:

| 테스트 | 확인 |
|---|---|
| low-defect selection | high-prior hallucination보다 grounded answer 선택 |
| small-gap abstain | 후보 defect가 거의 같으면 답하지 않음 |
| high-defect abstain | 모든 후보가 나쁘면 답하지 않음 |
| metric regression | baseline prior 선택 대비 hallucination 감소 |
| $\beta=0$ prior limit | posterior가 prior로 복귀 |
| large-$\beta$ concentration | 낮은 defect 후보로 posterior 농축 |
| defect decomposition | component 합과 clipped energy 일치 |
| claim fold | atomic evidence label이 candidate defect count로 접힘 |
| Rust/numpy parity | Rust kernel과 numpy fallback의 posterior 일치 |

## 다음 실험

실제 pipeline 실험은 candidate 생성·holdout·human audit과 비용을 사전등록해 false-allow 감소를 반증 가능하게 해야 한다.

1. 실제 RAG pipeline에서 후보 $N=8\sim16$개를 만든다.
2. 답변을 atomic claim으로 분해한다.
3. evidence retrieval로 `supported`, `unsupported`, `contradicted`를 채운다.
4. TruthfulQA, FEVER, HotpotQA, 내부 문서 QA에서 baseline과 비교한다.
5. 핵심 지표는 hallucination rate, abstention precision, answer rate, exact/F1, calibration error다.

SOTA 초과 주장은 위 4번 공개 benchmark에서 다음 조건을 만족할 때만 한다.

| 조건 | 기준 |
|---|---|
| 동일 base LLM | 같은 모델, 같은 retrieval budget, 같은 candidate count |
| 강한 baseline | greedy, self-consistency, RAG reranker, verifier reranker와 비교 |
| 비용 공개 | token/call 증가율과 latency 포함 |
| abstain 분리 | 답한 문제의 정확도와 전체 coverage를 함께 보고 |
| seed 반복 | 최소 3개 seed 또는 bootstrap confidence interval |

현재 지위는 `Tooling/Open test`다. synthetic 검산은 통과했지만, 실전 claim verifier와 benchmark 결과가 붙기 전까지 LLM 성능 향상으로 주장하지 않는다.
