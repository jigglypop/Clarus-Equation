# 증거 등급별 예시 항목 (가짜 데이터)

모두 Q-TEST-1("Σ_{k=1}^n k = n(n+1)/2")을 대상으로 한 가상 항목이다. 필드는 ledger-format 스키마.

## L0

```yaml
id: E-20260902-001
question: Q-TEST-1
attempt: 1
level: L0
verdict: continue
derivation: null
verification: {symbolic: skipped, numeric: skipped, lean: skipped}
adversary: {counterexamples: [], survived_checks: []}
```
근거: derivation이 없다.

## L1

```yaml
level: L1
derivation: derivations/Q-TEST-1/attempt-01.derivation.md
verification: {symbolic: fail, numeric: skipped, lean: skipped}
adversary: {counterexamples: [], survived_checks: [dimension]}
```
근거: 유도는 있으나 기호검증 실패, 수치검증 미실행.

## L2

```yaml
level: L2
verification: {symbolic: skipped, numeric: pass, lean: skipped}
adversary: {counterexamples: [], survived_checks: [dimension, n_equals_1, n_equals_2]}
```
근거: 수치 통과, 기호 미실행(예: sympy 미설치). survived가 3이어도 symbolic ≠ pass면 L2.

## L2 (기호 통과지만 반례 있음)

```yaml
level: L2
verification: {symbolic: pass, numeric: pass, lean: skipped}
adversary:
  counterexamples:
    - {input: "n=0", expected: "0", observed: "정의역 밖", note: "claim이 n≥1을 명시하지 않음"}
  survived_checks: [dimension, n_equals_1, n_equals_2]
```
근거: 반례가 비어 있지 않으므로 L3 불가.

## L3

```yaml
level: L3
verification: {symbolic: pass, numeric: pass, lean: skipped}
adversary: {counterexamples: [], survived_checks: [dimension, n_equals_1, induction_step, large_n_numeric]}
sourcer: null
```
근거: 기호 통과, 반례 없음, survived ≥ 3. sourcer 미실행이라 L4 아님.

## L4

```yaml
level: L4
verification: {symbolic: pass, numeric: pass, lean: skipped, lean_waived: true}
adversary: {counterexamples: [], survived_checks: [dimension, n_equals_1, induction_step, large_n_numeric]}
sourcer:
  prior_art:
    - {ref: "Gauss, 초등 산술", relation: identical, note: "알려진 결과. judge는 promote 대신 park(known result)"}
```
근거: L3 + sourcer 실행 + 사람이 lean 면제. 단 relation identical이면 verdict는 park.
