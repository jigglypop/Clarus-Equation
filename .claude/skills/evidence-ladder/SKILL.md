---
name: evidence-ladder
description: 증거 등급 L0~L4를 매기거나 읽을 때, 어떤 원장 항목을 논문(paper/)에 인용해도 되는지 판단할 때, 원장 항목의 level 필드를 채울 때, 추측 카드 항목의 등급과 지위를 구분할 때 반드시 참조. judge와 paper-writer가 쓴다.
---

# 증거 사다리

등급은 객관 기준으로만 매긴다. `ledger.py`가 같은 표를 코드로 재구현해 `level`이
근거와 불일치하면 원장 검증이 실패한다(이중 점검). 이유: LLM의 자기 평가는 관대하다.

| 등급 | 정의 | 판정 근거 (원장 필드) |
|---|---|---|
| L0 | 질문만 있음. 유도 없음 | `derivation` 없음 또는 파일 부재 |
| L1 | 유도는 있으나 기계검증 미실행 또는 실패 | `symbolic` ∈ {skipped, fail} 이고 `numeric` ≠ pass |
| L2 | 수치검증 통과, 기호검증 미통과/미실행 | `numeric` = pass 이고 `symbolic` ≠ pass. 기호 pass인데 반례가 있거나 survived_checks < 3이어도 L2 |
| L3 | 기호검증 통과 + adversary가 반례 못 찾음 | `symbolic` = pass ∧ `counterexamples` 비어 있음 ∧ `survived_checks` ≥ 3 |
| L4 | L3 + 선행문헌 대조 완료 + (선택) Lean 통과 | L3 ∧ `sourcer`가 mapping(실행됨) ∧ (`lean` = pass 또는 `lean_waived: true`) |

## 규칙

- **L3 이상만 논문에 인용 가능.** paper-writer가 L2 이하를 인용하면 안 되며,
  ko-academic-prose의 탈고 게이트가 이를 검사한다. 이유: 논문은 fail-closed다.
- sympy가 없는 환경에서는 `symbolic`이 항상 skipped이므로 최고 등급이 L2다. 2026-09-02 현재
  정책 허용 Python에 sympy가 있으므로 L3가 가능하다. 없어지면 같은 derivation을 재검증한다.
- `lean_waived`는 사람이 명시적으로 적을 때만 true다. 에이전트가 넣지 않는다.
- 등급을 낮추는 것은 언제나 허용, 올리는 것은 위 근거가 있을 때만.
- 원장 항목의 `level`은 judge가 근거 필드에서 계산해 적는다.

## 추측 카드 항목 (verdict adopt·refute)

카드 항목의 `derivation`은 카드 경로(`derivations/<Q>/F-NN.formula.md`)이고 `verification`은
카드의 verify 블록(극한 복원·차원 항등식) 결과다. 등급은 같은 표로 매기되 **뜻이 다르다**.

- 카드 L3 = "극한 복원·차원 항등식이 기호검증을 통과했고 adversary가 극한을 깨지 못함". 식이
  참이라는 뜻이 아니다. 논문 지위는 `[공리: 후보]`, 사전등록 숫자는 `[예측: 사전등록]`.
- 카드가 `[정리]`가 되는 것은 사다리 단이 모두 closed(각 단 L3)·cited일 때이고, 그 승격은
  paper-writer가 원장 항목들을 근거로 한다. 카드 항목 하나로 승격하지 않는다.
- adopt는 sourcer 실행이 필수이고 identical·special_case면 `validate`가 거부한다. 재발견은 등급이
  아니라 채택 여부를 결정한다.
- 외부기존 단(`ladder_cited`)은 등급이 없다. 문헌 참조로 닫히며 논문에서는 문헌을 인용한다.

## 인용 표기

논문에서 원장 항목을 인용할 때 각주에 항목 id를 남긴다: `[E-20260902-004]`.
이유: 나중에 등급이 바뀌면 추적해서 고쳐야 한다. 실제 예시는 `references/examples.md`.
