# 자기선택 recurrent deformation (M4-R): 최종 보고서

Status: COMPLETE

## 사후 봉인 (retroactive closure)

이 보고서는 2026-08-23에 작성된 사후 봉인이다. 실험, 계산, 감사를 새로 수행하지
않았으며, 기존 stage 파일(`20-audit.md`, `31-validation.md`)의 판정을 요약만 한다.
**이 보고서는 새 증거를 만들지 않았다.**

## 계약 질문

외부에서 정답 recurrent matrix를 계산해 설치하지 않고, 경험한 cue/value trajectory의
terminal error와 local trace만으로 후보 deformation을 생성한 뒤 native internal
rollout으로 선택하는 동일 규칙이, Loop 8 zero-store binding과 Loop 9 held-out
composition을 모두 통과하는가.

## 감사 판정 (20-audit.md 그대로)

- Status: COMPLETE, `Gate: REVISE` — Revision 2 (fold semantics) 재감사가 보류된
  상태의 게이트다. PASS가 아니다.
- 인가 범위는 M4-R basic과 unconditional controls, focused source tests,
  formula-discovery seeds `97401..97408`뿐. development-validation과 confirmation
  seed는 봉인.
- fold는 max-scale 75% 또는 instability trigger가 machine receipt에서 성립할 때만
  Revision 2로 열 수 있다.

## 검증 결과 (31-validation.md 그대로)

- focused test `1 passed`; `git diff --check` 통과. discovery는 인가된 seed
  `97401..97408`만 사용했고 validation/confirmation 미개봉이 출력으로 확인됨.
- Discovery 결과: Loop 8 basic task gate 8/8이지만 모든 seed에서
  `min_control_advantage = 0`; Loop 9 basic task gate 4/8이고 advantage는 한 번도
  양수가 아니었다. 즉 **selection-causal-gate 실패**이며 source parse/test 실패가
  아니다.
- fold receipt에 instability 없음; max-scale trigger는 Loop 8 seeds 97401, 97406,
  97407과 Loop 9 seeds 97402, 97407에서 참. fold는 Revision 2 수학/감사 전까지
  비활성으로 남았다.

판정: discovery 단계에서 자기선택 규칙은 matched control 대비 이득을 만들지
못했다. 어떤 경로 승격도 없다.

## 원장 인용

2026-08-23 기준 `_workspace/ce/brain-algorithm-route-ledger.md`에는 이 run을
인용하는 행이 없다 (run 이름·M4-R·seed 범위로 검색해 확인). 이 run이 계약의
PREDECESSOR_EVIDENCE로 인용한 원장 제약(M1 Loop 8 성공과 T1 Loop 9 실패의 동시
설명 의무, threshold/seed/decoder 재조정 금지)은 원장에 그대로 유효하다.

## 미완 항목

- BLOCKED: Revision 2 fold semantics 재감사 미실행. 재개 조건 — 보존된 fold
  receipt(max-scale trigger)를 근거로 fold 수학을 동결하고 독립 재감사를 통과한
  새 revision.
- development-validation seed와 confirmation seed는 미개봉 봉인 상태로 남는다.
  discovery의 causal-gate 실패가 해소되기 전에는 개봉 근거가 없다.
