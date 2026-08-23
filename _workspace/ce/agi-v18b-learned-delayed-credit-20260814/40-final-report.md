# AGI V18b reward-decoded delayed linear credit: final report

Status: ABANDONED

## 사후 봉인 (retroactive closure)

이 보고서는 2026-08-23에 작성된 사후 봉인이다. 구현·검증·confirmation을 새로
수행하지 않았고, 존재하는 stage 파일과 문서화된 외부 기록의 상태만 기록한다.
**이 보고서는 새 증거를 만들지 않았다.**

## 계약 질문

marked cue를 긴 distractor delay 너머로 유지하고, 학습자 자신의 binary action과
나중의 binary reward를 결합해 supervision을 복원하며, 이를 persistent linear
classifier에 누적해 coordinate cue에서 unseen dense composition으로 일반화할 수
있는가 (결정론적 합성 credit assignment; `AGI GO` 금지).

## 선행 관계

PREDECESSOR `_workspace/ce/_archive/agi-v18-learned-delayed-credit-20260814`는
E3 pathwise quantifier가 거짓으로 확정되어 ABANDONED로 닫혔다 (해당 run의
40-final-report). V18b는 그 후속 요건(전체 trajectory paired coupling 등)을
계약으로 세운 수리판이다. 보존 정리($a(2R-1)=y$, $w=\theta$ 복원)는 V18 보고서에
이미 기록되어 있고 여기서 재서술하지 않는다.

## 이 run에 존재하는 것 (있는 그대로)

- `00-contract.md`, `11-math.md`, `12-routes.md` 완결; `10-sources.md`는 정당화된
  skip.
- `20-audit.md`: `Gate: PASS` (P0 0, P1 0) — 단 좁은 등록 구현만 인가하며, 감사
  시점에 "confirmation seed도 confirmation artifact도 존재하지 않는다"고 명문.
- `artifacts/run_v18b_benchmark.py`: benchmark 스크립트 1건. 결과 JSON 없음.
- `30-implementation.md`, `31-validation.md`: **존재하지 않는다.**

## ABANDONED 근거

1. **run 단계 산출 증거 부재**: Gate PASS 이후 이 run 디렉토리에 구현·검증 stage와
   development/confirmation 결과 artifact가 기록되지 않았다.
2. **봉인 파손의 문서 기록**: 저장소에 `tests/test_v18b_benchmark.py`는 존재하나,
   2026-08-15의 독립 실행 기록
   (`agi-frontier-comparison-20260815/artifacts/internal-validation.md`)은
   `4 failed, 9 passed`를 보고한다 — production 모듈 이름의 hash suffix 불일치와
   sealed manifest byte 불일치로 "unit implementation은 존재하나 현재 유효한
   sealed confirmation 결과는 없다"(해당 파일 명문). 즉 run 밖에서 구현이
   진행됐으나 계약의 봉인 체계로 회수되지 못했다.
3. **후속 부재**: `docs/7_AGI/`에 V18b 봉인 문서가 없고, 이후 run들은 V18b를 범위
   제외로만 언급한다 (2026-08-23 검색 확인).

## 재개 조건

- 새 계약으로만 재개한다: 20-audit의 봉인 의무(격리 로딩, 정확 manifest, 2차
  무결성 감사)를 다시 동결하고, 현재 트리의 test/production 모듈 봉인 불일치를
  복구한 뒤, fresh development seed로 30/31을 기록한다.
- 기존 confirmation seed 블록은 개봉된 적이 없으며 봉인 상태를 유지한다.
