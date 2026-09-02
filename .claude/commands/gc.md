---
description: 문서·workspace 정리 점검 (읽기 전용 보고)
---

정리 점검을 수행하라. 삭제는 명시 요청 없이 하지 않는다.

1. `.claude\hooks\python.cmd links`와 `lint`로 깨진 링크·앵커·고아 문서·과정 서술·영어 비율 초과를 나열한다.
2. `.claude\hooks\python.cmd harness`로 `_workspace/` 상한(파일 수·크기·21일 무활동)을 검사하고 흡수·삭제 후보를 나열한다.
3. `paper/진전_원장.md` §5에서 30일 넘게 갱신되지 않은 항목과 §6에서 "다음 하나"가 비어 있는 트랙, `ledger/questions.yaml`의 escalated·parked 질문을 나열한다.
4. 대상·이유·제안(연결·흡수·보관·삭제)을 표로 보고한다.
