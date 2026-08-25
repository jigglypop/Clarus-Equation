$ce-research 스킬과 `.codex/harnesses/ce_research_docs_policy.md`로 후속 작업을 수행하라.

대상: $ARGUMENTS

1. 같은 부모 주장의 반례 수정, 식 재유도, 재검산, 상세화와 논문 개정이면 새 light run을 만들지 않고 선행 run과 저장소 루트 `docs/`의 기존 정본을 이어 쓴다. 외부 복제본의 `docs`는 대체 정본으로 쓰지 않는다.
2. 반례가 있으면 `.codex/harnesses/ce_counterexample_pivot_loop.md`에 따라 실패식을 음성대조군으로 고정하고 같은 run에서 구조 route를 `pivot`한다. 한 식의 실패로 부모 목표를 줄이지 않는다.
3. 독립 successor가 하네스의 새 run admission gate를 통과한 경우에만 계약에 `PREDECESSOR: _workspace/ce/<선행-run>`과 `CE_DOC=docs/<정본-경로>`를 적는다.
4. 새 light run이 필요한 경우 레인은 math-verifier만 실행한다. 선행 run이 감사한 결론은 재유도하지 않고 경로만 인용한다.
5. 원장을 먼저 안정화한 뒤 논문·유도는 `$ce-paper-write`로 기존 `docs` 정본에 한국어로 자세히 반영한다. 사용자가 다른 언어를 지정한 경우에만 바꾸며, `_workspace` final report만으로 완료하지 않는다.
6. 마지막 메시지에는 실제 사용한 경우만 `CE_RUN=...`을 쓰고, 정본을 고쳤다면 `CE_DOC=...`을 남긴다.
