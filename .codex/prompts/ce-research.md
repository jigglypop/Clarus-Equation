`$ce-research` 스킬로 명시적으로 연구 등급인 요청만 full run으로 수행하라.

주제: $ARGUMENTS

1. `.codex/harnesses/ce_research_docs_policy.md`를 먼저 적용한다. 정본은 저장소 루트 `docs/`로 고정한다. 같은 주제의 active·미완성·predecessor run과 루트 `docs`의 기존 정본을 찾고, 반례 수정·식 재유도·상세화·논문 개정이면 새 run을 만들지 않는다.
2. 후보 식의 반례가 있으면 `.codex/harnesses/ce_counterexample_pivot_loop.md`를 적용한다. 실패식을 음성대조군으로 고정하고 같은 run에서 구조 route를 사전등록해 `pivot`한다. 한 식의 실패만으로 부모 목표를 축소하지 않는다.
3. 하네스의 새 run admission gate를 통과한 경우에만 `.codex/hooks/run.cmd init _workspace/ce/<주제-슬러그>-<YYYYMMDD>`를 실행한다. `REUSE?`가 나오면 기존 run을 이어받는다.
4. run을 쓰면 00-contract.md에 질문·정의역·주장·기호·허용 오차와 최종 `CE_DOC=docs/<정본-경로>`를 고정한다.
5. 뇌·기억·의식 주제라면 먼저 `_workspace/ce/brain-algorithm-route-ledger.md`와 선행 run 12/31, 존재하는 40을 읽는다. 40이 없으면 마지막 numbered audit와 closure 부재를 쓰고, 이전 결과·퇴역 경로·후보 선택을 계약에 고정한다.
6. 독립적인 source/math 레인은 필요할 때 병렬 실행하고, 구현을 멈춘 안정 스냅샷을 한 번 감사한다.
7. 감사 Gate PASS 후 원장을 먼저 안정화한다. 논문·유도·자세한 설명은 `$ce-paper-write`로 기존 `docs` 정본에 단계별 유도, 반례와 수정식, 안정성, 수치·관측 판정, 미완성 다리까지 자세히 반영한다. 사용자가 다른 언어를 지정하지 않으면 정본 본문은 한국어로 쓴다.
8. subagent는 Git을 발행하지 않고 main에 run과 정본의 정확한 변경 manifest를 인계한다.
9. 마지막 메시지에는 run을 실제 사용했을 때만 `CE_RUN=...`을, 정본을 고쳤으면 항상 `CE_DOC=...`을 남긴다.
