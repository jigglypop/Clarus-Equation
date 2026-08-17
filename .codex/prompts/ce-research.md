`$ce-research` 스킬로 명시적으로 연구 등급인 요청만 full run으로 수행하라.

주제: $ARGUMENTS

1. `.codex/hooks/run.cmd init _workspace/ce/<주제-슬러그>-<YYYYMMDD>` 로 run을 만들고, REUSE? 목록이 나오면 기존 run을 이어받는다.
2. 00-contract.md에 질문·정의역·주장·기호·허용 오차를 고정한다.
3. 독립적인 source/math 레인은 필요할 때 병렬 실행하고, 구현을 멈춘 안정 스냅샷을 한 번 감사한다.
4. 감사 Gate PASS 후 승인된 구현·집필만 진행한다.
5. 마지막 메시지에 `CE_RUN=_workspace/ce/<run-id>` 한 줄을 남긴다.
