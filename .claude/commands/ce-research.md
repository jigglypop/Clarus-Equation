---
description: CE full 모드 연구 run (6단계 전체)
---

/ce-research 스킬로 full 모드 연구 run을 수행하라.

주제: $ARGUMENTS

1. `.claude/hooks/run.cmd init _workspace/ce/<주제-슬러그>-<YYYYMMDD>` 로 run을 만들고, REUSE? 목록이 나오면 기존 run을 이어받는다.
2. 00-contract.md에 질문·정의역·주장·기호·허용 오차를 고정한다.
3. 뇌·기억·의식 주제라면 먼저 `_workspace/ce/brain-algorithm-route-ledger.md`와 선행 run 12/31, 존재하는 40을 읽는다. 40이 없으면 마지막 numbered audit와 closure 부재를 쓰고, 이전 결과·퇴역 경로·후보 선택을 계약에 고정한다.
4. 독립적인 source/math 레인은 필요할 때 병렬 실행하고, 구현을 멈춘 안정 스냅샷을 한 번 감사한다.
5. 감사 Gate PASS 후 승인된 구현·집필만 진행한다. subagent는 Git을 발행하지 않고 main에 변경 manifest를 인계한다.
6. 마지막 메시지에 `CE_RUN=_workspace/ce/<run-id>` 한 줄을 남긴다.
