---
description: 활성 CE run 현황 보고
---

활성 CE run의 현황을 보고하라.

1. `_workspace/ce/.active-run`에서 run 경로를 읽는다. 없으면 direct child run 디렉터리를 읽기 전용으로 나열하며, 탐색 목적으로 `gc`를 실행하지 않는다.
2. `.claude/hooks/run.cmd status <run-dir>` 출력만으로 단계별 상태·Gate·수정 카운트를 보고한다. stage 파일을 다시 읽지 않는다.
3. 다음에 해야 할 최소 단계를 한 줄로 제안한다.
