활성 CE run의 현황을 보고하라.

1. `_workspace/ce/.active-run`에서 run 경로를 읽는다. 없으면 `hooks/run.cmd gc _workspace/ce`로 미완성 run을 나열한다.
2. `hooks/run.cmd status <run-dir>` 출력만으로 단계별 상태·Gate·수정 카운트를 보고한다. stage 파일을 다시 읽지 않는다.
3. 다음에 해야 할 최소 단계를 한 줄로 제안한다.
