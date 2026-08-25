연구 세션을 정리하라.

1. `.codex/hooks/run.cmd gc _workspace/ce`를 실행한다.
2. `ARCHIVED`, `STALE`, `REFERENCED`, `PINNED`, `COLLISION` 목록을 구분해 보고한다. 참조를 자동 재작성하거나 archive 증거를 자동 삭제하지 않는다.
3. STALE run마다 이어받을지, 40-final-report.md에 `Status: ABANDONED (사유)`를 쓸지 한 줄씩 제안한다. 방치를 기본값으로 두지 않는다.
4. 정확히 식별한 컴파일 target과 `__pycache__`처럼 재생성 가능한 cache만 실제 경로가 `_workspace/ce` 아래인지 확인한 뒤 별도로 삭제한다. stage 문서와 artifact는 cache로 추정하지 않는다.
