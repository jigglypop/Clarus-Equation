# 경로 레인

Status: OPEN_SCHEMA_GATE

1. 000541의 가장 작은 source byte를 전부 받아 SHA-256을 검증한다.
2. response values를 집계하지 않는 schema walker로 NWB object paths, shapes, dtypes, units만 기록한다.
3. identity/position/calcium/stimulus receipt를 판정한다.
4. gate를 통과하면 가장 이른 3 worms에서 window와 A3를 동결한다.
5. development failure가 spectral instability이면 등록된 normalization 수정 1회만 허용한다.
6. 나머지 5 worms를 한 번 확인한다.

000565 정적 자산 실패는 `artifacts/source-revision-1.md`에 보존했다. 000541에서도 calcium/identity row join이 없으면 더 큰 파일을 임의 탐색하지 않고 `BLOCKED_INPUT`으로 멈춘다.
