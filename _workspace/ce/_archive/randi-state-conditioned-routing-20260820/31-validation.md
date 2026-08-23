# 기계 검증

Status: COMPLETE

## 환경

`.codex/hooks/python.cmd doctor`는 다음 시스템 interpreter를 선택했다.

- Python: `C:\Users\dongh\AppData\Local\Programs\Python\Python311\python.exe`
- version: `3.11.9`
- bytecode/cache: disabled
- repository `PYTHONPATH`: configured

연구 skill의 별도 `run.ps1` 래퍼는 환경에 없어 stage gate는 수동 검사했다.

## 실행 명령

```powershell
.codex\hooks\python.cmd python `
  _workspace\ce\randi-state-conditioned-routing-20260820\artifacts\audit_source_join_schema.py `
  --nwb _workspace\ce\randi-state-conditioned-routing-20260820\artifacts\sub-24-segmentation.nwb `
  --schema _workspace\ce\randi-neural-propagation-source-audit-20260820\artifacts\e2syt-exemplar-schema.json `
  --output _workspace\ce\randi-state-conditioned-routing-20260820\artifacts\source-join-schema-audit.json
```

## 결과

| gate | 결과 |
|---|---|
| NWB bytes | `PASS`: 1,273,970 |
| NWB SHA-256 | `PASS`: `40e4a0da...959aef532e` |
| schema SHA-256 | `PASS`: `45e53bb2...1df631433e54f` |
| target DynamicTableRegion | `PASS`: 세 reference 모두 `TargetPlaneSegmentation` |
| target identity column/reference | `BLOCKED`: 없음 |
| response-side NeuroPAL mapping schema | `PASS_SCHEMA`: 있음 |
| event assignment/control receipt | `BLOCKED`: 없음 |
| fluorescence `data` value read | `PASS_BOUNDARY`: false |
| geometric matching | `PASS_BOUNDARY`: false |
| effect/endpoint 계산 | `PASS_BOUNDARY`: false |
| 최종 machine status | `BLOCKED_EXPLICIT_SOURCE_JOIN / BLOCKED_ASSIGNMENT_RECEIPT` |

스크립트 source compile은 통과했다. 같은 입력·명령으로 JSON을 덮어쓴 전후 SHA-256은
모두 `26ed26e811d388ac15be61a27293b2dad7f4532c8a4099f9b049ccbe14a9cdc9`로
일치했다.

## 해석 경계

**[산출]** 이 검사는 frozen exemplar schema에서 source identity와 assignment receipt가
입장하지 못했음을 보인다. 전체 Randi/OSF native source에 해당 정보가 없다는 증명도,
상태-조건부 라우팅 가설의 반증도 아니다. response value를 읽지 않았으므로 예측력이나
개입 효과에 관한 수치 결론은 없다.
