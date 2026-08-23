# 구현 기록

Status: COMPLETE

## 범위

이번 구현은 response effect model이 아니라 apparatus admission probe다. 선행 run의
동결 manifest에서 결과 독립 최소 크기 규칙으로 이미 선택한 DANDI segmentation
asset `d076d282-162a-4946-a1c8-68e72b6cce54`를 다시 내려받았다. 실제 bytes와
SHA-256은 선행 receipt와 일치한다.

`artifacts/audit_source_join_schema.py`는 표준 라이브러리만 사용해 다음을 수행한다.

1. NWB bytes와 SHA-256을 동결값에 대조한다.
2. 선행 schema inventory의 SHA-256을 동결값에 대조한다.
3. event table, target table, target DynamicTableRegion과 response-side `neuropal_ids`
   schema만 검사한다.
4. target identity 열·외부 table reference와 assignment/control 열의 존재 여부를
   결정론적으로 판정한다.
5. response `data` 값 비열람, geometric matching 미수행과 endpoint 비선택을 JSON에
   기록한다.

## apparatus 경계

`h5py`를 쓰는 value-level probe는 시스템 Python에 dependency가 없어 실행되지 않았다.
Windows Application Control을 우회하거나 workspace 가상환경을 사용하지 않았고,
dependency 설치도 하지 않았다. 미실행 초안은 제거하고, 이미 SHA로 동결되어 검증된
선행 schema inventory를 입력으로 삼는 표준 라이브러리 probe로 범위를 좁혔다.

따라서 이번 구현은 target ROI index 값, fluorescence 값, source centroid distance,
identity match 결과 또는 neural effect를 읽지 않았다. explicit schema join의 부재와
event-level assignment receipt 부재만 판정한다.

## 산출물

| 파일 | SHA-256 |
|---|---|
| `artifacts/sub-24-segmentation.nwb` | `40e4a0daac128d9cba743eb80c1fbfdb3f647a739129f07342d330959aef532e` |
| `artifacts/audit_source_join_schema.py` | `597bc82aa7c3ff94f220cdd4ab410d0abe9077f820cbe41932eb9814645975e8` |
| `artifacts/source-join-schema-audit.json` | `26ed26e811d388ac15be61a27293b2dad7f4532c8a4099f9b049ccbe14a9cdc9` |

