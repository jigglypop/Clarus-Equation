# publication-native event 복원 경로

Status: COMPLETE

## 결과 전 경로

| 순위 | 경로 | 현재 상태 | matched falsifier / STOP | 입장 상한 |
|---:|---|---|---|---|
| 1 | R3a OSF provider tree | `PASS_ACQUISITION_CANDIDATE` | full tree·immutable byte receipt 또는 deterministic selection 실패 | acquisition evidence |
| 2 | R3b Zenodo `pumpprobe` + OSF txt | `BLOCKED_SOURCE_INDEX_JOIN` | 26 incomplete sessions와 827 blank local-label joins | `PARTIAL_SOURCE_INDEX_SCHEMA / OBSERVATIONAL_SCHEMA_ONLY` |
| 3 | R1b frozen NWB conversion mapping | `STOP_FIELD_LOSS` | native event·sentinel·assignment receipt가 converter에서 보존되지 않음 | converter schema corroboration only |
| 4 | R2 geometric validation | `BLOCKED_SOURCE_JOIN` | frame·unit·transform·radius·ambiguity·tie rule receipt가 없음 | 새 immutable validation object가 있을 때만 재개 |

## R3a

**[산출]** OSF 공식 API는 678-object `raw_extracted_data` tree와 stable file IDs/download
route를 제공한다. 구현 레인은 모든 metadata page를 동결하고 object family count와
provider checksum을 검증한다. `gcamp` family는 manifest에는 남기되 bytes를 받지 않는다.

## R3b

**[산출]** predeclared non-outcome txt family의 actual bytes와 Zenodo `pumpprobe-1.1.zip`
source를 사용해 다음을 검사했다.

1. session별 six-family completeness와 unique numeric prefix;
2. `stim_neurons`, `stim_volume_i`, clock/event cardinality와 missing/sentinel domain;
3. `labels[stim_neurons]` index validity와 identity string provenance;
4. automatic/manual, failure와 exclusion field의 존재 여부;
5. source code가 해당 field를 읽고 쓰는 정확한 경로.

113개 세션 중 87개가 다섯 family를 모두 가졌고, 이 안의 4,457개 stimulation row에서
3,537개 nonnegative index join과 2,710개 nonblank local-label join을 얻었다. cardinality,
index domain, clock failure는 0이었지만 26개 세션이 불완전하고 827개 join label이
blank이므로 전체 `PASS_SOURCE_INDEX_JOIN`은 false다. 보존되는 좁은 지위는
`PARTIAL_SOURCE_INDEX_SCHEMA / OBSERVATIONAL_SCHEMA_ONLY`다. canonical identity의
confidence/provenance와 assignment receipt는 성립하지 않는다.

## R1b

**[산출]** frozen converter에서 `TargetPlaneSegmentation`, NeuroPAL 및 label/confidence/
comment schema, manual target 입출력, failed target의 `NaN` 표현은 확인했다. 그러나 native
`stim_neurons`/`stim_volume_i`, event sentinel `-1/-2/-3`, original assignment row의 보존은
확인되지 않았다. 따라서 이 mapping은 `STOP_FIELD_LOSS`다. native event·sentinel·manual/
complementary label·failed flag·canonical confidence/provenance·order/timing/dose가 event row로
보존된 frozen export가 있을 때만 재개한다. outcome을 읽어 matching하면 apparatus invalid다.

## R2

**[미완성]** 현재 조사 범위에는 독립 validation object가 없어 `BLOCKED_SOURCE_JOIN`이다.
outcome과 독립적인 immutable object가 coordinate frame, pixel/µm/z unit과 calibration,
registration transform, matching radius, multiple-candidate ambiguity rule, deterministic
tie-break, confidence/provenance를 함께 제공할 때만 재개한다. response-derived match는
`APPARATUS_INVALID`다.

## 자유도 규율

R3a는 predeclared filename token, byte size, canonical path 순서를 따른다. R3b/R1b는
publication Zenodo release와 이미 동결한 commit/tree를 쓴다. R2에는 하나의 transform/
radius/tie rule만 허용한다. response를 본 뒤 threshold, radius, source partition,
endpoint, decoder나 seed를 바꾸는 반복은 새 경로가 아니다.
