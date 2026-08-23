# Publication-native event 복원 사후 감사

Status: COMPLETE

Gate: PASS AFTER REVISION

Audited: 2026-08-20

Scope: outcome-blind OSF native text, publication-pinned `pumpprobe` source archive,
frozen NWB converter의 구현 후 안정 스냅샷

## 감사 결론

**[산출]** 구현은 계약의 acquisition 경계와 outcome 비열람 규칙을 지켰다. OSF의
허용 family 538개(9,346,585 bytes), Zenodo archive 1개, converter archive 1개를 합친
540 files / 10,681,846 bytes는 602 files / 32,000,000 bytes 상한 이하다. OSF frozen
manifest는 678 objects를 68 metadata pages에서 기록했다.

**[산출]** machine gate의 최종 판정은 타당하다.

| Gate | 결과 | 이유 |
|---|---|---|
| `PASS_SOURCE_INDEX_JOIN` | false | 113개 중 26개 session이 불완전하고 local-label join 827개가 blank |
| `PASS_SOURCE_JOIN` | false | canonical identity confidence/provenance가 없음 |
| `PASS_ASSIGNMENT_RECEIPT` | false | automatic/manual assignment와 original failure receipt가 없음 |
| `PASS_APPARATUS_INPUT` | false | 위 필수 입력 gate가 동시에 성립하지 않음 |
| overall | `BLOCKED_SOURCE_INDEX_JOIN` | empirical response 분석에 입장하지 않음 |

87개 complete session의 4,457 stimulation rows에서는 cardinality, index domain, clock
failure가 모두 0이었다. 3,537개 nonnegative index join 중 2,710개는 nonblank local label로
결합된다. 이 결과의 정확한 상한은
`PARTIAL_SOURCE_INDEX_SCHEMA / OBSERVATIONAL_SCHEMA_ONLY`이며, canonical source identity나
배정 메커니즘의 증거가 아니다.

## 경로 감사

- R3a: `PASS_ACQUISITION_CANDIDATE`. 공식 bytes와 deterministic selection receipt를 확보했다.
- R3b: `BLOCKED_SOURCE_INDEX_JOIN`. 부분 source-index schema만 보존한다.
- R1b: `STOP_FIELD_LOSS`. converter에서 NeuroPAL/manual-target schema는 보였지만 native
  event fields, sentinel semantics, assignment receipt의 rowwise 보존은 보이지 않았다.
- R2: `BLOCKED_SOURCE_JOIN`. 독립 geometric validation object가 없다. frame, units/z
  calibration, transform, radius, ambiguity rule, deterministic tie-break, confidence/provenance를
  가진 outcome-blind immutable object가 새로 제공될 때만 재개한다.

response, fluorescence, autoresponse, fitted kernel, effect, state, p/q value 또는
outcome-derived matching은 읽거나 수행하지 않았다. pre/post baseline, autoresponse-negative
selection, full-state predictive gain을 causal source×state routing으로 간주하지 않았다.

## P1 수정 확인

초기 사후 감사가 지적한 세 항목을 수정했다: `12-routes.md`의 열린 경로를 최종 상태로
닫았고, 구판 601/27 MB 상한을 602/32 MB로 바로잡았으며, OSF 실행 receipt를 68 pages로
통일했다. P2 권고에 따라 JSON field `all_six_family_complete`를 실제 의미인
`all_allowed_family_complete`로 바꿨다. 과학 판정을 바꾸는 P0 결함은 없었다.

## 안정 스냅샷

| 파일 | SHA-256 |
|---|---|
| `00-contract.md` | `f1fd4866a9ea96a89eb0d68ff7ddef0987cb3b78c56f37a6151a6293b8f4858b` |
| `10-sources.md` | `a063769dbb707d330158ea11741733abf16b23c491f2b341ef6624b08f0e7caa` |
| `11-math.md` | `f36c7bcee7e045f16e98c28c37da8eddbe659a91de0b0dee735e6c1a798160d8` |
| `12-routes.md` | `8994bf99388841e1deb3d62bb4d207084a2898442f895d2694b12490c088d5c2` |
| `30-implementation.md` | `8df46ef53d6b6a2fb0e26b96f664b0533b0c3ac9a8c0c8354fdbf328af6d18de` |
| `31-validation.md` | `3bff495394eb224dfe50acabc923f0fe82004ddb3ed1879f61a1825bb5d362a3` |
| audit script | `b6e39c19295c25049efdcee5e4de04c6e1f623ab621c6a7009d20ec23cb1e528` |
| machine JSON | `ba55f83e2436f4b2be373c0254d60378efff7d3dca5494895c68a1771c8ebbec` |
| OSF manifest | `962140bba5863602273a2cf9c32a97c7c1a40456e3620f3fa53ecc522686a8b3` |
| Zenodo archive | `e6a52ec8fbaa2cdb8da2b72549495d5c91f459786bcf61549b0c771e803a9378` |
| converter archive | `9d6d01e74243f5be17204d0a086a33a74a98c8e98ec2f7befa57089eee0bc603` |

독립 사후 감사의 수정 사항을 반영한 뒤 offline rerun은 같은
`BLOCKED_SOURCE_INDEX_JOIN` 결과를 냈고 `git diff --check`는 통과했다.
