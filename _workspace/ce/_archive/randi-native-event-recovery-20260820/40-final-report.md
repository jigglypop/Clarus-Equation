# publication-native event 복원의 종결 보고

Status: COMPLETE

## 초록

이 실행은 Randi 등(2023)의 공개 자료에서 자극 사건을 결과 자료와 독립적으로 복원하여, 상태 조건부 신경 라우팅 검정에 필요한 장치 입력을 만들 수 있는지 점검했다. 공식 OSF 트리와 제한된 비결과 텍스트 파일, Zenodo의 고정 `pumpprobe` 배포본, 고정 NWB 변환기 소스를 조사했지만 fluorescence·response·autoresponse·effect 관련 값을 읽거나 분석하지 않았다. 113개 세션 가운데 다섯 허용 파일군이 모두 갖추어진 세션은 87개였고, 그 안의 4,457개 자극 사건에서 3,537개 비음수 source-index join과 2,710개 비공백 local-label join이 확인되었다. 그러나 827개 join은 빈 local label이었고 26개 세션은 불완전했으며, canonical identity의 confidence/provenance와 원래의 assignment receipt도 복원되지 않았다. 그러므로 이 실행의 종결 지위는 `NATIVE_EVENT_RECOVERY_COMPLETE / PARTIAL_SOURCE_INDEX_SCHEMA / EMPIRICAL_ROUTE_BLOCKED`이며, 이는 효과의 부재나 존재에 관한 결론이 아니라 그 효과를 시험할 입력 영수증이 아직 부족하다는 결론이다.

## 문제와 경계

상태 조건부 라우팅을 검정하려면 각 자극 사건이 누구에게, 어떤 절차로 배정되었는지를 결과로부터 독립적으로 알아야 한다. 이 보고서에서 canonical identity란 recording-local index나 빈 문자열이 아니라, 해당 자극원을 고정된 신경세포 정체성에 연결하고 그 연결의 confidence 또는 provenance를 함께 남긴 식별자다. assignment receipt란 `(animal, session, event)`별로 자극 시각, source identity, 자동·수동 배정, 실패·예외, 용량과 순서를 결과 비열람 상태에서 되살릴 수 있는 기록을 뜻한다.

이 구분이 필요한 이유는 pre/post 차이와 autoresponse-음성 사건 선택이 자극의 무효 대조군을 자동으로 만들지 않기 때문이다. 시간 drift만 있어도 pre/post 차이는 생길 수 있고, 자극 후 반응을 보고 사건을 제외하면 post-treatment selection이 된다. 따라서 source label처럼 보이는 문자열이 있더라도, original assignment와 실패 기록이 없으면 능동 source 효과나 source×state 상호작용을 추정하는 단계로 넘어갈 수 없다.

## 동결된 입력과 복원 절차

**[공리: 입력 경계]** 1차 자료는 Randi et al., *Nature* (2023), DOI `10.1038/s41586-023-06683-4`, 데이터 프로젝트 OSF DOI `10.17605/OSF.IO/E2SYT`, 그리고 Zenodo `pumpprobe` DOI `10.5281/zenodo.8312985`로 고정했다. OSF `raw_extracted_data` 폴더 `671a5286badd54a2128707e3`의 공식 provider tree를 전 페이지 순회하고, 정확한 suffix가 `ds_name.txt`, `labels.txt`, `stim_neurons.txt`, `stim_volume_i.txt`, `t.txt`인 파일만 선택했다. `gcamp`을 비롯한 반응·효과·적합 관련 파일은 선택·열람하지 않았다.

선택은 family 우선순위, 선언 byte 수, canonical path 순서로 결정했고 총 538개 파일, 9,346,585 bytes에 멈췄다. 여기에 Zenodo `pumpprobe-1.1.zip` 한 개와 변환기 commit `3544c9bb59f90d5630fa1871850d990db9cafc18`의 archive 한 개만 더해 총 540개 파일, 10,681,846 bytes였다. 이는 계약의 602개 파일·32,000,000 bytes 상한 안이다. archive는 추출·import·실행·pickle load 없이 member metadata와 UTF-8 source text만 정적으로 점검했다.

**[산출]** OSF manifest의 SHA-256은 `962140bba5863602273a2cf9c32a97c7c1a40456e3620f3fa53ecc522686a8b3`이다. `pumpprobe-1.1.zip`은 1,287,278 bytes, 제공 MD5 `40d87e790193d38528b4ba0cecf23e8c`, local SHA-256 `e6a52ec8fbaa2cdb8da2b72549495d5c91f459786bcf61549b0c771e803a9378`이다. 변환 archive는 47,983 bytes, local SHA-256 `9d6d01e74243f5be17204d0a086a33a74a98c8e98ec2f7befa57089eee0bc603`이다. 이 해시는 획득 바이트를 다시 식별할 수 있게 하지만, 제공자 checksum이 존재하거나 검증되었다는 주장과는 별도로 기록했다.

## event source-index 결과

**[산출]** 113개 numeric-prefix 세션 중 87개가 허용된 다섯 파일군을 모두 지녔다. 이 87개 세션의 4,457개 stimulation row에서 `stim_neurons[event]`를 `labels[stim_neurons[event]]`로 결합했을 때, 3,537개가 비음수 source-index였고 2,710개는 비공백 local label에 도달했다. 827개는 index cardinality와 domain은 유효하지만 label 문자열이 비어 있었다. complete 세션 안에서는 row cardinality 실패, index-domain 실패, clock 실패가 모두 0이었지만, 26개 세션의 결손과 827개의 공백은 전체 자료가 source-index-to-label 완결성을 갖추었다는 명제를 반증한다.

이 수치는 좁지만 유용한 사실을 남긴다. 공식 bytes는 일부 complete 세션에서 event 번호와 recording-local source index, 그리고 일부 local label을 연결한다. 반면 이 local label은 canonical identity가 아니며, 빈 label을 임의의 다른 표나 공간 근접으로 메우면 결과 독립성·불확실성 기록을 잃는다. 따라서 세 기본 machine gate인 `PASS_SOURCE_INDEX_JOIN`, `PASS_SOURCE_JOIN`, `PASS_ASSIGNMENT_RECEIPT`은 모두 false이며, 이 셋을 모두 요구하는 합성 gate `PASS_APPARATUS_INPUT`도 false다. 기계 판정은 `BLOCKED_SOURCE_INDEX_JOIN`이다.

## 변환기와 기하 검증 경로

**[산출]** 고정 NWB 변환기에는 NeuroPAL segmentation, label·confidence·comment schema, 수동 target 파일 `targets_manually_located.txt`, 그리고 manually targeted PumpProbe ID를 다루는 코드가 있다. 그러나 native `stim_neurons`와 `stim_volume_i`, `-1/-2/-3` sentinel의 뜻, complementary/manual label, 원래 assignment·failure receipt를 NWB 산출물에 보존한다는 field-level 증거는 발견되지 않았다. 특히 수동 위치 결정 또는 실패 targeting을 NaN으로 처리하는 경로는 보였지만, 그것이 원자료의 완전한 사건 영수증을 대체하지는 않는다. 이 R1b 경로는 `STOP_FIELD_LOSS`다.

**[미완성]** R2의 독립 기하 검증 객체도 현재의 공식 입력에는 없다. frame, unit 및 z 규약, registration transform, matching radius, ambiguity 처리와 tie-break를 같은 immutable 객체에서 함께 확인할 수 없으므로, 좌표 근접만으로 canonical identity를 만들 수 없다. 이 경로의 지위는 `BLOCKED_SOURCE_JOIN`이다.

## 결론과 재개 조건

**[정리: 종결 판정]** 결과 비열람이라는 입력 경계 아래에서, 현재 공개된 자료만으로는 완전한 source-to-canonical-identity join과 complete assignment receipt가 동시에 복원되지 않는다. 따라서 이 실행은 source×state 라우팅 모델, 능동 source 효과, 무광 효과, 또는 AGI식 기능적 라우팅에 대한 실증 결론을 제공하지 않는다. 반대로, 그러한 결론이 아직 제공되지 않았음을 재현 가능한 기계 gate로 남긴다.

재개에는 결과와 독립적인 immutable 원자료가 필요하다. R3b를 다시 열려면 26개 결손 세션과 827개 빈 label을 설명하거나 보완하는 event-level object가 있어야 하고, local source index에서 canonical identity로 가는 confidence/provenance가 필요하다. R1b를 다시 열려면 converter가 native event, sentinel, 수동·실패·배정 정보를 실제 출력에 어떤 field로 보존하는지 byte-backed mapping이 필요하다. R2를 다시 열려면 위의 frame·unit·transform·radius·ambiguity·tie-break를 모두 고정한 독립 validation receipt가 필요하다. 이 중 어느 하나도 response 값, responder 선택, 효과 크기, state embedding으로 소급 구성해서는 안 된다.

## 재현 기록

동결된 산출물은 `artifacts/native_event_audit.json`이며, audit program은 `artifacts/native_event_recovery_audit.py`다. 네트워크 없이 기존 receipt를 재판정하는 명령은 다음과 같다.

```powershell
.codex\hooks\python.cmd python "_workspace\ce\randi-native-event-recovery-20260820\artifacts\native_event_recovery_audit.py" --run-dir "_workspace\ce\randi-native-event-recovery-20260820" --offline
```

예상 요약은 `{"files": 538, "overall": "BLOCKED_SOURCE_INDEX_JOIN", "sessions": 113, "source_index_join": false}`다. 구현과 검증의 세부 한계, provider checksum 분리, 선택 영수증은 `30-implementation.md`와 `31-validation.md`에 고정되어 있다. 이 실행은 `C:\Users\dongh\.codex\hooks\run.ps1`가 환경에 없었기 때문에, 같은 8-stage 이름과 수동 gate 검증으로 수행했다.

## 참고 자료

Randi, F. et al. “Neural signal propagation atlas of *Caenorhabditis elegans*.” *Nature* 623, 406–414 (2023). DOI: `10.1038/s41586-023-06683-4`.

OSF project `E2SYT`. DOI: `10.17605/OSF.IO/E2SYT`.

Leifer lab `pumpprobe` release. Zenodo DOI: `10.5281/zenodo.8312985`.

`catalystneuro/leifer_lab_to_nwb`, frozen commit `3544c9bb59f90d5630fa1871850d990db9cafc18`.
